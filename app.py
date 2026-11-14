"""BOM-to-Spend Tie-Out: the demo app.

    uv run streamlit run app.py

Reads out/*.csv and out/eval_*.json only. No API keys, no model loading: every match,
reason and metric was computed earlier and saved. Reviewer actions and the materiality
slider recompute buckets, product cost and leads live, in session state.
"""
from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from src import allocate

OUT = Path("out")
BUCKETS = ["production", "spares", "warranty", "unassigned"]
BUCKET_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#9a9890"]  # unassigned is neutral: it is the absence of a tie
SURFACE = "#ffffff"
HOW = {
    "rule_pn_exact": "Rule: part number", "rule_xref": "Rule: supplier crosswalk", "rule_xref_offbom": "Rule: crosswalk to a part on no BOM",
    "classifier": "Classifier", "llm": "Claude", "llm_material": "Claude, held by the materiality rule", "reviewer": "Reviewer (this session)",
}
MAIN_MODEL, SMALL_MODEL = "claude-opus-5-5", "claude-haiku-5-5"

st.set_page_config(page_title="BOM-to-Spend Tie-Out", layout="wide")


def usd(x: float) -> str:
    return f"${x:,.0f}"


def pct(x: float) -> str:
    return f"{x:.1%}"


@st.cache_data
def load():
    lines = pd.read_csv(OUT / "lines.csv", dtype={"po_id": str, "supplier_part_number": str})
    text = lines.select_dtypes(include=["object", "string"]).columns
    lines[text] = lines[text].fillna("")
    bom = pd.read_csv(OUT / "bom.csv")
    builds = pd.read_csv(OUT / "builds.csv", dtype={"period": str})
    evals = {p.stem: json.loads(p.read_text()) for p in sorted(OUT.glob("eval_*.json"))}
    # what rule-confirmed purchases say about each part: its usual price and who sells it
    confirmed = lines[lines["source_pre"].isin(["rule_pn_exact", "rule_xref"])]
    ref = confirmed.groupby("part").agg(ref_price=("unit_price_usd", "median"), ref_suppliers=("supplier_key", lambda s: sorted(set(s))))
    return lines, bom, builds, evals, ref


@st.cache_data
def compute(materiality: float, actions: tuple):
    """The whole downstream state for one materiality threshold and one set of reviewer actions."""
    base, bom, builds, _, _ = load()
    lines = base.copy()
    lines["decision"], lines["source"] = lines["decision_pre"], lines["source_pre"]
    lines = allocate.apply_materiality(lines, materiality)
    for po_id, action in actions:
        row = lines["po_id"] == po_id
        if action == "approve":
            lines.loc[row, "part"] = lines.loc[row, "suggested_part"]
        lines.loc[row, ["decision", "source"]] = [{"approve": "accept", "no_match": "no_match", "reject": "rejected"}[action], "reviewer"]
    lines = allocate.assign_buckets(lines)
    alloc, products = allocate.allocate(lines, bom, builds)
    return lines, alloc, products, allocate.find_leads(lines, bom)


base, bom, builds, evals, ref = load()
part_desc = bom.drop_duplicates("part_number").set_index("part_number")["description"]
st.session_state.setdefault("actions", {})

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Controls")
    materiality = st.slider("Materiality threshold (USD)", 0, 200_000, int(allocate.MATERIALITY_USD), 5_000,
                            help="A match that only Claude vouched for goes to a person when the line is worth more than this.")
    st.caption(f"{len(st.session_state.actions)} reviewer decisions this session")
    if st.button("Undo all reviewer decisions", disabled=not st.session_state.actions):
        st.session_state.actions = {}
        st.rerun()

lines, alloc, products, leads = compute(float(materiality), tuple(sorted(st.session_state.actions.items())))
start_lines = compute(float(allocate.MATERIALITY_USD), ())[0]
total = lines["spend_usd"].sum()
by_bucket = lines.groupby("bucket")["spend_usd"].sum().reindex(BUCKETS, fill_value=0.0)
tied = 1 - by_bucket["unassigned"] / total
tied_start = 1 - start_lines.loc[start_lines["bucket"] == "unassigned", "spend_usd"].sum() / total
queue = lines[lines["decision"] == "review"].sort_values("spend_usd", ascending=False)

# ------------------------------------------------------------------- header
st.title("BOM-to-Spend Tie-Out")
st.warning("**Synthetic data.** The products, parts, suppliers and prices are invented, and the true matches are planted so accuracy "
           "can be measured. Nothing here comes from a real company.", icon=":material/science:")

k1, k2, k3 = st.columns(3)
k1.metric("Tie-out rate", pct(tied), f"{(tied - tied_start) * 100:+.1f} pts this session" if abs(tied - tied_start) > 1e-9 else None,
          help="Share of spend tied to a product bucket: production, spares or warranty.")
k2.metric("Waiting for review", f"{len(queue)} lines", usd(queue["spend_usd"].sum()), delta_color="off")
k3.metric("Total spend", usd(total), f"{len(lines):,} purchase lines", delta_color="off")

tab_overview, tab_queue, tab_cost, tab_leads, tab_eval = st.tabs(
    ["Overview", f"Review queue ({len(queue)})", "Product cost", "Leads", "Evaluation"])

# ----------------------------------------------------------------- overview
with tab_overview:
    st.subheader("Where the spend ties")
    cols = st.columns(4)
    for col, b in zip(cols, BUCKETS):
        col.metric(b.capitalize(), pct(by_bucket[b] / total), usd(by_bucket[b]), delta_color="off")
    bar = pd.DataFrame({"bucket": BUCKETS, "spend": by_bucket.to_numpy(), "share": (by_bucket / total).to_numpy(), "order": range(4)})
    st.altair_chart(
        alt.Chart(bar).mark_bar(height=28, stroke=SURFACE, strokeWidth=2).encode(
            x=alt.X("spend:Q", stack="normalize", axis=alt.Axis(format="%", title="Share of spend", grid=False)),
            color=alt.Color("bucket:N", scale=alt.Scale(domain=BUCKETS, range=BUCKET_COLORS), legend=alt.Legend(title=None, orient="top")),
            order="order:Q",
            tooltip=[alt.Tooltip("bucket:N", title="Bucket"), alt.Tooltip("spend:Q", title="Spend", format="$,.0f"),
                     alt.Tooltip("share:Q", title="Share", format=".1%")],
        ).properties(height=90), width="stretch")
    inferred = lines[lines["bucket_inferred"]]
    st.caption(f"Unassigned includes the {len(queue)} lines waiting for review. The bucket was inferred from the match and the plant, "
               f"because the PO type was blank, on {len(inferred):,} lines ({usd(inferred['spend_usd'].sum())}).")

    st.subheader("From a spreadsheet lookup to the full pipeline")
    s2 = evals["eval_step2_local"]["splits"]["heldout"]
    s3 = evals["eval_step3"]
    clf, full = s3["backends"][s3["chosen_backend"]]["heldout"], s3["llm"][MAIN_MODEL]["heldout"]
    true_rate = full["true_tie_out"]
    stages = pd.DataFrame([
        ("1. Exact part-number lookup", s2["(a) naive exact part number"]["tie_out"], s2["(a) naive exact part number"]["precision"],
         s2["(a) naive exact part number"]["recall"], None, s2["(a) naive exact part number"]["wrong_match_spend"]),
        ("2. Rules (normalized numbers, crosswalk)", s2["(b) rules only"]["tie_out"], s2["(b) rules only"]["precision"],
         s2["(b) rules only"]["recall"], None, s2["(b) rules only"]["wrong_match_spend"]),
        ("3. Rules + embeddings, fixed cutoff", s2["(c) rules + embeddings"]["tie_out"], s2["(c) rules + embeddings"]["precision"],
         s2["(c) rules + embeddings"]["recall"], None, s2["(c) rules + embeddings"]["wrong_match_spend"]),
        ("4. Rules + calibrated classifier", clf["tie_out"], clf["accept_precision"], clf["recall"], clf["review_spend"], clf["wrong_match_spend"]),
        ("5. Rules + classifier + Claude", full["tie_out"], full["accept_precision"], full["recall"], full["review_spend"], full["wrong_match_spend"]),
    ], columns=["stage", "tie_out", "precision", "recall", "review_spend", "wrong"])
    left, right = st.columns([3, 2])
    with left:
        bars = alt.Chart(stages).mark_bar(height=22, color="#2a78d6", cornerRadiusEnd=4).encode(
            y=alt.Y("stage:N", sort=None, title=None, axis=alt.Axis(labelLimit=320)),
            x=alt.X("tie_out:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%", title="Spend tied out (held-out lines)")),
            tooltip=[alt.Tooltip("stage:N", title="Stage"), alt.Tooltip("tie_out:Q", title="Tied out", format=".1%"),
                     alt.Tooltip("precision:Q", title="Precision", format=".1%"), alt.Tooltip("recall:Q", title="Recall", format=".1%")])
        labels = bars.mark_text(align="left", dx=6).encode(text=alt.Text("tie_out:Q", format=".1%"))
        truth_df = pd.DataFrame({"x": [true_rate], "label": [f"true rate {true_rate:.1%}"]})
        rule = alt.Chart(truth_df).mark_rule(strokeDash=[4, 4], color="#52514e").encode(x="x:Q", tooltip=alt.Tooltip("label:N", title="Planted truth"))
        st.altair_chart((bars + labels + rule).properties(height=230), width="stretch")
        st.caption(f"Dashed line: the true tie-out rate on these lines, {true_rate:.1%}.")
    with right:
        st.dataframe(pd.DataFrame({
            "Stage": stages["stage"], "Tied out": stages["tie_out"].map(pct), "Precision": stages["precision"].map(pct),
            "Recall": stages["recall"].map(pct), "Left for review": stages["review_spend"].map(lambda v: "n/a" if pd.isna(v) else usd(v)),
            "Wrong-match $": stages["wrong"].map(usd)}), hide_index=True, width="stretch")
    st.caption(f"Measured on {full['lines']:,} held-out purchase lines ({usd(full['spend'])}) written in noise styles that were never used "
               "to tune the matcher. Stage 4 ties less than stage 3 because it refuses to guess: uncertain lines wait for review instead.")

# ------------------------------------------------------------- review queue
with tab_queue:
    st.subheader("Review queue, largest dollars first")
    if queue.empty:
        st.success("Nothing is waiting for review.")
    else:
        def why(r) -> str:
            if r["source"] == "llm_material":
                return f"Claude matched it, but the line is above the {usd(materiality)} materiality threshold"
            if pd.notna(r["llm_conf"]) and r["llm_part"]:
                return "Claude picked a part, below its confidence cutoff"
            if pd.notna(r["llm_conf"]):
                return "Claude leaned towards no BOM part, not confidently"
            return "Classifier unsure"

        view = pd.DataFrame({
            "Spend": queue["spend_usd"].map(usd), "Description on the PO": queue["item_description"],
            "Part-number field": queue["supplier_part_number"], "Supplier": queue["supplier"],
            "Suggested part": queue["suggested_part"], "Why it is here": queue.apply(why, axis=1)})
        picked = st.dataframe(view, hide_index=True, width="stretch", height=260, on_select="rerun", selection_mode="single-row", key="queue_table")
        rows = picked.selection.rows
        r = queue.iloc[rows[0] if rows and rows[0] < len(queue) else 0]
        sug = r["suggested_part"]

        st.markdown(f"#### PO line {r['po_id']}: {usd(r['spend_usd'])}")
        a, b = st.columns(2)
        with a:
            st.markdown("**The purchase line**")
            st.table(pd.DataFrame({"": {
                "Description": r["item_description"], "Part-number field": r["supplier_part_number"] or "(blank)", "Supplier": r["supplier"],
                "Plant": r["plant"], "PO date": r["po_date"], "PO type": r["po_type"] or "(blank)",
                "Quantity": f"{r['base_qty']:,.0f} units (bought per {r['uom']})", "Price per unit": f"${r['unit_price_usd']:.4f}"}}))
        with b:
            st.markdown("**The suggested match**")
            st.markdown(f"`{sug}` {part_desc.get(sug, '')}")
            st.markdown(f"**How it was decided:** {why(r)}. Classifier probability for its best candidate: {r['prob']:.2f}.")
            if r["llm_reason"]:
                answer = f"`{r['llm_part']}`" if r["llm_part"] else "no BOM part"
                st.info(f"**Claude's answer: {answer}, confidence {r['llm_conf']:.2f}.** {r['llm_reason']}")
            if sug in ref.index:
                ref_price, ref_sup = ref.at[sug, "ref_price"], ref.at[sug, "ref_suppliers"]
                st.markdown(f"**Price evidence:** this line is ${r['unit_price_usd']:.4f} per unit; rule-confirmed purchases of `{sug}` "
                            f"have a median of ${ref_price:.4f} ({r['unit_price_usd'] / ref_price - 1:+.0%}).")
                seen = "has" if r["supplier_key"] in ref_sup else "has no"
                st.markdown(f"**Supplier evidence:** this supplier {seen} rule-confirmed purchases of `{sug}`. "
                            f"Confirmed suppliers: {', '.join(s.title() for s in ref_sup)}.")
            else:
                st.markdown(f"**Price and supplier evidence:** no rule-confirmed purchases of `{sug}` yet, so there is nothing to compare against.")

        c1, c2, c3, _ = st.columns([1.2, 1.2, 1.4, 3])
        for col, label, action, kind in ((c1, f"Approve match to {sug}", "approve", "primary"), (c2, "Reject suggestion", "reject", "secondary"),
                                         (c3, "Confirm: not a BOM part", "no_match", "secondary")):
            if col.button(label, type=kind, key=f"{action}_{r['po_id']}"):
                st.session_state.actions[r["po_id"]] = action
                st.rerun()
        st.caption("Approve ties the line to the suggested part. Reject removes the suggestion and leaves the line untied for a manual lookup. "
                   "Not a BOM part closes the line as correctly unmatched. Tie-out, buckets, product cost and leads update immediately.")

    if st.session_state.actions:
        st.markdown("**Decided this session**")
        done = lines[lines["po_id"].isin(st.session_state.actions)]
        st.dataframe(pd.DataFrame({"PO line": done["po_id"], "Spend": done["spend_usd"].map(usd), "Description": done["item_description"],
                                   "Action": done["po_id"].map(st.session_state.actions), "Part": done["part"].where(done["decision"] == "accept", ""),
                                   "Bucket now": done["bucket"]}), hide_index=True, width="stretch")

# ------------------------------------------------------------- product cost
with tab_cost:
    st.subheader("Parts cost per unit: BOM standard versus estimated actual")
    st.caption("Estimated: matched production spend is split across the products that use each part, by BOM quantity x units built in the "
               "purchase's quarter. Purchases show what was bought, not what was consumed. Lines waiting for review are not included yet.")
    st.dataframe(pd.DataFrame({
        "Product": products["product_id"], "Units built": products["units_built"].map("{:,}".format),
        "Standard per unit": products["std_cost_per_unit"].map("${:.2f}".format),
        "Estimated actual per unit": products["actual_cost_per_unit_est"].map("${:.2f}".format),
        "Actual vs standard": products["actual_vs_std"].map("{:+.1%}".format),
        "Allocated production spend": products["allocated_production_spend_est"].map(usd)}), hide_index=True, width="stretch")

    product = st.selectbox("Parts driving the difference for", products["product_id"])
    units = products.set_index("product_id").at[product, "units_built"]
    use = bom[bom["product_id"] == product].groupby("part_number").agg(
        description=("description", "first"), assembly=("assembly_id", "first"), qty=("qty_per_unit", "sum"), std_cost=("std_cost", "first"))
    use["std_per_unit"] = use["qty"] * use["std_cost"]
    use["actual_per_unit"] = alloc[alloc["product_id"] == product].groupby("part_number")["allocated_usd"].sum().reindex(use.index).fillna(0.0) / units
    use["diff"] = use["actual_per_unit"] - use["std_per_unit"]
    drivers = use.reindex(use["diff"].abs().sort_values(ascending=False).index).head(12).reset_index()
    drivers["direction"] = drivers["diff"].map(lambda d: "above standard" if d > 0 else "below standard")
    drivers["label"] = drivers["part_number"] + "  " + drivers["description"]
    st.altair_chart(
        alt.Chart(drivers).mark_bar(height=16, cornerRadiusEnd=3).encode(
            y=alt.Y("label:N", sort=None, title=None, axis=alt.Axis(labelLimit=380)),
            x=alt.X("diff:Q", axis=alt.Axis(format="$.2f", title="Estimated actual minus standard, per unit built")),
            color=alt.Color("direction:N", scale=alt.Scale(domain=["above standard", "below standard"], range=["#eb6834", "#2a78d6"]),
                            legend=alt.Legend(title=None, orient="top")),
            tooltip=[alt.Tooltip("part_number:N", title="Part"), alt.Tooltip("description:N", title="Description"),
                     alt.Tooltip("std_per_unit:Q", title="Standard per unit", format="$.3f"),
                     alt.Tooltip("actual_per_unit:Q", title="Est. actual per unit", format="$.3f"),
                     alt.Tooltip("diff:Q", title="Difference", format="+$.3f")],
        ).properties(height=330), width="stretch")
    st.dataframe(pd.DataFrame({
        "Part": drivers["part_number"], "Description": drivers["description"], "Assembly": drivers["assembly"],
        "Qty per unit": drivers["qty"].map("{:g}".format), "Standard per unit": drivers["std_per_unit"].map("${:.3f}".format),
        "Est. actual per unit": drivers["actual_per_unit"].map("${:.3f}".format), "Difference": drivers["diff"].map("{:+.3f}".format),
        "Note": drivers["actual_per_unit"].map(lambda v: "no matched production spend yet" if v == 0 else "")}), hide_index=True, width="stretch")

# -------------------------------------------------------------------- leads
with tab_leads:
    st.subheader("Leads, not savings")
    st.caption("Each lead is a price difference worth a conversation, shown as a range: half the gap closed, up to all of it. "
               "None of it is booked. Volume commitments, timing, quality and logistics can explain a gap.")
    names = {"supplier_price_gap": "Same part, different supplier prices", "duplicate_part": "Duplicate internal part numbers",
             "spares_premium": "Spares bought at a premium"}
    chosen = st.multiselect("Lead type", list(names), default=list(names), format_func=names.get)
    shown = leads[leads["lead_type"].isin(chosen)].reset_index(drop=True)
    m1, m2 = st.columns(2)
    m1.metric("Leads", len(shown))
    m2.metric("Combined range", f"{usd(shown['low_usd'].sum())} to {usd(shown['high_usd'].sum())}")
    if shown.empty:
        st.info("No leads for this selection.")
    else:
        pick = st.dataframe(pd.DataFrame({
            "Type": shown["lead_type"].map(names), "Part": shown["part_number"], "Description": shown["description"],
            "Range": shown["low_usd"].map(usd) + " to " + shown["high_usd"].map(usd), "Evidence": shown["detail"]}),
            hide_index=True, width="stretch", height=300, on_select="rerun", selection_mode="single-row", key="leads_table")
        rows = pick.selection.rows
        lead = shown.iloc[rows[0] if rows and rows[0] < len(shown) else 0]
        parts_in = [lead["part_number"]] + ([lead["related"]] if lead["lead_type"] == "duplicate_part" else [])
        buckets_in = ["production", "spares"] if lead["lead_type"] == "spares_premium" else ["production"]
        under = lines[(lines["decision"] == "accept") & lines["part"].isin(parts_in) & lines["bucket"].isin(buckets_in)].sort_values("po_date")
        st.markdown(f"**Purchase lines behind the lead on `{'` and `'.join(parts_in)}`** ({len(under)} lines, {usd(under['spend_usd'].sum())}). "
                    "Select a row above to change the lead.")
        st.dataframe(pd.DataFrame({
            "PO line": under["po_id"], "Date": under["po_date"], "Plant": under["plant"], "Supplier": under["supplier"], "Part": under["part"],
            "Bucket": under["bucket"], "Units": under["base_qty"].map("{:,.0f}".format), "Price per unit": under["unit_price_usd"].map("${:.4f}".format),
            "Spend": under["spend_usd"].map(usd), "Description on the PO": under["item_description"],
            "Matched by": under["source"].map(HOW)}), hide_index=True, width="stretch", height=320)

# --------------------------------------------------------------- evaluation
with tab_eval:
    s3, s4 = evals["eval_step3"], evals["eval_step4"]
    llm = s3["llm"][MAIN_MODEL]
    st.subheader("Held-out versus tuning split, full pipeline")
    st.caption("The matcher was tuned on the tuning split only. The held-out split is written in noise styles it never saw: plant-specific "
               "codes, part-number typos, supplier catalog wording, description typos and Spanish.")
    rows = {"Purchase lines": lambda m: f"{m['lines']:,}", "Spend": lambda m: usd(m["spend"]),
            "Auto-accepted, share of lines": lambda m: pct(m["accept_share"]), "Precision of accepted matches": lambda m: f"{m['accept_precision']:.2%}",
            "Recall": lambda m: pct(m["recall"]), "Left for review": lambda m: f"{m['review_lines']} lines, {usd(m['review_spend'])}",
            "No-match calls that are right": lambda m: pct(m["no_match_accuracy"]), "Tie-out rate": lambda m: pct(m["tie_out"]),
            "True tie-out rate": lambda m: pct(m["true_tie_out"]), "Wrong-match dollars": lambda m: usd(m["wrong_match_spend"])}
    st.table(pd.DataFrame({"Held-out": {k: f(llm["heldout"]) for k, f in rows.items()}, "Tuning": {k: f(llm["tune"]) for k, f in rows.items()}}))

    st.subheader("Accuracy by noise style")
    styles = pd.DataFrame(s3["by_style"])
    for kind, title in (("id_style", "How the part-number field was written"), ("desc_style", "How the description was written")):
        sub = styles[styles["kind"] == kind]
        table = {}
        for split, name in (("heldout", "held-out"), ("tune", "tuning")):
            part = sub[sub["split"] == split].set_index("style")
            table[f"Lines ({name})"] = part["lines"].map("{:,}".format)
            table[f"Precision ({name})"] = part["accept_precision"].map(pct)
            table[f"Recall ({name})"] = part["recall"].map(pct)
            table[f"In review ({name})"] = part["review_lines"].astype(str)
        st.markdown(f"**{title}**")
        st.dataframe(pd.DataFrame(table).fillna("not in this split").rename_axis("Style").reset_index(), hide_index=True, width="stretch")

    st.subheader("Claude on the review band: cost and the small-model comparison")
    models = {}
    for model in (MAIN_MODEL, SMALL_MODEL):
        m, u = s3["llm"][model], s3["llm"][model]["usage"]
        models[model] = {
            "Lines sent (distinct calls)": f"{m['band_lines']} ({u['calls']})", "Input / output tokens": f"{u['input_tokens']:,} / {u['output_tokens']:,}",
            "Cost at list price": f"${u['cost_usd']:.2f}", "Cost per 1,000 purchase lines processed": f"${u['cost_usd'] / s3['total_lines'] * 1000:.2f}",
            "Cost per 1,000 lines sent to Claude": f"${u['cost_usd'] / m['band_lines'] * 1000:.2f}",
            "Answer right, held-out band": pct(m["heldout_answer_accuracy"]), "Answer right, tuning band": pct(m["tune_answer_accuracy"]),
            "Held-out review queue afterwards": f"{m['heldout']['review_lines']} lines, {usd(m['heldout']['review_spend'])}",
            "Held-out tie-out afterwards": pct(m["heldout"]["tie_out"])}
    st.table(pd.DataFrame(models))
    st.caption("Only the uncertain middle band goes to Claude, so cost scales with how messy the data is, not with its size. "
               "The small model is much cheaper, but its picks did not reach 99% precision on the tuning split, so none were auto-accepted.")

    st.subheader("Leads against the planted findings")
    st.dataframe(pd.DataFrame([{"Lead type": names[k], "Found": v["found"], "Planted": v["planted"], "Precision": f"{v['precision']:.0%}",
                                "Recall": f"{v['recall']:.0%}", "Range": f"{usd(v['low_usd'])} to {usd(v['high_usd'])}"}
                               for k, v in s4["leads"].items()]), hide_index=True, width="stretch")
    st.caption(f"Measured at the default {usd(s4['materiality_usd'])} materiality threshold, before any reviewer action.")

    st.subheader("Limits")
    st.markdown("""
- **The data is synthetic.** Accuracy is measured on noise this project generated. Real exports are messier in ways the generator does not model, so treat these numbers as a ceiling, not a forecast.
- **One product family.** Four drill models and 276 parts. A real manufacturer has thousands of parts and many near-identical ones.
- **Allocation is an estimate.** Spend is split by BOM quantity and build volume. It ignores inventory, scrap and timing between buying and building.
- **Buckets lean on the PO type.** Where it is blank, the bucket is inferred from the match and the plant. A blank, unmatched spare is counted as unassigned.
- **Duplicate detection is narrow.** It finds part numbers whose descriptions use the same words and numbers. Duplicates described differently are missed.
- **Leads are not savings.** A price gap can have a good reason. Each one needs a buyer's judgment before it means anything.
- **Reviewer decisions here last for the session.** A real deployment would store them and use them to retrain the matcher.
- **What a pilot would test:** match accuracy on one plant's real exports, how much spend lands in review, and how long a reviewer takes to clear it.
""")
