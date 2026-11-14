"""BOM-to-Spend Tie-Out: the demo app.

    uv run streamlit run app.py

The built-in sample reads out/*.csv and out/eval_*.json only: no API keys, no model loading.
The Upload tab runs the matcher on a company's own files and does need API keys.
Reviewer decisions are written to crosswalk.csv and applied to every line with the same text.
"""
from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from src import allocate, crosswalk, mapping, pipeline
from src.normalize import norm_desc, norm_pn, prepare

OUT = Path("out")
SAMPLE_DIR = Path("samples/second_company")
SAMPLE_FILES = {"boms": "bom_export.csv", "purchases": "po_lines.xlsx", "builds": "production_output.csv"}
KIND_LABEL = {"boms": "Bill of materials", "purchases": "Purchase order lines", "builds": "Production volumes"}
BUCKETS = ["production", "spares", "warranty", "unassigned"]
BUCKET_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#9a9890"]  # unassigned is neutral: it is the absence of a tie
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
SURFACE = "#ffffff"
HOW = {
    "rule_pn_exact": "Rule: part number", "rule_xref": "Rule: supplier crosswalk", "rule_xref_offbom": "Rule: crosswalk to a part on no BOM",
    "rule_pn_in_text": "Rule: part number in the description", "rule_crosswalk": "Reviewer crosswalk", "rule_crosswalk_none": "Reviewer crosswalk",
    "classifier": "Classifier", "embedding": "Nearest description only", "llm": "Claude", "llm_material": "Claude, held by the materiality rule",
    "crosswalk": "Reviewer crosswalk",
}
MAIN_MODEL, SMALL_MODEL = "claude-opus-5-5", "claude-haiku-5-5"

st.set_page_config(page_title="BOM-to-Spend Tie-Out", layout="wide")


def usd(x: float) -> str:
    return "n/a" if pd.isna(x) else f"${x:,.0f}"


def pct(x: float) -> str:
    return "n/a" if pd.isna(x) else f"{x:.1%}"


def quarter_of(dates: pd.Series) -> pd.Series:
    d = pd.to_datetime(dates)
    return d.dt.year.astype(str) + "Q" + d.dt.quarter.astype(str)


@st.cache_data
def load():
    lines = pd.read_csv(OUT / "lines.csv", dtype={"po_id": str, "supplier_part_number": str})
    text = lines.select_dtypes(include=["object", "string"]).columns
    lines[text] = lines[text].fillna("")
    bom = pd.read_csv(OUT / "bom.csv")
    builds = pd.read_csv(OUT / "builds.csv", dtype={"period": str})
    evals = {p.stem: json.loads(p.read_text()) for p in sorted(OUT.glob("eval_*.json"))}
    return lines, bom, builds, evals


@st.cache_data
def reference(base: pd.DataFrame) -> pd.DataFrame:
    """What rule-confirmed purchases say about each part: its usual price and who sells it."""
    confirmed = base[base["source_pre"].str.startswith("rule_") & (base["decision_pre"] == "accept")]
    return confirmed.groupby("part").agg(ref_price=("unit_price_usd", "median"), ref_suppliers=("supplier_key", lambda s: sorted(set(s))))


@st.cache_data
def compute(base: pd.DataFrame, bom: pd.DataFrame, builds: pd.DataFrame, materiality: float, rules: tuple, q_from: str, q_to: str):
    """The whole downstream state for one materiality threshold, one crosswalk and one date range."""
    lines = base.copy()
    lines["decision"], lines["source"] = lines["decision_pre"], lines["source_pre"]
    lines = allocate.apply_materiality(lines, materiality)
    lines = allocate.apply_crosswalk(lines, {k: (d, part) for k, d, part in rules})
    lines = lines[(lines["quarter"] >= q_from) & (lines["quarter"] <= q_to)].reset_index(drop=True)
    in_range = quarter_of(builds["period"]).between(q_from, q_to)
    lines = allocate.assign_buckets(lines)
    alloc, products = allocate.allocate(lines, bom, builds[in_range])
    return lines, alloc, products, allocate.find_leads(lines, bom)


sample_base, sample_bom, sample_builds, evals = load()
if st.session_state.pop("_switch_to_uploaded", False):
    st.session_state["dataset"] = "Uploaded files"
uploaded = st.session_state.get("uploaded")

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Controls")
    options = ["Built-in sample"] + (["Uploaded files"] if uploaded else [])
    if st.session_state.get("dataset") not in options:
        st.session_state["dataset"] = options[0]
    dataset = st.radio("Data", options, key="dataset")
    is_sample = dataset == "Built-in sample"
    base, bom, builds = (sample_base, sample_bom, sample_builds) if is_sample else (uploaded["lines"], uploaded["bom"], uploaded["builds"])

    materiality = st.slider("Materiality threshold (USD)", 0, 200_000, int(allocate.MATERIALITY_USD), 5_000,
                            help="A match that only Claude vouched for goes to a person when the line is worth more than this.")
    quarters = sorted(base["quarter"].unique())
    q_from, q_to = (st.select_slider("Date range (quarters)", options=quarters, value=(quarters[0], quarters[-1]), key=f"quarters_{dataset}")
                    if len(quarters) > 1 else (quarters[0], quarters[0]))

    st.subheader("Reviewer crosswalk")
    xw = crosswalk.load()
    st.caption(f"{len(xw)} decisions saved in crosswalk.csv. Each one applies to every line with the same text, now and on the next run.")
    b1, b2 = st.columns(2)
    if b1.button("Undo last", disabled=xw.empty):
        crosswalk.undo_last()
        st.rerun()
    if b2.button("Clear all", disabled=xw.empty):
        crosswalk.clear()
        st.rerun()
    st.download_button("Download crosswalk.csv", xw.to_csv(index=False), "crosswalk.csv", "text/csv", disabled=xw.empty)

rules = tuple(sorted((k, d, part) for k, (d, part) in crosswalk.as_rules(xw).items()))
lines, alloc, products, leads = compute(base, bom, builds, float(materiality), rules, q_from, q_to)
start_lines = compute(base, bom, builds, float(allocate.MATERIALITY_USD), (), q_from, q_to)[0]
ref = reference(base)
part_desc = bom.drop_duplicates("part_number").set_index("part_number")["description"]
total = lines["spend_usd"].sum()
by_bucket = lines.groupby("bucket")["spend_usd"].sum().reindex(BUCKETS, fill_value=0.0)
tied = 1 - by_bucket["unassigned"] / total
tied_start = 1 - start_lines.loc[start_lines["bucket"] == "unassigned", "spend_usd"].sum() / total
queue = lines[lines["decision"] == "review"].sort_values("spend_usd", ascending=False)

# ------------------------------------------------------------------- header
st.title("BOM-to-Spend Tie-Out")
if is_sample or uploaded.get("demo_sample"):
    st.warning("**Synthetic data.** The products, parts, suppliers and prices are invented, and the true matches are planted so accuracy "
               "can be measured. Nothing here comes from a real company.", icon=":material/science:")
if not is_sample:
    st.info(f"**Uploaded files:** {', '.join(uploaded['files'])}. Matched with rules, embeddings and Claude. The classifier is skipped "
            "until this company has confirmed matches to train on; every reviewer decision in the crosswalk is one.", icon=":material/upload_file:")

k1, k2, k3 = st.columns(3)
k1.metric("Tie-out rate", pct(tied), f"{(tied - tied_start) * 100:+.1f} pts from reviewer decisions and settings" if abs(tied - tied_start) > 1e-9 else None,
          help="Share of spend tied to a product bucket: production, spares or warranty.")
k2.metric("Waiting for review", f"{len(queue)} lines", usd(queue["spend_usd"].sum()), delta_color="off")
k3.metric("Total spend", usd(total), f"{len(lines):,} purchase lines", delta_color="off")

tab_overview, tab_queue, tab_cost, tab_leads, tab_trend, tab_eval, tab_upload = st.tabs(
    ["Overview", f"Review queue ({len(queue)})", "Product cost", "Leads", "Trends and inventory", "Evaluation", "Upload"])

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

    if is_sample:
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
    else:
        st.subheader("How this run was matched")
        stats = uploaded["stats"]
        st.markdown(f"{stats['lines']:,} purchase lines against {stats['bom_parts']:,} BOM parts. Rules settled {stats['settled_by_rules']:,} lines; "
                    f"{stats['sent_to_claude']:,} went to Claude with their five nearest BOM parts ({stats['claude']['calls']} distinct calls, "
                    f"${stats['claude']['cost_usd']:.2f} at list price). Embeddings: {stats['embedding_backend']}.")
        how = lines.assign(how=lines["source"].map(HOW).fillna(lines["source"])).groupby(["decision", "how"]).agg(
            lines=("po_id", "size"), spend=("spend_usd", "sum")).reset_index()
        st.dataframe(pd.DataFrame({"Outcome": how["decision"], "Decided by": how["how"], "Lines": how["lines"], "Spend": how["spend"].map(usd)}),
                     hide_index=True, width="stretch")
        for note in uploaded["notes"]:
            st.caption(note)
        st.caption("There is no ground truth for uploaded files, so accuracy cannot be reported here. The Evaluation tab shows how the same "
                   "steps score on the built-in sample and on external data.")

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
            return "Classifier unsure" if r["source"] == "classifier" else "Not judged by Claude: over the call limit or the call failed"

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
            st.markdown(f"**How it was decided:** {why(r)}. {'Classifier probability for its best candidate' if is_sample else 'Description similarity of the nearest BOM part'}: {r['prob']:.2f}.")
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
                crosswalk.add(r["supplier_part_number"], r["item_description"], sug, action, dataset)
                st.rerun()
        st.caption("Approve ties the line to the suggested part. Reject removes the suggestion and leaves the line untied for a manual lookup. "
                   "Not a BOM part closes the line as correctly unmatched. The decision is saved to crosswalk.csv and applies to every line with the "
                   "same text, so tie-out, buckets, product cost and leads update immediately.")

    if not xw.empty:
        st.markdown("**Reviewer crosswalk**")
        keys = lines["supplier_part_number"].map(norm_pn) + "|" + lines["item_description"].map(norm_desc)
        settled = keys.value_counts()
        shown_xw = xw.assign(lines_settled=[int(settled.get(f"{norm_pn(a)}|{norm_desc(d)}", 0)) for a, d in zip(xw["supplier_part_number"], xw["item_description"])])
        st.dataframe(shown_xw.rename(columns={"supplier_part_number": "Part-number field", "item_description": "Description on the PO", "part_number": "BOM part",
                                              "decision": "Decision", "decided_at": "Decided at", "dataset": "Data", "lines_settled": "Lines it settles here"}),
                     hide_index=True, width="stretch")
        st.caption("Undo, clear and download are in the sidebar.")

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
    st.download_button("Download product cost (CSV)", products.to_csv(index=False), "product_cost.csv", "text/csv")
    if products["std_cost_per_unit"].isna().all():
        st.info("The BOM has no standard cost, so the parts below are ranked by estimated actual cost per unit.")

    product = st.selectbox("Parts driving the difference for", products["product_id"])
    units = products.set_index("product_id").at[product, "units_built"]
    use = bom[bom["product_id"] == product].groupby("part_number").agg(
        description=("description", "first"), assembly=("assembly_id", "first"), qty=("qty_per_unit", "sum"), std_cost=("std_cost", "first"))
    use["std_per_unit"] = (use["qty"] * use["std_cost"]).fillna(0.0)
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
    st.download_button("Download leads (CSV)", shown.to_csv(index=False), "leads.csv", "text/csv", disabled=shown.empty)
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
    if not is_sample:
        st.info("Uploaded files have no ground truth. Everything below was measured on the built-in sample and on external data.")
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

    st.subheader("External tests: real data, nothing retrained")
    st.caption("Zero-shot: the classifier and every cutoff come from the synthetic tuning split. The downloaded data is not in the repository; "
               "`python3 -m src.external.fetch` gets it.")
    abt, ext = evals.get("eval_external_abt_buy"), evals.get("eval_external_parts")
    if abt:
        st.markdown(f"**A. Abt-Buy product matching** (University of Mannheim benchmark, fixed test split: {abt['test_pairs']} pairs, "
                    f"{abt['test_matches']} true matches)")
        st.dataframe(pd.DataFrame([{"Method": k, "Precision": f"{v['precision']:.2f}", "Recall": f"{v['recall']:.2f}", "F1": f"{v['f1']:.2f}"}
                                   for k, v in {**abt["results"], **abt["published"]}.items()]), hide_index=True, width="stretch")
        st.caption(f"The benchmark's validation split was used only to check the model-code rule and set two similarity cutoffs. "
                   f"Claude judged {abt['claude']['pairs_sent']} pairs for ${abt['claude']['cost_usd']:.2f}. The last two rows are the baselines published on the benchmark page.")
    if ext:
        st.markdown(f"**B. Real electronic parts** ({ext['bom_parts']} BOM parts from {ext['bom_files']} open-hardware Kitspace BOMs; "
                    f"{ext['mpns_found_in_catalog']} of their manufacturer part numbers found in the JLCPCB catalog of {ext['catalog_size']:,} parts; "
                    f"{ext['no_match_lines']} State of California purchase-order lines as true no-match cases)")
        table = []
        for cond, c in ext["conditions"].items():
            for stage_name, m in (("Rules + classifier", c["classifier_stage"]), ("Rules + classifier + Claude", c["with_claude"])):
                table.append({"Condition": cond, "Stage": stage_name, "Right": m["correct"], "Same-spec alternate": m["wrong_same_spec_alternate"],
                              "Wrong": m["wrong_other"], "In review": m["in_review"], "Called no match": m["called_no_match"],
                              "Precision": f"{m['precision']:.2f}", "Recall": f"{m['recall']:.2f}",
                              "No-match lines right": f"{m['no_match_accuracy']:.1%}", "No-match wrongly accepted": m["no_match_wrongly_accepted"]})
        st.dataframe(pd.DataFrame(table), hide_index=True, width="stretch")
        st.markdown("""
- **The classifier does not transfer.** Trained on synthetic drill parts, it sends almost every real electronics line to review. Rules and Claude carry this test. A real deployment needs the client's own confirmed matches to train on.
- **With the part number in the text, the rule that looks for it does most of the work**, and precision stays at 0.99.
- **With the part number hidden, recall is low (0.28) and precision holds (0.97).** A hobbyist BOM line such as "Red LED 5mm" does not carry enough to pick one catalog part, and the tool declines instead of guessing.
- **Real no-match lines are rejected correctly more than 99% of the time** after Claude. The two or three accepted in error were accepted by the classifier, not by Claude.
""")

    st.subheader("Experiment: self-training without truth labels")
    st.caption("Can the classifier be trained on a dataset's own rule-matched lines, with no confirmed labels? Positives are the lines a rule "
               "matched; negatives are their other nearest candidates. Part-number features are hidden and the matched number is stripped from the text.")
    sa, sb = evals.get("eval_selftrain"), evals.get("eval_external_selftrain")
    if sa:
        fit_a = sa["self_train_fit"]
        st.markdown(f"**(a) Synthetic data, held-out split.** Self-trained on {fit_a['train_lines']:,} rule-matched lines.")
        table = []
        for name, r in sa["results"].items():
            for stage_name, key in (("Rules + classifier", "classifier_stage"), ("Rules + classifier + Claude", "with_claude")):
                h = r[key]["heldout"]
                table.append({"Classifier": name, "Stage": stage_name, "Precision": f"{h['accept_precision']:.2%}", "Recall": pct(h["recall"]),
                              "Review queue": f"{h['review_lines']} lines, {usd(h['review_spend'])}", "Wrong-match $": usd(h["wrong_match_spend"])})
        st.dataframe(pd.DataFrame(table), hide_index=True, width="stretch")
    if sb:
        fit_b = sb["self_train_fit"]
        st.markdown(f"**(b) Real parts.** Trained on {fit_b['train_lines']} rule-matched lines from half the parts ({fit_b['positives']} usable positives); "
                    f"tested on the other half with the part number hidden, plus 500 no-match lines.")
        table = []
        for name, r in sb["results"].items():
            for stage_name, key in (("Rules + classifier", "classifier_stage"), ("Rules + classifier + Claude", "with_claude")):
                m = r[key]
                table.append({"Classifier": name, "Stage": stage_name, "Right": f"{m['right']} of {m['test_lines']}", "Wrong": m["wrong"],
                              "Precision": f"{m['precision']:.2f}", "Recall": f"{m['recall']:.2f}", "Review queue": f"{m['review_queue_lines']} lines",
                              "No-match lines wrongly accepted": m["no_match_wrongly_accepted"]})
        st.dataframe(pd.DataFrame(table), hide_index=True, width="stretch")
        st.caption("The catalog has prices but no quantities, so wrong-match dollars cannot be computed for the real-parts test.")
    if sa or sb:
        st.markdown("""
- **It does not replace confirmed labels.** On the synthetic data the self-trained classifier accepts more lines, but its precision falls from 100% to about 97.4% and it produces wrong-match dollars where the truth-trained one has none. Rule-matched lines always have a true part, so the model never sees a line that should be left unmatched.
- **On real parts it barely moves.** Only a third of the training lines had their true part among the five nearest candidates, the thresholds collapse to "accept almost nothing", and recall goes from 0.00 to 0.02. Its one gain: it stops wrongly accepting no-match lines.
- **Conclusion:** self-training is not a shortcut. The classifier needs reviewer-confirmed matches, which is what the crosswalk collects.
""")

    st.subheader("Limits")
    st.markdown("""
- **The data is synthetic.** Accuracy is measured on noise this project generated. Real exports are messier in ways the generator does not model, so treat these numbers as a ceiling, not a forecast.
- **The classifier is specific to the data it was trained on.** On real electronics parts it abstained on almost everything (see the external tests above).
- **One product family.** Four drill models and 276 parts. A real manufacturer has thousands of parts and many near-identical ones.
- **Allocation is an estimate.** Spend is split by BOM quantity and build volume. It ignores inventory, scrap and timing between buying and building.
- **Buckets lean on the PO type.** Where it is blank, the bucket is inferred from the match and the plant. A blank, unmatched spare is counted as unassigned.
- **Duplicate detection is narrow.** It finds part numbers whose descriptions use the same words and numbers. Duplicates described differently are missed.
- **Leads are not savings.** A price gap can have a good reason. Each one needs a buyer's judgment before it means anything.
- **Uploaded files skip the classifier.** They are matched with rules, embeddings and Claude, and there is no ground truth to score the result against.
- **The crosswalk is a local file.** Reviewer decisions are stored in crosswalk.csv and reused as the first rule; a real deployment would also retrain the classifier on them.
- **Trends and inventory are rough.** They have no opening stock, scrap, or lead time between buying and building.
- **What a pilot would test:** match accuracy on one plant's real exports, how much spend lands in review, and how long a reviewer takes to clear it.
""")

# ------------------------------------------------------- trends and inventory
with tab_trend:
    st.subheader("Tie-out by quarter")
    tq = lines.assign(un=lines["spend_usd"].where(lines["bucket"] == "unassigned", 0.0)).groupby("quarter")[["un", "spend_usd"]].sum().reset_index()
    tq["tie_out"] = 1 - tq["un"] / tq["spend_usd"]
    st.altair_chart(
        alt.Chart(tq).mark_line(color=SERIES[0], strokeWidth=2, point=alt.OverlayMarkDef(size=70, color=SERIES[0])).encode(
            x=alt.X("quarter:N", title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("tie_out:Q", axis=alt.Axis(format="%", title="Spend tied out"), scale=alt.Scale(zero=False)),
            tooltip=[alt.Tooltip("quarter:N", title="Quarter"), alt.Tooltip("tie_out:Q", title="Tied out", format=".1%"),
                     alt.Tooltip("spend_usd:Q", title="Spend", format="$,.0f")]).properties(height=220), width="stretch")
    st.caption("Use the date range in the sidebar to narrow every tab to a period.")

    st.subheader("Estimated actual versus standard cost per unit, by quarter")
    units_q = builds.assign(quarter=quarter_of(builds["period"])).groupby(["product_id", "quarter"])["units_built"].sum()
    aq = alloc.groupby(["product_id", "quarter"])["allocated_usd"].sum().to_frame("spend").join(units_q, how="inner").reset_index()
    aq = aq[aq["units_built"] > 0].merge(products[["product_id", "std_cost_per_unit"]], on="product_id")
    aq["vs_std"] = aq["spend"] / aq["units_built"] / aq["std_cost_per_unit"] - 1
    aq = aq.dropna(subset=["vs_std"])
    if aq.empty:
        st.info("This needs a standard cost on the BOM and at least one quarter with units built.")
    else:
        product_ids = sorted(aq["product_id"].unique())
        st.altair_chart(
            alt.Chart(aq).mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=60)).encode(
                x=alt.X("quarter:N", title=None, axis=alt.Axis(labelAngle=0)),
                y=alt.Y("vs_std:Q", axis=alt.Axis(format="+%", title="Estimated actual vs standard")),
                color=alt.Color("product_id:N", scale=alt.Scale(domain=product_ids, range=SERIES[:len(product_ids)]), legend=alt.Legend(title=None, orient="top")),
                tooltip=[alt.Tooltip("product_id:N", title="Product"), alt.Tooltip("quarter:N", title="Quarter"),
                         alt.Tooltip("vs_std:Q", title="Actual vs standard", format="+.1%"), alt.Tooltip("units_built:Q", title="Units built", format=",")],
            ).properties(height=260), width="stretch")
        st.caption("An estimate. Parts are bought ahead of the builds that use them, so a single quarter can swing; lines still in review are not included.")

    st.subheader("Estimated inventory: bought minus built")
    st.caption("An estimate, not a stock count: cumulative production purchases minus BOM quantity x units built, over the selected date range. "
               "It ignores opening stock, scrap and parts issued to service.")
    need = bom.groupby(["part_number", "product_id"])["qty_per_unit"].sum().reset_index().merge(units_q.reset_index(), on="product_id")
    need["need"] = need["qty_per_unit"] * need["units_built"]
    need_q = need.groupby(["part_number", "quarter"])["need"].sum()
    bought = lines[(lines["decision"] == "accept") & (lines["bucket"] == "production")]
    bought_q = bought.groupby(["part", "quarter"])["base_qty"].sum().rename_axis(["part_number", "quarter"])
    inv = pd.DataFrame({"bought": bought_q.groupby("part_number").sum(), "needed": need_q.groupby("part_number").sum()}).fillna(0.0)
    price = bought.groupby("part")["spend_usd"].sum() / bought.groupby("part")["base_qty"].sum()
    inv["units"] = inv["bought"] - inv["needed"]
    inv["value"] = inv["units"] * price.reindex(inv.index)
    inv = inv.dropna(subset=["value"])
    if inv.empty:
        st.info("No matched production purchases in this range.")
    else:
        top = inv.reindex(inv["value"].abs().sort_values(ascending=False).index).head(15)
        st.dataframe(pd.DataFrame({
            "Part": top.index, "Description": top.index.map(part_desc), "Bought": top["bought"].map("{:,.0f}".format),
            "Needed for builds": top["needed"].map("{:,.0f}".format), "Est. units on hand": top["units"].map("{:+,.0f}".format),
            "Est. value": top["value"].map(lambda v: f"{'-' if v < 0 else ''}${abs(v):,.0f}"),
            "Note": top["units"].map(lambda u: "bought less than built: opening stock, or purchases not yet matched" if u < 0 else "")}),
            hide_index=True, width="stretch")
        part_pick = st.selectbox("Cumulative bought versus needed for", list(top.index), format_func=lambda x: f"{x}  {part_desc.get(x, '')}")
        qs = sorted(lines["quarter"].unique())
        cum = pd.DataFrame({"Bought": bought_q.xs(part_pick).reindex(qs).fillna(0.0).cumsum() if part_pick in bought_q.index.get_level_values(0) else 0.0,
                            "Needed for builds": need_q.xs(part_pick).reindex(qs).fillna(0.0).cumsum() if part_pick in need_q.index.get_level_values(0) else 0.0},
                           index=pd.Index(qs, name="quarter")).reset_index().melt("quarter", var_name="series", value_name="units")
        st.altair_chart(
            alt.Chart(cum).mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=60)).encode(
                x=alt.X("quarter:N", title=None, axis=alt.Axis(labelAngle=0)), y=alt.Y("units:Q", axis=alt.Axis(format=",", title="Cumulative units")),
                color=alt.Color("series:N", scale=alt.Scale(domain=["Bought", "Needed for builds"], range=SERIES[:2]), legend=alt.Legend(title=None, orient="top")),
                tooltip=[alt.Tooltip("series:N", title="Series"), alt.Tooltip("quarter:N", title="Quarter"), alt.Tooltip("units:Q", title="Cumulative units", format=",.0f")],
            ).properties(height=240), width="stretch")

# ------------------------------------------------------------------- upload
with tab_upload:
    st.subheader("Run it on your own exports")
    st.caption("CSV or Excel. Column names do not need to match ours: Claude proposes a mapping and you confirm it. "
               "This step needs ANTHROPIC_API_KEY (and uses OPENAI_API_KEY for embeddings if present). It sends the headers and 20 sample rows of "
               "each file to Claude for the mapping, then purchase descriptions and BOM parts to the embedding model and Claude for matching.")
    up = st.session_state.setdefault("up_files", {})
    for col, kind in zip(st.columns(3), KIND_LABEL):
        f = col.file_uploader(KIND_LABEL[kind], type=["csv", "xlsx", "xls"], key=f"file_{kind}")
        if f is not None:
            up[kind] = (f.name, f.getvalue())
        if kind in up:
            col.caption(f"Loaded: {up[kind][0]}")
    if st.button("Load the second sample: the same kind of data with different column names"):
        for kind, name in SAMPLE_FILES.items():
            up[kind] = (name, (SAMPLE_DIR / name).read_bytes())
        st.session_state.pop("up_map", None)
        st.rerun()

    if len(up) == 3:
        raw = {kind: mapping.read_table(*up[kind]) for kind in KIND_LABEL}
        if st.button("Ask Claude to map the columns", type="secondary" if "up_map" in st.session_state else "primary"):
            try:
                with st.spinner("Claude is reading the headers and 20 sample rows of each file"):
                    proposals = {kind: mapping.propose(kind, raw[kind])[0] for kind in KIND_LABEL}
            except Exception as e:  # most often a missing API key
                st.error(f"Could not get a mapping from Claude: {e}")
            else:
                st.session_state["up_map"] = proposals
                for kind, prop in proposals.items():
                    for row in prop.itertuples():
                        st.session_state[f"map_{kind}_{row.canonical}"] = row.source or "(none)"
                st.rerun()

        if "up_map" in st.session_state:
            st.markdown("**Confirm or fix the mapping**")
            chosen, problems = {}, []
            for kind, prop in st.session_state["up_map"].items():
                chosen[kind] = {}
                with st.expander(f"{KIND_LABEL[kind]}: {up[kind][0]}, {len(raw[kind]):,} rows", expanded=True):
                    st.dataframe(raw[kind].head(5), hide_index=True, width="stretch")
                    grid = st.columns(4)
                    for i, row in enumerate(prop.itertuples()):
                        with grid[i % 4]:
                            pick = st.selectbox(f"{row.canonical} ({'required' if row.required else 'optional'})", ["(none)"] + list(raw[kind].columns),
                                                key=f"map_{kind}_{row.canonical}", help=row.meaning)
                            st.caption(f"Claude: {row.confidence:.0%} sure. {row.note}" if row.source else f"Claude found no column. {row.note}")
                            chosen[kind][row.canonical] = None if pick == "(none)" else pick
                            if row.required and pick == "(none)":
                                problems.append(f"{KIND_LABEL[kind]}: {row.canonical} is required")
                    used = [v for v in chosen[kind].values() if v]
                    if len(used) != len(set(used)):
                        problems.append(f"{KIND_LABEL[kind]}: one source column is used twice")
            for problem in problems:
                st.error(problem)
            st.info("On uploaded data the matcher uses rules, embeddings and Claude only. The classifier is skipped until this company has "
                    "confirmed matches to train it on. Decisions already in the crosswalk are applied first.")
            if st.button("Run the pipeline", type="primary", disabled=bool(problems)):
                try:
                    with st.spinner("Matching: rules, then embeddings, then Claude on the open lines. A minute or two the first time; cached after that."):
                        canon, notes = {}, []
                        for kind in KIND_LABEL:
                            canon[kind], kind_notes = mapping.apply_mapping(kind, raw[kind], chosen[kind])
                            notes += kind_notes
                        inp = prepare(canon["purchases"], canon["boms"], canon["builds"])
                        dec, stats = pipeline.run(inp, crosswalk.as_rules(crosswalk.load()))
                        st.session_state["uploaded"] = dict(
                            lines=allocate.build_lines(inp, dec), bom=inp.boms, builds=inp.builds, stats=stats, notes=notes,
                            files=[up[kind][0] for kind in KIND_LABEL], demo_sample=all(up[kind][0] == SAMPLE_FILES[kind] for kind in KIND_LABEL))
                        st.session_state["_switch_to_uploaded"] = True
                except Exception as e:
                    st.error(f"The run failed: {e}")
                else:
                    st.rerun()
    if uploaded:
        st.success(f"Last run: {', '.join(uploaded['files'])}: {uploaded['stats']['lines']:,} purchase lines. Select \"Uploaded files\" under Data in the sidebar to view it.")

