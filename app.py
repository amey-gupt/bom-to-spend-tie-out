"""BOM-to-Spend Tie-Out: the demo app.

    uv run streamlit run app.py

The built-in sample reads out/*.csv and out/eval_*.json only: no API keys, no model loading.
The "Your data" tab runs the matcher on a company's own files and does need API keys.
Reviewer decisions are saved (crosswalk.csv) and applied to every line with the same text.
"""
from __future__ import annotations

import html
import json
from datetime import date, timedelta
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from src import allocate, crosswalk, mapping, pipeline
from src.normalize import norm_desc, norm_pn, numbers_in, prepare

OUT = Path("out")
SAMPLE_DIR = Path("samples/second_company")
SAMPLE_FILES = {"boms": "bom_export.csv", "purchases": "po_lines.xlsx", "builds": "production_output.csv"}
KIND_LABEL = {"boms": "Bill of materials", "purchases": "Purchase order lines", "builds": "Production volumes"}
UPLOAD_LIMIT_MB = 5  # also set as server.maxUploadSize in .streamlit/config.toml
BUCKETS = ["production", "spares", "warranty", "unassigned"]
BLUE, ORANGE, AQUA, YELLOW, VIOLET, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6", "#9a9890"
BUCKET_COLORS = [BLUE, AQUA, YELLOW, GRAY]  # unassigned is neutral: it is the absence of a tie
SERIES = [BLUE, AQUA, YELLOW, VIOLET]       # orange is kept for warnings and wrong matches only
SURFACE = "#ffffff"
HOW = {
    "rule_pn_exact": "Rule: part number", "rule_xref": "Rule: supplier cross-reference", "rule_xref_offbom": "Rule: cross-reference to a part on no BOM",
    "rule_pn_in_text": "Rule: part number in the description", "rule_crosswalk": "Saved decision", "rule_crosswalk_none": "Saved decision",
    "classifier": "Model", "embedding": "Nearest description only", "llm": "Claude", "llm_material": "Claude, waiting for sign-off",
    "uom_mismatch": "Held: bought in a unit that cannot be converted", "crosswalk": "Saved decision",
}
MAIN_MODEL, SMALL_MODEL = "claude-opus-5-5", "claude-haiku-5-5"
SYNTHETIC_NOTICE = ("The products, parts, suppliers and prices are invented, and the true matches are planted so accuracy can be measured. "
                    "Nothing here comes from a real company.")
PAGE = 12  # rows per table page: tables are sized to their content so the mouse wheel scrolls the page

st.set_page_config(page_title="BOM-to-Spend Tie-Out", layout="wide")
st.markdown("""<style>
.block-container {padding-top: 1.4rem; padding-bottom: 2rem;}
header[data-testid="stHeader"] {background: transparent; height: 2.2rem;}
.app-title {font-size: 1.15rem; font-weight: 650; line-height: 1.3; margin-bottom: .15rem;}
.kpi-label {font-size: .78rem; color: #52514e; line-height: 1.2;}
.kpi-value {font-size: 1.5rem; font-weight: 650; line-height: 1.25; color: #0b0b0b;}
.kpi-note {font-size: .76rem; color: #52514e; line-height: 1.2;}
.pair {border: 1px solid #e4e3df; border-radius: 8px; padding: .55rem .8rem; margin-bottom: .5rem; font-size: .95rem;}
.pair .who {font-size: .74rem; color: #52514e;}
.tag {display: inline-block; font-size: .74rem; padding: .05rem .45rem; border-radius: 999px; background: #eef3fb; color: #1c5cab; margin-right: .3rem;}
</style>""", unsafe_allow_html=True)


def usd(x: float) -> str:
    return "n/a" if pd.isna(x) else f"${x:,.0f}"


def short(x: float) -> str:
    """$29.7M, $1.9M, $415K: for headlines. The full value goes in a tooltip."""
    if pd.isna(x):
        return "n/a"
    sign, x = ("-" if x < 0 else ""), abs(x)
    if x >= 1e6:
        return f"{sign}${x / 1e6:.1f}M"
    return f"{sign}${x / 1e3:.0f}K" if x >= 1e3 else f"{sign}${x:.0f}"


def pct(x: float) -> str:
    return "n/a" if pd.isna(x) else f"{x:.1%}"


def kpi(label: str, value: str, hover: str, note: str = "") -> str:
    return (f'<div title="{html.escape(hover)}"><div class="kpi-label">{html.escape(label)}</div>'
            f'<div class="kpi-value">{html.escape(value)}</div><div class="kpi-note">{html.escape(note)}</div></div>')


def quarter_of(dates: pd.Series) -> pd.Series:
    d = pd.to_datetime(dates)
    return d.dt.year.astype(str) + "Q" + d.dt.quarter.astype(str)


def table(df: pd.DataFrame, key: str, select: bool = False, page_size: int = PAGE):
    """A table sized to its rows, paged when long. With select=True returns the picked row's position in df, or None."""
    pages = max(1, -(-len(df) // page_size))
    page = min(st.session_state.get(f"{key}_page", 0), pages - 1)
    view = df.iloc[page * page_size:(page + 1) * page_size]
    picked = None
    if select:
        event = st.dataframe(view, hide_index=True, width="stretch", height="content", on_select="rerun", selection_mode="single-row", key=f"{key}_p{page}")
        picked = page * page_size + event.selection.rows[0] if event.selection.rows else None
    else:
        st.dataframe(view, hide_index=True, width="stretch", height="content")
    if pages > 1:
        a, b, c = st.columns([1, 2, 1])
        if a.button("Previous", key=f"{key}_prev", disabled=page == 0):
            st.session_state[f"{key}_page"] = page - 1
            st.rerun()
        b.caption(f"Page {page + 1} of {pages} ({len(df)} rows)")
        if c.button("Next", key=f"{key}_next", disabled=page >= pages - 1):
            st.session_state[f"{key}_page"] = page + 1
            st.rerun()
    return picked


def mark_shared(text: str, other: str) -> str:
    """The text with every word or number it shares with `other` in bold."""
    shared, numbers = set(norm_desc(other).split()), numbers_in(norm_desc(other))
    out = []
    for token in str(text).split():
        n = norm_desc(token)
        hit = bool(set(n.split()) & shared) or bool(numbers_in(n) & numbers)
        out.append(f"<b>{html.escape(token)}</b>" if hit else html.escape(token))
    return " ".join(out)


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
    """The whole downstream state for one sign-off threshold, one set of saved decisions and one date range."""
    lines = base.copy()
    lines["decision"], lines["source"] = lines["decision_pre"], lines["source_pre"]
    lines = allocate.apply_materiality(lines, materiality)
    lines = allocate.apply_crosswalk(lines, {k: (d, part) for k, d, part in rules})
    lines = allocate.apply_units(lines, bom)
    lines = lines[(lines["quarter"] >= q_from) & (lines["quarter"] <= q_to)].reset_index(drop=True)
    in_range = quarter_of(builds["period"]).between(q_from, q_to)
    lines = allocate.assign_buckets(lines)
    alloc, products = allocate.allocate(lines, bom, builds[in_range])
    return lines, alloc, products, allocate.find_leads(lines, bom, builds[in_range])


@st.cache_data
def requirements(bom: pd.DataFrame, builds: pd.DataFrame, q_from: str, q_to: str) -> pd.DataFrame:
    """Gross requirement of every bought part per product and build month, using the BOM in effect that month."""
    return allocate.requirements(bom, builds[quarter_of(builds["period"]).between(q_from, q_to)])


def engineering_changes(bom: pd.DataFrame) -> list[tuple[str, str, str]]:
    """(date, product, what changed) for every BOM row that came into effect on a date."""
    if "effective_from" not in bom:
        return []
    b = bom.assign(effective_from=bom["effective_from"].fillna("").astype(str), effective_to=bom["effective_to"].fillna("").astype(str))
    out = []
    for r in b[b["effective_from"] != ""].itertuples():
        day_before = (date.fromisoformat(r.effective_from) - timedelta(days=1)).isoformat()
        old = b[(b["product_id"] == r.product_id) & (b["effective_to"] == day_before) & (b["assembly_id"] == r.assembly_id)]
        same = old[old["part_number"] == r.part_number]
        if len(same):
            what = f"{r.part_number} quantity {same['qty_per_unit'].iat[0]:g} to {r.qty_per_unit:g}"
        else:
            what = f"{old['part_number'].iat[0]} replaced by {r.part_number}" if len(old) else f"{r.part_number} added"
        out.append((r.effective_from, r.product_id, what))
    return sorted(out)


sample_base, sample_bom, sample_builds, evals = load()
if st.session_state.pop("_switch_to_uploaded", False):
    st.session_state["dataset"] = "Your files"
uploaded = st.session_state.get("uploaded")

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Settings")
    options = ["Built-in sample"] + (["Your files"] if uploaded else [])
    if st.session_state.get("dataset") not in options:
        st.session_state["dataset"] = options[0]
    dataset = st.radio("Data", options, key="dataset")
    is_sample = dataset == "Built-in sample"
    base, bom, builds = (sample_base, sample_bom, sample_builds) if is_sample else (uploaded["lines"], uploaded["bom"], uploaded["builds"])

    materiality = st.slider("Needs sign-off above", 0, 200_000, int(allocate.MATERIALITY_USD), 5_000, format="$%d",
                            help="A match that only Claude vouched for waits for a person when the line is worth more than this.")
    quarters = sorted(base["quarter"].unique())
    q_from, q_to = (st.select_slider("Date range", options=quarters, value=(quarters[0], quarters[-1]), key=f"quarters_{dataset}")
                    if len(quarters) > 1 else (quarters[0], quarters[0]))
    st.caption("The date range narrows every tab to those quarters.")

xw = crosswalk.load()
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

# How far reviewing the largest lines would move tie-out, if each is approved. A line already tied by its PO type gains nothing.
gain = queue["spend_usd"].where(queue["bucket"] == "unassigned", 0.0)
top_n = int((gain.cumsum() < 0.8 * gain.sum()).sum()) + 1 if gain.sum() > 0 else 0
reach = tied + gain.head(top_n).sum() / total

# ------------------------------------------------------------------- header
h_title, h1, h2, h3 = st.columns([2.2, 1, 1, 1])
with h_title:
    st.markdown('<div class="app-title">BOM-to-Spend Tie-Out</div>', unsafe_allow_html=True)
    if is_sample or uploaded.get("demo_sample"):
        st.badge("Synthetic data", icon=":material/science:", color="orange", help=SYNTHETIC_NOTICE)
    else:
        st.badge("Your files", icon=":material/upload_file:", color="blue", help=", ".join(uploaded["files"]))
moved = (tied - tied_start) * 100
h1.markdown(kpi("Tie-out rate", pct(tied), f"{usd(total - by_bucket['unassigned'])} of {usd(total)} is tied to production, spares or warranty",
                f"{moved:+.1f} pts from your decisions and settings" if abs(moved) > 1e-9 else ""), unsafe_allow_html=True)
h2.markdown(kpi("Waiting for a decision", f"{len(queue)} lines", f"{usd(queue['spend_usd'].sum())} across {len(queue)} purchase lines",
                short(queue["spend_usd"].sum())), unsafe_allow_html=True)
h3.markdown(kpi("Total spend", short(total), f"{usd(total)} across {len(lines):,} purchase lines", f"{len(lines):,} purchase lines"), unsafe_allow_html=True)
st.markdown(f"**{pct(tied)}** of spend is tied to a product, and **{len(queue)}** lines are waiting for a decision. "
            + (f"Review the **{top_n}** largest lines to reach **{pct(reach)}**." if top_n else "Nothing is waiting for review."),
            help="Assumes each of those lines is approved. Lines already tied through their order type add nothing.")

tab_overview, tab_review, tab_cost, tab_leads, tab_evidence, tab_data = st.tabs(
    ["Overview", f"Review ({len(queue)})", "Product cost", "Savings leads", "Evidence", "Your data"])

# ----------------------------------------------------------------- overview
with tab_overview:
    st.subheader("Where the spend ties")
    for col, b in zip(st.columns(4), BUCKETS):
        col.markdown(kpi(b.capitalize(), pct(by_bucket[b] / total), usd(by_bucket[b]), short(by_bucket[b])), unsafe_allow_html=True)
    names_b = [b.capitalize() for b in BUCKETS]
    bar = pd.DataFrame({"bucket": names_b, "spend": by_bucket.to_numpy(), "share": (by_bucket / total).to_numpy(), "order": range(4)})
    st.altair_chart(
        alt.Chart(bar).mark_bar(height=28, stroke=SURFACE, strokeWidth=2).encode(
            x=alt.X("spend:Q", stack="normalize", axis=alt.Axis(format="%", title="Share of spend", grid=False)),
            color=alt.Color("bucket:N", scale=alt.Scale(domain=names_b, range=BUCKET_COLORS), legend=alt.Legend(title=None, orient="top")),
            order="order:Q",
            tooltip=[alt.Tooltip("bucket:N", title="Bucket"), alt.Tooltip("spend:Q", title="Spend", format="$,.0f"),
                     alt.Tooltip("share:Q", title="Share", format=".1%")],
        ).properties(height=90), width="stretch")
    inferred = lines[lines["bucket_inferred"]]
    st.caption(f"Unassigned includes the {len(queue)} lines waiting for a decision. Where the order type was blank, the bucket was inferred from the "
               f"match and the plant: {len(inferred):,} lines ({short(inferred['spend_usd'].sum())}).")

    st.subheader("Tie-out by quarter")
    tq = lines.assign(un=lines["spend_usd"].where(lines["bucket"] == "unassigned", 0.0)).groupby("quarter")[["un", "spend_usd"]].sum().reset_index()
    tq["tie_out"] = 1 - tq["un"] / tq["spend_usd"]
    st.altair_chart(
        alt.Chart(tq).mark_line(color=SERIES[0], strokeWidth=2, point=alt.OverlayMarkDef(size=70, color=SERIES[0])).encode(
            x=alt.X("quarter:N", title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("tie_out:Q", axis=alt.Axis(format="%", title="Spend tied out"), scale=alt.Scale(zero=False)),
            tooltip=[alt.Tooltip("quarter:N", title="Quarter"), alt.Tooltip("tie_out:Q", title="Tied out", format=".1%"),
                     alt.Tooltip("spend_usd:Q", title="Spend", format="$,.0f")]).properties(height=220), width="stretch")

    if is_sample:
        st.subheader("From a spreadsheet lookup to the full pipeline")
        s2 = evals["eval_step2_local"]["splits"]["heldout"]
        s3 = evals["eval_step3"]
        clf, full = s3["backends"][s3["chosen_backend"]]["heldout"], s3["llm"][MAIN_MODEL]["heldout"]
        true_rate = full["true_tie_out"]
        stages = pd.DataFrame([
            ("1. Exact part-number lookup", s2["(a) naive exact part number"]["tie_out"], s2["(a) naive exact part number"]["precision"],
             s2["(a) naive exact part number"]["recall"], None, s2["(a) naive exact part number"]["wrong_match_spend"]),
            ("2. Rules: cleaned part numbers, cross-reference", s2["(b) rules only"]["tie_out"], s2["(b) rules only"]["precision"],
             s2["(b) rules only"]["recall"], None, s2["(b) rules only"]["wrong_match_spend"]),
            ("3. Rules + nearest description, fixed cutoff", s2["(c) rules + embeddings"]["tie_out"], s2["(c) rules + embeddings"]["precision"],
             s2["(c) rules + embeddings"]["recall"], None, s2["(c) rules + embeddings"]["wrong_match_spend"]),
            ("4. Rules + trained model", clf["tie_out"], clf["accept_precision"], clf["recall"], clf["review_spend"], clf["wrong_match_spend"]),
            ("5. Rules + trained model + Claude", full["tie_out"], full["accept_precision"], full["recall"], full["review_spend"], full["wrong_match_spend"]),
        ], columns=["stage", "tie_out", "precision", "recall", "review_spend", "wrong"])
        kinds = ["No wrong matches", "Wrong matches"]
        stages["kind"] = stages["wrong"].map(lambda w: kinds[1] if w > 0 else kinds[0])
        stages["label"] = [f"{t:.1%}" + (f"   {short(w)} wrong" if w > 0 else "") for t, w in zip(stages["tie_out"], stages["wrong"])]
        bars = alt.Chart(stages).mark_bar(cornerRadiusEnd=4).encode(
            y=alt.Y("stage:N", sort=None, title=None, axis=alt.Axis(labelLimit=360), scale=alt.Scale(paddingInner=0.3, paddingOuter=0.15)),
            x=alt.X("tie_out:Q", scale=alt.Scale(domain=[0, 1.12]), axis=alt.Axis(format="%", title="Spend tied out (held-out lines)", values=[0, .2, .4, .6, .8, 1])),
            color=alt.Color("kind:N", scale=alt.Scale(domain=kinds, range=[BLUE, ORANGE]), legend=alt.Legend(title=None, orient="top")),
            tooltip=[alt.Tooltip("stage:N", title="Stage"), alt.Tooltip("tie_out:Q", title="Tied out", format=".1%"),
                     alt.Tooltip("precision:Q", title="Precision", format=".1%"), alt.Tooltip("recall:Q", title="Recall", format=".1%"),
                     alt.Tooltip("wrong:Q", title="Wrong-match dollars", format="$,.0f")])
        labels = alt.Chart(stages).mark_text(align="left", dx=6, color="#0b0b0b").encode(y=alt.Y("stage:N", sort=None), x="tie_out:Q", text="label:N")
        truth_df = pd.DataFrame({"x": [true_rate], "label": [f"true rate {true_rate:.1%}"]})
        rule = alt.Chart(truth_df).mark_rule(strokeDash=[4, 4], color="#52514e").encode(x="x:Q", tooltip=alt.Tooltip("label:N", title="Planted truth"))
        st.altair_chart((bars + labels + rule).properties(height=alt.Step(34)), width="stretch")
        st.caption(f"Dashed line: the true tie-out rate on these lines, {true_rate:.1%}. Measured on {full['lines']:,} held-out purchase lines "
                   f"({short(full['spend'])}) written in styles never used to tune the matcher. Stage 4 ties less than stage 3 because it refuses "
                   "to guess: uncertain lines wait for review instead.")
        st.table(pd.DataFrame({
            "Stage": stages["stage"], "Tied out": stages["tie_out"].map(pct), "Precision": stages["precision"].map(pct),
            "Recall": stages["recall"].map(pct), "Left for review": stages["review_spend"].map(lambda v: "n/a" if pd.isna(v) else short(v)),
            "Wrong-match dollars": stages["wrong"].map(usd)}).set_index("Stage"))
    else:
        st.subheader("How this run was matched")
        stats = uploaded["stats"]
        st.markdown(f"{stats['lines']:,} purchase lines against {stats['bom_parts']:,} BOM parts. Rules settled {stats['settled_by_rules']:,} lines; "
                    f"{stats['sent_to_claude']:,} went to Claude with their five nearest BOM parts ({stats['claude']['calls']} distinct calls, "
                    f"${stats['claude']['cost_usd']:.2f} at list price).")
        how = lines.assign(how=lines["source"].map(HOW).fillna(lines["source"])).groupby(["decision", "how"]).agg(
            lines=("po_id", "size"), spend=("spend_usd", "sum")).reset_index()
        st.dataframe(pd.DataFrame({"Outcome": how["decision"], "Decided by": how["how"], "Lines": how["lines"], "Spend": how["spend"].map(usd)}),
                     hide_index=True, width="stretch", height="content")
        for note in uploaded["notes"]:
            st.caption(note)
        st.caption("Your own files have no answer key, so accuracy cannot be reported here. The Evidence tab shows how the same steps score "
                   "on the built-in sample and on outside data.")

# ------------------------------------------------------------------- review
with tab_review:
    if queue.empty:
        st.success("Nothing is waiting for review.")
    else:
        def tags(r) -> list[str]:
            if r["source"] == "uom_mismatch":
                return ["Unit mismatch"]
            if r["source"] == "llm_material":
                return [f"Over {short(materiality)}"]
            if pd.notna(r["llm_conf"]):
                return ["Low confidence"]
            return ["Low confidence"] if r["source"] == "classifier" else ["Not judged"]

        q = queue.assign(tag=queue.apply(lambda r: ", ".join(tags(r)), axis=1), gain=gain)
        groups = q.groupby(["supplier_part_number", "item_description"], sort=False).agg(
            lines=("po_id", "size"), spend=("spend_usd", "sum"), gain=("gain", "sum"), suggested=("suggested_part", "first"),
            tag=("tag", lambda s: ", ".join(dict.fromkeys(s))), first=("po_id", "first")).reset_index().sort_values("spend", ascending=False).reset_index(drop=True)
        pos = min(st.session_state.get("review_pos", 0), len(groups) - 1)

        left, right = st.columns([5, 6], gap="large")
        with left:
            st.markdown(f"**{len(groups)} items** covering {len(queue)} lines, largest dollars first")
            picked = table(pd.DataFrame({
                "Open": ["▶" if i == pos else "" for i in range(len(groups))], "Total": groups["spend"].map(short), "Lines": groups["lines"],
                "Why": groups["tag"], "Suggested part": groups["suggested"], "Text on the PO": groups["item_description"]}),
                key=f"queue_{len(xw)}", select=True)
            if picked is not None and picked != pos:
                st.session_state["review_pos"] = picked
                st.rerun()
            st.caption("Lines with identical text are one item: one decision settles all of them.")

        g = groups.iloc[pos]
        r = q[q["po_id"] == g["first"]].iloc[0]
        sug = g["suggested"]
        with right:
            st.markdown(f"Approving settles **{g['lines']} {'line' if g['lines'] == 1 else 'lines'}** ({usd(g['spend'])}) and moves tie-out by "
                        f"**{g['gain'] / total * 100:+.2f} pts**.")
            c1, c2, c3 = st.columns([1.5, 1.1, 1.3])
            for col, label, action, kind in ((c1, f"Approve: {sug}", "approve", "primary"), (c2, "Reject suggestion", "reject", "secondary"),
                                             (c3, "Not a BOM part", "no_match", "secondary")):
                if col.button(label, type=kind, key=f"{action}_{g['first']}", width="stretch"):
                    crosswalk.add(r["supplier_part_number"], r["item_description"], sug, action, dataset)
                    st.rerun()  # the list shifts up, so the same position now holds the next item

            bom_text = part_desc.get(sug, "")
            st.markdown(
                f'<div class="pair"><div class="who">Text on the PO</div>{mark_shared(r["item_description"], bom_text)}</div>'
                f'<div class="pair"><div class="who">BOM description of {html.escape(str(sug))}</div>{mark_shared(bom_text, r["item_description"])}</div>',
                unsafe_allow_html=True)
            st.markdown("".join(f'<span class="tag">{html.escape(t)}</span>' for t in g["tag"].split(", ")), unsafe_allow_html=True)
            if r["source"] == "uom_mismatch":
                st.warning(r["uom_issue"])
            if r["llm_reason"]:
                answer = f"`{r['llm_part']}`" if r["llm_part"] else "no BOM part"
                st.info(f"**Claude's answer: {answer}, confidence {r['llm_conf']:.0%}.** {r['llm_reason']}")
            st.markdown(f"**{'Model confidence' if is_sample else 'Closeness of the nearest BOM description'}:** {r['prob']:.0%}")
            if sug in ref.index:
                ref_price, ref_sup = ref.at[sug, "ref_price"], ref.at[sug, "ref_suppliers"]
                st.markdown(f"**Price:** this line is ${r['unit_price_usd']:.4f} per unit; confirmed purchases of `{sug}` have a median of "
                            f"${ref_price:.4f} ({r['unit_price_usd'] / ref_price - 1:+.0%}).")
                st.markdown(f"**Supplier:** this supplier {'has' if r['supplier_key'] in ref_sup else 'has no'} confirmed purchases of `{sug}`. "
                            f"Confirmed suppliers: {', '.join(s.title() for s in ref_sup)}.")
            else:
                st.markdown(f"**Price and supplier:** no confirmed purchases of `{sug}` yet, so there is nothing to compare against.")
            st.markdown(f"**Part-number field** {r['supplier_part_number'] or '(blank)'} &nbsp;·&nbsp; **Supplier** {r['supplier']} &nbsp;·&nbsp; "
                        f"**Plant** {r['plant'] or '(blank)'}  \n**Order date** {r['po_date']} &nbsp;·&nbsp; **Order type** {r['po_type'] or '(blank)'} "
                        f"&nbsp;·&nbsp; **Quantity** {r['qty']:,.2f} {r['uom']} &nbsp;·&nbsp; **Price per unit** ${r['unit_price_usd']:.4f}"
                        + (f"  \nLargest of {g['lines']} lines with this text." if g["lines"] > 1 else ""))
            st.caption("Approve ties these lines to the suggested part. Reject removes the suggestion and leaves them untied for a manual lookup. "
                       "Not a BOM part closes them as correctly unmatched.")

    st.divider()
    s1, s2_, s3_, s4_ = st.columns([3, 1, 1, 1.4])
    s1.markdown(f"**Saved decisions: {len(xw)}.** Each one applies to every line with the same text, now and on the next run.")
    if s2_.button("Undo last", disabled=xw.empty, width="stretch"):
        crosswalk.undo_last()
        st.rerun()
    if s3_.button("Clear all", disabled=xw.empty, width="stretch"):
        crosswalk.clear()
        st.rerun()
    s4_.download_button("Download (CSV)", xw.to_csv(index=False), "saved_decisions.csv", "text/csv", disabled=xw.empty, width="stretch")
    if not xw.empty:
        keys = lines["supplier_part_number"].map(norm_pn) + "|" + lines["item_description"].map(norm_desc)
        settled = keys.value_counts()
        shown_xw = xw.assign(lines_settled=[int(settled.get(f"{norm_pn(a)}|{norm_desc(d)}", 0)) for a, d in zip(xw["supplier_part_number"], xw["item_description"])])
        table(shown_xw.rename(columns={"supplier_part_number": "Part-number field", "item_description": "Text on the PO", "part_number": "BOM part",
                                       "decision": "Decision", "decided_at": "Decided at", "dataset": "Data", "lines_settled": "Lines it settles here"}),
              key="saved")

# ------------------------------------------------------------- product cost
with tab_cost:
    st.subheader("Parts cost per unit: standard versus estimated actual")
    cost = products.melt("product_id", ["std_cost_per_unit", "actual_cost_per_unit_est"], var_name="series", value_name="cost").dropna(subset=["cost"])
    cost["series"] = cost["series"].map({"std_cost_per_unit": "BOM standard", "actual_cost_per_unit_est": "Estimated actual"})
    cost_bars = alt.Chart(cost).mark_bar(cornerRadiusEnd=3).encode(
        y=alt.Y("product_id:N", title=None, scale=alt.Scale(paddingInner=0.35, paddingOuter=0.15)), yOffset=alt.YOffset("series:N", sort=["BOM standard", "Estimated actual"]),
        x=alt.X("cost:Q", axis=alt.Axis(format="$.0f", title="Parts cost per unit built")),
        color=alt.Color("series:N", scale=alt.Scale(domain=["BOM standard", "Estimated actual"], range=[BLUE, AQUA]), legend=alt.Legend(title=None, orient="top")),
        tooltip=[alt.Tooltip("product_id:N", title="Product"), alt.Tooltip("series:N", title="Measure"), alt.Tooltip("cost:Q", title="Per unit", format="$.2f")])
    cost_labels = alt.Chart(cost).mark_text(align="left", dx=5, color="#0b0b0b").encode(
        y=alt.Y("product_id:N"), yOffset=alt.YOffset("series:N", sort=["BOM standard", "Estimated actual"]), x="cost:Q", text=alt.Text("cost:Q", format="$.2f"))
    st.altair_chart((cost_bars + cost_labels).properties(height=alt.Step(20)), width="stretch")
    st.caption("Estimated: matched production spend is split across the products that use each part, by gross BOM quantity x units built in the "
               "purchase's quarter. Purchases show what was bought, not what was consumed. Lines waiting for a decision are not included yet.")
    st.dataframe(pd.DataFrame({
        "Product": products["product_id"], "Units built": products["units_built"].map("{:,}".format),
        "Standard per unit": products["std_cost_per_unit"].map(lambda v: "n/a" if pd.isna(v) else f"${v:.2f}"),
        "Estimated actual per unit": products["actual_cost_per_unit_est"].map("${:.2f}".format),
        "Actual vs standard": products["actual_vs_std"].map(lambda v: "n/a" if pd.isna(v) else f"{v:+.1%}"),
        "Allocated production spend": products["allocated_production_spend_est"].map(usd)}), hide_index=True, width="stretch", height="content")
    st.download_button("Download product cost (CSV)", products.to_csv(index=False), "product_cost.csv", "text/csv")

    product_key, asof_key = f"cost_product_{dataset}", f"bom_as_of_{dataset}"
    changes = engineering_changes(bom)
    day_min, day_max = pd.to_datetime(base["po_date"]).min().date(), pd.to_datetime(base["po_date"]).max().date()
    if changes:
        day_max = max(day_max, date.fromisoformat(changes[-1][0]))
        st.subheader("Engineering changes")
        st.caption("Click one to open that product's BOM as it stood on the day the change took effect.")
        for i, (day_text, prod_id, what) in enumerate(changes):
            if st.button(f"{day_text}   {prod_id}: {what}", key=f"ec_{i}", icon=":material/history:"):
                st.session_state[product_key], st.session_state[asof_key] = prod_id, date.fromisoformat(day_text)

    st.subheader("Parts driving the difference")
    if products["std_cost_per_unit"].isna().all():
        st.info("The BOM has no standard cost, so the parts below are ranked by estimated actual cost per unit.")
    product = st.selectbox("Product", products["product_id"], key=product_key)
    units = products.set_index("product_id").at[product, "units_built"]
    req = requirements(bom, builds, q_from, q_to)
    mine = req[req["product_id"] == product]
    use = mine.assign(std_value=mine["need"] * mine["std_cost"]).groupby("part_number").agg(
        description=("description", "first"), assembly=("assembly_id", "first"), need=("need", "sum"), std_value=("std_value", "sum"))
    use["qty"] = use["need"] / units  # average gross quantity per unit built, with the BOM in effect each month
    use["std_per_unit"] = (use["std_value"] / units).fillna(0.0)
    use["actual_per_unit"] = alloc[alloc["product_id"] == product].groupby("part_number")["allocated_usd"].sum().reindex(use.index).fillna(0.0) / units
    use["diff"] = use["actual_per_unit"] - use["std_per_unit"]
    drivers = use.reindex(use["diff"].abs().sort_values(ascending=False).index).head(12).reset_index()
    drivers["direction"] = drivers["diff"].map(lambda d: "Above standard" if d > 0 else "Below standard")
    drivers["label"] = drivers["part_number"] + "  " + drivers["description"]
    st.altair_chart(
        alt.Chart(drivers).mark_bar(height=16, cornerRadiusEnd=3).encode(
            y=alt.Y("label:N", sort=None, title=None, axis=alt.Axis(labelLimit=380)),
            x=alt.X("diff:Q", axis=alt.Axis(format="$.2f", title="Estimated actual minus standard, per unit built")),
            color=alt.Color("direction:N", scale=alt.Scale(domain=["Above standard", "Below standard"], range=[VIOLET, BLUE]),
                            legend=alt.Legend(title=None, orient="top")),
            tooltip=[alt.Tooltip("part_number:N", title="Part"), alt.Tooltip("description:N", title="Description"),
                     alt.Tooltip("std_per_unit:Q", title="Standard per unit", format="$.3f"),
                     alt.Tooltip("actual_per_unit:Q", title="Est. actual per unit", format="$.3f"),
                     alt.Tooltip("diff:Q", title="Difference", format="+$.3f")],
        ).properties(height=330), width="stretch")
    st.dataframe(pd.DataFrame({
        "Part": drivers["part_number"], "Description": drivers["description"], "Assembly": drivers["assembly"],
        "Gross qty per unit": drivers["qty"].map("{:.3g}".format), "Standard per unit": drivers["std_per_unit"].map("${:.3f}".format),
        "Est. actual per unit": drivers["actual_per_unit"].map("${:.3f}".format), "Difference": drivers["diff"].map("{:+.3f}".format),
        "Note": drivers["actual_per_unit"].map(lambda v: "no matched production spend yet" if v == 0 else "")}), hide_index=True, width="stretch", height="content")

    st.subheader("The BOM in effect on a date")
    st.caption("Allocation, standard cost and the bought-versus-required lead use the BOM rows in effect in each build month.")
    st.session_state.setdefault(asof_key, day_max)
    if not day_min <= st.session_state[asof_key] <= day_max:
        st.session_state[asof_key] = day_max
    day = st.date_input("BOM as of", min_value=day_min, max_value=day_max, key=asof_key).isoformat()
    full_bom = bom[bom["product_id"] == product].copy()
    for col in ("effective_from", "effective_to", "parent_part"):
        full_bom[col] = full_bom[col].fillna("").astype(str) if col in full_bom else ""
    active = allocate.in_effect(full_bom, day)
    now = full_bom[active]
    change = (now["effective_from"].map(lambda d: f"new or changed on {d}" if d else "") + now["effective_to"].map(lambda d: f" (ends {d})" if d else "")).str.strip()
    table(pd.DataFrame({
        "Change": change, "Part": now["part_number"], "Description": now["description"], "Assembly": now["assembly_id"],
        "Parent": now["parent_part"], "Make or buy": now.get("make_buy", "buy"), "Qty per parent": now["qty_per_unit"].map("{:g}".format), "Unit": now["uom"],
        "Scrap": now.get("scrap_pct", pd.Series(0.0, index=now.index)).fillna(0.0).map("{:.0%}".format)}).sort_values("Change", ascending=False),
        key=f"bom_{product}_{day}")
    gone = full_bom[~active & (full_bom["effective_to"] != "") & (full_bom["effective_to"] < day)]
    later = full_bom[~active & (full_bom["effective_from"] > day)]
    st.caption(f"{len(now)} rows in effect on {day}; {int((change != '').sum())} of them carry an engineering change. "
               + ("No longer in effect: " + "; ".join(f"{x.part_number} x{x.qty_per_unit:g} (until {x.effective_to})" for x in gone.itertuples()) + ". " if len(gone) else "")
               + ("Not yet in effect: " + "; ".join(f"{x.part_number} x{x.qty_per_unit:g} (from {x.effective_from})" for x in later.itertuples()) + "." if len(later) else ""))

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
    st.caption("An estimate, not a stock count: cumulative production purchases minus the gross BOM requirement (scrap included, multi-level BOMs "
               "flattened) x units built, in the BOM's unit of measure, over the selected date range. "
               "It ignores opening stock, scrap and parts issued to service.")
    need_q = requirements(bom, builds, q_from, q_to).groupby(["part_number", "quarter"])["need"].sum()
    bought = lines[(lines["decision"] == "accept") & (lines["bucket"] == "production")]
    bought_q = bought.groupby(["part", "quarter"])["qty_bom"].sum().rename_axis(["part_number", "quarter"])
    inv = pd.DataFrame({"bought": bought_q.groupby("part_number").sum(), "needed": need_q.groupby("part_number").sum()}).fillna(0.0)
    price = bought.groupby("part")["spend_usd"].sum() / bought.groupby("part")["qty_bom"].sum()
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
            hide_index=True, width="stretch", height="content")
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

# ------------------------------------------------------------ savings leads
with tab_leads:
    names = {"supplier_price_gap": "Same part, different supplier prices", "duplicate_part": "Duplicate internal part numbers",
             "over_purchase": "Bought more than the builds required", "spares_premium": "Spares bought at a premium"}
    present = [k for k in names if (leads["lead_type"] == k).any()]
    chosen = st.pills("Lead type", present, selection_mode="multi", default=present, format_func=names.get, key=f"lead_types_{dataset}") or []
    shown = leads[leads["lead_type"].isin(chosen)].reset_index(drop=True)
    low, high = shown["low_usd"].sum(), shown["high_usd"].sum()
    m1, m2, m3 = st.columns(3)
    m1.markdown(kpi("Combined range", f"{short(low)} to {short(high)}", f"{usd(low)} to {usd(high)}: half of each gap closed, up to all of it"), unsafe_allow_html=True)
    m2.markdown(kpi("Leads", f"{len(shown)}", f"{len(shown)} leads across {len(chosen)} types"), unsafe_allow_html=True)
    m3.markdown(kpi("Share of spend", f"{low / total:.1%} to {high / total:.1%}", f"{usd(low)} to {usd(high)} of {usd(total)}"), unsafe_allow_html=True)
    st.markdown("**Leads, not booked savings.** Each is a price or quantity gap worth a conversation. Volume commitments, timing, quality and "
                "logistics can explain a gap.")
    if shown.empty:
        st.info("No leads for this selection.")
    else:
        by_type = shown.groupby("lead_type").agg(low=("low_usd", "sum"), high=("high_usd", "sum"), n=("part_number", "size")).reindex(chosen).dropna().reset_index()
        by_type["name"] = by_type["lead_type"].map(names)
        by_type["label"] = [f"{short(a)} to {short(b)}  ({int(n)} leads)" for a, b, n in zip(by_type["low"], by_type["high"], by_type["n"])]
        ranges = alt.Chart(by_type).mark_bar(height=16, color=BLUE, cornerRadius=4).encode(
            y=alt.Y("name:N", sort=None, title=None, axis=alt.Axis(labelLimit=300)),
            x=alt.X("low:Q", axis=alt.Axis(format="$~s", title="Range: half the gap closed, up to all of it"), scale=alt.Scale(domainMin=0)), x2="high:Q",
            tooltip=[alt.Tooltip("name:N", title="Lead type"), alt.Tooltip("low:Q", title="Low", format="$,.0f"),
                     alt.Tooltip("high:Q", title="High", format="$,.0f"), alt.Tooltip("n:Q", title="Leads")])
        range_labels = alt.Chart(by_type).mark_text(align="left", dx=6, color="#0b0b0b").encode(y=alt.Y("name:N", sort=None), x="high:Q", text="label:N")
        st.altair_chart((ranges + range_labels).properties(height=34 * len(by_type) + 50), width="stretch")

        pick = table(pd.DataFrame({
            "Type": shown["lead_type"].map(names), "Part": shown["part_number"], "Description": shown["description"],
            "Range": shown["low_usd"].map(usd) + " to " + shown["high_usd"].map(usd), "What the data shows": shown["detail"]}),
            key=f"leads_{'_'.join(chosen)}", select=True)
        st.download_button("Download leads (CSV)", shown.to_csv(index=False), "leads.csv", "text/csv")
        lead = shown.iloc[pick if pick is not None else 0]
        parts_in = [lead["part_number"]] + ([lead["related"]] if lead["lead_type"] == "duplicate_part" else [])
        buckets_in = ["production", "spares"] if lead["lead_type"] == "spares_premium" else ["production"]
        under = lines[(lines["decision"] == "accept") & lines["part"].isin(parts_in) & lines["bucket"].isin(buckets_in)].sort_values("po_date")
        st.markdown(f"**Purchase lines behind the lead on `{'` and `'.join(parts_in)}`** ({len(under)} lines, {usd(under['spend_usd'].sum())}). "
                    "Select a row above to change the lead.")
        table(pd.DataFrame({
            "PO line": under["po_id"], "Date": under["po_date"], "Plant": under["plant"], "Supplier": under["supplier"], "Part": under["part"],
            "Bucket": under["bucket"], "Quantity (BOM unit)": under["qty_bom"].map("{:,.0f}".format), "Price per unit": under["unit_price_usd"].map("${:.4f}".format),
            "Spend": under["spend_usd"].map(usd), "Text on the PO": under["item_description"],
            "Matched by": under["source"].map(HOW)}), key=f"under_{lead['lead_type']}_{lead['part_number']}")

# ----------------------------------------------------------------- evidence
with tab_evidence:
    if not is_sample:
        st.info("Uploaded files have no ground truth. Everything below was measured on the built-in sample and on external data.")
    s3, s4 = evals["eval_step3"], evals["eval_step4"]
    llm = s3["llm"][MAIN_MODEL]

    abt_card, xw_card = evals.get("eval_external_abt_buy"), evals.get("eval_crosswalk")
    held = llm["heldout"]
    card1, card2, card3 = st.columns(3)
    with card1.container(border=True):
        st.markdown("**Held-out accuracy**")
        st.markdown(kpi("Precision of accepted matches", f"{held['accept_precision']:.1%}", f"{held['accept_lines']:,} accepted lines; {usd(held['wrong_match_spend'])} matched wrongly"),
                    unsafe_allow_html=True)
        st.markdown(kpi("Tie-out", f"{held['tie_out']:.1%}", f"on {held['lines']:,} held-out lines, {usd(held['spend'])}", f"true rate {held['true_tie_out']:.1%}"),
                    unsafe_allow_html=True)
        st.caption("Lines written in styles the matcher was never tuned on. Synthetic data.")
    with card2.container(border=True):
        st.markdown("**Outside benchmark: Abt-Buy**")
        if abt_card:
            ours = abt_card["results"]["Rules + embeddings + Claude"]["f1"]
            published = ", ".join(f"{v['f1']:.2f}" for v in abt_card["published"].values())
            st.markdown(kpi("F1, nothing retrained", f"{ours:.2f}", f"{abt_card['test_pairs']} test pairs, {abt_card['test_matches']} true matches",
                            f"published baselines {published}"), unsafe_allow_html=True)
        st.caption("Real product listings from two shops, fixed test split from the University of Mannheim benchmark page.")
    with card3.container(border=True):
        st.markdown("**Review work over six quarters**")
        if xw_card:
            arm_with, arm_without = (pd.DataFrame(rows_x) for rows_x in xw_card["arms"].values())
            st.markdown(kpi("Lines left for a person", f"{int(arm_with['review_lines'].sum())} vs {int(arm_without['review_lines'].sum())}",
                            f"{usd(arm_with['review_spend'].sum())} vs {usd(arm_without['review_spend'].sum())} of spend left for review",
                            "with saved decisions vs without"), unsafe_allow_html=True)
        st.caption("A simulated reviewer clears each quarter's queue. Item text repeats by construction in this data.")
    st.caption("Details are in the sections below.")

    with st.expander("Held-out versus tuning split", expanded=False):
        st.caption("The matcher was tuned on the tuning split only. The held-out split is written in noise styles it never saw: plant-specific "
                   "codes, part-number typos, supplier catalog wording, description typos and Spanish.")
        rows = {"Purchase lines": lambda m: f"{m['lines']:,}", "Spend": lambda m: usd(m["spend"]),
                "Auto-accepted, share of lines": lambda m: pct(m["accept_share"]), "Precision of accepted matches": lambda m: f"{m['accept_precision']:.2%}",
                "Recall": lambda m: pct(m["recall"]), "Left for review": lambda m: f"{m['review_lines']} lines, {usd(m['review_spend'])}",
                "No-match calls that are right": lambda m: pct(m["no_match_accuracy"]), "Tie-out rate": lambda m: pct(m["tie_out"]),
                "True tie-out rate": lambda m: pct(m["true_tie_out"]), "Wrong-match dollars": lambda m: usd(m["wrong_match_spend"])}
        st.table(pd.DataFrame({"Held-out": {k: f(llm["heldout"]) for k, f in rows.items()}, "Tuning": {k: f(llm["tune"]) for k, f in rows.items()}}))

    with st.expander("Accuracy by noise style", expanded=False):
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
            st.dataframe(pd.DataFrame(table).fillna("not in this split").rename_axis("Style").reset_index(), hide_index=True, width="stretch", height="content")

    with st.expander("Claude on uncertain lines: cost and the small-model comparison", expanded=False):
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

    with st.expander("Savings leads against the planted findings", expanded=False):
        st.dataframe(pd.DataFrame([{"Lead type": names[k], "Found": v["found"], "Planted": v["planted"], "Precision": f"{v['precision']:.0%}",
                                    "Recall": f"{v['recall']:.0%}", "Range": f"{usd(v['low_usd'])} to {usd(v['high_usd'])}"}
                                   for k, v in s4["leads"].items()]), hide_index=True, width="stretch", height="content")
        st.caption(f"Measured at the default {usd(s4['materiality_usd'])} materiality threshold, before any reviewer action.")

    with st.expander("External tests: real data, nothing retrained", expanded=False):
        st.caption("Zero-shot: the classifier and every cutoff come from the synthetic tuning split. The downloaded data is not in the repository; "
                   "`python3 -m src.external.fetch` gets it.")
        abt, ext = evals.get("eval_external_abt_buy"), evals.get("eval_external_parts")
        if abt:
            st.markdown(f"**A. Abt-Buy product matching** (University of Mannheim benchmark, fixed test split: {abt['test_pairs']} pairs, "
                        f"{abt['test_matches']} true matches)")
            st.dataframe(pd.DataFrame([{"Method": k, "Precision": f"{v['precision']:.2f}", "Recall": f"{v['recall']:.2f}", "F1": f"{v['f1']:.2f}"}
                                       for k, v in {**abt["results"], **abt["published"]}.items()]), hide_index=True, width="stretch", height="content")
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
            st.dataframe(pd.DataFrame(table), hide_index=True, width="stretch", height="content")
            st.markdown("""
    - **The classifier does not transfer.** Trained on synthetic drill parts, it either sends real electronics lines to review or calls them "no match" outright, depending on the training run. Rules and Claude carry this test. A real deployment needs the client's own confirmed matches to train on.
    - **With the part number in the text, the rule that looks for it does most of the work**, and precision stays at 0.99.
    - **With the part number hidden, recall is low and precision holds (0.97).** A hobbyist BOM line such as "Red LED 5mm" does not carry enough to pick one catalog part, and the tool declines instead of guessing.
    - **Real no-match lines are rejected correctly more than 99% of the time** after Claude. The two or three accepted in error were accepted by the classifier, not by Claude.
    """)

    with st.expander("Sign-off by item (added after seeing held-out results)", expanded=False):
        im = s3.get("item_materiality")
        if im and im.get("before", {}).get("heldout"):
            st.caption(f"When the lines sharing one item text add up to more than {usd(im['threshold_usd'])}, the classifier may not close the item as "
                       "\"no match\" on its own: it goes to Claude, and to a person if Claude is not confident either way. "
                       "This rule was added after seeing held-out results, so the \"after\" column is not a clean held-out measurement.")
            rows_im = {"Precision of accepted matches": lambda m: f"{m['accept_precision']:.2%}", "Recall": lambda m: pct(m["recall"]),
                       "Review queue": lambda m: f"{m['review_lines']} lines, {usd(m['review_spend'])}",
                       "No-match calls that are right": lambda m: pct(m["no_match_accuracy"]),
                       "Spend with a BOM part dropped as no match": lambda m: usd(m["no_match_missed_spend"]), "Wrong-match dollars": lambda m: usd(m["wrong_match_spend"])}
            st.table(pd.DataFrame({"Held-out, before": {k: f(im["before"]["heldout"]) for k, f in rows_im.items()},
                                   "Held-out, after": {k: f(im["after"]["heldout"]) for k, f in rows_im.items()}}))
            st.caption(f"{im['held_from_classifier_no_match']} lines across both splits were kept open by this rule. "
                       f"Lines sent to Claude rose from {im['before']['band_lines']} to {im['after']['band_lines']}.")

    with st.expander("Experiment: do saved decisions cut review work over time?", expanded=False):
        xe = evals.get("eval_crosswalk")
        if xe:
            st.caption("Purchases are processed one quarter at a time. Each quarter the pipeline runs with the decisions saved so far, then a simulated "
                       "reviewer clears that quarter's review queue from the truth and the decisions are saved. The comparison is the same run with nothing saved.")
            plain = {"With the crosswalk": "With saved decisions", "No crosswalk": "Without saved decisions"}
            xr = pd.concat([pd.DataFrame(rows_x).assign(arm=plain.get(arm, arm)) for arm, rows_x in xe["arms"].items()])
            arms = [plain.get(a, a) for a in xe["arms"]]
            st.altair_chart(
                alt.Chart(xr).mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=70)).encode(
                    x=alt.X("quarter:N", title=None, axis=alt.Axis(labelAngle=0)), y=alt.Y("review_lines:Q", title="Lines left for review"),
                    color=alt.Color("arm:N", scale=alt.Scale(domain=arms, range=SERIES[:2]), legend=alt.Legend(title=None, orient="top")),
                    tooltip=[alt.Tooltip("arm:N", title="Run"), alt.Tooltip("quarter:N", title="Quarter"), alt.Tooltip("review_lines:Q", title="Review lines"),
                             alt.Tooltip("review_spend:Q", title="Review spend", format="$,.0f"), alt.Tooltip("claude_calls:Q", title="Claude calls"),
                             alt.Tooltip("settled_by_crosswalk:Q", title="Settled by a saved decision")]).properties(height=240), width="stretch")
            st.dataframe(pd.DataFrame({
                "Run": xr["arm"], "Quarter": xr["quarter"], "Lines": xr["lines"], "Settled by a saved decision": xr["settled_by_crosswalk"],
                "Claude calls": xr["claude_calls"], "Review lines": xr["review_lines"], "Review spend": xr["review_spend"].map(usd),
                "Precision": xr["precision"].map("{:.2%}".format), "Wrong-match $": xr["wrong_match_spend"].map(usd)}), hide_index=True, width="stretch", height="content")
            tot = xr.groupby("arm")[["review_lines", "review_spend", "claude_calls", "settled_by_crosswalk", "crosswalk_right"]].sum()
            w, n = tot.loc[arms[0]], tot.loc[arms[1]]
            st.markdown(f"""
    - **Review work:** {int(w['review_lines'])} lines ({usd(w['review_spend'])}) over six quarters with saved decisions, against {int(n['review_lines'])} lines ({usd(n['review_spend'])}) without.
    - **Claude calls:** {int(w['claude_calls'])} against {int(n['claude_calls'])}. The saving is smaller because most Claude calls are for lines Claude settles itself, which never reach a reviewer.
    - **Accuracy of saved decisions:** they settled {int(w['settled_by_crosswalk'])} later lines and {int(w['crosswalk_right'])} were right.
    - **Caveat:** item text repeats by construction in this data. The generator reuses one item text per plant, supplier and part about 80% of the time, so the size of the effect says little about a real company's exports. The classifier used here was also trained on lines from all six quarters.
    """)

    with st.expander("Experiment: training the model without confirmed labels", expanded=False):
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
            st.dataframe(pd.DataFrame(table), hide_index=True, width="stretch", height="content")
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
            st.dataframe(pd.DataFrame(table), hide_index=True, width="stretch", height="content")
            st.caption("The catalog has prices but no quantities, so wrong-match dollars cannot be computed for the real-parts test.")
        if sa or sb:
            st.markdown("""
    - **It does not replace confirmed labels.** On the synthetic data the self-trained classifier accepts more lines, but its precision falls from 100% to about 96.5% and it produces wrong-match dollars where the truth-trained one has none. Rule-matched lines always have a true part, so the model never sees a line that should be left unmatched.
    - **On real parts it barely moves.** Only a third of the training lines had their true part among the five nearest candidates, the thresholds collapse to "accept almost nothing", and recall goes from 0.00 to 0.02. Its one gain: it stops wrongly accepting no-match lines.
    - **Conclusion:** self-training is not a shortcut. The classifier needs reviewer-confirmed matches, which is what the crosswalk collects.
    """)

    with st.expander("Limits", expanded=True):
        st.markdown("""
    - **The data is synthetic.** Accuracy is measured on noise this project generated. Real exports are messier in ways the generator does not model, so treat these numbers as a ceiling, not a forecast.
    - **The classifier is specific to the data it was trained on.** On real electronics parts it abstained on almost everything (see the external tests above).
    - **One product family.** Four drill models and 280 bought parts. A real manufacturer has thousands of parts and many near-identical ones.
    - **Units convert only within a dimension.** Kilograms to grams or feet to metres are converted; wire bought by weight against a BOM in metres is not, and goes to review. On a purchase line the unit "M" is read as "per thousand" unless the matched part is measured in length.
    - **Purchased versus required is a gap, not a finding.** It can mean stock build, scrap above standard, service use, or a BOM quantity that is too low.
    - **Engineering changes count from the first of a month.** The BOM is taken as of the first day of each build month, so a change dated mid-month applies from the next month. Purchases are compared with the requirement, not corrected for it.
    - **Allocation is an estimate.** Spend is split by BOM quantity and build volume. It ignores inventory, scrap and timing between buying and building.
    - **Buckets lean on the PO type.** Where it is blank, the bucket is inferred from the match and the plant. A blank, unmatched spare is counted as unassigned.
    - **Duplicate detection is narrow.** It finds part numbers whose descriptions use the same words and numbers. Duplicates described differently are missed.
    - **Leads are not savings.** A price gap can have a good reason. Each one needs a buyer's judgment before it means anything.
    - **Your own files skip the trained model.** They are matched with rules, embeddings and Claude, and there is no ground truth to score the result against.
    - **Saved decisions live in a local file.** They are reused as the first rule on the next run; a real deployment would also retrain the model on them.
    - **Trends and inventory are rough.** They have no opening stock, scrap, or lead time between buying and building.
    - **What a pilot would test:** match accuracy on one plant's real exports, how much spend lands in review, and how long a reviewer takes to clear it.
    """)


# ---------------------------------------------------------------- your data
with tab_data:
    st.subheader("Run it on your own exports")
    st.caption("CSV or Excel. Column names do not need to match ours: Claude proposes a mapping and you confirm it. This needs API keys set up on this "
               "machine. It sends the headers and 20 sample rows of each file to Claude for the mapping, then purchase descriptions and BOM parts to "
               "the embedding model and Claude for matching.")
    up = st.session_state.setdefault("up_files", {})

    st.markdown("#### 1. Upload")
    if st.button("Try the second sample", type="primary", help="The same kind of data as the built-in sample, exported with different column names."):
        for kind, name in SAMPLE_FILES.items():
            up[kind] = (name, (SAMPLE_DIR / name).read_bytes())
        st.session_state.pop("up_map", None)
        st.rerun()
    st.caption(f"Or upload your own three files, up to {UPLOAD_LIMIT_MB} MB each.")
    for col, kind in zip(st.columns(3), KIND_LABEL):
        f = col.file_uploader(KIND_LABEL[kind], type=["csv", "xlsx", "xls"], key=f"file_{kind}")
        if f is not None:
            up[kind] = (f.name, f.getvalue())
        if kind in up:
            col.caption(f"Loaded: {up[kind][0]}")

    if len(up) == 3:
        raw = {kind: mapping.read_table(*up[kind]) for kind in KIND_LABEL}
        st.markdown("#### 2. Confirm columns")
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
            chosen_map, problems = {}, []
            for kind, prop in st.session_state["up_map"].items():
                chosen_map[kind] = {}
                with st.expander(f"{KIND_LABEL[kind]}: {up[kind][0]}, {len(raw[kind]):,} rows", expanded=True):
                    st.dataframe(raw[kind].head(5), hide_index=True, width="stretch", height="content")
                    grid = st.columns(4)
                    for i, row in enumerate(prop.itertuples()):
                        with grid[i % 4]:
                            pick_col = st.selectbox(f"{row.canonical} ({'required' if row.required else 'optional'})", ["(none)"] + list(raw[kind].columns),
                                                    key=f"map_{kind}_{row.canonical}", help=row.meaning)
                            st.caption(f"Claude: {row.confidence:.0%} sure. {row.note}" if row.source else f"Claude found no column. {row.note}")
                            chosen_map[kind][row.canonical] = None if pick_col == "(none)" else pick_col
                            if row.required and pick_col == "(none)":
                                problems.append(f"{KIND_LABEL[kind]}: {row.canonical} is required")
                    used = [v for v in chosen_map[kind].values() if v]
                    if len(used) != len(set(used)):
                        problems.append(f"{KIND_LABEL[kind]}: one source column is used twice")
            for problem in problems:
                st.error(problem)

            st.markdown("#### 3. Run")
            st.info("On your own data the matcher uses rules, nearest descriptions and Claude only. The trained model is skipped until this company "
                    "has confirmed matches to train it on. Saved decisions are applied first.")
            if st.button("Run the pipeline", type="primary", disabled=bool(problems)):
                try:
                    with st.spinner("Matching: rules, then nearest descriptions, then Claude on the open lines. A minute or two the first time."):
                        canon, notes = {}, []
                        for kind in KIND_LABEL:
                            canon[kind], kind_notes = mapping.apply_mapping(kind, raw[kind], chosen_map[kind])
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
        st.success(f"Last run: {', '.join(uploaded['files'])}: {uploaded['stats']['lines']:,} purchase lines. Select \"Your files\" under Data in the sidebar to view it.")
