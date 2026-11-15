"""BOM-to-Spend Tie-Out: the demo app.

    uv run streamlit run app.py

The built-in sample reads out/*.csv and out/eval_*.json only: no API keys, no model loading.
The "Your data" tab runs the matcher on a company's own files and does need API keys.
Reviewer decisions are saved (crosswalk.csv) and applied to every line with the same text.
"""
from __future__ import annotations

import hmac
import html
import json
from datetime import date, timedelta
from pathlib import Path

import sys

import re

import altair as alt
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


@st.cache_resource
def _build() -> dict:
    return {"stamp": None}


# A host can swap in new code and results without restarting the process. When that happens the already-imported
# src modules and the cached results are from the old version, which breaks the page (a lead without its owner,
# a purchase line without its source row). So: when any file in src/ or out/ has changed, drop both and start clean.
_stamp = max((f.stat().st_mtime for f in [*Path("src").glob("*.py"), *Path("out").glob("*")]), default=0)
if _build()["stamp"] != _stamp:
    for _name in [n for n in sys.modules if n == "src" or n.startswith("src.")]:
        del sys.modules[_name]
    st.cache_data.clear()
    _build()["stamp"] = _stamp

from src import adjudicate, allocate, crosswalk, mapping, match, pipeline, settings

try:  # animated charts; if the component cannot load (for example on a host that blocks it), static charts are used instead
    from streamlit_echarts import JsCode, st_echarts
    HAVE_ECHARTS = True
except Exception:
    HAVE_ECHARTS = False
from src.normalize import norm_desc, norm_pn, numbers_in, prepare

OUT = Path("out")
SAMPLE_DIR = Path("samples/second_company")
SAMPLE_FILES = {"boms": "bom_export.csv", "purchases": "po_lines.xlsx", "builds": "production_output.csv"}
KIND_LABEL = {"boms": "Bill of materials", "purchases": "Purchase order lines", "builds": "Production volumes"}
UPLOAD_LIMIT_MB = 5  # also set as server.maxUploadSize in .streamlit/config.toml
SAMPLE_CACHE = Path("cache/llm_sample")  # committed Claude answers for the second sample: it runs with no Claude calls
HOSTED = settings.hosted()               # a public deployment: access code, limits, and nothing written to disk
HOSTED_MAX_LINES, HOSTED_MAX_CALLS, HOSTED_MAX_RUNS = 2_000, 150, 3
HOSTED_DAILY_RUNS = int(settings.secret("DAILY_RUN_LIMIT", "40"))  # across all visitors: bounds API spend when there is no access code
adjudicate.PERSIST = match.PERSIST = not HOSTED
BUCKETS = ["production", "spares", "warranty", "unassigned"]
# One set of tokens for the whole app, taken from the Opportunities page.
BG, SURFACE_C, INK, MUTED, LINE = "#EDF0F4", "#FFFFFF", "#0F1B2D", "#5B6778", "#D9DEE6"
MONEY, MONEY_ON_DARK = "#087F43", "#63E2A4"
BLUE, VIOLET, YELLOW, ORANGE, GRAY = "#2457D6", "#7A3FD1", "#9A5200", "#E2703A", "#9AA5B4"  # buyer, engineer, service, wrong, neutral
AQUA = MONEY
OPPORTUNITIES_HEIGHT = 420  # the floor; the stylesheet stretches the page to fill the window below the tabs
BUCKET_COLORS = [BLUE, AQUA, YELLOW, GRAY]  # unassigned is neutral: it is the absence of a tie
SERIES = [BLUE, AQUA, YELLOW, VIOLET]       # orange is kept for warnings and wrong matches only
SURFACE = SURFACE_C
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

st.set_page_config(page_title="BOM-to-Spend Tie-Out", layout="wide", initial_sidebar_state="collapsed")
FONT = "Inter, sans-serif"
LIGHT_BLUE = "#B9CBF5"
st.markdown("""<style>
.block-container {padding-top: .6rem; padding-bottom: 1rem;}
header[data-testid="stHeader"] {background: transparent; height: 2rem;}
h1, h2, h3, h4 {font-family: "Bricolage Grotesque", Inter, sans-serif !important; letter-spacing: -0.01em;}
h3 {margin-top: 1.6rem !important; padding-bottom: .4rem !important;}
.topbar {display: flex; align-items: center; gap: .7rem; margin: 0 0 -.4rem 0;}
.app-title {font-family: "Bricolage Grotesque", Inter, sans-serif; font-size: 1.1rem; font-weight: 700;}
.badge {font-size: .76rem; font-weight: 600; padding: .12rem .6rem; border-radius: 999px; background: #E4ECFF; color: #2457D6;}
.badge.warn {background: #FFEFD6; color: #9A5200;}
/* the dark band that opens every tab */
.hero-band {background: #0F1B2D; color: #fff; border-radius: 22px; padding: 1.1rem 1.6rem 1.2rem; margin-bottom: .4rem;}
.hero-line {font-size: .95rem; color: #B7C2D3;}
.hero-big {font-family: "Bricolage Grotesque", Inter, sans-serif; font-weight: 700; font-size: clamp(2.6rem, 5vw, 4.4rem); line-height: 1.05;
           letter-spacing: -0.02em; color: #63E2A4; margin: .15rem 0 .55rem; font-variant-numeric: tabular-nums;}
.hero-big.text {font-size: clamp(1.9rem, 3.4vw, 2.9rem);}
.hero-pills span {display: inline-block; font-size: .84rem; padding: .28rem .75rem; border-radius: 999px; background: #1E2D45; color: #DCE4F0; margin-right: .45rem;}
.kpi-label {font-size: .78rem; color: #5B6778; line-height: 1.2;}
.kpi-value {font-family: "Bricolage Grotesque", Inter, sans-serif; font-size: 1.5rem; font-weight: 700; line-height: 1.25; color: #087F43; font-variant-numeric: tabular-nums;}
.kpi-note {font-size: .76rem; color: #5B6778; line-height: 1.2;}
/* one card surface: white, 1.5px border, 18px radius, on the grey page */
.card, .pair, [data-testid="stVegaLiteChart"], [data-testid="stDataFrame"], [data-testid="stExpander"] details, [data-testid="stTable"],
[class*="st-key-card_"] {background: #FFFFFF; border: 1.5px solid #D9DEE6 !important; border-radius: 18px !important;}
.card {padding: .45rem .8rem; min-height: 4.6rem;}
[data-testid="stVegaLiteChart"] {padding: .5rem 0 .2rem 0; overflow: hidden;}
[class*="st-key-card_"] .card {border: 0 !important; padding: 0; min-height: 0; margin-bottom: .5rem;}
[data-testid="stDataFrame"] {padding: .35rem; overflow: hidden;}
[data-testid="stTable"] {padding: .2rem .4rem; overflow: hidden;}
[class*="st-key-card_"] {padding: .9rem 1rem;}
.pair {padding: .55rem .8rem; margin-bottom: .5rem; font-size: .95rem;}
.pair .who {font-size: .74rem; color: #5B6778;}
.tag {display: inline-block; font-size: .74rem; font-weight: 600; padding: .1rem .55rem; border-radius: 999px; background: #E4ECFF; color: #2457D6; margin-right: .3rem;}
kbd {font-size: .72rem; padding: .05rem .35rem; border: 1.5px solid #D9DEE6; border-radius: 4px; background: #fff;}
button kbd {color: #0F1B2D !important; background: #EDF0F4 !important; border-color: #D9DEE6 !important;}
button[kind="primary"] kbd {color: #FFFFFF !important; background: rgba(255,255,255,.2) !important; border-color: rgba(255,255,255,.55) !important;}
/* buttons are pills; the primary one is green (the theme's primary colour) */
.stButton button, .stDownloadButton button {border-radius: 999px !important; border-width: 1.5px !important; font-weight: 600;}
/* the Opportunities page fills the window below the tabs, so the browser page itself does not scroll */
.st-key-period_pop {position: absolute !important; top: -46px; right: 0; width: auto !important; z-index: 5;}
.st-key-period_pop button {min-height: 36px !important; padding: 2px 14px !important; background: #FFFFFF !important; border: 1.5px solid #D9DEE6 !important; white-space: nowrap;}
.st-key-period_pop button p {font-size: 14px;}
[data-testid="stVerticalBlock"]:has(> .st-key-period_pop) {position: relative;}
[data-testid="stIFrame"], [data-testid="stElementContainer"]:has(> [data-testid="stIFrame"]) {height: calc(100vh - 132px) !important; min-height: 420px; border-radius: 22px;}
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"], [data-testid="stExpandSidebarButton"] {display: none !important;}
[data-testid="stPopover"] button {border-radius: 999px !important; border-width: 1.5px !important; padding: .1rem .9rem; min-height: 2rem;}
/* Phones. Nothing below applies above 640 px, so the desktop layout is untouched. */
@media (max-width: 640px) {
  /* All five tabs fit a 390 px screen without the scroll arrow. */
  [data-testid="stTabs"] [role="tablist"] {gap: 5px !important;}
  [data-testid="stTab"] p {font-size: 13px !important;}
  /* Charts whose labels need room keep a readable width and scroll sideways inside their card. */
  .st-key-card_flow, .st-key-wide_stages, .st-key-wide_drivers {overflow-x: auto !important;}
  .st-key-card_flow > div {min-width: 600px;}
  .st-key-wide_stages [data-testid="stVegaLiteChart"], .st-key-wide_drivers [data-testid="stVegaLiteChart"] {min-width: 600px;}
}
</style>""", unsafe_allow_html=True)


def note(text: str, label: str = "How to read this"):
    """A caption of at most 12 words. Anything longer becomes a short label with the full text in its tooltip."""
    if len(text.split()) <= 12:
        st.caption(text)
    else:
        st.caption(label, help=text)


def show_chart(chart, **kwargs):
    """Charts sit on a white card, so they draw on white, not on the page's grey."""
    padded = chart.properties(padding={"left": 16, "right": 22, "top": 8, "bottom": 8})
    return st.altair_chart(padded.configure(background=SURFACE_C).configure_view(strokeWidth=0), **kwargs)


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
    return (f'<div class="card" title="{html.escape(hover)}"><div class="kpi-label">{html.escape(label)}</div>'
            f'<div class="kpi-value">{html.escape(value)}</div><div class="kpi-note">{html.escape(note)}</div></div>')


def quarter_of(dates: pd.Series) -> pd.Series:
    d = pd.to_datetime(dates)
    return d.dt.year.astype(str) + "Q" + d.dt.quarter.astype(str)


def table(df: pd.DataFrame, key: str, select: bool = False, page_size: int = PAGE, column_config: dict | None = None):
    """A table sized to its rows, paged when long. With select=True returns the picked row's position in df, or None."""
    pages = max(1, -(-len(df) // page_size))
    page = min(st.session_state.get(f"{key}_page", 0), pages - 1)
    view = df.iloc[page * page_size:(page + 1) * page_size]
    picked = None
    if select:
        event = st.dataframe(view, hide_index=True, width="stretch", height="content", on_select="rerun", selection_mode="single-row", key=f"{key}_p{page}",
                             column_config=column_config)
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
    notes_path = OUT / "similar_part_notes.json"
    evals["similar_part_notes"] = json.loads(notes_path.read_text()) if notes_path.exists() else {}
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
def waiting_by_product(lines: pd.DataFrame, bom: pd.DataFrame, builds: pd.DataFrame, q_from: str, q_to: str) -> pd.Series:
    """Production spend each product would gain if every suggestion in the review queue were approved. Until then that
    spend is in no product's cost, so cost against standard reads low."""
    in_range = builds[quarter_of(builds["period"]).between(q_from, q_to)]
    now = allocate.allocate(lines, bom, in_range)[1].set_index("product_id")["allocated_production_spend_est"]
    sim = lines.copy()
    ok = (sim["decision"] == "review") & sim["suggested_part"].isin(set(bom["part_number"]))
    sim.loc[ok, "part"], sim.loc[ok, "decision"] = sim.loc[ok, "suggested_part"], "accept"
    then = allocate.allocate(allocate.assign_buckets(sim), bom, in_range)[1].set_index("product_id")["allocated_production_spend_est"]
    return (then - now).clip(lower=0.0)


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


@st.cache_resource
def daily_runs() -> dict:
    """One counter for the whole server, shared by every visitor. Holds a date and a count, nothing else."""
    return {"day": "", "runs": 0}


def saved_load() -> pd.DataFrame:
    """Saved reviewer decisions. On a public host they live in this browser session only, never in a shared file."""
    if HOSTED:
        return st.session_state.setdefault("saved_decisions", pd.DataFrame(columns=crosswalk.COLUMNS))
    return crosswalk.load()


def saved_add(supplier_part_number: str, item_description: str, part: str, decision: str, dataset: str):
    if HOSTED:
        st.session_state["saved_decisions"] = crosswalk.append(saved_load(), supplier_part_number, item_description, part, decision, dataset)
    else:
        crosswalk.add(supplier_part_number, item_description, part, decision, dataset)


def saved_undo():
    if HOSTED:
        st.session_state["saved_decisions"] = saved_load().iloc[:-1]
    else:
        crosswalk.undo_last()


def saved_clear():
    if HOSTED:
        st.session_state["saved_decisions"] = pd.DataFrame(columns=crosswalk.COLUMNS)
    else:
        crosswalk.clear()


sample_base, sample_bom, sample_builds, evals = load()
similar_notes = evals.get("similar_part_notes", {})
if st.session_state.pop("_switch_to_uploaded", False):
    st.session_state["dataset"] = "Your files"
uploaded = st.session_state.get("uploaded")

# ----------------------------------------------------------------- settings
# There is no sidebar. Each control sits in the tab it belongs to (the data switch in Your data, the sign-off threshold in
# Review, the period on Opportunities) and its value is read here from session state, before anything is computed.
options = ["Built-in sample"] + (["Your files"] if uploaded else [])
if st.session_state.get("dataset") not in options:
    st.session_state["dataset"] = options[0]
dataset = st.session_state["dataset"]
is_sample = dataset == "Built-in sample"
base, bom, builds = (sample_base, sample_bom, sample_builds) if is_sample else (uploaded["lines"], uploaded["bom"], uploaded["builds"])

SIGNOFF_STEPS = list(range(0, 200_001, 5_000))
if st.session_state.get("signoff") not in SIGNOFF_STEPS:
    st.session_state["signoff"] = int(allocate.MATERIALITY_USD)
materiality = st.session_state["signoff"]

quarters = sorted(base["quarter"].unique())
period_key = f"quarters_{dataset}"      # the slider's own key; Streamlit drops it whenever the slider is not drawn
period_kept = f"period_{dataset}"       # our copy, which survives
def _valid_period(v):
    return isinstance(v, (tuple, list)) and len(v) == 2 and all(q in quarters for q in v)
if _valid_period(st.session_state.get(period_key)):
    st.session_state[period_kept] = tuple(st.session_state[period_key])
elif not _valid_period(st.session_state.get(period_kept)):
    st.session_state[period_kept] = (quarters[0], quarters[-1])
q_from, q_to = st.session_state[period_kept]

xw = saved_load()
rules = tuple(sorted((k, d, part) for k, (d, part) in crosswalk.as_rules(xw).items()))
lines, alloc, products, leads = compute(base, bom, builds, float(materiality), rules, q_from, q_to)
# Results held in a browser session from before an update may lack newer columns. Fill them, so an old session degrades instead of failing.
for _col, _default in (("source_file", "purchase file"), ("source_row", "?")):
    if _col not in lines:
        lines[_col] = _default
if "owner" not in leads:
    leads["owner"] = leads["lead_type"].map(allocate.OWNER)
if "check" not in leads:
    leads["check"] = leads["lead_type"].map(allocate.CHECK)
start_lines = compute(base, bom, builds, float(allocate.MATERIALITY_USD), (), q_from, q_to)[0]
# Cost against standard leaves out spend that is waiting in review. Where that is more than a tenth of a product's standard
# cost the percentage would mislead, so the product is shown as "in review" instead.
REVIEW_SHARE_LIMIT = 0.10
products["waiting_usd"] = products["product_id"].map(waiting_by_product(lines, bom, builds, q_from, q_to)).fillna(0.0)
products["cost_in_review"] = products["waiting_usd"] > REVIEW_SHARE_LIMIT * products["std_cost_per_unit"] * products["units_built"]
review_spend = float(lines.loc[lines["decision"] == "review", "spend_usd"].sum())
review_note = f"Excludes {short(review_spend)} waiting in review." if review_spend > 0 else ""
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
synthetic = is_sample  # the built-in sample only; anything run through Your data is labelled "Your files"
st.markdown(
    '<div class="topbar"><span class="app-title">BOM-to-Spend Tie-Out</span>'
    + (f'<span class="badge warn" title="{html.escape(SYNTHETIC_NOTICE)}">Synthetic data</span>' if synthetic
       else f'<span class="badge" title="{html.escape(", ".join(uploaded["files"]))}">Your files</span>') + "</div>", unsafe_allow_html=True)

LEAD_NAMES = {"supplier_price_gap": "Same part, different supplier prices", "spares_premium": "Spares bought at a premium",
              "duplicate_part": "Duplicate internal part numbers", "over_purchase": "Bought more than the builds required",
              "similar_parts": "Similar parts to consolidate"}
names = LEAD_NAMES
LEAD_KIND = {"supplier_price_gap": "Two supplier prices", "spares_premium": "Spares premium", "duplicate_part": "Duplicate part",
             "over_purchase": "Over-bought", "similar_parts": "Similar parts"}
LEAD_CHECKS = {
    "supplier_price_gap": ["Volume commitments and contract terms", "Order quantity, freight and currency", "Capacity and quality at the cheaper supplier"],
    "spares_premium": ["Order quantity and freight on small service orders", "Same revision as the production part", "Service-level terms with the supplier"],
    "duplicate_part": ["Engineering confirms the two parts are interchangeable", "Stock on hand of each number", "Which number suppliers quote against"],
    "over_purchase": ["Scrap above the BOM allowance", "Stock built up for later builds", "Parts issued to service"],
    "similar_parts": ["Engineering must confirm", "Fit, form and function of both parts", "Volume commitments, order quantity and timing"],
}


def plain_noun(description: str) -> list[str]:
    """The part's name as words: lower case, except acronyms (PCBA, MOSFET, LED), which stay in capitals."""
    return [w if (w.isupper() and sum(c.isalpha() for c in w) >= 2) else w.lower() for w in str(description).split(",")[0].split()]


def lead_action(r) -> str:
    """What to check before acting on the lead, in 6 to 10 words. Leading qualifiers are dropped from long part names."""
    if r["lead_type"] == "duplicate_part":
        return f"Confirm {r['part_number']} and {r['related']} are interchangeable, then merge"
    if r["lead_type"] == "similar_parts":
        return f"Ask engineering if {r['part_number']} can replace {r['related']}"
    make = {"supplier_price_gap": "Check terms, then move the {} to the cheaper supplier", "spares_premium": "Check if spares of the {} qualify for production pricing",
            "over_purchase": "Find out why the {} is over-bought"}[r["lead_type"]]
    words = plain_noun(r["description"])
    if len(words) > 1 and words[-1] in ("left", "right", "upper", "lower"):  # "housing shell left" reads as "left housing shell"
        words = [words[-1]] + words[:-1]
    if len(make.format(" ".join(words)).split()) > 10:  # the article goes first, then leading qualifiers
        make = make.replace("the {}", "{}")
    while len(make.format(" ".join(words)).split()) > 10 and len(words) > 1:
        words = words[1:]
    return make.format(" ".join(words))


# Leads, each split across products by where the part's allocated spend went. "Similar parts" are ideas for engineering
# and are kept out of every total.
L = leads.copy()
L["action"] = L.apply(lead_action, axis=1) if len(L) else ""
part_weights = alloc.groupby(["part_number", "product_id"])["allocated_usd"].sum()
shares = []
for r in L.itertuples():
    parts_ = [r.part_number] + ([r.related] if r.lead_type in ("duplicate_part", "similar_parts") else [])
    w = part_weights[part_weights.index.get_level_values(0).isin(parts_)].groupby("product_id").sum()
    shares.append((w / w.sum()).to_dict() if w.sum() > 0 else {})
L["shares"] = shares
firm = L[L["lead_type"] != "similar_parts"]
by_product = {}
for r in firm.itertuples():
    for product_id, share in r.shares.items():
        entry = by_product.setdefault(product_id, [0.0, 0.0, 0])
        entry[0] += r.low_usd * share
        entry[1] += r.high_usd * share
        entry[2] += 1


def hero(big: str, line: str, pills: list[str], text: bool = False):
    """The dark band that opens every tab: one big figure, one line, up to three pills."""
    st.markdown(f'<div class="hero-band"><div class="hero-line">{html.escape(line)}</div>'
                f'<div class="hero-big{" text" if text else ""}">{html.escape(big)}</div>'
                f'<div class="hero-pills">{"".join(f"<span>{html.escape(p)}</span>" for p in pills[:3])}</div></div>', unsafe_allow_html=True)


def opportunities_data() -> dict:
    """Everything the Opportunities page reads, built from the same lines, leads and shares as the rest of the app."""
    ok = lines[(lines["decision"] == "accept") & lines["qty_bom"].notna()]
    display = lines.groupby("supplier_key")["supplier"].agg(lambda s: max(s, key=len))  # the fullest spelling of each supplier's name
    price_of = lambda frame: frame["spend_usd"].sum() / frame["qty_bom"].sum() if frame["qty_bom"].sum() else None
    row = lambda label, value, text=None: {"label": label, "value": round(float(value), 4), "text": text or f"${value:,.2f}"}
    out = []
    for i, r in enumerate(L.itertuples()):
        pair = [r.part_number] + ([r.related] if r.lead_type in ("duplicate_part", "similar_parts") else [])
        buckets = ["production", "spares"] if r.lead_type == "spares_premium" else ["production"]
        under = ok[ok["part"].isin(pair) & ok["bucket"].isin(buckets)]
        prod = under[under["bucket"] == "production"]
        compare = []
        if r.lead_type == "supplier_price_gap":
            by_sup = prod.groupby("supplier_key").apply(price_of, include_groups=False).dropna().sort_values()
            compare = [row(display[k], v) for k, v in by_sup.items()]
        elif r.lead_type == "spares_premium":
            spares = under[under["bucket"] == "spares"]
            compare = [row(label, price_of(f)) for label, f in (("Production", prod), ("Spares", spares)) if price_of(f)]
        elif r.lead_type in ("duplicate_part", "similar_parts"):
            compare = [row(part, price_of(prod[prod["part"] == part])) for part in pair if price_of(prod[prod["part"] == part])]
        elif r.lead_type == "over_purchase":
            found = re.search(r"bought ([\d,]+) (\S+) against ([\d,]+)", r.detail)
            if found:
                compare = [row(label, float(found.group(n).replace(",", "")), f"{found.group(n)} {found.group(2)}") for label, n in (("Bought", 1), ("Needed", 3))]
        checks = list(LEAD_CHECKS[r.lead_type])
        note = similar_notes.get("|".join(sorted(map(str, pair)))) if r.lead_type == "similar_parts" else None
        if note:
            checks.append(f"What differs (A is {note['a']}, B is {note['b']}): {note['note']}")
        top = (under[under["bucket"] == "spares"] if r.lead_type == "spares_premium" else under).sort_values("spend_usd", ascending=False).head(6)
        out.append({
            "id": f"L{i}", "type": r.lead_type, "kind": LEAD_KIND[r.lead_type], "owner": r.owner, "part": r.part_number, "action": r.action,
            "low": float(r.low_usd), "high": float(r.high_usd), "in_total": r.lead_type != "similar_parts", "compare": compare, "checks": checks,
            "products": [{"id": k, "share": round(float(v), 4)} for k, v in sorted(r.shares.items(), key=lambda kv: -kv[1])],
            "line_count": int(len(under)),
            "lines": [{"date": x.po_date, "supplier": x.supplier, "price": round(float(x.unit_price_usd), 4), "spend": round(float(x.spend_usd), 2),
                       "source": f"{x.source_file}, row {x.source_row}"} for x in top.itertuples()]})
    return {"spend_total": float(total), "tied_share": float(tied), "waiting_lines": int(len(queue)),
            "synthetic": bool(is_sample),
            "review_note": review_note,
            "products": [{"id": r.product_id, "vs_std": None if pd.isna(r.actual_vs_std) else float(r.actual_vs_std), "in_review": bool(r.cost_in_review)}
                         for r in products.itertuples()], "leads": out}


tab_labels = ["Opportunities", f"Review ({len(queue)})", "Products", "Evidence", "Your data"]
# The selected tab is remembered by its label, and the Review label carries a count. When the count changes (a decision, the
# sign-off slider), carry the selection over to the new label, or the reader is thrown back to the first tab.
if str(st.session_state.get("main_tabs", "")).startswith("Review (") and st.session_state["main_tabs"] not in tab_labels:
    st.session_state["main_tabs"] = tab_labels[1]
tab_opps, tab_review, tab_cost, tab_evidence, tab_data = st.tabs(tab_labels, key="main_tabs", on_change="rerun")

# ------------------------------------------------------------ opportunities
with tab_opps:
    if len(quarters) > 1:
        # Drawn out of the flow; opportunities.html moves it into the slot it keeps free in its filter row.
        with st.popover(f"Period: {q_from} to {q_to}", key="period_pop"):
            st.select_slider("Date range", options=quarters, value=(q_from, q_to), key=period_key)
            st.caption("Narrows every tab to these quarters.")
    page = Path("opportunities.html").read_text().replace("/*DATA*/null/*END*/", json.dumps(opportunities_data()).replace("</", "<\\/"))
    # scrolling=True lets a phone scroll the page inside its frame; on a desktop the page fits its frame and never scrolls.
    components.html(page, height=OPPORTUNITIES_HEIGHT, scrolling=True)

# ----------------------------------------------------------------- products
with tab_cost:
    if by_product:
        best_id, (best_low, best_high, best_n) = max(by_product.items(), key=lambda kv: kv[1][1])
        best = products.set_index("product_id").loc[best_id]
        hero(best_id, "The product with the largest lead range", [f"{short(best_low)} to {short(best_high)} in leads", f"{best_n} leads touch it"]
             + (["Cost vs standard: in review"] if best["cost_in_review"] else [f"Estimated actual {best['actual_vs_std']:+.1%} vs standard"]
                if pd.notna(best["actual_vs_std"]) else []))
    else:
        hero("No leads yet", "Products", [])
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
    show_chart((cost_bars + cost_labels).properties(height=alt.Step(20)), width="stretch")
    if review_note:
        st.caption(review_note)
    note("Estimated: matched production spend is split across the products that use each part, by gross BOM quantity x units built in the "
               "purchase's quarter. Purchases show what was bought, not what was consumed. Lines waiting for a decision are not included yet.")
    st.dataframe(pd.DataFrame({
        "Product": products["product_id"], "Units built": products["units_built"].map("{:,}".format),
        "Standard per unit": products["std_cost_per_unit"].map(lambda v: "n/a" if pd.isna(v) else f"${v:.2f}"),
        "Estimated actual per unit": products["actual_cost_per_unit_est"].map("${:.2f}".format),
        "Actual vs standard": ["in review" if r.cost_in_review else "n/a" if pd.isna(r.actual_vs_std) else f"{r.actual_vs_std:+.1%}" for r in products.itertuples()],
        "Allocated production spend": products["allocated_production_spend_est"].map(usd)}), hide_index=True, width="stretch", height="content")
    # The file says what the screen says: "in review" where the percentage is withheld.
    cost_file = products.drop(columns="cost_in_review").rename(columns={"waiting_usd": "waiting_in_review_usd"})
    cost_file["actual_vs_std"] = cost_file["actual_vs_std"].astype(object).mask(products["cost_in_review"], "in review")
    st.download_button("Download product cost (CSV)", cost_file.to_csv(index=False), "product_cost.csv", "text/csv")

    st.subheader("Shared parts: who uses them and who sells them")
    req = requirements(bom, builds, q_from, q_to)
    matched = lines[(lines["decision"] == "accept") & lines["part"].isin(set(req["part_number"]))]
    part_spend = matched.groupby("part")["spend_usd"].sum().sort_values(ascending=False)
    n_parts = st.select_slider("Parts shown, largest spend first", options=[20, 40, 60, 80], value=40, key=f"net_n_{dataset}")
    top_parts = list(part_spend.head(n_parts).index)
    buys = matched[matched["part"].isin(top_parts) & matched["qty_bom"].notna()].groupby(["part", "supplier_key"]).agg(
        spend=("spend_usd", "sum"), qty=("qty_bom", "sum"), name=("supplier", lambda x: x.value_counts().index[0])).reset_index()
    sup_name = buys.groupby("supplier_key")["name"].first()
    uses = req[req["part_number"].isin(top_parts)].groupby(["product_id", "part_number"], as_index=False)["need"].sum()
    net_key = f"net_part_{dataset}"
    if HAVE_ECHARTS and top_parts:
        size = lambda v, top, lo, hi: float(round(lo + (hi - lo) * (v / top) ** 0.5, 1))
        sup_spend = buys.groupby("supplier_key")["spend"].sum()
        corners = [(430, 170), (930, 170), (430, 400), (930, 400), (680, 285), (180, 285), (1180, 285), (680, 60)]  # products are pinned apart
        nodes = ([{"name": pid, "category": 0, "symbolSize": 44, "label": {"show": True, "fontWeight": 600}, "x": corners[i % 8][0], "y": corners[i % 8][1], "fixed": True}
                  for i, pid in enumerate(sorted(uses["product_id"].unique()))]
                 + [{"name": part, "category": 1, "symbolSize": size(part_spend[part], part_spend.iloc[0], 9, 38), "value": int(round(part_spend[part])),
                     "label": {"show": i < 8}} for i, part in enumerate(top_parts)]
                 + [{"name": sup_name[k], "category": 2, "symbolSize": size(v, sup_spend.max(), 9, 30), "value": int(round(v)), "label": {"show": bool(v > 0.03 * sup_spend.sum())}}
                    for k, v in sup_spend.items()])
        links = ([{"source": u.product_id, "target": u.part_number, "lineStyle": {"type": "solid"}} for u in uses.itertuples()]
                 + [{"source": b_.part, "target": sup_name[b_.supplier_key], "lineStyle": {"type": "dashed"}} for b_ in buys.itertuples()])
        with st.container(key="card_net"):
            clicked = st_echarts({"baseOption": {
                "textStyle": {"fontFamily": FONT}, "color": [BLUE, AQUA, VIOLET],
                "legend": {"top": 4, "data": ["Product", "Part", "Supplier"], "textStyle": {"fontFamily": FONT}},
                "tooltip": {"valueFormatter": JsCode("function (v) { return v ? '$' + Math.round(v).toLocaleString('en-US') : ''; }")},
                "series": [{
                    "type": "graph", "layout": "force", "roam": True, "draggable": True, "top": 40,
                    "categories": [{"name": "Product"}, {"name": "Part"}, {"name": "Supplier"}],
                    "force": {"repulsion": 190, "edgeLength": [30, 110], "gravity": 0.09, "layoutAnimation": True},
                    "label": {"position": "right", "fontFamily": FONT, "fontSize": 11, "color": INK},
                    "lineStyle": {"color": "#9a9890", "opacity": 0.45, "width": 1},
                    "emphasis": {"focus": "adjacency", "label": {"show": True}, "lineStyle": {"width": 2}},
                    "data": nodes, "links": links,
                }]},
                # The products are pinned at positions laid out for a wide chart. On a phone, zoom out so the whole network shows.
                # The rule without a query restores the normal view whenever the chart is wider than that.
                "media": [{"query": {"maxWidth": 700}, "option": {"series": [{"zoom": 0.3, "label": {"fontSize": 9}}]}},
                          {"option": {"series": [{"zoom": 1, "label": {"fontSize": 11}}]}}]},
                height="560px", renderer="svg", key=f"net_{dataset}_{n_parts}",
                events={"click": "function (params) { return params.dataType === 'node' ? params.name : null; }"})
        clicked = clicked.get("chart_event") if isinstance(clicked, dict) else clicked
        if clicked in top_parts and clicked != st.session_state.get("net_last_click"):
            st.session_state["net_last_click"], st.session_state[net_key] = clicked, clicked
        note("Solid lines: a product uses the part. Dashed lines: the part is bought from the supplier. Node size is spend. "
                   "Click a part, or pick one below. Drag to rearrange, scroll to zoom.")
    elif top_parts:
        show_chart(
            alt.Chart(part_spend.head(n_parts).rename_axis("part").reset_index(name="spend")).mark_bar(color=BLUE, cornerRadiusEnd=3).encode(
                y=alt.Y("part:N", sort=None, title=None), x=alt.X("spend:Q", axis=alt.Axis(format="$~s", title="Matched spend")),
                tooltip=[alt.Tooltip("part:N", title="Part"), alt.Tooltip("spend:Q", title="Spend", format="$,.0f")]).properties(height=alt.Step(16)),
            width="stretch")
    if top_parts:
        if st.session_state.get(net_key) not in top_parts:
            st.session_state[net_key] = top_parts[0]
        chosen_part = st.selectbox("Part", top_parts, key=net_key)
        note(part_desc.get(chosen_part, ""))
        n1, n2 = st.columns(2)
        with n1:
            st.markdown(f"**Products using `{chosen_part}`**")
            used = req[req["part_number"] == chosen_part].groupby("product_id").agg(need=("need", "sum"), units=("units_built", "sum")).reset_index()
            st.dataframe(pd.DataFrame({"Product": used["product_id"], "Gross qty per unit": (used["need"] / used["units"]).map("{:.3g}".format),
                                       "Required over the period": used["need"].map("{:,.0f}".format)}), hide_index=True, width="stretch", height="content")
        with n2:
            st.markdown(f"**Suppliers of `{chosen_part}`** ({usd(part_spend[chosen_part])} matched)")
            sold = buys[buys["part"] == chosen_part].sort_values("spend", ascending=False)
            st.dataframe(pd.DataFrame({"Supplier": sold["name"], "Price per unit": (sold["spend"] / sold["qty"]).map("${:.4f}".format),
                                       "Quantity": sold["qty"].map("{:,.0f}".format), "Spend": sold["spend"].map(usd)}),
                         hide_index=True, width="stretch", height="content")

    product_key, asof_key = f"cost_product_{dataset}", f"bom_as_of_{dataset}"
    changes = engineering_changes(bom)
    day_min, day_max = pd.to_datetime(base["po_date"]).min().date(), pd.to_datetime(base["po_date"]).max().date()
    if changes:
        day_max = max(day_max, date.fromisoformat(changes[-1][0]))
        st.subheader("Engineering changes")
        note("Click one to open that product's BOM as it stood on the day the change took effect.")
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
    with st.container(key="wide_drivers"):  # on a phone this chart keeps a readable width and scrolls sideways
        show_chart(
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
    note("Allocation, standard cost and the bought-versus-required lead use the BOM rows in effect in each build month.")
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
    note(f"{len(now)} rows in effect on {day}; {int((change != '').sum())} of them carry an engineering change. "
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
        show_chart(
            alt.Chart(aq).mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=60)).encode(
                x=alt.X("quarter:N", title=None, axis=alt.Axis(labelAngle=0)),
                y=alt.Y("vs_std:Q", axis=alt.Axis(format="+%", title="Estimated actual vs standard")),
                color=alt.Color("product_id:N", scale=alt.Scale(domain=product_ids, range=SERIES[:len(product_ids)]), legend=alt.Legend(title=None, orient="top")),
                tooltip=[alt.Tooltip("product_id:N", title="Product"), alt.Tooltip("quarter:N", title="Quarter"),
                         alt.Tooltip("vs_std:Q", title="Actual vs standard", format="+.1%"), alt.Tooltip("units_built:Q", title="Units built", format=",")],
            ).properties(height=260), width="stretch")
        note("An estimate. Parts are bought ahead of the builds that use them, so a single quarter can swing; lines still in review are not included.")

    st.subheader("Estimated inventory: bought minus built")
    note("An estimate, not a stock count: cumulative production purchases minus the gross BOM requirement (scrap included, multi-level BOMs "
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
        show_chart(
            alt.Chart(cum).mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=60)).encode(
                x=alt.X("quarter:N", title=None, axis=alt.Axis(labelAngle=0)), y=alt.Y("units:Q", axis=alt.Axis(format=",", title="Cumulative units")),
                color=alt.Color("series:N", scale=alt.Scale(domain=["Bought", "Needed for builds"], range=SERIES[:2]), legend=alt.Legend(title=None, orient="top")),
                tooltip=[alt.Tooltip("series:N", title="Series"), alt.Tooltip("quarter:N", title="Quarter"), alt.Tooltip("units:Q", title="Cumulative units", format=",.0f")],
            ).properties(height=240), width="stretch")

# ------------------------------------------------------------------- review
with tab_review:
    hero(f"{len(queue)} lines", "Waiting for a decision", [f"{short(queue['spend_usd'].sum())} waiting", f"Top {top_n} decisions reach {pct(reach)} tied",
                                                           f"{pct(tied)} tied now"])
    sign_col, _ = st.columns([2, 3])
    sign_col.select_slider("Needs sign-off above", options=SIGNOFF_STEPS, format_func=lambda v: f"${v:,}", key="signoff",
                           help="A match that only Claude vouched for waits for a person when the line is worth more than this.")
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

        left, right = st.columns([6, 5], gap="large")
        with left:
            st.markdown(f"**{len(groups)} items** covering {len(queue)} lines, largest dollars first")
            picked = table(pd.DataFrame({
                "Total": [("▶ " if i == pos else "") + short(v) for i, v in enumerate(groups["spend"])], "Lines": groups["lines"],
                "Text on the PO": groups["item_description"], "Suggested part": groups["suggested"], "Why": groups["tag"]}),
                key=f"queue_{len(xw)}", select=True,
                column_config={"Total": st.column_config.TextColumn(width=78), "Lines": st.column_config.NumberColumn(width=52),
                               "Text on the PO": st.column_config.TextColumn(width=250), "Suggested part": st.column_config.TextColumn(width=104)})
            if picked is not None and picked != pos:
                st.session_state["review_pos"] = picked
                st.rerun()
            note("Lines with identical text are one item: one decision settles all of them.")

        g = groups.iloc[pos]
        r = q[q["po_id"] == g["first"]].iloc[0]
        sug = g["suggested"]
        with right:
            st.markdown(f"Approving settles **{g['lines']} {'line' if g['lines'] == 1 else 'lines'}** ({usd(g['spend'])}) and moves tie-out by "
                        f"**{g['gain'] / total * 100:+.2f} pts**.")
            c1, c2, c3 = st.columns([1.5, 1.1, 1.3])
            for col, label, action, kind, keyboard in ((c1, f"Approve: {sug}", "approve", "primary", "A"), (c2, "Reject", "reject", "secondary", "X"),
                                                       (c3, "Not a BOM part", "no_match", "secondary", "N")):
                if col.button(label, type=kind, key=f"{action}_{g['first']}", width="stretch", shortcut=keyboard):
                    saved_add(r["supplier_part_number"], r["item_description"], sug, action, dataset)
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
            st.markdown(f"**Source** {r['source_file']}, row {r['source_row']} &nbsp;·&nbsp; "
                        f"**Part-number field** {r['supplier_part_number'] or '(blank)'} &nbsp;·&nbsp; **Supplier** {r['supplier']} &nbsp;·&nbsp; "
                        f"**Plant** {r['plant'] or '(blank)'}  \n**Order date** {r['po_date']} &nbsp;·&nbsp; **Order type** {r['po_type'] or '(blank)'} "
                        f"&nbsp;·&nbsp; **Quantity** {r['qty']:,.2f} {r['uom']} &nbsp;·&nbsp; **Price per unit** ${r['unit_price_usd']:.4f}"
                        + (f"  \nLargest of {g['lines']} lines with this text." if g["lines"] > 1 else ""))
            st.markdown("Keyboard: <kbd>A</kbd> approve &nbsp; <kbd>X</kbd> reject &nbsp; <kbd>N</kbd> not a BOM part", unsafe_allow_html=True)
            note("Approve ties these lines to the suggested part. Reject removes the suggestion and leaves them untied for a manual lookup. "
                       "Not a BOM part closes them as correctly unmatched.")

    st.divider()
    s1, s2_, s3_, s4_ = st.columns([3, 1, 1, 1.4])
    s1.markdown(f"**Saved decisions: {len(xw)}.** Each one applies to every line with the same text, now and on the next run."
                + (" Here they last for this browser session only." if HOSTED else ""))
    if s2_.button("Undo last", disabled=xw.empty, width="stretch"):
        saved_undo()
        st.rerun()
    if s3_.button("Clear all", disabled=xw.empty, width="stretch"):
        saved_clear()
        st.rerun()
    s4_.download_button("Download (CSV)", xw.to_csv(index=False), "saved_decisions.csv", "text/csv", disabled=xw.empty, width="stretch")
    if not xw.empty:
        keys = lines["supplier_part_number"].map(norm_pn) + "|" + lines["item_description"].map(norm_desc)
        settled = keys.value_counts()
        shown_xw = xw.assign(lines_settled=[int(settled.get(f"{norm_pn(a)}|{norm_desc(d)}", 0)) for a, d in zip(xw["supplier_part_number"], xw["item_description"])])
        table(shown_xw.rename(columns={"supplier_part_number": "Part-number field", "item_description": "Text on the PO", "part_number": "BOM part",
                                       "decision": "Decision", "decided_at": "Decided at", "dataset": "Data", "lines_settled": "Lines it settles here"}),
              key="saved")

# ----------------------------------------------------------------- evidence
with tab_evidence:
    if not is_sample:
        st.info("Uploaded files have no ground truth. Everything below was measured on the built-in sample and on external data.")
    s3, s4 = evals["eval_step3"], evals["eval_step4"]
    llm = s3["llm"][MAIN_MODEL]
    hero(pct(llm["heldout"]["tie_out"]), "Of spend tied, on styles the matcher never saw", [
        f"{round(llm['heldout']['accept_lines'] * (1 - llm['heldout']['accept_precision']))} wrong matches in this test", f"True rate {pct(llm['heldout']['true_tie_out'])}",
        f"{llm['heldout']['lines']:,} held-out lines"])

    abt_card, xw_card = evals.get("eval_external_abt_buy"), evals.get("eval_crosswalk")
    held = llm["heldout"]
    card1, card2, card3 = st.columns(3)
    with card1.container(key="card_ev1"):
        st.markdown("**Held-out accuracy**")
        st.markdown(kpi("Precision of accepted matches", f"{held['accept_precision']:.1%}", f"{held['accept_lines']:,} accepted lines; {usd(held['wrong_match_spend'])} matched wrongly"),
                    unsafe_allow_html=True)
        st.markdown(kpi("Tie-out", f"{held['tie_out']:.1%}", f"on {held['lines']:,} held-out lines, {usd(held['spend'])}", f"true rate {held['true_tie_out']:.1%}"),
                    unsafe_allow_html=True)
        note("Lines written in styles the matcher was never tuned on. Synthetic data.")
    with card2.container(key="card_ev2"):
        st.markdown("**Outside benchmark: Abt-Buy**")
        if abt_card:
            ours = abt_card["results"]["Rules + embeddings + Claude"]["f1"]
            published = ", ".join(f"{v['f1']:.2f}" for v in abt_card["published"].values())
            st.markdown(kpi("F1, nothing retrained", f"{ours:.2f}", f"{abt_card['test_pairs']} test pairs, {abt_card['test_matches']} true matches",
                            f"published baselines {published}"), unsafe_allow_html=True)
        note("Real product listings from two shops, fixed test split from the University of Mannheim benchmark page.")
    with card3.container(key="card_ev3"):
        st.markdown("**Review work over six quarters**")
        if xw_card:
            arm_with, arm_without = (pd.DataFrame(rows_x) for rows_x in xw_card["arms"].values())
            st.markdown(kpi("Lines left for a person", f"{int(arm_with['review_lines'].sum())} vs {int(arm_without['review_lines'].sum())}",
                            f"{usd(arm_with['review_spend'].sum())} vs {usd(arm_without['review_spend'].sum())} of spend left for review",
                            "with saved decisions vs without"), unsafe_allow_html=True)
        note("A simulated reviewer clears each quarter's queue. Item text repeats by construction in this data.")
    note("Details are in the sections below.")

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
        with st.container(key="wide_stages"):  # on a phone this chart keeps a readable width and scrolls sideways
            show_chart((bars + labels + rule).properties(height=alt.Step(34)), width="stretch")
        note(f"Dashed line: the true tie-out rate on these lines, {true_rate:.1%}. Measured on {full['lines']:,} held-out purchase lines "
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
        if stats.get("not_sent"):
            st.warning(f"Limit reached: {stats['call_limit']} Claude calls per run. {stats['not_sent']:,} smaller lines were not sent and are waiting in Review.")
        for note in uploaded["notes"]:
            note(note)
        note("Your own files have no answer key, so accuracy cannot be reported here. The Evidence tab shows how the same steps score "
                   "on the built-in sample and on outside data.")

    st.subheader("Where the spend ties")
    for col, b in zip(st.columns(4), BUCKETS):
        col.markdown(kpi(b.capitalize(), pct(by_bucket[b] / total), usd(by_bucket[b]), short(by_bucket[b])), unsafe_allow_html=True)
    names_b = [b.capitalize() for b in BUCKETS]
    st.subheader("How the spend got there")
    decided = lines["source"].map(lambda v: "Saved decisions" if v in ("crosswalk", "rule_crosswalk") else "Rules" if v.startswith("rule_")
                                  else "Claude" if v == "llm" else "Model")
    decided = decided.mask(lines["decision"].isin(["no_match", "rejected"]), "No match").mask(lines["decision"] == "review", "Waiting for a decision")
    flow = lines.assign(decided=decided, bucket_name=lines["bucket"].str.capitalize()).groupby(["decided", "bucket_name"])["spend_usd"].sum().reset_index()
    left_order = [g for g in ("Rules", "Model", "Claude", "Saved decisions", "Waiting for a decision", "No match") if (flow["decided"] == g).any()]
    if HAVE_ECHARTS:
        left_total, right_total = flow.groupby("decided")["spend_usd"].sum(), flow.groupby("bucket_name")["spend_usd"].sum()
        l_name = {g: f"{g}  {short(left_total[g])}" for g in left_order}
        r_name = {b: f"{b}  {short(right_total.get(b, 0.0))}" for b in names_b}
        with st.container(key="card_flow"):
            st_echarts({
                "textStyle": {"fontFamily": FONT},
                "tooltip": {"trigger": "item", "valueFormatter": JsCode("function (v) { return '$' + Math.round(v).toLocaleString('en-US'); }")},
                "series": [{
                    "type": "sankey", "left": 8, "right": 150, "top": 12, "bottom": 12, "nodeWidth": 14, "nodeGap": 14, "draggable": False,
                    "layoutIterations": 0, "emphasis": {"focus": "adjacency"},
                    "label": {"fontFamily": FONT, "fontSize": 13, "color": INK},
                    "lineStyle": {"color": "target", "opacity": 0.32, "curveness": 0.5},
                    "data": [{"name": l_name[g], "itemStyle": {"color": "#6f6e68"}} for g in left_order]
                            + [{"name": r_name[b], "itemStyle": {"color": c}} for b, c in zip(names_b, BUCKET_COLORS) if right_total.get(b, 0.0) > 0],
                    "links": [{"source": l_name[r.decided], "target": r_name[r.bucket_name], "value": float(round(r.spend_usd, 2))} for r in flow.itertuples()],
                }]}, height="330px", key="flow")
        note("Left: how each line was decided. Right: the bucket its spend landed in. Widths are dollars; the chart updates after every review decision.")
    else:
        show_chart(
            alt.Chart(flow).mark_bar(height=22, stroke=SURFACE, strokeWidth=2).encode(
                y=alt.Y("decided:N", sort=left_order, title=None), x=alt.X("spend_usd:Q", axis=alt.Axis(format="$~s", title="Spend")),
                color=alt.Color("bucket_name:N", scale=alt.Scale(domain=names_b, range=BUCKET_COLORS), legend=alt.Legend(title=None, orient="top")),
                tooltip=[alt.Tooltip("decided:N", title="Decided by"), alt.Tooltip("bucket_name:N", title="Bucket"),
                         alt.Tooltip("spend_usd:Q", title="Spend", format="$,.0f")]).properties(height=alt.Step(30)), width="stretch")
    inferred = lines[lines["bucket_inferred"]]
    note(f"Unassigned includes the {len(queue)} lines waiting for a decision. Where the order type was blank, the bucket was inferred from the "
               f"match and the plant: {len(inferred):,} lines ({short(inferred['spend_usd'].sum())}).")

    st.subheader("Tie-out by quarter")
    tq = lines.assign(un=lines["spend_usd"].where(lines["bucket"] == "unassigned", 0.0)).groupby("quarter")[["un", "spend_usd"]].sum().reset_index()
    tq["tie_out"] = 1 - tq["un"] / tq["spend_usd"]
    show_chart(
        alt.Chart(tq).mark_line(color=SERIES[0], strokeWidth=2, point=alt.OverlayMarkDef(size=70, color=SERIES[0])).encode(
            x=alt.X("quarter:N", title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("tie_out:Q", axis=alt.Axis(format="%", title="Spend tied out"), scale=alt.Scale(zero=False)),
            tooltip=[alt.Tooltip("quarter:N", title="Quarter"), alt.Tooltip("tie_out:Q", title="Tied out", format=".1%"),
                     alt.Tooltip("spend_usd:Q", title="Spend", format="$,.0f")]).properties(height=220), width="stretch")


    with st.expander("Held-out versus tuning split", expanded=False):
        note("The matcher was tuned on the tuning split only. The held-out split is written in noise styles it never saw: plant-specific "
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
        note("Only the uncertain middle band goes to Claude, so cost scales with how messy the data is, not with its size. "
                   "The small model is much cheaper, but its picks did not reach 99% precision on the tuning split, so none were auto-accepted.")

    with st.expander("Savings leads against the planted findings", expanded=False):
        st.dataframe(pd.DataFrame([{"Lead type": names[k], "Found": v["found"], "Planted": v["planted"], "Precision": f"{v['precision']:.0%}",
                                    "Recall": f"{v['recall']:.0%}", "Range": f"{usd(v['low_usd'])} to {usd(v['high_usd'])}"}
                                   for k, v in s4["leads"].items()]), hide_index=True, width="stretch", height="content")
        note(f"Measured at the default {usd(s4['materiality_usd'])} materiality threshold, before any reviewer action.")
        sim = s4.get("similar_parts")
        if sim:
            st.warning(f"**Similar parts to consolidate is unmeasured.** Nothing was planted for it, so there is no precision or recall. "
                       f"It produces {sim['found']} pairs worth {usd(sim['low_usd'])} to {usd(sim['high_usd'])}, every one of which engineering must confirm; "
                       "they are kept out of the headline range.")

    with st.expander("External tests: real data, nothing retrained", expanded=False):
        note("Zero-shot: the classifier and every cutoff come from the synthetic tuning split. The downloaded data is not in the repository; "
                   "`python3 -m src.external.fetch` gets it.")
        abt, ext = evals.get("eval_external_abt_buy"), evals.get("eval_external_parts")
        if abt:
            st.markdown(f"**A. Abt-Buy product matching** (University of Mannheim benchmark, fixed test split: {abt['test_pairs']} pairs, "
                        f"{abt['test_matches']} true matches)")
            st.dataframe(pd.DataFrame([{"Method": k, "Precision": f"{v['precision']:.2f}", "Recall": f"{v['recall']:.2f}", "F1": f"{v['f1']:.2f}"}
                                       for k, v in {**abt["results"], **abt["published"]}.items()]), hide_index=True, width="stretch", height="content")
            note(f"The benchmark's validation split was used only to check the model-code rule and set two similarity cutoffs. "
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
            note(f"When the lines sharing one item text add up to more than {usd(im['threshold_usd'])}, the classifier may not close the item as "
                       "\"no match\" on its own: it goes to Claude, and to a person if Claude is not confident either way. "
                       "This rule was added after seeing held-out results, so the \"after\" column is not a clean held-out measurement.")
            rows_im = {"Precision of accepted matches": lambda m: f"{m['accept_precision']:.2%}", "Recall": lambda m: pct(m["recall"]),
                       "Review queue": lambda m: f"{m['review_lines']} lines, {usd(m['review_spend'])}",
                       "No-match calls that are right": lambda m: pct(m["no_match_accuracy"]),
                       "Spend with a BOM part dropped as no match": lambda m: usd(m["no_match_missed_spend"]), "Wrong-match dollars": lambda m: usd(m["wrong_match_spend"])}
            st.table(pd.DataFrame({"Held-out, before": {k: f(im["before"]["heldout"]) for k, f in rows_im.items()},
                                   "Held-out, after": {k: f(im["after"]["heldout"]) for k, f in rows_im.items()}}))
            note(f"{im['held_from_classifier_no_match']} lines across both splits were kept open by this rule. "
                       f"Lines sent to Claude rose from {im['before']['band_lines']} to {im['after']['band_lines']}.")

    with st.expander("Experiment: do saved decisions cut review work over time?", expanded=False):
        xe = evals.get("eval_crosswalk")
        if xe:
            note("Purchases are processed one quarter at a time. Each quarter the pipeline runs with the decisions saved so far, then a simulated "
                       "reviewer clears that quarter's review queue from the truth and the decisions are saved. The comparison is the same run with nothing saved.")
            plain = {"With the crosswalk": "With saved decisions", "No crosswalk": "Without saved decisions"}
            xr = pd.concat([pd.DataFrame(rows_x).assign(arm=plain.get(arm, arm)) for arm, rows_x in xe["arms"].items()])
            arms = [plain.get(a, a) for a in xe["arms"]]
            show_chart(
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
        note("Can the classifier be trained on a dataset's own rule-matched lines, with no confirmed labels? Positives are the lines a rule "
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
            note("The catalog has prices but no quantities, so wrong-match dollars cannot be computed for the real-parts test.")
        if sa or sb:
            st.markdown("""
    - **It does not replace confirmed labels.** On the synthetic data the self-trained classifier accepts more lines, but its precision falls from 100% to about 96.5% and it produces wrong-match dollars where the truth-trained one has none. Rule-matched lines always have a true part, so the model never sees a line that should be left unmatched.
    - **On real parts it barely moves.** Only a third of the training lines had their true part among the five nearest candidates, the thresholds collapse to "accept almost nothing", and recall goes from 0.00 to 0.02. Its one gain: it stops wrongly accepting no-match lines.
    - **Conclusion:** self-training is not a shortcut. The classifier needs reviewer-confirmed matches, which is what the crosswalk collects.
    """)

    with st.expander("Limits", expanded=False):
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
    - **"Similar parts to consolidate" is unmeasured.** Nothing was planted for it, so it has no precision or recall. It pairs parts whose descriptions differ in one close number or the package, which can pair parts that are not interchangeable at all. Engineering must confirm every one.
- **Lead status is kept for the session only.** Marking a lead validated or rejected is not saved to disk.
- **Leads are not savings.** A price gap can have a good reason. Each one needs a buyer's judgment before it means anything.
    - **Your own files skip the trained model.** They are matched with rules, embeddings and Claude, and there is no ground truth to score the result against.
    - **Saved decisions live in a local file.** They are reused as the first rule on the next run; a real deployment would also retrain the model on them.
    - **Trends and inventory are rough.** They have no opening stock, scrap, or lead time between buying and building.
    - **What a pilot would test:** match accuracy on one plant's real exports, how much spend lands in review, and how long a reviewer takes to clear it.
    """)


# ---------------------------------------------------------------- your data
with tab_data:
    hero("Run it on your own exports", "CSV or Excel, any column names", ["1 Upload", "2 Confirm columns", "3 Run"], text=True)
    if uploaded:
        st.radio("Showing in every tab", options, key="dataset", horizontal=True)
    code = settings.secret("ACCESS_CODE") if HOSTED else ""  # optional: with no code set, uploads are open to everyone
    unlocked = not code or st.session_state.get("unlocked", False)
    with st.container():
        up = st.session_state.setdefault("up_files", {})

        st.markdown("#### 1. Upload")
        if st.button("Try the second sample", type="primary", help="The same kind of data as the built-in sample, exported with different column names."):
            for kind, name in SAMPLE_FILES.items():
                up[kind] = (name, (SAMPLE_DIR / name).read_bytes())
            st.session_state.pop("up_map", None)
            st.rerun()
        note(f"Or upload your own three files, up to {UPLOAD_LIMIT_MB} MB each. Headers and 20 sample rows of each file go to Claude for the "
             "column mapping; purchase descriptions and BOM parts go to the embedding model and Claude for matching."
             + (f" On this hosted copy: at most {HOSTED_MAX_LINES:,} purchase lines, {HOSTED_MAX_CALLS} Claude calls per run and "
                f"{HOSTED_MAX_RUNS} runs per session. Your files and decisions are visible only in your own session, are not written to disk, "
                "and are gone when you reload the page." if HOSTED else ""), label="Limits and privacy")
        if unlocked:
            for col, kind in zip(st.columns(3), KIND_LABEL):
                f = col.file_uploader(KIND_LABEL[kind], type=["csv", "xlsx", "xls"], key=f"file_{kind}")
                if f is not None and f.size > UPLOAD_LIMIT_MB * 1_000_000:
                    col.error(f"{f.name} is over the {UPLOAD_LIMIT_MB} MB limit.")
                elif f is not None:
                    up[kind] = (f.name, f.getvalue())
                if kind in up:
                    col.caption(f"Loaded: {up[kind][0]}")
        else:  # the sample above makes no API calls and stays open; real uploads spend the owner's API budget
            st.markdown("Uploading your own files on this hosted copy needs an access code. The sample above does not.")
            code_col, _ = st.columns([2, 3])
            entered = code_col.text_input("Access code", type="password", key="access_code_entry")
            if code_col.button("Unlock"):
                if hmac.compare_digest(entered.encode(), code.encode()):
                    st.session_state["unlocked"] = True
                    st.rerun()
                else:
                    st.error("That code is not right.")

        if len(up) == 3:
            raw = {kind: mapping.read_table(*up[kind]) for kind in KIND_LABEL}
            is_demo = all(up[kind][0] == SAMPLE_FILES[kind] and up[kind][1] == (SAMPLE_DIR / SAMPLE_FILES[kind]).read_bytes() for kind in KIND_LABEL)
            # The committed sample reads its answers from the repository. Anything else is cached locally, or not at all on a public host.
            llm_cache = SAMPLE_CACHE if is_demo else None if HOSTED else pipeline.UPLOAD_CACHE
            runs_used = st.session_state.get("runs_used", 0)
            blocked = [] if is_demo or unlocked else ["Running uploaded files needs the access code."]
            if HOSTED and not is_demo and len(raw["purchases"]) > HOSTED_MAX_LINES:
                blocked.append(f"The purchase file has {len(raw['purchases']):,} lines. This hosted copy accepts at most {HOSTED_MAX_LINES:,} per run.")
            counter = daily_runs()
            if counter["day"] != date.today().isoformat():
                counter.update(day=date.today().isoformat(), runs=0)
            if HOSTED and not is_demo and counter["runs"] >= HOSTED_DAILY_RUNS:
                blocked.append(f"This hosted copy has reached today's limit of {HOSTED_DAILY_RUNS} runs on uploaded files across all visitors. "
                               "The second sample still works.")
            if HOSTED and runs_used >= HOSTED_MAX_RUNS:
                blocked.append(f"This session has used its {HOSTED_MAX_RUNS} runs. Reload the page to start a new session.")
            for message in blocked:
                st.error(message)
            st.markdown("#### 2. Confirm columns")
            if st.button("Ask Claude to map the columns", type="secondary" if "up_map" in st.session_state else "primary"):
                try:
                    with st.spinner("Claude is reading the headers and 20 sample rows of each file"):
                        proposals = {kind: mapping.propose(kind, raw[kind], cache=llm_cache)[0] for kind in KIND_LABEL}
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
                                note(f"Claude: {row.confidence:.0%} sure. {row.note}" if row.source else f"Claude found no column. {row.note}")
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
                if st.button("Run the pipeline", type="primary", disabled=bool(problems or blocked)):
                    try:
                        with st.spinner("Matching: rules, then nearest descriptions, then Claude on the open lines. A minute or two the first time."):
                            canon, notes = {}, []
                            for kind in KIND_LABEL:
                                canon[kind], kind_notes = mapping.apply_mapping(kind, raw[kind], chosen_map[kind])
                                notes += kind_notes
                            inp = prepare(canon["purchases"], canon["boms"], canon["builds"], source_file=up["purchases"][0])
                            # The sample runs without saved decisions so every answer comes from the committed cache; they still apply afterwards.
                            dec, stats = pipeline.run(inp, {} if is_demo else crosswalk.as_rules(saved_load()), cache=llm_cache,
                                                      max_calls=HOSTED_MAX_CALLS if HOSTED and not is_demo else pipeline.MAX_CALLS)
                            st.session_state["runs_used"] = runs_used + 1
                            if HOSTED and not is_demo:
                                counter["runs"] += 1
                            st.session_state["uploaded"] = dict(
                                lines=allocate.build_lines(inp, dec), bom=inp.boms, builds=inp.builds, stats=stats, notes=notes,
                                files=[up[kind][0] for kind in KIND_LABEL], demo_sample=is_demo)
                            st.session_state["_switch_to_uploaded"] = True
                    except Exception as e:
                        st.error(f"The run failed: {e}")
                    else:
                        st.rerun()
        if uploaded:
            st.success(f"Last run: {', '.join(uploaded['files'])}: {uploaded['stats']['lines']:,} purchase lines. Every tab now shows these files; "
                       "switch back with \"Showing in every tab\" at the top of this tab.")
