"""Buckets, allocation to products, and savings leads.

Reads out/decisions.csv and data/*.csv only. Never reads data/truth/.

    python3 -m src.allocate         # writes out/lines.csv, out/product_cost.csv, out/leads.csv

Everything here is an estimate from purchase records: purchases show what was bought,
not what a product consumed. Leads are leads, not booked savings.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .normalize import Inputs, load_inputs, norm_desc, norm_pn, numbers_in, parse_uom

MATERIALITY_USD = 25_000     # a Claude-accepted line above this goes to a person anyway
PRICE_GAP_MIN = 0.05         # ignore supplier price differences smaller than this
SPARES_PREMIUM_MIN = 0.35    # spares price over production price before it becomes a lead
OVER_PURCHASE_MIN = 0.15     # bought this far above the gross requirement before it becomes a lead
LEAD_LOW = 0.5               # low end of every lead's range: half the gap closed


def load_lines(inp: Inputs, decisions_path: str | Path = "out/decisions.csv") -> pd.DataFrame:
    return build_lines(inp, pd.read_csv(decisions_path, dtype={"po_id": str}, keep_default_na=False))


def build_lines(inp: Inputs, dec: pd.DataFrame) -> pd.DataFrame:
    cols = ["po_id", "po_date", "plant", "supplier", "supplier_key", "supplier_part_number", "item_description", "uom", "po_type",
            "qty", "base_qty", "unit_price_usd", "spend_usd"]
    lines = inp.purchases[cols].merge(dec[["po_id", "decision", "source", "part", "prob", "llm_part", "llm_conf", "llm_reason"]], on="po_id", validate="1:1")
    # the matcher's verdict before the materiality rule or any reviewer action; the app starts from these
    lines["decision_pre"], lines["source_pre"] = lines["decision"], lines["source"]
    lines["suggested_part"] = lines["llm_part"].where(lines["llm_part"] != "", lines["part"])
    date = pd.to_datetime(lines["po_date"])
    lines["quarter"] = date.dt.year.astype(str) + "Q" + date.dt.quarter.astype(str)
    return lines


def apply_materiality(lines: pd.DataFrame, threshold: float = MATERIALITY_USD) -> pd.DataFrame:
    """A match that only Claude vouched for is not allowed to move a large amount of money on its own."""
    out = lines.copy()
    big = (out["decision"] == "accept") & (out["source"] == "llm") & (out["spend_usd"] > threshold)
    out.loc[big, ["decision", "source"]] = ["review", "llm_material"]
    return out


def apply_crosswalk(lines: pd.DataFrame, crosswalk: dict) -> pd.DataFrame:
    """Reviewer decisions, keyed by item text, override the matcher on every line with that text."""
    out = lines.copy()
    if not crosswalk:
        return out
    key = out["supplier_part_number"].map(norm_pn) + "|" + out["item_description"].map(norm_desc)
    for k, (decision, part) in crosswalk.items():
        row = key == k
        if decision == "approve":
            out.loc[row, "part"] = part
        out.loc[row, ["decision", "source"]] = [{"approve": "accept", "no_match": "no_match", "reject": "rejected"}[decision], "crosswalk"]
    return out


def in_effect(boms: pd.DataFrame, as_of: str | None = None) -> pd.Series:
    """Which BOM rows are in effect on a date (ISO text). No date means the current BOM: rows with no end date."""
    if "effective_to" not in boms:
        return pd.Series(True, index=boms.index)
    start, end = boms["effective_from"].fillna("").astype(str), boms["effective_to"].fillna("").astype(str)
    if as_of is None:
        return end == ""
    return ((start == "") | (start <= as_of)) & ((end == "") | (end >= as_of))


def flatten_bom(boms: pd.DataFrame, as_of: str | None = None) -> pd.DataFrame:
    """Explode a multi-level BOM down to the bought parts, per product, using the rows in effect on as_of.

    qty_per_unit in the result is the gross quantity of the bought part per finished unit: the net quantities
    multiplied down the tree, each divided by (1 - scrap_pct) of its own row. Raises ValueError on a cycle or
    on a parent that is not on the product's BOM. A BOM without parent_part is treated as single-level.
    """
    b = boms[in_effect(boms, as_of)].copy()
    for col, default in (("scrap_pct", 0.0), ("parent_part", ""), ("make_buy", "buy")):
        if col not in b:
            b[col] = default
    b["parent_part"] = b["parent_part"].fillna("").astype(str)
    b["gross"] = b["qty_per_unit"] / (1 - b["scrap_pct"].fillna(0.0).clip(0, 0.95))
    out = []
    for product, rows in b.groupby("product_id", sort=False):
        children, parent_of = {}, {}
        for r in rows.itertuples():
            children.setdefault(r.parent_part, []).append(r)
            parent_of.setdefault(r.part_number, r.parent_part)
        for part in parent_of:  # walk up from every row: a repeat means a cycle, a dead end means an orphan
            seen, cur = [part], parent_of[part]
            while cur != "":
                if cur in seen:
                    raise ValueError(f"BOM cycle in {product}: {' -> '.join(seen[seen.index(cur):] + [cur])}")
                if cur not in parent_of:
                    raise ValueError(f"BOM error in {product}: parent {cur} of {seen[-1]} is not on the BOM")
                seen.append(cur)
                cur = parent_of[cur]

        def walk(parent: str, mult: float):
            for r in children.get(parent, []):
                if r.part_number in children or r.make_buy == "make":
                    walk(r.part_number, mult * r.gross)  # an in-house item: its own scrap raises everything below it
                else:
                    out.append(dict(product_id=product, part_number=r.part_number, assembly_id=r.assembly_id, description=r.description,
                                    uom=r.uom, std_cost=r.std_cost, qty_per_unit=mult * r.gross))
        walk("", 1.0)
    flat = pd.DataFrame(out, columns=["product_id", "part_number", "assembly_id", "description", "uom", "std_cost", "qty_per_unit"])
    return flat.groupby(["product_id", "part_number"], sort=False).agg(
        assembly_id=("assembly_id", "first"), description=("description", "first"), uom=("uom", "first"), std_cost=("std_cost", "first"),
        qty_per_unit=("qty_per_unit", "sum")).reset_index()


def requirements(boms: pd.DataFrame, builds: pd.DataFrame) -> pd.DataFrame:
    """Gross requirement of every bought part, per product and build month, using the BOM in effect that month.

    The BOM is taken as of the first day of each build month, so an engineering change dated mid-month counts from
    the following month. Columns: product_id, part_number, period, quarter, units_built, qty_per_unit, std_cost, need.
    """
    b = builds.copy()
    month = pd.to_datetime(b["period"])
    b["as_of"], b["quarter"] = month.dt.strftime("%Y-%m-01"), month.dt.year.astype(str) + "Q" + month.dt.quarter.astype(str)
    versions, out = {}, []
    for as_of, g in b.groupby("as_of"):
        key = tuple(in_effect(boms, as_of))  # months with the same rows in effect share one flattened BOM
        if key not in versions:
            versions[key] = flatten_bom(boms, as_of)
        out.append(versions[key].merge(g[["product_id", "period", "quarter", "units_built"]], on="product_id"))
    cols = ["product_id", "part_number", "assembly_id", "description", "uom", "std_cost", "qty_per_unit", "period", "quarter", "units_built"]
    req = pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=cols)
    req["need"] = req["qty_per_unit"] * req["units_built"]
    return req


def apply_units(lines: pd.DataFrame, boms: pd.DataFrame) -> pd.DataFrame:
    """Express each line's quantity in the unit its BOM part is measured in (qty_bom).

    Same dimension: converted (KG to G, FT to M, box of 100 to each). Different dimension (wire bought by weight,
    measured by length on the BOM): no conversion is possible without a factor, so qty_bom is blank, uom_issue says
    why, and a matched line goes to review unless a reviewer has already decided it.
    """
    out = lines.copy()
    bom_uom = boms.drop_duplicates("part_number").set_index("part_number")["uom"].to_dict()
    qty_bom, issue = [], []
    for part, uom, qty in zip(out["part"], out["uom"], out["qty"]):
        if part not in bom_uom:
            qty_bom.append(np.nan); issue.append("")
            continue
        (ldim, lsize), (bdim, bsize) = parse_uom(uom), parse_uom(bom_uom[part], bom_side=True)
        if str(uom).upper().strip() == "M" and bdim == "length":
            ldim, lsize = "length", 1.0
        if ldim == bdim:
            qty_bom.append(qty * lsize / bsize); issue.append("")
        else:
            qty_bom.append(np.nan)
            issue.append(f"Bought in {uom} ({ldim}) but the BOM measures {part} in {bom_uom[part]} ({bdim}); the quantity cannot be converted without a factor")
    out["qty_bom"], out["uom_issue"] = qty_bom, issue
    hold = (out["decision"] == "accept") & (out["uom_issue"] != "") & ~out["source"].isin(["crosswalk", "rule_crosswalk"])
    out.loc[hold, ["decision", "source"]] = ["review", "uom_mismatch"]
    return out


def assign_buckets(lines: pd.DataFrame) -> pd.DataFrame:
    """production | spares | warranty | unassigned, plus bucket_inferred.

    PO type present: service -> spares, warranty -> warranty, production -> production if matched to a BOM part.
    PO type blank:   a matched line is spares at a service location and production elsewhere (inferred).
    A line that is neither matched nor typed service/warranty ties to nothing: unassigned.
    Lines still in review count as unmatched until a person confirms them.
    """
    out = lines.copy()
    matched = out["decision"] == "accept"
    typed = out[out["po_type"] != ""]
    service_share = typed.assign(svc=typed["po_type"].isin(["service", "warranty"]) * typed["spend_usd"]).groupby("plant")[["svc", "spend_usd"]].sum()
    service_plants = set(service_share.index[service_share["svc"] > 0.5 * service_share["spend_usd"]])  # learned from the typed lines
    at_service = out["plant"].isin(service_plants)

    out["bucket"], out["bucket_inferred"] = "unassigned", False
    out.loc[out["po_type"] == "service", "bucket"] = "spares"
    out.loc[out["po_type"] == "warranty", "bucket"] = "warranty"
    out.loc[(out["po_type"] == "production") & matched, "bucket"] = "production"
    blank = (out["po_type"] == "") & matched
    out.loc[blank & at_service, "bucket"] = "spares"
    out.loc[blank & ~at_service, "bucket"] = "production"
    out.loc[blank, "bucket_inferred"] = True
    return out


def tie_out(lines: pd.DataFrame) -> pd.DataFrame:
    by = lines.groupby("bucket")["spend_usd"].sum().reindex(["production", "spares", "warranty", "unassigned"], fill_value=0.0)
    return pd.DataFrame({"spend_usd": by, "share": by / by.sum()})


# ----------------------------------------------------------------- allocation

def allocate(lines: pd.DataFrame, boms: pd.DataFrame, builds: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split matched production spend across products by gross quantity per unit x units_built in the purchase's quarter.
    The BOM is flattened to bought parts first, so parts inside in-house subassemblies are allocated like any other.
    Each build month uses the BOM rows in effect that month, so an engineering change moves the weights from its date.

    Returns (allocation rows: product, quarter, part, spend; product table with estimated actual cost per unit
    beside the BOM standard cost).
    """
    req = requirements(boms, builds)
    demand = req.groupby(["part_number", "product_id", "quarter"], as_index=False)["need"].sum()
    whole = demand.groupby(["part_number", "product_id"])["need"].sum().rename("need_all").reset_index()

    prod = lines[(lines["bucket"] == "production") & (lines["decision"] == "accept")]
    spend = prod.groupby(["part", "quarter"])["spend_usd"].sum().rename("spend").reset_index().rename(columns={"part": "part_number"})
    rows = spend.merge(demand[["part_number", "product_id", "quarter", "need"]], on=["part_number", "quarter"], how="left").merge(
        whole, on=["part_number", "product_id"], how="left")
    # weight by the quarter's need; if nothing using the part was built that quarter, fall back to the whole period
    q_total = rows.groupby(["part_number", "quarter"])["need"].transform("sum")
    weight = np.where(q_total > 0, rows["need"], rows["need_all"])
    rows["allocated_usd"] = rows["spend"] * weight / pd.Series(weight).groupby([rows["part_number"], rows["quarter"]]).transform("sum").to_numpy()
    alloc = rows[["product_id", "quarter", "part_number", "allocated_usd"]]

    units_built = builds.groupby("product_id")["units_built"].sum()
    std = req.assign(c=req["need"] * req["std_cost"]).groupby("product_id")["c"].sum(min_count=1) / units_built  # average over the period's builds
    products = pd.DataFrame({
        "units_built": units_built,
        "std_cost_per_unit": std,
        "allocated_production_spend_est": alloc.groupby("product_id")["allocated_usd"].sum(),
    }).fillna({"allocated_production_spend_est": 0.0})
    products["actual_cost_per_unit_est"] = products["allocated_production_spend_est"] / products["units_built"]
    products["actual_vs_std"] = products["actual_cost_per_unit_est"] / products["std_cost_per_unit"] - 1
    return alloc, products.reset_index()


# ---------------------------------------------------------------------- leads

def _vwap(df: pd.DataFrame, by) -> pd.DataFrame:
    g = df[df["qty_bom"].notna()].groupby(by).agg(spend=("spend_usd", "sum"), qty=("qty_bom", "sum"))
    g["price"] = g["spend"] / g["qty"]
    return g


def find_leads(lines: pd.DataFrame, boms: pd.DataFrame, builds: pd.DataFrame | None = None) -> pd.DataFrame:
    """Each lead carries a dollar range: LEAD_LOW of the gap closed, up to all of it."""
    bought_rows = boms[boms["make_buy"] != "make"] if "make_buy" in boms else boms
    bom_parts = bought_rows.drop_duplicates("part_number")[["part_number", "description"]].reset_index(drop=True)
    bom_parts["desc_norm"] = bom_parts["description"].map(norm_desc)
    ok = lines[lines["decision"] == "accept"]
    prod, spares = ok[ok["bucket"] == "production"], ok[ok["bucket"] == "spares"]
    name = lines.groupby("supplier_key")["supplier"].agg(lambda s: s.value_counts().index[0])
    leads = []

    # 1. same part, different unit price across suppliers (production buys only, volume-weighted)
    by_sup = _vwap(prod, ["part", "supplier_key"]).reset_index()
    for part, g in by_sup.groupby("part"):
        if len(g) < 2:
            continue
        low = g.loc[g["price"].idxmin()]
        dear = g[g["price"] > low["price"] * (1 + PRICE_GAP_MIN)]
        if dear.empty:
            continue
        gap = ((dear["price"] - low["price"]) * dear["qty"]).sum()
        leads.append(dict(lead_type="supplier_price_gap", part_number=part, related="; ".join(name[dear["supplier_key"]]),
                          detail=f"{name[low['supplier_key']]} at ${low['price']:.4f}; " + "; ".join(
                              f"{name[s]} at ${pr:.4f} (+{pr / low['price'] - 1:.0%})" for s, pr in zip(dear["supplier_key"], dear["price"])),
                          gap_usd=gap))

    # 2. two internal part numbers for one physical part: same family, same words, same numbers
    parts = bom_parts.assign(family=bom_parts["part_number"].str.split("-").str[0],
                             words=bom_parts["desc_norm"].map(lambda d: frozenset(d.split())), nums=bom_parts["desc_norm"].map(numbers_in))
    price = _vwap(prod, "part")
    same_part = {}  # duplicate number -> the number it duplicates, so quantities can be pooled below
    for _, g in parts.groupby(["family", "words", "nums"]):
        if len(g) < 2:
            continue
        a, b = sorted(g["part_number"])[:2]
        same_part[b] = a
        if a in price.index and b in price.index:
            cheap, dear = sorted([a, b], key=lambda x: price.at[x, "price"])
            gap = (price.at[dear, "price"] - price.at[cheap, "price"]) * price.at[dear, "qty"]
            detail = f"{cheap} at ${price.at[cheap, 'price']:.4f}; {dear} at ${price.at[dear, 'price']:.4f} (+{price.at[dear, 'price'] / price.at[cheap, 'price'] - 1:.0%})"
        else:
            gap, detail = 0.0, "same description; no matched purchases on one of them yet"
        leads.append(dict(lead_type="duplicate_part", part_number=a, related=b, detail=detail, gap_usd=gap))

    # 3. spares bought at a premium over the production price of the same part
    sp, pr = _vwap(spares, "part"), _vwap(prod, "part")
    both = sp.join(pr, lsuffix="_spares", rsuffix="_prod", how="inner")
    both = both[both["price_spares"] > both["price_prod"] * (1 + SPARES_PREMIUM_MIN)]
    for part, r in both.iterrows():
        leads.append(dict(lead_type="spares_premium", part_number=part, related="",
                          detail=f"spares at ${r['price_spares']:.4f} vs production at ${r['price_prod']:.4f} (+{r['price_spares'] / r['price_prod'] - 1:.0%})",
                          gap_usd=(r["price_spares"] - r["price_prod"]) * r["qty_spares"]))

    # 4. bought more than the builds required, over the trailing 12 months. A gap, not a diagnosis.
    if builds is not None and len(prod):
        dates = pd.to_datetime(prod["po_date"])
        end = pd.to_datetime(lines["po_date"]).max()
        start = end - pd.DateOffset(months=12)
        pooled = prod[dates > start].assign(part=lambda d: d["part"].replace(same_part))  # duplicates are one physical part
        recent = _vwap(pooled, "part")
        months = pd.to_datetime(builds["period"]).dt.to_period("M")
        req = requirements(boms, builds[(months > start.to_period("M")) & (months <= end.to_period("M"))])  # the BOM in effect each month
        req = req.assign(part_number=req["part_number"].replace(same_part))
        need, uom = req.groupby("part_number")["need"].sum(), req.drop_duplicates("part_number").set_index("part_number")["uom"]
        for part, r in recent.iterrows():
            required = need.get(part, 0.0)
            if required > 0 and r["qty"] > required * (1 + OVER_PURCHASE_MIN):
                leads.append(dict(lead_type="over_purchase", part_number=part, related="",
                                  detail=f"bought {r['qty']:,.0f} {uom[part]} against {required:,.0f} required by builds in the 12 months to "
                                         f"{end:%b %Y} (+{r['qty'] / required - 1:.0%}); cause not determined",
                                  gap_usd=(r["qty"] - required) * r["price"]))

    out = pd.DataFrame(leads, columns=["lead_type", "part_number", "related", "detail", "gap_usd"])
    out["low_usd"], out["high_usd"] = (out["gap_usd"] * LEAD_LOW).round(0), out["gap_usd"].round(0)
    desc = bom_parts.set_index("part_number")["description"]
    out.insert(2, "description", out["part_number"].map(desc))
    return out.drop(columns="gap_usd").sort_values("high_usd", ascending=False).reset_index(drop=True)


def run(decisions_path: str | Path = "out/decisions.csv", data_dir: str | Path = "data", materiality: float = MATERIALITY_USD):
    inp = load_inputs(data_dir)
    lines = assign_buckets(apply_units(apply_materiality(load_lines(inp, decisions_path), materiality), inp.boms))
    alloc, products = allocate(lines, inp.boms, inp.builds)
    return inp, lines, alloc, products, find_leads(lines, inp.boms, inp.builds)


def write_outputs(inp: Inputs, lines, alloc, products, leads, out: str | Path = "out"):
    """Everything the app needs, so it can run from out/ alone."""
    out = Path(out)
    out.mkdir(exist_ok=True)
    for name, df in (("lines", lines), ("allocation", alloc), ("product_cost", products), ("leads", leads),
                     ("bom", inp.boms), ("builds", inp.builds)):
        df.to_csv(out / f"{name}.csv", index=False)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--materiality", type=float, default=MATERIALITY_USD, help="USD; Claude-accepted lines above this go to review")
    args = ap.parse_args()
    inp, lines, alloc, products, leads = run(materiality=args.materiality)
    write_outputs(inp, lines, alloc, products, leads)

    t = tie_out(lines)
    print(f"tie-out rate {1 - t.at['unassigned', 'share']:.1%} of ${t['spend_usd'].sum():,.0f}")
    for bucket, r in t.iterrows():
        print(f"  {bucket:<11} ${r['spend_usd']:>12,.0f}  {r['share']:6.1%}")
    inferred = lines[lines["bucket_inferred"]]
    review = lines[lines["decision"] == "review"]
    material = (lines["source"] == "llm_material").sum()
    print(f"bucket inferred (blank PO type) on {len(inferred)} lines, ${inferred['spend_usd'].sum():,.0f}")
    print(f"review queue {len(review)} lines, ${review['spend_usd'].sum():,.0f} ({material} moved there by the ${args.materiality:,.0f} materiality rule)")
    print("\nparts cost per unit, estimated from purchases vs BOM standard")
    print(products.to_string(index=False, formatters={"std_cost_per_unit": "${:.2f}".format, "actual_cost_per_unit_est": "${:.2f}".format,
                                                      "allocated_production_spend_est": "${:,.0f}".format, "actual_vs_std": "{:+.1%}".format}))
    s = leads.groupby("lead_type").agg(leads=("part_number", "size"), low_usd=("low_usd", "sum"), high_usd=("high_usd", "sum"))
    print("\nleads (ranges, not booked savings)")
    print(s.to_string(formatters={"low_usd": "${:,.0f}".format, "high_usd": "${:,.0f}".format}))
    print("wrote out/lines.csv, allocation.csv, product_cost.csv, leads.csv, bom.csv, builds.csv")
