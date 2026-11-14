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

from .normalize import Inputs, load_inputs, norm_desc, norm_pn, numbers_in

MATERIALITY_USD = 25_000     # a Claude-accepted line above this goes to a person anyway
PRICE_GAP_MIN = 0.05         # ignore supplier price differences smaller than this
SPARES_PREMIUM_MIN = 0.35    # spares price over production price before it becomes a lead
LEAD_LOW = 0.5               # low end of every lead's range: half the gap closed


def load_lines(inp: Inputs, decisions_path: str | Path = "out/decisions.csv") -> pd.DataFrame:
    return build_lines(inp, pd.read_csv(decisions_path, dtype={"po_id": str}, keep_default_na=False))


def build_lines(inp: Inputs, dec: pd.DataFrame) -> pd.DataFrame:
    cols = ["po_id", "po_date", "plant", "supplier", "supplier_key", "supplier_part_number", "item_description", "uom", "po_type",
            "base_qty", "unit_price_usd", "spend_usd"]
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
    """Split matched production spend across products by qty_per_unit x units_built in the purchase's quarter.

    Returns (allocation rows: product, quarter, part, spend; product table with estimated actual cost per unit
    beside the BOM standard cost).
    """
    builds = builds.copy()
    period = pd.to_datetime(builds["period"])
    builds["quarter"] = period.dt.year.astype(str) + "Q" + period.dt.quarter.astype(str)
    units_q = builds.groupby(["product_id", "quarter"])["units_built"].sum().rename("units").reset_index()
    usage = boms.groupby(["part_number", "product_id"])["qty_per_unit"].sum().reset_index()

    demand = usage.merge(units_q, on="product_id")
    demand["need"] = demand["qty_per_unit"] * demand["units"]
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

    std = boms.assign(c=boms["qty_per_unit"] * boms["std_cost"]).groupby("product_id")["c"].sum(min_count=1)
    products = pd.DataFrame({
        "units_built": builds.groupby("product_id")["units_built"].sum(),
        "std_cost_per_unit": std,
        "allocated_production_spend_est": alloc.groupby("product_id")["allocated_usd"].sum(),
    }).fillna({"allocated_production_spend_est": 0.0})
    products["actual_cost_per_unit_est"] = products["allocated_production_spend_est"] / products["units_built"]
    products["actual_vs_std"] = products["actual_cost_per_unit_est"] / products["std_cost_per_unit"] - 1
    return alloc, products.reset_index()


# ---------------------------------------------------------------------- leads

def _vwap(df: pd.DataFrame, by) -> pd.DataFrame:
    g = df.groupby(by).agg(spend=("spend_usd", "sum"), qty=("base_qty", "sum"))
    g["price"] = g["spend"] / g["qty"]
    return g


def find_leads(lines: pd.DataFrame, boms: pd.DataFrame) -> pd.DataFrame:
    """Each lead carries a dollar range: LEAD_LOW of the gap closed, up to all of it."""
    bom_parts = boms.drop_duplicates("part_number")[["part_number", "description"]].reset_index(drop=True)
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
    for _, g in parts.groupby(["family", "words", "nums"]):
        if len(g) < 2:
            continue
        a, b = sorted(g["part_number"])[:2]
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

    out = pd.DataFrame(leads, columns=["lead_type", "part_number", "related", "detail", "gap_usd"])
    out["low_usd"], out["high_usd"] = (out["gap_usd"] * LEAD_LOW).round(0), out["gap_usd"].round(0)
    desc = bom_parts.set_index("part_number")["description"]
    out.insert(2, "description", out["part_number"].map(desc))
    return out.drop(columns="gap_usd").sort_values("high_usd", ascending=False).reset_index(drop=True)


def run(decisions_path: str | Path = "out/decisions.csv", data_dir: str | Path = "data", materiality: float = MATERIALITY_USD):
    inp = load_inputs(data_dir)
    lines = assign_buckets(apply_materiality(load_lines(inp, decisions_path), materiality))
    alloc, products = allocate(lines, inp.boms, inp.builds)
    return inp, lines, alloc, products, find_leads(lines, inp.boms)


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
