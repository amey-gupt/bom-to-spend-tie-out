"""Synthetic data generator with planted ground truth.

    python3 -m src.generate            # writes data/ and data/truth/

Two layers, kept separate on purpose:
  1. build_backbone(): clean engineering data (parts, BOMs, builds, sourcing).
  2. make_purchases(): the purchasing view of the same parts, with noise.
A different backbone (for example real BOMs) can be passed to layer 2.

The matcher may read data/*.csv only. data/truth/ is for evaluation.
All data is synthetic. Standard library only; fixed seed; deterministic.
"""
from __future__ import annotations

import argparse
import csv
import math
import random
from collections import defaultdict
from dataclasses import dataclass, field, replace
from datetime import date, timedelta
from pathlib import Path

from . import noise
from .catalog import CAT, CATS, FX_USD_PER_UNIT, INDIRECT, MATERIALS, PATTERNS, PRODUCTS, SERVICE_ONLY, SUBASSEMBLIES, SUPPLIERS, Supplier

SEED = 20261114
MONTHS = [(2025 + m // 12, m % 12 + 1) for m in range(18)]  # 2025-01 .. 2026-06
N_QUARTERS = len(MONTHS) // 3

N_DUPLICATES = 8
N_OTHER_FAMILY = 40
N_SPARE_LINES, N_WARRANTY_LINES, N_SERVICE_ONLY_LINES, N_OTHER_FAMILY_LINES, N_INDIRECT_LINES = 320, 160, 200, 220, 250
XREF_COVERAGE = 0.40       # share of supplier part numbers that have a crosswalk row
BLANK_PO_TYPE = 0.15
PLANTED_PREMIUM_PARTS = 15


@dataclass
class Part:
    part_number: str
    cat: str
    noun: str
    qual: str
    attrs: tuple
    revision: str
    base_price: float
    commodity: str
    pack: bool = False
    fail: bool = False
    description: str = ""
    dup_of: str | None = None


@dataclass
class Source:
    supplier: Supplier
    share: float
    price: float  # USD per base unit


@dataclass
class Backbone:
    parts: dict = field(default_factory=dict)        # part_number -> Part, parts on a BOM
    bom: list = field(default_factory=list)          # rows of boms.csv
    builds: list = field(default_factory=list)       # rows of builds.csv
    sourcing: dict = field(default_factory=dict)     # part_number -> [Source], BOM and off-BOM parts
    other_family: list = field(default_factory=list)  # parts of a product line whose BOMs were not supplied
    service_only: list = field(default_factory=list)  # spares that are on no production BOM
    duplicates: list = field(default_factory=list)   # (part_number, duplicate part_number)
    findings: list = field(default_factory=list)     # planted findings, for evaluating leads


# ------------------------------------------------------------------ backbone

def build_backbone(rng: random.Random) -> Backbone:
    bb = Backbone()
    used, counters = set(), {}

    def next_pn(prefix: str) -> str:
        counters[prefix] = counters.get(prefix, rng.randint(10000, 80000)) + rng.randint(1, 9)
        return f"{prefix}-{counters[prefix]}"

    def new_part(cat, voltage=None) -> Part:
        for _ in range(500):
            qual = rng.choice(cat.quals)
            attrs = tuple((k, voltage if k == "V" and cat.share == "volt" and voltage else rng.choice(vals)) for k, vals in cat.attrs)
            if (cat.key, qual, attrs) not in used:
                break
        else:
            raise RuntimeError(f"attribute grid exhausted for {cat.key}")
        used.add((cat.key, qual, attrs))
        lo, hi = cat.price
        part = Part(next_pn(cat.prefix), cat.key, cat.noun, qual, attrs, rng.choice("AABBCD"),
                    round(math.exp(rng.uniform(math.log(lo), math.log(hi))), 4), cat.commodity, cat.pack, cat.fail)
        part.description = noise.canonical(part)
        return part

    def source(part: Part, n=None):
        pool = [s for s in SUPPLIERS if s.commodity == part.commodity]
        n = min(len(pool), n or rng.choices([1, 2, 3], weights=[65, 27, 8])[0])
        chosen = rng.sample(pool, n)
        first = rng.uniform(0.55, 0.8)
        shares = {1: [1.0], 2: [first, 1 - first], 3: [0.5, 0.3, 0.2]}[n]
        out = []
        for i, (sup, share) in enumerate(zip(chosen, shares)):
            factor = 1.0
            if i > 0:
                planted = rng.random() < 0.45
                factor = rng.uniform(1.08, 1.30) if planted else rng.uniform(0.98, 1.03)
                if planted:
                    bb.findings.append(dict(finding_type="supplier_price_gap", part_number=part.part_number, related=sup.name,
                                            detail=f"{factor:.3f}x the price of {chosen[0].name}"))
            out.append(Source(sup, share, part.base_price * factor))
        bb.sourcing[part.part_number] = out

    # BOM positions: each category fills some positions per assembly; products share parts per a pattern.
    for cat in CATS:
        for assembly, count in cat.slots:
            for _ in range(count):
                prods = cat.only or tuple(range(len(PRODUCTS)))
                pattern = rng.choice(PATTERNS[cat.share]) if not cat.only else tuple(range(len(prods)))
                qty, variants = rng.choice(cat.qty), {}
                for pi, vi in zip(prods, pattern):
                    prod = PRODUCTS[pi]
                    if vi not in variants:
                        variants[vi] = new_part(cat, prod.voltage)
                        bb.parts[variants[vi].part_number] = variants[vi]
                    p = variants[vi]
                    q = (3 if prod.voltage == 12 else 5) if cat.key == "cell" else qty
                    bb.bom.append(dict(product_id=prod.product_id, assembly_id=assembly, part_number=p.part_number,
                                       revision=p.revision, description=p.description, qty_per_unit=q, uom="EA",
                                       std_cost=p.base_price))  # engineering's standard cost: set once, not updated as prices move
    for part in list(bb.parts.values()):
        source(part)

    # Planted: the same physical part set up under a second internal number for one product.
    users = defaultdict(list)
    for row in bb.bom:
        users[row["part_number"]].append(row["product_id"])
    candidates = [pn for pn, prods in users.items() if len(prods) >= 2 and CAT[bb.parts[pn].cat].share in ("common", "mixed")]
    for pn in rng.sample(candidates, N_DUPLICATES):
        orig = bb.parts[pn]
        clone = replace(orig, part_number=next_pn(CAT[orig.cat].prefix), revision="A", dup_of=pn)
        clone.description = noise.alt_canonical(clone)
        bb.parts[clone.part_number] = clone
        for row in bb.bom:
            if row["part_number"] == pn and row["product_id"] == users[pn][-1]:
                row.update(part_number=clone.part_number, revision=clone.revision, description=clone.description)
        taken = {s.supplier.sid for s in bb.sourcing[pn]}
        pool = [s for s in SUPPLIERS if s.commodity == orig.commodity and s.sid not in taken] or [bb.sourcing[pn][0].supplier]
        factor = rng.uniform(1.06, 1.22)
        bb.sourcing[clone.part_number] = [Source(rng.choice(pool), 1.0, orig.base_price * factor)]
        bb.duplicates.append((pn, clone.part_number))
        bb.findings.append(dict(finding_type="duplicate_part", part_number=pn, related=clone.part_number,
                                detail=f"same physical part; duplicate bought at {factor:.3f}x"))

    # Build volumes, monthly, with seasonality and one mid-period launch.
    for prod in PRODUCTS:
        for m, (y, mo) in enumerate(MONTHS):
            units = 0 if m < prod.launch_month else int(prod.base_units * (1 + 0.2 * math.sin(2 * math.pi * (mo - 9) / 12)) * rng.uniform(0.9, 1.1))
            bb.builds.append(dict(product_id=prod.product_id, period=f"{y}-{mo:02d}", units_built=units))

    # Hard negatives: parts of another product line (same categories, near-neighbor part numbers, no BOM supplied).
    for _ in range(N_OTHER_FAMILY):
        part = new_part(rng.choice([c for c in CATS if not c.only]), voltage=rng.choice([36, 40]))
        bb.other_family.append(part)
        source(part, n=1)
    # Service-only items: legitimately absent from every production BOM.
    for noun, qual, attrs, lo, hi in SERVICE_ONLY:
        part = Part(next_pn("SVC"), "service", noun, qual, attrs, "A", round(rng.uniform(lo, hi), 2), "Service")
        part.description = noise.canonical(part)
        bb.service_only.append(part)
        source(part, n=1)
    return bb


# ------------------------------------------------------------ purchase lines

BUCKET = {"production": "production", "spare": "spares", "service_only": "spares", "warranty": "warranty",
          "other_family": "unassigned", "indirect": "unassigned"}
PO_TYPE = {"production": "production", "spare": "service", "service_only": "service", "warranty": "warranty",
           "other_family": "production", "indirect": ""}
PLANT_PREFIX = {"P01": "GV", "P02": "MX", "SDC": "SV"}


def make_purchases(bb: Backbone, rng: random.Random):
    """Noise layer. Returns (purchases rows, ground-truth rows, supplier crosswalk rows)."""
    supplier_pns, plant_codes, item_master = {}, {}, {}

    def supplier_pn(sup: Supplier, pn: str) -> str:
        if (sup.sid, pn) not in supplier_pns:
            tag = "".join(w[0] for w in sup.name.split()[:2]).upper()
            scheme = int(sup.sid[1:]) % 3
            supplier_pns[sup.sid, pn] = [f"{tag}-{rng.randint(100000, 999999)}",
                                         f"{rng.randint(10000, 99999)}{rng.choice('ABKTX')}{rng.randint(100, 999)}",
                                         f"{tag}{rng.randint(1000, 9999)}-{rng.randint(10, 99)}"][scheme]
        return supplier_pns[sup.sid, pn]

    def plant_code(plant: str, pn: str) -> str:
        if (plant, pn) not in plant_codes:
            plant_codes[plant, pn] = f"{PLANT_PREFIX[plant]}-{rng.randint(1000000, 9999999):07d}"
        return plant_codes[plant, pn]

    # Crosswalk first, so supplier part numbers are fixed before any line is rendered.
    xref = []
    for pn, sources in bb.sourcing.items():
        for src in sources:
            spn = supplier_pn(src.supplier, pn)
            if rng.random() < XREF_COVERAGE:
                xref.append(dict(supplier=src.supplier.name, supplier_part_number=spn, part_number=pn))

    quarter_units = defaultdict(lambda: [0] * N_QUARTERS)  # product -> units built per quarter
    for row in bb.builds:
        y, mo = map(int, row["period"].split("-"))
        quarter_units[row["product_id"]][((y - 2025) * 12 + mo - 1) // 3] += row["units_built"]
    plant_of = {p.product_id: p.plant for p in PRODUCTS}
    demand = defaultdict(lambda: defaultdict(lambda: [0] * N_QUARTERS))  # part -> plant -> base units per quarter
    for row in bb.bom:
        for q in range(N_QUARTERS):
            demand[row["part_number"]][plant_of[row["product_id"]]][q] += row["qty_per_unit"] * quarter_units[row["product_id"]][q]

    def day_in(q: int) -> date:
        return date(2025 + q // 4, 3 * (q % 4) + 1, 1) + timedelta(days=rng.randint(0, 88))

    lines = []  # (kind, part or indirect text, plant, supplier, date, base qty, USD price per base unit)

    for pn, part in bb.parts.items():
        drift = rng.uniform(-0.01, 0.02)  # price change per quarter
        for plant, per_q in demand[pn].items():
            for q, need in enumerate(per_q):
                for src in bb.sourcing[pn]:
                    total = need * src.share * rng.uniform(0.96, 1.08)  # bought is never exactly what was built
                    for piece in ([1.0] if rng.random() < 0.85 else [0.6, 0.4]):
                        if total * piece >= 1:
                            price = src.price * (1 + drift * q) * rng.uniform(0.995, 1.005)
                            lines.append(("production", part, plant, src.supplier, day_in(q), total * piece, price))

    def total_need(pn: str, q: int) -> int:
        return sum(per_q[q] for per_q in demand[pn].values())

    bom_parts = list(bb.parts.values())
    weights = [8 if p.fail else 1 for p in bom_parts]
    premium_parts = {p.part_number: rng.uniform(1.5, 1.9) for p in rng.sample([p for p in bom_parts if p.fail], PLANTED_PREMIUM_PARTS)}
    for pn, factor in premium_parts.items():
        bb.findings.append(dict(finding_type="spares_premium", part_number=pn, related="", detail=f"spares bought at about {factor:.2f}x production price"))

    def service_plant() -> str:
        return "SDC" if rng.random() < 0.87 else rng.choice(["P01", "P02"])

    for _ in range(N_SPARE_LINES):
        part = rng.choices(bom_parts, weights=weights)[0]
        q = rng.randrange(N_QUARTERS)
        qty = max(5, total_need(part.part_number, q) * (rng.uniform(0.02, 0.07) if part.fail else rng.uniform(0.005, 0.03)))
        src = bb.sourcing[part.part_number][0]
        factor = premium_parts.get(part.part_number, 1.0) * rng.uniform(1.05, 1.25)
        lines.append(("spare", part, service_plant(), src.supplier, day_in(q), qty, src.price * factor))
    for _ in range(N_WARRANTY_LINES):
        part = rng.choice([p for p in bom_parts if p.fail])
        q = rng.randrange(N_QUARTERS)
        qty = max(3, total_need(part.part_number, q) * rng.uniform(0.005, 0.03))
        src = bb.sourcing[part.part_number][0]
        lines.append(("warranty", part, service_plant(), src.supplier, day_in(q), qty, src.price * rng.uniform(1.0, 1.2)))
    for _ in range(N_SERVICE_ONLY_LINES):
        part = rng.choice(bb.service_only)
        src = bb.sourcing[part.part_number][0]
        lines.append(("service_only", part, service_plant(), src.supplier, day_in(rng.randrange(N_QUARTERS)), rng.randint(20, 400),
                      src.price * rng.uniform(0.97, 1.03)))
    for _ in range(N_OTHER_FAMILY_LINES):
        part = rng.choice(bb.other_family)
        src = bb.sourcing[part.part_number][0]
        lines.append(("other_family", part, rng.choice(["P01", "P02"]), src.supplier, day_in(rng.randrange(N_QUARTERS)),
                      rng.randint(300, 8000), src.price * rng.uniform(0.97, 1.03)))
    mro = [s for s in SUPPLIERS if s.commodity == "MRO"]
    for _ in range(N_INDIRECT_LINES):
        text, lo, hi = rng.choice(INDIRECT)
        lines.append(("indirect", text, rng.choice(["P01", "P02", "SDC"]), rng.choice(mro), day_in(rng.randrange(N_QUARTERS)),
                      rng.randint(1, 40), rng.uniform(lo, hi)))

    def item_text(part: Part, plant: str, sup: Supplier):
        """ID, description and unit as keyed for this plant + supplier + part. Mostly stable, like a real item master."""
        key = (plant, sup.sid, part.part_number)
        if key in item_master and rng.random() < 0.8:
            return item_master[key]
        id_style, desc_style = noise.pick_styles(rng, plant)
        text = (id_style, desc_style,
                noise.render_id(id_style, part.part_number, part.revision, supplier_pn(sup, part.part_number), plant_code(plant, part.part_number), rng),
                noise.render_desc(part, desc_style, rng), *noise.pick_uom(rng, part.pack))
        return item_master.setdefault(key, text)

    lines.sort(key=lambda l: l[4])
    purchases, truth = [], []
    for n, (kind, item, plant, sup, day, base_qty, price) in enumerate(lines, start=1):
        if kind == "indirect":
            id_style, desc_style = ("blank", "exact") if rng.random() < 0.6 else ("supplier", "upper_reorder")
            id_text = "" if id_style == "blank" else f"{rng.randint(100000, 999999)}"
            desc, uom, units = (item if desc_style == "exact" else item.upper()), "EA", 1
        else:
            id_style, desc_style, id_text, desc, uom, units = item_text(item, plant, sup)
        po_id = f"45{n:08d}"
        purchases.append(dict(
            po_id=po_id, po_date=day.isoformat(), plant=plant, supplier=noise.supplier_name_variant(sup.name, rng),
            supplier_part_number=id_text, item_description=desc, qty=max(1, math.ceil(base_qty / units)), uom=uom,
            unit_price=round(price * units / FX_USD_PER_UNIT[sup.currency], 4), currency=sup.currency,
            po_type="" if rng.random() < BLANK_PO_TYPE else PO_TYPE[kind]))
        truth.append(dict(
            po_id=po_id, true_part_number=item.part_number if kind in ("production", "spare", "warranty") else "",
            true_bucket=BUCKET[kind], line_kind=kind, true_supplier=sup.name, units_per_uom=units,
            id_style=id_style, desc_style=desc_style, noise_style=f"{id_style}+{desc_style}", split=noise.split_of(id_style, desc_style)))
    return purchases, truth, xref


# ------------------------------------------------- structure and materials

def add_structure(bb: Backbone, purchases: list, truth: list, seed: int):
    """Multi-level BOMs, scrap, and continuous materials, layered on top of the flat discrete-part data.

    Uses its own random stream and only appends, so every purchase line generated above keeps its content.
    - Gearbox and motor become in-house ("make") subassemblies; their parts hang under them via parent_part.
    - Four materials are measured in area, length or mass, with scrap, and bought in assorted units.
      Some wire and steel lines are bought by weight: a different dimension that cannot be converted without a factor.
    - Two materials are bought well above their gross requirement (planted, for the purchased-vs-required lead).
    """
    rng = random.Random(seed + 1)
    for row in bb.bom:
        row.update(scrap_pct=0.0, parent_part="", make_buy="buy")

    sub_pn, sub_scrap, counter = {}, {}, {}
    for prod in PRODUCTS:
        for assembly, (prefix, name, scrap) in SUBASSEMBLIES.items():
            counter[prefix] = counter.get(prefix, rng.randint(1000, 8000)) + rng.randint(1, 9)
            pn = f"{prefix}-{counter[prefix]}"
            sub_pn[prod.product_id, assembly], sub_scrap[prod.product_id, assembly] = pn, scrap
            for row in bb.bom:
                if row["product_id"] == prod.product_id and row["assembly_id"] == assembly:
                    row["parent_part"] = pn
            bb.bom.append(dict(product_id=prod.product_id, assembly_id=assembly, part_number=pn, revision="A", description=f"{name}, {prod.product_id}",
                               qty_per_unit=1, uom="EA", std_cost="", scrap_pct=scrap, parent_part="", make_buy="make"))

    quarter_units = defaultdict(lambda: [0] * N_QUARTERS)
    for row in bb.builds:
        y, mo = map(int, row["period"].split("-"))
        quarter_units[row["product_id"]][((y - 2025) * 12 + mo - 1) // 3] += row["units_built"]
    suppliers = [s for s in SUPPLIERS if s.commodity == "Materials"]
    for row, t in zip(purchases, truth):
        t["cross_dimension"] = False
    extra = []
    for m, sup in zip(MATERIALS, suppliers):
        counter[m["prefix"]] = rng.randint(10000, 80000)
        part = Part(f"{m['prefix']}-{counter[m['prefix']]}", "material", m["noun"], m["qual"], m["attrs"], "A", m["price"], "Materials")
        part.description = noise.canonical(part)
        gross_per_unit = {}
        for i, prod in enumerate(PRODUCTS):
            parent = sub_pn.get((prod.product_id, m["assembly"]), "")
            bb.bom.append(dict(product_id=prod.product_id, assembly_id=m["assembly"], part_number=part.part_number, revision="A",
                               description=part.description, qty_per_unit=m["qty"][i], uom=m["uom"], std_cost=m["price"],
                               scrap_pct=m["scrap"], parent_part=parent, make_buy="buy"))
            gross_per_unit[prod.product_id] = m["qty"][i] / (1 - m["scrap"]) / (1 - sub_scrap.get((prod.product_id, m["assembly"]), 0.0))
        if m["over"] > 1.15:
            bb.findings.append(dict(finding_type="over_purchase", part_number=part.part_number, related="",
                                    detail=f"bought about {m['over']:.2f}x the gross requirement"))
        supplier_pn, master = f"{''.join(w[0] for w in sup.name.split()[:2]).upper()}-{rng.randint(100000, 999999)}", {}
        for plant in ("P01", "P02"):
            code = f"{PLANT_PREFIX[plant]}-{rng.randint(1000000, 9999999)}"
            for q in range(N_QUARTERS):
                need = sum(gross_per_unit[p.product_id] * quarter_units[p.product_id][q] for p in PRODUCTS if p.plant == plant)
                bought = need * rng.uniform(0.98, 1.06) * m["over"]
                for piece in ([1.0] if rng.random() < 0.6 else [0.55, 0.45]):
                    if bought * piece <= 0:
                        continue
                    cross = bool(m["cross"]) and rng.random() < m["cross"][2]
                    if cross:
                        uom, per_bom_unit = m["cross"][0], m["cross"][1]
                        qty, price, factor = bought * piece * per_bom_unit, m["price"] / per_bom_unit, ""
                    else:
                        uom, factor = rng.choice(m["units"])
                        qty, price = bought * piece / factor, m["price"] * factor
                    if (plant, cross) not in master:  # one item-master text per plant and buying unit
                        id_style, desc_style = noise.pick_styles(rng, plant)
                        master[plant, cross] = (id_style, desc_style, noise.render_id(id_style, part.part_number, "A", supplier_pn, code, rng),
                                                noise.render_desc(part, desc_style, rng))
                    id_style, desc_style, id_text, desc = master[plant, cross]
                    day = date(2025 + q // 4, 3 * (q % 4) + 1, 1) + timedelta(days=rng.randint(0, 88))
                    extra.append((
                        dict(po_id="", po_date=day.isoformat(), plant=plant, supplier=noise.supplier_name_variant(sup.name, rng),
                             supplier_part_number=id_text, item_description=desc, qty=round(qty, 2), uom=uom,
                             unit_price=round(price * rng.uniform(0.99, 1.01), 4), currency="USD", po_type="" if rng.random() < BLANK_PO_TYPE else "production"),
                        dict(po_id="", true_part_number=part.part_number, true_bucket="production", line_kind="production", true_supplier=sup.name,
                             units_per_uom=factor, id_style=id_style, desc_style=desc_style, noise_style=f"{id_style}+{desc_style}",
                             split=noise.split_of(id_style, desc_style), cross_dimension=cross)))

    merged = sorted(list(zip(purchases, truth)) + extra, key=lambda pair: pair[0]["po_date"])  # stable: existing lines keep their order
    for n, (row, t) in enumerate(merged, start=1):
        row["po_id"] = t["po_id"] = f"45{n:08d}"
    purchases[:], truth[:] = [r for r, _ in merged], [t for _, t in merged]
    return len(extra)


# -------------------------------------------------------------------- output

def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def generate(out: Path, seed: int = SEED) -> dict:
    rng = random.Random(seed)
    bb = build_backbone(rng)
    purchases, truth, xref = make_purchases(bb, rng)
    material_lines = add_structure(bb, purchases, truth, seed)

    write_csv(out / "boms.csv", bb.bom)
    write_csv(out / "builds.csv", bb.builds)
    write_csv(out / "purchases.csv", purchases)
    write_csv(out / "supplier_xref.csv", xref)
    write_csv(out / "fx_rates.csv", [dict(currency=c, usd_per_unit=r) for c, r in FX_USD_PER_UNIT.items()])
    write_csv(out / "truth" / "ground_truth.csv", truth)
    write_csv(out / "truth" / "duplicate_parts.csv", [dict(part_number=a, duplicate_part_number=b) for a, b in bb.duplicates])
    write_csv(out / "truth" / "planted_findings.csv", bb.findings)

    # Self-checks: the files must agree with each other.
    assert len({p["po_id"] for p in purchases}) == len(purchases)
    bom_pns = {r["part_number"] for r in bb.bom if r["make_buy"] == "buy"}
    assert set(bb.parts) <= bom_pns, "every discrete part is on a BOM"
    assert all(t["true_part_number"] in bom_pns for t in truth if t["true_part_number"])

    spend, by_split, by_style = defaultdict(float), defaultdict(int), defaultdict(int)
    for p, t in zip(purchases, truth):
        spend[t["true_bucket"]] += p["qty"] * p["unit_price"] * FX_USD_PER_UNIT[p["currency"]]
        by_split[t["split"]] += 1
        by_style[t["id_style"]] += 1
        by_style[t["desc_style"] + " (desc)"] += 1
    unit_cost = defaultdict(float)
    for r in bb.bom:
        if r["make_buy"] == "buy":  # ignores scrap; the flattened, scrap-adjusted cost is computed in allocate.py
            unit_cost[r["product_id"]] += r["qty_per_unit"] * r["std_cost"]
    return dict(material_lines=material_lines, parts=len(bom_pns), bom_lines=len(bb.bom), purchases=len(purchases), xref=len(xref), spend=dict(spend),
                split=dict(by_split), styles=dict(by_style), unit_cost=dict(unit_cost), findings=len(bb.findings))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=Path("data"))
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    s = generate(args.out, args.seed)

    total = sum(s["spend"].values())
    print(f"unique BOM parts {s['parts']} | BOM lines {s['bom_lines']} | purchase lines {s['purchases']} | crosswalk rows {s['xref']} | planted findings {s['findings']}")
    print("standard parts cost per unit: " + ", ".join(f"{k} ${v:.2f}" for k, v in s["unit_cost"].items()))
    print(f"total spend ${total:,.0f}")
    for bucket in ("production", "spares", "warranty", "unassigned"):
        print(f"  {bucket:<11} ${s['spend'][bucket]:>12,.0f}  {s['spend'][bucket] / total:6.1%}")
    print(f"true tie-out rate {1 - s['spend']['unassigned'] / total:.1%}")
    n = s["purchases"]
    print("split: " + ", ".join(f"{k} {v / n:.0%}" for k, v in sorted(s["split"].items())))
    print("styles: " + ", ".join(f"{k} {v / n:.0%}" for k, v in sorted(s["styles"].items())))
    print(f"wrote {args.out}/")


if __name__ == "__main__":
    main()
