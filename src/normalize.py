"""Load the CSV exports and put both sides into comparable form.

Reads data/*.csv only. Never reads data/truth/.

    python3 -m src.normalize        # prints a sample of normalized purchase lines
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

_REV = re.compile(r"^(.*\d)(?:REV|R)?[A-Z]$")
_LEGAL = re.compile(r"\b(INC|LLC|CORP|CO|LTD|GMBH|S A DE C V|SA DE CV)\b")
_UNITS = "MM|KW|W|V|VDC|A|AH|MAH|MA|UF|AWG|K|OHM"


def norm_pn(raw: str) -> str:
    """Part-number key: uppercase, no punctuation, no trailing revision letter.
    'mtr-75168b', 'MTR 75168 REV B' and 'MTR.75168/B' all become 'MTR75168'."""
    key = re.sub(r"[^A-Z0-9]", "", str(raw).upper())
    m = _REV.match(key)
    return m.group(1) if m else key


def norm_supplier(raw: str) -> str:
    key = re.sub(r"[^A-Z0-9 ]", " ", str(raw).upper())
    return re.sub(r"\s+", " ", _LEGAL.sub(" ", key)).strip()


def norm_desc(raw: str) -> str:
    """Description text for embedding: uppercase, punctuation out, dimensions and units written one way
    ('M3 x 8' -> 'M3X8', '100 W' -> '100W')."""
    s = re.sub(r"[,;]", " ", str(raw).upper())
    for _ in range(3):
        s = re.sub(r"(\d)\s*X\s*(\d)", r"\1X\2", s)
    s = re.sub(rf"(\d)\s+({_UNITS})\b", r"\1\2", s)
    return re.sub(r"\s+", " ", s).strip()


_NUM = re.compile(r"(\d+(?:\.\d+)?)(KW|MAH)?")


def numbers_in(desc_norm: str) -> frozenset:
    """Numeric attributes in a normalized description, units reconciled (0.1KW -> 100, 2000MAH -> 2)."""
    out = set()
    for value, unit in _NUM.findall(desc_norm):
        v = float(value) * (1000 if unit == "KW" else 0.001 if unit == "MAH" else 1)
        out.add(f"{round(v, 4):g}")
    return frozenset(out)


def canonical_suppliers(keys) -> dict:
    """Map each supplier key to the longest key it is a prefix of, so truncated vendor names fold into the full one."""
    uniq = sorted(set(keys), key=len, reverse=True)
    out = {}
    for k in uniq:
        out[k] = next((full for full in uniq if len(k) >= 8 and full.startswith(k)), k)
    return out


# unit -> (dimension, size in the dimension's base unit: each, gram, metre, square metre)
UNITS = {
    "EA": ("count", 1.0), "EACH": ("count", 1.0), "PC": ("count", 1.0), "PCS": ("count", 1.0), "C": ("count", 100.0),
    "G": ("mass", 1.0), "GM": ("mass", 1.0), "GR": ("mass", 1.0), "KG": ("mass", 1000.0), "LB": ("mass", 453.592), "LBS": ("mass", 453.592),
    "MTR": ("length", 1.0), "FT": ("length", 0.3048), "KM": ("length", 1000.0),
    "M2": ("area", 1.0), "SQM": ("area", 1.0), "FT2": ("area", 0.092903), "SQFT": ("area", 0.092903),
}


def parse_uom(raw: str, bom_side: bool = False) -> tuple[str, float]:
    """(dimension, size in base units). 'BX100' -> (count, 100), 'KG' -> (mass, 1000), 'FT' -> (length, 0.3048).

    'M' is ambiguous: on a purchase line it usually means per thousand, on a BOM it means metres. A purchase line
    is read as per thousand here and re-read as metres after matching, if the matched part is measured in length.
    """
    s = str(raw).upper().strip()
    if s == "M":
        return ("length", 1.0) if bom_side else ("count", 1000.0)
    if s in UNITS:
        return UNITS[s]
    digits = re.search(r"\d+", s)
    return "count", float(digits.group()) if digits else 1.0


def units_per_uom(raw: str) -> float:
    """Base units in one purchase unit. 'BX100' -> 100, 'C' -> 100, 'M' -> 1000, 'PCS' -> 1, 'KG' -> 1000 (grams)."""
    return parse_uom(raw)[1]


@dataclass
class Inputs:
    purchases: pd.DataFrame  # one row per purchase line, with normalized columns added
    boms: pd.DataFrame
    parts: pd.DataFrame      # one row per unique BOM part number
    builds: pd.DataFrame
    xref: pd.DataFrame       # supplier part number -> internal part number, where the company has one


FX_DEFAULT = {"USD": 1.0, "EUR": 1.08}


def prepare(purchases: pd.DataFrame, boms: pd.DataFrame, builds: pd.DataFrame, xref: pd.DataFrame | None = None, fx: dict | None = None,
            source_file: str = "purchases.csv") -> Inputs:
    """Add the normalized columns to tables that already use the canonical column names."""
    fx = fx or FX_DEFAULT
    p = purchases.copy()
    if "source_row" not in p:  # where each line came from, so any number can be traced back: row 1 is the header
        p["source_row"] = range(2, len(p) + 2)
    if "source_file" not in p:
        p["source_file"] = source_file
    p["qty"] = p["qty"].astype(float)
    p["unit_price"] = p["unit_price"].astype(float)
    p["pn_key"] = p["supplier_part_number"].map(norm_pn)
    p["supplier_key"] = p["supplier"].map(norm_supplier)
    p["supplier_key"] = p["supplier_key"].map(canonical_suppliers(p["supplier_key"]))
    p["desc_norm"] = p["item_description"].map(norm_desc)
    p["units_per_uom"] = p["uom"].map(units_per_uom)
    p["spend_usd"] = p["qty"] * p["unit_price"] * p["currency"].map(lambda c: fx.get(c, 1.0))
    p["base_qty"] = p["qty"] * p["units_per_uom"]
    p["unit_price_usd"] = p["spend_usd"] / p["base_qty"]  # per base unit, comparable across suppliers

    boms = boms.copy()
    for col, default in (("scrap_pct", 0.0), ("parent_part", ""), ("make_buy", "buy"), ("effective_from", ""), ("effective_to", "")):  # all optional
        if col not in boms:
            boms[col] = default
    boms["qty_per_unit"] = boms["qty_per_unit"].astype(float)
    boms["std_cost"] = pd.to_numeric(boms["std_cost"], errors="coerce")
    boms["scrap_pct"] = pd.to_numeric(boms["scrap_pct"], errors="coerce").fillna(0.0)
    # only bought parts can appear on a purchase order; in-house subassemblies are never match candidates
    parts = boms[boms["make_buy"] != "make"].drop_duplicates("part_number")[["part_number", "revision", "description"]].reset_index(drop=True)
    parts["pn_key"] = parts["part_number"].map(norm_pn)
    parts["desc_norm"] = parts["description"].map(norm_desc)

    builds = builds.copy()
    builds["units_built"] = builds["units_built"].astype(int)

    xref = pd.DataFrame(columns=["supplier", "supplier_part_number", "part_number"]) if xref is None else xref.copy()
    xref["spn_key"] = xref["supplier_part_number"].map(norm_pn)
    xref["supplier_key"] = xref["supplier"].map(norm_supplier)
    return Inputs(p, boms, parts, builds, xref)


def load_inputs(data_dir: str | Path = "data") -> Inputs:
    d = Path(data_dir)
    read = lambda name: pd.read_csv(d / name, dtype=str, keep_default_na=False)
    fx = read("fx_rates.csv").set_index("currency")["usd_per_unit"].astype(float).to_dict()
    return prepare(read("purchases.csv"), read("boms.csv"), read("builds.csv"), read("supplier_xref.csv"), fx)


if __name__ == "__main__":
    inp = load_inputs()
    cols = ["supplier_part_number", "pn_key", "item_description", "desc_norm", "uom", "units_per_uom", "unit_price_usd"]
    print(inp.purchases[cols].sample(12, random_state=0).to_string(index=False))
    print(f"\n{len(inp.purchases)} purchase lines, {len(inp.parts)} BOM parts, {len(inp.xref)} crosswalk rows, "
          f"${inp.purchases['spend_usd'].sum():,.0f} total spend")
