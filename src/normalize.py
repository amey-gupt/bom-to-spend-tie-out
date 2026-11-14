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


def units_per_uom(raw: str) -> int:
    """Base units (each) in one purchase unit. 'BX100' -> 100, 'C' -> 100, 'M' -> 1000, 'PCS' -> 1."""
    s = str(raw).upper().strip()
    if s == "C":
        return 100
    if s == "M":
        return 1000
    digits = re.search(r"\d+", s)
    return int(digits.group()) if digits else 1


@dataclass
class Inputs:
    purchases: pd.DataFrame  # one row per purchase line, with normalized columns added
    boms: pd.DataFrame
    parts: pd.DataFrame      # one row per unique BOM part number
    builds: pd.DataFrame
    xref: pd.DataFrame       # supplier part number -> internal part number, where the company has one


def load_inputs(data_dir: str | Path = "data") -> Inputs:
    d = Path(data_dir)
    read = lambda name: pd.read_csv(d / name, dtype=str, keep_default_na=False)

    fx = read("fx_rates.csv").set_index("currency")["usd_per_unit"].astype(float)
    p = read("purchases.csv")
    p["qty"] = p["qty"].astype(float)
    p["unit_price"] = p["unit_price"].astype(float)
    p["pn_key"] = p["supplier_part_number"].map(norm_pn)
    p["supplier_key"] = p["supplier"].map(norm_supplier)
    p["supplier_key"] = p["supplier_key"].map(canonical_suppliers(p["supplier_key"]))
    p["desc_norm"] = p["item_description"].map(norm_desc)
    p["units_per_uom"] = p["uom"].map(units_per_uom)
    p["spend_usd"] = p["qty"] * p["unit_price"] * p["currency"].map(fx)
    p["base_qty"] = p["qty"] * p["units_per_uom"]
    p["unit_price_usd"] = p["spend_usd"] / p["base_qty"]  # per base unit, comparable across suppliers

    boms = read("boms.csv")
    boms["qty_per_unit"] = boms["qty_per_unit"].astype(float)
    parts = boms.drop_duplicates("part_number")[["part_number", "revision", "description"]].reset_index(drop=True)
    parts["pn_key"] = parts["part_number"].map(norm_pn)
    parts["desc_norm"] = parts["description"].map(norm_desc)

    builds = read("builds.csv")
    builds["units_built"] = builds["units_built"].astype(int)

    xref = read("supplier_xref.csv")
    xref["spn_key"] = xref["supplier_part_number"].map(norm_pn)
    xref["supplier_key"] = xref["supplier"].map(norm_supplier)
    return Inputs(p, boms, parts, builds, xref)


if __name__ == "__main__":
    inp = load_inputs()
    cols = ["supplier_part_number", "pn_key", "item_description", "desc_norm", "uom", "units_per_uom", "unit_price_usd"]
    print(inp.purchases[cols].sample(12, random_state=0).to_string(index=False))
    print(f"\n{len(inp.purchases)} purchase lines, {len(inp.parts)} BOM parts, {len(inp.xref)} crosswalk rows, "
          f"${inp.purchases['spend_usd'].sum():,.0f} total spend")
