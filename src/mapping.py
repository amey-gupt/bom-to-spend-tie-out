"""Map a company's own column names onto ours, with Claude proposing and a person confirming.

Claude sees the headers and the first 20 rows of each file and returns, for every canonical column,
the source column that holds it (or none) with a confidence. Nothing is run until a person confirms.
"""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
from pydantic import BaseModel

from . import adjudicate

CACHE = Path("cache/llm_uploads")  # not committed: holds text from uploaded files

# kind -> [(canonical column, required, meaning)]
CANONICAL = {
    "boms": [
        ("product_id", True, "the finished product or model the row belongs to"),
        ("part_number", True, "the company's internal part or item number for the component"),
        ("description", True, "engineering description of the component"),
        ("qty_per_unit", True, "how many of the component go into one unit of the product"),
        ("assembly_id", False, "sub-assembly or BOM section the component sits in"),
        ("revision", False, "engineering revision of the part"),
        ("uom", False, "unit of measure of the BOM quantity"),
        ("std_cost", False, "standard cost of one component, in USD"),
    ],
    "purchases": [
        ("po_date", True, "purchase order or posting date"),
        ("supplier", True, "supplier or vendor name"),
        ("item_description", True, "free-text description of what was bought"),
        ("qty", True, "quantity ordered, in the order unit"),
        ("unit_price", True, "price per order unit"),
        ("supplier_part_number", False, "the part or item number keyed on the PO line: internal, supplier's, or a plant code"),
        ("po_id", False, "purchase order number or line identifier"),
        ("plant", False, "plant, site or location that bought it"),
        ("uom", False, "order unit of measure (each, box of 100, ...)"),
        ("currency", False, "currency of the unit price"),
        ("po_type", False, "order type that tells production from service/spares or warranty buys"),
    ],
    "builds": [
        ("product_id", True, "the finished product or model"),
        ("period", True, "month (or date) of production"),
        ("units_built", True, "units produced in the period"),
    ],
}
KIND_NAME = {"boms": "bill of materials", "purchases": "purchase order lines", "builds": "production (build) volumes"}

SYSTEM = """You map the columns of a manufacturer's data export onto a fixed target schema.

You get the kind of file, the target columns with their meanings, the file's headers, and its first rows. For every target column, name the one source column that holds that information, exactly as its header is written, or an empty string if the file has no such column. Use the values in the sample rows, not just the header names. Never use one source column for two target columns. confidence is the probability (0 to 1) that your choice for that target column is right. note is a few words on why, or what to check."""


class ColumnChoice(BaseModel):
    canonical: str
    source: str
    confidence: float
    note: str


class FileMapping(BaseModel):
    columns: list[ColumnChoice]


def read_table(name: str, data: bytes) -> pd.DataFrame:
    """CSV or Excel, everything as text."""
    if name.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(io.BytesIO(data), dtype=str)
    else:
        df = pd.read_csv(io.BytesIO(data), dtype=str, keep_default_na=False)
    df.columns = [str(c).strip() for c in df.columns]
    return df.fillna("")


def propose(kind: str, df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Ask Claude for the mapping. Returns (one row per canonical column: source, confidence, note; usage)."""
    target = "\n".join(f"- {c} ({'required' if req else 'optional'}): {meaning}" for c, req, meaning in CANONICAL[kind])
    prompt = (f"File kind: {KIND_NAME[kind]}\n\nTarget columns\n{target}\n\nSource headers\n{list(df.columns)}\n\n"
              f"First rows (CSV)\n{df.head(20).to_csv(index=False)}")
    answers, usage = adjudicate.ask_many([prompt], system=SYSTEM, schema=FileMapping, cache=CACHE)
    got = {c["canonical"]: c for c in (answers[prompt] or {}).get("columns", [])}
    rows = []
    for c, req, meaning in CANONICAL[kind]:
        a = got.get(c, {})
        source = a.get("source", "") if a.get("source", "") in df.columns else ""
        rows.append(dict(canonical=c, required=req, meaning=meaning, source=source,
                         confidence=float(a.get("confidence", 0.0)) if source else 0.0, note=a.get("note", "no answer from Claude" if not a else "")))
    return pd.DataFrame(rows), usage


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace(r"[^\d.\-eE]", "", regex=True), errors="coerce")


def _po_type(v: str) -> str:
    v = str(v).lower()
    if any(k in v for k in ("warr", "wty", "rma")):
        return "warranty"
    if any(k in v for k in ("serv", "svc", "spare", "after")):
        return "service"
    return "production" if "prod" in v or v in ("nb", "standard", "std") else ""


def apply_mapping(kind: str, df: pd.DataFrame, mapping: dict) -> tuple[pd.DataFrame, list[str]]:
    """Rename to the canonical columns, fill optional ones, clean values. Returns (table, notes on what was dropped or defaulted)."""
    notes = []
    out = pd.DataFrame({c: (df[mapping[c]].astype(str).str.strip() if mapping.get(c) else "") for c, _, _ in CANONICAL[kind]}, index=df.index)
    if kind == "boms":
        out["qty_per_unit"] = _num(out["qty_per_unit"])
        out["std_cost"] = _num(out["std_cost"])
        out.loc[out["assembly_id"] == "", "assembly_id"] = "ALL"
        out.loc[out["uom"] == "", "uom"] = "EA"
        keep = (out["part_number"] != "") & out["qty_per_unit"].notna()
        if out["std_cost"].isna().all():
            notes.append("No standard cost column: product cost will show the estimated actual only.")
    elif kind == "purchases":
        out["qty"], out["unit_price"] = _num(out["qty"]), _num(out["unit_price"])
        out["po_date"] = pd.to_datetime(out["po_date"], errors="coerce").dt.strftime("%Y-%m-%d")
        out["po_type"] = out["po_type"].map(_po_type)
        out["currency"] = out["currency"].str.upper().where(out["currency"] != "", "USD")
        out.loc[out["uom"] == "", "uom"] = "EA"
        if (out["po_id"] == "").any() or out["po_id"].duplicated().any():
            out["po_id"] = [f"UP{i:06d}" for i in range(len(out))]
            notes.append("PO identifiers were missing or repeated, so lines were numbered.")
        keep = out["po_date"].notna() & (out["qty"] > 0) & out["unit_price"].notna()
    else:
        out["units_built"] = _num(out["units_built"]).fillna(0).astype(int)
        out["period"] = pd.to_datetime(out["period"], errors="coerce").dt.strftime("%Y-%m")
        keep = out["period"].notna() & (out["product_id"] != "")
    if (~keep).any():
        notes.append(f"{int((~keep).sum())} {KIND_NAME[kind]} rows dropped: a required value was blank or unreadable.")
    return out[keep].reset_index(drop=True), notes
