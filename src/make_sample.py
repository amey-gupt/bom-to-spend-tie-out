"""Write the second demo sample: the same kind of data as a different company would export it.

    python3 -m src.make_sample      # writes samples/second_company/

A slice of the synthetic data (two products, one plant plus the service center, first quarter of 2025)
with different column names, a US date format, a different order-type vocabulary, and the purchase
lines as an Excel file. It exists to demo the upload and column-mapping flow. Synthetic, like the rest.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

OUT = Path("samples/second_company")
PRODUCTS = ["DR-12C", "DR-18"]

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    read = lambda name: pd.read_csv(f"data/{name}", dtype=str, keep_default_na=False)

    bom = read("boms.csv")
    bom = bom[bom["product_id"].isin(PRODUCTS)]
    bom.rename(columns={"product_id": "Model", "assembly_id": "Sub-Assy", "part_number": "Item No.", "revision": "Rev", "description": "Item Text",
                        "qty_per_unit": "Qty/Unit", "uom": "UoM", "std_cost": "Std Cost (USD)"}).to_csv(OUT / "bom_export.csv", index=False)

    po = read("purchases.csv")
    po = po[(po["po_date"] < "2025-04-01") & po["plant"].isin(["P01", "SDC"])].copy()
    po["po_date"] = pd.to_datetime(po["po_date"]).dt.strftime("%m/%d/%Y")
    po["po_type"] = po["po_type"].map({"production": "PROD", "service": "SERV", "warranty": "WARR", "": ""})
    po["qty"], po["unit_price"] = po["qty"].astype(int), po["unit_price"].astype(float)
    po.rename(columns={"po_id": "PO Number", "po_date": "Order Date", "plant": "Site", "supplier": "Vendor Name", "supplier_part_number": "Vendor Item",
                       "item_description": "Short Text", "qty": "Order Qty", "uom": "Order Unit", "unit_price": "Net Price", "currency": "Curr.",
                       "po_type": "Order Type"}).to_excel(OUT / "po_lines.xlsx", index=False)

    builds = read("builds.csv")
    builds = builds[builds["product_id"].isin(PRODUCTS) & (builds["period"] < "2025-04")].copy()
    builds["period"] = pd.to_datetime(builds["period"]).dt.strftime("%b %Y")
    builds.rename(columns={"product_id": "Model", "period": "Month", "units_built": "Units Completed"}).to_csv(OUT / "production_output.csv", index=False)
    print(f"wrote {OUT}/: {len(bom)} BOM rows, {len(po)} purchase lines, {len(builds)} build rows")
