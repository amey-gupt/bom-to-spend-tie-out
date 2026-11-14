"""The reviewer crosswalk: every decision a person makes, kept so it never has to be made twice.

crosswalk.csv maps purchase text (supplier part number + item description) to a BOM part, or to
"not a BOM part". It is applied as the first rule on the next run and to every line with the same text.
The file is local working state and is not committed.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from .normalize import norm_desc, norm_pn

PATH = Path("crosswalk.csv")
COLUMNS = ["supplier_part_number", "item_description", "part_number", "decision", "decided_at", "dataset"]


def load(path: Path = PATH) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def add(supplier_part_number: str, item_description: str, part_number: str, decision: str, dataset: str, path: Path = PATH):
    """decision: approve (ties the text to part_number), no_match (confirmed not a BOM part), reject (suggestion was wrong)."""
    row = dict(supplier_part_number=supplier_part_number, item_description=item_description,
               part_number=part_number if decision == "approve" else "", decision=decision,
               decided_at=datetime.now().isoformat(timespec="seconds"), dataset=dataset)
    pd.concat([load(path), pd.DataFrame([row])], ignore_index=True).to_csv(path, index=False)


def undo_last(path: Path = PATH):
    load(path).iloc[:-1].to_csv(path, index=False)


def clear(path: Path = PATH):
    path.unlink(missing_ok=True)


def as_rules(df: pd.DataFrame) -> dict:
    """{normalized part-number field | normalized description: (decision, part)}; a later decision replaces an earlier one."""
    return {f"{norm_pn(r.supplier_part_number)}|{norm_desc(r.item_description)}": (r.decision, r.part_number) for r in df.itertuples()}
