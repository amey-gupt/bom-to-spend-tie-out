"""One line from Claude on what differs between the two parts of each "similar parts" lead.

    python3 -m src.lead_notes       # top 20 pairs by range; writes out/similar_part_notes.json

The app only reads the saved file. Answers are cached in cache/llm_leads/ (committed), so a rerun makes no calls.
Reads out/leads.csv and out/bom.csv only.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from pydantic import BaseModel

from . import adjudicate

CACHE = Path("cache/llm_leads")
TOP = 20
SYSTEM = """You are helping an engineer review two parts on a manufacturer's bills of materials that look alike.

In one sentence of at most 30 words, say what physically differs between them and what an engineer would have to check before one could replace the other. Do not say whether they are interchangeable: that is the engineer's decision. Plain words, no preamble."""


class Note(BaseModel):
    difference: str


def pair_key(a: str, b: str) -> str:
    return "|".join(sorted([a, b]))


if __name__ == "__main__":
    leads = pd.read_csv("out/leads.csv", keep_default_na=False)
    desc = pd.read_csv("out/bom.csv").drop_duplicates("part_number").set_index("part_number")["description"]
    pairs = leads[leads["lead_type"] == "similar_parts"].sort_values("high_usd", ascending=False).head(TOP)
    prompts = {pair_key(r.part_number, r.related): f"Part A: {r.part_number}: {desc[r.part_number]}\nPart B: {r.related}: {desc[r.related]}" for r in pairs.itertuples()}
    answers, usage = adjudicate.ask_many(list(prompts.values()), system=SYSTEM, schema=Note, cache=CACHE)
    labels = {pair_key(r.part_number, r.related): (r.part_number, r.related) for r in pairs.itertuples()}  # which part the note calls A and B
    notes = {key: dict(a=labels[key][0], b=labels[key][1], note=answers[q]["difference"]) for key, q in prompts.items() if answers[q]}
    Path("out/similar_part_notes.json").write_text(json.dumps(notes, indent=2))
    print(f"{len(notes)} notes, {usage['new_calls']} new Claude calls, ${usage['cost_usd']:.2f}")
    for key, note in list(notes.items())[:3]:
        print(" ", key, "->", note["note"])
