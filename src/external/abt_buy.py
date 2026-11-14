"""External test A: Abt-Buy product matching, fixed test split from the Mannheim benchmark page.

    python3 -m src.external.fetch abt_buy
    python3 -m src.external.abt_buy           # writes out/eval_external_abt_buy.json

Zero-shot: nothing is trained on this data. The task is pair classification (is this Abt record the
same product as this Buy record?), so the pipeline is applied pair by pair:
  rules       the Abt model code appears in the Buy record's name
  embeddings  cosine similarity of the two records' text; pairs below a floor are non-matches
  Claude      judges the pairs between the floor and a rule match
The benchmark's validation split is used only to check the rule and set the two similarity cutoffs.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np
from pydantic import BaseModel

from .. import adjudicate, match

DATA = Path("data/external/abt_buy")
LLM_CACHE = Path("cache/llm_external")  # not committed: answers quote the downloaded records
PUBLISHED = {"SVM (benchmark page)": (0.96, 0.71, 0.81), "Random Forest (benchmark page)": (0.95, 0.77, 0.85)}

SYSTEM = """You decide whether two product listings from two different online shops are the same product.

Same product means the same model: same brand, same model number or name, same variant (size, capacity, color where it is part of the model). Accessories, bundles, different sizes or generations of a product line are not the same product. Listings are terse and written differently by each shop; a model number in one may be missing or formatted differently in the other.

Answer with match true or false, confidence as the probability (0 to 1) that your answer is right, and reason as one sentence naming the evidence."""


class PairVerdict(BaseModel):
    match: bool
    confidence: float
    reason: str


def _records(name: str) -> dict:
    with (DATA / name).open(encoding="utf-8", errors="replace") as f:
        return {r["subject_id"]: r for r in csv.DictReader(f)}


def _pairs(name: str) -> list[tuple[str, str, bool]]:
    with (DATA / name).open() as f:
        return [(r["source_id"], r["target_id"], r["matching"] == "True") for r in csv.DictReader(f)]


def _alnum(s: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", s.upper())


def rule(abt: dict, buy: dict) -> bool:
    """The Abt name usually ends in ' - MODELCODE'. A match needs that code inside the Buy name."""
    code = _alnum(abt["name"].rsplit(" - ", 1)[-1]) if " - " in abt["name"] else ""
    return len(code) >= 5 and any(c.isdigit() for c in code) and code in _alnum(buy["name"])


def prf(pred: np.ndarray, true: np.ndarray) -> dict:
    tp = int((pred & true).sum())
    p, r = tp / max(int(pred.sum()), 1), tp / max(int(true.sum()), 1)
    return dict(precision=p, recall=r, f1=2 * p * r / max(p + r, 1e-9), predicted=int(pred.sum()), positives=int(true.sum()))


def features(pairs, abt, buy):
    a_text = [f"{abt[a]['name']}. {abt[a]['description']}" for a, _, _ in pairs]
    b_text = [f"{buy[b]['name']}. {buy[b]['description']} {buy[b]['manufacturer']}" for _, b, _ in pairs]
    sim = (match.embed(a_text, "openai") * match.embed(b_text, "openai")).sum(axis=1)
    ruled = np.array([rule(abt[a], buy[b]) for a, b, _ in pairs])
    return sim, ruled, np.array([t for _, _, t in pairs])


def main():
    abt, buy = _records("1_abt.csv"), _records("2_buy.csv")
    val, test = _pairs("gs_val.csv"), _pairs("gs_test.csv")

    # Validation split: check the rule, set the cutoffs. No model is fit.
    v_sim, v_rule, v_true = features(val, abt, buy)
    rule_val = prf(v_rule, v_true)
    grid = np.arange(0.20, 0.95, 0.01)
    cutoff = float(max(grid, key=lambda t: prf(v_rule | (v_sim >= t), v_true)["f1"]))             # best F1 without Claude
    floor = float(max(t for t in grid if ((v_sim >= t) | v_rule)[v_true].mean() >= 0.99))          # keeps 99% of true matches

    sim, ruled, true = features(test, abt, buy)
    band = ~ruled & (sim >= floor)
    prompts = {i: f"Listing A (shop 1)\n- name: {abt[a]['name']}\n- description: {abt[a]['description']}\n- price: {abt[a]['price'] or 'not given'}\n\n"
                  f"Listing B (shop 2)\n- name: {buy[b]['name']}\n- description: {buy[b]['description']}\n- manufacturer: {buy[b]['manufacturer']}\n"
                  f"- price: {buy[b]['price'] or 'not given'}"
               for i, (a, b, _) in enumerate(test) if band[i]}
    answers, usage = adjudicate.ask_many(list(prompts.values()), system=SYSTEM, schema=PairVerdict, cache=LLM_CACHE)
    claude = np.zeros(len(test), bool)
    for i, q in prompts.items():
        claude[i] = bool(answers[q] and answers[q]["match"])

    report = dict(
        dataset="Abt-Buy, Mannheim Benchmark Matching Tasks, fixed test split", test_pairs=len(test), test_matches=int(true.sum()),
        validation_use=dict(rule=rule_val, embedding_cutoff=cutoff, claude_floor=floor),
        results={"Rules only": prf(ruled, true), "Rules + embeddings": prf(ruled | (sim >= cutoff), true),
                 "Rules + embeddings + Claude": prf(ruled | claude, true)},
        published={k: dict(precision=p, recall=r, f1=f) for k, (p, r, f) in PUBLISHED.items()},
        claude=dict(pairs_sent=int(band.sum()), **usage),
    )
    Path("out").mkdir(exist_ok=True)
    Path("out/eval_external_abt_buy.json").write_text(json.dumps(report, indent=2))

    print(f"Abt-Buy test split: {len(test)} pairs, {int(true.sum())} matches")
    print(f"validation only: rule precision {rule_val['precision']:.2f} recall {rule_val['recall']:.2f}; embedding cutoff {cutoff:.2f}; Claude floor {floor:.2f}")
    print(f"{'method':<34}{'precision':>10}{'recall':>8}{'F1':>7}")
    for name, m in {**report["results"], **report["published"]}.items():
        print(f"{name:<34}{m['precision']:>10.2f}{m['recall']:>8.2f}{m['f1']:>7.2f}")
    print(f"Claude: {int(band.sum())} pairs sent, {usage['calls']} calls, ${usage['cost_usd']:.2f} ({usage['failed_calls']} failed)")


if __name__ == "__main__":
    main()
