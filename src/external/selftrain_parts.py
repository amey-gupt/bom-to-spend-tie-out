"""Experiment (b): self-training on real parts. Kitspace BOMs against the JLCPCB catalog.

    python3 -m src.external.selftrain_parts      # writes out/eval_external_selftrain.json

The BOM parts are split in half at random. Training uses only what a company would have without labels:
purchase lines for the first half whose part number is written in the text, which the rule matches. The
matched number is stripped and the part-number features hidden before training. The test is the other
half of the parts with the part number hidden, plus the 500 California no-match lines.
Compared with the classifier transferred from the synthetic data, on the same test lines.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

from .. import adjudicate, match, score as scoring
from . import kitspace_jlc as k

BACKEND = "openai"


def lines_for(catalog: pd.DataFrame, owner: dict, visible: bool) -> pd.DataFrame:
    return pd.DataFrame({
        "supplier": "JLCPCB", "supplier_part_number": "C" + catalog["lcsc"].astype(str),
        "item_description": catalog["description"] + ((" " + catalog["mpn"]) if visible else ""),
        "unit_price_usd": catalog["unit_price"], "true_part": catalog["mpn"].str.upper().map(owner), "lcsc": catalog["lcsc"]})


def metrics(dec: pd.DataFrame, p: pd.DataFrame) -> dict:
    real, accept = (p["true_part"] != "").to_numpy(), (dec["decision"] == "accept").to_numpy()
    right = accept & (dec["part"].to_numpy() == p["true_part"].to_numpy())
    review, none = (dec["decision"] == "review").to_numpy(), (dec["decision"] == "no_match").to_numpy()
    return dict(
        test_lines=int(real.sum()), accepted=int((accept & real).sum()), right=int(right.sum()), wrong=int((accept & real & ~right).sum()),
        precision=float(right.sum() / max((accept & real).sum(), 1)), recall=float(right.sum() / max(real.sum(), 1)),
        in_review=int((review & real).sum()), called_no_match=int((none & real).sum()),
        review_queue_lines=int(review.sum()),
        no_match_lines=int((~real).sum()), no_match_right=int((none & ~real).sum()), no_match_in_review=int((review & ~real).sum()),
        no_match_wrongly_accepted=int((accept & ~real).sum()))


def main():
    parts, owner = k.build_parts(k.load_boms())
    catalog = k.catalog_rows(set(owner))
    catalog = catalog.assign(part=catalog["mpn"].str.upper().map(owner))
    negatives = k.no_match_lines()

    with_lines = sorted(catalog["part"].unique())
    random.Random(k.SEED).shuffle(with_lines)
    train_parts = set(with_lines[:len(with_lines) // 2])
    train_cat, test_cat = catalog[catalog["part"].isin(train_parts)], catalog[~catalog["part"].isin(train_parts)]

    # Training data: first-half lines with the part number visible, so the rule can label them. No truth is read.
    train_inp = k.make_inputs(parts, lines_for(train_cat, owner, visible=True))
    train_rules = match.rule_match(train_inp)
    self_scorer, info = scoring.self_train(train_inp, train_rules, BACKEND, use_history=False)
    transferred = scoring.Scorer.load(BACKEND)

    # Test: second-half parts with the part number hidden, plus the no-match lines.
    test_inp = k.make_inputs(parts, pd.concat([lines_for(test_cat, owner, visible=False), negatives], ignore_index=True))
    p = test_inp.purchases
    rules = match.rule_match(test_inp)
    cand_parts, cand_sims = match.embedding_candidates(test_inp, backend=BACKEND)
    pairs = scoring.build_pairs(test_inp, rules, cand_parts, cand_sims)

    report = dict(bom_parts=len(parts), parts_with_catalog_lines=len(with_lines), train_parts=len(train_parts), test_parts=len(with_lines) - len(train_parts),
                  train_lines_available=len(train_cat), self_train_fit=info, results={})
    print(f"{len(train_cat)} first-half lines, {info['train_lines']} matched by the rule and used for training "
          f"({info['train_pairs']} pairs, {info['positives']} positives); accept at p >= {info['accept_at']:.3f}, no match below {info['reject_below']:.3f}")
    print(f"test: {len(test_cat)} second-half lines with the part number hidden + {len(negatives)} no-match lines")
    for name, scorer in (("Transferred from the synthetic data", transferred), ("Self-trained on this dataset's rule-matched lines", self_scorer)):
        dec = scoring.decide(rules, pairs, scorer)
        band = dec.index[dec["decision"] == "review"]
        verdicts, usage = adjudicate.adjudicate(adjudicate.build_prompts(test_inp, rules, pairs, band), cache=k.LLM_CACHE)
        final = adjudicate.apply(dec, verdicts, transferred.llm_accept_conf, transferred.llm_reject_conf)
        report["results"][name] = dict(classifier_stage=metrics(dec, p), with_claude=metrics(final, p), claude=dict(lines_sent=len(band), **usage))
        print(f"\n=== {name} ===")
        for label, m in (("rules + classifier", report["results"][name]["classifier_stage"]), ("rules + classifier + Claude", report["results"][name]["with_claude"])):
            print(f"  {label:<28} accepted {m['accepted']:>3} of {m['test_lines']}: {m['right']} right, {m['wrong']} wrong | precision {m['precision']:.2f} recall {m['recall']:.2f} | "
                  f"review queue {m['review_queue_lines']} lines | no-match lines: {m['no_match_right']} of {m['no_match_lines']} right, {m['no_match_wrongly_accepted']} wrongly accepted")
        print(f"  Claude: {len(band)} lines sent, {usage['calls']} calls, ${usage['cost_usd']:.2f}")
    Path("out").mkdir(exist_ok=True)
    Path("out/eval_external_selftrain.json").write_text(json.dumps(report, indent=2, default=float))
    print("\nwrote out/eval_external_selftrain.json")


if __name__ == "__main__":
    main()
