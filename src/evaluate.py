"""Score the matcher against the planted ground truth.

The only module that reads data/truth/.

    python3 -m src.evaluate         # prints the comparison and writes out/eval_step2.json

Definitions (per purchase line; a duplicate part number counts as the same part)
  precision        correct matches / lines the method matched
  recall           correct matches / lines that truly have a BOM part
  dollar accuracy  share of spend whose outcome is right (right part, or correctly left unmatched)
  tie-out rate     share of spend tied to a product: matched to a BOM part, or flagged
                   service/warranty by its PO type. The true rate is 1 - truly unassigned spend.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import match
from .normalize import load_inputs


def load_truth(data_dir: str | Path = "data") -> tuple[pd.DataFrame, dict]:
    d = Path(data_dir) / "truth"
    truth = pd.read_csv(d / "ground_truth.csv", dtype=str, keep_default_na=False)
    canon = {}  # duplicate part number -> the part it duplicates
    for a, b in pd.read_csv(d / "duplicate_parts.csv", dtype=str).itertuples(index=False):
        canon[b] = a
    return truth, canon


def score(pred: pd.Series, truth: pd.DataFrame, purchases: pd.DataFrame, canon: dict, mask: np.ndarray) -> dict:
    same = lambda s: s.map(lambda x: canon.get(x, x))
    pred, true = same(pred[mask]), same(truth["true_part_number"][mask])
    spend, po_type = purchases["spend_usd"][mask], purchases["po_type"][mask]
    matched, has_part, right = pred != "", true != "", pred == true
    correct = (matched & right).sum()
    tied = matched | po_type.isin(["service", "warranty"])
    return dict(
        lines=int(mask.sum()),
        precision=correct / max(matched.sum(), 1),
        recall=correct / max(has_part.sum(), 1),
        dollar_accuracy=spend[right].sum() / spend.sum(),
        tie_out=spend[tied].sum() / spend.sum(),
        true_tie_out=spend[truth["true_bucket"][mask] != "unassigned"].sum() / spend.sum(),
        wrong_match_spend=float(spend[matched & ~right].sum()),
    )


def pick_threshold(rules, cand_parts, cand_sims, truth, purchases, canon, tune) -> float:
    """Similarity cutoff with the best F1 on the tuning split. Held-out lines play no part."""
    best, best_f1 = match.DEFAULT_THRESHOLD, -1.0
    for t in np.arange(0.50, 0.96, 0.01):
        s = score(match.combine(rules, cand_parts, cand_sims, t), truth, purchases, canon, tune)
        f1 = 2 * s["precision"] * s["recall"] / max(s["precision"] + s["recall"], 1e-9)
        if f1 > best_f1:
            best, best_f1 = float(round(t, 2)), f1
    return best


def main(data_dir: str | Path = "data"):
    inp = load_inputs(data_dir)
    p = inp.purchases
    truth, canon = load_truth(data_dir)
    assert (truth["po_id"] == p["po_id"]).all()

    rules = match.rule_match(inp)
    cand_parts, cand_sims = match.embedding_candidates(inp)
    tune, held = (truth["split"] == "tune").to_numpy(), (truth["split"] == "heldout").to_numpy()
    threshold = pick_threshold(rules, cand_parts, cand_sims, truth, p, canon, tune)

    methods = {
        "(a) naive exact part number": match.naive_match(inp),
        "(b) rules only": rules["rule_part"],
        "(c) rules + embeddings": match.combine(rules, cand_parts, cand_sims, threshold),
    }
    report = {"embedding_threshold": threshold, "threshold_chosen_on": "tune split, best F1", "splits": {}}
    for split, mask in (("heldout", held), ("tune", tune)):
        rows = {name: score(pred, truth, p, canon, mask) for name, pred in methods.items()}
        report["splits"][split] = rows
        first = next(iter(rows.values()))
        print(f"\n{split.upper()} split: {first['lines']} lines, true tie-out rate {first['true_tie_out']:.1%}")
        print(f"{'method':<30}{'precision':>10}{'recall':>9}{'$ accuracy':>12}{'tie-out':>9}{'vs true':>9}{'wrong-match $':>15}")
        for name, s in rows.items():
            print(f"{name:<30}{s['precision']:>10.1%}{s['recall']:>9.1%}{s['dollar_accuracy']:>12.1%}{s['tie_out']:>9.1%}"
                  f"{(s['tie_out'] - s['true_tie_out']) * 100:>+8.1f}p{s['wrong_match_spend']:>15,.0f}")

    # How good are the candidates themselves, on held-out lines the rules left open that truly have a part?
    same = np.vectorize(lambda x: canon.get(x, x))
    open_ = held & (rules["rule_method"] == "").to_numpy() & (truth["true_part_number"] != "").to_numpy()
    hits = same(cand_parts[open_]) == same(truth["true_part_number"].to_numpy()[open_])[:, None]
    report["heldout_open_lines"] = int(open_.sum())
    report["recall_at_1"], report["recall_at_5"] = float(hits[:, 0].mean()), float(hits.any(axis=1).mean())
    print(f"\nembedding threshold {threshold} (chosen on the tune split)")
    print(f"held-out lines the rules left open: {open_.sum()}; true part is the top candidate for {hits[:, 0].mean():.1%}, "
          f"in the top {match.TOP_K} for {hits.any(axis=1).mean():.1%}")

    by_style = []
    pred = methods["(c) rules + embeddings"]
    for col in ("id_style", "desc_style"):
        for style, idx in truth[held].groupby(col).groups.items():
            m = np.zeros(len(truth), bool); m[idx] = True
            s = score(pred, truth, p, canon, m)
            by_style.append(dict(style=f"{col}={style}", lines=s["lines"], precision=s["precision"], recall=s["recall"]))
    report["heldout_by_style_c"] = by_style
    print("\n(c) on the held-out split, by noise style")
    for r in by_style:
        print(f"  {r['style']:<26}{r['lines']:>6} lines  precision {r['precision']:.1%}  recall {r['recall']:.1%}")

    Path("out").mkdir(exist_ok=True)
    Path("out/eval_step2.json").write_text(json.dumps(report, indent=2, default=float))
    print("\nwrote out/eval_step2.json")


if __name__ == "__main__":
    main()
