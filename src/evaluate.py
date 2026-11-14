"""Score the matcher against the planted ground truth.

The only module that reads data/truth/.

    python3 -m src.evaluate step2 [--backend local|openai]   # naive vs rules vs rules + embeddings
    python3 -m src.evaluate step3 [--no-llm]                  # classifier, three outcomes, Claude on the review band
    python3 -m src.evaluate step4 [--materiality USD]         # buckets, tie-out by bucket, leads vs planted findings
    python3 -m src.evaluate selftrain                         # experiment: classifier trained on rule-matched lines only
    python3 -m src.evaluate crosswalk                         # experiment: does the reviewer crosswalk cut review work over time

step3 also trains the classifier (labels come from the truth, tuning split only) and saves it to models/.

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

from . import adjudicate, allocate, match, score as scoring
from .normalize import Inputs, load_inputs


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


def step2(data_dir: str | Path = "data", backend: str = "local"):
    inp = load_inputs(data_dir)
    p = inp.purchases
    truth, canon = load_truth(data_dir)
    assert (truth["po_id"] == p["po_id"]).all()

    rules = match.rule_match(inp)
    cand_parts, cand_sims = match.embedding_candidates(inp, backend=backend)
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
    Path(f"out/eval_step2_{backend}.json").write_text(json.dumps(report, indent=2, default=float))
    print(f"\nwrote out/eval_step2_{backend}.json")
    return report


# ---------------------------------------------------------------------- step 3

def outcome_metrics(dec: pd.DataFrame, truth: pd.DataFrame, purchases: pd.DataFrame, canon: dict, mask: np.ndarray) -> dict:
    same = lambda s: s.map(lambda x: canon.get(x, x))
    d, spend, po_type = dec[mask], purchases["spend_usd"][mask], purchases["po_type"][mask]
    true = same(truth["true_part_number"][mask])
    accept, review, none = d["decision"] == "accept", d["decision"] == "review", d["decision"] == "no_match"
    right = accept & (same(d["part"]) == true)
    by_model = accept & ~d["source"].str.startswith("rule_")  # accepted by the classifier or the LLM, not by a rule
    total = spend.sum()
    return dict(
        lines=int(mask.sum()), spend=float(total),
        accept_lines=int(accept.sum()), accept_share=float(accept.mean()), accept_spend_share=float(spend[accept].sum() / total),
        accept_precision=float(right.sum() / max(accept.sum(), 1)),
        model_accept_lines=int(by_model.sum()), model_accept_precision=float((right & by_model).sum() / max(by_model.sum(), 1)),
        recall=float(right.sum() / max((true != "").sum(), 1)),
        review_lines=int(review.sum()), review_spend=float(spend[review].sum()), review_spend_share=float(spend[review].sum() / total),
        no_match_lines=int(none.sum()), no_match_accuracy=float((none & (true == "")).sum() / max(none.sum(), 1)),
        no_match_missed_spend=float(spend[none & (true != "")].sum()),
        tie_out=float(spend[accept | po_type.isin(["service", "warranty"])].sum() / total),
        true_tie_out=float(spend[truth["true_bucket"][mask] != "unassigned"].sum() / total),
        wrong_match_spend=float(spend[accept & ~right].sum()),
    )


def print_outcomes(title: str, m: dict):
    print(f"\n{title}")
    print(f"  auto-accept   {m['accept_lines']:>5} lines ({m['accept_share']:.1%} of lines, {m['accept_spend_share']:.1%} of spend), precision {m['accept_precision']:.2%}"
          f"  [accepted by model, not rules: {m['model_accept_lines']} lines at {m['model_accept_precision']:.2%}]")
    print(f"  review queue  {m['review_lines']:>5} lines, ${m['review_spend']:,.0f} ({m['review_spend_share']:.1%} of spend)")
    print(f"  no match      {m['no_match_lines']:>5} lines, {m['no_match_accuracy']:.1%} truly have no BOM part (${m['no_match_missed_spend']:,.0f} wrongly dropped)")
    print(f"  recall {m['recall']:.1%} | tie-out {m['tie_out']:.1%} vs true {m['true_tie_out']:.1%} | wrong-match ${m['wrong_match_spend']:,.0f}")


def _llm_threshold(conf: pd.Series, right: pd.Series, target: float) -> float:
    """Lowest confidence cutoff whose answers are right at the target rate on the tuning split; 1.01 = never auto-resolve."""
    for c in np.arange(0.50, 1.0, 0.01):
        keep = conf >= c
        if keep.sum() >= 20 and right[keep].mean() >= target:
            return float(round(c, 2))
    return 1.01


def step3(data_dir: str | Path = "data", backends=("local", "openai"), use_llm: bool = True):
    inp = load_inputs(data_dir)
    p = inp.purchases
    truth, canon = load_truth(data_dir)
    same = np.vectorize(lambda x: canon.get(x, x))
    true_part = same(truth["true_part_number"].to_numpy())
    tune, held = (truth["split"] == "tune").to_numpy(), (truth["split"] == "heldout").to_numpy()
    rules = match.rule_match(inp)
    open_ = (rules["rule_method"] == "").to_numpy()
    # Cross-validation folds leave one description style out at a time, so calibration and thresholds are set
    # on styles the fold's model never saw. That is the situation the held-out split tests.
    style = pd.factorize(truth["desc_style"])[0]
    protect = scoring.material_items(p, allocate.MATERIALITY_USD)  # material items the classifier may not close on its own
    previous = json.loads(Path("out/eval_step3.json").read_text()) if Path("out/eval_step3.json").exists() else {}
    report, runs = {"backends": {}}, {}

    for backend in backends:
        cand_parts, cand_sims = match.embedding_candidates(inp, backend=backend)
        pairs = scoring.build_pairs(inp, rules, cand_parts, cand_sims)
        label = (same(pairs["cand"].to_numpy()) == true_part[pairs["line"]]).astype(int)
        line_tune, line_open = tune[pairs["line"]], open_[pairs["line"]]
        # Training rows, tuning split only: lines the rules left open, plus rule-settled lines with the part number hidden.
        a, b = pairs[line_tune & line_open], scoring.without_pn(pairs[line_tune & ~line_open])
        train = pd.concat([a, b], ignore_index=True)
        y = np.concatenate([label[line_tune & line_open], label[line_tune & ~line_open]])
        scorer, info = scoring.fit(train, y, style[train["line"]], np.arange(len(train)) < len(a), backend)
        dec = scoring.decide(rules, pairs, scorer, protect)
        hits = same(cand_parts[held & open_ & (true_part != "")]) == true_part[held & open_ & (true_part != "")][:, None]
        tune_open = tune & open_ & (true_part != "")
        tune_top1 = float((same(cand_parts[tune_open][:, 0]) == true_part[tune_open]).mean())
        report["backends"][backend] = dict(fit=info, tune_recall_at_1=tune_top1, recall_at_1=float(hits[:, 0].mean()), recall_at_5=float(hits.any(axis=1).mean()),
                                           heldout=outcome_metrics(dec, truth, p, canon, held), tune=outcome_metrics(dec, truth, p, canon, tune))
        runs[backend] = (scorer, pairs, dec)
        print(f"\n=== embedding backend: {backend} ({match.BACKENDS[backend]}) ===")
        print(f"held-out lines the rules left open: true part is the top candidate for {hits[:, 0].mean():.1%}, in the top 5 for {hits.any(axis=1).mean():.1%}")
        print(f"classifier trained on {info['train_pairs']} tune pairs; accept at p >= {info['accept_at']:.3f}, no match below {info['reject_below']:.3f} "
              f"(out-of-fold on tune: {info['oof_accept_share']:.1%} of open lines accepted at {info['oof_accept_precision']:.2%} precision)")
        print_outcomes("HELD-OUT, rules + classifier", report["backends"][backend]["heldout"])

    # The LLM stage runs once, on the backend whose top candidate is right more often on the tuning split.
    # (Out-of-fold accept share was used before; it differs by a point or two between backends and flips with the data.)
    best = max(backends, key=lambda b: report["backends"][b]["tune_recall_at_1"])
    print("\ntuning-split top-candidate accuracy: " + ", ".join(f"{b} {report['backends'][b]['tune_recall_at_1']:.1%}" for b in backends) + f" -> {best}")
    scorer, pairs, dec = runs[best]
    report["chosen_backend"] = best
    if use_llm:
        band = dec.index[dec["decision"] == "review"]
        prompts = adjudicate.build_prompts(inp, rules, pairs, band)
        print(f"\n=== Claude on the review band ({best} backend): {len(band)} lines, {prompts['prompt'].nunique()} distinct items ===")
        report["llm"], classifier_dec = {}, dec
        for model in (adjudicate.SMALL_MODEL, adjudicate.MODEL):  # the main model runs last, so its decisions are the ones saved
            verdicts, usage = adjudicate.adjudicate(prompts, model)
            answered = verdicts["llm_conf"].notna()
            v_right = pd.Series(same(verdicts["llm_part"].to_numpy()) == true_part[verdicts.index], index=verdicts.index)
            in_tune = pd.Series(tune[verdicts.index], index=verdicts.index) & answered
            in_held = pd.Series(held[verdicts.index], index=verdicts.index) & answered
            picks, nones = in_tune & (verdicts["llm_part"] != ""), in_tune & (verdicts["llm_part"] == "")
            accept_conf = _llm_threshold(verdicts["llm_conf"][picks], v_right[picks], scoring.ACCEPT_PRECISION)
            reject_conf = _llm_threshold(verdicts["llm_conf"][nones], v_right[nones], 1 - scoring.REJECT_MISS_RATE)
            dec = adjudicate.apply(classifier_dec, verdicts, accept_conf, reject_conf)
            report["llm"][model] = dict(usage=usage, band_lines=len(band), unanswered_lines=int((~answered).sum()),
                                        accept_conf=accept_conf, reject_conf=reject_conf,
                                        heldout_band_lines=int(in_held.sum()), heldout_answer_accuracy=float(v_right[in_held].mean()),
                                        tune_answer_accuracy=float(v_right[in_tune].mean()),
                                        heldout=outcome_metrics(dec, truth, p, canon, held), tune=outcome_metrics(dec, truth, p, canon, tune))
            print(f"\n--- {model}: {usage['calls']} calls, {usage['input_tokens']:,} input + {usage['output_tokens']:,} output tokens, "
                  f"${usage['cost_usd']:.2f} at list price ({usage['failed_calls']} failed) ---")
            print(f"answer is right on {v_right[in_held].mean():.1%} of {in_held.sum()} held-out band lines ({v_right[in_tune].mean():.1%} on tune)")
            print(f"confidence cutoffs set on tune: accept a pick at >= {accept_conf}, accept 'none' at >= {reject_conf} (1.01 = never)")
            print_outcomes(f"HELD-OUT, rules + classifier + {model}", report["llm"][model]["heldout"])
            print_outcomes(f"TUNE, rules + classifier + {model}", report["llm"][model]["tune"])
        scorer.llm_accept_conf, scorer.llm_reject_conf = accept_conf, reject_conf

        # Item-level materiality was added after seeing held-out results. Keep the last results from before it for comparison.
        main = report["llm"][adjudicate.MODEL]
        before = previous.get("item_materiality", {}).get("before") or {k: previous.get("llm", {}).get(adjudicate.MODEL, {}).get(k) for k in ("heldout", "tune", "band_lines")}
        report["item_materiality"] = dict(threshold_usd=allocate.MATERIALITY_USD, protected_lines=int(protect.sum()),
                                          held_from_classifier_no_match=int((protect & (scoring.decide(rules, pairs, scorer)["decision"] == "no_match").to_numpy()).sum()),
                                          before=before, after={k: main[k] for k in ("heldout", "tune", "band_lines")})

    # Full-pipeline results by noise style, for both splits (the app's evaluation tab reads this).
    report["by_style"] = []
    for split, split_mask in (("heldout", held), ("tune", tune)):
        for col in ("id_style", "desc_style"):
            for style_name in sorted(truth.loc[split_mask, col].unique()):
                m = outcome_metrics(dec, truth, p, canon, split_mask & (truth[col] == style_name).to_numpy())
                report["by_style"].append(dict(split=split, kind=col, style=style_name, lines=m["lines"], accept_share=m["accept_share"],
                                               accept_precision=m["accept_precision"], recall=m["recall"], review_lines=m["review_lines"],
                                               wrong_match_spend=m["wrong_match_spend"]))
    report["total_lines"] = int(len(p))

    scorer.save()
    Path("out").mkdir(exist_ok=True)
    dec.insert(0, "po_id", p["po_id"])
    dec.to_csv("out/decisions.csv", index=False)
    Path("out/eval_step3.json").write_text(json.dumps(report, indent=2, default=float))
    print(f"\nsaved models/scorer_{best}.pkl, out/decisions.csv, out/eval_step3.json")
    return report


# ---------------------------------------------------------------------- step 4

def step4(data_dir: str | Path = "data", materiality: float | None = None):
    from . import allocate
    inp, lines, alloc, products, leads = allocate.run(data_dir=data_dir, materiality=allocate.MATERIALITY_USD if materiality is None else materiality)
    truth, canon = load_truth(data_dir)
    assert (truth["po_id"] == lines["po_id"]).all()
    spend, held = lines["spend_usd"], (truth["split"] == "heldout")
    right = lines["bucket"] == truth["true_bucket"]
    report = {"materiality_usd": allocate.MATERIALITY_USD if materiality is None else materiality}

    print("BUCKETS")
    for name, mask in (("all lines", pd.Series(True, index=lines.index)), ("held-out", held)):
        acc = spend[right & mask].sum() / spend[mask].sum()
        inferred = mask & lines["bucket_inferred"]
        acc_inf = spend[right & inferred].sum() / max(spend[inferred].sum(), 1e-9)
        report[f"bucket_accuracy_{name}"] = dict(by_spend=float(acc), inferred_lines=int(inferred.sum()), inferred_by_spend=float(acc_inf))
        print(f"  {name}: bucket right for {acc:.1%} of spend; where the bucket was inferred ({inferred.sum()} lines), right for {acc_inf:.1%}")
    rows = {}
    print(f"\n  {'bucket':<12}{'tool':>15}{'share':>8}{'truth':>15}{'share':>8}{'tool - truth':>14}")
    for b in ("production", "spares", "warranty", "unassigned"):
        got, true = spend[lines["bucket"] == b].sum(), spend[truth["true_bucket"] == b].sum()
        rows[b] = dict(tool_usd=float(got), truth_usd=float(true))
        print(f"  {b:<12}{got:>15,.0f}{got / spend.sum():>8.1%}{true:>15,.0f}{true / spend.sum():>8.1%}{got - true:>+14,.0f}")
    tied, true_tied = 1 - rows["unassigned"]["tool_usd"] / spend.sum(), 1 - rows["unassigned"]["truth_usd"] / spend.sum()
    review = lines["decision"] == "review"
    report.update(buckets=rows, tie_out=float(tied), true_tie_out=float(true_tied), review_lines=int(review.sum()), review_spend=float(spend[review].sum()),
                  moved_by_materiality=int((lines["source"] == "llm_material").sum()))
    print(f"  tie-out rate {tied:.1%} vs true {true_tied:.1%}; review queue {review.sum()} lines, ${spend[review].sum():,.0f} "
          f"({report['moved_by_materiality']} lines moved there by the ${report['materiality_usd']:,.0f} materiality rule)")

    print("\nPARTS COST PER UNIT (estimated from purchases) VS BOM STANDARD")
    print(products.to_string(index=False, formatters={"std_cost_per_unit": "${:.2f}".format, "actual_cost_per_unit_est": "${:.2f}".format,
                                                      "allocated_production_spend_est": "${:,.0f}".format, "actual_vs_std": "{:+.1%}".format}))
    report["products"] = products.to_dict("records")

    # Lines bought in a different dimension from the BOM unit must be held with a unit reason.
    cross = truth["cross_dimension"] == "True"
    flagged = lines["uom_issue"] != ""
    held_for_units = lines["source"] == "uom_mismatch"
    report["units"] = dict(cross_dimension_lines=int(cross.sum()), flagged=int((flagged & cross).sum()), held_in_review=int((held_for_units & cross).sum()),
                           wrongly_flagged=int((flagged & ~cross & (lines["decision"] == "accept")).sum()),
                           converted_lines=int((lines["qty_bom"].notna() & (lines["qty_bom"] != lines["qty"]) & (lines["decision"] == "accept")).sum()))
    u = report["units"]
    print(f"\nUNITS: {u['cross_dimension_lines']} lines bought in another dimension; {u['flagged']} flagged, {u['held_in_review']} held in review with a unit reason; "
          f"{u['wrongly_flagged']} accepted lines wrongly flagged; {u['converted_lines']} accepted lines converted to the BOM unit")

    print("\nLEADS VS PLANTED FINDINGS")
    planted = pd.read_csv(Path(data_dir) / "truth" / "planted_findings.csv", dtype=str, keep_default_na=False)
    key = lambda df, kind, col: ({frozenset(x) for x in zip(df.loc[df[col] == kind, "part_number"], df.loc[df[col] == kind, "related"])}
                                 if kind == "duplicate_part" else set(df.loc[df[col] == kind, "part_number"]))
    report["leads"] = {}
    for kind in ("supplier_price_gap", "duplicate_part", "spares_premium", "over_purchase"):
        found, want = key(leads, kind, "lead_type"), key(planted, kind, "finding_type")
        hit = found & want
        mine = leads[leads["lead_type"] == kind]
        r = dict(found=len(found), planted=len(want), precision=len(hit) / max(len(found), 1), recall=len(hit) / max(len(want), 1),
                 low_usd=float(mine["low_usd"].sum()), high_usd=float(mine["high_usd"].sum()))
        if kind == "over_purchase":  # an engineering change that purchasing did not follow also shows up as a real gap
            ec_path = Path(data_dir) / "truth" / "engineering_changes.csv"
            ec = pd.read_csv(ec_path, dtype=str, keep_default_na=False) if ec_path.exists() else pd.DataFrame(columns=["part_number", "new_part_number"])
            r["explained_by_engineering_change"] = sorted((found - want) & (set(ec["part_number"]) | set(ec["new_part_number"])))
        if kind == "spares_premium":  # a planted premium can only be seen if spares of that part were actually bought
            bought = set(truth.loc[truth["line_kind"] == "spare", "true_part_number"])
            r["planted_with_spares_bought"] = len(want & bought)
            r["recall_of_detectable"] = len(hit) / max(len(want & bought), 1)
        report["leads"][kind] = r
        extra = f", {r['recall_of_detectable']:.0%} of the {r['planted_with_spares_bought']} that had spares bought" if kind == "spares_premium" else ""
        if kind == "over_purchase" and r["explained_by_engineering_change"]:
            extra = f"; not planted but on a part with an engineering change: {', '.join(r['explained_by_engineering_change'])}"
        print(f"  {kind:<20} found {r['found']:>3}, planted {r['planted']:>3}: precision {r['precision']:.0%}, recall {r['recall']:.0%}{extra}; "
              f"range ${r['low_usd']:,.0f} to ${r['high_usd']:,.0f}")

    allocate.write_outputs(inp, lines, alloc, products, leads)
    Path("out/eval_step4.json").write_text(json.dumps(report, indent=2, default=float))
    print("\nwrote out/eval_step4.json and the step-4 outputs")
    return report


# ------------------------------------------------------------ self-training

def selftrain(data_dir: str | Path = "data", backend: str = "openai"):
    """Experiment: a classifier trained on the dataset's own rule-matched lines (no truth labels) against the
    truth-trained one, both scored on the held-out split. Truth is used here for scoring only."""
    inp = load_inputs(data_dir)
    p = inp.purchases
    truth, canon = load_truth(data_dir)
    tune, held = (truth["split"] == "tune").to_numpy(), (truth["split"] == "heldout").to_numpy()
    rules = match.rule_match(inp)
    cand_parts, cand_sims = match.embedding_candidates(inp, backend=backend)
    pairs = scoring.build_pairs(inp, rules, cand_parts, cand_sims)
    scorers = {"Truth-trained (tuning-split labels)": scoring.Scorer.load(backend)}
    scorers["Self-trained (rule-matched lines, no truth)"], info = scoring.self_train(inp, rules, backend)
    cut = scorers["Truth-trained (tuning-split labels)"]

    report = {"backend": backend, "self_train_fit": info, "results": {}}
    print(f"self-trained on {info['train_lines']} rule-matched lines ({info['train_pairs']} pairs, {info['positives']} positives); "
          f"accept at p >= {info['accept_at']:.3f}, no match below {info['reject_below']:.3f}")
    protect = scoring.material_items(p, allocate.MATERIALITY_USD)
    for name, scorer in scorers.items():
        dec = scoring.decide(rules, pairs, scorer, protect)
        band = dec.index[dec["decision"] == "review"]
        verdicts, usage = adjudicate.adjudicate(adjudicate.build_prompts(inp, rules, pairs, band))
        final = adjudicate.apply(dec, verdicts, cut.llm_accept_conf, cut.llm_reject_conf)
        report["results"][name] = dict(
            classifier_stage=dict(heldout=outcome_metrics(dec, truth, p, canon, held), tune=outcome_metrics(dec, truth, p, canon, tune)),
            with_claude=dict(heldout=outcome_metrics(final, truth, p, canon, held), tune=outcome_metrics(final, truth, p, canon, tune)),
            claude=dict(lines_sent=len(band), **usage))
        print(f"\n=== {name} ===")
        print_outcomes("HELD-OUT, rules + classifier", report["results"][name]["classifier_stage"]["heldout"])
        print_outcomes("HELD-OUT, rules + classifier + Claude", report["results"][name]["with_claude"]["heldout"])
        print(f"  Claude: {len(band)} lines sent, {usage['calls']} calls, ${usage['cost_usd']:.2f}")
    Path("out").mkdir(exist_ok=True)
    Path("out/eval_selftrain.json").write_text(json.dumps(report, indent=2, default=float))
    print("\nwrote out/eval_selftrain.json")
    return report


# ------------------------------------------------------ crosswalk over time

def crosswalk_experiment(data_dir: str | Path = "data", backend: str = "openai"):
    """Does the crosswalk cut review work over time? Purchases are processed one quarter at a time, in date order.
    Each quarter the full pipeline runs with the crosswalk built so far; then a simulated reviewer resolves that
    quarter's review queue from the truth and the decisions are saved. Compared with the same run with no crosswalk.

    Two caveats: item text repeats by construction in this data (the generator reuses one item-master text per plant,
    supplier and part about 80% of the time), and the classifier is the one trained on tuning-split lines from all quarters.
    """
    inp = load_inputs(data_dir)
    p = inp.purchases
    truth, canon = load_truth(data_dir)
    same = np.vectorize(lambda x: canon.get(x, x))
    true_raw, true_part = truth["true_part_number"].to_numpy(), same(truth["true_part_number"].to_numpy())
    date = pd.to_datetime(p["po_date"])
    quarter = (date.dt.year.astype(str) + "Q" + date.dt.quarter.astype(str)).to_numpy()
    scorer = scoring.Scorer.load(backend)
    report = {"backend": backend, "arms": {}}

    for arm, use in (("With the crosswalk", True), ("No crosswalk", False)):
        saved, rows = {}, []
        for q in sorted(set(quarter)):
            idx = np.flatnonzero(quarter == q)
            inp_q = Inputs(p.iloc[idx].reset_index(drop=True), inp.boms, inp.parts, inp.builds, inp.xref)
            pq = inp_q.purchases
            rules = match.rule_match(inp_q, saved if use else None)
            cand_parts, cand_sims = match.embedding_candidates(inp_q, backend=backend)
            pairs = scoring.build_pairs(inp_q, rules, cand_parts, cand_sims)
            dec = scoring.decide(rules, pairs, scorer, scoring.material_items(pq, allocate.MATERIALITY_USD))
            band = dec.index[dec["decision"] == "review"]
            usage = dict(calls=0, cost_usd=0.0)
            if len(band):
                verdicts, usage = adjudicate.adjudicate(adjudicate.build_prompts(inp_q, rules, pairs, band))
                dec = adjudicate.apply(dec, verdicts, scorer.llm_accept_conf, scorer.llm_reject_conf)
            else:
                dec = dec.assign(llm_part="", llm_conf=np.nan, llm_reason="")
            dec.insert(0, "po_id", pq["po_id"])
            lines = allocate.apply_units(allocate.apply_materiality(allocate.build_lines(inp_q, dec)), inp.boms)

            t, spend = true_part[idx], lines["spend_usd"].to_numpy()
            accept, review = (lines["decision"] == "accept").to_numpy(), (lines["decision"] == "review").to_numpy()
            right = accept & (same(lines["part"].to_numpy()) == t)
            by_xw = rules["rule_method"].isin(["crosswalk", "crosswalk_none"]).to_numpy()
            xw_right = by_xw & (same(rules["rule_part"].to_numpy()) == t)
            rows.append(dict(quarter=q, lines=len(idx), spend=float(spend.sum()), settled_by_crosswalk=int(by_xw.sum()), crosswalk_right=int(xw_right.sum()),
                             claude_lines=len(band), claude_calls=usage["calls"], claude_cost_usd=usage["cost_usd"],
                             review_lines=int(review.sum()), review_spend=float(spend[review].sum()),
                             precision=float(right.sum() / max(accept.sum(), 1)), recall=float(right.sum() / max((t != "").sum(), 1)),
                             wrong_match_spend=float(spend[accept & ~right].sum()), crosswalk_entries=len(saved)))
            if use:  # the reviewer clears this quarter's queue; each decision is saved against the item's text
                for i in np.flatnonzero(review):
                    key = f"{pq.at[i, 'pn_key']}|{pq.at[i, 'desc_norm']}"
                    saved[key] = ("approve", true_raw[idx[i]]) if true_raw[idx[i]] else ("no_match", "")
        report["arms"][arm] = rows
        print(f"\n=== {arm} ===")
        print(f"{'quarter':<9}{'lines':>6}{'by crosswalk':>14}{'(right)':>9}{'Claude calls':>14}{'review lines':>14}{'review $':>12}{'precision':>11}{'wrong-match $':>15}")
        for r in rows:
            print(f"{r['quarter']:<9}{r['lines']:>6}{r['settled_by_crosswalk']:>14}{r['crosswalk_right']:>9}{r['claude_calls']:>14}{r['review_lines']:>14}"
                  f"{r['review_spend']:>12,.0f}{r['precision']:>11.2%}{r['wrong_match_spend']:>15,.0f}")
        tot = {k: sum(r[k] for r in rows) for k in ("settled_by_crosswalk", "crosswalk_right", "claude_calls", "claude_cost_usd", "review_lines", "review_spend", "wrong_match_spend")}
        print(f"{'total':<9}{sum(r['lines'] for r in rows):>6}{tot['settled_by_crosswalk']:>14}{tot['crosswalk_right']:>9}{tot['claude_calls']:>14}{tot['review_lines']:>14}"
              f"{tot['review_spend']:>12,.0f}{'':>11}{tot['wrong_match_spend']:>15,.0f}   Claude ${tot['claude_cost_usd']:.2f}")
    Path("out").mkdir(exist_ok=True)
    Path("out/eval_crosswalk.json").write_text(json.dumps(report, indent=2, default=float))
    print("\nwrote out/eval_crosswalk.json")
    return report


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["step2", "step3", "step4", "selftrain", "crosswalk"])
    ap.add_argument("--backend", choices=list(match.BACKENDS), default="local")
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--materiality", type=float, default=None)
    args = ap.parse_args()
    {"step2": lambda: step2(backend=args.backend), "step3": lambda: step3(use_llm=not args.no_llm),
     "step4": lambda: step4(materiality=args.materiality), "selftrain": selftrain, "crosswalk": crosswalk_experiment}[args.step]()
