"""Rerank embedding candidates with a calibrated classifier and sort lines into three outcomes.

Reads data/*.csv only (through normalize.py). Labels are passed in by the caller: in the
evaluation they come from the planted truth, in production from reviewer-confirmed matches.

    python3 -m src.score            # applies the saved model, writes out/decisions.csv

Outcomes per purchase line
  accept    rules settled it, or the best candidate's calibrated probability clears the accept threshold
  no_match  crosswalk says the part is on no supplied BOM, or no candidate reaches the reject threshold
  review    everything in between; this band goes to the LLM and then to a person
"""
from __future__ import annotations

import pickle
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold

from . import match
from .normalize import Inputs, load_inputs, norm_desc, norm_pn, numbers_in

MODEL_DIR = Path("models")
PN_FEATURES = ["pn_sim", "pn_prefix_match", "pn_looks_internal", "has_pn"]
FEATURES = ["emb_sim", "emb_rank", "emb_lead", "tfidf_sim", "tfidf_lead", "num_jaccard", "num_line_in_cand", "num_cand_in_line",
            "n_line_nums", "desc_len", "log_price_ratio", "sup_seen_part", "sup_share_prefix", *PN_FEATURES]
ACCEPT_PRECISION = 0.99   # auto-accept threshold is set to reach this on the tuning split
REJECT_MISS_RATE = 0.01   # at most this share of 'no match' calls may hide a true candidate, on the tuning split


def _lead(values: np.ndarray) -> np.ndarray:
    """Each candidate's score minus the best score among the line's other candidates."""
    order = np.sort(values, axis=1)
    best, second = order[:, -1:], order[:, -2:-1]
    return values - np.where(values == best, second, best)


def build_pairs(inp: Inputs, rules: pd.DataFrame, cand_parts: np.ndarray, cand_sims: np.ndarray, leave_own_line_out: bool = False) -> pd.DataFrame:
    """One row per (purchase line, candidate part) with the features the classifier sees.

    leave_own_line_out: when a rule-matched line is itself a training example, its own price and supplier must not
    count as evidence for its own part, or the price and supplier features would give the answer away.
    """
    p, parts = inp.purchases, inp.parts.set_index("part_number")
    n, k = cand_parts.shape
    line = np.repeat(np.arange(n), k)
    cand = cand_parts.ravel()
    part_pos = parts.index.get_indexer(cand)

    # character n-gram TF-IDF: robust to abbreviations, truncation and typos where word embeddings are not
    tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True).fit(pd.concat([parts["desc_norm"], p["desc_norm"]]))
    line_m, part_m = tfidf.transform(p["desc_norm"]), tfidf.transform(parts["desc_norm"])
    tfidf_sim = np.asarray(line_m[line].multiply(part_m[part_pos]).sum(axis=1)).ravel().reshape(n, k)

    # numbers and units
    line_nums = p["desc_norm"].map(numbers_in).to_numpy()[line]
    cand_nums = parts["desc_norm"].map(numbers_in).to_numpy()[part_pos]
    inter = np.array([len(a & b) for a, b in zip(line_nums, cand_nums)])
    n_line = np.array([len(a) for a in line_nums])
    n_cand = np.array([len(b) for b in cand_nums])

    # what the rule-matched lines already tell us about each part: its price and who supplies it
    settled = p[rules["rule_part"] != ""].assign(part=rules["rule_part"])
    ref_price = settled.groupby("part")["unit_price_usd"].median()
    part_suppliers = settled.groupby("part")["supplier_key"].agg(set).to_dict()
    prefix = settled["part"].str.split("-").str[0]
    prefix_share = (settled.groupby([prefix, "supplier_key"]).size() / settled.groupby(prefix).size()).to_dict()
    sup = p["supplier_key"].to_numpy()[line]
    cand_prefix = np.array([c.split("-")[0] for c in cand])

    # part-number strings
    line_pn = p["pn_key"].to_numpy()[line]
    cand_pn = parts["pn_key"].to_numpy()[part_pos]
    bom_prefixes = set(parts.index.str.split("-").str[0])
    line_prefix = np.array([m.group(1) if (m := re.match(r"^([A-Z]{3})\d", x)) else "" for x in line_pn])

    out = pd.DataFrame({
        "line": line, "cand": cand,
        "emb_sim": cand_sims.ravel(), "emb_rank": np.tile(np.arange(k), n), "emb_lead": _lead(cand_sims).ravel(),
        "tfidf_sim": tfidf_sim.ravel(), "tfidf_lead": _lead(tfidf_sim).ravel(),
        "num_jaccard": inter / np.maximum(n_line + n_cand - inter, 1),
        "num_line_in_cand": inter / np.maximum(n_line, 1), "num_cand_in_line": inter / np.maximum(n_cand, 1),
        "n_line_nums": n_line, "desc_len": p["desc_norm"].str.len().to_numpy()[line],
        "log_price_ratio": np.abs(np.log(p["unit_price_usd"].to_numpy()[line] / ref_price.reindex(cand).to_numpy())),  # NaN when no reference
        "sup_seen_part": [float(s in part_suppliers.get(c, ())) for s, c in zip(sup, cand)],
        "sup_share_prefix": [prefix_share.get((pre, s), 0.0) for pre, s in zip(cand_prefix, sup)],
        "pn_sim": [SequenceMatcher(None, a, b).ratio() if a else 0.0 for a, b in zip(line_pn, cand_pn)],
        "pn_prefix_match": (line_prefix == cand_prefix).astype(float),
        "pn_looks_internal": np.isin(line_prefix, list(bom_prefixes)).astype(float),
        "has_pn": (line_pn != "").astype(float),
    })
    if leave_own_line_out:
        prices = settled.groupby("part")["unit_price_usd"].apply(list).to_dict()
        counts = settled.groupby(["part", "supplier_key"]).size().to_dict()
        own_part, price_of = rules["rule_part"].to_numpy(), p["unit_price_usd"].to_numpy()
        for row in np.flatnonzero((own_part[line] == cand) & (own_part[line] != "")):
            i, c = line[row], cand[row]
            others = list(prices[c])
            others.remove(price_of[i])
            out.at[row, "log_price_ratio"] = abs(np.log(price_of[i] / np.median(others))) if others else np.nan
            out.at[row, "sup_seen_part"] = float(counts[c, sup[row]] > 1)
    return out


def without_pn(pairs: pd.DataFrame) -> pd.DataFrame:
    """The same pairs as if the part-number field had been left blank. Lets lines that the rules
    already settled teach the classifier about descriptions, prices and suppliers."""
    out = pairs.copy()
    out[PN_FEATURES] = 0.0
    return out


@dataclass
class Scorer:
    clf: HistGradientBoostingClassifier
    calibrator: IsotonicRegression
    accept_at: float
    reject_below: float
    backend: str
    llm_accept_conf: float = 1.01  # set by evaluate.py once the LLM band has been scored on the tuning split
    llm_reject_conf: float = 1.01
    unused: tuple = ()  # features that were entirely missing in training; zeroed so they cannot matter

    def proba(self, pairs: pd.DataFrame) -> np.ndarray:
        X = pairs[FEATURES].assign(**{c: 0.0 for c in getattr(self, "unused", ())})
        return self.calibrator.predict(self.clf.predict_proba(X)[:, 1])

    def save(self):
        MODEL_DIR.mkdir(exist_ok=True)
        (MODEL_DIR / f"scorer_{self.backend}.pkl").write_bytes(pickle.dumps(self))

    @staticmethod
    def load(backend: str = "local") -> "Scorer":
        return pickle.loads((MODEL_DIR / f"scorer_{backend}.pkl").read_bytes())


def _new_clf():
    return HistGradientBoostingClassifier(max_depth=3, max_iter=150, learning_rate=0.06, l2_regularization=1.0, random_state=0)


def best_per_line(pairs: pd.DataFrame, prob: np.ndarray) -> pd.DataFrame:
    scored = pairs[["line", "cand"]].assign(prob=prob)
    return scored.loc[scored.groupby("line")["prob"].idxmax()].set_index("line")


def fit(pairs: pd.DataFrame, label: np.ndarray, group: np.ndarray, decides: np.ndarray, backend: str) -> tuple[Scorer, dict]:
    """Train on labeled pairs. Calibration and both thresholds use out-of-fold predictions only,
    so they are not flattered by the model having seen the same rows.

    group    rows sharing a group stay in one fold; the evaluation groups by noise style, so every
             out-of-fold prediction is for a style that fold's model never saw
    decides  True for rows of lines the classifier would really decide (not settled by rules)
    """
    unused = tuple(c for c in FEATURES if pairs[c].isna().all())
    X = pairs[FEATURES].assign(**{c: 0.0 for c in unused})
    oof = np.zeros(len(pairs))
    for train, test in GroupKFold(n_splits=min(5, len(np.unique(group)))).split(X, label, group):
        oof[test] = _new_clf().fit(X.iloc[train], label[train]).predict_proba(X.iloc[test])[:, 1]
    calibrator = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(oof, label)
    prob = calibrator.predict(oof)

    real = pairs[decides].assign(label=label[decides])
    best = best_per_line(real, prob[decides])
    best["correct"] = real.set_index(["line", "cand"])["label"].reindex(list(zip(best.index, best["cand"]))).to_numpy()
    best["has_true_cand"] = real.groupby("line")["label"].max()

    grid = np.unique(np.round(best["prob"], 4))
    accept_at = next((t for t in grid if best.loc[best["prob"] >= t, "correct"].mean() >= ACCEPT_PRECISION), 1.01)
    reject_below = max((t for t in grid if best.loc[best["prob"] < t, "has_true_cand"].mean() <= REJECT_MISS_RATE), default=0.0)
    reject_below = min(reject_below, accept_at)

    accepted = best["prob"] >= accept_at
    info = dict(train_pairs=len(pairs), deciding_lines=len(best), accept_at=float(accept_at), reject_below=float(reject_below),
                oof_accept_share=float(accepted.mean()), oof_accept_precision=float(best.loc[accepted, "correct"].mean()),
                oof_review_share=float(((best["prob"] < accept_at) & (best["prob"] >= reject_below)).mean()))
    return Scorer(_new_clf().fit(X, label), calibrator, float(accept_at), float(reject_below), backend, unused=unused), info


def strip_matched_numbers(inp: Inputs, rules: pd.DataFrame) -> Inputs:
    """A copy of the inputs where every rule-matched line has its matched part number removed from the description."""
    key_of = inp.parts.set_index("part_number")["pn_key"]
    p = inp.purchases.copy()
    for i in rules.index[rules["rule_part"] != ""]:
        key = key_of[rules.at[i, "rule_part"]]
        kept = [tok for tok in re.split(r"(?<=[\s,;()])|(?=[\s,;()])", p.at[i, "item_description"]) if norm_pn(tok) != key]
        p.at[i, "item_description"] = "".join(kept).strip()
    p["desc_norm"] = p["item_description"].map(norm_desc)
    return Inputs(p, inp.boms, inp.parts, inp.builds, inp.xref)


def self_train(inp: Inputs, rules: pd.DataFrame, backend: str, train_lines: np.ndarray | None = None, use_history: bool = True) -> tuple[Scorer, dict]:
    """Train the classifier on one dataset with no truth labels.

    Positives are the lines a rule matched, paired with the part the rule chose; negatives are those lines' other
    nearest candidates. The part-number features are hidden and the matched number is stripped from the text, so
    the model has to learn from descriptions, numbers, prices and suppliers, which is what the open lines offer.
    Calibration and thresholds come from folds that hold out whole parts.

    train_lines  optional boolean mask limiting which rule-matched lines may be used
    use_history  False drops the price and supplier evidence (for data where the lines to score will have none)
    """
    matched = (rules["rule_part"] != "").to_numpy() & (np.ones(len(rules), bool) if train_lines is None else train_lines)
    stripped = strip_matched_numbers(inp, rules)
    cand_parts, cand_sims = match.embedding_candidates(stripped, backend=backend)
    history = rules if use_history else rules.assign(rule_part="", rule_method="")
    pairs = build_pairs(stripped, history, cand_parts, cand_sims, leave_own_line_out=use_history)
    pairs = without_pn(pairs[matched[pairs["line"]]]).reset_index(drop=True)
    target = rules["rule_part"].to_numpy()[pairs["line"]]
    label = (pairs["cand"].to_numpy() == target).astype(int)
    scorer, info = fit(pairs, label, pd.factorize(target)[0], np.ones(len(pairs), bool), backend)
    info.update(train_lines=int(matched.sum()), positives=int(label.sum()), lines_with_rule_part_in_top5=float(pairs.assign(y=label).groupby("line")["y"].max().mean()))
    return scorer, info


def material_items(purchases: pd.DataFrame, threshold: float) -> np.ndarray:
    """True for lines whose item is material: spend summed over every line with the same text is above the threshold."""
    item = purchases["pn_key"] + "|" + purchases["desc_norm"]
    return (purchases["spend_usd"].groupby(item).transform("sum") > threshold).to_numpy()


def decide(rules: pd.DataFrame, pairs: pd.DataFrame, scorer: Scorer, protect: np.ndarray | None = None) -> pd.DataFrame:
    """Per purchase line: decision, where it came from, the part (best candidate even when not accepted), probability.

    protect: lines the classifier may not close as "no match" on its own (material items). They join the review band
    instead, so Claude sees them and, if Claude is not confident either way, a person does.
    """
    best = best_per_line(pairs, scorer.proba(pairs))
    out = pd.DataFrame({"decision": "review", "source": "classifier", "part": best["cand"], "prob": best["prob"]}, index=rules.index)
    out.loc[out["prob"] >= scorer.accept_at, "decision"] = "accept"
    closed = (out["prob"] < scorer.reject_below).to_numpy()
    out.loc[closed & ~protect if protect is not None else closed, "decision"] = "no_match"
    hit = rules["rule_part"] != ""
    out.loc[hit, ["decision", "part", "prob"]] = np.column_stack([np.full(hit.sum(), "accept"), rules.loc[hit, "rule_part"], np.ones(hit.sum())])
    off = rules["rule_method"].isin(["xref_offbom", "crosswalk_none"])
    out.loc[off, ["decision", "prob"]] = ["no_match", 0.0]
    out.loc[hit | off, "source"] = "rule_" + rules.loc[hit | off, "rule_method"]
    out["prob"] = out["prob"].astype(float)
    return out


def run(backend: str = "local") -> pd.DataFrame:
    inp = load_inputs()
    rules = match.rule_match(inp)
    cand_parts, cand_sims = match.embedding_candidates(inp, backend=backend)
    from .allocate import MATERIALITY_USD
    out = decide(rules, build_pairs(inp, rules, cand_parts, cand_sims), Scorer.load(backend), material_items(inp.purchases, MATERIALITY_USD))
    out.insert(0, "po_id", inp.purchases["po_id"])
    out["spend_usd"] = inp.purchases["spend_usd"].round(2)
    return out


if __name__ == "__main__":
    out = run()
    Path("out").mkdir(exist_ok=True)
    out.to_csv("out/decisions.csv", index=False)
    summary = out.groupby("decision").agg(lines=("po_id", "size"), spend_usd=("spend_usd", "sum"))
    print(summary.assign(share_of_spend=(summary["spend_usd"] / summary["spend_usd"].sum()).map("{:.1%}".format)).to_string())
    print("wrote out/decisions.csv (classifier stage only; run src.evaluate step3 for the LLM band)")
