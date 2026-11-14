"""The matcher for a company's own files: rules, then embeddings, then Claude. No classifier.

The classifier in score.py is trained on confirmed matches. A new company has none yet, and the
external tests showed a classifier trained elsewhere does not transfer, so it is skipped here: every
line the rules leave open goes to Claude with its five nearest BOM parts. Reviewer decisions collect
in the crosswalk and are the training data a later classifier would use.

Never reads data/truth/.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

from . import adjudicate, match
from .normalize import Inputs
from .settings import secret

LLM_ACCEPT_CONF, LLM_REJECT_CONF = 0.72, 0.50  # Claude's confidence cutoffs, set on the synthetic tuning split
MAX_CALLS = 600                                # ceiling on distinct Claude calls per run; the largest lines go first
UPLOAD_CACHE = Path("cache/llm_uploads")       # not committed: holds text from uploaded files


def run(inp: Inputs, crosswalk: dict | None = None, backend: str | None = None, cache: Path | None = UPLOAD_CACHE,
        max_calls: int = MAX_CALLS) -> tuple[pd.DataFrame, dict]:
    """Returns (decisions per purchase line, run statistics). cache=None keeps Claude's answers in memory only."""
    if backend is None:  # OpenAI when a key is available; the local model only if it is installed
        texts = set(inp.parts["desc_norm"]) | set(inp.purchases["desc_norm"])
        if secret("OPENAI_API_KEY") or match.shipped_embeddings_cover(texts):
            backend = "openai"
        elif importlib.util.find_spec("sentence_transformers"):
            backend = "local"
        else:
            raise RuntimeError("No embedding model is available: add an OpenAI key, or install the local model.")
    p = inp.purchases
    rules = match.rule_match(inp, crosswalk)
    cand_parts, cand_sims = match.embedding_candidates(inp, backend=backend)

    dec = pd.DataFrame({"decision": "review", "source": "embedding", "part": cand_parts[:, 0], "prob": cand_sims[:, 0].astype(float)}, index=p.index)
    hit = rules["rule_part"] != ""
    dec.loc[hit, "decision"], dec.loc[hit, "part"], dec.loc[hit, "prob"] = "accept", rules.loc[hit, "rule_part"], 1.0
    none = rules["rule_method"].isin(["xref_offbom", "crosswalk_none"])
    dec.loc[none, ["decision", "prob"]] = ["no_match", 0.0]
    settled = hit | none
    dec.loc[settled, "source"] = "rule_" + rules.loc[settled, "rule_method"]

    open_lines = dec.index[~settled]
    n, k = cand_parts.shape
    pairs = pd.DataFrame({"line": [i for i in range(n) for _ in range(k)], "cand": cand_parts.ravel(), "emb_rank": list(range(k)) * n})
    prompts = adjudicate.build_prompts(inp, rules, pairs, open_lines)
    by_spend = prompts.assign(spend=p.loc[prompts.index, "spend_usd"]).sort_values("spend", ascending=False)
    keep = by_spend["prompt"].drop_duplicates().head(max_calls)
    prompts = prompts[prompts["prompt"].isin(set(keep))]
    verdicts, usage = adjudicate.adjudicate(prompts, cache=cache)
    final = adjudicate.apply(dec, verdicts, LLM_ACCEPT_CONF, LLM_REJECT_CONF)
    final.insert(0, "po_id", p["po_id"])

    stats = dict(lines=len(p), bom_parts=len(inp.parts), embedding_backend=match.BACKENDS[backend],
                 settled_by_rules=int(settled.sum()), sent_to_claude=len(prompts), not_sent=int(len(open_lines) - len(prompts)), call_limit=max_calls, claude=usage)
    return final, stats
