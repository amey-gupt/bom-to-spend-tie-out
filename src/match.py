"""Match purchase lines to BOM parts: rules first, then embedding candidates.

Reads data/*.csv only (through normalize.py). Never reads data/truth/.

    python3 -m src.match            # writes out/matches.csv

Stages
  naive   raw part-number string equality (the baseline a spreadsheet VLOOKUP gives)
  rules   normalized part number, then the supplier crosswalk
  embed   top-k BOM parts by description similarity, for lines the rules did not settle
"""
from __future__ import annotations

import hashlib
import os
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .normalize import Inputs, load_inputs, norm_pn

BACKENDS = {"local": "sentence-transformers/all-MiniLM-L6-v2", "openai": "text-embedding-3-small"}
CACHE_DIR = Path("cache")
TOP_K = 5
DEFAULT_THRESHOLD = 0.80  # evaluate.py picks the working threshold on the tuning split


def naive_match(inp: Inputs) -> pd.Series:
    """Exact string equality between the purchase line's part-number field and a BOM part number."""
    known = set(inp.parts["part_number"])
    return inp.purchases["supplier_part_number"].where(inp.purchases["supplier_part_number"].isin(known), "")


def rule_match(inp: Inputs) -> pd.DataFrame:
    """Returns rule_part ('' when none) and rule_method per purchase line.

    pn_exact     normalized part number equals a BOM part's
    xref         crosswalk maps the supplier's part number to a BOM part
    xref_offbom  crosswalk maps it to an internal part that is on no supplied BOM: a settled 'no match'
    pn_in_text   exactly one BOM part number is written out inside the item description
    """
    p = inp.purchases
    by_key = inp.parts.drop_duplicates("pn_key", keep=False).set_index("pn_key")["part_number"]
    xref = inp.xref.drop_duplicates("spn_key", keep=False).set_index("spn_key")["part_number"]
    on_bom = set(inp.parts["part_number"])

    long_keys = {k for k in by_key.index if len(k) >= 6}
    part, method = [], []
    for key, text in zip(p["pn_key"], p["item_description"]):
        in_text = long_keys & {norm_pn(tok) for tok in re.split(r"[\s,;()]+", text)} if long_keys else set()
        if key and key in by_key.index:
            part.append(by_key[key]); method.append("pn_exact")
        elif key and key in xref.index:
            target = xref[key]
            part.append(target if target in on_bom else ""); method.append("xref" if target in on_bom else "xref_offbom")
        elif len(in_text) == 1:
            part.append(by_key[in_text.pop()]); method.append("pn_in_text")
        else:
            part.append(""); method.append("")
    return pd.DataFrame({"rule_part": part, "rule_method": method}, index=p.index)


# ----------------------------------------------------------------- embeddings

def load_env():
    """API keys come from the environment, or from .env.local / .env (both gitignored)."""
    for name in (".env.local", ".env"):
        if Path(name).exists():
            for line in Path(name).read_text().splitlines():
                if "=" in line and not line.lstrip().startswith("#"):
                    key, value = line.split("=", 1)
                    os.environ.setdefault(key.strip(), value.strip().strip('"\''))


def _encode(texts: list[str], backend: str) -> np.ndarray:
    if backend == "local":
        from sentence_transformers import SentenceTransformer  # imported late: slow to load, not needed on a warm cache
        return SentenceTransformer(BACKENDS[backend]).encode(texts, batch_size=128, normalize_embeddings=True, show_progress_bar=False)
    from openai import OpenAI
    load_env()
    client, rows = OpenAI(), []
    for i in range(0, len(texts), 500):
        rows += [d.embedding for d in client.embeddings.create(model=BACKENDS[backend], input=texts[i:i + 500]).data]
    vectors = np.asarray(rows)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def embed(texts: list[str], backend: str = "local") -> np.ndarray:
    """Unit-length embeddings, cached on disk per (model, text) so nothing is computed or paid for twice."""
    CACHE_DIR.mkdir(exist_ok=True)
    path = CACHE_DIR / f"embeddings_{hashlib.sha1(BACKENDS[backend].encode()).hexdigest()[:10]}.pkl"
    cache = pickle.loads(path.read_bytes()) if path.exists() else {}
    missing = sorted({t for t in texts if t not in cache})
    if missing:
        cache.update(zip(missing, _encode(missing, backend).astype(np.float32)))
        path.write_bytes(pickle.dumps(cache))
    return np.stack([cache[t] for t in texts])


def embedding_candidates(inp: Inputs, k: int = TOP_K, backend: str = "local") -> tuple[np.ndarray, np.ndarray]:
    """For every purchase line, the k most similar BOM parts by description: (part numbers, cosine similarities)."""
    part_vecs = embed(inp.parts["desc_norm"].tolist(), backend)
    line_vecs = embed(inp.purchases["desc_norm"].tolist(), backend)
    sims = line_vecs @ part_vecs.T
    top = np.argsort(-sims, axis=1)[:, :k]
    return inp.parts["part_number"].to_numpy()[top], np.take_along_axis(sims, top, axis=1)


def combine(rules: pd.DataFrame, cand_parts: np.ndarray, cand_sims: np.ndarray, threshold: float) -> pd.Series:
    """Rules win when they settled the line; otherwise take the best embedding candidate if it clears the threshold."""
    by_embedding = np.where(cand_sims[:, 0] >= threshold, cand_parts[:, 0], "")
    return pd.Series(np.where(rules["rule_method"] != "", rules["rule_part"], by_embedding), index=rules.index)


def run(data_dir: str | Path = "data", threshold: float = DEFAULT_THRESHOLD) -> pd.DataFrame:
    inp = load_inputs(data_dir)
    rules = rule_match(inp)
    cand_parts, cand_sims = embedding_candidates(inp)
    out = pd.DataFrame({"po_id": inp.purchases["po_id"], "naive_part": naive_match(inp), **rules})
    for i in range(TOP_K):
        out[f"cand{i + 1}"] = cand_parts[:, i]
        out[f"sim{i + 1}"] = cand_sims[:, i].round(4)
    out["matched_part"] = combine(rules, cand_parts, cand_sims, threshold)
    return out


if __name__ == "__main__":
    out = run()
    Path("out").mkdir(exist_ok=True)
    out.to_csv("out/matches.csv", index=False)
    n = len(out)
    print(f"{n} purchase lines")
    print(f"  naive exact match     {(out['naive_part'] != '').sum():>5}  {(out['naive_part'] != '').mean():.1%}")
    for method, count in out["rule_method"].value_counts().items():
        print(f"  rule {method or '(not settled)':<16} {count:>5}  {count / n:.1%}")
    print(f"  matched at threshold {DEFAULT_THRESHOLD}: {(out['matched_part'] != '').mean():.1%}")
    print("wrote out/matches.csv")
