"""Send the uncertain middle band to Claude: one call per distinct item, with its five candidates.

Claude returns a part number or "none", a confidence, and a one-sentence reason that is shown
to the reviewer as evidence. Every response is cached on disk, so reruns and the demo need no network.
Never reads data/truth/.
"""
from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anthropic
import pandas as pd
from pydantic import BaseModel

from .match import load_env
from .normalize import Inputs

MODEL = "claude-opus-5-5"
SMALL_MODEL = "claude-haiku-5-5"  # the smallest current Claude model, for the cost comparison
PRICE_PER_MTOK = {"claude-opus-5-5": (4.00, 20.00), "claude-haiku-5-5": (0.10, 0.50)}  # USD (input, output), list price
CACHE = Path("cache/llm")

SYSTEM = """You match purchase-order lines to parts on a manufacturer's bills of materials (BOMs).

You get one purchase line as keyed by purchasing and five candidate BOM parts. Decide which candidate, if any, is the same physical part.

How purchasing text differs from engineering text:
- Descriptions are abbreviated, reordered, truncated, mistyped, written in a supplier's catalog wording, or in Spanish.
- Units are written many ways (0.1 kW = 100 W, 2000 mAh = 2.0 Ah, D28 = 28 mm diameter, L40 = 40 mm long, 14T or Z=14 = 14 teeth).
- The part-number field may hold the internal number (possibly with a revision suffix or a typo), the supplier's own number, a plant-specific code, or nothing.
- Prices are per single unit in USD. Spares and warranty buys can cost up to about twice the production price; a different supplier can differ by up to about 30%.

Rules:
- Every attribute stated in the purchase line (dimensions, ratings, material, color, finish) must agree with the candidate. One clear conflict rules a candidate out.
- A part number that is clearly an internal number of the same family but belongs to none of the candidates is strong evidence for "none": the company buys parts for products whose BOMs are not in this set.
- Service kits, complete assemblies, chargers, consumables and shop supplies are not BOM parts: answer "none".
- If the text fits more than one candidate and nothing else separates them, pick the most likely one and give a low confidence.

Answer with part_number set to one candidate's part number exactly, or "none". confidence is the probability (0 to 1) that your answer is right. reason is one sentence a procurement reviewer can check, naming the evidence."""


class Verdict(BaseModel):
    part_number: str
    confidence: float
    reason: str


def build_prompts(inp: Inputs, rules: pd.DataFrame, pairs: pd.DataFrame, lines) -> pd.DataFrame:
    """One prompt per purchase line in `lines`. Lines with the same item text and candidates get the same prompt."""
    p, parts = inp.purchases, inp.parts.set_index("part_number")
    settled = p[rules["rule_part"] != ""].assign(part=rules["rule_part"])
    ref_price = settled.groupby("part")["unit_price_usd"].median()
    suppliers = settled.groupby("part")["supplier"].agg(lambda s: next(iter(s.value_counts().index), ""))
    usage = inp.boms.groupby("part_number").apply(
        lambda g: f"{g['assembly_id'].iloc[0]} assembly of {', '.join(dict.fromkeys(g['product_id']))}", include_groups=False)
    cands = pairs[pairs["line"].isin(lines)].sort_values(["line", "emb_rank"]).groupby("line")["cand"].agg(list)

    item_cols = ["supplier_part_number", "item_description", "supplier_key", "plant", "uom", "po_type"]
    sub = p.loc[lines]
    price = sub.groupby(item_cols)["unit_price_usd"].transform("median")
    rows = []
    for i, row in sub.iterrows():
        text = [
            "Purchase line",
            f"- part-number field: {row['supplier_part_number'] or '(blank)'}",
            f"- description: {row['item_description']}",
            f"- supplier: {row['supplier']}",
            f"- plant: {row['plant']}; unit of measure: {row['uom']}; PO type: {row['po_type'] or '(blank)'}",
            f"- price per single unit: ${price[i]:.4f}",
            "", "Candidates",
        ]
        for c in cands[i]:
            known = f"${ref_price[c]:.4f} from {suppliers[c]}" if c in ref_price.index else "no confirmed purchases yet"
            text.append(f"- {c} | {parts.at[c, 'description']} | used in {usage[c]} | confirmed purchase price: {known}")
        rows.append((i, "\n".join(text), cands[i]))
    return pd.DataFrame(rows, columns=["line", "prompt", "cands"]).set_index("line")


def _ask(client: anthropic.Anthropic, prompt: str, model: str, system: str = SYSTEM, schema=Verdict, cache: Path = CACHE) -> dict | None:
    """One cached call. The cache entry keeps the verdict and the token usage that produced it."""
    path = cache / f"{hashlib.sha256((model + system + prompt).encode()).hexdigest()}.json"
    if path.exists():
        hit = json.loads(path.read_text())
        if "input_tokens" in hit:
            return hit
    try:
        resp = client.messages.parse(model=model, max_tokens=4000, system=system, output_config={"effort": "low"},
                                     messages=[{"role": "user", "content": prompt}], output_format=schema)
    except anthropic.RateLimitError:
        return None  # the SDK already retried; leave uncached so a rerun picks it up
    except anthropic.APIStatusError as e:
        print(f"  Claude API error {e.status_code}: {e.message}")
        return None
    except anthropic.APIConnectionError:
        return None
    if resp.stop_reason != "end_turn" or resp.parsed_output is None:
        return None
    entry = {**resp.parsed_output.model_dump(), "model": model,
             "input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens}
    path.write_text(json.dumps(entry))
    return entry


def ask_many(prompts: list[str], model: str = MODEL, system: str = SYSTEM, schema=Verdict, cache: Path = CACHE, workers: int = 8) -> tuple[dict, dict]:
    """Ask each distinct prompt once. Returns ({prompt: answer or None}, usage)."""
    load_env()
    cache.mkdir(parents=True, exist_ok=True)
    client = anthropic.Anthropic(max_retries=4)
    unique = list(dict.fromkeys(prompts))
    with ThreadPoolExecutor(workers) as pool:
        answers = dict(zip(unique, pool.map(lambda q: _ask(client, q, model, system, schema, cache), unique)))
    done = [a for a in answers.values() if a is not None]
    tokens_in, tokens_out = sum(a["input_tokens"] for a in done), sum(a["output_tokens"] for a in done)
    price_in, price_out = PRICE_PER_MTOK[model]
    return answers, dict(model=model, calls=len(done), failed_calls=len(unique) - len(done), input_tokens=tokens_in, output_tokens=tokens_out,
                         cost_usd=round((tokens_in * price_in + tokens_out * price_out) / 1e6, 4))


def adjudicate(prompts: pd.DataFrame, model: str = MODEL, workers: int = 8, cache: Path = CACHE) -> tuple[pd.DataFrame, dict]:
    """Returns (verdicts, usage). Verdicts hold llm_part ('' for none), llm_conf and llm_reason per line; lines whose
    call failed get llm_conf NaN and stay in review. Usage is tokens and list-price cost over the distinct calls,
    including calls answered from the cache (what it cost to produce them)."""
    answers, usage = ask_many(prompts["prompt"].tolist(), model, cache=cache, workers=workers)
    rows = []
    for line, prompt, cands in prompts.itertuples():
        v = answers[prompt]
        if v is None:
            rows.append((line, "", float("nan"), ""))
            continue
        part = v["part_number"].strip()
        valid = part in cands
        # an answer outside the candidate list that is not "none" is unusable: keep the line in review
        conf = v["confidence"] if valid or part.lower() == "none" else float("nan")
        rows.append((line, part if valid else "", min(max(conf, 0.0), 1.0), v["reason"]))
    return pd.DataFrame(rows, columns=["line", "llm_part", "llm_conf", "llm_reason"]).set_index("line"), usage


def apply(decisions: pd.DataFrame, verdicts: pd.DataFrame, accept_conf: float, reject_conf: float) -> pd.DataFrame:
    """Fold Claude's verdicts into the decisions. Confident answers resolve the line; the rest stay in
    review with Claude's suggestion and reason attached."""
    out = decisions.join(verdicts)
    out[["llm_part", "llm_reason"]] = out[["llm_part", "llm_reason"]].fillna("")
    in_band = out["decision"] == "review"
    take = in_band & (out["llm_part"] != "") & (out["llm_conf"] >= accept_conf)
    drop = in_band & (out["llm_part"] == "") & (out["llm_conf"] >= reject_conf)
    out.loc[take, ["decision", "source"]] = ["accept", "llm"]
    out.loc[take, "part"] = out.loc[take, "llm_part"]
    out.loc[drop, ["decision", "source"]] = ["no_match", "llm"]
    return out
