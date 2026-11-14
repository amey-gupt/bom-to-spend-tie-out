"""External test B: real electronic parts. Kitspace BOMs against the JLCPCB catalog, plus real no-match lines.

    python3 -m src.external.fetch kitspace jlcpcb ca_po
    python3 -m src.external.kitspace_jlc      # writes out/eval_external_parts.json

BOM side       lines from open-hardware 1-click-bom.tsv files: a description and one or more MPNs.
Purchase side  the JLCPCB catalog row for each of those MPNs: LCSC number, catalog description, price.
Truth          MPN equality. Two conditions: the MPN is appended to the purchase text, or it is hidden.
No-match side  500 State of California purchase-order lines, none of which is on these BOMs.

Zero-shot: the classifier and thresholds are the ones trained on the synthetic tuning split. Nothing is refit.
"""
from __future__ import annotations

import csv
import json
import random
import re
import sqlite3
from pathlib import Path

import pandas as pd

from .. import adjudicate, match, score as scoring
from ..normalize import Inputs, norm_desc, norm_pn, norm_supplier, numbers_in

DATA = Path("data/external")
LLM_CACHE = Path("cache/llm_external")  # not committed: answers quote the downloaded records
N_NO_MATCH = 500
SEED = 20261114


def load_boms() -> pd.DataFrame:
    """One row per BOM position that has both a description and at least one MPN."""
    rows = []
    for path in sorted((DATA / "kitspace").glob("*.tsv")):
        text = path.read_text(encoding="utf-8", errors="replace")
        table = list(csv.reader(text.splitlines(), delimiter="\t" if "\t" in text.splitlines()[0] else ","))
        head = [h.strip().lower() for h in table[0]]
        if "description" not in head:
            continue
        d_col = head.index("description")
        mpn_cols = [i for i, h in enumerate(head) if re.fullmatch(r"mp_?n\d*", h)]
        for r in table[1:]:
            r = r + [""] * (len(head) - len(r))
            mpns = [r[i].strip() for i in mpn_cols if r[i].strip()]
            if r[d_col].strip() and mpns:
                rows.append(dict(project=path.stem, description=r[d_col].strip(), mpns=mpns))
    return pd.DataFrame(rows)


def build_parts(bom_lines: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Unique BOM parts keyed by their first MPN. Returns (parts, any listed MPN -> part number)."""
    owner, parts = {}, {}
    for r in bom_lines.itertuples():
        key = next((owner[m.upper()] for m in r.mpns if m.upper() in owner), r.mpns[0])
        p = parts.setdefault(key, dict(part_number=key, description=r.description, projects=set()))
        p["projects"].add(r.project)
        if len(r.description) > len(p["description"]):
            p["description"] = r.description
        for m in r.mpns:
            owner.setdefault(m.upper(), key)
    return pd.DataFrame(parts.values()), owner


def catalog_rows(mpns: set) -> pd.DataFrame:
    """Catalog rows whose manufacturer part number equals one of the BOM MPNs (case-insensitive). One scan of 4M rows."""
    con = sqlite3.connect(f"file:{DATA / 'jlcpcb' / 'cache.sqlite3'}?mode=ro", uri=True)
    con.execute("create temp table want(mpn text primary key)")
    con.executemany("insert or ignore into want values (?)", [(m,) for m in mpns])
    rows = con.execute("select c.lcsc, c.mfr, c.manufacturer, c.subcategory, c.package, c.description, c.price, c.stock "
                       "from jlc_components c join want w on upper(c.mfr) = w.mpn where c.description != ''").fetchall()
    df = pd.DataFrame(rows, columns=["lcsc", "mpn", "manufacturer", "subcategory", "package", "description", "price", "stock"])
    df["unit_price"] = df["price"].map(lambda s: float(m.group(1)) if (m := re.search(r":([\d.]+)", s or "")) else float("nan"))
    return df.sort_values("stock", ascending=False).drop_duplicates("mpn")  # one catalog row per MPN: the best stocked


def no_match_lines() -> pd.DataFrame:
    recs = json.loads((DATA / "ca_po" / "purchase_orders.json").read_text())["result"]["records"]
    recs = [r for r in recs if (r.get("Item Name") or "").strip()]
    rows = []
    for r in random.Random(SEED).sample(recs, N_NO_MATCH):
        name, desc = r["Item Name"].strip(), (r.get("Item Description") or "").strip()
        price = re.sub(r"[^\d.]", "", r.get("Unit Price") or "")
        rows.append(dict(supplier=r.get("Supplier Name") or "unknown", supplier_part_number="",
                         item_description=name if desc in ("", name) else f"{name}. {desc}",
                         unit_price_usd=float(price) if price else float("nan"), true_part="", lcsc=None))
    return pd.DataFrame(rows)


def make_inputs(parts: pd.DataFrame, purchases: pd.DataFrame) -> Inputs:
    p = purchases.reset_index(drop=True).copy()
    p["po_id"] = [f"EXT{i:05d}" for i in range(len(p))]
    p["plant"], p["uom"], p["po_type"] = "", "EA", ""
    p["pn_key"], p["supplier_key"] = p["supplier_part_number"].map(norm_pn), p["supplier"].map(norm_supplier)
    p["desc_norm"] = p["item_description"].map(norm_desc)
    parts = parts.assign(revision="", pn_key=parts["part_number"].map(norm_pn), desc_norm=parts["description"].map(norm_desc))
    boms = pd.DataFrame([dict(part_number=r.part_number, assembly_id="PCB", product_id=proj) for r in parts.itertuples() for proj in sorted(r.projects)])
    return Inputs(p, boms, parts.drop(columns="projects"), pd.DataFrame(), pd.DataFrame(columns=["spn_key", "part_number"]))


def run_condition(parts, catalog, owner, negatives, scorer, visible: bool) -> dict:
    cat = pd.DataFrame({
        "supplier": "JLCPCB", "supplier_part_number": "C" + catalog["lcsc"].astype(str),
        "item_description": catalog["description"] + ((" " + catalog["mpn"]) if visible else ""),
        "unit_price_usd": catalog["unit_price"], "true_part": catalog["mpn"].str.upper().map(owner), "lcsc": catalog["lcsc"]})
    inp = make_inputs(parts, pd.concat([cat, negatives], ignore_index=True))
    p = inp.purchases
    rules = match.rule_match(inp)
    cand_parts, cand_sims = match.embedding_candidates(inp, backend=scorer.backend)
    pairs = scoring.build_pairs(inp, rules, cand_parts, cand_sims)
    dec = scoring.decide(rules, pairs, scorer)
    band = dec.index[dec["decision"] == "review"]
    verdicts, usage = adjudicate.adjudicate(adjudicate.build_prompts(inp, rules, pairs, band), cache=LLM_CACHE)
    final = adjudicate.apply(dec, verdicts, scorer.llm_accept_conf, scorer.llm_reject_conf)

    # A wrong accept is a same-spec alternate when the accepted BOM part's own catalog row has the same
    # subcategory, package and numeric attributes as the purchased catalog row.
    spec = {r.mpn.upper(): (r.subcategory, r.package, numbers_in(norm_desc(r.description))) for r in catalog.itertuples()}
    part_mpns = {}
    for mpn, part in owner.items():
        part_mpns.setdefault(part, set()).add(mpn)
    line_spec = dict(zip(p.index[:len(cat)], (spec[m.upper()] for m in catalog["mpn"])))

    def stage(d: pd.DataFrame) -> dict:
        is_cat = p["true_part"] != ""
        accept = d["decision"] == "accept"
        right = accept & (d["part"] == p["true_part"])
        wrong = accept & is_cat & ~right
        alternates = sum(any(spec.get(m) == line_spec[i] for m in part_mpns.get(d.at[i, "part"], ())) for i in d.index[wrong])
        neg = ~is_cat
        return dict(
            catalog_lines=int(is_cat.sum()), accepted=int((accept & is_cat).sum()), correct=int(right.sum()),
            wrong=int(wrong.sum()), wrong_same_spec_alternate=int(alternates), wrong_other=int(wrong.sum()) - int(alternates),
            in_review=int(((d["decision"] == "review") & is_cat).sum()), called_no_match=int(((d["decision"] == "no_match") & is_cat).sum()),
            precision=float(right.sum() / max((accept & is_cat).sum(), 1)), recall=float(right.sum() / max(is_cat.sum(), 1)),
            precision_counting_alternates=float((right.sum() + alternates) / max((accept & is_cat).sum(), 1)),
            no_match_lines=int(neg.sum()), no_match_correct=int(((d["decision"] == "no_match") & neg).sum()),
            no_match_in_review=int(((d["decision"] == "review") & neg).sum()), no_match_wrongly_accepted=int((accept & neg).sum()),
            no_match_accuracy=float(((d["decision"] == "no_match") & neg).sum() / max(neg.sum(), 1)))

    return dict(rules=rules["rule_method"].value_counts().to_dict(), classifier_stage=stage(dec), with_claude=stage(final),
                claude=dict(lines_sent=len(band), **usage))


def main():
    bom_lines = load_boms()
    parts, owner = build_parts(bom_lines)
    catalog = catalog_rows(set(owner))
    negatives = no_match_lines()
    scorer = scoring.Scorer.load("openai")
    report = dict(
        bom_files=int(bom_lines["project"].nunique()), bom_positions=len(bom_lines), bom_parts=len(parts), bom_mpns=len(owner),
        mpns_found_in_catalog=len(catalog), catalog_size=4_276_372, no_match_lines=len(negatives),
        zero_shot=f"classifier and cutoffs from the synthetic tuning split (accept at p >= {scorer.accept_at:.3f}, Claude pick at >= {scorer.llm_accept_conf})",
        conditions={"MPN visible in the purchase text": run_condition(parts, catalog, owner, negatives, scorer, True),
                    "MPN hidden": run_condition(parts, catalog, owner, negatives, scorer, False)})
    Path("out").mkdir(exist_ok=True)
    Path("out/eval_external_parts.json").write_text(json.dumps(report, indent=2))

    print(f"{report['bom_files']} BOM files, {report['bom_positions']} positions with description and MPN, {report['bom_parts']} unique parts; "
          f"{report['mpns_found_in_catalog']} of {report['bom_mpns']} MPNs found in the catalog; {len(negatives)} no-match lines")
    for name, c in report["conditions"].items():
        print(f"\n{name}  (rules: {c['rules']})")
        for label, m in (("rules + classifier", c["classifier_stage"]), ("rules + classifier + Claude", c["with_claude"])):
            print(f"  {label:<28} accepted {m['accepted']:>3} of {m['catalog_lines']}: {m['correct']} right, {m['wrong_same_spec_alternate']} same-spec alternates, "
                  f"{m['wrong_other']} wrong; {m['in_review']} in review, {m['called_no_match']} called no match | precision {m['precision']:.2f} "
                  f"(with alternates {m['precision_counting_alternates']:.2f}) recall {m['recall']:.2f}")
            print(f"  {'':<28} no-match lines: {m['no_match_correct']} of {m['no_match_lines']} right ({m['no_match_accuracy']:.1%}), "
                  f"{m['no_match_in_review']} in review, {m['no_match_wrongly_accepted']} wrongly accepted")
        print(f"  Claude: {c['claude']['lines_sent']} lines sent, {c['claude']['calls']} calls, ${c['claude']['cost_usd']:.2f}")


if __name__ == "__main__":
    main()
