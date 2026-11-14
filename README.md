# BOM-to-Spend Tie-Out

Links a manufacturer's bills of materials (BOMs) to its purchasing records using messy CSV exports, with no system integration. Built at the Kearney PERLab x Velocity AI Hackathon (Corporate Track 3, BOM Cost Linkage).

**All data in this repo is synthetic.** The products, parts, suppliers and prices are invented, and the true matches are planted so accuracy can be measured.

## Status

Steps 1 and 2 of 6 are done: the data generator, and the rules + embedding matcher with its evaluation. Scoring with a review queue, allocation and the app are not built yet.

## Results so far (held-out noise styles, 1,669 lines)

| Method | Precision | Recall | Dollar-weighted accuracy | Tie-out rate (true: 98.7%) |
|---|---|---|---|---|
| Naive exact part number | 100.0% | 21.8% | 27.6% | 28.2% |
| Rules only | 100.0% | 40.0% | 34.2% | 34.4% |
| Rules + embeddings | 95.2% | 86.9% | 85.7% | 85.4% |

The embedding similarity threshold (0.66) was chosen on the tuning split only. Reproduce with `uv run python -m src.evaluate`.

## Generate the data

```
python3 -m src.generate
```

The generator uses the standard library only, with a fixed seed, and gives the same output every run. The matcher needs `uv sync` first (pandas, sentence-transformers on CPU); embeddings are cached in `cache/`.

| File | Contents |
|---|---|
| `data/boms.csv` | 4 cordless drill models, 8 assemblies each, parts shared across models |
| `data/builds.csv` | units built per product per month, Jan 2025 to Jun 2026 |
| `data/purchases.csv` | purchase lines with noisy part numbers, descriptions, units and supplier names |
| `data/supplier_xref.csv` | partial supplier-part-number crosswalk (about 40% coverage) |
| `data/fx_rates.csv` | currency conversion to USD |
| `data/truth/ground_truth.csv` | per purchase line: true part, true bucket, noise style, tune/held-out split |
| `data/truth/duplicate_parts.csv` | internal part numbers that are the same physical part |
| `data/truth/planted_findings.csv` | planted supplier price gaps, duplicate parts and spares premiums |

The matcher may read `data/*.csv` only. `data/truth/` is for evaluation.

## Noise and the held-out split

Each purchase line gets an ID style (exact, revision/punctuation variant, supplier part number, plant-specific code, blank, typo) and a description style (exact, reordered, abbreviated, truncated, alternate units, supplier catalog wording, typo, Spanish). Lines whose style is plant code, ID typo, catalog wording, description typo or Spanish are marked `heldout` and must not be used for tuning, so accuracy on them is not just a measure of this generator.

Purchases also include lines that correctly match no BOM: service-only items, parts of a product line whose BOMs were not supplied, and indirect items.

## Layout

```
src/catalog.py   products, part categories, suppliers
src/noise.py     noise layer (independent of the backbone)
src/generate.py  backbone, purchase lines, ground truth
src/normalize.py load exports; normalize part numbers, units, currency, text
src/match.py     naive baseline, rules, embedding candidates (top 5)
src/evaluate.py  scores against data/truth/ (the only module that reads it)
data/            generated CSVs
```
