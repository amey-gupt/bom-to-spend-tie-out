# BOM-to-Spend Tie-Out

Links a manufacturer's bills of materials (BOMs) to its purchasing records using messy CSV exports, with no system integration. Built at the Kearney PERLab x Velocity AI Hackathon (Corporate Track 3, BOM Cost Linkage).

**All data in this repo is synthetic.** The products, parts, suppliers and prices are invented, and the true matches are planted so accuracy can be measured.

## Status

Step 1 of 6 is done: the data generator. Matching, allocation and the app are not built yet.

## Generate the data

```
python3 -m src.generate
```

Standard library only, fixed seed, same output every run.

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
data/            generated CSVs
```
