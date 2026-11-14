# BOM-to-Spend Tie-Out

Links a manufacturer's bills of materials (BOMs) to its purchasing records using messy CSV exports, with no system integration. Built at the Kearney PERLab x Velocity AI Hackathon (Corporate Track 3, BOM Cost Linkage).

**All data in this repo is synthetic.** The products, parts, suppliers and prices are invented, and the true matches are planted so accuracy can be measured.

## Status

Steps 1 to 4 of 6 are done: the data generator, the matcher (rules, embeddings, calibrated classifier, Claude on the review band), and allocation with buckets and leads. The app is not built yet.

## Results so far (held-out noise styles, 1,669 lines, $9.8M, true tie-out rate 98.7%)

| Stage | Tied-out spend | Precision of accepted matches | Recall | Review queue | Wrong-match dollars |
|---|---|---|---|---|---|
| Naive exact part number | 28.2% | 100.0% | 21.8% | n/a | $0 |
| Rules only | 34.4% | 100.0% | 40.0% | n/a | $0 |
| Rules + embeddings, fixed threshold | 85.4% | 95.2% | 86.9% | n/a | $86,119 |
| Rules + classifier | 77.7% | 100.0% | 87.0% | 287 lines, $2.23M | $0 |
| Rules + classifier + Claude | 98.0% | 100.0% | 99.1% | 13 lines, $48K | $0 |

How the numbers are kept honest:

- The classifier is trained on the tuning split only. Its accept threshold is set for at least 99% precision using leave-one-noise-style-out cross-validation, so the threshold is chosen on styles the model had not seen.
- Claude sees only the review band (647 lines, 451 distinct items across both splits), with the five candidates. Its confidence cutoffs are set on the tuning split.
- Claude cost: 451 calls on `claude-opus-5-5` used 623,414 input and 51,015 output tokens, $3.51 at list price. The same calls on `claude-haiku-5-5` cost $0.11, but its picks never reached 99% precision on the tuning split, so none were auto-accepted and the held-out review queue stayed at 216 lines ($2.13M).
- On the tuning split the full pipeline is not perfect: 99.95% precision and $18,953 of wrong-match spend.
- All of this is measured on synthetic noise. Real exports will be messier in ways the generator does not model.

Embedding backends on the same held-out lines (rules + classifier, before Claude): OpenAI `text-embedding-3-small` puts the true part first for 95.6% of open lines and auto-accepts 79.3% of lines; local `all-MiniLM-L6-v2` gets 89.9% and 74.5%. Both reach 100% precision on accepted matches.

## Buckets, product cost and leads (all 4,421 lines, $29.3M)

| Bucket | Tool | Truth |
|---|---|---|
| Production | 81.8% | 87.3% |
| Spares | 6.1% | 6.2% |
| Warranty | 1.1% | 1.3% |
| Unassigned | 11.0% | 5.2% |

The tool reports an 89.0% tie-out rate against a true 94.8%. The gap is deliberate: any match that only Claude vouched for and that is worth more than $25,000 goes to a person, which holds 20 lines and about $1.6M in the review queue. With that rule off, tie-out is 94.4% and the bucket is right for 99.1% of spend.

Leads against the planted findings: supplier price gaps 46 found for 45 planted (98% precision, 100% recall), duplicate part numbers 8 of 8, spares premiums 15 of 15. Every lead is shown as a range, from half the gap closed to all of it, and is a lead, not a booked saving.

Product cost is an estimate: matched production spend is split across products by BOM quantity x units built per quarter, and shown beside the BOM standard cost.

Reproduce with `uv run python -m src.evaluate step2` and `uv run python -m src.evaluate step3` and `uv run python -m src.evaluate step4`. Step 3 needs `OPENAI_API_KEY` and `ANTHROPIC_API_KEY` in the environment or in `.env.local` on the first run; every API response is cached under `cache/`.

## Generate the data

```
python3 -m src.generate
```

The generator uses the standard library only, with a fixed seed, and gives the same output every run. The matcher needs `uv sync` first (pandas, sentence-transformers on CPU); embeddings are cached in `cache/`.

| File | Contents |
|---|---|
| `data/boms.csv` | 4 cordless drill models, 8 assemblies each, parts shared across models, with standard cost |
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
src/score.py     pair features, calibrated classifier, accept / review / no match
src/adjudicate.py Claude on the review band: part or "none", confidence, one-sentence reason
src/allocate.py  buckets, materiality rule, allocation to products, leads
src/evaluate.py  trains and scores against data/truth/ (the only module that reads it)
data/            generated CSVs
```
