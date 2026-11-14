# BOM-to-Spend Tie-Out

Links a manufacturer's bills of materials (BOMs) to its purchasing records using messy CSV exports, with no system integration. Built at the Kearney PERLab x Velocity AI Hackathon (Corporate Track 3, BOM Cost Linkage).

**All data in this repo is synthetic.** The products, parts, suppliers and prices are invented, and the true matches are planted so accuracy can be measured.

## Status

Steps 1 to 5 of 6 are done: the data generator, the matcher (rules, embeddings, calibrated classifier, Claude on the review band), allocation with buckets and leads, and the app.

## Run the app

```
uv sync
uv run streamlit run app.py
```

Then open http://localhost:8501. The app reads the committed snapshot in `out/` only, so it needs no API keys and starts in about a second. It has seven tabs:

- **Overview:** tie-out rate, the four buckets, and each stage from a plain part-number lookup to the full pipeline.
- **Review queue:** uncertain lines, largest dollars first, with the suggested part, Claude's reason, and price and supplier evidence. Approve, reject or mark as not a BOM part; tie-out, buckets, product cost and leads update at once. The materiality slider is in the sidebar.
- **Product cost:** BOM standard versus estimated actual per unit, and the parts driving the difference.
- **Leads:** price differences shown as ranges, with the purchase lines behind each one. They are leads, not savings.
- **Evaluation:** held-out versus tuning results, accuracy by noise style, Claude cost, and limits.

- **Trends and inventory:** tie-out and estimated actual versus standard by quarter, and an estimated inventory view (bought minus built). Both are rough estimates.
- **Upload:** run the matcher on your own CSV or Excel files (see below).

The sidebar has the data switch, the materiality slider, a date range, and the reviewer crosswalk.

### Reviewer crosswalk

Every approve, reject or not-a-BOM-part decision is written to `crosswalk.csv` (purchase text and part-number field to BOM part). It applies at once to every line with the same text, persists across sessions, and is the first rule on the next pipeline run. Undo, clear and download are in the sidebar. The file is local working state and is gitignored. Leads and product cost also have download buttons.

### Your own files

The Upload tab takes a BOM, purchase lines and production volumes as CSV or Excel. Column names do not need to match: Claude reads the headers and 20 sample rows of each file and proposes a mapping with a confidence per column, which you confirm or fix before anything runs. This needs `ANTHROPIC_API_KEY` (and uses `OPENAI_API_KEY` for embeddings if present, otherwise a local model), in the environment or in `.env.local`.

Uploaded data is matched with rules, embeddings and Claude only. The classifier is skipped: it has to be trained on a company's own confirmed matches, and the external tests show one trained elsewhere does not transfer. There is no ground truth for uploaded files, so the app reports how lines were decided, not accuracy.

`samples/second_company/` holds a small second sample to try this with: a slice of the synthetic data exported with different column names, US dates, a different order-type vocabulary, and the purchase lines as Excel. "Load the second sample" in the Upload tab uses it. On that slice the mapping was right for all 22 columns, and matching reached 100% precision and 97% recall against the planted truth (256 Claude calls, $2.00, about 85 seconds uncached).

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

Reproduce with `uv run python -m src.evaluate step2` and `uv run python -m src.evaluate step3` and `uv run python -m src.evaluate step4`. To rebuild `out/` from scratch, step 3 needs `OPENAI_API_KEY` for the embeddings (in the environment or in `.env.local`); the embedding cache is not committed. Claude's responses are committed under `cache/llm/`, so no Anthropic key is needed unless the data or prompts change.

## External tests (real data, zero-shot)

Nothing is retrained for these: the classifier and all cutoffs come from the synthetic tuning split. The downloaded data is not committed.

```
python3 -m src.external.fetch                 # about 400 MB download, 2.6 GB on disk, into data/external/
uv run python -m src.external.abt_buy         # needs OPENAI_API_KEY and ANTHROPIC_API_KEY
uv run python -m src.external.kitspace_jlc
```

**A. Abt-Buy product matching**, fixed test split from the University of Mannheim benchmark page (710 pairs, 109 matches):

| Method | Precision | Recall | F1 |
|---|---|---|---|
| Rules only | 0.97 | 0.68 | 0.80 |
| Rules + embeddings | 0.95 | 0.81 | 0.87 |
| Rules + embeddings + Claude | 0.98 | 0.91 | 0.94 |
| SVM (published on the page) | 0.96 | 0.71 | 0.81 |
| Random Forest (published on the page) | 0.95 | 0.77 | 0.85 |

The benchmark's validation split was used only to check the model-code rule and to set two similarity cutoffs. Claude judged 229 pairs ($1.04).

**B. Real electronic parts.** 250 BOM parts from 14 open-hardware Kitspace BOMs (24 were fetched; 14 have both a description and an MPN). 208 of their 354 MPNs are in the JLCPCB catalog (4.3M parts), and each becomes a purchase line with the LCSC number, catalog description and price. Truth is MPN equality. 500 State of California purchase-order lines are added as true no-match cases.

| Condition | Stage | Right | Same-spec alternate | Wrong | In review | Called no match | Precision | Recall | No-match lines right |
|---|---|---|---|---|---|---|---|---|---|
| MPN visible | Rules + classifier | 145 | 0 | 0 | 63 | 0 | 1.00 | 0.70 | 4.4% |
| MPN visible | + Claude | 163 | 0 | 1 | 5 | 39 | 0.99 | 0.78 | 99.6% |
| MPN hidden | Rules + classifier | 1 | 0 | 0 | 204 | 3 | 1.00 | 0.00 | 0.0% |
| MPN hidden | + Claude | 59 | 0 | 2 | 18 | 129 | 0.97 | 0.28 | 99.4% |

What this shows: the classifier, trained on synthetic drill parts, does not transfer and sends nearly every real line to review; rules and Claude carry the test. With the MPN hidden, recall is low because a BOM line like "Red LED 5mm" does not identify one catalog part, and the tool declines instead of guessing. Two or three no-match lines were wrongly accepted, all by the classifier.

Data sources and their terms, as checked on 2026-11-14:

| Source | What is used | License or terms found |
|---|---|---|
| [Mannheim Benchmark Matching Tasks](http://data.dws.informatik.uni-mannheim.de/benchmarkmatchingtasks/index.html) | Abt-Buy records and fixed splits | No license is stated on the page. The data originates from the Leipzig database group's benchmark. |
| Kitspace `1-click-bom.tsv` files on GitHub | BOM lines from 24 repositories (listed in `src/external/fetch.py`) | Varies by repository. Among those GitHub reported: MIT, GPL-2.0, GPL-3.0; several have no detected license, and 12 lookups failed. |
| JLCPCB parts catalog via [yaqwsx/jlcparts](https://github.com/yaqwsx/jlcparts) | Catalog rows for the matched MPNs | The mirror's code is MIT. The catalog content is JLCPCB's and carries no open license that I found. |
| [California Purchase Order Data 2012-2015](https://data.ca.gov/dataset/purchase-order-data) | 500 sampled lines | The dataset's license field on data.ca.gov is empty. |

Because several of these have no clear license, none of the downloaded data, and none of Claude's cached answers about it, is committed. Only aggregate results are.

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
src/pipeline.py  rules + embeddings + Claude for uploaded files (no classifier)
src/mapping.py   Claude column mapping and value cleaning for uploaded files
src/crosswalk.py reviewer decisions, reused as the first rule
src/make_sample.py writes samples/second_company/
src/external/    fetch script and two zero-shot tests on real data
app.py           Streamlit app; reads out/ only
out/             committed snapshot of results the app reads
data/            generated CSVs
```
