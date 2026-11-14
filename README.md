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

Then open http://localhost:8501. The app reads the committed snapshot in `out/` only, so it needs no API keys. The header shows the tie-out rate, the lines waiting for a decision, total spend, and how far reviewing the largest lines would move tie-out. It has six tabs:

- **Overview:** the four buckets, a spend-flow chart from how each line was decided to the bucket it landed in (it updates after every review decision), tie-out by quarter, and each stage from a plain part-number lookup to the full pipeline.
- **Review:** the queue on the left, grouped so lines with identical text are one item, and the detail on the right: decision buttons, what approving would settle, the PO text and BOM description with shared words in bold, Claude's reason, and price and supplier evidence. Saved decisions (undo, clear, download) are at the bottom. Keyboard: A approves, X rejects, N marks not a BOM part.
- **Product cost:** BOM standard versus estimated actual per unit, a network of shared parts with the products that use them and the suppliers that sell them (click a part for its products and price per supplier), the engineering changes, the parts driving the difference, the BOM in effect on any date, the cost trend by quarter and an estimated inventory view.
- **Savings leads:** price and quantity gaps shown as ranges, by type, with the purchase lines behind each one. They are leads, not booked savings.
- **Evidence:** three headline results, then the full evaluation in collapsed sections, and the limits.
- **Your data:** run the matcher on your own CSV or Excel files in three steps (see below). Uploads are limited to 5 MB per file.

The sidebar (collapsed by default) has the data switch, the "Needs sign-off above" threshold and a date range.

The flow chart and the network use ECharts through the `streamlit-echarts` component, which animates changes. If that component cannot be imported, the app falls back to static Altair charts. This has been checked locally in a headless browser, not on Streamlit Community Cloud. The font is Inter (SIL Open Font License), served from `static/` so the app needs no network for it.

### Saved decisions

Every approve, reject or not-a-BOM-part decision is written to `crosswalk.csv` (purchase text and part-number field to BOM part). It applies at once to every line with the same text, persists across sessions, and is the first rule on the next pipeline run. Undo, clear and download are at the bottom of the Review tab. The file is local working state and is gitignored. Leads and product cost also have download buttons.

### Your own files

The Your data tab takes a BOM, purchase lines and production volumes as CSV or Excel. Column names do not need to match: Claude reads the headers and 20 sample rows of each file and proposes a mapping with a confidence per column, which you confirm or fix before anything runs. This needs `ANTHROPIC_API_KEY` (and uses `OPENAI_API_KEY` for embeddings if present, otherwise a local model), in the environment or in `.env.local`.

Uploaded data is matched with rules, embeddings and Claude only. The classifier is skipped: it has to be trained on a company's own confirmed matches, and the external tests show one trained elsewhere does not transfer. There is no ground truth for uploaded files, so the app reports how lines were decided, not accuracy.

`samples/second_company/` holds a small second sample to try this with: a slice of the synthetic data exported with different column names, US dates, a different order-type vocabulary, and the purchase lines as Excel. "Try the second sample" in the Your data tab uses it. It includes the multi-level BOM columns and the continuous materials. On that slice the mapping was right for all 25 columns, and matching reached 99.7% precision and 97.5% recall against the planted truth (259 Claude calls, $2.03, about 85 seconds uncached).

## Deploying to Streamlit Community Cloud

I have not deployed this myself; the hosted mode below was tested locally with the same secrets.

1. Push the repository to GitHub.
2. On share.streamlit.io choose **Create app**, pick the repository and branch, and set the main file to `app.py`. Choose Python 3.12.
3. Under **Advanced settings, Secrets**, paste:

   ```toml
   HOSTED = "true"
   ACCESS_CODE = "choose-a-code"
   OPENAI_API_KEY = "..."
   ANTHROPIC_API_KEY = "..."
   ```

4. Deploy.

| Secret | What it does |
|---|---|
| `HOSTED` | Turns on hosted mode: the access code, the limits below, and no writes to disk. |
| `ACCESS_CODE` | Unlocks the Your data tab. Without it that tab is switched off; the other tabs stay open to everyone. Compared in constant time. |
| `OPENAI_API_KEY` | Embeddings for uploaded files. The local embedding model is not installed on the host. |
| `ANTHROPIC_API_KEY` | Column mapping and matching for uploaded files. |

The second sample needs neither key: its Claude answers (`cache/llm_sample/`) and its embeddings (`cache/sample_embeddings.npz`) are committed, so "Try the second sample" makes no API calls. It still needs the access code, because the whole tab does.

Keys are read from Streamlit secrets first, then from the environment or `.env.local`, and are passed straight to the API clients; they are never printed or logged. `.streamlit/secrets.toml` is gitignored.

Hosted limits, each shown on the page when it is hit: 5 MB per file, 2,000 purchase lines per run, 150 Claude calls per run (the largest lines go first; the rest wait in Review), and 3 runs per browser session.

What is kept where on the host: uploaded files, the matched results and saved decisions live in the browser session's memory only. Nothing derived from uploaded text is written to disk, and there is no shared decisions file. Uploaded text is sent to OpenAI and Anthropic for matching, as the tab says.

Dependencies: Community Cloud uses `uv.lock` when a repository has one, ahead of `requirements.txt`. The default install in both is the same small set, without torch or sentence-transformers. Those, with scikit-learn, are in the optional `local` extra, needed only to rebuild the results: `uv sync --extra local`.

## Results so far (held-out noise styles, 1,708 lines, $10.1M, true tie-out rate 98.8%)

| Stage | Tied-out spend | Precision of accepted matches | Recall | Review queue | Wrong-match dollars |
|---|---|---|---|---|---|
| Naive exact part number | 27.5% | 100.0% | 21.3% | n/a | $0 |
| Rules only | 34.2% | 100.0% | 40.1% | n/a | $0 |
| Rules + embeddings, fixed threshold | 84.1% | 95.3% | 86.3% | n/a | $86,119 |
| Rules + classifier | 77.1% | 100.0% | 83.7% | 328 lines, $2.35M | $0 |
| Rules + classifier + Claude | 97.9% | 100.0% | 98.8% | 18 lines, $55K | $0 |

How the numbers are kept honest:

- The classifier is trained on the tuning split only. Its accept threshold is set for at least 99% precision using leave-one-noise-style-out cross-validation, so the threshold is chosen on styles the model had not seen.
- Claude sees only the review band (614 lines, 392 distinct items across both splits), with the five candidates. Its confidence cutoffs are set on the tuning split.
- Claude cost: 392 calls on `claude-opus-5-5` used 543,378 input and 47,275 output tokens, $3.12 at list price. The same calls on `claude-haiku-5-5` cost $0.10, but its picks never reached 99% precision on the tuning split, so none were auto-accepted and the held-out review queue stayed at 274 lines ($2.25M).
- "No match" calls are the weak spot: on held-out, 96.6% of them are right, and $13,536 of spend that does have a BOM part is dropped. This figure is after the item-level materiality rule below, which was added after seeing held-out results.
- On the tuning split the full pipeline is not perfect: 99.95% precision and $18,953 of wrong-match spend.
- All of this is measured on synthetic noise. Real exports will be messier in ways the generator does not model.

Embedding backends on the same held-out lines (rules + classifier, before Claude): OpenAI `text-embedding-3-small` puts the true part first for 95.7% of open lines and auto-accepts 76.4% of lines; local `all-MiniLM-L6-v2` gets 90.2% and 74.8%. Both reach 100% precision on accepted matches. The backend for the Claude stage is chosen on the tuning split, by how often the true part is the top candidate.

## Buckets, product cost and leads (all 4,487 lines, $29.7M)

| Bucket | Tool | Truth |
|---|---|---|
| Production | 81.5% | 87.4% |
| Spares | 6.0% | 6.1% |
| Warranty | 1.1% | 1.3% |
| Unassigned | 11.4% | 5.1% |

The tool reports an 88.6% tie-out rate against a true 94.9%, with 100 lines ($1.91M) in review. The gap is deliberate: any match that only Claude vouched for and that is worth more than $25,000 goes to a person, which holds 20 lines and about $1.6M in the review queue. With that rule off, tie-out is 93.9% and the bucket is right for 98.6% of spend.

Leads against the planted findings: supplier price gaps 46 found for 45 planted (98% precision, 100% recall), duplicate part numbers 8 of 8, spares premiums 15 of 15, purchased-above-requirement 1 of 2 planted, plus 2 more on the parts with an engineering change (see below). Every lead is shown as a range, from half the gap closed to all of it, and is a lead, not a booked saving.

Product cost is an estimate: matched production spend is split across products by gross BOM quantity x units built per quarter, and shown beside the BOM standard cost.

## Effective-dated BOM revisions

- **BOM columns:** `effective_from` and `effective_to`; blank means open-ended.
- **Two engineering changes are in the data,** on the BOM only. From 2025-10-01 a screw on the DR-18 goes from 4 per unit to 2. From 2026-01-01 a resistor position on the DR-18H switches to another resistor already used in the same assembly. No purchase line was added, removed or altered, as if purchasing had not caught up with engineering.
- **What uses the dates:** allocation, standard cost per unit and the purchased-versus-required lead all use the BOM rows in effect in each build month (taken as of the first of the month). Standard cost per unit is the average over the period's builds.
- **In the app:** the Product cost tab has a date picker that shows the BOM in effect on that date, with changed rows marked and retired or future rows listed.

Checked after the change: `purchases.csv`, `builds.csv`, `supplier_xref.csv` and the ground truth are byte-identical; steps 2 and 3 give identical results and `out/decisions.csv` is identical; no new Claude calls were made.

What moved in step 4:

| | Before | After |
|---|---|---|
| Tie-out rate, buckets, review queue | 88.6%, 100 lines | unchanged |
| DR-18 standard cost per unit | $90.48 | $90.46 |
| DR-18H standard cost per unit | $87.85 | $87.84 |
| Allocated production spend | | DR-18 -$61, DR-18H -$805, DR-20 +$867 |
| Purchased-above-requirement leads | 1 (grease) | 3: grease, plus the two changed parts |
| Other leads (69) | | unchanged, same ranges |

The two new leads are real gaps in this data: purchasing kept buying at the old rate. The screw shows +25% ($880 to $1,759) and the replaced resistor +18% ($1,017 to $2,034). They are not in the planted over-purchase list, so that lead's precision against the planted list reads 33%; the evaluation reports them separately as explained by an engineering change.

## Materiality by item

**This rule was added after seeing held-out results**, so its "after" numbers are not a clean held-out measurement. The earlier results showed the classifier closing some material items as "no match" when they did have a BOM part.

Spend is summed over every line with the same item text. If the total is above the materiality threshold ($25,000), the classifier may not close the item as "no match" on its own: the lines go to Claude, and to a reviewer if Claude is not confident either way. A confident "none" from Claude still closes them.

| Held-out split | Before | After |
|---|---|---|
| Precision of accepted matches | 100.00% | 100.00% |
| Recall | 98.8% | 98.8% |
| Review queue | 11 lines, $25K | 18 lines, $55K |
| No-match calls that are right | 92.3% | 96.6% |
| Spend with a BOM part dropped as no match | $43,572 | $13,536 |
| Wrong-match dollars | $0 | $0 |

The rule kept 80 lines open across both splits and raised the lines sent to Claude from 534 to 614 ($2.76 to $3.12). The tuning split is unchanged. The $13,536 still dropped is lines where Claude itself answered "none" with moderate confidence.

## Experiment: does the crosswalk cut review work over time?

Purchases are processed one quarter at a time, in date order. Each quarter the full pipeline runs with the crosswalk built so far; then a simulated reviewer clears that quarter's review queue from the truth, and those decisions are saved. The comparison is the same run with no crosswalk. Reproduce with `uv run python -m src.evaluate crosswalk`.

| Quarter | Lines | Settled by crosswalk | Claude calls (with / without) | Review lines (with / without) | Review spend (with / without) | Precision (with / without) |
|---|---|---|---|---|---|---|
| 2025Q1 | 719 | 0 | 101 / 101 | 14 / 14 | $352K / $352K | 99.83% / 99.83% |
| 2025Q2 | 706 | 16 | 86 / 100 | 0 / 16 | $0 / $276K | 100% / 100% |
| 2025Q3 | 756 | 13 | 89 / 104 | 5 / 15 | $5K / $358K | 100% / 100% |
| 2025Q4 | 770 | 16 | 77 / 94 | 4 / 19 | $36K / $514K | 100% / 100% |
| 2026Q1 | 766 | 22 | 83 / 99 | 2 / 18 | $3K / $202K | 100% / 99.84% |
| 2026Q2 | 770 | 20 | 69 / 84 | 1 / 15 | $18K / $202K | 100% / 100% |
| Total | 4,487 | 87 | 505 / 582 | 26 / 97 | $414K / $1.90M | |

- Review work falls by about three quarters: 26 lines against 97 over six quarters, and $414K against $1.90M.
- Claude calls fall by 13% (505 against 582, $3.98 against $4.65). Most Claude calls are for lines Claude settles itself, which never reach a reviewer or the crosswalk.
- All 87 lines the crosswalk settled were right.
- **Item text repeats by construction in this data.** The generator reuses one item text per plant, supplier and part about 80% of the time, so the size of this effect says little about a real company's exports. The classifier used was also trained on tuning-split lines from all six quarters.

## Units of measure, scrap and multi-level BOMs

- **BOM columns:** `uom`, `scrap_pct`, `parent_part` and `make_buy`. The gearbox and motor are in-house subassemblies with their own components; the gearbox carries 1% scrap.
- **Continuous materials:** steel sheet by area, magnet wire by length, grease and potting resin by mass, each with scrap, bought in assorted units (KG, LB, FT, FT2 and so on). 66 purchase lines.
- **Unit conversion:** purchase quantities are converted to the BOM unit within the same dimension (713 accepted lines, including boxes to each). A line bought in another dimension, such as wire by weight against a BOM in metres, cannot be converted without a factor: it goes to review with that reason. All 4 such lines were caught and no other line was flagged. On a purchase line the unit `M` is read as "per thousand" unless the matched part is measured in length.
- **Flattening:** the BOM tree is exploded to bought parts before allocation. Gross requirement is net quantity divided by (1 - scrap), multiplied down the tree. A cycle or a parent that is not on the BOM raises an error that names the path.
- **Purchased versus required:** a lead when production purchases over the trailing 12 months exceed the gross requirement by more than 15%. It is shown as a gap and no cause is assigned. Duplicate part numbers are pooled first, otherwise one half of each pair looks over-bought.
- **What it found:** two materials were planted at about 1.3x their requirement. Grease is found (+30%, a $2,840 to $5,681 lead). Magnet wire is missed: two of its lines were bought by weight and are held for a unit conversion, so its counted purchases fall below the threshold.

Reproduce (after `uv sync --extra local`) with `uv run python -m src.evaluate step2` and `uv run python -m src.evaluate step3` and `uv run python -m src.evaluate step4`. To rebuild `out/` from scratch, step 3 needs `OPENAI_API_KEY` for the embeddings (in the environment or in `.env.local`); the embedding cache is not committed. Claude's responses are committed under `cache/llm/`, so no Anthropic key is needed unless the data or prompts change.

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
| MPN visible | Rules + classifier | 145 | 0 | 0 | 54 | 9 | 1.00 | 0.70 | 54.8% |
| MPN visible | + Claude | 159 | 0 | 1 | 5 | 43 | 0.99 | 0.76 | 99.6% |
| MPN hidden | Rules + classifier | 1 | 0 | 0 | 104 | 103 | 1.00 | 0.00 | 37.0% |
| MPN hidden | + Claude | 34 | 0 | 1 | 13 | 160 | 0.97 | 0.16 | 99.4% |

What this shows: the classifier, trained on synthetic drill parts, does not transfer; rules and Claude carry the test. How it fails depends on the training run: an earlier classifier sent nearly every real line to review, while the current one calls about half the hidden-MPN lines "no match" outright, which cut recall after Claude from 0.28 to 0.16. With the MPN hidden, recall is low because a BOM line like "Red LED 5mm" does not identify one catalog part, and the tool declines instead of guessing. Two or three no-match lines were wrongly accepted, all by the classifier.

Data sources and their terms, as checked on 2026-11-14:

| Source | What is used | License or terms found |
|---|---|---|
| [Mannheim Benchmark Matching Tasks](http://data.dws.informatik.uni-mannheim.de/benchmarkmatchingtasks/index.html) | Abt-Buy records and fixed splits | No license is stated on the page. The data originates from the Leipzig database group's benchmark. |
| Kitspace `1-click-bom.tsv` files on GitHub | BOM lines from 24 repositories (listed in `src/external/fetch.py`) | Varies by repository. Among those GitHub reported: MIT, GPL-2.0, GPL-3.0; several have no detected license, and 12 lookups failed. |
| JLCPCB parts catalog via [yaqwsx/jlcparts](https://github.com/yaqwsx/jlcparts) | Catalog rows for the matched MPNs | The mirror's code is MIT. The catalog content is JLCPCB's and carries no open license that I found. |
| [California Purchase Order Data 2012-2015](https://data.ca.gov/dataset/purchase-order-data) | 500 sampled lines | The dataset's license field on data.ca.gov is empty. |

Because several of these have no clear license, none of the downloaded data, and none of Claude's cached answers about it, is committed. Only aggregate results are.

## Experiment: self-training without truth labels

Can the classifier be trained on a dataset's own rule-matched lines, with no confirmed labels? Positives are the lines a rule matched, negatives are their other top-5 candidates, the part-number features are hidden, and the matched number is stripped from the text. A line's own price and supplier are also left out of the evidence for its own part.

```
uv run python -m src.evaluate selftrain
uv run python -m src.external.selftrain_parts     # needs the external data
```

**(a) Synthetic data, held-out split** (self-trained on 2,254 rule-matched lines from the whole dataset):

| Classifier | Stage | Precision | Recall | Review queue | Wrong-match dollars |
|---|---|---|---|---|---|
| Truth-trained | Rules + classifier | 100.00% | 83.7% | 328 lines, $2.35M | $0 |
| Truth-trained | + Claude | 100.00% | 98.8% | 18 lines, $55K | $0 |
| Self-trained | Rules + classifier | 96.51% | 95.8% | 145 lines, $1.01M | $75,999 |
| Self-trained | + Claude | 96.60% | 98.4% | 16 lines, $37K | $75,999 |

**(b) Real parts** (trained on 70 rule-matched lines from half the parts, of which 24 gave a usable positive; tested on 97 lines from the other half with the part number hidden, plus 500 no-match lines):

| Classifier | Stage | Right | Wrong | Precision | Recall | Review queue | No-match lines wrongly accepted |
|---|---|---|---|---|---|---|---|
| Transferred from synthetic | Rules + classifier | 0 | 0 | n/a | 0.00 | 361 lines | 3 |
| Transferred from synthetic | + Claude | 17 | 0 | 1.00 | 0.18 | 8 lines | 3 |
| Self-trained | Rules + classifier | 2 | 0 | 1.00 | 0.02 | 595 lines | 0 |
| Self-trained | + Claude | 26 | 1 | 0.96 | 0.27 | 10 lines | 0 |

The catalog has no quantities, so wrong-match dollars cannot be computed for (b).

Self-training is not a shortcut. On the synthetic data it accepts more lines but loses the 99% precision guarantee: rule-matched lines always have a true part, so the model never sees a line that should stay unmatched, and its own cross-validated precision estimate (99.3%) is optimistic. On real parts the training set is too small and too weak, the accept threshold collapses to 1.0, and recall moves from 0.00 to 0.02. The classifier needs reviewer-confirmed matches, which is what the crosswalk collects.

## Generate the data

```
python3 -m src.generate
```

The generator uses the standard library only, with a fixed seed, and gives the same output every run. The matcher needs `uv sync` first (pandas, sentence-transformers on CPU); embeddings are cached in `cache/`.

| File | Contents |
|---|---|
| `data/boms.csv` | 4 cordless drill models, 8 assemblies each, parts shared across models; standard cost, unit of measure, scrap, parent part, make/buy and effective dates |
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
