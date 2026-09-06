# Pocket Vakil — Findings

Things that cost time to discover and would cost time to rediscover.

Every Open India Law number below was measured on 2026-09-06 against snapshot `v2026.08`.
Each carries its sample size. "Complete scan" = every row of the file(s) named.
Extrapolations carry a 95% CI from between-row-group spread.

**All token counts are provisional.** They use GPT-2's 50k BPE as a stand-in — the
project tokenizer does not exist yet. See *Decision 4*.

---

## Locked

**Compute** (OLM planner, do not exceed): 300M params · context 2048 · 6B tokens
processed · ~2B unique × 3 epochs · 2 × 20h H100.
Check: 6ND = 1.08e19 FLOPs → ~10h at 30% MFU.
Arch: d_model 1024, 24 layers, 16 heads.

**Task:** situation description in, subset of 100 IPC sections out. Closed set, cannot
abstain. Trained and measured on ILSI court-fact text; accepts layperson phrasing at
demo time but that performance is unmeasured and expected lower (ILSIC, EACL 2026).

**Output format:** generative. 100 special tokens (`<ipc_379>`), targets sorted
ascending + EOS. Report parse-failure rate, don't repair it.

**Corpus:** Open India Law `v2026.08`, SC + Allahabad/Bombay/Calcutta/Delhi/MP/Madras,
`year <= 2020`, English only. No criminal filter, no `section_type` filter.

---

## Decisions

**1. Calcutta stays in.** It yields 5.4M tokens — 0.14% of the 7-court total — because
98% of its rows are post-2020 (both sampled row groups agree: 98%/98%). Kept for ILSI
court coverage, not for volume.
*Tension, so it isn't rediscovered:* the `year <= 2020` cutoff exists to avoid teaching
IPC and BNS numbering for identical conduct, but it guts the court distribution ILSI
draws from. Calcutta is the extreme case. Revisiting the cutoff means revisiting this.

**2. Corpus mix: cap the two largest courts only.** Bombay and Madras are capped at pack
time. The other five are taken whole (~1.39B combined); the remaining ~610M to the 2B
target is split evenly between the two capped courts.

| Court | Full yield | Capped target | Share of 2B |
|---|---:|---:|---:|
| Supreme Court | 241.7M | 241.7M (whole) | 12.1% |
| Allahabad | 344.4M | 344.4M (whole) | 17.2% |
| Bombay | 1,409.9M | **305.0M** | 15.3% |
| Calcutta | 5.4M | 5.4M (whole) | 0.3% |
| Delhi | 473.1M | 473.1M (whole) | 23.7% |
| Madhya Pradesh | 323.3M | 323.3M (whole) | 16.2% |
| Madras | 1,016.1M | **305.0M** | 15.3% |
| **Total** | **3,814.0M** | **1,997.9M** | 100% |

Consequence to be aware of: capping only the top two makes **Delhi the largest single
contributor** at 23.7%.

**Cap at pack time, not extraction.** Extract every court in full so the mix can be
changed without re-extracting — extraction is the slow half.

**3. Reassembly overlap trim: FIXED** in `build_corpus.py` (2026-09-06). Strip the header
per chunk → trim in that coordinate space → apply the rest of the cleaning once to the
joined document. **The ordering is load-bearing; do not reorder it back.** Evidence and
post-fix numbers under *Chunk geometry* below.

**5. Full extraction is the default; `process()` streams.** `--hc-groups` now defaults to
all row groups, with `--hc-groups N` kept as the smoke-test sampler (`pick_groups`
unchanged). This required restructuring `process()` to write documents as they complete:
buffering a whole file measured **~9.4 GB of RAM for Bombay** on a 16 GB machine, and the
old default only avoided it by sampling 2 row groups.

Cases **do** straddle row-group boundaries — rare (~0.03–0.06%) but real, and one measured
case (`DLHC011384132013`) had 13 of its 22 chunks in the previous row group. Straddlers are
carried forward and emitted once complete; truncating them would be exactly the silent
corruption *Decision 3* removed. Completeness test is `len(chunks) == total_chunks`, which
is safe: that field is never null, never disagrees within a case, and is never exceeded
(verified across 37k+ cases in 4 files). Anything still incomplete at end-of-file is
emitted with what exists — that matches the old behaviour and loses nothing.

Verified equivalent: identical document sets **and identical document contents** against
the pre-restructure code on SC row groups 28+29 and Delhi 39+40 (40,000 rows each,
re-chunked to 8 row groups). Peak memory 99→50 MB and 181→42 MB; the new peak is flat in
file size, the old grows. Output *line order* now follows completion rather than
first appearance — the corpus is a set of documents, so this is immaterial.

**4. chars/token to be re-measured** with the project's own BPE, after the 16k-vs-32k
vocab decision and before packing. The 4.24 figure below is GPT-2's 50k BPE and will not
transfer — an in-domain 32k vocab should compress legal boilerplate better, a 16k vocab
worse. Every token total in this file inherits that ratio and is provisional until then.

---

## Open India Law — measured

**Access:** read parquet directly via `HfFileSystem` + `pyarrow` range requests. No full
download needed; a single row group is a 5–12s read.

| Documented | Measured | Sample |
|---|---|---|
| `text_original` is clean | Identical to `text`, **20,000/20,000**. Both carry the header. | SC row group 28, complete |
| 628,863 central sections | **74,484** | complete scan, `in_central_legislation.parquet` |
| IPC at `IND_central_1860_45` | absent. IPC is **`IND_REP_659_1860`** — 774 rows, 533 sections, `act_status=repealed`, `in_force=False` | complete scan |
| `section_type` has `facts` | **21 rows in 60,000 (0.035%)**. Dominant: `body` 49.7%, `paragraph` 40.6%, `section` 8.7% — structural, not rhetorical | SC row groups 5/28/50 |

Corpus totals confirmed exactly: **32,572,660 chunks**, **53,558,383,765 bytes (53.6 GB)**
across 26 judgment files (complete scan of all 26 parquet footers).

### Language — an SC-only problem

| Court | Rows (complete scan) | English |
|---|---:|---:|
| Supreme Court | 1,113,313 | **33.35%** (hin 27.62%, pun 27.53%) |
| Allahabad | 2,007,927 | **100.00%** |
| Bombay | 2,788,384 | **100.00%** |
| Calcutta | 535,453 | **100.00%** |
| Delhi | 1,064,459 | **100.00%** |
| Madhya Pradesh | 970,908 | **100.00%** |
| Madras | 2,610,596 | **100.00%** |

The "33.3% English" figure is **Supreme Court only**. It is not a corpus-wide property.

**So `language_code == "en"` is load-bearing for SC and a no-op for all six High Courts.**
Keep it — it costs nothing and guards the SC path.

`language_code` is reliable: `en` rows are 100.0% Latin script with median 34.8% English
stopwords, 0/300 suspect (n=200–300 per language, header stripped first, seeded). Non-`en`
rows are dominated by their native script. Note the header is English even on non-English
chunks, so strip it before any script check.

### Translations share `case_id` — filter language first

Complete scan of SC: **37,014 distinct `case_id`**, 36,753 distinct titles, of which
**36,158 (98.4%) appear in >1 language**; 36,412/37,014 case_ids carry >1 language;
35,596 have at least one English chunk.

Each language is a **parallel full chunk set** — its own `total_chunks`, its own
`chunk_index` from 0. So grouping by `case_id` before filtering language staples the
Hindi and Punjabi versions onto the English judgment.

**Filter language first, then group by `case_id`.** After the `en` filter,
`chunk_index` alone orders correctly: 0 collisions across 1,874 English cases, against
553 colliding case_ids (8.3%) pre-filter — exactly the multi-language ones.

**Again: load-bearing for SC, a no-op for the six High Courts**, which are single-language.

### Cleaning

Strip the `Case: … Section: …` header, `[TITLE]`/`[SECTION]` markers, and single-letter
lines (law-report margin markers).

- Header present on **60,000/60,000** rows — on *every* chunk, not just the first. The
  `build_corpus.py` regex strips 100%, 0 failures. **3.170% of characters**, mean 116.2 chars.
- Marker/margin/blank cleanup removes a further **median 37 chars** (p90 92, max 479) from
  **95.5%** of chunks. This is what breaks the overlap trim — see below.
- OCR damage is real (`D~CEMBER`, `No.1834of2013`), reportedly worse pre-2000.
  **Not quantified** — no error-rate measurement was run.

### Chunk geometry — and the overlap-trim bug

Measured on 18,175 consecutive chunk pairs, SC English:

- Spans: p10 2,679 · **p50 3,749** · p90 4,055 · max 4,096. (Not a flat 4096.)
- Overlaps: p10 −2 · **p50 195** · p90 475. (Not a flat 512.)
- **19.5% of consecutive pairs have a negative overlap — a gap.** Source text is simply
  absent between those chunks. Dataset property, not a code bug; nothing to fix.

**The bug — fixed 2026-09-06.** `char_start`/`char_end` are in **raw-minus-header**
coordinates — verified exactly on 2,994/3,000 chunks. The old `reassemble()` cleaned each
chunk *first*, then sliced `body[overlap:]` with that raw-coordinate offset. Cleaning had
already shrunk the text, so the slice cut past the overlap into real content. Worked
example — correct: `'Appellant was convicted by the Special Judge…'`; old output:
`'convicted by the Special Judge…'`.

Fix: strip header per chunk → trim → clean the joined document once. Contiguous seams are
joined with `""` (an exact trim leaves no gap to bridge, and a `"\n"` there would itself
break a word); gap seams keep `"\n"` so missing source text stays visible.

Re-measured on the same sample (1,644 multi-chunk complete SC English docs, row groups
5/28/50):

| | before | after |
|---|---:|---:|
| seams losing text | 68.3% (8,584/12,562) | **0.0%** (0/12,563) |
| seams cutting mid-word | 41.7% (5,238) | **0.0%** (0) |
| aggregate character loss | −0.241% | **−0.000%** |
| chars lost per seam | median 4, mean 10.9, max 215 | 0 / 0.0 / 0 |

Validated against an independent reconstruction (each chunk's body placed at its absolute
`char_start` in a buffer — does not use the trim logic at all): **398/400 documents
byte-identical**. The 2 exceptions are artifacts of that oracle, not the fix — in both the
fixed output is correct (buffer produced `Callphonex."Callphonex."` and `deposition..`;
fixed code produced each once).

**Overlapping chunk regions are not always byte-identical: 1,288 disagreements in 787,754
overlap-character comparisons (0.16%)**, mostly a margin letter present in one chunk's
rendering and absent from the other. First-write-wins reconstruction garbles those; taking
the later chunk's continuation verbatim, as the trim does, stays self-consistent.

Gap seams are unchanged and unfixable — 19.5% of pairs, source text genuinely absent.

### Court and year fields

**Courts are selected by filename, never by the `court` column.** All 7 target filenames
resolve. Do not "improve" this by filtering on the column — it is inconsistent:

- Bombay appears three ways: `Bombay High Court` (39,626), `High Court of Bombay` (292),
  `High Court of Mumbai` (7)
- Nulls in every HC sample, up to **2.7%** (Delhi, 1,085/40,000)

**`year` has zero nulls across all 11,091,040 rows** in the 7 target files (complete scan
of parquet column statistics; min year 1950). The falsy guard at `build_corpus.py:106`
(`if r["year"] and …`) is therefore a **no-op** — harmless, but it protects nothing.

Row groups are **not** date-ordered — per-group year medians and post-2020 fractions agree
closely across a file (Calcutta 98%/98%, Bombay 26%/27%, SC 12.1/12.2/15.3%) — and are
case-aligned (partial-document rate ~0.0%). This is why row-group sampling extrapolates
cleanly.

Post-2020 share varies enormously by court and drives the yields above: Calcutta 98%,
Allahabad ~69%, Madras ~52%, MP 46%, Delhi ~38%, Bombay ~26%, SC ~13%.

### Deduplication — measured and accepted

Pooled 14,898 reassembled documents across all 7 courts (seeded sample):

- Exact duplicates: **0.26%** (39 redundant docs)
- Near-duplicates (MinHash, 5-word shingles, Jaccard ≥ 0.8): **1.13%** (169 docs)
- **Zero cross-court**, exact or near. SC had zero near-dup pairs of any kind.

Unhandled in the pipeline, and accepted at this level — it will not distort ~2B tokens.
**Caveat: the cross-court negative is weak.** HC samples were 2 row groups each, so an SC
judgment duplicated inside an HC file could easily fall outside the sample.

### IPC text

Use ILSI's `sec_src` — its keys match the label vocabulary. OIL is a cross-check only:
filtering `in_force == True` drops **all 774** IPC rows (complete scan).

---

## Corpus yield

Extrapolated from row-group samples: SC 3 of 56 row groups (60,000 rows, 5.4%); each High
Court 2 row groups at n/4 and 3n/4 (40,000 rows each). Row groups are date-unbiased and
case-aligned (above), so these extrapolate cleanly; CIs are indicative given n=2–3.

**Supreme Court, complete pipeline** (en + `year<=2020` + clean + reassemble + min 500 chars):

```
1,113,313 chunks (complete scan) → 31,435 ± 2,210 judgments
1,041,117,019 ± 145,930,793 chars  ·  241.7M ± 34.9M tokens
mean 33,120 chars/doc
```

**7-court total: 3.81B tokens** (range 3.68–3.94B from per-row-group spread) — see the
per-court table under *Decision 2*.

These were measured with the pre-fix reassembly, which dropped ~0.24% of characters. The
fix recovers that, so true yields are ~0.24% higher — immaterial against a ±14% CI, and
not worth re-running the extrapolation for.

**Against the ~2B target that is 1.9x headroom. Extraction is not the critical path.**
SC alone (242M) is 8x short, so full High Court extraction is still required — but once
run, volume stops being the constraint. SC + Bombay + Madras alone clear 2B.

**The shipped default is wrong for the real run.** `--hc-groups 2` samples 2 row groups
per High Court and yields **0.32B — 6.3x short of target**. Must be changed to full
extraction before the extraction run.

### Length distribution

1,684 complete SC English judgments:

| | p10 | p50 | p90 | p99 | mean | max |
|---|---:|---:|---:|---:|---:|---:|
| chars | 8,265 | 22,803 | 58,111 | 188,276 | 33,120 | 1,928,420 |
| tokens | 1,991 | 5,356 | 13,667 | 44,071 | 7,737 | 441,541 |

Mean judgment ~7.7k tokens, so pretraining packs into 2048-token windows and **~89% of
judgments (~98% of tokens) exceed a single window**. But **10.7% of judgments fit
entirely** in 2048 tokens — those hold 1.91% of all tokens. The model does see some whole
judgments; it does not see most of them.

### chars/token

**4.240** measured on 600 random cleaned English chunks (2,052,153 chars, seeded);
**4.281** across 1,684 full documents (13.0M chars). Tokenizer: GPT-2 50k byte-level BPE.

**The 4.0 constant used in the earlier SC estimate was wrong** — stated here only because
a reader who divides characters by 4 will otherwise re-derive the same error. It
overstates token yield by **7.0%**, which is how the earlier "~260M tokens" figure arose
from a character count (1041.3M) that was itself accurate.

Re-measure with the project tokenizer before packing (*Decision 4*).

---

## Pending fixes

1. **Re-measure chars/token** with the project BPE after the vocab decision
   (*Decision 4*); every token figure here is provisional until then.

Done, both 2026-09-06: reassembly overlap trim (*Decision 3*); full-extraction default
plus the streaming restructure it required (*Decision 5*).

---

## ILSI

> **Still uninspected.** Nothing in this block has been measured — every figure is from
> the source papers/repo, carried over unverified. `data/ilsi/` is empty.

Zenodo DOI `10.5281/zenodo.6053791`. ~43k/10k/13k, 100 labels, multi-label.
JSONL: `id`, `text` (list of sentences — join first), `labels`.
Use train/dev/test + `sec_src`. Skip `type_map`, `label_tree`, `citation_network`,
metapath schemas — those are for LeSICiN's graph.

---

## Next

1. Full English extraction, SC + 6 HCs → 3.81B available, ~2B after the *Decision 2* caps.
   Raise `--hc-groups` to full first. No longer the critical path, but still the long pole
   in wall-clock terms.
2. Run `inspect_ilsi.py` → p90 fact length (does 2048 fit?), labels with <10 examples
   (caps macro-F1), exact label string format (drives the 100 special tokens).
3. Tokenizer on corpus + ILSI + `sec_src`, + 100 label tokens. Compare 16k vs 32k —
   32k costs ~33M embedding params (11% of budget).
4. Pre-tokenize and pack to shards **on laptop**. No tokenization on GPU time.

Session notes: checkpoint to persistent storage and test restore *during* session 1;
budget 1–2h of session 2 for setup and re-upload; 30-min smoke test before the long run.
