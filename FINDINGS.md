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

**Tokenizer: 32k byte-level BPE, tied embeddings. FROZEN** — `bpe_32000_final.json`,
fit to the Decision 2 capped mix. Reasoning in *Decision 4*. Retraining it invalidates
every shard and every checkpoint; the hash in `data/shards/manifest.json` is what ties
a run to its vocabulary.

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
time. The other five are taken whole; the remainder to the 2B target is split evenly
between the two capped courts.

**Updated 2026-09-06 against the completed extraction** (see *Corpus yield — measured*).
The five whole courts now sum to **1,297.0M**, not the ~1.39B the extrapolation implied,
so **the per-court cap rises from 305.0M to 351.5M**. Both capped courts still hold far
more than that (Bombay 1,069.6M, Madras 803.0M), so the mix is unchanged in shape.

| Court | Extrapolated | **Measured** | Capped target | Share of 2B |
|---|---:|---:|---:|---:|
| Supreme Court | 241.7M | **242.4M** | 242.4M (whole) | 12.1% |
| Allahabad | 344.4M | **340.3M** | 340.3M (whole) | 17.0% |
| Bombay | 1,409.9M | **1,069.6M** | **351.5M** | 17.6% |
| Calcutta | 5.4M | **4.8M** | 4.8M (whole) | 0.2% |
| Delhi | 473.1M | **432.7M** | 432.7M (whole) | 21.6% |
| Madhya Pradesh | 323.3M | **276.8M** | 276.8M (whole) | 13.8% |
| Madras | 1,016.1M | **803.0M** | **351.5M** | 17.6% |
| **Total** | **3,813.9M** | **3,169.5M** | **2,000.0M** | 100% |

Consequence to be aware of: capping only the top two still makes **Delhi the largest
single contributor**, now at 21.6%.

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

**4. Vocab: 32k, tied embeddings. DECIDED 2026-09-06 and frozen.** Both sizes were
trained and measured (`src/train_tokenizer.py`, *Tokenizer* below).

| | 16k | **32k** |
|---|---:|---:|
| chars/token, held-out corpus | 4.228 | **4.418** |
| chars/token, ILSI facts | 4.467 | **4.604** |
| embedding params, tied | 16.4M (5.1%) | **32.8M (9.8%)** |

**The deciding number is that 16k gave 4.228 on corpus — no better than GPT-2's 4.24.**
An in-domain 16k vocabulary bought *nothing* on the pretraining corpus over a generic
50k one, so the whole premise of a domain vocab fails at that size. 32k's 4.418 is a
genuine 4.5% gain, and that is what justifies the extra 16.4M parameters at 9.8% of the
budget tied. **Tied, not untied** — untied would be 65.5M (17.8%), which is real model
capacity spent on the embedding table rather than on depth.

Consequence: the 4.24 GPT-2 ratio that every earlier token figure in this file inherits
is superseded by **4.418** (corpus) and **4.604** (ILSI facts). Figures below still
carrying 4.28 are flagged where they matter.

**4b. Budget: 2B unique × 3 epochs, with the Decision 2 caps.** Kept the caps rather than
taking more unique tokens: 3.07B is available, so the alternative was reachable. Court
diversity is worth more than unique-token count here — uncapped, Bombay and Madras alone
are 59% of the corpus, and the evaluation set draws on all seven courts. Spending the
extra tokens would have bought volume at the cost of skewing pretraining toward two
registries.

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

## Corpus yield — extrapolated (superseded)

> Kept for the method and its error, not its numbers. The real figures are in
> *Corpus yield — measured* below; this block ran 16.9% hot.

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
extraction before the extraction run. *(Fixed — Decision 5.)*

---

## Corpus yield — measured

**Full extraction completed 2026-09-06**, 38 minutes wall (19:49–20:27), all 7 courts,
558 row groups, unattended. Superseding the extrapolations above.

```
data/corpus.jsonl   2,977,598 documents   13,565.5M chars   14.34 GB
mean 4,555 chars/doc
~3,169.5M tokens at 4.28 chars/token  (provisional -- Decision 4)
```

| Court | Docs | Mchars | Mtokens | Predicted | Delta |
|---|---:|---:|---:|---:|---:|
| Supreme Court | 31,869 | 1,037.4 | 242.4 | 241.7 | **+0.3%** |
| Allahabad | 524,294 | 1,456.3 | 340.3 | 344.4 | **−1.2%** |
| Bombay | 1,186,639 | 4,578.0 | 1,069.6 | 1,409.9 | **−24.1%** |
| Calcutta | 9,285 | 20.6 | 4.8 | 5.4 | −10.9% |
| Delhi | 223,903 | 1,851.8 | 432.7 | 473.1 | −8.5% |
| Madhya Pradesh | 323,322 | 1,184.7 | 276.8 | 323.3 | −14.4% |
| Madras | 678,286 | 3,436.8 | 803.0 | 1,016.1 | **−21.0%** |
| **Total** | **2,977,598** | **13,565.5** | **3,169.5** | **3,813.9** | **−16.9%** |

**The 2B target holds with 1.58x headroom** (3.17B available), so nothing about the plan
changes — but the extrapolation ran **16.9% hot overall**, and the error is not uniform.

**Where the sampling was trustworthy and where it was not.** SC (+0.3%) and Allahabad
(−1.2%) were near-exact. The three largest misses are Bombay (−24.1%), Madras (−21.0%)
and MP (−14.4%). SC was sampled at 3 row groups of 56; every High Court at 2. So
**2 row groups is too thin a sample for a large file** — the between-group variance
FINDINGS treated as small is real at that n, and the CIs quoted above ("indicative given
n=2–3") were justified in the caveat and too narrow in practice. Do not extrapolate a
large parquet from 2 row groups again.

Document counts also reveal how differently the courts are chunked: Bombay yields
1.19M documents against the SC's 31.9k, and mean document length falls from 32,553
chars (SC) to 3,858 (Bombay) and 5,067 (Madras). The corpus-wide mean of 4,555
chars/doc is therefore an HC figure, not a judgment-length figure — the SC's own mean is
7x higher. Anything reasoning about document length must do it per court.

3 SC cases were emitted incomplete (missing chunks in the source, not straddlers) —
`process()` emits what exists rather than dropping them, per *Decision 5*.

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
2. **Inline margin letters survive cleaning.** `MARGIN` strips single letters alone on a
   line; law reports also print them inline and those remain — **42,018 occurrences across
   1,519/1,682 docs (90%), 802 per M chars** in the SC smoke test (`placed in C melit
   list`, `scaling D down three of the claims`). No text is lost, so this is noise the
   tokenizer will spend vocabulary on rather than corruption. Deferred, not accepted.
3. `build_corpus.py:205` and `inspect_ilsi.py:54` both estimate tokens as `chars/4`,
   the exact error *chars/token* below warns about — it overstates by 7.0%. The final
   run therefore printed "~3391M tokens" where 4.28 gives 3,169.5M. Cosmetic; deferred.

Done, all 2026-09-06: reassembly overlap trim (*Decision 3*); full-extraction default
plus the streaming restructure it required (*Decision 5*); the extraction run itself.

---

## ILSI

Measured 2026-09-06, complete scans of all three splits. Fetched by `src/fetch_ilsi.py`.

### Source: the Hub mirror, not Zenodo

ILSI is mirrored inside **IL-TUR** (`Exploration-Lab/IL-TUR`, ACL 2024) under `lsi/`,
readable with the same `HfFileSystem` + `pyarrow` path as the corpus. Splits match the
published sizes exactly, so this is ILSI and not a re-cut of it:

| | rows | PLAN.md |
|---|---:|---|
| `lsi/train` | 42,750 | ~43k |
| `lsi/dev` | 10,181 | ~10k |
| `lsi/test` | 13,019 | ~13k |
| `lsi/statutes` | 100 | 100 target sections (`sec_src`) |

232 MB total. **Downloaded, not streamed** — unlike the 53.6 GB corpus this is read
repeatedly (inspection, tokenizer, every fine-tuning epoch, evaluation) and has to be on
the GPU box for session 2 anyway.

Zenodo DOI `10.5281/zenodo.6053791` remains the citable original and is the place to
verify masking against source, but it is **not needed to proceed** — and its API did not
respond within 120s from here. Cite IL-TUR as the actual provenance in the write-up.

Still skip `type_map`, `label_tree`, `citation_network` and the metapath schemas — those
are for LeSICiN's graph.

### Labels are integers, and the mapping had to be established

A row's `labels` reads `[69, 9, 3]`, not section strings. The integers index the **row
order of the statutes file**, which is sorted ascending by section number.

Confirmed, not assumed. Resolving that way puts **Section 34 (11,731) · 302 (10,380) ·
506 (7,881) · 323 (7,728) · 307 (7,121)** at the top of the train frequency table —
common intention, murder, criminal intimidation, hurt, attempt to murder. That is the
charging pattern of Indian criminal law; a wrong offset gives noise. Structural checks
also hold: exactly 100 statutes, label indices span exactly 0–99, all ids in `Section N`
form. `fetch_ilsi.py:check()` re-asserts these every run.

`fetch_ilsi.py` writes both forms — `labels` as section ids, `label_ids` as the untouched
integers — plus `data/ilsi/label_map.json`. **That map is load-bearing**: it defines the
100 special tokens and the reverse map at evaluation. Freeze it with the tokenizer.

**9 of the 100 section ids are not plain numbers.** Special tokens cannot be numeric-only.
Range is `<ipc_2>` .. `<ipc_511>`, and the exceptions are:

```
<ipc_120B> <ipc_174A> <ipc_229A> <ipc_294(b)> <ipc_304A>
<ipc_304B> <ipc_366A> <ipc_376(2)> <ipc_498A>
```

**Two contain parentheses** — `<ipc_294(b)>` and `<ipc_376(2)>`. Letter suffixes are
harmless, but parentheses are worth checking against the tokenizer library before the
vocab is frozen: added-token handling is not uniformly punctuation-safe, and this is
exactly the class of thing that is unfixable after packing. Substituting a safe character
is fine — decide it now and record it, because `label_map.json` is the contract.

Statutes row order is already ascending by section number, which is exactly the target
ordering *Locked / Output format* requires — so sorting the integer indices sorts the
sections.

### Fact text is masked

Statute and party mentions are replaced with `<SECTION>`, `<ACT>`, `<ENTITY>` placeholders.
Deliberate — unmasked, the answer is printed in the input and the task is trivial. Two
consequences: the tokenizer must handle these placeholders, and they **widen the
court-fact vs layperson gap** in *Locked / Task*, because real user input contains no
placeholders at all. The demo mismatch is now structural, not just register.

### Label distribution — the rare-label worry was wrong

| | train | dev | test |
|---|---:|---:|---:|
| distinct labels | 100 | 100 | 100 |
| labels/doc mean | 3.71 | 3.92 | 3.90 |
| labels/doc p50 / p90 / max | 3 / 7 / 31 | 3 / 7 / 28 | 3 / 7 / 23 |
| labels with <50 examples | **0** | 0 | 0 |
| labels with <10 examples | **0** | 0 | 0 |
| rarest label count | **272** | 68 | 90 |

All 100 labels appear in all three splits. **No label is rare** — the rarest in train has
272 examples. `PLAN.md` §5 and the old *Next* item both treat "labels with <10 training
examples" as a cap on achievable macro-F1. **It is not a constraint here; drop that
concern.** Macro-F1 will be bounded by model capacity, not by label starvation.

Skew is head-heavy but not pathological: the **top 10 labels cover ~41% of all label
mentions** (train 41.2%, dev 41.0%, test 40.7%) — consistent across splits, so dev is a
fair proxy for test.

`sec_src` key alignment is exact: **100/100** train labels have matching section text
(`inspect_ilsi.py`).

### Label lists are not deduplicated — affects the fine-tuning target

**4 documents repeat a label in their own `labels` list**: train 3, dev 1, test 0.
Measured examples: `[61, 61, 62, 62]` and `[93, 94, 94, 94, 97, 97, 97]`.

Found because it broke the baseline outright — `csr_matrix` *sums* duplicate coordinates,
so those rows held 2s, the target typed as `multiclass-multioutput` rather than a binary
indicator, and `OneVsRestClassifier` refused it. Loud failure, easy fix (`set()` per row).

**The silent version of this bug is the one that matters.** *Locked / Output format*
specifies a target of sorted label tokens plus EOS. Built naively from `labels`, those
4 documents teach the model to emit `<ipc_395> <ipc_395>` — a repeated token in the
supervision signal, which then surfaces at evaluation as malformed output and is counted
against the parse-failure rate the plan says to report honestly. **Dedup when building the
target sequence**, not only in the baseline. Only 4 documents, and nothing will flag it.

### Fact length — 2048 does NOT fit a quarter of the data

Complete scans, tokens at the provisional 4.24 chars/token (*Decision 4* — re-measure):

| | train | dev | test |
|---|---:|---:|---:|
| chars p50 | 2,700 | 3,396 | 3,349 |
| chars p90 | 16,663 | 17,768 | 17,158 |
| chars max | 530,815 | 439,274 | 812,632 |
| tokens p50 | 636 | 800 | 789 |
| **tokens p90** | **3,929** | **4,190** | **4,046** |
| tokens p99 | 11,398 | 12,754 | 11,992 |
| **% docs > 2048 tokens** | **24.3%** | 28.2% | 26.7% |
| sentences/doc p50 | 18 | 22 | 22 |

The median fact fits 2048 comfortably (~636 tokens), but **p90 is roughly double the
context and ~a quarter of every split overflows it.**

Measured with real GPT-2 BPE rather than the char proxy, **24.8%** of train exceeds 2048
(10,589/42,750) — the 4.24-chars/token estimate above was accurate to half a point.

**Resolved: head-2048.** See *Truncation policy* below — tested, and the choice of end
turns out to cost almost nothing.

---

## Baseline — the number to beat

TF-IDF (1–2 gram, min_df 2, 200k features, sublinear) + one-vs-rest logistic regression
(liblinear, dual, C=1.0). `src/baseline_tfidf.py` → `runs/baseline_tfidf.json`. 165s CPU.

```
ILSI test (13,019 docs)      micro-F1 0.4941     macro-F1 0.2858
threshold 0.20, tuned on dev (dev micro 0.4969) -- test never touched
micro-P 0.4986   micro-R 0.4897
labels with F1 = 0.0: 10/100
no prediction at all: 935 docs (7.18%)
```

**The SLM has to beat micro 0.494 / macro 0.286.** Two caveats when comparing: the
baseline reads the *whole* fact text where the SLM gets 2048 tokens, so it is mildly
advantaged; and it has no generative failure mode, so its 7.18% no-prediction rate is the
nearest analogue to — not the same thing as — the parse-failure rate.

**The 10 zero-F1 labels are not the rare ones.** All hold 272–543 training examples:

```
Section 4 (367)   114 (349)   200 (287)   228 (329)   299 (385)
      308 (543)   384 (351)   419 (530)   450 (272)   457 (363)
```

Meanwhile Section 229A scores F1 0.743 on 602 examples and 174A scores 0.721 on 487. So
the failures are **semantic confusability, not data starvation** — 299 (culpable homicide,
definitional) is inseparable from 302/304 in bag-of-words terms, and 114 collides with
34/149. This corroborates *Label distribution* above: the macro ceiling is
representational, and the 0.286-vs-0.494 gap is where the headroom is.

---

## Truncation policy — decided

Context is locked at 2048 and 24.8% of ILSI train exceeds it (10,589/42,750, measured
with real GPT-2 BPE, not the char proxy; dev 2,906, test 3,512). `src/truncation_test.py`
→ `runs/truncation_test.json`. Probe is the TF-IDF baseline retrained per strategy, with
train/dev/test all truncated the same way — truncating test alone would measure a
train/test mismatch instead of information loss.

| strategy | micro-F1 | macro-F1 | zero-F1 | no-pred | label-pairs losing all top-50 feats | docs losing everything |
|---|---:|---:|---:|---:|---:|---:|
| untruncated | 0.4941 | 0.2858 | 10 | 7.18% | — | — |
| **head-2048** | 0.4890 | 0.2809 | 9 | 5.96% | 54/15,009 = 0.36% | 1 (0.03%) |
| tail-2048 | 0.4866 | **0.2820** | 11 | 5.67% | 77/15,009 = 0.51% | 0 |
| head+tail 1024 | **0.4899** | 0.2774 | 11 | 6.87% | **36/15,009 = 0.24%** | 0 |

**The finding is that it barely matters.** Every strategy lands within 1.5% relative of
untruncated on micro and 3% on macro; the spread *between* strategies is 0.0033 micro and
0.0046 macro, almost certainly inside run-to-run noise on a single split. No strategy
loses the label evidence: even the worst drops all top-50 features for only 0.51% of
(document, label) pairs, and at most one document out of 3,512 loses everything.

**Decision: head-2048.** Simplest to implement, best macro of the two that lead on micro,
and no measurable penalty. The concern that motivated the test — that judgment fact
sections state the determining conduct late, so head-truncation would silently gut the
signal — **is not supported**. Head loses less signal than tail (0.36% vs 0.51%).

**What this does not settle.** TF-IDF is order-insensitive: it measures whether the
evidence *tokens survive*, not whether an autoregressive model can use them at that
position. A 300M causal LM may well be more sensitive to where the evidence sits.
Revisit if fine-tuned F1 comes in far below this baseline — but do not spend GPU time
on the question up front, because the lexical evidence says there is little to win.

---

## Next

1. Tokenizer on corpus + ILSI + `sec_src`, + the 100 label tokens from `label_map.json`.
   Compare 16k vs 32k — 32k costs ~33M embedding params (11% of budget). Must also handle
   the `<SECTION>` / `<ACT>` / `<ENTITY>` masks and the parenthesised label tokens.
2. **Re-measure chars/token** with that BPE (*Decision 4*) and restate every token figure
   in this file, including the 3,169.5M total and the *Decision 2* cap of 351.5M.
3. Pre-tokenize and pack to shards **on laptop**, applying the *Decision 2* caps at pack
   time. No tokenization on GPU time.
4. Fine-tuning target construction — dedup labels (*ILSI / Label lists*), head-2048
   truncation.

Done 2026-09-06: ILSI fetched and inspected (`fetch_ilsi.py`); full corpus extraction
(38 min, 14.34 GB); TF-IDF baseline; truncation policy decided.

Session notes: checkpoint to persistent storage and test restore *during* session 1;
budget 1–2h of session 2 for setup and re-upload; 30-min smoke test before the long run.
