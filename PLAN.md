# Project Plan: SLM for Indian Legal Statute Identification

**Bootcamp:** IAIRO SLM Bootcamp (Aug 15 – Sep 19)
**Framework:** OpenLanguageModel (OLM) — from-scratch model training
**Compute:** 2 × 20-hour H100 sessions

> Measured data findings and gotchas live in `FINDINGS.md`. This document is the
> stable plan; it is not updated as things are discovered.

---

## 1. Objective

Train a small language model, entirely from scratch, that takes a description of a
situation's facts and identifies which section(s) of the Indian Penal Code apply.

**Example:**
- Input: *"The accused entered the complainant's house at night and took away jewellery without consent."*
- Output: applicable IPC section(s) covering house-trespass and theft.

Multi-label: one situation can map to more than one section.

**What is measured vs what is claimed.** Training and evaluation both use ILSI, whose
inputs are fact descriptions extracted from judgments — formal register, named parties
(PW-1, A-2), dates, procedural framing. All reported precision/recall/F1 therefore
describe **court-fact input**.

The model will also answer input phrased casually, as in the example above, and demos
will show this. But no labelled layperson test set is in scope, so that performance is
**unquantified**, and Paul et al. (ILSIC, EACL 2026 Findings) found that models trained
purely on court judgments do poorly on laypeople queries. The write-up reports the two
separately rather than letting a court-fact F1 sit next to a layperson demo.

## 2. Compute envelope

Fixed by the OLM cost-frontier planner. **Not to be exceeded.**

| Item | Value |
|---|---|
| Parameters | 300M |
| Context length | 2048 |
| Tokens processed | 6B (hard ceiling) |
| Unique tokens | ~2B, 3 epochs |
| GPU | 2 × 20h H100 |

Check: 6ND = 6 × 3e8 × 6e9 = 1.08e19 FLOPs. H100 BF16 ≈ 989 TFLOPs dense; at 30% MFU
that is ~10 hours of pretraining. 300M × 20 tokens/param = 6B, so this point is
compute-optimal by construction.

Indicative architecture: d_model 1024, 24 layers, 16 heads
(12 · n_layer · d_model² ≈ 302M non-embedding).

The remaining hours are deliberately kept as recovery room, not spent on a larger model.

## 3. Datasets

### 3.1 Pretraining corpus

- **Source:** Open India Law (`vaquill/open-india-law` on Hugging Face), snapshot
  `v2026.08`. CC BY 4.0, gated behind contact-sharing. Attribution mandatory.

- **Content:** 12,848,644 judgments (1950–2025), Supreme Court and all 25 High Courts,
  as 32,572,660 extracted-text chunks totalling 53.6 GB; plus 22,265 enactments from
  India Code. Judgment PDFs are not re-hosted — this is the extracted-text layer.

- **Filters:**

  1. **Courts:** Supreme Court plus the six High Courts ILSI draws from
     (Allahabad, Bombay, Calcutta, Delhi, Madhya Pradesh, Madras).
  2. **`language_code == "en"`.** Only 33.3% of the corpus is English; Hindi and
     Punjabi are ~27.5% each. Mandatory, not optional.
  3. **`year <= 2020`.** The Bharatiya Nyaya Sanhita replaced the IPC in July 2024, so
     post-2024 judgments cite BNS numbers for conduct ILSI labels with IPC numbers
     (theft: IPC 379 vs BNS 303). Training across both teaches two competing numbering
     schemes for identical facts. Also aligns the corpus with ILSI's vintage.
  4. **No criminal-domain filter.** Would require regex on `case_number` prefixes —
     crude, silently lossy, an extra failure point. Pretraining teaches legal English;
     fine-tuning teaches the task. Fine-tuning and evaluation are both entirely
     criminal, which is the consistency that matters.
  5. **No `section_type` filter.** Measured as unusable — `facts` was 1 row in 2000.
     See `FINDINGS.md`.

- **Preprocessing:**
  - Strip the `Case: … Section: …` contextual header. It is present in **both** `text`
    and `text_original`, which are identical — the documented clean field does not exist.
  - Strip `[TITLE]` / `[SECTION]` markers and single-letter lines (law-report margin
    markers).
  - **Filter language before grouping by `case_id`.** Translations share a `case_id`
    with the original, so grouping first concatenates language versions of the same
    judgment.
  - Chunks are 4096 chars with 512 chars overlap. Group by `case_id`, sort by
    `chunk_index`, trim overlap via `char_start`/`char_end`.
  - Tokenizer trained on this corpus plus ILSI train plus `sec_src`, with 100 label
    tokens added, so section references like "s. 379 IPC" tokenize sensibly.

- **Access pattern:** read parquet directly via `HfFileSystem` + `pyarrow` (range
  requests, no full download). Debug against the Supreme Court file, then process the
  six High Court files, writing only surviving rows locally.

- **Volume:** Supreme Court alone yields ~260M tokens. Full English extraction across
  all seven courts is required to reach the ~2B target. This is the critical path.

### 3.2 Fine-tuning + evaluation dataset

- **Dataset:** ILSI — Indian Legal Statute Identification (Paul, Goyal & Ghosh,
  *LeSICiN*, AAAI 2022).

- **Content:** ~66,090 fact descriptions from the Supreme Court and six High Courts,
  each labelled with applicable sections from a target set of 100 IPC statutes.

- **Splits:** ~43k train / ~10k dev / ~13k test, pre-defined.

- **Access:** code at `github.com/Law-AI/LeSICiN`; data on Zenodo under DOI
  `10.5281/zenodo.6053791`.

- **Format:** JSONL with `id`, `text` (a list of sentences, joined before tokenizing),
  and `labels`.

- **What we use and skip:** train/dev/test plus `sec_src` (text of the 100 target
  sections, keyed to match the label vocabulary). Skip `type_map`, `label_tree`,
  `citation_network` and metapath schemas — those exist for LeSICiN's heterogeneous
  graph, which we are not building.

- **IPC text:** `sec_src` is primary. Open India Law's legislation subset is a
  cross-check only; the IPC there is `IND_REP_659_1860`, marked `repealed` with
  `in_force = False`, so filtering on `in_force` drops it entirely.

## 4. Model output format

Generative, not a classification head — the pretraining is a causal LM either way, and
generation is what the bootcamp is about.

- 100 special tokens added to the tokenizer vocabulary, one per section (`<ipc_379>`).
  Each section is a single token, so partial or malformed section numbers are
  impossible by construction.
- Fine-tuning target: a sorted sequence of label tokens followed by EOS. Labels sorted
  ascending by section number, so the model is not penalised for producing the right
  set in an arbitrary order.
- Parsing: collect label tokens from the output.
- **Parse-failure rate is measured and reported, not silently repaired.** If the model
  emits ordinary text instead of label tokens, that counts as a failure. A near-zero
  rate is itself a result worth reporting.

## 5. Pipeline

1. **Corpus build.** Full English extraction, SC + six High Courts, `year <= 2020`,
   reassembled and cleaned. Target ~2B tokens.
2. **ILSI inspection.** p90 fact length in tokens (does 2048 suffice?), count of labels
   with <10 training examples (caps achievable macro-F1), exact label string format.
3. **Tokenizer.** BPE on corpus + ILSI train + `sec_src`, plus the 100 label tokens.
   Compare 16k vs 32k vocab — 32k costs ~33M embedding params, 11% of the budget.
4. **Pack.** Pre-tokenize the whole corpus into binary shards **on laptop**. No
   tokenization on GPU time.
5. **Baseline.** TF-IDF + one-vs-rest logistic regression on ILSI, before any GPU time.
   Cheap, and a from-scratch SLM that cannot beat it is a signal worth having early.
6. **Pretraining.** Causal LM in OLM, 300M params, context 2048, 3 epochs over ~2B
   tokens. Checkpoint to persistent storage throughout.
7. **Fine-tuning.** ILSI train split, fact description → sorted label-token sequence.
8. **Evaluation.** ILSI test split, precision/recall/F1 micro and macro, plus
   parse-failure rate. Published ILSI baselines as context, not a target to beat.
9. **Write-up.**

## 6. Timeline

| Phase | Work |
|---|---|
| Early | Corpus extraction, ILSI inspection, tokenizer, packing, TF-IDF baseline — all local |
| Session 1 (20h H100) | 30-min smoke test at full config, then pretraining with checkpointing |
| Session 2 (20h H100) | Restore checkpoint, finish pretraining, fine-tune, evaluate |
| Late | Error analysis, write-up |

Session notes: verify checkpoint restore *during* session 1, not at the start of
session 2. Budget 1–2h of session 2 for environment setup and data re-upload. The
GPU directory wipes between slots, so plan the re-upload.

## 7. Limitations (to state explicitly in the write-up)

- Research/learning project, evaluated only on ILSI. Not a validated legal tool, and
  predictions are not legal advice.
- **Closed 100-label set with no abstention.** Given a non-criminal fact pattern the
  model will still emit IPC sections, confidently and wrongly. A worked out-of-domain
  example will demonstrate this.
- Metrics cover court-fact input only; layperson-phrased performance is unmeasured and
  expected to be lower (§1).
- Labels are IPC, not BNS — the model does not know post-2024 numbering.
- Coverage capped at ILSI's 100 sections, not the full ~511-section IPC.
- Performance bounded by a 300M-parameter model on ~2B tokens, against full-scale
  baselines (InLegalBERT, LeSICiN) trained on the same task.
- Corpus caveats: Open India Law is a community/commercial release, not peer-reviewed,
  and several of its documented figures did not survive measurement (`FINDINGS.md`).
  Text comes from PDF extraction with OCR fallback; pre-2000 scans are materially
  noisier.

## 8. Key References

- Paul, S., Goyal, P., Ghosh, S. — *LeSICiN: A Heterogeneous Graph-based Approach for Automatic Legal Statute Identification from Indian Legal Documents*, AAAI 2022 (introduces the ILSI dataset)
- Paul, S. et al. — *Pre-training Transformers on Indian Legal Text* (InLegalBERT)
- Paul, S. et al. — *ILSIC: Corpora for Identifying Indian Legal Statutes from Queries by Laymen*, EACL 2026 Findings
- IL-TUR: Benchmark for Indian Legal Text Understanding and Reasoning, ACL 2024
- OpenLanguageModel — cost frontier planner, `olm-cost-frontier-demo.pages.dev`
