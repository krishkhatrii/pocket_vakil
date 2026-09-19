# Run: <id, e.g. pretrain-s1>

Copy this file to `runs/<id>.md` **before** starting the session. Fill the config
block before the GPU clock starts; fill the rest as you go, not afterwards.

The point of this file is week five, when you cannot remember which corpus produced
which F1. Write down what you actually did, including the parts that went wrong.

---

## Identity

| | |
|---|---|
| Run id | |
| Date / session | |
| Purpose | smoke test / pretrain / resume / fine-tune / eval |
| Started from | scratch, or `runs/<id>.md` checkpoint at step N |

## Data

| | |
|---|---|
| Corpus | `data/corpus.jsonl`, 2,977,598 docs, 13,565.5M chars |
| Shards | `data/shards/`, manifest total tokens: |
| Tokenizer file | `bpe_32000_final.json` |
| **Tokenizer sha256** | |
| Vocab size | 32,000 (+104 added) |
| Seq len | 2048 |
| Mix | Decision 2 caps — Bombay/Madras capped, five whole |

> The tokenizer hash ties this run's token ids to one vocabulary. If it does not
> match `data/shards/manifest.json`, the run is not comparable to the others and
> the shards are not the ones you think they are. Check it, do not assume it.

## Model config

| | |
|---|---|
| Params (non-embedding) | |
| d_model / layers / heads | 1024 / 24 / 16 |
| Embeddings | tied |
| Context | 2048 |
| Precision | |
| Optimiser / LR / schedule | |
| Warmup / weight decay / grad clip | |
| Batch size / grad accum / tokens per step | |
| Seed | |

## Budget

| | Planned | Actual |
|---|---|---|
| Tokens processed | 6B ceiling | |
| Epochs over unique | 3 × ~2B | |
| GPU hours | 20 | |
| MFU | ~30% target | |

---

## Loss curve

Record at checkpoints, not continuously. Include the step you restored from.

| Step | Tokens seen | Train loss | Val loss | LR | Wall clock | Checkpoint path |
|---|---|---|---|---|---|---|
| | | | | | | |

## Timeline

| Wall clock | Event |
|---|---|
| | session start, environment up |
| | data uploaded (size, how long) |
| | smoke test done |
| | training start |
| | **checkpoint restore verified** ← do this during session 1, not at the start of session 2 |
| | training end |

## Results

| Metric | Value |
|---|---|
| Final train / val loss | |
| ILSI test micro-F1 | |
| ILSI test macro-F1 | |
| Labels with F1 = 0.0 | |
| Parse-failure rate | |
| Empty predictions | |

Baseline to beat: **micro 0.4941 / macro 0.2858** (`runs/baseline_tfidf.json`).
The baseline sees whole fact text; this model sees 2048 tokens with head
truncation, so it is at a mild disadvantage. Say so when comparing.

---

## What broke

Every failure, including the ones you fixed in two minutes. The two-minute fixes
are the ones that recur at 3am in session 2.

| What | Symptom | Cause | Fix |
|---|---|---|---|
| | | | |

## What changed from the plan

Anything that differs from `PLAN.md` or `FINDINGS.md`, and why. If a decision in
either document turned out to be wrong, say so here and update it there.

-

## Notes for the next session

Things future-you needs and will not otherwise remember: environment setup gotchas,
the exact re-upload command, how long each stage really took, what to do differently.

-
