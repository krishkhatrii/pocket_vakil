# Pocket Vakil

A 300M-parameter language model trained from scratch to identify which Indian Penal
Code sections apply to a described situation.

IAIRO SLM Bootcamp project. Built with [OpenLanguageModel](https://openlanguagemodel.github.io/openlanguagemodel/).

---

## Setup

```bash
python -m venv .vakil_env
.vakil_env\Scripts\activate          # Windows
pip install -U huggingface_hub pyarrow

hf auth login
```

The corpus is gated. Accept the terms at
[huggingface.co/datasets/vaquill/open-india-law](https://huggingface.co/datasets/vaquill/open-india-law)
**while logged in**, or every read returns 403. Note that file listings and the README
are served publicly on gated repos, so those succeeding does not mean you have access.

## Run order

```bash
# 1. Build the pretraining corpus (long - run overnight)
python src/build_corpus.py                  # full extraction, all 7 courts
python src/build_corpus.py --row-groups 3   # quick trial first

# 2. Inspect ILSI (download from Zenodo DOI 10.5281/zenodo.6053791)
python src/inspect_ilsi.py data/ilsi/
```

Then: tokenizer → pack to shards → TF-IDF baseline → pretrain → fine-tune → evaluate.

## Scripts

| Script | Does | Produces |
|---|---|---|
| `build_corpus.py` | Filters Open India Law to English, `year <= 2020`, six ILSI High Courts + SC. Strips headers and markers, reassembles chunks by `case_id`, trims overlap. | `data/corpus.jsonl` |
| `inspect_ilsi.py` | Label vocab, labels-per-doc, label frequency skew, text lengths, `sec_src` key alignment. | stdout |

### build_corpus.py flags

| Flag | Default | Notes |
|---|---|---|
| `--row-groups` | all | Cap SC row groups. Use `3` for a trial run. |
| `--hc-groups` | all | Cap row groups per High Court. Use `2` for a smoke test. |
| `--sc-only` | off | Skip High Courts. |
| `--year-max` | 2020 | IPC/BNS cutoff. Don't change without reading `PLAN.md` §4.1. |
| `--min-chars` | 500 | Drop short documents after cleaning. |

Output is one JSON object per line:
`{case_id, court, year, title, text, source}` where `source` is `sc` or `hc`.

## Layout

```
pocket_vakil/
├── README.md
├── PLAN.md              what we're building and why (stable)
├── FINDINGS.md          measured facts and gotchas (append as discovered)
├── WRITEUP.md           the deliverable (fill in as results land)
│
├── src/
│   ├── build_corpus.py
│   ├── inspect_ilsi.py
│   ├── train_tokenizer.py
│   ├── pack.py
│   └── baseline_tfidf.py
│
├── data/                gitignored — large and reproducible
│   ├── corpus.jsonl     pretraining corpus, cleaned
│   ├── ilsi/            train/dev/test + sec_src, from Zenodo
│   ├── tokenizer/       vocab + merges — frozen once training starts
│   └── shards/          packed uint16 token windows for the GPU
│
├── runs/                one file per training run
└── notes/               scratch, delete freely
```

Nothing in `data/` is committed. `corpus.jsonl` runs to ~17–18 GB at full extraction
(provisional — see below) and git keeps history forever; all of it is reproducible from
the scripts.

**The tokenizer is frozen once training starts.** Retraining it invalidates every packed
shard, because token IDs shift. Keep exactly one and record its vocab size in `runs/`.

**Shards are why packing happens locally.** Training memory-maps a shard and reads
slices — no parsing or tokenizing in the hot loop. The packed ~2B tokens at 2 bytes each
is ~4 GB, against ~17–18 GB of JSONL for the full extraction it is packed from, so
re-uploading between the two GPU sessions is faster too.

**The ~17–18 GB figure is provisional.** It is 3.81B measured tokens at 4.28 chars/token,
and that ratio is GPT-2's 50k BPE standing in for a project tokenizer that does not exist
yet. A 16k vocab would compress worse and a 32k in-domain vocab better, so re-measure
after the vocab decision — `FINDINGS.md` *Decision 4*. Note it does not fit comfortably
on this laptop's free disk; check before the overnight run.

## Runs

One file per training run in `runs/`, recording: corpus version and token count,
tokenizer vocab size, model config, hyperparameters, wall time, final metrics.

Do this every time. In week five you will not remember which corpus produced which F1.

## Compute

300M params · context 2048 · 6B tokens processed · 2 × 20h H100. Fixed — see `PLAN.md` §3.

Tokenize and pack locally. GPU time is for gradient steps only.

## Attribution

Pretraining data: [Open India Law](https://huggingface.co/datasets/vaquill/open-india-law)
(Vaquill AI), CC BY 4.0. Attribution is a condition of the licence.

Fine-tuning data: ILSI, from Paul, Goyal & Ghosh, *LeSICiN*, AAAI 2022.

## Disclaimer

Learning project. Not a legal tool. Outputs are not legal advice.
