# Pocket Vakil

A 332M-parameter decoder-only language model, trained from scratch, that identifies
which Indian Penal Code sections apply to a described set of case facts.

Built for the IAIRO SLM Bootcamp with [OpenLanguageModel](https://openlanguagemodel.github.io/openlanguagemodel/).
Weights: [`Pocket-Vakil/pocket-vakil-332m-ipc`](https://huggingface.co/Pocket-Vakil/pocket-vakil-332m-ipc)

---

## Quick start

```bash
pip install torch transformers
python demo_layman.py
```

Describe an incident in plain English:
the woman was harassed for dowry by her husband and in-laws and died of
burns within three years of marriage
raw output (0.6s): '<ipc_304B><ipc_498A><|endoftext|>'
sections: ['304B', '498A']


Weights download from the Hub on first run (~1.3 GB, then cached). Inference is
sub-second on CPU.

## Results

Full 13,019-row ILSI test split:

| Metric | Pocket Vakil | TF-IDF baseline |
|---|---|---|
| Micro-F1 | 0.4791 | 0.4941 |
| Macro-F1 | **0.3585** | 0.2860 |
| Exact match | 0.0767 | — |
| Labels with zero F1 | **0/100** | — |
| Parse failures | 0.07% | — |

Slightly below the baseline on micro-F1, well above it on macro-F1. Every one of the
100 IPC sections is predicted correctly at least sometimes — the baseline collapses on
rare labels.

**Ablation.** A v3 variant with a heavier instruction mix (73/19/8 vs v2's 81/11/8)
scored worse across the board: micro 0.4494, macro 0.3042, 17 dead labels. The v2
mix is measured, not assumed.

## Model

| | |
|---|---|
| Parameters | 331,964,416 |
| Architecture | Decoder-only transformer, from scratch (no pretrained weights) |
| Attention | MHA, 16 heads, head_dim 64 |
| Normalization | LayerNorm |
| Activation | SwiGLU, ff_multiplier 2.625 |
| Position | RoPE |
| Context | 2048 |
| Precision | bf16 |
| Tokenizer | Custom BPE, 32,000 vocab, 100 IPC label tokens |

**Task framing.** Multi-label classification done generatively. The 100 IPC sections
are special tokens in the vocabulary, emitted sorted ascending followed by EOS. No
classification head. Trade-off: no per-label confidence scores, so no threshold tuning.

**Prompt format** (required — the model emits bare EOS without it):
{passage}

TASK: identify the applicable IPC sections
ANSWER:


## Training

| | |
|---|---|
| Pretraining | Open India Law, ~2B tokens, 49 shards, SC + 7 High Courts |
| Fine-tuning | ILSI, 52,750 mixed examples (81% IPC / 11% instruction / 8% continuation) |
| Compute | 1 × H100 NVL (93 GB), 18.6 GPU-hours across two 20h slots |
| Optimizer | AdamW, β=(0.9, 0.999), wd 0.1, grad clip 1.0 |
| Batch | 12 micro × 16 accum = 393,024 tokens/step |
| Tokens seen | 3,945,960,960 (11.8 tokens/param, ~60% of Chinchilla-optimal) |
| Final loss | 1.2069 pretrain (ppl 3.34) · 0.8057 fine-tune (ppl 2.24) |

Bombay and Madras are capped at pack time so no single court dominates.

**LR schedule.** Slot 1: 6e-4 peak, 500 warmup, cosine to 6e-5. Slot 2 restarts at
2e-4 with 100-step rewarmup rather than resuming — the Slot 1 cosine had already
decayed to its floor, so resuming would have trained at near-zero LR.

## Key finding: a padding bug cost 2× the measured score

First eval returned 0.2408 micro-F1 with a 43% parse-failure rate. Two hypotheses
ruled out: prompt-format mismatch (verified character-exact, 0 differing positions)
and truncation (doubling the generation budget changed nothing).

Actual cause: training right-padded with EOS, eval left-padded with EOS (id 0, a real
stop token, not a dedicated pad id) and passed no attention mask. At batch 16, 52.3%
of attended positions were EOS padding — the model was reading "stop" repeatedly
before seeing the case.

Three paths, identical 40 rows:

| Path | Micro-F1 | Parse failures |
|---|---|---|
| Batch-1, unpadded | 0.4695 | 0% |
| Right-padded, batch 16 | 0.4713 | 0% |
| Left-padded, batch 16 (original) | 0.2304 | 37.5% |

Failing rows carried a mean 1,728 pad tokens against 597 for succeeding ones.
Right-padding is also 2.5× faster (16.6s/row vs 41.2s/row).

The diagnostic that caught it was parse-failure rate: precision on parsed outputs
held at ~0.47 throughout. High precision with high failure rate means a pipeline
bug, not a bad model.

Full numbers in `archive/patch_evaluate.py` and `src/diag_padding.py`.

## Limitations

- **Cannot abstain.** Fixed label set, no null class, no confidence threshold. On
  non-criminal input it returns Section 34 (common intention), which is never a
  standalone offence.
- Exact match is 0.077 — useful as a candidate set for review, not as a final answer.
- Macro-F1 0.36 vs micro 0.48: rare sections still missed often.
- Trained on court-authored prose. Plain-language input works better than expected
  but the distribution gap is unmeasured.
- Undertrained at 11.8 tokens/param; loss was still falling when compute ran out.
- Per-shard SHA256 hashes were not computed before the compute node was reclaimed.
  The corpus is reproducible from `src/build_corpus.py` + `data/shards/manifest.json`.

## Reproducing

```bash
pip install -U huggingface_hub pyarrow
hf auth login
```

The corpus is gated — accept the terms at
[vaquill/open-india-law](https://huggingface.co/datasets/vaquill/open-india-law)
while logged in, or reads return 403. File listings are public on gated repos, so a
successful listing does not mean you have access.

```bash
python src/build_corpus.py --row-groups 3   # trial
python src/build_corpus.py                  # full (~13.4 GB, run overnight)
python src/fetch_ilsi.py
python src/train_tokenizer.py
python src/pack.py
python src/baseline_tfidf.py
python src/train.py                         # pretrain, then fine-tune
python src/evaluate.py
```

**Known path issue:** `src/build_mixed_ft_v3.py` and `src/patch_labels.py` reference
`data/ilsi_ft_v1/`, which is `data/ilsi_ft/` on disk. The v3 data was built on the
compute node and not retrieved before the window closed.

**The tokenizer is frozen once training starts.** Retraining it shifts token IDs and
invalidates every packed shard. `data/ilsi/label_map.json` is frozen alongside it —
it defines the 100 special tokens and the reverse map used at eval.

## Layout
```
pocket_vakil/
├── README.md
├── PLAN.md design decisions
├── FINDINGS.md measured facts and gotchas
├── demo_layman.py plain-language demo
├── predict.py inference entry point (also on HF)
├── inference_hf.py reproduces reported results via transformers
├── convert_to_hf.py OLM → HF conversion
├── make_plots.py generates submission figures
│
├── src/ pipeline: corpus → tokenizer → pack → train → eval
├── runs/ per-run logs and metrics
├── logs/ extraction and verification logs
├── submission/ frozen bootcamp submission bundle
├── archive/ one-off diagnostics and superseded artifacts
└── data/ manifests only; corpus and shards gitignored
```


Weights, checkpoints, packed shards, and the raw corpus are gitignored — ~32 GB, all
reproducible or published on the Hub.

## Attribution

Pretraining: [Open India Law](https://huggingface.co/datasets/vaquill/open-india-law)
(Vaquill AI), CC BY 4.0. Attribution is a licence condition.

Fine-tuning: ILSI, from Paul, Goyal & Ghosh, *LeSICiN*, AAAI 2022. Accessed via the
IL-TUR mirror (`Exploration-Lab/IL-TUR`, ACL 2024) — cite both.

## Disclaimer

Not a legal tool. Outputs are candidate sections for human review, not legal advice.