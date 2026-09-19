---
license: apache-2.0
language: [en]
tags: [legal, indian-law, ipc, multi-label-classification, from-scratch]
pipeline_tag: text-generation
---

# Pocket Vakil 332M — IPC Statute Identification

A 332M-parameter language model trained **from scratch** on 2.0B tokens of
public Indian case law, then fine-tuned to identify applicable Indian Penal
Code sections from case facts.

Built for the IAIRO / Pramana SLM++ Bootcamp (Team 15).

## Usage

Requires `trust_remote_code=True` — the architecture is custom.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "Karishma-24/pocket-vakil-332m-ipc", trust_remote_code=True)
tok = AutoTokenizer.from_pretrained(
    "Karishma-24/pocket-vakil-332m-ipc", trust_remote_code=True)
```

**The task frame is required.** Prompts must end with:
<case facts>\n\n### TASK: identify the applicable IPC sections\n### ANSWER:\n

Without it the model emits EOS immediately. Output is a sequence of
`<ipc_NNN>` tokens terminated by EOS.

## Results — ILSI test split, 13,019 cases, 100 sections

| Metric | Pocket Vakil | tf-idf baseline |
|---|---|---|
| micro-F1 | 0.4791 | 0.4941 |
| macro-F1 | **0.3585** | 0.2858 |
| exact match | 0.0767 | — |
| labels with F1 = 0 | **0 / 100** | — |

Parse failure rate 0.07%, zero empty outputs.

For reference, LeSICiN (AAAI 2022, the paper that introduced ILSI) reports
28% macro-F1; a 2025 model reports 0.486 micro / 0.355 macro. Setups are not
identical — treat as a reference range, not a leaderboard comparison.

## Architecture

332M total / 299.2M non-embedding · 24 layers · d_model 1024 · 16 heads ·
**MHA** · **RoPE (interleaved pairs)** · **LayerNorm** (not RMSNorm) ·
**SwiGLU** · tied embeddings · biases on all projections · context 2048 ·
bf16 · 32,000-token BPE vocab including 100 `<ipc_*>` label tokens.

No stock HF architecture matches this combination, hence the custom modelling
code.

## Training

Pretraining: 3.93B tokens over ~2.0B unique (1.97 epochs), 10,049 steps,
2 × H100, ~17.2 GPU-hours. AdamW, peak LR 6e-4, cosine. **11.8 tokens per
parameter — roughly 60% of Chinchilla-optimal; the model is undertrained.**

SFT: 52,750 examples (81% IPC, 11% instruction-following, 8% continuation),
3 epochs, peak LR 5e-5, ~1.4 GPU-hours.

## Limitations

- Closed 100-section output vocabulary; cannot express anything outside it
- No abstention — emits sections even on out-of-distribution input
- Inputs truncated to ~2,025 tokens; sections cited late in long judgments
  may be missed
- Domain-pure training corpus: near-chance on general knowledge and
  instruction-following benchmarks
- **Not legal advice.** Intended to surface candidates for human review.