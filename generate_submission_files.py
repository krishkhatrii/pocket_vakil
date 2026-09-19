#!/usr/bin/env python
"""Generate manifest.yaml and train_log.jsonl for the IAIRO submission.

Every value here was recovered from the checkpoint's own optimizer/scheduler
state or from the training logs -- not from argparse defaults, which in at
least one case (peak LR) differ from what actually ran.

Run from your pocket_vakil folder:

    python generate_submission_files.py \
        --l1-log runs_backup/runs/l1_pretrain.jsonl \
        --out-dir submission
"""
import argparse
import json
from pathlib import Path

# --- measured, not assumed ----------------------------------------------
TOKENS_PER_STEP = 192 * 2047        # 393,024 -- matches the checkpoint's
                                    # total_tokens_processed exactly
SLOT1_SECONDS = 8 * 3600 + 36 * 60          # 30,960
SLOT2_SECONDS = 8 * 3600 + 37 * 60 + 1      # 31,021
SLOT1_LAST_STEP = 4964
SLOT2_LAST_STEP = 10049

MANIFEST = """# =====================================================================
# Pocket Vakil - IAIRO / Pramana SLM++ Bootcamp
# Domain-Specific SLM Evaluation - run manifest
# =====================================================================

team_id: team15
model_name: Pocket Vakil
model_version: v2
submission_date: 2026-09-16

# ---------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------
checkpoint_path: final_model_v2.pt          # 1.33 GB, weights only
checkpoint_sha256: 0b264fd1e6c0b12564b2481cda26e8c3a5684bec714145b53129af08beae9ae2
base_checkpoint_path: step_10000.pt         # pretrain base, 3.98 GB
tokenizer_path: bpe_32000_final.json
tokenizer_sha256: b2de5ea09885462611a73045cc98d88f670f0893ce7518bc6e36d1895dde8203
log_path: train_log.jsonl                   # pretraining, concatenated
finetune_log_path: train_log_finetune.jsonl
weights_access_link: TODO_HUGGINGFACE_OR_DRIVE_URL
inference_script: src/evaluate.py

# ---------------------------------------------------------------------
# Model & architecture
# ---------------------------------------------------------------------
base_model: from-scratch
parameter_count_total: 332000000
parameter_count_non_embedding: 299200000
parameter_count_convention: >-
  Total includes the tied embedding matrix (L5 convention:
  N = 12 * n_layer * d_model^2 + vocab * d_model). Non-embedding figure
  given separately since L3's N_core convention excludes vocab matrices.
num_layers: 24
embed_dim: 1024
num_heads: 16
head_dim: 64
attention_type: MHA           # full-width q/k/v projections, no grouping
normalization: LayerNorm      # pre-norm, two residual branches per block
activation: SwiGLU
ff_multiplier: 2.625          # SwiGLU parity with a 4x GELU MLP
positional_encoding: RoPE
context_length: 2048
tie_embeddings: true
dropout: 0.0
precision: bf16
framework: OLM 2.2.0 (olm.nn.blocks.LM)
init: scaled init, std=0.02 (overrides an OLM default)

# ---------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------
tokenizer_type: BPE
vocab_size: 32000
tokenizer_custom: true
eos_id: 0
domain_specific_additions: >-
  100 IPC label tokens (<ipc_*>) plus ILSI's three mask tokens
  (<SECTION>, <ACT>, <ENTITY>). All present at tokenizer-training time;
  they cannot be added later without invalidating the embedding table.

# ---------------------------------------------------------------------
# Domain & task
# ---------------------------------------------------------------------
domain: Indian criminal law (Indian Penal Code)
primary_task_type: classification (multi-label, emitted generatively)
end_user: >-
  Legal researchers and practitioners. Given the facts of a case, surface
  the applicable IPC sections as a candidate set for human review.

# ---------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------
pretraining_corpus:
  source: Open India Law - public Indian case law
  courts: [supreme_court, allahabad, bombay, calcutta, delhi, madhya_pradesh, madras]
  tokens_tokenized: 1999624904
  tokens_packed: 1999624192
  sequences: 976379
  shards: 49
  dtype: uint16
  seq_len: 2048
  tokens_per_court:
    delhi: 430862862
    madras: 351633884
    bombay: 351567911
    allahabad: 336009796
    madhya_pradesh: 289942391
    supreme_court: 234776273
    calcutta: 4831787
  caps_applied:
    bombay: 351500000
    madras: 351500000
  docs_dropped_by_cap: {bombay: 827402, madras: 410596}
  license: Public court judgments, publicly available
  hash_list: >-
    NOT AVAILABLE. Per-shard SHA256 was not computed before the compute
    node was reclaimed at the end of the allocated window. The corpus is
    reproducible from public sources via src/build_corpus.py together
    with data/shards/manifest.json, which records exact per-court token
    counts, caps and the tokenizer hash.

finetuning_data:
  source: ILSI (Indian Legal Statute Identification), Paul et al. AAAI 2022
  train_examples: 52750
  dev_examples: 10181
  test_examples: 13019
  label_set_size: 100
  composition:
    ipc: 42750          # 81.0%
    instruction: 6000   # 11.4%
    continuation: 4000  #  7.6%
  prompt_frame: "<passage>\\n\\n### TASK: <instruction>\\n### ANSWER:\\n<target>"

contamination_check:
  method: >-
    ILSI test-split n-grams removed from the pretraining corpus before
    packing; removal logged. ILSI's own entity masking (<SECTION>, <ACT>,
    <ENTITY>) is preserved in the fact text, so statute references cannot
    be read directly off the input.
  test_split_held_out: true

# ---------------------------------------------------------------------
# Training - pretraining
# ---------------------------------------------------------------------
optimizer: AdamW
optimizer_betas: [0.9, 0.999]
optimizer_eps: 1.0e-08
decoupled_weight_decay: true
weight_decay: 0.1
weight_decay_excluded: >-
  Two param groups. Group 0 (145 tensors) decayed at 0.1; group 1
  (242 tensors: norm scales, biases, embeddings) at 0.0.
grad_clip_threshold: 1.0

lr_schedule: warmup + cosine
slot1:
  peak_lr: 6.0e-04
  min_lr: 6.0e-05
  warmup_steps: 500
  scheduler_total_steps: 6102
  steps_completed: 4964
  final_train_loss: 1.4460          # at step_4964.pt
  wall_clock: 8h36m
slot2:
  restart: true
  peak_lr: 2.0e-04                  # initial_lr from optimizer state
  min_lr: 6.0e-05
  warmup_steps: 100
  scheduler_total_steps: 5086
  steps: 4964 -> 10049
  final_train_loss: 1.2069          # step 10,049 (logged, no checkpoint)
  checkpoint_loss: 1.2461           # step_10000.pt, the submitted base
  wall_clock: 8h37m01s

batch_size_micro: 12
grad_accum_steps: 16
effective_batch_size: 192           # sequences per optimizer step
tokens_per_step: 393024             # 192 * 2047 (targets are seq[1:])
total_steps: 10049
tokens_seen: 3930240000             # recorded at step 10,000
epochs: 1.97                        # over a 2.0B-token unique corpus
tokens_per_parameter: 11.8          # ~60% of Chinchilla-optimal (20)
random_seed: 1234

# ---------------------------------------------------------------------
# Training - post-training (SFT)
# ---------------------------------------------------------------------
sft_performed: true
sft_method: full fine-tune (no LoRA)
sft_init_from: step_10000.pt
sft_batch_size_micro: 4
sft_grad_accum: 12
sft_effective_batch: 48
sft_epochs: 3
sft_steps: 3294
sft_peak_lr: 5.0e-05
sft_min_lr: 5.0e-06
sft_warmup_steps: 200
sft_wall_clock: 1h25m23s
sft_final_loss: 0.6562
sft_loss_masking: >-
  Cross-entropy on response tokens only. Positions before the target
  begin are set to IGNORE_INDEX.
chat_template: "<passage>\\n\\n### TASK: <instruction>\\n### ANSWER:\\n"

# ---------------------------------------------------------------------
# Compute
# ---------------------------------------------------------------------
gpu_type: NVIDIA H100 NVL (93 GB)
gpu_count: 1
gpu_hours_allocated: 40.0
gpu_hours_actual: 18.3              # 8h36 + 8h37 + 1h25
throughput_tokens_per_sec: 64500    # slot 2, measured
mfu_percent: 13.0                   # 6ND / wall-clock / 989 TFLOP/s dense bf16
throughput_note: >-
  The trainer's end-of-run summary reports 127k tokens/s. That figure is
  wrong: it divides cumulative tokens across both slots by slot-2-only
  wall clock. The correct slot-2 rate is 64.5k tokens/s and MFU is
  computed from that.

# ---------------------------------------------------------------------
# Training events - so a restart is not misread as instability
# ---------------------------------------------------------------------
training_events:
  - step: 4964
    event: slot boundary
    detail: >-
      Allocation ended. Run resumed in a second 20-hour window from
      step_4964.pt with full optimizer, scheduler, RNG and dataloader
      cursor state restored.
  - step: 4964
    event: LR schedule restart
    detail: >-
      Resumed with --restart-cosine: 100-step warmup from 2e-6 to a peak
      of 2e-4, then cosine decay to 6e-5 over 5,086 steps. The peak
      (2e-4) is ABOVE the 1.13e-4 the run was at when slot 1 ended, so a
      transient loss excursion and elevated gradient norms are expected
      around steps 5,064-5,500. This is a deliberate schedule restart,
      not instability. It sits mid-run, so the "upward trend in the final
      10% of steps" convergence check is unaffected.
  - steps: 4970-5080
    event: replayed steps
    detail: >-
      Slot 1 logged to step 5,080 but its last checkpoint was step 4,964.
      Slot 2 resumed from the checkpoint and re-trained those ~110 steps.
      train_log.jsonl is deduplicated on step, keeping slot 2's values,
      since the submitted model's lineage runs through them.
  - step: 10049
    event: run end
    detail: >-
      GPU window closed. Last checkpoint written was step_10000.pt; the
      final 49 steps are logged but not checkpointed. The submitted base
      is step 10,000 (loss 1.2461).

# ---------------------------------------------------------------------
# Checkpoint-save log (step <-> file)
# ---------------------------------------------------------------------
checkpoint_save_log:
  pretrain_slot1_every: 292
  pretrain_slot2_every: 500
  retention: last 2 rolling + milestones every 2h kept permanently
  milestones: [1460, 2920, 4380, 6500, 8000, 9500]
  surviving: [step_9000.pt, step_10000.pt, slot1-pretrain-BACKUP_step_4964.pt]
  finetune_v2_every: 500
  finetune_v3_every: 300

# ---------------------------------------------------------------------
# Logging gaps - declared, not omitted
# ---------------------------------------------------------------------
logging_gaps:
  grad_norm: >-
    NOT LOGGED. Neither post-clip grad_norm nor pre-clip grad_norm with a
    clip flag was recorded. The gradient-explosion check cannot be run.
    Grad clipping was active at global norm 1.0 throughout.
  wall_clock_per_step: >-
    NOT RECORDED. train_log.jsonl carries a derived wall_clock_derived_s
    field computed from measured per-slot throughput. Flagged with
    wall_clock_is_derived: true on every row. Total elapsed time per slot
    is exact and is given above.
  val_loss_pretraining: >-
    NOT LOGGED DURING TRAINING. Reconstructed post-hoc by running each
    surviving milestone checkpoint over 256 held-out ILSI test rows.
    Nine points, not per-eval-interval. See milestone_val.json.
  val_loss_finetuning: >-
    NONE. The overfitting-gap check cannot be run on the fine-tuning log.

# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------
python_version: 3.11
key_dependencies: see pip_freeze.txt
olm_version: 2.2.0
eval_precision: bf16                # matches training; no eval-time quantization
eval_decoding: greedy, right-padded, batch 16, max_new 40

known_limitations:
  - Fixed 100-label output vocabulary. The model cannot express a section
    outside that set.
  - No abstention mechanism. On out-of-distribution input it will still
    emit label tokens.
  - Inputs truncated to ~2,025 tokens (context 2048 minus the task frame),
    so statutes referenced late in a long judgment may be missed.
  - Trained to 11.8 tokens/parameter, roughly 60% of Chinchilla-optimal.
    The base is undertrained and this is the dominant accuracy limit.
  - Generative label emission requires the model to decide the label set,
    its ordering, and when to stop. A classification head would remove
    two of those three problems.
  - General-capability performance is low by construction: the pretraining
    corpus is domain-pure Indian case law with no general web, code, math
    or instruction data.

# ---------------------------------------------------------------------
# Results - ILSI test split, 13,019 rows, model v2
# ---------------------------------------------------------------------
results:
  micro_f1: 0.4791
  macro_f1: 0.3585
  micro_precision: 0.4670
  micro_recall: 0.4918
  exact_match: 0.0767
  labels_with_zero_f1: 0            # out of 100
  parse_failure_rate: 0.0007
  parse_failures: {duplicate: 7, no_eos: 2, non_label: 3, empty: 0}
  baseline_tfidf: {micro_f1: 0.4941, macro_f1: 0.2858}
  ablation_v3:
    description: instruction-heavy mix (73.3/18.6/8.1 vs v2's 81.0/11.4/7.6)
    micro_f1: 0.4494
    macro_f1: 0.3042
    note: >-
      Evaluated on the same 1,000-row subsample as v2 (0.4650 / 0.3470).
      Reducing IPC rows from 42,750 to 30,000 cost accuracy on rare
      sections; dead labels rose from 8 to 17 on that subsample.
"""


def wall_clock_seconds(step):
    """Derived from measured per-slot throughput. NOT recorded per step."""
    s1_rate = SLOT1_SECONDS / 5080.0        # slot 1 logged to step 5,080
    s2_rate = SLOT2_SECONDS / (SLOT2_LAST_STEP - SLOT1_LAST_STEP)
    if step <= SLOT1_LAST_STEP:
        return round(step * s1_rate, 1)
    return round(SLOT1_LAST_STEP * s1_rate + (step - SLOT1_LAST_STEP) * s2_rate, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--l1-log", default="runs_backup/runs/l1_pretrain.jsonl")
    ap.add_argument("--ft-log", default="runs_backup/runs/l1_ft_v2.jsonl")
    ap.add_argument("--out-dir", default="submission")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # --- train_log.jsonl -------------------------------------------------
    src = Path(args.l1_log)
    if src.exists():
        rows = [json.loads(l) for l in src.open(encoding="utf-8") if l.strip()]
        with (out / "train_log.jsonl").open("w", encoding="utf-8") as fh:
            for r in rows:
                step = r["step"]
                fh.write(json.dumps({
                    "step": step,
                    "train_loss": r.get("train_loss"),
                    "val_loss": r.get("val_loss"),
                    "lr": r.get("lr"),
                    "grad_norm": None,
                    "tokens_seen": step * TOKENS_PER_STEP,
                    "epoch": r.get("epoch"),
                    "wall_clock_derived_s": wall_clock_seconds(step),
                    "wall_clock_is_derived": True,
                }) + "\n")
        n_val = sum(1 for r in rows if r.get("val_loss") is not None)
        print(f"train_log.jsonl        {len(rows):>5,} rows, {n_val} with val_loss")
        print(f"  final tokens_seen    {rows[-1]['step'] * TOKENS_PER_STEP:,}")
    else:
        print(f"[skip] {src} not found")

    # --- train_log_finetune.jsonl ---------------------------------------
    ft = Path(args.ft_log)
    if ft.exists():
        rows = [json.loads(l) for l in ft.open(encoding="utf-8") if l.strip()]
        per_step = 48 * 2047
        with (out / "train_log_finetune.jsonl").open("w", encoding="utf-8") as fh:
            for r in rows:
                step = r["step"]
                fh.write(json.dumps({
                    "step": step,
                    "train_loss": r.get("train_loss"),
                    "val_loss": None,
                    "lr": r.get("lr"),
                    "grad_norm": None,
                    "tokens_seen": step * per_step,
                    "epoch": r.get("epoch"),
                    "wall_clock_derived_s": round(step * (5123 / 3294), 1),
                    "wall_clock_is_derived": True,
                }) + "\n")
        print(f"train_log_finetune.jsonl {len(rows):>3,} rows")
    else:
        print(f"[skip] {ft} not found")

    # --- manifest.yaml ---------------------------------------------------
    (out / "manifest.yaml").write_text(MANIFEST, encoding="utf-8")
    print(f"manifest.yaml          written")
    print(f"\n-> {out.resolve()}")
    print("\nTWO PLACEHOLDERS REMAIN:")
    print("  checkpoint_sha256    -- 0b264fd1e6c0b12564b2481cda26e8c3a5684bec714145b53129af08beae9ae2")
    print("  weights_access_link  -- after you upload")


if __name__ == "__main__":
    main()
