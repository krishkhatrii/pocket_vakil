"""
Training for Pocket Vakil, on OLM's Trainer, with our patches applied at runtime.

    pretrain()   causal LM over the packed uint16 shards
    finetune()   ILSI fact -> IPC label tokens, loss on label tokens only

    python src/train.py pretrain --config prod_300m --run-id pretrain-s1
    python src/train.py pretrain --config prod_300m --run-id pretrain-s1 --resume
    python src/train.py finetune --config prod_300m --run-id ft-s2 \
        --init-from checkpoints/pretrain-s1/step_20000.pt

Nothing in OLM's source is edited. `olm_patches.apply_all()` runs first and aborts
the process if the node's OLM does not expose what we patch.

FINE-TUNING MASK. OLM's Trainer yields `(input_ids, labels)` and calls
`self.loss(logits, labels)` with two arguments -- there is no third slot for a
mask. OLM's `MaskedCELoss` takes one but its signature is `(logits, y, mask)`, so
it cannot be used without patching the Trainer's call site. The supported route is
`ignore_index`: OLM's default `CrossEntropyLoss` already passes
`ignore_index=-100`, so the dataset writes -100 at every non-label position and
masking falls out of OLM's own loss.

DATA-POSITION RESUME. OLM's Trainer always iterates its dataloader from the start,
so a naive resume re-shows data already trained on -- which on a 3-epoch budget
quietly becomes a 4th partial epoch over part of the corpus. `ResumableSampler`
owns the epoch permutation and a cursor into it; both are checkpointed.
"""

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

import olm_patches

from olm.data.datasets import DataLoader
from olm.train import Trainer
from olm.train import TrainerCallback          # exported from olm.train.trainer
from olm.train.callbacks import MetricsLoggerCallback
from olm.train.losses import CrossEntropyLoss
from olm.train.optim import AdamW

from model import CONFIGS, build_model, param_breakdown, EOS_ID

EXPECTED_TOKENIZER_SHA = "b2de5ea09885462611a73045cc98d88f670f0893ce7518bc6e36d1895dde8203"
TOKENIZER = Path("data/tokenizer/bpe_32000_final.json")
SHARDS = Path("data/shards")
ILSI_FT = Path("data/ilsi_ft")
RUNS = Path("runs")
DEFAULT_CKPT_DIR = Path("checkpoints")      # relative to project root

IGNORE_INDEX = -100


# --------------------------------------------------------------------------
# guards


def verify_tokenizer(check_shards: bool) -> str:
    """Refuse to run against a vocabulary that did not produce the data.

    A mismatch does not crash -- it trains a healthy-looking model on token ids
    that mean something else. First thing that runs, on both stages.
    """
    if not TOKENIZER.exists():
        raise SystemExit(f"TOKENIZER MISSING: {TOKENIZER}")
    actual = hashlib.sha256(TOKENIZER.read_bytes()).hexdigest()
    if actual != EXPECTED_TOKENIZER_SHA:
        raise SystemExit(
            "\n" + "=" * 70 +
            "\nTOKENIZER HASH MISMATCH -- refusing to run\n" + "=" * 70 +
            f"\n  expected {EXPECTED_TOKENIZER_SHA}\n  actual   {actual}\n"
            "Token ids would not match the packed shards.\n" + "=" * 70)
    if check_shards:
        man_path = SHARDS / "manifest.json"
        if not man_path.exists():
            raise SystemExit(f"SHARD MANIFEST MISSING: {man_path}")
        man = json.loads(man_path.read_text(encoding="utf-8"))
        got = man["tokenizer"]["sha256"]
        if got != EXPECTED_TOKENIZER_SHA:
            raise SystemExit(
                "\nSHARD MANIFEST TOKENIZER MISMATCH -- refusing to run\n"
                f"  manifest {got}\n  expected {EXPECTED_TOKENIZER_SHA}\n"
                "The shards were packed with a different vocabulary.")
    print(f"[verify] tokenizer sha256 {actual[:16]}... OK"
          f"{' + shard manifest OK' if check_shards else ''}")
    return actual


# --------------------------------------------------------------------------
# resumable sampling


class ResumableSampler(Sampler):
    """Deterministic per-epoch permutation with a checkpointable cursor.

    OLM's Trainer restarts its dataloader every epoch and has no notion of where
    a run left off, so the position has to live somewhere we control. Given
    (seed, epoch) the permutation is reproducible, so a checkpoint only needs the
    epoch and an index into it.

    On resume the current epoch yields `perm[cursor:]`; every later epoch yields
    its full permutation. Iterating advances `cursor` so a checkpoint written
    mid-epoch records exactly how far the run got.
    """

    def __init__(self, n: int, seed: int = 1234, shuffle: bool = True):
        self.n, self.seed, self.shuffle = n, seed, shuffle
        self.epoch, self.cursor = 0, 0

    def _perm(self):
        if not self.shuffle:
            return np.arange(self.n)
        return np.random.default_rng(self.seed + self.epoch).permutation(self.n)

    def __iter__(self):
        perm = self._perm()
        start = self.cursor
        for i in range(start, self.n):
            self.cursor = i + 1
            yield int(perm[i])
        self.epoch += 1
        self.cursor = 0

    def __len__(self):
        return max(self.n - self.cursor, 0)

    def peek_next(self):
        """The index the next __iter__ would yield first. Used by resume tests."""
        return int(self._perm()[self.cursor]) if self.cursor < self.n else None

    def state_dict(self):
        return {"epoch": self.epoch, "cursor": self.cursor,
                "seed": self.seed, "n": self.n}

    def load_state_dict(self, s):
        if s.get("n") != self.n or s.get("seed") != self.seed:
            print(f"[sampler] WARNING checkpoint sampler (n={s.get('n')}, "
                  f"seed={s.get('seed')}) differs from current "
                  f"(n={self.n}, seed={self.seed}); position may not align")
        self.epoch, self.cursor = s["epoch"], s["cursor"]


# --------------------------------------------------------------------------
# datasets -- both yield (input_ids, labels), OLM's Trainer contract


class PretrainShardDataset(Dataset):
    """Packed uint16 shards as (input_ids, labels) pairs.

    Shards are memory-mapped: the pack does not need to be resident, and the OS
    page cache handles reuse across epochs better than anything here would.
    """

    def __init__(self, shard_dir: Path, context_length: int):
        man = json.loads((shard_dir / "manifest.json").read_text(encoding="utf-8"))
        self.packed_len = man["seq_len"]
        if context_length > self.packed_len:
            raise SystemExit(
                f"context_length {context_length} exceeds packed seq_len "
                f"{self.packed_len}")
        # Next-token prediction needs one more token than it predicts. A packed
        # row of 2048 therefore yields 2047 inputs and 2047 targets: the final
        # token is a target only. Model capacity stays context_length; the
        # training example is one token shorter, which is the standard trade and
        # costs 0.05% of positions.
        self.take = min(context_length + 1, self.packed_len)
        self.context_length = self.take - 1
        self.arrays = [np.load(shard_dir / s["file"], mmap_mode="r")
                       for s in man["shards"]]
        self.offsets = np.cumsum([0] + [s["sequences"] for s in man["shards"]])
        self.total = int(self.offsets[-1])
        self.total_tokens = man["total_tokens_packed"]

    def __len__(self):
        return self.total

    def __getitem__(self, i):
        s = int(np.searchsorted(self.offsets, i, side="right") - 1)
        row = self.arrays[s][i - self.offsets[s]]
        w = np.asarray(row[: self.take], dtype=np.int64)
        return torch.from_numpy(w[:-1].copy()), torch.from_numpy(w[1:].copy())


class ILSIFinetuneDataset(Dataset):
    """ILSI (fact -> IPC label tokens) with the loss confined to label tokens.

    Sequence is `input_ids + target_ids`. With next-token prediction on
    x = seq[:-1], y = seq[1:], position i predicts seq[i+1], so a label-token
    prediction lives at i >= n_input - 1. That off-by-one is load-bearing: at
    i = n_input - 1 the model sees the last fact token and must emit the first
    label, the transition being taught. Everything earlier is IGNORE_INDEX.
    """

    def __init__(self, path: Path, context_length: int):
        self.rows = [json.loads(l) for l in path.open(encoding="utf-8")]
        self.context_length = context_length
        self.n_truncated = sum(
            1 for r in self.rows
            if len(r["input_ids"]) + len(r["target_ids"]) > context_length + 1)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        tgt = r["target_ids"]
        # Keep the whole target, trim the fact: dropping label tokens would
        # silently change the supervision.
        inp = r["input_ids"][: self.context_length + 1 - len(tgt)]
        seq = inp + tgt
        n_in = len(inp)
        x = np.full(self.context_length, EOS_ID, dtype=np.int64)
        y = np.full(self.context_length, IGNORE_INDEX, dtype=np.int64)
        xs = np.asarray(seq[:-1], dtype=np.int64)
        ys = np.asarray(seq[1:], dtype=np.int64)
        x[: len(xs)] = xs
        start = max(n_in - 1, 0)
        y[start: len(ys)] = ys[start:]
        return torch.from_numpy(x), torch.from_numpy(y)


# --------------------------------------------------------------------------
# checkpointing


class PocketCheckpointCallback(TrainerCallback):
    """Checkpoints with sampler position, plus a size-aware retention policy.

    OLM's own CheckpointCallback saves neither the dataloader position nor RNG
    state, and keeps N most-recent by count. At 300M parameters a checkpoint is
    ~4 GB (weights + AdamW moments), and node disk is unknown, so retention is:

      * the last `keep_last` step checkpoints (rolling, default 2)
      * one milestone every `milestone_hours` (default 2h), never deleted

    Milestones are what let a run be salvaged if the last two checkpoints turn
    out to straddle a divergence.
    """

    def __init__(self, ckpt_dir: Path, save_every: int, sampler=None,
                 keep_last: int = 2, milestone_hours: float = 2.0,
                 run_id: str = "run", extra: dict = None,
                 verify_first: bool = True, abort_on_bad_first: bool = True):
        self.dir = Path(ckpt_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.save_every = save_every
        self.sampler = sampler
        self.keep_last = keep_last
        self.milestone_seconds = milestone_hours * 3600
        self.run_id = run_id
        self.extra = extra or {}
        self.start = time.time()
        self.last_milestone = self.start
        self.verify_first = verify_first
        self.abort_on_bad_first = abort_on_bad_first
        self.verified = False

    def on_step_end(self, trainer, step: int, loss: float) -> None:
        if step % self.save_every != 0:
            return
        now = time.time()
        is_milestone = (now - self.last_milestone) >= self.milestone_seconds
        path = self.dir / (f"milestone_{step}.pt" if is_milestone
                           else f"step_{step}.pt")
        self._save(trainer, step, path, loss)

        # Verify the FIRST checkpoint by reading it back. A checkpoint that
        # cannot be loaded is worth knowing about at minute 30, not at hour 12
        # when it is the only thing standing between a crash and starting over.
        if self.verify_first and not self.verified:
            self.verified = True
            ok, report = verify_checkpoint_file(path, trainer.model)
            (self.dir / "first_checkpoint_verification.json").write_text(
                json.dumps(report, indent=2, default=str), encoding="utf-8")
            if ok:
                print(f"[ckpt-verify] PASS {path.name}: "
                      f"{report['tensors']} tensors, "
                      f"{report['params']:,} params, all finite, "
                      f"reload {report['reload_seconds']:.1f}s")
            else:
                msg = ("\n" + "!" * 70 +
                       f"\n[ckpt-verify] FAIL on the FIRST checkpoint {path.name}\n" +
                       "\n".join(f"  - {p}" for p in report["problems"]) +
                       "\nCheckpointing is broken. Continuing would spend the whole "
                       "slot\nproducing a model that cannot be resumed or "
                       "evaluated.\n" + "!" * 70)
                print(msg, flush=True)
                if self.abort_on_bad_first:
                    raise SystemExit(msg)

        if is_milestone:
            self.last_milestone = now
            print(f"[ckpt] MILESTONE {path.name} (kept permanently)")
        else:
            self._prune()

    def _save(self, trainer, step, path, loss):
        payload = {
            "step": step,
            "epoch": trainer.current_epoch,
            "loss": loss,
            "model_state_dict": trainer.model.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "scaler_state_dict": trainer.scaler.state_dict(),
            "losses": trainer.losses,
            "total_tokens_processed": trainer.total_tokens_processed,
            "sampler_state": self.sampler.state_dict() if self.sampler else None,
            "rng": {
                "torch": torch.get_rng_state(),
                "numpy": np.random.get_state(),
                "cuda": (torch.cuda.get_rng_state_all()
                         if torch.cuda.is_available() else None),
            },
            "tokenizer_sha256": EXPECTED_TOKENIZER_SHA,
            "run_id": self.run_id,
            **self.extra,
        }
        if trainer.scheduler is not None:
            payload["scheduler_state_dict"] = trainer.scheduler.state_dict()
        tmp = path.with_suffix(".tmp")
        torch.save(payload, tmp)
        tmp.replace(path)      # atomic: a crash never leaves a plausible-looking
        (self.dir / "latest.txt").write_text(path.name, encoding="utf-8")
        print(f"[ckpt] {path.name} step {step:,} loss {loss:.4f} "
              f"({path.stat().st_size/1e9:.2f} GB)", flush=True)

    def _prune(self):
        steps = sorted(self.dir.glob("step_*.pt"),
                       key=lambda p: int(p.stem.split("_")[1]))
        for old in steps[: -self.keep_last] if self.keep_last else []:
            old.unlink()
            print(f"[ckpt] pruned {old.name}")


def verify_checkpoint_file(path: Path, model) -> tuple:
    """Read a checkpoint back and prove it is usable. Returns (ok, report).

    Checks, in the order they would bite:
      * the file loads at all (truncation, disk-full, interrupted write)
      * every key the resume path reads is present
      * the state dict matches the live model exactly -- no missing or extra
        tensors, so `load_state_dict(strict=True)` will succeed later
      * every weight tensor is finite (NaN/Inf from a diverged step or from
        memory corruption)
      * the optimiser carries real state, not an empty shell

    A checkpoint that fails any of these is not a checkpoint.
    """
    report = {"path": str(path), "problems": [], "tensors": 0, "params": 0}
    t0 = time.time()
    try:
        ck = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        report["problems"].append(f"torch.load failed: {type(e).__name__}: {e}")
        return False, report
    report["reload_seconds"] = time.time() - t0
    report["size_gb"] = path.stat().st_size / 1e9

    for key in ("model_state_dict", "optimizer_state_dict", "step", "sampler_state"):
        if key not in ck:
            report["problems"].append(f"missing key: {key}")
    sd = ck.get("model_state_dict")
    if not isinstance(sd, dict) or not sd:
        report["problems"].append("model_state_dict missing or empty")
        return False, report

    live = model.state_dict()
    missing = set(live) - set(sd)
    extra = set(sd) - set(live)
    if missing:
        report["problems"].append(f"{len(missing)} tensors missing "
                                  f"(e.g. {sorted(missing)[:3]})")
    if extra:
        report["problems"].append(f"{len(extra)} unexpected tensors "
                                  f"(e.g. {sorted(extra)[:3]})")

    nonfinite, shape_mismatch = [], []
    n_params = 0
    for k, v in sd.items():
        if not torch.is_tensor(v):
            continue
        report["tensors"] += 1
        n_params += v.numel()
        if k in live and tuple(v.shape) != tuple(live[k].shape):
            shape_mismatch.append(f"{k}: {tuple(v.shape)} vs {tuple(live[k].shape)}")
        if v.is_floating_point() and not torch.isfinite(v).all():
            nonfinite.append(k)
    report["params"] = n_params
    if shape_mismatch:
        report["problems"].append(f"shape mismatch: {shape_mismatch[:3]}")
    if nonfinite:
        report["problems"].append(
            f"{len(nonfinite)} tensors contain NaN/Inf (e.g. {nonfinite[:3]})")

    opt = ck.get("optimizer_state_dict") or {}
    if not opt.get("state"):
        report["problems"].append(
            "optimizer_state_dict has no per-parameter state; resume would "
            "restart Adam moments from zero")

    report["ok"] = not report["problems"]
    return report["ok"], report


def save_final_model(model, path: Path, cfg, extra: dict = None):
    """Weights-only export -- the file to download first when time is short.

    No optimiser moments, so it is a third the size of a training checkpoint and
    is all that `evaluate.py` needs. Carries the config and tokenizer hash so the
    weights can never be silently paired with the wrong vocabulary.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "config": cfg.to_dict(),
        "config_name": cfg.name,
        "tokenizer_sha256": EXPECTED_TOKENIZER_SHA,
        "saved": time.strftime("%Y-%m-%d %H:%M:%S"),
        "note": "weights only; no optimizer state. Load with evaluate.py --ckpt",
        **(extra or {}),
    }
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)
    print(f"[final] {path} ({path.stat().st_size/1e9:.2f} GB, weights only)")
    return path


def find_latest_checkpoint(ckpt_dir: Path):
    """Newest checkpoint by step, milestones included."""
    if not ckpt_dir.exists():
        return None
    cands = list(ckpt_dir.glob("step_*.pt")) + list(ckpt_dir.glob("milestone_*.pt"))
    if not cands:
        return None
    return max(cands, key=lambda p: int(p.stem.split("_")[1]))


def _as_byte_states(states):
    """Coerce a sequence of CUDA RNG states to CPU uint8 tensors, or None.

    torch requires ByteTensors here; a checkpoint written by another torch build
    can carry a different dtype. Returns None if coercion itself fails.
    """
    try:
        return [s.detach().cpu().to(torch.uint8) if torch.is_tensor(s)
                else torch.as_tensor(s, dtype=torch.uint8) for s in states]
    except Exception as e:
        print(f"[rng] WARNING cuda RNG states could not be coerced to uint8 "
              f"({type(e).__name__}: {e})", flush=True)
        return None


def restore_rng(ck):
    """Restore RNG state, best-effort. **Never fatal.**

    A resume that dies here is holding a valid model, optimizer and data position
    and refusing to use them -- the worst trade available on a timed slot. RNG
    state governs dropout and sampling draws from this point forward, so losing it
    costs exact reproducibility, not correctness. Each generator is restored
    independently so one bad entry cannot take the others down with it.

    What is *saved* is unchanged; this only makes reading it back forgiving.
    """
    r = ck.get("rng") or {}

    def attempt(name, fn):
        try:
            fn()
            return True
        except Exception as e:
            print(f"[rng] WARNING could not restore {name} RNG "
                  f"({type(e).__name__}: {e}); continuing with the current state",
                  flush=True)
            return False

    # Not written by _save today; restored if a future checkpoint carries it.
    if r.get("python") is not None:
        attempt("python", lambda: random.setstate(r["python"]))

    if r.get("torch") is not None:
        attempt("torch cpu", lambda: torch.set_rng_state(
            r["torch"].cpu() if torch.is_tensor(r["torch"]) else r["torch"]))

    if r.get("numpy") is not None:
        attempt("numpy", lambda: np.random.set_state(r["numpy"]))

    # `is not None`, not truthiness: bool() on a multi-element tensor raises, and
    # a malformed checkpoint could put a bare tensor here instead of a list.
    if r.get("cuda") is not None and torch.cuda.is_available():
        states = r["cuda"]
        n_dev = torch.cuda.device_count()
        if not isinstance(states, (list, tuple)):
            print(f"[rng] WARNING cuda RNG state is {type(states).__name__}, "
                  f"expected a list; skipping cuda RNG only", flush=True)
        elif len(states) != n_dev:
            print(f"[rng] WARNING checkpoint holds {len(states)} cuda RNG state(s) "
                  f"but this node has {n_dev} device(s); skipping cuda RNG only",
                  flush=True)
        else:
            prepared = _as_byte_states(states)
            if prepared is not None and not attempt(
                    "cuda", lambda: torch.cuda.set_rng_state_all(list(prepared))):
                retry = _as_byte_states(states)
                if retry is not None:
                    attempt("cuda (retry after uint8 coercion)",
                            lambda: torch.cuda.set_rng_state_all(list(retry)))


def cosine_lr_at(step, peak_lr, warmup_steps, total_steps, min_lr):
    """LR the scheduler emits when `last_epoch == step`.

    Mirrors WarmupCosineScheduler.get_lr (olm/train/schedulers/warmup.py:102-119)
    exactly, including the `+1` in the warmup factor and the clamp of progress at
    1.0. Used only to preview the schedule; the scheduler itself still computes
    every LR it applies.
    """
    import math
    if step < warmup_steps:
        return peak_lr * (step + 1) / max(1, warmup_steps)
    decay = max(1, total_steps - warmup_steps)
    progress = min((step - warmup_steps) / decay, 1.0)
    return min_lr + (peak_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))


def restart_cosine_schedule(trainer, peak_lr, warmup_steps, total_steps, min_lr):
    """Replace the restored cosine with a fresh one. Only called for --restart-cosine.

    `load_state` restores the scheduler via `self.__dict__.update(state_dict)`
    (olm/train/schedulers/base.py:93), so the checkpoint's base_lrs, warmup_steps,
    total_steps and last_epoch overwrite whatever the CLI just built. This runs
    AFTER that restore and puts the CLI values back.

    param_group['lr'] is rewritten too, because the trainer calls
    optimizer.step() BEFORE scheduler.step() (trainer.py:361-365): without this
    the first update after resume would run at the checkpoint's decayed LR.
    Optimizer moment state (exp_avg/exp_avg_sq) is not touched, nor is the sampler.
    """
    s = trainer.scheduler
    if s is None:
        raise SystemExit("--restart-cosine: trainer has no scheduler to restart")
    groups = trainer.optimizer.param_groups
    s.base_lrs = [peak_lr] * len(groups)     # a list, one entry per param group
    s.warmup_steps = warmup_steps
    s.total_steps = total_steps
    s.min_lr = min_lr
    s.last_epoch = 0
    s._step_count = 1                        # mirror a freshly-built scheduler
    lr0 = cosine_lr_at(0, peak_lr, warmup_steps, total_steps, min_lr)
    s._last_lr = [lr0] * len(groups)
    for g in groups:
        g["lr"] = lr0
        g["initial_lr"] = peak_lr
    return lr0


def print_schedule(peak_lr, warmup_steps, total_steps, min_lr, run_max_steps):
    """Preview the fresh curve. --restart-total-steps is the cosine horizon only;
    the run still stops at the computed max_steps, so say so when they differ."""
    print(f"[restart-cosine] peak {peak_lr:.3e} | warmup {warmup_steps:,} | "
          f"total {total_steps:,} | min {min_lr:.3e}")
    for k in (0, 100, 1000, total_steps):
        tag = "warmup" if k < warmup_steps else "cosine"
        print(f"[restart-cosine]   step {k:>7,}  lr "
              f"{cosine_lr_at(k, peak_lr, warmup_steps, total_steps, min_lr):.6e}"
              f"  ({tag})")
    if run_max_steps < total_steps:
        print(f"[restart-cosine] NOTE run stops at max_steps {run_max_steps:,}, "
              f"but the cosine is drawn over {total_steps:,} steps -- the curve "
              f"will be truncated")
    elif run_max_steps > total_steps:
        print(f"[restart-cosine] NOTE run continues to {run_max_steps:,} steps, "
              f"past the cosine horizon {total_steps:,}; LR pins at min_lr after "
              f"that (progress clamps at 1.0)")


# --------------------------------------------------------------------------
# shared plumbing


def build_trainer(model, dataloader, cfg, args, total_steps, callbacks, prec):
    trainer = Trainer(
        model=model,
        optimizer=AdamW,
        dataloader=dataloader,
        device=args.device,
        context_length=cfg.max_seq_len,
        grad_accum_steps=args.grad_accum,
        use_amp=prec.use_amp,
        loss=CrossEntropyLoss,          # ignore_index=-100 -> fine-tuning mask
        callbacks=callbacks,
        grad_clip_norm=args.grad_clip,
        warmup_steps=args.warmup_steps,
        total_steps=total_steps,
        min_lr=args.lr * args.min_lr_ratio,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        use_warmup_cosine=True,
    )
    olm_patches.apply_precision_to_trainer(trainer, prec)
    return trainer


def load_weights(model, path: Path, device: str):
    ck = torch.load(path, map_location=device, weights_only=False)
    if isinstance(ck, dict) and ck.get("tokenizer_sha256") not in (
            None, EXPECTED_TOKENIZER_SHA):
        raise SystemExit(f"{path} was trained under a different tokenizer")
    state = ck.get("model_state_dict", ck) if isinstance(ck, dict) else ck
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[init] WARNING missing={list(missing)[:4]} "
              f"unexpected={list(unexpected)[:4]}")
    print(f"[init] weights from {path} "
          f"(step {ck.get('step') if isinstance(ck, dict) else '?'})")
    return ck


def ckpt_dir_for(args, run_id) -> Path:
    return Path(args.checkpoint_dir) / run_id


# --------------------------------------------------------------------------
# stages


def _run(args, stage):
    cfg = CONFIGS[args.config]
    prec = olm_patches.apply_all(device=args.device)
    verify_tokenizer(check_shards=(stage == "pretrain"))

    if stage == "pretrain":
        ds = PretrainShardDataset(SHARDS, cfg.max_seq_len)
        print(f"[data] {len(ds):,} sequences x {cfg.max_seq_len} "
              f"({len(ds)*cfg.max_seq_len/1e9:.2f}B tokens/epoch)")
    else:
        ds = ILSIFinetuneDataset(ILSI_FT / "train.jsonl", cfg.max_seq_len)
        print(f"[data] ILSI train {len(ds):,} examples "
              f"({ds.n_truncated:,} need input truncation)")

    sampler = ResumableSampler(len(ds), seed=args.seed, shuffle=True)
    loader = DataLoader(ds, batch_size=args.batch_size, sampler=sampler,
                        shuffle=False, num_workers=args.num_workers,
                        drop_last=True, pin_memory=(args.device != "cpu"))

    model = build_model(cfg)
    pc = param_breakdown(model)
    print(f"[model] {cfg.name}: {pc['non_embedding']/1e6:.1f}M non-embedding, "
          f"{pc['total']/1e6:.1f}M total")

    steps_per_epoch = max(len(ds) // (args.batch_size * args.grad_accum), 1)
    total_steps = args.max_steps or int(steps_per_epoch * args.epochs)
    print(f"[sched] {args.epochs} epochs, {steps_per_epoch:,} steps/epoch, "
          f"{total_steps:,} total steps")

    if stage == "finetune" and args.init_from:
        load_weights(model, Path(args.init_from), args.device)
    elif stage == "finetune":
        print("[init] WARNING no --init-from; fine-tuning from random init")

    cdir = ckpt_dir_for(args, args.run_id)
    ckpt_cb = PocketCheckpointCallback(
        cdir, save_every=args.ckpt_every, sampler=sampler,
        keep_last=args.keep_last, milestone_hours=args.milestone_hours,
        run_id=args.run_id, extra={"config": cfg.name, "stage": stage},
        verify_first=args.verify_first, abort_on_bad_first=True)
    callbacks = [ckpt_cb,
                 MetricsLoggerCallback(log_dir=str(RUNS / args.run_id / "metrics"),
                                       log_every=args.log_every)]

    trainer = build_trainer(model, loader, cfg, args, total_steps, callbacks, prec)

    if args.restart_cosine and not args.resume:
        raise SystemExit("--restart-cosine requires --resume (there is no restored "
                         "schedule to replace on a fresh run)")

    if args.resume:
        latest = find_latest_checkpoint(cdir)
        if latest:
            # Build the scheduler before restoring, so its state has somewhere to
            # land; epochs=0 constructs it without consuming any data.
            trainer.train(epochs=0)
            ck = torch.load(latest, map_location=args.device, weights_only=False)
            step = trainer.load_state(ck, restore_sampler=sampler)
            restore_rng(ck)
            print(f"[resume] {latest.name}: step {step:,}, "
                  f"sampler epoch {sampler.epoch} cursor {sampler.cursor:,}, "
                  f"lr {trainer.optimizer.param_groups[0]['lr']:.3e}")
            if args.restart_cosine:
                r_peak = (args.restart_peak_lr if args.restart_peak_lr is not None
                          else args.lr)
                r_warm = (args.restart_warmup_steps
                          if args.restart_warmup_steps is not None
                          else args.warmup_steps)
                r_total = (args.restart_total_steps
                           if args.restart_total_steps is not None else total_steps)
                r_min = (args.restart_min_lr if args.restart_min_lr is not None
                         else args.lr * args.min_lr_ratio)
                lr0 = restart_cosine_schedule(trainer, r_peak, r_warm, r_total, r_min)
                print(f"[restart-cosine] discarded checkpoint schedule; "
                      f"last_epoch reset to 0, lr now {lr0:.3e}")
                print_schedule(r_peak, r_warm, r_total, r_min, total_steps)
        else:
            print(f"[resume] no checkpoint in {cdir}; starting fresh")

    write_run_header(RUNS / args.run_id, args.run_id, cfg, args, prec, stage, ds)
    trainer.train(epochs=int(args.epochs), log_interval=args.log_every,
                  max_steps=total_steps)

    if stage == "finetune":
        # Written unconditionally at the end of fine-tuning: this is the artefact
        # worth rescuing first if the slot is running out.
        save_final_model(
            trainer.model, Path(args.final_model or (cdir / "final_model.pt")), cfg,
            extra={"run_id": args.run_id, "stage": stage,
                   "steps": trainer.global_step,
                   "final_loss": trainer.losses[-1] if trainer.losses else None,
                   "init_from": args.init_from})
    return trainer


def write_run_header(run_dir, run_id, cfg, args, prec, stage, ds):
    run_dir.mkdir(parents=True, exist_ok=True)
    md = run_dir / f"{run_id}.md"
    if md.exists():
        return
    md.write_text("\n".join([
        f"# Run: {run_id}", "",
        "| | |", "|---|---|",
        f"| Stage | {stage} |",
        f"| Date | {time.strftime('%Y-%m-%d %H:%M')} |",
        f"| Preset | {cfg.name} ({cfg.embed_dim}d/{cfg.num_layers}L/{cfg.num_heads}H) |",
        f"| Context | {cfg.max_seq_len} |",
        f"| Precision | {prec.name} ({prec.reason}) |",
        f"| OLM | {olm_patches.olm_version()} |",
        f"| Patches | {', '.join(olm_patches.applied_patches())} |",
        f"| Tokenizer sha256 | {EXPECTED_TOKENIZER_SHA} |",
        f"| Examples | {len(ds):,} |",
        f"| Batch / accum | {args.batch_size} / {args.grad_accum} |",
        f"| LR / warmup / clip | {args.lr} / {args.warmup_steps} / {args.grad_clip} |",
        f"| Checkpoint dir | {ckpt_dir_for(args, run_id)} |",
        "", "## Loss curve", "",
        "| Step | Tokens seen | Train loss | LR | Wall clock | Checkpoint |",
        "|---|---|---|---|---|---|", "",
        "## What broke", "", "| What | Symptom | Cause | Fix |", "|---|---|---|---|", "",
    ]), encoding="utf-8")


def pretrain(args):
    return _run(args, "pretrain")


def finetune(args):
    return _run(args, "finetune")


def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["pretrain", "finetune"])
    ap.add_argument("--config", default="prod_300m", choices=list(CONFIGS))
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--min-lr-ratio", type=float, default=0.1)
    ap.add_argument("--warmup-steps", type=int, default=500)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--checkpoint-dir", default=str(DEFAULT_CKPT_DIR))
    ap.add_argument("--ckpt-every", type=int, default=500)
    ap.add_argument("--keep-last", type=int, default=2)
    ap.add_argument("--milestone-hours", type=float, default=2.0)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--restart-cosine", action="store_true",
                    help="on resume, discard the checkpoint's LR schedule and "
                         "start a fresh warmup+cosine (requires --resume)")
    ap.add_argument("--restart-peak-lr", type=float, default=None,
                    help="new peak LR for --restart-cosine (default: --lr)")
    ap.add_argument("--restart-warmup-steps", type=int, default=None,
                    help="new warmup length (default: --warmup-steps)")
    ap.add_argument("--restart-total-steps", type=int, default=None,
                    help="new cosine horizon, NOT the run's stopping condition "
                         "(default: the run's computed total)")
    ap.add_argument("--restart-min-lr", type=float, default=None,
                    help="new LR floor (default: --lr * --min-lr-ratio)")
    ap.add_argument("--init-from", type=str, default=None)
    ap.add_argument("--final-model", type=str, default=None,
                    help="weights-only export path (finetune); "
                         "default <checkpoint-dir>/<run-id>/final_model.pt")
    ap.add_argument("--no-verify-first-ckpt", dest="verify_first",
                    action="store_false", default=True,
                    help="skip reading back the first checkpoint (not advised)")
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    return ap


def main():
    args = build_argparser().parse_args()
    if args.stage == "finetune":
        # ILSI is ~43k examples; pretrain defaults would be far too aggressive.
        if args.lr == 6e-4:
            args.lr = 5e-5
        if args.warmup_steps == 500:
            args.warmup_steps = 100
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    (pretrain if args.stage == "pretrain" else finetune)(args)


if __name__ == "__main__":
    main()
