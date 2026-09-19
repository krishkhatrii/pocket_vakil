"""
Runtime patches to OLM, applied from our code. **Nothing here edits OLM source.**

The training node has its own OLM install of unknown version. Any change made by
editing `external/olm` would simply not exist there, so every modification we need
is applied at import time to whatever OLM is present.

That trade has a cost: a patch is a bet that an attribute still exists and still
means what we think. `check_olm_api()` settles that bet in the first seconds of
the run rather than in hour six, by asserting every attribute we touch and
refusing to continue if one is missing.

What is patched and why:

  bf16 autocast   OLM's Trainer calls `autocast(device_type, enabled=use_amp)`
                  with no dtype, so CUDA defaults to fp16 with GradScaler. We
                  force bfloat16 and disable the scaler (bf16 needs no loss
                  scaling). fp16 on a 300M model over 2B tokens is a real
                  overflow risk we have no time to debug on the node.

  resume          OLM writes checkpoints but never reads them back; there is no
                  resume of any kind. We attach `load_state`.

  data position   OLM's Trainer always iterates its dataloader from the start,
                  so a resumed run re-shows data. Handled by ResumableSampler in
                  train.py, which the checkpoint saves and restores.

Usage (always in this order):

    import olm_patches
    olm_patches.check_olm_api()          # fails loudly if the API moved
    prec = olm_patches.apply_all()       # returns the precision actually chosen
"""

import importlib
from dataclasses import dataclass

import torch

# (import path, attribute) pairs we monkey-patch or depend on. If the node's OLM
# renames any of these, the run must stop immediately.
REQUIRED_API = [
    ("olm", "__version__"),
    ("olm.train.trainer.trainer", "autocast"),
    ("olm.train.trainer.trainer", "GradScaler"),
    ("olm.train.trainer.trainer", "Trainer"),
    ("olm.train", "Trainer"),
    ("olm.train.optim", "AdamW"),
    ("olm.train.losses", "CrossEntropyLoss"),
    ("olm.train", "TrainerCallback"),
    ("olm.train.callbacks", "MetricsLoggerCallback"),
    ("olm.train.schedulers", "WarmupCosineScheduler"),
    ("olm.data.datasets", "DataLoader"),
    ("olm.nn.blocks.LM", "LM"),
]

# Attributes on Trainer instances that our checkpointing and resume rely on.
REQUIRED_TRAINER_ATTRS = [
    "model", "optimizer", "scaler", "scheduler", "dataloader",
    "global_step", "current_epoch", "losses", "device_type", "use_amp",
    "total_tokens_processed", "training_state", "train",
]

# CrossEntropyLoss must ignore -100 or the fine-tuning mask silently trains on
# the fact text. Cheap to verify, catastrophic to assume.
IGNORE_INDEX = -100

_APPLIED = []


@dataclass
class Precision:
    dtype: torch.dtype
    name: str
    use_amp: bool
    scaler_enabled: bool
    reason: str


def olm_version() -> str:
    import olm
    return getattr(olm, "__version__", "unknown")


def check_olm_api(verbose: bool = True) -> str:
    """Assert every attribute we patch or depend on exists. Fail loudly if not."""
    ver = olm_version()
    missing = []
    for mod_path, attr in REQUIRED_API:
        try:
            mod = importlib.import_module(mod_path)
        except Exception as e:
            missing.append(f"{mod_path} (import failed: {type(e).__name__}: {e})")
            continue
        if not hasattr(mod, attr):
            missing.append(f"{mod_path}.{attr}")

    if not missing:
        from olm.train.trainer.trainer import Trainer
        for attr in REQUIRED_TRAINER_ATTRS:
            names = Trainer.__init__.__code__.co_names
            if not hasattr(Trainer, attr) and attr not in names:
                missing.append(f"Trainer.{attr}")

    if missing:
        raise SystemExit(
            "\n" + "=" * 70 +
            f"\nOLM API MISMATCH -- installed version {ver}\n" + "=" * 70 +
            "\nThese attributes are patched or required by pocket_vakil but are "
            "missing:\n  - " + "\n  - ".join(missing) +
            "\n\nThis build was written against OLM 2.2.0 (submodule eeab195).\n"
            "The node's OLM differs. Refusing to run: a silently-skipped patch "
            "means\nfp16 overflow, a broken resume, or an unmasked fine-tuning "
            "loss.\n" + "=" * 70)

    # Verify the ignore_index contract the fine-tuning mask depends on.
    from olm.train.losses import CrossEntropyLoss
    ce = CrossEntropyLoss()
    logits = torch.zeros(1, 4, 8)
    labels = torch.full((1, 4), IGNORE_INDEX)
    labels[0, -1] = 3
    loss = float(ce(logits, labels))
    if not (loss == loss and loss != 0.0):  # NaN would mean nothing was ignored
        raise SystemExit(
            f"OLM CrossEntropyLoss does not honour ignore_index={IGNORE_INDEX} "
            f"(loss={loss}). The fine-tuning mask would train on fact text.")

    if verbose:
        print(f"[olm_patches] OLM {ver}: API check passed "
              f"({len(REQUIRED_API)} module attrs, "
              f"{len(REQUIRED_TRAINER_ATTRS)} trainer attrs, ignore_index OK)")
    return ver


def select_precision(device: str) -> Precision:
    """bf16 when the card supports it, fp32 loudly when it does not. Never fp16.

    fp16 is excluded on purpose rather than used as a fallback: it needs loss
    scaling, and a scaler that runs away mid-run on an unattended 20h slot is
    exactly the failure we cannot afford. fp32 is slower and always correct.
    """
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        return Precision(torch.float32, "fp32", False, False, "cpu device")

    supported = False
    try:
        supported = torch.cuda.is_bf16_supported()
    except Exception as e:
        print(f"[olm_patches] WARNING is_bf16_supported() raised {type(e).__name__}")

    if supported:
        return Precision(torch.bfloat16, "bf16", True, False, "cuda bf16 supported")

    name = torch.cuda.get_device_name(0)
    print("\n" + "!" * 70)
    print(f"[olm_patches] WARNING: {name} reports NO bf16 support.")
    print("Falling back to fp32. This is SLOWER and may not fit the time budget.")
    print("Deliberately NOT falling back to fp16 -- silent overflow on an")
    print("unattended run is worse than a slow run you can plan around.")
    print("!" * 70 + "\n")
    return Precision(torch.float32, "fp32", False, False,
                     f"{name} lacks bf16 support")


def patch_autocast(dtype: torch.dtype) -> None:
    """Force OLM's Trainer autocast to `dtype`.

    OLM imports `autocast` at module scope in trainer.py and calls it without a
    dtype. Replacing that module-level symbol is the least invasive hook: it does
    not touch Trainer.train's source and survives subclasses (DDP/FSDP trainers
    import the same symbol path).
    """
    import olm.train.trainer.trainer as tt
    if getattr(tt, "_pv_autocast_patched", False):
        return
    original = tt.autocast

    def autocast_with_dtype(device_type, *args, **kwargs):
        kwargs.setdefault("dtype", dtype)
        return original(device_type, *args, **kwargs)

    tt.autocast = autocast_with_dtype
    tt._pv_autocast_patched = True
    _APPLIED.append(f"autocast->{dtype}")


def apply_precision_to_trainer(trainer, prec: Precision) -> None:
    """Disable the GradScaler when using bf16/fp32. Call after Trainer(...)."""
    from torch.amp import GradScaler
    trainer.use_amp = prec.use_amp
    trainer.scaler = GradScaler(trainer.device_type, enabled=prec.scaler_enabled)
    _APPLIED.append(f"scaler_enabled={prec.scaler_enabled}")


def patch_resume() -> None:
    """Attach `Trainer.load_state`. OLM has no resume of its own."""
    from olm.train.trainer.trainer import Trainer
    if hasattr(Trainer, "load_state"):
        return

    def load_state(self, checkpoint: dict, restore_sampler=None):
        """Restore model/optimizer/scaler/scheduler/step from a checkpoint dict.

        Restoring the scheduler's own state is what continues the cosine decay
        instead of restarting it -- the failure mode that is nearly invisible in
        a loss curve.
        """
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if checkpoint.get("optimizer_state_dict"):
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if checkpoint.get("scaler_state_dict") is not None:
            try:
                self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
            except Exception:
                pass  # scaler disabled under bf16; state is meaningless
        if self.scheduler is not None and checkpoint.get("scheduler_state_dict"):
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.global_step = checkpoint.get("step", 0)
        self.current_epoch = checkpoint.get("epoch", 0)
        self.losses = list(checkpoint.get("losses", []))
        self.total_tokens_processed = checkpoint.get("total_tokens_processed", 0)
        if restore_sampler is not None and checkpoint.get("sampler_state"):
            restore_sampler.load_state_dict(checkpoint["sampler_state"])
        return self.global_step

    Trainer.load_state = load_state
    _APPLIED.append("Trainer.load_state")


def apply_all(device: str = "cpu", verbose: bool = True) -> Precision:
    """Run the API check, choose precision, apply every patch. Returns Precision."""
    check_olm_api(verbose=verbose)
    prec = select_precision(device)
    if prec.dtype is not torch.float32:
        patch_autocast(prec.dtype)
    patch_resume()
    if verbose:
        print(f"[olm_patches] precision: {prec.name} ({prec.reason})")
        print(f"[olm_patches] applied: {', '.join(_APPLIED) or 'none'}")
    return prec


def applied_patches():
    return list(_APPLIED)


def environment_report(device: str = "cuda") -> dict:
    """Everything about the node that changes how the run behaves."""
    import platform
    rep = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "olm": olm_version(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": getattr(torch.version, "cuda", None),
        "device_requested": device,
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        rep.update({
            "gpu_name": props.name,
            "gpu_count": torch.cuda.device_count(),
            "vram_total_gb": round(props.total_memory / 1e9, 2),
            "sm_count": props.multi_processor_count,
            "capability": f"{props.major}.{props.minor}",
        })
        try:
            rep["bf16_supported"] = torch.cuda.is_bf16_supported()
        except Exception:
            rep["bf16_supported"] = None
    else:
        rep["gpu_name"] = None
        rep["bf16_supported"] = False
    return rep
