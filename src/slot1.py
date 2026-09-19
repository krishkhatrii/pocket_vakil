"""
Pre-flight tools for the production slot. Measures the node; assumes nothing.

    python src/slot1.py env                     # GPU, VRAM, bf16, torch, OLM, patches
    python src/slot1.py verify                  # tokenizer hash + shard manifest
    python src/slot1.py probe --config prod_300m
    python src/slot1.py smoke --config prod_300m --batch-size N --minutes 20

`probe` finds the largest batch size that fits by actually running forward+backward
and catching OOM, then reports peak memory. `smoke` measures real tokens/sec on
whatever card is present and extrapolates the full pretrain wall-clock from that
measurement -- no assumed hardware, no assumed MFU.

If the extrapolation does not fit the slot, `smoke` prints the token count that
WOULD fit, so the decision is a number at minute 30 rather than a surprise at
hour 19.
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

import olm_patches
from model import CONFIGS, build_model, param_breakdown
from train import (EXPECTED_TOKENIZER_SHA, SHARDS, ILSI_FT, PretrainShardDataset,
                   ResumableSampler, verify_tokenizer)

from olm.data.datasets import DataLoader
from olm.train.losses import CrossEntropyLoss
from olm.train.optim import AdamW

# Dense bf16/fp16 peak TFLOPs, for MFU only. Unknown cards report MFU as None
# rather than guessing -- a wrong MFU is worse than no MFU.
PEAK_TFLOPS = {
    "H100": 989.0, "H200": 989.0, "A100": 312.0, "L40S": 362.0, "L40": 181.0,
    "A6000": 155.0, "A40": 149.0, "RTX 4090": 165.0, "V100": 125.0,
    "RTX 3090": 71.0, "T4": 65.0,
}


def peak_tflops_for(gpu_name):
    if not gpu_name:
        return None
    for key, val in PEAK_TFLOPS.items():
        if key.lower().replace(" ", "") in gpu_name.lower().replace(" ", ""):
            return val
    return None


def cmd_env(args):
    rep = olm_patches.environment_report(args.device)
    prec = olm_patches.apply_all(device=args.device)
    rep["precision"] = prec.name
    rep["precision_reason"] = prec.reason
    rep["patches_applied"] = olm_patches.applied_patches()
    rep["peak_tflops_known"] = peak_tflops_for(rep.get("gpu_name"))

    print("\n" + "=" * 66)
    print("ENVIRONMENT REPORT")
    print("=" * 66)
    for k, v in rep.items():
        print(f"  {k:22s} {v}")
    if rep["cuda_available"] and rep["peak_tflops_known"] is None:
        print(f"\n  NOTE: {rep.get('gpu_name')} is not in the peak-TFLOPs table, "
              "so MFU\n        cannot be computed. tokens/sec and the wall-clock "
              "extrapolation\n        are measured directly and remain valid.")
    print("=" * 66)
    if args.out:
        Path(args.out).write_text(json.dumps(rep, indent=2, default=str),
                                  encoding="utf-8")
    return rep


def checkpoint_bytes(total_params: int) -> int:
    """Bytes one training checkpoint costs on disk.

    fp32 weights (4B/param) plus AdamW's two moments (8B/param) = 12B/param.
    Gradients are not saved. Tied embeddings are a single tensor, so the optimiser
    keeps one moment pair for them, not two -- which is why this is computed from
    the real parameter count rather than a remembered GB figure.
    """
    return total_params * 12


def cmd_disk(args):
    """Does the checkpoint retention policy fit the free space on this node?"""
    import shutil as _shutil

    cfg = CONFIGS[args.config]
    model = build_model(cfg)
    pc = param_breakdown(model)
    per_ckpt = checkpoint_bytes(pc["total"])
    final_only = pc["total"] * 4          # final_model.pt, weights only

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    usage = _shutil.disk_usage(ckpt_dir)
    free = usage.free

    # Milestones are never pruned, so they are what grows without bound.
    n_milestones = max(int(math.ceil(args.train_hours / args.milestone_hours)), 1)
    rolling = args.keep_last * per_ckpt
    milestones = n_milestones * per_ckpt
    # Worst instant: rolling set + every milestone + the temp file of an
    # in-flight write + the final weights-only export.
    peak = rolling + milestones + per_ckpt + final_only
    need_gb = peak / 1e9

    print("\n" + "=" * 66)
    print("DISK SPACE vs CHECKPOINT RETENTION")
    print("=" * 66)
    print(f"  checkpoint dir       {ckpt_dir.resolve()}")
    print(f"  filesystem free      {free/1e9:.1f} GB of {usage.total/1e9:.1f} GB")
    print(f"  model                {cfg.name}, {pc['total']/1e6:.1f}M params")
    print(f"  per checkpoint       {per_ckpt/1e9:.2f} GB "
          f"(weights {pc['total']*4/1e9:.2f} + AdamW moments {pc['total']*8/1e9:.2f})")
    print(f"  final_model.pt       {final_only/1e9:.2f} GB (weights only)")
    print("-" * 66)
    print(f"  policy               keep_last={args.keep_last}, "
          f"milestone every {args.milestone_hours}h over {args.train_hours}h "
          f"= {n_milestones} milestones")
    print(f"  rolling              {rolling/1e9:.2f} GB")
    print(f"  milestones           {milestones/1e9:.2f} GB")
    print(f"  in-flight temp       {per_ckpt/1e9:.2f} GB")
    print(f"  final export         {final_only/1e9:.2f} GB")
    print(f"  PEAK REQUIRED        {need_gb:.2f} GB")
    print("=" * 66)

    headroom = free - peak
    fits = headroom > args.reserve_gb * 1e9
    out = {"free_gb": free / 1e9, "per_checkpoint_gb": per_ckpt / 1e9,
           "n_milestones": n_milestones, "peak_required_gb": need_gb,
           "reserve_gb": args.reserve_gb, "fits": fits,
           "keep_last": args.keep_last, "milestone_hours": args.milestone_hours}

    if fits:
        print(f"  VERDICT: FITS -- {headroom/1e9:.1f} GB spare "
              f"(after {args.reserve_gb} GB reserve)")
    else:
        print(f"  VERDICT: DOES NOT FIT -- short by "
              f"{(peak + args.reserve_gb*1e9 - free)/1e9:.2f} GB")
        print("\n  Retention policies that WOULD fit:")
        budget = free - args.reserve_gb * 1e9 - final_only - per_ckpt
        suggestions = []
        for keep in (2, 1):
            n_ms = int(budget / per_ckpt) - keep
            if n_ms >= 1:
                hrs = args.train_hours / n_ms
                suggestions.append((keep, n_ms, hrs))
        if suggestions:
            for keep, n_ms, hrs in suggestions:
                print(f"    --keep-last {keep} --milestone-hours {hrs:.1f}  "
                      f"-> {n_ms} milestones, "
                      f"{((keep+n_ms)*per_ckpt + per_ckpt + final_only)/1e9:.1f} GB")
            k, n_ms, hrs = suggestions[0]
            out["suggested"] = {"keep_last": k, "milestone_hours": round(hrs, 1)}
        else:
            print(f"    NONE. Even keep_last=1 with no milestones needs "
                  f"{(2*per_ckpt + final_only)/1e9:.1f} GB.")
            print("    Free disk space, or point --checkpoint-dir at a larger volume.")
            out["suggested"] = None
    print("=" * 66 + "\n")

    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    if not fits and args.strict:
        raise SystemExit(2)
    return out


def cmd_verify(args):
    verify_tokenizer(check_shards=True)
    man = json.loads((SHARDS / "manifest.json").read_text(encoding="utf-8"))
    n_files = len(man["shards"])
    on_disk = sorted(SHARDS.glob("shard_*.npy"))
    missing = [s["file"] for s in man["shards"]
               if not (SHARDS / s["file"]).exists()]
    bad_size = [s["file"] for s in man["shards"]
                if (SHARDS / s["file"]).exists()
                and (SHARDS / s["file"]).stat().st_size != s["bytes"]]
    print(f"[verify] shards: {n_files} in manifest, {len(on_disk)} on disk")
    print(f"[verify] total packed tokens: {man['total_tokens_packed']:,}")
    print(f"[verify] seq_len {man['seq_len']}, dtype {man['dtype']}")
    if missing or bad_size:
        raise SystemExit(f"SHARD PROBLEM missing={missing[:3]} "
                         f"size_mismatch={bad_size[:3]}")
    ft = ILSI_FT / "train.jsonl"
    print(f"[verify] ILSI fine-tune data present: {ft.exists()}")
    print("[verify] all checks passed")


def _one_step(model, opt, x, y, prec, device):
    """A real forward+backward+step, so the memory measured is the memory used."""
    with torch.autocast(device_type=device.split(":")[0], dtype=prec.dtype,
                        enabled=prec.use_amp):
        logits = model(x)
        loss = CrossEntropyLoss()(logits, y)
    loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)
    return float(loss.detach())


def cmd_probe(args):
    """Largest batch size that survives forward+backward, measured not guessed."""
    prec = olm_patches.apply_all(device=args.device)
    cfg = CONFIGS[args.config]
    device = args.device
    if not device.startswith("cuda"):
        print("[probe] not a CUDA device; batch probing is meaningless on CPU")
        return

    results = []
    best = None
    for bs in args.candidates:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            model = build_model(cfg).to(device)
            opt = AdamW(model.parameters(), lr=1e-5)
            x = torch.randint(0, cfg.vocab_size, (bs, cfg.max_seq_len),
                              device=device)
            y = torch.randint(0, cfg.vocab_size, (bs, cfg.max_seq_len),
                              device=device)
            for _ in range(3):
                _one_step(model, opt, x, y, prec, device)
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated() / 1e9
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            results.append({"batch_size": bs, "peak_gb": round(peak, 2),
                            "vram_gb": round(total, 2), "ok": True})
            best = bs
            print(f"[probe] batch {bs:>3}: OK   peak {peak:6.2f} GB "
                  f"/ {total:.1f} GB ({100*peak/total:.0f}%)")
        except torch.cuda.OutOfMemoryError:
            results.append({"batch_size": bs, "ok": False, "error": "OOM"})
            print(f"[probe] batch {bs:>3}: OOM")
            break
        except Exception as e:
            results.append({"batch_size": bs, "ok": False,
                            "error": f"{type(e).__name__}: {e}"})
            print(f"[probe] batch {bs:>3}: {type(e).__name__}: {e}")
            break
        finally:
            del model, opt
            try:
                del x, y
            except NameError:
                pass
            torch.cuda.empty_cache()

    if best is None:
        raise SystemExit("[probe] nothing fits, not even the smallest candidate")
    # One step down from the max, to leave headroom for allocator fragmentation
    # over a 20h run. The largest size that fits once often does not fit at 3am.
    safe = args.candidates[max(args.candidates.index(best) - 1, 0)] \
        if len(args.candidates) > 1 and best != args.candidates[0] else best
    print(f"\n[probe] largest that fits: {best}   recommended (headroom): {safe}")
    out = {"config": cfg.name, "results": results, "max_batch": best,
           "recommended_batch": safe}
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def cmd_smoke(args):
    """Measure tokens/sec on this card, then extrapolate the whole pretrain."""
    prec = olm_patches.apply_all(device=args.device)
    verify_tokenizer(check_shards=True)
    cfg = CONFIGS[args.config]
    device = args.device

    ds = PretrainShardDataset(SHARDS, cfg.max_seq_len)
    sampler = ResumableSampler(len(ds), seed=1234, shuffle=True)
    loader = DataLoader(ds, batch_size=args.batch_size, sampler=sampler,
                        shuffle=False, num_workers=args.num_workers,
                        drop_last=True, pin_memory=device.startswith("cuda"))

    model = build_model(cfg).to(device)
    pc = param_breakdown(model)
    opt = AdamW(model.parameters(), lr=1e-5, weight_decay=0.1)
    model.train()

    it = iter(loader)
    # Warm up outside the timed window: allocator, autotune and the first CUDA
    # kernels are one-off costs that would otherwise depress the measurement.
    for _ in range(args.warmup_batches):
        x, y = next(it)
        _one_step(model, opt, x.to(device), y.to(device), prec, device)
    if device.startswith("cuda"):
        torch.cuda.synchronize()

    deadline = time.time() + args.minutes * 60
    t0 = time.time()
    n_tok = n_batches = 0
    losses = []
    while time.time() < deadline and n_batches < args.max_batches:
        x, y = next(it)
        losses.append(_one_step(model, opt, x.to(device), y.to(device), prec, device))
        n_tok += x.numel()
        n_batches += 1
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.time() - t0

    tps = n_tok / elapsed
    env = olm_patches.environment_report(device)
    peak_tf = peak_tflops_for(env.get("gpu_name"))
    flops_per_token = 6 * pc["non_embedding"]
    achieved_tflops = tps * flops_per_token / 1e12
    mfu = (achieved_tflops / peak_tf) if peak_tf else None

    epoch_tokens = len(ds) * cfg.max_seq_len
    total_tokens = epoch_tokens * args.epochs
    pretrain_hours = total_tokens / tps / 3600

    reserve = args.finetune_hours + args.eval_hours + args.setup_hours
    budget = args.slot_hours - reserve
    fits = pretrain_hours <= budget
    affordable_tokens = budget * 3600 * tps

    print("\n" + "=" * 66)
    print("SMOKE TEST RESULT")
    print("=" * 66)
    print(f"  GPU                  {env.get('gpu_name')} "
          f"({env.get('vram_total_gb')} GB), precision {prec.name}")
    print(f"  model                {cfg.name}, {pc['non_embedding']/1e6:.1f}M non-emb")
    print(f"  batch                {args.batch_size} x {cfg.max_seq_len} tokens")
    print(f"  measured             {n_batches:,} batches, {n_tok/1e6:.1f}M tokens, "
          f"{elapsed/60:.1f} min")
    print(f"  loss                 {losses[0]:.3f} -> {losses[-1]:.3f}")
    print(f"  THROUGHPUT           {tps:,.0f} tokens/sec")
    print(f"  achieved             {achieved_tflops:,.1f} TFLOPs "
          f"(6N per token, N={pc['non_embedding']/1e6:.0f}M)")
    print(f"  MFU                  "
          f"{f'{100*mfu:.1f}%  (peak {peak_tf} TFLOPs)' if mfu else 'unknown GPU -- not computed'}")
    print("-" * 66)
    print(f"  pretrain target      {args.epochs} x {epoch_tokens/1e9:.2f}B "
          f"= {total_tokens/1e9:.2f}B tokens")
    print(f"  EXTRAPOLATED         {pretrain_hours:.2f} h of pretraining")
    print(f"  slot budget          {args.slot_hours:.1f} h total - "
          f"{reserve:.1f} h reserved (setup {args.setup_hours}, "
          f"finetune {args.finetune_hours}, eval {args.eval_hours}) "
          f"= {budget:.2f} h available")
    print("=" * 66)
    if fits:
        print(f"  VERDICT: FITS with {budget - pretrain_hours:.2f} h to spare")
    else:
        over = pretrain_hours - budget
        print(f"  VERDICT: DOES NOT FIT -- over by {over:.2f} h")
        print(f"  Tokens that WOULD fit in {budget:.2f} h: "
              f"{affordable_tokens/1e9:.2f}B "
              f"({100*affordable_tokens/total_tokens:.0f}% of target)")
        print(f"  Options: reduce to {affordable_tokens/epoch_tokens:.2f} epochs "
              f"over the {epoch_tokens/1e9:.2f}B pack,")
        print(f"           or cut unique tokens to "
              f"{affordable_tokens/args.epochs/1e9:.2f}B and keep "
              f"{args.epochs} epochs.")
    print("=" * 66 + "\n")

    out = {
        "gpu": env.get("gpu_name"), "vram_gb": env.get("vram_total_gb"),
        "precision": prec.name, "config": cfg.name,
        "batch_size": args.batch_size, "context": cfg.max_seq_len,
        "tokens_per_sec": tps, "achieved_tflops": achieved_tflops,
        "peak_tflops": peak_tf, "mfu": mfu,
        "loss_first": losses[0], "loss_last": losses[-1],
        "epoch_tokens": epoch_tokens, "epochs": args.epochs,
        "total_tokens": total_tokens,
        "pretrain_hours_extrapolated": pretrain_hours,
        "slot_hours": args.slot_hours, "reserved_hours": reserve,
        "budget_hours": budget, "fits": fits,
        "affordable_tokens": affordable_tokens,
    }
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2, default=str),
                                  encoding="utf-8")
    return out


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    p = sub.add_parser("env"); p.add_argument("--device", default=dev)
    p.add_argument("--out", default=None); p.set_defaults(fn=cmd_env)

    p = sub.add_parser("verify"); p.add_argument("--device", default=dev)
    p.add_argument("--out", default=None); p.set_defaults(fn=cmd_verify)

    p = sub.add_parser("disk")
    p.add_argument("--config", default="prod_300m", choices=list(CONFIGS))
    p.add_argument("--checkpoint-dir", default="./checkpoints")
    p.add_argument("--train-hours", type=float, default=12.0,
                   help="expected pretrain duration, drives milestone count")
    p.add_argument("--milestone-hours", type=float, default=2.0)
    p.add_argument("--keep-last", type=int, default=2)
    p.add_argument("--reserve-gb", type=float, default=5.0,
                   help="free space to leave for logs, temp and the OS")
    p.add_argument("--strict", action="store_true",
                   help="exit 2 if the policy does not fit")
    p.add_argument("--device", default=dev)
    p.add_argument("--out", default=None); p.set_defaults(fn=cmd_disk)

    p = sub.add_parser("probe")
    p.add_argument("--config", default="prod_300m", choices=list(CONFIGS))
    p.add_argument("--device", default=dev)
    p.add_argument("--candidates", type=int, nargs="+",
                   default=[1, 2, 4, 8, 12, 16, 24, 32])
    p.add_argument("--out", default=None); p.set_defaults(fn=cmd_probe)

    p = sub.add_parser("smoke")
    p.add_argument("--config", default="prod_300m", choices=list(CONFIGS))
    p.add_argument("--device", default=dev)
    p.add_argument("--batch-size", type=int, required=True)
    p.add_argument("--minutes", type=float, default=20.0)
    p.add_argument("--max-batches", type=int, default=10**9)
    p.add_argument("--warmup-batches", type=int, default=5)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--slot-hours", type=float, default=20.0)
    p.add_argument("--setup-hours", type=float, default=1.0)
    p.add_argument("--finetune-hours", type=float, default=2.0)
    p.add_argument("--eval-hours", type=float, default=0.5)
    p.add_argument("--out", default=None); p.set_defaults(fn=cmd_smoke)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
