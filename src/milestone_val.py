#!/usr/bin/env python
"""Reconstruct a validation-loss curve from the milestone checkpoints.

Pretraining logged only step / epoch / train_loss / learning_rate. The intake
form requires validation loss curves, and the milestone checkpoints are the
only remaining route to one. They are deleted when the slot window closes, so
this has to run tonight.

Held-out set: the ILSI test split's fact text (input_ids). That split is held out of fine-tuning. NOTE: explicit n-gram
decontamination against the pretraining corpus is not evidenced in the build
pipeline, so treat this as approximately-held-out. It is computed post-hoc, not during training, and must be labelled as
such wherever it is reported.

Batch size 1 by design: no padding, so no mask-correctness questions enter the
numbers. Writes a small JSON -- download that, not the 30GB of checkpoints.

Usage:
    PYTHONPATH=src python src/milestone_val.py --device cuda --n 256
"""
import argparse
import glob
import json
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from model import CONFIGS, build_model

TEST = Path("data/ilsi_ft/test.jsonl")


def find_checkpoints(extra):
    paths = []
    paths += glob.glob("checkpoints/slot1-pretrain/milestone_*.pt")
    paths += glob.glob("checkpoints/slot1-pretrain/step_*.pt")
    paths += glob.glob("checkpoints/*BACKUP*.pt")
    paths += list(extra or [])
    seen, out = set(), []
    for p in paths:
        rp = os.path.realpath(p)
        if rp not in seen and os.path.isfile(p):
            seen.add(rp)
            out.append(p)
    return out


def load_val_rows(n, ctx):
    rows = []
    with TEST.open(encoding="utf-8") as fh:
        for line in fh:
            if len(rows) >= n:
                break
            ids = json.loads(line)["input_ids"]
            if len(ids) >= 32:
                rows.append(ids[:ctx])
    return rows


@torch.no_grad()
def val_loss(model, rows, device, use_bf16):
    total_nll, total_tok = 0.0, 0
    for ids in rows:
        x = torch.tensor([ids], dtype=torch.long, device=device)
        if use_bf16:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x)
        else:
            logits = model(x)
        logits = logits[:, :-1, :].float()
        target = x[:, 1:]
        nll = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target.reshape(-1),
            reduction="sum",
        )
        total_nll += float(nll)
        total_tok += target.numel()
    mean = total_nll / max(total_tok, 1)
    return mean, total_tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="prod_300m", choices=list(CONFIGS))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n", type=int, default=256, help="held-out rows")
    ap.add_argument("--ckpt", action="append", help="extra checkpoint path(s)")
    ap.add_argument("--out", default="runs/milestone_val.json")
    ap.add_argument("--fp32", action="store_true",
                    help="disable bf16 autocast (slower, more comparable)")
    args = ap.parse_args()

    cfg = CONFIGS[args.config]
    use_bf16 = (not args.fp32) and args.device.startswith("cuda") \
        and torch.cuda.is_bf16_supported()

    rows = load_val_rows(args.n, cfg.max_seq_len)
    tok_total = sum(len(r) for r in rows)
    print(f"held-out: {len(rows)} rows, {tok_total:,} tokens | "
          f"bf16={use_bf16}")

    ckpts = find_checkpoints(args.ckpt)
    print(f"found {len(ckpts)} checkpoints\n")

    model = build_model(cfg).to(args.device)
    model.eval()

    results = []
    for path in ckpts:
        t0 = time.time()
        try:
            ck = torch.load(path, map_location="cpu", weights_only=False)
            state = ck.get("model_state_dict", ck) if isinstance(ck, dict) else ck
            model.load_state_dict(state)
            step = ck.get("step") if isinstance(ck, dict) else None
            train_loss = ck.get("loss") if isinstance(ck, dict) else None
            toks = ck.get("total_tokens_processed") if isinstance(ck, dict) else None
            del ck, state

            mean, n_tok = val_loss(model, rows, args.device, use_bf16)
            rec = {
                "checkpoint": os.path.basename(path),
                "step": step,
                "train_loss_at_ckpt": train_loss,
                "total_tokens_processed": toks,
                "val_loss": round(mean, 6),
                "val_ppl": round(float(torch.exp(torch.tensor(mean))), 4),
                "val_tokens": n_tok,
                "seconds": round(time.time() - t0, 1),
            }
            results.append(rec)
            print(f"  {rec['checkpoint']:<34} step {str(step):>6}  "
                  f"val_loss {mean:.4f}  ppl {rec['val_ppl']:.3f}  "
                  f"({rec['seconds']}s)")
        except Exception as exc:
            print(f"  {os.path.basename(path):<34} FAILED: {exc}")
            results.append({"checkpoint": os.path.basename(path),
                            "error": str(exc)})

    results.sort(key=lambda r: (r.get("step") is None, r.get("step") or 0))
    payload = {
        "note": "Post-hoc held-out loss on the ILSI test split fact text. "
                "Computed after training, not during. Batch size 1, no padding.",
        "config": args.config,
        "held_out_rows": len(rows),
        "held_out_tokens": tok_total,
        "bf16_autocast": use_bf16,
        "results": results,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
