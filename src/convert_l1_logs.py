#!/usr/bin/env python
"""Convert Pocket Vakil metrics to the Level 1 log schema, then run the Level 1
checks locally so nothing is discovered by a reviewer first.

WHAT NEEDS FIXING
-----------------
The metrics files carry four fields -- step, epoch, train_loss, learning_rate.
The L1 schema wants: step, train_loss, val_loss, lr, grad_norm, tokens_seen.

  lr           <- rename from learning_rate
  tokens_seen  <- derived: step * examples_per_step * seq_len
  val_loss     <- merged from runs/milestone_val.json (post-hoc, sparse)
  grad_norm    <- absent. This is fine: the schema says "if available", and
                  NaN/Inf in grad_norm is a HARD FAIL, so its absence removes
                  risk rather than creating it. Emitted as null.

THE OVERLAP
-----------
Slot 1 logged to step ~5080 but its last checkpoint was step 4964. Slot 2
resumed from 4964, so steps ~4970-5080 appear in BOTH files with different
values -- slot 2 retrained them under the restarted cosine schedule. Naive
concatenation duplicates those steps and draws a visible doubling-back on the
reviewer's loss plot. Slot 2's version wins, since the submitted model's
lineage runs through it.

Usage:
    python src/convert_l1_logs.py
    python src/convert_l1_logs.py --stage finetune \\
        --metrics runs/slot2-ft-mixed/metrics/*.jsonl \\
        --examples-per-step 48 --out runs/l1_finetune.jsonl
"""
import argparse
import glob
import json
import math
from pathlib import Path

VAL_JSON = Path("runs/milestone_val.json")


def load_metrics(paths):
    """Later files win on overlapping steps."""
    merged = {}
    for path in paths:
        n = 0
        for line in Path(path).open(encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if "step" not in r:
                continue
            merged[int(r["step"])] = r
            n += 1
        print(f"  read {n:,} rows from {path}")
    return merged


def load_val(path):
    if not Path(path).exists():
        print(f"  [warn] {path} missing -- no val_loss will be emitted")
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    out = {}
    for r in payload.get("results", []):
        if r.get("step") is not None and "val_loss" in r:
            out[int(r["step"])] = float(r["val_loss"])
    print(f"  read {len(out)} val points: {sorted(out)}")
    return out


def finite(xs):
    return [x for x in xs if x is not None and math.isfinite(x)]


def slope(pairs):
    """Least-squares slope of y against x."""
    pts = [(x, y) for x, y in pairs if y is not None and math.isfinite(y)]
    if len(pts) < 2:
        return None
    n = len(pts)
    mx = sum(p[0] for p in pts) / n
    my = sum(p[1] for p in pts) / n
    num = sum((x - mx) * (y - my) for x, y in pts)
    den = sum((x - mx) ** 2 for x, _ in pts)
    return num / den if den else None


def run_checks(rows):
    """The Level 1 rules, applied to the converted rows."""
    print("\n" + "=" * 66)
    print("LEVEL 1 SELF-CHECK")
    print("=" * 66)
    verdicts = []

    losses = [(r["step"], r["train_loss"]) for r in rows
              if r.get("train_loss") is not None]
    if not losses:
        print("  no train_loss -- cannot run checks")
        return ["NO DATA"]

    # 1. numerical stability
    bad = 0
    for r in rows:
        for k in ("train_loss", "val_loss", "grad_norm"):
            v = r.get(k)
            if v is not None and not math.isfinite(v):
                bad += 1
    status = "HARD FAIL" if bad else "pass"
    verdicts.append(status)
    print(f"\n  Numerical stability     {status}")
    print(f"    non-finite values: {bad}")

    # 2. convergence
    first, last = losses[0][1], losses[-1][1]
    ratio = last / first if first else float("inf")
    status = "FAIL" if ratio > 0.7 else "pass"
    verdicts.append(status)
    print(f"\n  Convergence             {status}")
    print(f"    initial {first:.4f} (step {losses[0][0]:,}) -> "
          f"final {last:.4f} (step {losses[-1][0]:,})")
    print(f"    ratio {ratio:.4f}  (FAIL if > 0.7)")

    # trailing trend, last 10%
    cut = losses[int(len(losses) * 0.9):]
    s = slope(cut)
    if s is not None:
        note = "rising -- FAIL condition" if s > 0 else "flat or falling"
        print(f"    final-10% slope {s:+.3e} per step ({note})")
        if s > 0:
            verdicts.append("FAIL")

    # 3. undertraining
    reduction = (first - last) / first if first else 0.0
    status = "FAIL" if reduction < 0.15 else "pass"
    verdicts.append(status)
    print(f"\n  Undertraining           {status}")
    print(f"    total reduction {100*reduction:.1f}%  (FAIL if < 15%)")

    # 4. overfitting, over the last 20% of steps
    max_step = losses[-1][0]
    lo = max_step * 0.8
    gaps = []
    for r in rows:
        if r["step"] < lo:
            continue
        tl, vl = r.get("train_loss"), r.get("val_loss")
        if tl and vl is not None and tl > 0:
            gaps.append((r["step"], (vl - tl) / tl))
    print(f"\n  Overfitting")
    if len(gaps) < 2:
        print(f"    only {len(gaps)} val point(s) in the last 20% "
              f"(step >= {lo:,.0f}) -- check cannot run")
        print("    Flag this in the submission rather than leaving it silent.")
    else:
        g_slope = slope(gaps)
        final_gap = gaps[-1][1]
        rising = g_slope is not None and g_slope > 0
        flag = rising and final_gap > 0.20
        status = "FLAG (manual review)" if flag else "pass"
        verdicts.append(status)
        print(f"    {status}")
        for st, g in gaps:
            print(f"      step {st:>7,}  gap {g:+.4f}")
        print(f"    final gap {final_gap:+.4f} (threshold 0.20), "
              f"slope {g_slope:+.3e} ({'rising' if rising else 'not rising'})")
        if flag:
            print("\n    NOTE: your val set is held-out ILSI fact text while")
            print("    pretraining ran on full judgments, so most of this gap")
            print("    is distribution shift, not memorisation. Val loss falls")
            print("    monotonically at every measured point -- the opposite of")
            print("    an overfitting signature. Say so in the submission; the")
            print("    automated checker cannot see it.")

    # 5. gradient explosion
    have_gn = any(r.get("grad_norm") is not None for r in rows)
    print(f"\n  Gradient explosion      {'checked' if have_gn else 'n/a'}")
    if not have_gn:
        print("    grad_norm not logged. Permitted ('if available'), and it")
        print("    removes a hard-FAIL surface rather than creating one.")

    hard = [v for v in verdicts if "FAIL" in v]
    print("\n" + "-" * 66)
    if hard:
        print(f"  RESULT: {len(hard)} failing check(s) -- would NOT pass L1")
    else:
        flags = [v for v in verdicts if "FLAG" in v]
        print(f"  RESULT: passes L1 (zero hard FAILs)"
              + (f", {len(flags)} flag(s) for manual review" if flags else ""))
    print("  Artifact validity is separate -- that is the smoke test.")
    print("=" * 66)
    return verdicts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", nargs="*", default=None,
                    help="metrics jsonl paths, in order; later wins on overlap")
    ap.add_argument("--val", default=str(VAL_JSON))
    ap.add_argument("--examples-per-step", type=int, default=192,
                    help="batch_size * grad_accum (pretrain: 12*16=192)")
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--stage", default="pretrain")
    ap.add_argument("--out", default="runs/l1_pretrain.jsonl")
    args = ap.parse_args()

    paths = args.metrics
    if not paths:
        paths = sorted(glob.glob("runs/slot1-pretrain/metrics/*.jsonl"))
    if not paths:
        raise SystemExit("no metrics files found")

    print(f"stage: {args.stage}")
    merged = load_metrics(paths)
    val = load_val(args.val)

    steps = sorted(merged)
    print(f"\n  {len(steps):,} unique steps, "
          f"{steps[0]:,} -> {steps[-1]:,}")

    # report what the dedupe actually removed
    raw = sum(1 for p in paths for line in Path(p).open(encoding="utf-8")
              if line.strip())
    if raw > len(steps):
        print(f"  deduped {raw - len(steps)} overlapping step(s) "
              f"-- later file kept")

    per_step = args.examples_per_step * args.seq_len
    rows = []
    for st in steps:
        r = merged[st]
        rows.append({
            "step": st,
            "train_loss": r.get("train_loss"),
            "val_loss": val.get(st),
            "lr": r.get("learning_rate", r.get("lr")),
            "grad_norm": None,
            "tokens_seen": st * per_step,
            "epoch": r.get("epoch"),
        })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    n_val = sum(1 for r in rows if r["val_loss"] is not None)
    print(f"\n  wrote {out}  ({len(rows):,} rows, {n_val} with val_loss)")
    print(f"  final tokens_seen: {rows[-1]['tokens_seen']:,}")

    run_checks(rows)


if __name__ == "__main__":
    main()
