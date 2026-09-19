#!/usr/bin/env python
"""Generate the four submission figures.

  fig1_pretrain_loss.png   Pretraining loss, LR restart annotated
  fig2_val_curve.png       Reconstructed held-out loss vs tokens
  fig3_per_label_f1.png    Per-label F1 against label frequency
  fig4_ablation.png        v2 vs v3 mix ablation

Run from the pocket_vakil folder:

    python make_plots.py
"""
import argparse
import collections
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TOKENS_PER_STEP = 192 * 2047
RESTART_STEP = 4964


def load_jsonl(p):
    return [json.loads(l) for l in Path(p).open(encoding="utf-8") if l.strip()]


def fig1(log, out):
    rows = load_jsonl(log)
    steps = [r["step"] for r in rows]
    loss = [r["train_loss"] for r in rows]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(steps, loss, lw=0.9, color="#2b6cb0")
    ax.axvline(RESTART_STEP, color="#c53030", ls="--", lw=1.2)
    ax.annotate(
        "slot boundary + LR restart\n(peak 2e-4, above the 1.13e-4\nthe run resumed from)",
        xy=(RESTART_STEP, max(loss) * 0.55),
        xytext=(RESTART_STEP + 900, max(loss) * 0.72),
        fontsize=8, color="#c53030",
        arrowprops=dict(arrowstyle="->", color="#c53030", lw=0.9))

    ax2 = ax.twinx()
    ax2.plot(steps, [r["lr"] for r in rows], lw=0.8, color="#dd6b20", alpha=0.55)
    ax2.set_ylabel("learning rate", color="#dd6b20", fontsize=9)
    ax2.tick_params(axis="y", labelcolor="#dd6b20", labelsize=8)

    ax.set_xlabel("step")
    ax.set_ylabel("training loss")
    ax.set_title("Pocket Vakil 332M — pretraining loss\n"
                 "10,049 steps · 3.93B tokens · 2 × H100 slots", fontsize=11)
    ax.grid(alpha=0.25)
    ax.set_ylim(0, min(max(loss), 10.5))
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    print(f"  {out}")


def fig2(val_json, out):
    payload = json.loads(Path(val_json).read_text(encoding="utf-8"))
    pts = sorted(
        ((r["step"], r["val_loss"]) for r in payload["results"]
         if r.get("step") is not None and "val_loss" in r),
        key=lambda x: x[0])
    # one point per step (the two step-10000 files are the same weights)
    seen, uniq = set(), []
    for s, v in pts:
        if s not in seen:
            seen.add(s)
            uniq.append((s, v))

    toks = [s * TOKENS_PER_STEP / 1e9 for s, _ in uniq]
    vals = [v for _, v in uniq]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(toks, vals, "o-", color="#2f855a", lw=1.6, ms=5)
    for t, v in zip(toks, vals):
        ax.annotate(f"{v:.3f}", (t, v), textcoords="offset points",
                    xytext=(0, 7), fontsize=7, ha="center")

    ax.axvline(RESTART_STEP * TOKENS_PER_STEP / 1e9,
               color="#c53030", ls="--", lw=1.1)
    ax.text(RESTART_STEP * TOKENS_PER_STEP / 1e9 + 0.05, max(vals) * 0.985,
            "slot boundary\n(epoch 2 begins)", fontsize=8, color="#c53030")

    ax.set_xlabel("tokens seen (billions)")
    ax.set_ylabel("held-out loss (ILSI fact text)")
    ax.set_title("Reconstructed validation curve\n"
                 "slot 1: −0.241 nats · slot 2: −0.023 nats for the same wall clock",
                 fontsize=11)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    print(f"  {out}")


def fig3(results, test_jsonl, out):
    d = json.loads(Path(results).read_text(encoding="utf-8"))
    per_label = d["per_label_f1"]

    freq = collections.Counter()
    for line in Path(test_jsonl).open(encoding="utf-8"):
        freq.update(json.loads(line)["labels"])

    pairs = [(freq[k], v, k) for k, v in per_label.items() if k in freq]
    pairs.sort()
    x = [p[0] for p in pairs]
    y = [p[1] for p in pairs]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.scatter(x, y, s=26, alpha=0.7, color="#2b6cb0", edgecolor="none")
    ax.axhline(d["macro_f1"], color="#dd6b20", ls="--", lw=1.2,
               label=f"macro-F1 {d['macro_f1']:.4f}")
    ax.axhline(0, color="#c53030", lw=1.0, alpha=0.5)

    # label the extremes only
    for n, f1, name in pairs[:2] + pairs[-3:]:
        ax.annotate(name.replace("Section ", "S"), (n, f1),
                    textcoords="offset points", xytext=(4, 4), fontsize=7)

    worst = min(pairs, key=lambda p: p[1])
    ax.annotate(f"lowest: {worst[2]} at F1 {worst[1]:.3f}\n"
                f"no label scores zero",
                xy=(worst[0], worst[1]), xytext=(worst[0] * 1.6, 0.06),
                fontsize=8, color="#2f855a",
                arrowprops=dict(arrowstyle="->", color="#2f855a", lw=0.9))

    ax.set_xscale("log")
    ax.set_xlabel("label frequency in test split (log scale)")
    ax.set_ylabel("per-label F1")
    ax.set_ylim(-0.03, 1.0)
    ax.set_title("Per-label F1 vs label frequency — 100 IPC sections\n"
                 "0 / 100 labels with F1 = 0 across 13,019 test cases",
                 fontsize=11)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    print(f"  {out}")

    lo = [p[1] for p in pairs if p[0] < 200]
    hi = [p[1] for p in pairs if p[0] > 2000]
    print(f"     rarest (<200 cases):  mean F1 {sum(lo)/len(lo):.4f}  n={len(lo)}")
    print(f"     common (>2000 cases): mean F1 {sum(hi)/len(hi):.4f}  n={len(hi)}")


def fig4(out):
    # both measured on the same 1,000-row subsample
    metrics = ["micro-F1", "macro-F1"]
    v2 = [0.4650, 0.3470]
    v3 = [0.4494, 0.3042]
    xs = range(len(metrics))
    w = 0.35

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 4.5),
                                  gridspec_kw={"width_ratios": [1.4, 1]})

    ax.bar([i - w / 2 for i in xs], v2, w, label="v2  (81.0 / 11.4 / 7.6)",
           color="#2b6cb0")
    ax.bar([i + w / 2 for i in xs], v3, w, label="v3  (73.3 / 18.6 / 8.1)",
           color="#a0aec0")
    for i, (a, b) in enumerate(zip(v2, v3)):
        ax.text(i - w / 2, a + 0.006, f"{a:.4f}", ha="center", fontsize=8)
        ax.text(i + w / 2, b + 0.006, f"{b:.4f}", ha="center", fontsize=8)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 0.56)
    ax.set_ylabel("score")
    ax.set_title("Fine-tune mix ablation\nsame 1,000 test rows", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, axis="y")

    ax2.bar(["v2", "v3"], [8, 17], color=["#2b6cb0", "#a0aec0"], width=0.5)
    ax2.text(0, 8.3, "8", ha="center", fontsize=9)
    ax2.text(1, 17.3, "17", ha="center", fontsize=9)
    ax2.set_ylabel("labels with F1 = 0  (of 100)")
    ax2.set_ylim(0, 21)
    ax2.set_title("Cutting IPC rows starves\nthe rare sections", fontsize=11)
    ax2.grid(alpha=0.25, axis="y")

    fig.tight_layout()
    fig.savefig(out, dpi=200)
    print(f"  {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-log", default="submission/train_log.jsonl")
    ap.add_argument("--val-json", default="runs_backup/runs/milestone_val.json")
    ap.add_argument("--results", default="runs_backup/runs/eval_FULL_v2_n13019.json")
    ap.add_argument("--test-jsonl", default="data/ilsi_ft/test.jsonl")
    ap.add_argument("--out-dir", default="submission/figures")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    print("writing figures:")

    for fn, a in [
        (fig1, (args.train_log, out / "fig1_pretrain_loss.png")),
        (fig2, (args.val_json, out / "fig2_val_curve.png")),
        (fig3, (args.results, args.test_jsonl, out / "fig3_per_label_f1.png")),
        (fig4, (out / "fig4_ablation.png",)),
    ]:
        try:
            fn(*a)
        except Exception as exc:
            print(f"  [FAILED] {fn.__name__}: {exc}")

    print(f"\n-> {out.resolve()}")


if __name__ == "__main__":
    main()
