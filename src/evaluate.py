"""
Evaluate a fine-tuned Pocket Vakil checkpoint on the ILSI test split.

    python src/evaluate.py --ckpt runs/ft-s2/checkpoints/step_8000.pt \
        --config prod_300m

**The task is multi-label.** Measured on the built data: 88.1% of test rows carry
more than one IPC section (train 78.6%, max 31 labels on a single row). So the
headline metrics are micro- and macro-F1 over label *sets*. Plain accuracy is
reported only as exact-set-match, which is a deliberately harsh number -- getting
4 of 5 sections right scores zero on it -- and it is not the metric to optimise.

    micro-F1   pools TP/FP/FN across all labels; dominated by frequent sections
    macro-F1   averages per-label F1; dominated by the tail
    exact match  every predicted section correct and none missing

**Parse failures are measured, not repaired** (PLAN.md section 4). Four kinds are
counted separately because they mean different things:

    duplicate   same label emitted twice -- the failure the ILSI dedup exists to
                prevent, so a nonzero rate points back at the data
    no_eos      hit the generation budget without stopping
    non_label   emitted ordinary text where a label token was required
    empty       stopped immediately, predicting nothing

They overlap, so `any` is the headline. Whatever labels *were* extracted are still
scored -- dropping failed rows would flatter F1 by removing the hardest cases.

Decoding is greedy and recomputes the prefix each step (no KV cache, no sampling).
Targets average ~5 tokens, so the cost is bounded and the result is deterministic.
"""

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

from model import CONFIGS, build_model, EOS_ID

TOKENIZER = Path("data/tokenizer/bpe_32000_final.json")
ILSI_FT = Path("data/ilsi_ft")
LABEL_MAP = Path("data/ilsi/label_map.json")

# src/baseline_tfidf.py on the same test split, whole (untruncated) fact text.
BASELINE = {"micro_f1": 0.4941, "macro_f1": 0.2858}


def multilabel_scores(gold, pred):
    """Micro/macro/per-label F1 and exact-match over sets of section names."""
    labels = sorted({l for s in gold for l in s} | {l for s in pred for l in s})
    tp, fp, fn = Counter(), Counter(), Counter()
    exact = 0
    for g, p in zip(gold, pred):
        exact += (g == p)
        for l in p & g:
            tp[l] += 1
        for l in p - g:
            fp[l] += 1
        for l in g - p:
            fn[l] += 1
    TP, FP, FN = sum(tp.values()), sum(fp.values()), sum(fn.values())
    micro_p = TP / max(TP + FP, 1)
    micro_r = TP / max(TP + FN, 1)
    micro_f1 = 2 * TP / max(2 * TP + FP + FN, 1)
    per = {l: 2 * tp[l] / max(2 * tp[l] + fp[l] + fn[l], 1) for l in labels}
    macro_f1 = sum(per.values()) / max(len(per), 1)
    return {
        "micro_f1": micro_f1, "macro_f1": macro_f1,
        "micro_precision": micro_p, "micro_recall": micro_r,
        "exact_match": exact / max(len(gold), 1),
        "tp": TP, "fp": FP, "fn": FN, "per_label_f1": per,
    }


def parse_generation(ids, eos_id, id_to_label):
    """Emitted ids -> (label set, failure flags). Strict: nothing is repaired."""
    flags = {"duplicate": False, "no_eos": False, "non_label": False, "empty": False}
    labels, seen = [], set()
    saw_eos = False
    for t in ids:
        if t == eos_id:
            saw_eos = True
            break
        if t in id_to_label:
            name = id_to_label[t]
            if name in seen:
                flags["duplicate"] = True
            else:
                seen.add(name)
                labels.append(name)
        else:
            flags["non_label"] = True
    flags["no_eos"] = not saw_eos
    flags["empty"] = not labels
    return set(labels), flags


@torch.no_grad()
def greedy_generate(model, prompts, max_new, eos_id, device, batch_size, ctx):
    """Greedy decode with left padding, so every row's last position is real."""
    out = []
    for i in range(0, len(prompts), batch_size):
        chunk = [p[-ctx:] for p in prompts[i:i + batch_size]]
        width = max(len(p) for p in chunk)
        X = np.full((len(chunk), width), eos_id, dtype=np.int64)
        for b, p in enumerate(chunk):
            X[b, width - len(p):] = p
        x = torch.from_numpy(X).to(device)
        done = torch.zeros(len(chunk), dtype=torch.bool, device=device)
        gen = [[] for _ in chunk]
        for _ in range(max_new):
            logits = model(x[:, -ctx:])
            nxt = logits[:, -1, :].float().argmax(-1)
            for b in range(len(chunk)):
                if not done[b]:
                    gen[b].append(int(nxt[b]))
                    if int(nxt[b]) == eos_id:
                        done[b] = True
            if bool(done.all()):
                break
            x = torch.cat([x, nxt.unsqueeze(1)], dim=1)
        out.extend(gen)
        if (i // batch_size) % 20 == 0:
            print(f"   {min(i+batch_size, len(prompts)):,}/{len(prompts):,}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default="prod_300m", choices=list(CONFIGS))
    ap.add_argument("--split", default="test")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new", type=int, default=40)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = CONFIGS[args.config]
    tok = Tokenizer.from_file(str(TOKENIZER))
    eos_id = tok.token_to_id("<|endoftext|>")
    assert eos_id == EOS_ID, f"EOS id drift: {eos_id} != {EOS_ID}"

    meta = json.loads(LABEL_MAP.read_text(encoding="utf-8"))
    id_to_label = {}
    for section, token in zip(meta["sections"], meta["special_tokens"]):
        tid = tok.token_to_id(token)
        assert tid is not None, f"{token} missing from tokenizer"
        id_to_label[tid] = section

    rows = [json.loads(l)
            for l in (ILSI_FT / f"{args.split}.jsonl").open(encoding="utf-8")]
    if args.limit:
        rows = rows[: args.limit]
    n = len(rows)
    card = Counter(len(r["labels"]) for r in rows)
    multi = sum(v for k, v in card.items() if k > 1)
    print(f"{n:,} {args.split} rows | {len(id_to_label)} label tokens | "
          f"multi-label {multi:,} ({100*multi/n:.1f}%), max {max(card)} labels/row")

    model = build_model(cfg).to(args.device)
    ck = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    state = ck.get("model_state_dict", ck) if isinstance(ck, dict) else ck
    model.load_state_dict(state)
    model.eval()
    print(f"loaded {args.ckpt} "
          f"(step {ck.get('step') if isinstance(ck, dict) else '?'})")

    t0 = time.time()
    gen = greedy_generate(model, [r["input_ids"] for r in rows], args.max_new,
                          eos_id, args.device, args.batch_size, cfg.max_seq_len)
    print(f"generated in {time.time()-t0:.0f}s")

    gold, pred, flags = [], [], []
    for r, g in zip(rows, gen):
        s, f = parse_generation(g, eos_id, id_to_label)
        gold.append(set(r["labels"]))
        pred.append(s)
        flags.append(f)

    m = multilabel_scores(gold, pred)
    fail_counts = {k: sum(f[k] for f in flags)
                   for k in ("duplicate", "no_eos", "non_label", "empty")}
    any_fail = sum(any(f.values()) for f in flags)
    zero_f1 = sorted(l for l, v in m["per_label_f1"].items() if v == 0.0)

    result = {
        "checkpoint": args.ckpt, "config": cfg.name, "split": args.split, "n": n,
        "task": "multi-label", "multi_label_rows": multi,
        **{k: v for k, v in m.items() if k != "per_label_f1"},
        "labels_with_zero_f1": len(zero_f1), "zero_f1_sections": zero_f1,
        "parse_failures": {**fail_counts, "any": any_fail,
                           "any_rate": any_fail / n,
                           **{f"{k}_rate": v / n for k, v in fail_counts.items()}},
        "per_label_f1": m["per_label_f1"],
        "baseline_tfidf": BASELINE,
        "delta_vs_baseline": {
            "micro_f1": m["micro_f1"] - BASELINE["micro_f1"],
            "macro_f1": m["macro_f1"] - BASELINE["macro_f1"],
        },
    }
    out = Path(args.out) if args.out else Path(args.ckpt).parent / f"eval_{args.split}.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"\n=== ILSI {args.split} ({n:,} rows, multi-label) ===")
    print(f"   micro-F1     {m['micro_f1']:.4f}   "
          f"(baseline {BASELINE['micro_f1']:.4f}, {m['micro_f1']-BASELINE['micro_f1']:+.4f})")
    print(f"   macro-F1     {m['macro_f1']:.4f}   "
          f"(baseline {BASELINE['macro_f1']:.4f}, {m['macro_f1']-BASELINE['macro_f1']:+.4f})")
    print(f"   micro-P / R  {m['micro_precision']:.4f} / {m['micro_recall']:.4f}")
    print(f"   exact match  {m['exact_match']:.4f}   (all sections right, none missing)")
    print(f"   labels with F1 = 0.0: {len(zero_f1)}/100")
    print(f"\n   PARSE FAILURES (measured, not repaired)")
    for k in ("duplicate", "no_eos", "non_label", "empty"):
        print(f"      {k:10s} {fail_counts[k]:6,}  ({100*fail_counts[k]/n:5.2f}%)")
    print(f"      {'ANY':10s} {any_fail:6,}  ({100*any_fail/n:5.2f}%)")
    print(f"\n   -> {out}")


if __name__ == "__main__":
    main()
