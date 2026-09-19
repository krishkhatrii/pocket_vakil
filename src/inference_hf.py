#!/usr/bin/env python
"""Pocket Vakil — reproduce the reported ILSI results using only `transformers`.

The original evaluation script imports OLM, which was supplied inside the
bootcamp compute environment and is not installable from a public index. This
script needs nothing beyond torch + transformers, so a reviewer can reproduce
every reported number without that dependency.

The converted HF model is numerically identical to the OLM model the results
were produced with (max |delta logit| = 0.000e+00, argmax agreement 100%), so
the numbers this script prints are the numbers in the submission.

Decoding matches the reported configuration exactly:
  * greedy, no sampling
  * RIGHT padding, each row's logits read at its own true last position
  * prompts truncated to (ctx - max_new) so the window never exceeds context
  * parse failures are measured, never repaired, and score as-is

Usage:
    python inference_hf.py --model pocket-vakil-332m-ipc \\
        --data data/ilsi_ft_v2/test.jsonl --batch-size 16

    python inference_hf.py --model pocket-vakil-332m-ipc \\
        --data data/ilsi_ft_v2/test.jsonl --limit 200     # quick check

    python inference_hf.py --model pocket-vakil-332m-ipc --predict "<case facts>"
"""
import argparse
import json
import time
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def label_token_ids(tok):
    """Map IPC label token ids to the section names used in the gold labels.

    The tokenizer stores these as <ipc_302>; the ILSI gold labels are
    "Section 302". Comparing the raw token strings against the gold names
    silently yields zero true positives -- every prediction becomes a false
    positive and every gold label a false negative, with no error raised.
    """
    out = {}
    for token, tid in tok.get_vocab().items():
        if token.startswith("<ipc_") and token.endswith(">"):
            out[tid] = "Section " + token[5:-1]
    return out


@torch.no_grad()
def greedy_generate(model, tok, prompts, max_new, eos_id, device, batch_size, ctx):
    """Right-padded greedy decode. Each row reads its own last real position."""
    out = []
    prompt_ctx = max(1, ctx - max_new)

    for i in range(0, len(prompts), batch_size):
        chunk = [p[-prompt_ctx:] for p in prompts[i:i + batch_size]]
        n = len(chunk)
        lens = [len(p) for p in chunk]
        width = max(lens)

        x = torch.full((n, width + max_new), eos_id, dtype=torch.long)
        mask = torch.zeros((n, width + max_new), dtype=torch.long)
        for b, p in enumerate(chunk):
            x[b, :len(p)] = torch.tensor(p, dtype=torch.long)
            mask[b, :len(p)] = 1
        x, mask = x.to(device), mask.to(device)

        cur = torch.tensor(lens, dtype=torch.long, device=device)
        done = torch.zeros(n, dtype=torch.bool, device=device)
        gen = [[] for _ in chunk]
        rows = torch.arange(n, device=device)

        for _ in range(max_new):
            upto = int(cur.max())
            logits = model(input_ids=x[:, :upto],
                           attention_mask=mask[:, :upto]).logits
            nxt = logits[rows, cur - 1, :].float().argmax(-1)

            for b in range(n):
                if not done[b]:
                    t = int(nxt[b])
                    gen[b].append(t)
                    if t == eos_id:
                        done[b] = True
            if bool(done.all()):
                break

            alive = ~done
            x[rows[alive], cur[alive]] = nxt[alive]
            mask[rows[alive], cur[alive]] = 1
            cur = cur + alive.long()

        out.extend(gen)
        if (i // batch_size) % 20 == 0:
            print(f"   {min(i+batch_size, len(prompts)):,}/{len(prompts):,}",
                  flush=True)
    return out


def parse_generation(ids, eos_id, id_to_label):
    """Return (labels, failure_kind_or_None). Failures are measured, not fixed."""
    seen, labels, fail = set(), [], None
    hit_eos = False
    for t in ids:
        if t == eos_id:
            hit_eos = True
            break
        if t not in id_to_label:
            fail = fail or "non_label"
            continue
        name = id_to_label[t]
        if name in seen:
            fail = fail or "duplicate"
            continue
        seen.add(name)
        labels.append(name)
    if not hit_eos:
        fail = fail or "no_eos"
    if not labels:
        fail = "empty"
    return labels, fail


def score(gold, pred, all_labels):
    tp = fp = fn = 0
    per = {l: [0, 0, 0] for l in all_labels}
    for g, p in zip(gold, pred):
        gs, ps = set(g), set(p)
        for l in ps & gs:
            tp += 1
            per[l][0] += 1
        for l in ps - gs:
            fp += 1
            per.setdefault(l, [0, 0, 0])[1] += 1
        for l in gs - ps:
            fn += 1
            per.setdefault(l, [0, 0, 0])[2] += 1

    def f1(t, f_p, f_n):
        return 2 * t / (2 * t + f_p + f_n) if (2 * t + f_p + f_n) else 0.0

    micro = f1(tp, fp, fn)
    per_f1 = {l: f1(*v) for l, v in per.items()}
    macro = sum(per_f1.values()) / len(per_f1) if per_f1 else 0.0
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return micro, macro, prec, rec, tp, fp, fn, per_f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="pocket-vakil-332m-ipc")
    ap.add_argument("--data", default="data/ilsi_ft_v2/test.jsonl")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new", type=int, default=40)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="eval_hf_result.json")
    ap.add_argument("--predict", default=None,
                    help="run on one raw text instead of the test split")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True).to(device)
    model.eval()

    id_to_label = label_token_ids(tok)
    eos_id = tok.eos_token_id
    ctx = model.config.max_position_embeddings
    print(f"{len(id_to_label)} IPC label tokens | eos {eos_id} | ctx {ctx}")

    FRAME = "\n\n### TASK: identify the applicable IPC sections\n### ANSWER:\n"

    # ---- single-text mode -------------------------------------------------
    if args.predict:
        ids = tok.encode(args.predict)[: ctx - args.max_new - 32]
        ids = ids + tok.encode(FRAME)
        gen = greedy_generate(model, tok, [ids], args.max_new, eos_id,
                              device, 1, ctx)[0]
        labels, fail = parse_generation(gen, eos_id, id_to_label)
        print(f"\npredicted sections: {labels or '(none)'}")
        if fail:
            print(f"parse note: {fail}")
        return

    # ---- test split -------------------------------------------------------
    rows = [json.loads(l) for l in Path(args.data).open(encoding="utf-8")
            if l.strip()]
    if args.limit:
        rows = rows[: args.limit]
    if "labels" not in rows[0]:
        raise SystemExit("rows have no 'labels' field -- wrong data file?")

    gold = [r["labels"] for r in rows]
    all_labels = sorted({l for g in gold for l in g})
    overlap = set(id_to_label.values()) & set(all_labels)
    if not overlap:
        raise SystemExit(
            f"label namespace mismatch: model emits e.g. "
            f"{sorted(id_to_label.values())[:3]} but gold labels are "
            f"{all_labels[:3]}. Scoring would report 0.0 with no error."
        )
    print(f"label namespace overlap: {len(overlap)}/{len(all_labels)}")
    card = Counter(len(g) for g in gold)
    multi = sum(v for k, v in card.items() if k > 1)
    print(f"{len(rows):,} rows | {len(all_labels)} labels | "
          f"multi-label {multi:,} ({100*multi/len(rows):.1f}%)")

    t0 = time.time()
    gen = greedy_generate(model, tok, [r["input_ids"] for r in rows],
                          args.max_new, eos_id, device, args.batch_size, ctx)
    print(f"generated in {time.time()-t0:.0f}s")

    pred, fails = [], Counter()
    for g in gen:
        labels, fail = parse_generation(g, eos_id, id_to_label)
        pred.append(labels)
        if fail:
            fails[fail] += 1

    micro, macro, prec, rec, tp, fp, fn, per_f1 = score(gold, pred, all_labels)
    exact = sum(set(g) == set(p) for g, p in zip(gold, pred)) / len(gold)
    zero = [l for l, v in per_f1.items() if v == 0.0]

    n = len(rows)
    print(f"\n=== ILSI test ({n:,} rows, multi-label) ===")
    print(f"   micro-F1     {micro:.4f}")
    print(f"   macro-F1     {macro:.4f}")
    print(f"   micro-P / R  {prec:.4f} / {rec:.4f}")
    print(f"   exact match  {exact:.4f}")
    print(f"   labels with F1 = 0: {len(zero)}/{len(per_f1)}")
    print(f"\n   PARSE FAILURES (measured, not repaired)")
    for k in ("duplicate", "no_eos", "non_label", "empty"):
        print(f"      {k:<12}{fails[k]:>5}  ({100*fails[k]/n:5.2f}%)")
    print(f"      {'ANY':<12}{sum(fails.values()):>5}  "
          f"({100*sum(fails.values())/n:5.2f}%)")

    Path(args.out).write_text(json.dumps({
        "model": args.model, "data": args.data, "n": n,
        "batch_size": args.batch_size, "max_new": args.max_new,
        "device": device, "decoding": "greedy, right-padded, attention mask",
        "micro_f1": micro, "macro_f1": macro,
        "micro_precision": prec, "micro_recall": rec,
        "exact_match": exact, "tp": tp, "fp": fp, "fn": fn,
        "labels_with_zero_f1": len(zero), "zero_f1_sections": zero,
        "parse_failures": dict(fails), "per_label_f1": per_f1,
    }, indent=2), encoding="utf-8")
    print(f"\n   -> {args.out}")


if __name__ == "__main__":
    main()
