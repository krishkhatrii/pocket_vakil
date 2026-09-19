"""Measure what evaluate.py's left-pad-with-EOS geometry costs on ILSI.

`greedy_generate` left-pads every batch to the batch's longest prompt using
`eos_id` as the pad value, and passes no attention mask. Fine-tuning did the
opposite: `ILSIFinetuneDataset.__getitem__` right-pads to a fixed 2048, so under
causal attention the real tokens never attend to pad and the first fact token is
always at absolute position 0. At eval a short row sharing a batch with a long
one attends over a run of EOS -- the token trained to mean "stop" -- before it
ever reaches the fact text.

This measures that, and nothing else, by running the SAME rows three ways:

    Path A   batch size 1    -> width == own length, zero pad. This reproduces
                               training's "no pad before real content" geometry.
    Path B   batch size 16   -> file-order batches, the current default, where
                               ~52% of attended positions are EOS pad.
    Path C   batch size 16   -> identical rows and batches to B, but RIGHT-padded,
                               which is what fine-tuning did.

C is the control that makes B interpretable. Right-padding leaves every row's real
tokens at positions 0..L-1 with pad only after them, and causal attention never
looks forward, so C should reproduce A up to batched floating-point
nondeterminism. If A and C agree, the harness is sound and the single remaining
variable between B and C is which side the padding is on. If A and C disagree,
something other than padding side is moving the numbers and B-vs-A cannot be read
as a padding result.

Paths A and B import `greedy_generate` and `parse_generation` from evaluate.py
and call them unchanged. Nothing here reimplements or corrects them -- a local "fixed"
copy would measure the fix, not the thing being diagnosed. Scoring likewise
reuses evaluate.py's `multilabel_scores`, so the F1 numbers are comparable to a
real eval run.

Batch composition is preserved exactly. Paths B and C call their generate
function once per 16-row group, which is what the full-split call does internally at each `i`, so
each row sees the identical padding it would see in a normal run. Path A chunks
by 10 only to emit progress; at batch size 1 every row is its own batch, so
chunking cannot change the result.

    python src/diag_padding.py                          # mps, first 100 rows
    python src/diag_padding.py --device cpu --limit 40
    python src/diag_padding.py --limit 500 --out scratchpad/diag_500.json

Reads nothing it does not already need and writes only its --out JSON.
"""

import argparse
import contextlib
import io
import json
import time
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

# Imported unchanged -- this is the code under test, not a copy of it.
from evaluate import (
    ILSI_FT,
    LABEL_MAP,
    TOKENIZER,
    greedy_generate,
    multilabel_scores,
    parse_generation,
)
from model import CONFIGS, EOS_ID, build_model

FAIL_KEYS = ("duplicate", "no_eos", "non_label", "empty")


@torch.no_grad()
def greedy_generate_rightpad(model, prompts, max_new, eos_id, device, batch_size,
                             ctx):
    """evaluate.py's greedy_generate, verbatim, with padding moved to the right.

    Local to this diagnostic. evaluate.py is not touched, and this is never used
    to score a real run -- it exists so Path C can hold batch size and batch
    composition fixed while flipping only the padding side.

    Exactly three things differ from the original:

      1. pad goes AFTER the real tokens (`X[b, :len(p)]`), not before
      2. each row's logits are read at its own last real index, not at the
         shared final column -- with left padding those coincide, with right
         padding they do not
      3. each row's generated token is written at its own cursor and that cursor
         advances per row, replacing the single shared `torch.cat` column

    Deliberately unchanged, because they are the thing being measured: same
    eos_id as the pad value, same `x[:, -ctx:]` re-slice, same stopping
    condition, same max_new, no attention mask, no KV cache, no dtype or device
    handling of its own.
    """
    out = []
    for i in range(0, len(prompts), batch_size):
        chunk = [p[-ctx:] for p in prompts[i:i + batch_size]]
        width = max(len(p) for p in chunk)
        X = np.full((len(chunk), width), eos_id, dtype=np.int64)
        for b, p in enumerate(chunk):
            X[b, :len(p)] = p                      # CHANGE 1: pad on the right
        x = torch.from_numpy(X).to(device)
        done = torch.zeros(len(chunk), dtype=torch.bool, device=device)
        gen = [[] for _ in chunk]
        cur = [len(p) for p in chunk]              # CHANGE 3: per-row write cursor
        for _ in range(max_new):
            view = x[:, -ctx:]
            # The re-slice is kept as-is, so once x outgrows ctx it drops leading
            # columns -- which under right padding are real tokens. Offset the
            # per-row index by however many columns it removed.
            off = x.shape[1] - view.shape[1]
            logits = model(view)
            # CHANGE 2: own last real index, not the shared final column.
            idx = [max(cur[b] - 1 - off, 0) for b in range(len(chunk))]
            nxt = torch.stack([logits[b, idx[b], :]
                               for b in range(len(chunk))]).float().argmax(-1)
            for b in range(len(chunk)):
                if not done[b]:
                    gen[b].append(int(nxt[b]))
                    if int(nxt[b]) == eos_id:
                        done[b] = True
            if bool(done.all()):
                break
            # CHANGE 3: grow by one pad column, then each row writes at its own
            # cursor. Rows are independent along the batch dim, so a finished
            # row leaving its tail as pad cannot affect any other row.
            x = torch.cat([x, torch.full((len(chunk), 1), eos_id, dtype=x.dtype,
                                         device=device)], dim=1)
            for b in range(len(chunk)):
                if not done[b]:
                    x[b, cur[b]] = nxt[b]
                    cur[b] += 1
        out.extend(gen)
        if (i // batch_size) % 20 == 0:
            print(f"   {min(i+batch_size, len(prompts)):,}/{len(prompts):,}", flush=True)
    return out


def resolve_device(name):
    """Honour --device, but refuse to silently run somewhere else."""
    if name == "mps":
        if not torch.backends.mps.is_available():
            raise SystemExit(
                "MPS unavailable (needs macOS 12.3+, arm64, and an MPS-enabled "
                "torch build). Re-run with --device cpu.")
        return torch.device("mps")
    return torch.device("cpu")


def load_model(ckpt_path, cfg, device, dtype):
    """Same load path as evaluate.py:168-175, plus an optional dtype cast.

    fp32 is the default because evaluate.py loads fp32 and never casts; matching
    it keeps this diagnostic comparable to the run it is diagnosing. --dtype
    float16 is offered for MPS memory headroom and falls back on error.
    """
    model = build_model(cfg).to(device)
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ck.get("model_state_dict", ck) if isinstance(ck, dict) else ck
    model.load_state_dict(state)
    model.eval()

    if dtype == "float16":
        try:
            model = model.half()
            with torch.no_grad():                      # prove a forward works
                model(torch.zeros((1, 8), dtype=torch.long, device=device))
            print("[dtype] float16")
        except Exception as e:
            model = model.float()
            print(f"[dtype] float16 failed on {device.type} "
                  f"({type(e).__name__}: {e}); using float32")
    else:
        print("[dtype] float32 (matches evaluate.py)")

    name = ck.get("config_name") if isinstance(ck, dict) else None
    print(f"[ckpt] {ckpt_path} (config_name={name!r}, "
          f"step={ck.get('step') if isinstance(ck, dict) else '?'})")
    return model


def run_path(model, prompts, *, batch_size, group, max_new, eos_id, device, ctx,
             label, gen_fn=greedy_generate):
    """Drive `gen_fn` in groups, returning (generations, widths, pads, secs).

    `group` rows are handed to gen_fn per call. For Paths B and C group ==
    batch_size, so each call reproduces exactly one batch of the full run.
    gen_fn's own progress prints are captured so the progress lines below stay
    readable; its behaviour is untouched.

    Pad accounting is identical for both padding sides: gen_fn clips to
    `p[-ctx:]`, takes `width = max(len)`, and pads every row to it, so a row's
    pad count is `width - len(p)` whichever side it lands on.
    """
    gens, pads, widths = [], [], []
    t0 = time.time()
    for i in range(0, len(prompts), group):
        chunk = prompts[i:i + group]
        # Mirror greedy_generate:108-109 to recover the geometry it will use.
        clipped = [p[-ctx:] for p in chunk]
        for j in range(0, len(clipped), batch_size):
            sub = clipped[j:j + batch_size]
            w = max(len(p) for p in sub)
            for p in sub:
                widths.append(w)
                pads.append(w - len(p))
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            gens.extend(gen_fn(model, chunk, max_new, eos_id, device,
                               batch_size, ctx))
        done = min(i + group, len(prompts))
        print(f"   [{label}] {done:,}/{len(prompts):,} rows "
              f"({time.time()-t0:.0f}s)", flush=True)
    return gens, widths, pads, time.time() - t0


def summarise(records, gold):
    """evaluate.py's own metric code over this path's predictions."""
    pred = [set(r["parsed_labels"]) for r in records]
    m = multilabel_scores(gold, pred)
    n = max(len(records), 1)
    return {
        "micro_f1": m["micro_f1"],
        "macro_f1": m["macro_f1"],
        "exact_match": m["exact_match"],
        "parse_failure_rate": sum(r["parse_failure"] for r in records) / n,
        "empty_rate": sum(r["empty"] for r in records) / n,
        "n_parse_failures": sum(r["parse_failure"] for r in records),
        "n_empty": sum(r["empty"] for r in records),
    }


def mean(xs):
    return sum(xs) / len(xs) if xs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/final_model.pt")
    ap.add_argument("--config", default="prod_300m", choices=list(CONFIGS))
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--device", default="mps", choices=["mps", "cpu"])
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16"])
    ap.add_argument("--max-new", type=int, default=40)
    ap.add_argument("--batch-b", type=int, default=16,
                    help="Path B batch size (Path A is always 1)")
    ap.add_argument("--out", default="scratchpad/diag_padding.json")
    args = ap.parse_args()

    device = resolve_device(args.device)
    cfg = CONFIGS[args.config]
    ctx = cfg.max_seq_len

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
    rows = rows[: args.limit]                      # FIRST N, file order
    prompts = [r["input_ids"] for r in rows]
    gold = [set(r["labels"]) for r in rows]
    lens = [len(p) for p in prompts]
    print(f"{len(rows):,} {args.split} rows (first {args.limit}, file order) | "
          f"device {device.type} | ctx {ctx} | prompt len "
          f"min {min(lens):,} / mean {mean(lens):.0f} / max {max(lens):,}")

    model = load_model(args.ckpt, cfg, device, args.dtype)

    print(f"\n--- PATH A: batch size 1 (zero padding) ---")
    gen_a, wid_a, pad_a, secs_a = run_path(
        model, prompts, batch_size=1, group=10, max_new=args.max_new,
        eos_id=eos_id, device=device, ctx=ctx, label="A")

    print(f"\n--- PATH B: batch size {args.batch_b} (file order, current default) ---")
    gen_b, wid_b, pad_b, secs_b = run_path(
        model, prompts, batch_size=args.batch_b, group=args.batch_b,
        max_new=args.max_new, eos_id=eos_id, device=device, ctx=ctx, label="B")

    print(f"\n--- PATH C: batch size {args.batch_b} (file order, RIGHT-padded) ---")
    gen_c, wid_c, pad_c, secs_c = run_path(
        model, prompts, batch_size=args.batch_b, group=args.batch_b,
        max_new=args.max_new, eos_id=eos_id, device=device, ctx=ctx, label="C",
        gen_fn=greedy_generate_rightpad)

    recs_a, recs_b, recs_c = [], [], []
    for i, r in enumerate(rows):
        for gen, wid, pad, recs, with_pad in (
                (gen_a, wid_a, pad_a, recs_a, False),
                (gen_b, wid_b, pad_b, recs_b, True),
                (gen_c, wid_c, pad_c, recs_c, True)):
            labels, flags = parse_generation(gen[i], eos_id, id_to_label)
            rec = {
                "row_index": i,
                "id": r["id"],
                "prompt_len": len(prompts[i]),
                "raw_decoded": tok.decode(gen[i], skip_special_tokens=False)[:120],
                "n_generated": len(gen[i]),
                "parsed_labels": sorted(labels),
                "gold_labels": sorted(gold[i]),
                "empty": flags["empty"],
                "parse_failure": any(flags[k] for k in FAIL_KEYS),
                "flags": flags,
            }
            if with_pad:
                rec["batch_max_len"] = wid[i]
                rec["pad_count"] = pad[i]
            recs.append(rec)

    sum_a, sum_b = summarise(recs_a, gold), summarise(recs_b, gold)
    sum_c = summarise(recs_c, gold)
    sum_a["wall_clock_seconds"] = secs_a
    sum_b["wall_clock_seconds"] = secs_b
    sum_c["wall_clock_seconds"] = secs_c

    def pad_split(recs):
        return ([r["pad_count"] for r in recs if r["parse_failure"]],
                [r["pad_count"] for r in recs if not r["parse_failure"]])

    fail_pads, ok_pads = pad_split(recs_b)
    fail_pads_c, ok_pads_c = pad_split(recs_c)
    # The control: right padding should leave A and C identical, so any row here
    # is a row where something other than padding side moved the prediction.
    a_vs_c_diff = [i for i in range(len(rows))
                   if set(recs_a[i]["parsed_labels"])
                   != set(recs_c[i]["parsed_labels"])]
    b_vs_c_diff = [i for i in range(len(rows))
                   if set(recs_b[i]["parsed_labels"])
                   != set(recs_c[i]["parsed_labels"])]
    a_ok_b_fail = [i for i in range(len(rows))
                   if not recs_a[i]["parse_failure"] and recs_b[i]["parse_failure"]]
    b_ok_a_fail = [i for i in range(len(rows))
                   if recs_a[i]["parse_failure"] and not recs_b[i]["parse_failure"]]

    w = 78
    print("\n" + "=" * w)
    print(f"PADDING DIAGNOSTIC -- {len(rows):,} rows, {args.split} split")
    print("=" * w)
    hb, hc = f"B (bs={args.batch_b} left)", f"C (bs={args.batch_b} right)"
    print(f"\n1. F1 (evaluate.py multilabel_scores)")
    print(f"   {'':20s} {'A (bs=1)':>12s} {hb:>16s} {hc:>17s} "
          f"{'B-A':>9s} {'C-A':>9s}")
    for key in ("micro_f1", "macro_f1", "exact_match"):
        print(f"   {key:20s} {sum_a[key]:12.4f} {sum_b[key]:16.4f} "
              f"{sum_c[key]:17.4f} {sum_b[key]-sum_a[key]:+9.4f} "
              f"{sum_c[key]-sum_a[key]:+9.4f}")

    print(f"\n2. Failure rates")
    for key in ("parse_failure_rate", "empty_rate"):
        print(f"   {key:20s} {sum_a[key]:12.4f} {sum_b[key]:16.4f} "
              f"{sum_c[key]:17.4f} {sum_b[key]-sum_a[key]:+9.4f} "
              f"{sum_c[key]-sum_a[key]:+9.4f}")
    print(f"   {'(counts)':20s} "
          f"{sum_a['n_parse_failures']:5d}f/{sum_a['n_empty']:<3d}e "
          f"{sum_b['n_parse_failures']:11d}f/{sum_b['n_empty']:<3d}e "
          f"{sum_c['n_parse_failures']:12d}f/{sum_c['n_empty']:<3d}e")

    print(f"\n3. Pad count vs outcome")
    for name, fp, op in (("B (left)", fail_pads, ok_pads),
                         ("C (right)", fail_pads_c, ok_pads_c)):
        print(f"   Path {name}")
        print(f"      failing    n={len(fp):4d}  mean pad "
              f"{mean(fp) if fp else float('nan'):8.1f}")
        print(f"      succeeding n={len(op):4d}  mean pad "
              f"{mean(op) if op else float('nan'):8.1f}")
        if fp and op:
            print(f"      difference           {mean(fp)-mean(op):+8.1f} "
                  f"pad tokens on failing rows")

    print(f"\n4. Disagreement")
    print(f"   A parsed OK, B failed  : {len(a_ok_b_fail):4d}  rows {a_ok_b_fail[:12]}")
    print(f"   B parsed OK, A failed  : {len(b_ok_a_fail):4d}  rows {b_ok_a_fail[:12]}")
    print(f"   A vs C label sets differ: {len(a_vs_c_diff):4d}  rows {a_vs_c_diff[:12]}")
    print(f"      (control -- expected ~0; nonzero means something other than")
    print(f"       padding side is moving predictions, so read B-A with care)")
    print(f"   B vs C label sets differ: {len(b_vs_c_diff):4d}  rows {b_vs_c_diff[:12]}")
    print(f"      (same batch size and composition -- padding side is the only"
          f" variable)")

    print(f"\n5. Wall clock")
    print(f"   Path A (bs=1)      {secs_a:8.1f}s  ({secs_a/max(len(rows),1):.2f}s/row)")
    print(f"   Path B (bs={args.batch_b:2d} left) {secs_b:8.1f}s  ({secs_b/max(len(rows),1):.2f}s/row)")
    print(f"   Path C (bs={args.batch_b:2d} right){secs_c:8.1f}s  ({secs_c/max(len(rows),1):.2f}s/row)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "meta": {
            "checkpoint": args.ckpt, "config": cfg.name, "split": args.split,
            "limit": args.limit, "n": len(rows), "device": device.type,
            "dtype": args.dtype, "max_new": args.max_new,
            "path_a_batch_size": 1, "path_b_batch_size": args.batch_b,
            "path_c_batch_size": args.batch_b, "path_c_padding": "right",
            "ctx": ctx,
        },
        "summary": {"path_a": sum_a, "path_b": sum_b, "path_c": sum_c},
        "pad_analysis": {
            "path_b": {
                "mean_pad_failing": mean(fail_pads),
                "mean_pad_succeeding": mean(ok_pads),
                "n_failing": len(fail_pads), "n_succeeding": len(ok_pads),
            },
            "path_c": {
                "mean_pad_failing": mean(fail_pads_c),
                "mean_pad_succeeding": mean(ok_pads_c),
                "n_failing": len(fail_pads_c), "n_succeeding": len(ok_pads_c),
            },
        },
        "disagreement": {"a_ok_b_fail": a_ok_b_fail, "b_ok_a_fail": b_ok_a_fail,
                         "a_vs_c_label_set_diff": a_vs_c_diff,
                         "b_vs_c_label_set_diff": b_vs_c_diff},
        "rows": {"path_a": recs_a, "path_b": recs_b, "path_c": recs_c},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n   -> {out}")


if __name__ == "__main__":
    main()
