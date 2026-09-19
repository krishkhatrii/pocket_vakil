"""
Build ILSI into the generative fine-tuning format.

    input   fact description, head-truncated
    target  sorted IPC label tokens, then EOS

Per PLAN.md section 4: generative, not a classification head, with one special token
per section so a malformed section number is impossible by construction.

**Labels are deduplicated, and that is not a formality.** Four ILSI rows repeat a
label in their own list (train 3, dev 1) -- e.g. [61, 61, 62, 62]. Built naively,
those rows teach the model to emit `<ipc_395>` twice, and a repeated label token is
exactly what the evaluation counts as a parse failure. Four rows out of 65,950, no
error raised, and the damage shows up as a metric you would then try to explain.
`assert_no_duplicates` fails loudly rather than trusting the dedup ran.

**The input budget is per-document, not a fixed 2048.** The target occupies
len(labels) + 1 tokens of the same window, so the input is truncated to
`seq_len - len(target)`. A document with 31 labels gives up 32 tokens of input that
a document with 3 labels keeps. Truncating the input to a flat 2048 and then
appending the target would push every long document past the context.

There is no explicit input->target delimiter. None is needed and none is possible:
the tokenizer is frozen, so no new separator token can be added, and the label
tokens never occur in fact text -- the first `<ipc_*>` is itself the boundary.

Usage:
    python src/build_finetune_targets.py
    python src/build_finetune_targets.py --seq-len 2048
"""

import argparse
import json
import time
from collections import Counter
from pathlib import Path

from tokenizers import Tokenizer

ILSI = Path("data/ilsi")
OUT = Path("data/ilsi_ft")
TOKENIZER = Path("data/tokenizer/bpe_32000_final.json")

EOS = "<|endoftext|>"
SEQ_LEN = 2048
SPLITS = ("train", "dev", "test")


def load(split):
    return [json.loads(l) for l in (ILSI / f"{split}.jsonl").open(encoding="utf-8")]


def dedup(rows, split):
    """Deduplicate labels per row, preserving nothing about original order.

    Order does not matter here because the target is sorted immediately after; what
    matters is that the multiset becomes a set.
    """
    affected = []
    for r in rows:
        labs = r["labels"]
        if len(labs) != len(set(labs)):
            c = Counter(labs)
            affected.append({
                "split": split, "id": r["id"],
                "raw": labs,
                "repeated": {k: v for k, v in c.items() if v > 1},
                "n_before": len(labs), "n_after": len(set(labs)),
            })
        r["labels"] = list(set(labs))
    return affected


def assert_no_duplicates(rows, split):
    bad = [r["id"] for r in rows if len(r["labels"]) != len(set(r["labels"]))]
    if bad:
        raise SystemExit(f"{split}: {len(bad)} rows still hold duplicate labels: {bad[:5]}")


def build_order(sections):
    """Canonical label order: the label_map index, which is ascending by section.

    One comparator, defined once, used for every split -- so 'sorted ascending by
    section number' means the same thing in train, dev, test and at evaluation.
    Sorting the section *strings* would not do this: 'Section 100' < 'Section 34'
    lexicographically.
    """
    return {s: j for j, s in enumerate(sections)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-len", type=int, default=SEQ_LEN)
    ap.add_argument("--batch", type=int, default=500)
    args = ap.parse_args()

    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    tok = Tokenizer.from_file(str(TOKENIZER))
    eos_id = tok.token_to_id(EOS)
    meta = json.loads((ILSI / "label_map.json").read_text(encoding="utf-8"))
    sections, tokens = meta["sections"], meta["special_tokens"]
    order = build_order(sections)
    tok_of = dict(zip(sections, tokens))

    # 1. every label has a special token that exists and is exactly one id
    missing, split_ids = [], []
    ids_of = {}
    for s, t in zip(sections, tokens):
        tid = tok.token_to_id(t)
        if tid is None:
            missing.append(t)
            continue
        enc = tok.encode(t, add_special_tokens=False).ids
        if len(enc) != 1:
            split_ids.append((t, enc))
        ids_of[s] = tid
    if missing or split_ids:
        raise SystemExit(f"label tokens missing {missing} / split {split_ids}")
    print(f"label tokens: {len(ids_of)}/100 present, all single-id, "
          f"eos_id {eos_id}")

    report = {"tokenizer": TOKENIZER.name, "seq_len": args.seq_len,
              "eos_id": eos_id, "splits": {}, "duplicates": []}
    worst = {"labels": 0, "id": None, "split": None}
    examples = []

    for split in SPLITS:
        rows = load(split)
        affected = dedup(rows, split)
        assert_no_duplicates(rows, split)
        report["duplicates"].extend(affected)

        texts = [" ".join(r["text"]) for r in rows]
        # Encode once, uncapped, so the pre-truncation distribution is measurable.
        full = []
        for i in range(0, len(texts), args.batch):
            full.extend(e.ids for e in
                        tok.encode_batch(texts[i:i + args.batch],
                                         add_special_tokens=False))

        n_over_raw = n_trunc = 0
        combined, in_lens, tgt_lens = [], [], []
        path = OUT / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for r, ids in zip(rows, full):
                labs = sorted(r["labels"], key=lambda s: order[s])
                target = [ids_of[s] for s in labs] + [eos_id]
                budget = args.seq_len - len(target)
                if len(ids) + len(target) > args.seq_len:
                    n_over_raw += 1
                inp = ids[:budget]              # head truncation
                if len(inp) < len(ids):
                    n_trunc += 1
                if len(labs) > worst["labels"]:
                    worst = {"labels": len(labs), "id": r["id"], "split": split,
                             "target_tokens": len(target)}
                f.write(json.dumps({
                    "id": r["id"],
                    "input_ids": inp,
                    "target_ids": target,
                    "labels": labs,
                    "n_input": len(inp),
                    "n_target": len(target),
                    "n_total": len(inp) + len(target),
                    "truncated": len(inp) < len(ids),
                    "n_input_untruncated": len(ids),
                }, ensure_ascii=False) + "\n")
                combined.append(len(inp) + len(target))
                in_lens.append(len(ids))
                tgt_lens.append(len(target))
                if split == "train" and len(examples) < 3 and 2 <= len(labs) <= 5:
                    examples.append((r["id"], texts[len(examples)], inp, target, labs))

        combined.sort()
        in_s = sorted(in_lens)
        p = lambda a, q: a[min(int(len(a) * q), len(a) - 1)]
        report["splits"][split] = {
            "rows": len(rows),
            "rows_with_duplicate_labels": len(affected),
            "input_untruncated_p50": p(in_s, .5), "input_untruncated_p90": p(in_s, .9),
            "input_untruncated_max": in_s[-1],
            "rows_where_input_plus_target_exceeds_seq_len": n_over_raw,
            "pct_exceeds": 100 * n_over_raw / len(rows),
            "rows_truncated": n_trunc,
            "pct_truncated": 100 * n_trunc / len(rows),
            "target_min": min(tgt_lens), "target_max": max(tgt_lens),
            "target_mean": sum(tgt_lens) / len(tgt_lens),
            "combined_p50": p(combined, .5), "combined_p90": p(combined, .9),
            "combined_max": combined[-1],
            "combined_over_seq_len_after_build": sum(1 for x in combined
                                                     if x > args.seq_len),
            "file": path.name,
        }
        r = report["splits"][split]
        print(f"  {split:6s} {len(rows):6,} rows | dup {len(affected)} | "
              f"input p90 {r['input_untruncated_p90']:,} | truncated {r['pct_truncated']:.1f}% | "
              f"combined max {r['combined_max']:,} | over-limit after build "
              f"{r['combined_over_seq_len_after_build']}")

    report["worst_case_labels"] = worst
    report["format"] = {
        "input_ids": "head-truncated fact description, no special tokens",
        "target_ids": "IPC label token ids ascending by section, then EOS",
        "sequence": "input_ids + target_ids, total <= seq_len by construction",
        "delimiter": "none; label tokens never occur in fact text, so the first "
                     "<ipc_*> is the boundary. No separator token can be added -- "
                     "the tokenizer is frozen.",
        "loss": "stored separately so loss can be masked to target_ids only",
        "sort_key": "index into label_map.json sections (ascending by section number)",
    }
    (OUT / "manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\nworst case: {worst['labels']} labels ({worst['split']} {worst['id']}) "
          f"-> {worst['target_tokens']} target tokens, "
          f"input budget {args.seq_len - worst['target_tokens']:,}")
    print(f"duplicates fixed: {len(report['duplicates'])}")
    for d in report["duplicates"]:
        print(f"   {d['split']:5s} {d['id']:>10s}  {d['n_before']}->{d['n_after']}  "
              f"repeated {d['repeated']}")

    print("\n" + "=" * 70)
    print("WORKED EXAMPLES")
    print("=" * 70)
    for cid, raw, inp, target, labs in examples:
        print(f"\n--- {cid} ---")
        print(f"RAW FACT ({len(raw):,} chars): {raw[:300]}...")
        print(f"\nINPUT ({len(inp):,} tokens), decoded head: "
              f"{tok.decode(inp[:60], skip_special_tokens=False)[:240]!r}")
        print(f"INPUT tail: {tok.decode(inp[-30:], skip_special_tokens=False)[-140:]!r}")
        print(f"\nTARGET ids: {target}")
        print(f"TARGET decoded: {tok.decode(target, skip_special_tokens=False)!r}")
        print(f"TARGET labels : {labs}")
    print(f"\n-> {OUT}/  [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
