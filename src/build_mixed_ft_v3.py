#!/usr/bin/env python
"""Build data/ilsi_ft_v3/ -- an instruction-heavy remix.

WHY v3 EXISTS
-------------
v2 was 81% IPC / 11.4% instruction / 7.6% continuation. That weighting was
built to protect IPC accuracy. But IPC accuracy is not scored anywhere in the
evaluation framework -- Level 3 is an organiser-authored domain benchmark, not
your label-prediction task. Instruction Following is 30% of Level 2 and is the
only component still reachable with the GPU time left.

So v3 rebalances toward what is actually scored:

    IPC          30,000  (~58%)   subsampled from 42,750
    instruction  14,000  (~27%)   was 11.4%
    continuation  8,000  (~15%)   was 7.6%

Row count stays near 52k so training time is unchanged (~3,300 steps at
48 examples/step over 3 epochs).

WHAT ELSE CHANGED
-----------------
Fourteen generators instead of six. The framework names six checker families;
the point of the extra generators is variety WITHIN those families, since a
model that has seen exactly one JSON key-set has memorised a string rather
than learned a format.

Everything else -- the task frame, build-time truncation, target derivation --
is identical to v2. Ground truth is still computed from the passage, never
guessed, so every target is exactly checkable.

Usage:
    PYTHONPATH=src python src/build_mixed_ft_v3.py
"""
import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

from tokenizers import Tokenizer

from model import EOS_ID

TOKENIZER = Path("data/tokenizer/bpe_32000_final.json")
SRC = Path("data/ilsi_ft_v1")          # the ORIGINAL rows, not v2's reframed
OUT = Path("data/ilsi_ft_v3")

IPC_TASK = "identify the applicable IPC sections"
CONT_TASK = "continue the passage"
FRAME = "\n\n### TASK: {task}\n### ANSWER:\n"


def sentences(text):
    text = re.sub(r"\s+", " ", text).strip()
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z(])", text)
    return [p.strip() for p in parts if len(p.strip()) > 20]


def words(text):
    return [w for w in re.sub(r"\s+", " ", text).strip().split(" ") if w]


# ------------------------------------------------------------ generators
# Each returns (instruction, target) or None. Grouped by the framework's
# six checker families.

# --- family: exact sentence count
def g_sentence_count(text, rng):
    s = sentences(text)
    n = rng.choice([2, 3, 4])
    if len(s) < n:
        return None
    return (f"write exactly {n} sentences from the passage", " ".join(s[:n]))


def g_nth_sentence(text, rng):
    s = sentences(text)
    if len(s) < 3:
        return None
    n = rng.randint(2, min(4, len(s)))
    return (f"give sentence number {n} from the passage", s[n - 1])


def g_first_last(text, rng):
    s = sentences(text)
    if len(s) < 3:
        return None
    return ("give the first and the last sentence, numbered",
            f"1. {s[0]}\n2. {s[-1]}")


# --- family: valid JSON with specified keys
def g_json_basic(text, rng):
    s = sentences(text)
    if not s:
        return None
    w = words(s[0])
    if len(w) < 4:
        return None
    return ("return a JSON object with keys first_word, word_count, n_sentences",
            json.dumps({"first_word": w[0], "word_count": len(w),
                        "n_sentences": len(s)}))


def g_json_stats(text, rng):
    s = sentences(text)
    if not s:
        return None
    w = words(text)
    return ("return a JSON object with keys n_words, n_chars, last_word",
            json.dumps({"n_words": len(w), "n_chars": len(text.strip()),
                        "last_word": w[-1] if w else ""}))


def g_json_numbers(text, rng):
    nums = re.findall(r"\b\d[\d,]*\b", text)
    if len(nums) < 2:
        return None
    return ("return a JSON object with keys count and values, "
            "where values is an array of the numbers in the passage",
            json.dumps({"count": len(nums[:6]), "values": nums[:6]}))


# --- family: exact point count
def g_point_count(text, rng):
    s = sentences(text)
    n = rng.choice([2, 3, 4])
    if len(s) < n:
        return None
    return (f"list exactly {n} numbered points from the passage",
            "\n".join(f"{i+1}. {x}" for i, x in enumerate(s[:n])))


def g_bullet_count(text, rng):
    s = sentences(text)
    n = rng.choice([2, 3])
    if len(s) < n:
        return None
    return (f"list exactly {n} bullet points from the passage",
            "\n".join(f"- {x}" for x in s[:n]))


# --- family: word limit
def g_word_limit(text, rng):
    w = words(text)
    n = rng.choice([10, 15, 20, 25, 30])
    if len(w) < n + 5:
        return None
    return (f"answer in at most {n} words", " ".join(w[:n]))


def g_comma_list(text, rng):
    w = words(text)
    n = rng.choice([5, 8, 12])
    if len(w) < n + 3:
        return None
    return (f"list the first {n} words of the passage, comma separated",
            ", ".join(w[:n]))


# --- family: defined text transformation
def g_case(text, rng):
    s = sentences(text)
    if not s:
        return None
    mode = rng.choice(["upper", "lower", "title"])
    if mode == "upper":
        return ("rewrite the first sentence in upper case", s[0].upper())
    if mode == "lower":
        return ("rewrite the first sentence in lower case", s[0].lower())
    return ("rewrite the first sentence in title case", s[0].title())


def g_reverse(text, rng):
    s = sentences(text)
    if not s:
        return None
    return ("rewrite the first sentence with the word order reversed",
            " ".join(reversed(words(s[0]))))


def g_strip_digits(text, rng):
    s = sentences(text)
    if not s or not re.search(r"\d", s[0]):
        return None
    out = re.sub(r"\s+", " ", re.sub(r"\d", "", s[0])).strip()
    if len(out) < 15:
        return None
    return ("rewrite the first sentence with all digits removed", out)


# --- family: structured extraction
def g_extract_numbers(text, rng):
    nums = re.findall(r"\b\d[\d,]*\b", text)
    if len(nums) < 2:
        return None
    return ("extract every number in the passage, comma separated",
            ", ".join(nums[:8]))


GENERATORS = [
    g_sentence_count, g_nth_sentence, g_first_last,
    g_json_basic, g_json_stats, g_json_numbers,
    g_point_count, g_bullet_count,
    g_word_limit, g_comma_list,
    g_case, g_reverse, g_strip_digits,
    g_extract_numbers,
]


def frame_ids(tok, task):
    return tok.encode(FRAME.format(task=task)).ids


def make_row(rid, kind, passage_ids, marker_ids, target_ids, ctx):
    budget = ctx + 1 - len(marker_ids) - len(target_ids)
    if budget < 32:
        return None
    return {"id": rid, "kind": kind,
            "input_ids": list(passage_ids[:budget]) + list(marker_ids),
            "target_ids": list(target_ids)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--ipc-cap", type=int, default=30000)
    ap.add_argument("--n-instruction", type=int, default=14000)
    ap.add_argument("--n-continuation", type=int, default=8000)
    ap.add_argument("--cont-target-tokens", type=int, default=48)
    ap.add_argument("--max-target", type=int, default=160)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    if not SRC.exists():
        raise SystemExit(f"{SRC} not found. The original rows were moved to "
                         f"data/ilsi_ft_v1 when the symlink was created.")

    rng = random.Random(args.seed)
    tok = Tokenizer.from_file(str(TOKENIZER))
    OUT.mkdir(parents=True, exist_ok=True)

    ipc_marker = frame_ids(tok, IPC_TASK)
    cont_marker = frame_ids(tok, CONT_TASK)

    # ---- dev and test: reframed in full, never subsampled ----
    for split in ("dev", "test"):
        src = SRC / f"{split}.jsonl"
        if not src.exists():
            continue
        n = 0
        with src.open(encoding="utf-8") as fin, \
                (OUT / f"{split}.jsonl").open("w", encoding="utf-8") as fout:
            for line in fin:
                r = json.loads(line)
                row = make_row(r["id"], "ipc", r["input_ids"], ipc_marker,
                               r["target_ids"], args.ctx)
                if row:
                    fout.write(json.dumps(row) + "\n")
                    n += 1
        print(f"  {split}: {n:,} rows")

    # ---- train: IPC subsample ----
    all_train = [json.loads(l) for l in
                 (SRC / "train.jsonl").open(encoding="utf-8")]
    rng.shuffle(all_train)
    ipc_rows = []
    for r in all_train[:args.ipc_cap]:
        row = make_row(r["id"], "ipc", r["input_ids"], ipc_marker,
                       r["target_ids"], args.ctx)
        if row:
            ipc_rows.append(row)
    print(f"  train ipc: {len(ipc_rows):,} (from {len(all_train):,})")

    # ---- synthetic rows, drawn from the UNUSED remainder ----
    pool = [r["input_ids"] for r in all_train[args.ipc_cap:]]
    if len(pool) < 2000:
        pool = [r["input_ids"] for r in all_train]
    rng.shuffle(pool)
    print(f"  passage pool: {len(pool):,}")

    extra = []
    made = {"instruction": 0, "continuation": 0}
    used = Counter()
    gi = skipped = 0

    for ids in pool:
        if made["instruction"] >= args.n_instruction and \
                made["continuation"] >= args.n_continuation:
            break

        if made["continuation"] < args.n_continuation and rng.random() < 0.35:
            if len(ids) < 256:
                continue
            cut = len(ids) // 2
            tgt = list(ids[cut:cut + args.cont_target_tokens]) + [EOS_ID]
            row = make_row(f"cont_{made['continuation']}", "continuation",
                           ids[:cut], cont_marker, tgt, args.ctx)
            if row:
                extra.append(row)
                made["continuation"] += 1
            continue

        if made["instruction"] >= args.n_instruction:
            continue

        try:
            text = tok.decode(ids[:512])
        except Exception:
            skipped += 1
            continue

        gen = GENERATORS[gi % len(GENERATORS)]
        gi += 1
        out = gen(text, rng)
        if not out:
            skipped += 1
            continue
        instruction, target_text = out
        tgt = tok.encode(target_text).ids
        if not tgt or len(tgt) > args.max_target:
            skipped += 1
            continue
        row = make_row(f"instr_{made['instruction']}", "instruction",
                       ids[:512], frame_ids(tok, instruction),
                       tgt + [EOS_ID], args.ctx)
        if row:
            extra.append(row)
            made["instruction"] += 1
            used[gen.__name__] += 1

    print(f"  generated: {made['instruction']:,} instruction, "
          f"{made['continuation']:,} continuation ({skipped:,} skipped)")

    rows = ipc_rows + extra
    rng.shuffle(rows)
    with (OUT / "train.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    kinds = Counter(r["kind"] for r in rows)
    total = len(rows)
    print(f"\ntrain: {total:,} rows")
    for k, v in sorted(kinds.items()):
        print(f"  {k:<13} {v:>7,}  ({100*v/total:.1f}%)")

    print("\ngenerator usage:")
    for name, c in used.most_common():
        print(f"  {name:<20} {c:>6,}")

    (OUT / "manifest.json").write_text(json.dumps({
        "note": "Instruction-heavy remix. IPC subsampled; instruction and "
                "continuation rows generated from held-out passages with "
                "programmatically derived ground truth.",
        "frame": FRAME, "ipc_task": IPC_TASK, "ctx": args.ctx,
        "counts": dict(kinds), "generators": dict(used), "seed": args.seed,
    }, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    for kind in ("ipc", "instruction", "continuation"):
        r = next((x for x in rows if x["kind"] == kind), None)
        if r:
            print(f"\n--- {kind}")
            print("  prompt tail :", repr(tok.decode(r["input_ids"][-50:])))
            print("  target      :", repr(tok.decode(r["target_ids"])))


if __name__ == "__main__":
    main()
