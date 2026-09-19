#!/usr/bin/env python
"""Build a reframed, mixed fine-tuning set: data/ilsi_ft_v2/

WHY THE FORMAT CHANGES
----------------------
The slot-1 fine-tune fed the model `fact_text + label_tokens` with no
separator, no instruction and no marker of any kind. The eval prompt was a
character-exact prefix of the training sequence. So the model had no signal
distinguishing "predict IPC sections" from "continue this text", and after 3
epochs it answers *everything* with label tokens -- measured: 4/5 generic
prompts decode to an empty string, which is a hard FAIL under L1 artifact
validity.

Adding data without fixing the framing reproduces the same failure. So every
row here carries a uniform task frame built from ORDINARY vocabulary tokens
(no new special tokens -- the embedding table cannot absorb them now):

    <passage tokens> ### TASK: <instruction>\\n### ANSWER:\\n <target tokens>

THE TRUNCATION TRAP
-------------------
ILSIFinetuneDataset does:  inp = r["input_ids"][: ctx + 1 - len(tgt)]
That trims from the END, so a marker appended to input_ids would be silently
deleted on exactly the long rows that most need it. This builder therefore
truncates the passage at BUILD time and appends the marker after, so the
dataset's own truncation is a no-op. Every emitted row satisfies
len(input_ids) + len(target_ids) <= ctx + 1.

ROW TYPES
---------
  ipc          -- your existing rows, reframed. target_ids unchanged.
  instruction  -- the six checker families the evaluation framework names for
                  its Instruction Following (I) score. Targets are derived
                  programmatically from the passage, so ground truth is exact
                  by construction. This is ordinary SFT on a task family, not
                  training on anyone's eval set.
  continuation -- target is ordinary prose. This is the piece that actually
                  protects the text distribution and stops the label collapse.

IMPORTANT: test.jsonl and dev.jsonl are reframed too. evaluate.py builds
prompts straight from r["input_ids"], so evaluating a v2 model against v1 test
rows would measure the wrong thing.

Usage:
    PYTHONPATH=src python src/build_mixed_ft.py --n-instruction 6000 --n-continuation 4000
"""
import argparse
import json
import random
import re
from pathlib import Path

from tokenizers import Tokenizer

from model import EOS_ID

TOKENIZER = Path("data/tokenizer/bpe_32000_final.json")
SRC = Path("data/ilsi_ft")
OUT = Path("data/ilsi_ft_v2")

IPC_TASK = "identify the applicable IPC sections"
CONT_TASK = "continue the passage"

FRAME = "\n\n### TASK: {task}\n### ANSWER:\n"

# ---------------------------------------------------------------- helpers


def sentences(text):
    """Deliberately conservative splitter -- abbreviations and decimals break
    naive splitting on '.', and the framework's own checkers use a real
    sentencizer, so anything we cannot split cleanly we simply skip."""
    text = re.sub(r"\s+", " ", text).strip()
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z(])", text)
    return [p.strip() for p in parts if len(p.strip()) > 20]


def words(text):
    return [w for w in re.sub(r"\s+", " ", text).strip().split(" ") if w]


# ------------------------------------------------- instruction generators
# Each returns (instruction_text, target_text) or None if the passage does not
# support that task. Ground truth is derived, never guessed.


def t_sentence_count(text, rng):
    sents = sentences(text)
    n = rng.choice([2, 3, 4])
    if len(sents) < n:
        return None
    return (f"write exactly {n} sentences from the passage",
            " ".join(sents[:n]))


def t_json_keys(text, rng):
    sents = sentences(text)
    if not sents:
        return None
    ws = words(sents[0])
    if len(ws) < 4:
        return None
    obj = {"first_word": ws[0], "word_count": len(ws), "n_sentences": len(sents)}
    return ("return a JSON object with keys first_word, word_count, n_sentences",
            json.dumps(obj))


def t_point_count(text, rng):
    sents = sentences(text)
    n = rng.choice([2, 3, 4])
    if len(sents) < n:
        return None
    body = "\n".join(f"{i+1}. {s}" for i, s in enumerate(sents[:n]))
    return (f"list exactly {n} numbered points from the passage", body)


def t_word_limit(text, rng):
    ws = words(text)
    n = rng.choice([10, 15, 20, 25])
    if len(ws) < n + 5:
        return None
    return (f"answer in at most {n} words", " ".join(ws[:n]))


def t_transform(text, rng):
    sents = sentences(text)
    if not sents:
        return None
    mode = rng.choice(["upper", "lower", "reverse_words"])
    s = sents[0]
    if mode == "upper":
        return ("rewrite the first sentence in upper case", s.upper())
    if mode == "lower":
        return ("rewrite the first sentence in lower case", s.lower())
    return ("rewrite the first sentence with the word order reversed",
            " ".join(reversed(words(s))))


def t_extract(text, rng):
    nums = re.findall(r"\b\d[\d,]*\b", text)
    if len(nums) < 2:
        return None
    return ("extract every number in the passage, comma separated",
            ", ".join(nums[:8]))


GENERATORS = [t_sentence_count, t_json_keys, t_point_count,
              t_word_limit, t_transform, t_extract]


# ---------------------------------------------------------------- builder


def frame_ids(tok, task):
    return tok.encode(FRAME.format(task=task)).ids


def make_row(rid, kind, passage_ids, marker_ids, target_ids, ctx):
    """Truncate the passage so the frame and target always survive."""
    budget = ctx + 1 - len(marker_ids) - len(target_ids)
    if budget < 32:
        return None
    inp = list(passage_ids[:budget]) + list(marker_ids)
    return {"id": rid, "kind": kind,
            "input_ids": inp, "target_ids": list(target_ids)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--n-instruction", type=int, default=6000)
    ap.add_argument("--n-continuation", type=int, default=4000)
    ap.add_argument("--cont-target-tokens", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-target", type=int, default=160)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    tok = Tokenizer.from_file(str(TOKENIZER))
    OUT.mkdir(parents=True, exist_ok=True)

    ipc_marker = frame_ids(tok, IPC_TASK)
    cont_marker = frame_ids(tok, CONT_TASK)
    print(f"IPC frame = {len(ipc_marker)} tokens: "
          f"{FRAME.format(task=IPC_TASK)!r}")

    # ---- reframe all three splits (eval prompts must match training) ----
    counts = {}
    for split in ("train", "dev", "test"):
        src = SRC / f"{split}.jsonl"
        if not src.exists():
            print(f"  [skip] {src} missing")
            continue
        n_out = 0
        with src.open(encoding="utf-8") as fin, \
                (OUT / f"{split}.jsonl").open("w", encoding="utf-8") as fout:
            for line in fin:
                r = json.loads(line)
                row = make_row(r["id"], "ipc", r["input_ids"], ipc_marker,
                               r["target_ids"], args.ctx)
                if row:
                    fout.write(json.dumps(row) + "\n")
                    n_out += 1
        counts[split] = n_out
        print(f"  {split}: {n_out:,} ipc rows -> {OUT/f'{split}.jsonl'}")

    # ---- source passages for the synthetic rows, from TRAIN only ----
    need = args.n_instruction + args.n_continuation
    pool = []
    with (SRC / "train.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            if len(pool) >= need * 3:
                break
            pool.append(json.loads(line)["input_ids"])
    rng.shuffle(pool)
    print(f"\npassage pool: {len(pool):,}")

    extra, gi, made = [], 0, {"instruction": 0, "continuation": 0}
    fails = 0

    for ids in pool:
        if made["instruction"] >= args.n_instruction and \
                made["continuation"] >= args.n_continuation:
            break

        # decode a bounded window; full 2048-token decodes are wasteful here
        window = ids[:512]
        try:
            text = tok.decode(window)
        except Exception:
            fails += 1
            continue

        if made["continuation"] < args.n_continuation and rng.random() < 0.4:
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

        if made["instruction"] < args.n_instruction:
            gen = GENERATORS[gi % len(GENERATORS)]
            gi += 1
            out = gen(text, rng)
            if not out:
                fails += 1
                continue
            instruction, target_text = out
            tgt = tok.encode(target_text).ids
            if not tgt or len(tgt) > args.max_target:
                fails += 1
                continue
            tgt = tgt + [EOS_ID]
            row = make_row(f"instr_{made['instruction']}", "instruction",
                           window, frame_ids(tok, instruction), tgt, args.ctx)
            if row:
                extra.append(row)
                made["instruction"] += 1

    print(f"generated: {made['instruction']:,} instruction, "
          f"{made['continuation']:,} continuation ({fails:,} skipped)")

    # ---- append to train and shuffle, so the mix is interleaved ----
    train_path = OUT / "train.jsonl"
    rows = [json.loads(l) for l in train_path.open(encoding="utf-8")]
    rows.extend(extra)
    rng.shuffle(rows)
    with train_path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    kinds = {}
    for r in rows:
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
    total = len(rows)
    print(f"\ntrain: {total:,} rows")
    for k, v in sorted(kinds.items()):
        print(f"  {k:<13} {v:>7,}  ({100*v/total:.1f}%)")

    manifest = {
        "note": "Reframed + mixed fine-tune set. Every row carries a task "
                "frame in ordinary vocabulary tokens. Passages truncated at "
                "build time so ILSIFinetuneDataset's own truncation is a "
                "no-op and the frame always survives.",
        "frame": FRAME,
        "ipc_task": IPC_TASK,
        "ctx": args.ctx,
        "counts": kinds,
        "splits": counts,
        "seed": args.seed,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2),
                                       encoding="utf-8")

    # ---- show one row of each kind, decoded, so you can eyeball it ----
    print("\n" + "=" * 60)
    for kind in ("ipc", "instruction", "continuation"):
        r = next((x for x in rows if x["kind"] == kind), None)
        if not r:
            continue
        print(f"\n--- {kind} (in {len(r['input_ids'])} tok, "
              f"tgt {len(r['target_ids'])} tok)")
        print("  prompt tail :", repr(tok.decode(r["input_ids"][-60:])))
        print("  target      :", repr(tok.decode(r["target_ids"])))


if __name__ == "__main__":
    main()
