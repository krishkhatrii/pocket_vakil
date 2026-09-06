"""
Inspect ILSI before designing the fine-tuning stage.

Point it at wherever you unzipped the Zenodo download:
    python inspect_ilsi.py /path/to/ilsi

Answers:
  - how many examples per split, and is the label vocab really 100?
  - how many sections does a typical fact cite?
  - how skewed is the label distribution? (drives macro-F1 expectations)
  - how long are the fact texts? (drives context length)
  - do sec_src keys match the strings used in `labels`?
"""

import json
import sys
from collections import Counter
from pathlib import Path


def load_jsonl(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def survey(root):
    files = sorted(root.rglob("*.jsonl")) + sorted(root.rglob("*.json"))
    print(f"Found {len(files)} data files under {root}:\n")
    for f in files:
        print(f"   {f.relative_to(root)}  ({f.stat().st_size / 1e6:.1f} MB)")
    return files


def describe_split(path):
    rows = load_jsonl(path)
    print(f"\n=== {path.name} ===")
    print(f"   {len(rows):,} instances")
    print(f"   keys: {list(rows[0].keys())}")

    # `text` is documented as a list of sentences
    t = rows[0].get("text")
    if isinstance(t, list):
        print(f"   text is a list of {len(t)} sentences; first: {str(t[0])[:120]}")
        lengths = [sum(len(s) for s in r["text"]) for r in rows]
        sents = [len(r["text"]) for r in rows]
        print(f"   sentences/doc: min {min(sents)} / median {sorted(sents)[len(sents)//2]} "
              f"/ max {max(sents)}")
    else:
        lengths = [len(str(r.get("text", ""))) for r in rows]

    lengths.sort()
    print(f"   chars/doc: median {lengths[len(lengths)//2]:,} "
          f"/ p90 {lengths[int(len(lengths)*0.9)]:,} / max {lengths[-1]:,}")
    print(f"   ~tokens at 4 chars/token: median {lengths[len(lengths)//2]//4:,} "
          f"/ p90 {lengths[int(len(lengths)*0.9)]//4:,}")

    return rows


def label_stats(rows, name):
    labelled = [r for r in rows if r.get("labels")]
    if not labelled:
        print(f"   no labels in {name}")
        return Counter()

    per_doc = [len(r["labels"]) for r in labelled]
    per_doc.sort()
    freq = Counter(l for r in labelled for l in r["labels"])

    print(f"\n   labels/doc: mean {sum(per_doc)/len(per_doc):.2f} "
          f"/ median {per_doc[len(per_doc)//2]} / max {per_doc[-1]}")
    print(f"   distinct labels: {len(freq)}")
    print(f"   example label strings: {list(freq)[:6]}")

    counts = sorted(freq.values(), reverse=True)
    total = sum(counts)
    print(f"   top-10 labels cover {100*sum(counts[:10])/total:.1f}% of all label mentions")
    print(f"   rarest 20 labels: {counts[-20:]}")
    print(f"   labels with <50 examples: {sum(1 for c in counts if c < 50)}")
    print(f"   labels with <10 examples: {sum(1 for c in counts if c < 10)}")

    return freq


def check_sec_src(files, train_freq):
    sec = next((f for f in files if "sec" in f.name.lower()), None)
    if not sec:
        print("\n   no sec_src file found")
        return

    rows = load_jsonl(sec)
    print(f"\n=== {sec.name} ===")
    print(f"   {len(rows):,} sections")
    print(f"   keys: {list(rows[0].keys())}")
    print(f"   first id: {rows[0].get('id')}")
    body = rows[0].get("text")
    print(f"   first text: {str(body[0] if isinstance(body, list) else body)[:200]}")

    ids = {r["id"] for r in rows}
    labels = set(train_freq)
    print(f"\n   sec_src ids: {len(ids)}   train label vocab: {len(labels)}")
    print(f"   labels with matching section text: {len(labels & ids)}")
    missing = labels - ids
    if missing:
        print(f"   MISSING text for {len(missing)}: {sorted(missing)[:10]}")


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    files = survey(root)

    train_freq = Counter()
    for f in files:
        if f.suffix != ".jsonl":
            continue
        if any(k in f.name.lower() for k in ("train", "dev", "test", "val")):
            rows = describe_split(f)
            freq = label_stats(rows, f.name)
            if "train" in f.name.lower():
                train_freq = freq

    check_sec_src(files, train_freq)


if __name__ == "__main__":
    main()
