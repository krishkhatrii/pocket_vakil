"""
Fetch ILSI to data/ilsi/ as JSONL, and resolve its integer labels to section ids.

ILSI ships on Zenodo (DOI 10.5281/zenodo.6053791), but it is also mirrored inside
IL-TUR on the Hub, which is faster to reach and needs no unzip step. Splits match
the published sizes exactly: 42,750 / 10,181 / 13,019 and 100 target sections.

Unlike the pretraining corpus this is downloaded, not streamed. It is 232 MB and
gets read many times -- inspection, tokenizer training, several fine-tuning epochs,
evaluation -- and it has to sit on the GPU box for session 2 regardless.

The labels need resolving. IL-TUR stores them as integers, not section strings:
a row reads [69, 9, 3]. Those index the row order of the statutes file, which is
sorted ascending by section number. Verified rather than assumed -- resolving that
way puts Sections 34, 302, 506, 323 and 307 at the top of the training frequency
table, which is exactly the charging pattern of Indian criminal law. A wrong
mapping gives noise instead. `check()` re-asserts the structural half of that at
every run.

Both forms are written: `labels` as section ids, `label_ids` as the untouched
integers, so the mapping stays auditable.

Usage:
    python src/fetch_ilsi.py
    python src/fetch_ilsi.py --out data/ilsi
"""

import argparse
import json
import re
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

REPO = "Exploration-Lab/IL-TUR"
BASE = f"datasets/{REPO}/lsi/"

SPLITS = {
    "train": "train-00000-of-00001.parquet",
    "dev": "dev-00000-of-00001.parquet",
    "test": "test-00000-of-00001.parquet",
}
STATUTES = "statutes-00000-of-00001.parquet"

N_LABELS = 100

fs = HfFileSystem()


def open_pf(name):
    return pq.ParquetFile(fs.open(BASE + name, "rb"))


def token_name(section_id):
    """'Section 120B' -> '<ipc_120B>'.

    Suffixed sections are real -- 120B, 498A, 366A -- so this cannot be numeric.
    """
    return "<ipc_{}>".format(re.sub(r"^Section\s+", "", section_id).strip())


def load_statutes():
    """The 100 target sections, in the row order the integer labels index."""
    rows = open_pf(STATUTES).read().to_pylist()
    out = []
    for r in rows:
        text = r["text"]
        out.append({
            "id": r["id"],
            "text": text if isinstance(text, list) else [text],
        })
    return out


def check(statutes, max_seen):
    """Fail loudly if the mapping's assumptions stop holding upstream."""
    n = len(statutes)
    if n != N_LABELS:
        raise SystemExit(f"expected {N_LABELS} sections, got {n} -- mapping unsafe")
    bad = [s["id"] for s in statutes if not s["id"].startswith("Section ")]
    if bad:
        raise SystemExit(f"section ids not in 'Section N' form: {bad[:5]}")
    if max_seen >= n:
        raise SystemExit(f"label index {max_seen} out of range for {n} sections")


def convert(name, filename, statutes, out_dir):
    """Stream one split to JSONL, resolving integer labels to section ids."""
    pf = open_pf(filename)
    ids = [s["id"] for s in statutes]
    path = out_dir / f"{name}.jsonl"
    rows = mentions = max_seen = 0

    with path.open("w", encoding="utf-8") as out:
        for g in range(pf.metadata.num_row_groups):
            for r in pf.read_row_group(g).to_pylist():
                raw = list(r["labels"] or [])
                if raw:
                    max_seen = max(max_seen, max(raw))
                # Ascending by section number is the fine-tuning target order,
                # and statutes is already sorted that way -- so sort on the index.
                raw.sort()
                text = r["text"]
                out.write(json.dumps({
                    "id": r["id"],
                    "text": text if isinstance(text, list) else [text],
                    "labels": [ids[i] for i in raw],
                    "label_ids": raw,
                }, ensure_ascii=False) + "\n")
                rows += 1
                mentions += len(raw)
            print(f"   {name:6s} {g+1}/{pf.metadata.num_row_groups} groups, "
                  f"{rows:,} rows", end="\r")

    print(f"   {name:6s} {rows:7,} rows, {mentions:8,} label mentions, "
          f"{mentions/max(rows,1):.2f}/doc  -> {path}")
    return rows, max_seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/ilsi"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"Statutes ({REPO}):")
    statutes = load_statutes()
    check(statutes, 0)
    print(f"   {len(statutes)} sections, {statutes[0]['id']} .. {statutes[-1]['id']}")

    # Named sec_src so inspect_ilsi.py's key-alignment check finds it.
    sec_path = args.out / "sec_src.jsonl"
    with sec_path.open("w", encoding="utf-8") as f:
        for s in statutes:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"   -> {sec_path}")

    print("\nSplits:")
    total = 0
    max_seen = 0
    for name, filename in SPLITS.items():
        n, m = convert(name, filename, statutes, args.out)
        total += n
        max_seen = max(max_seen, m)

    check(statutes, max_seen)

    # The tokenizer needs the special tokens; evaluation needs the reverse map.
    map_path = args.out / "label_map.json"
    with map_path.open("w", encoding="utf-8") as f:
        json.dump({
            "source": f"{REPO}/lsi",
            "note": "index -> section id, by statutes row order (ascending by section)",
            "sections": [s["id"] for s in statutes],
            "special_tokens": [token_name(s["id"]) for s in statutes],
        }, f, indent=2, ensure_ascii=False)

    print(f"\n=== {args.out} ===")
    print(f"   {total:,} instances across {len(SPLITS)} splits")
    print(f"   {len(statutes)} sections -> {map_path}")
    print(f"   special tokens: {token_name(statutes[0]['id'])} .. "
          f"{token_name(statutes[-1]['id'])}")


if __name__ == "__main__":
    main()
