#!/usr/bin/env python
"""Restore the `labels` field dropped by build_mixed_ft.py / _v3.py.

The reframing builders carried only id / kind / input_ids / target_ids into
the v2 and v3 splits. evaluate.py reads r["labels"] as the gold label set, so
every eval against a reframed test split dies with KeyError: 'labels'.

This copies `labels` back from data/ilsi_ft_v1 by matching on `id`. Only
test.jsonl and dev.jsonl matter -- ILSIFinetuneDataset never reads `labels`,
so train.jsonl is patched only where an id matches (instruction and
continuation rows have no gold labels and are left alone).

Idempotent: safe to run twice.

Usage:
    python src/patch_labels.py
"""
import json
from pathlib import Path

V1 = Path("data/ilsi_ft_v1")
TARGETS = ["ilsi_ft_v2", "ilsi_ft_v3"]
SPLITS = ["test", "dev", "train"]


def gold_map(split):
    path = V1 / f"{split}.jsonl"
    if not path.exists():
        return None
    out = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if "labels" in r:
                out[r["id"]] = r["labels"]
    return out


def main():
    if not V1.exists():
        raise SystemExit(f"{V1} not found")

    caches = {s: gold_map(s) for s in SPLITS}
    for s, m in caches.items():
        print(f"  v1 {s}: {len(m) if m else 0:,} rows with labels")

    for target in TARGETS:
        base = Path("data") / target
        if not base.exists():
            print(f"\n[skip] {base} missing")
            continue
        print(f"\n{target}")
        for split in SPLITS:
            path = base / f"{split}.jsonl"
            gold = caches.get(split)
            if not path.exists() or not gold:
                continue
            rows = [json.loads(l) for l in path.open(encoding="utf-8")]
            n = miss = 0
            for r in rows:
                if r.get("labels") is not None:
                    continue
                lab = gold.get(r["id"])
                if lab is not None:
                    r["labels"] = lab
                    n += 1
                elif r.get("kind", "ipc") == "ipc":
                    miss += 1
            with path.open("w", encoding="utf-8") as fh:
                for r in rows:
                    fh.write(json.dumps(r) + "\n")
            have = sum(1 for r in rows if r.get("labels") is not None)
            flag = f"  [{miss} ipc rows still unmatched]" if miss else ""
            print(f"  {split:<6} {len(rows):>7,} rows | patched {n:>7,} | "
                  f"now have labels {have:,}{flag}")


if __name__ == "__main__":
    main()
