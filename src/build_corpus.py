"""
Build the corpus: full English Supreme Court + High Court text.

One output file, corpus.jsonl, used for both tokenizer training and pretraining.
Each line: {"case_id", "court", "year", "title", "text", "source"}

Handles what the diagnostics found:
  - SC is 67% translations (HCs 100% en) -> language_code == "en"
  - both text fields carry a header    -> strip it
  - [TITLE] / [SECTION] markers        -> strip all bracket markers
  - law-report margin letters          -> drop single-character lines
  - chunks overlap (p50 195, p90 475)  -> trim via char_start / char_end
  - 19.5% of pairs gap instead of overlap -> left alone, source text is missing

Usage:
    python build_corpus.py                    # full extraction, all 7 courts
    python build_corpus.py --hc-groups 2      # smoke test, 2 row groups per HC
    python build_corpus.py --sc-only
    python build_corpus.py --row-groups 3     # quick trial, SC only
"""

import argparse
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

REPO = "vaquill/open-india-law"
OUT = Path("data/corpus.jsonl")

SC = "in_supreme-court_judgments.parquet"

# The six High Courts ILSI draws from
HC = [
    "in_allahabad_judgments.parquet",
    "in_bombay_judgments.parquet",
    "in_calcutta_judgments.parquet",
    "in_delhi_judgments.parquet",
    "in_madhya-pradesh_judgments.parquet",
    "in_madras_judgments.parquet",
]

COLS = ["case_id", "chunk_index", "total_chunks", "text", "char_start", "char_end",
        "language_code", "year", "title", "court"]

# "Case: FOO versus BAR [[cite]] (1979)\nSection: PARAGRAPH\n\n"
HEADER = re.compile(r"\ACase:.*?\nSection:[^\n]*\n+", re.DOTALL)
# [TITLE] # , [SECTION] ## , [HEADNOTE] etc.
MARKER = re.compile(r"\[[A-Z][A-Z_ ]*\]\s*#*\s*")
# Margin letters printed alone on a line in law reports
MARGIN = re.compile(r"^\s*[A-Za-z]\s*$", re.MULTILINE)

fs = HfFileSystem()


def clean(raw):
    t = HEADER.sub("", raw or "")
    t = MARKER.sub("", t)
    t = MARGIN.sub("", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def reassemble(chunks):
    """Sort by chunk_index and splice out overlap between consecutive chunks.

    Order matters and is not arbitrary. char_start / char_end index the source
    text with the header removed and nothing else removed (measured: they equal
    len(text) - len(header) on 2,994 of 3,000 chunks). So the trim has to run in
    that coordinate space: strip the header per chunk, trim, and apply the rest
    of the cleaning once at the end. Cleaning a chunk before trimming it shifts
    every offset left and the slice then eats real text.

    Negative overlaps are gaps -- source text missing between chunks, ~19.5% of
    pairs. Nothing can recover it; the seam gets a newline so it stays visible.
    """
    chunks = sorted(chunks, key=lambda r: r["chunk_index"])
    parts, prev_end = [], None

    for r in chunks:
        body = HEADER.sub("", r["text"] or "")
        if not body:
            continue
        sep = ""
        if prev_end is not None and r["char_start"] is not None:
            overlap = prev_end - r["char_start"]
            if 0 < overlap < len(body):
                body = body[overlap:]
            elif overlap >= len(body):
                continue
            elif overlap < 0:
                sep = "\n"
        if parts:
            parts.append(sep)
        parts.append(body)
        prev_end = r["char_end"]

    return clean("".join(parts))


def pick_groups(total, n):
    """Evenly spaced row groups, so a sample isn't skewed to one era."""
    if n is None or n >= total:
        return list(range(total))
    step = max(total // (n + 1), 1)
    return sorted({min(step * (i + 1), total - 1) for i in range(n)})


def process(filename, n_groups, year_max, source, out, min_chars):
    """Stream one file to disk.

    Holding a whole file's chunks in memory does not survive full extraction --
    Bombay alone measured ~9.4 GB. So documents are written as soon as they are
    complete and only in-flight cases stay resident.

    A case is complete when we hold total_chunks of its chunks (verified: that
    field is never null, never disagrees within a case, and is never exceeded).
    Cases do straddle row-group boundaries -- rare, but one measured case had 13
    of its 22 chunks in the previous group -- so incomplete cases are carried
    forward, never truncated.
    """
    pf = pq.ParquetFile(fs.open(f"datasets/{REPO}/{filename}", "rb"))
    groups = pick_groups(pf.metadata.num_row_groups, n_groups)

    pending = defaultdict(list)
    seen = written = chars = 0

    def emit(chunks):
        nonlocal written, chars
        text = reassemble(chunks)
        if len(text) < min_chars:
            return
        head = chunks[0]
        out.write(json.dumps({
            "case_id": head["case_id"],
            "court": head.get("court"),
            "year": head.get("year"),
            "title": head.get("title"),
            "text": text,
            "source": source,
        }, ensure_ascii=False) + "\n")
        written += 1
        chars += len(text)

    for i, g in enumerate(groups):
        for r in pf.read_row_group(g, columns=COLS).to_pylist():
            seen += 1
            if r["language_code"] != "en":
                continue
            if r["year"] and r["year"] > year_max:
                continue
            chunks = pending[r["case_id"]]
            chunks.append(r)
            if len(chunks) == r["total_chunks"]:
                emit(pending.pop(r["case_id"]))
        print(f"   [{datetime.now():%Y-%m-%d %H:%M:%S}] {filename[:34]:34s} "
              f"{i+1}/{len(groups)} groups, {seen:,} seen, {written:,} docs, "
              f"{chars/1e6:.1f}M chars, {len(pending):,} open", flush=True)

    # Whatever never completed is missing chunks in the source, not a straddler.
    # Emit what exists -- dropping it would truncate judgments silently.
    incomplete = len(pending)
    for case_id in list(pending):
        emit(pending.pop(case_id))

    print(f"   {filename[:34]:34s} {len(groups):3d} groups -> "
          f"{written:7,} docs, {chars/1e6:8.1f}M chars"
          f"{f', {incomplete:,} incomplete' if incomplete else ''}")
    return written, chars


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--row-groups", type=int, default=None,
                    help="cap SC row groups (trial runs)")
    ap.add_argument("--hc-groups", type=int, default=None,
                    help="cap row groups per High Court (smoke tests); default all")
    ap.add_argument("--sc-only", action="store_true")
    ap.add_argument("--year-max", type=int, default=2020)
    ap.add_argument("--min-chars", type=int, default=500)
    args = ap.parse_args()

    total_docs = total_chars = 0

    with OUT.open("w", encoding="utf-8") as out:
        print("Supreme Court:")
        d, c = process(SC, args.row_groups, args.year_max, "sc", out, args.min_chars)
        total_docs, total_chars = d, c

        if not args.sc_only:
            print("\nHigh Courts:")
            for f in HC:
                try:
                    d, c = process(f, args.hc_groups, args.year_max, "hc",
                                   out, args.min_chars)
                    total_docs += d
                    total_chars += c
                except Exception as e:
                    print(f"   {f[:34]:34s} FAILED: {e}")

    print(f"\n=== {OUT} ===")
    print(f"   {total_docs:,} documents")
    print(f"   {total_chars/1e6:.1f}M chars  (~{total_chars/4e6:.0f}M tokens)")
    print(f"   mean {total_chars // max(total_docs,1):,} chars/doc")

    with OUT.open(encoding="utf-8") as f:
        s = json.loads(f.readline())
    print(f"\n--- sample: {str(s['title'])[:60]} ({s['year']}) ---")
    print(s["text"][:600])


if __name__ == "__main__":
    main()
