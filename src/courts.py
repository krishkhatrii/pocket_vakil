"""
Where each court sits in corpus.jsonl, and the Decision 2 caps.

Shared by train_tokenizer.py and pack.py deliberately. Both have to agree on the
court boundaries exactly -- the tokenizer is fit to the capped mix and the pack
applies the same caps -- and a correctness-critical table duplicated across two
files is a table that eventually disagrees with itself.

Courts cannot be identified from the `court` column: it is inconsistent (Bombay
appears as three different strings) and up to 2.7% null. They are identified by
position instead. build_corpus.py writes one file at a time in COURTS order, so
each court owns a contiguous line range, and the per-court document counts come
from the extraction summary in logs/extract.log.

Verified 2026-09-06: the ranges sum to 2,977,598 lines, which is the actual line
count of corpus.jsonl, and case_id prefixes agree with the ranges throughout
(SC numeric, UPHC, HCBM, WBCHC*, DLHC, MPHC, HCMA/HCMD). `check_line_count` and
`PREFIXES` keep both of those honest at runtime.
"""

# (name, documents, measured chars, case_id prefix) in file write order
COURTS = [
    ("supreme_court",  31_869, 1_037_400_000, ""),
    ("allahabad",     524_294, 1_456_300_000, "UPHC"),
    ("bombay",      1_186_639, 4_578_000_000, "HCBM"),
    ("calcutta",        9_285,    20_600_000, "WBCHC"),
    ("delhi",         223_903, 1_851_800_000, "DLHC"),
    ("madhya_pradesh", 323_322, 1_184_700_000, "MPHC"),
    ("madras",        678_286, 3_436_800_000, "HCM"),
]

TOTAL_DOCS = sum(c[1] for c in COURTS)  # 2,977,598

# Decision 2: cap the two largest, take the other five whole.
CAPPED = {"bombay": 351_500_000, "madras": 351_500_000}
TARGET_TOKENS = 2_000_000_000


def ranges():
    """[(name, start_line, end_line, chars, prefix)], end exclusive."""
    out, s = [], 0
    for name, docs, chars, prefix in COURTS:
        out.append((name, s, s + docs, chars, prefix))
        s += docs
    return out


def court_of(line_index):
    for name, a, b, _, _ in ranges():
        if a <= line_index < b:
            return name
    raise IndexError(f"line {line_index} past end of corpus ({TOTAL_DOCS})")


def check_line_count(n):
    if n != TOTAL_DOCS:
        raise SystemExit(
            f"corpus.jsonl has {n:,} lines, court table expects {TOTAL_DOCS:,}. "
            "The table is derived from a specific extraction -- re-derive it from "
            "logs/extract.log before packing anything.")


def capped_chars(chars_per_token):
    """Per-court character budget under the caps, for proportional sampling.

    The caps are token budgets; sampling text needs them in characters, so they
    are converted with the tokenizer's own measured ratio rather than a constant.
    """
    out = {}
    for name, _, chars, _ in COURTS:
        cap = CAPPED.get(name)
        out[name] = min(chars, cap * chars_per_token) if cap else chars
    return out
