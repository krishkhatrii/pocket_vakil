"""
Pack the corpus into uint16 token shards for the GPU.

Runs on the laptop. Training memory-maps a shard and reads slices -- no parsing and
no tokenizing in the hot loop, and ~4 GB of shards re-uploads between the two GPU
sessions far faster than 14.34 GB of JSONL.

Applies the Decision 2 caps here rather than at extraction, so the mix can change
without re-extracting: Bombay and Madras are truncated to their token budgets, the
other five courts are taken whole. Courts come from `courts.py` (position in the
file, not the unreliable `court` column).

Documents are concatenated with EOS between them and cut into fixed 2048-token
sequences. A document is not padded to a boundary -- that would waste a large
fraction of every window given the length distribution -- so sequences do span
document boundaries, and EOS is the only signal marking where one judgment ends.

uint16 is safe by assertion, not by assumption: 32,000 + 104 added tokens is far
under 65,535, but `check_dtype` fails loudly rather than silently wrapping if the
vocabulary ever grows.

Writes shards plus a manifest recording per-court token counts, the shard list with
row counts, the tokenizer hash and the total. The hash is what ties a training run
to the exact vocabulary that produced its ids.

Usage:
    python src/pack.py
    python src/pack.py --limit-docs 20000     # smoke test
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

import courts

CORPUS = Path("data/corpus.jsonl")
TOKENIZER = Path("data/tokenizer/bpe_32000_final.json")
OUTDIR = Path("data/shards")

SEQ_LEN = 2048
EOS = "<|endoftext|>"
SHARD_SEQS = 20_000          # 20k x 2048 x 2B = ~82 MB per shard
DTYPE = np.uint16


def check_dtype(vocab_size):
    if vocab_size > np.iinfo(DTYPE).max:
        raise SystemExit(
            f"vocab {vocab_size:,} exceeds {DTYPE.__name__} max "
            f"{np.iinfo(DTYPE).max:,} -- ids would wrap silently. Widen the dtype.")


class ShardWriter:
    """Fixed-length sequences to disk, one file per SHARD_SEQS.

    Writes whole sequences only and closes each shard before recording it, so a
    shard listed in the manifest is a shard that is complete on disk. An
    interrupted run leaves the tail shard unlisted rather than half-written and
    indistinguishable from a good one.
    """

    def __init__(self, outdir, seq_len, shard_seqs):
        self.outdir, self.seq_len, self.shard_seqs = outdir, seq_len, shard_seqs
        self.shards, self.n_seqs, self.idx = [], 0, 0
        # Preallocated, not a list of lists: 20,000 x 2048 Python ints is ~1.1 GB
        # of boxed integers, against 82 MB here.
        self.buf = np.empty((shard_seqs, seq_len), dtype=DTYPE)
        self.fill = 0

    def add(self, seq):
        self.buf[self.fill] = seq
        self.fill += 1
        self.n_seqs += 1
        if self.fill >= self.shard_seqs:
            self.flush()

    def flush(self):
        if not self.fill:
            return
        arr = self.buf[:self.fill]
        path = self.outdir / f"shard_{self.idx:04d}.npy"
        np.save(path, arr)
        size = path.stat().st_size
        self.shards.append({"file": path.name, "sequences": int(arr.shape[0]),
                            "tokens": int(arr.size), "bytes": size,
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()[:16]})
        self.idx += 1
        self.fill = 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-len", type=int, default=SEQ_LEN)
    ap.add_argument("--shard-seqs", type=int, default=SHARD_SEQS)
    ap.add_argument("--limit-docs", type=int, default=None, help="smoke test")
    ap.add_argument("--batch", type=int, default=200)
    args = ap.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    for old in OUTDIR.glob("shard_*.npy"):
        old.unlink()

    tok = Tokenizer.from_file(str(TOKENIZER))
    tok_hash = hashlib.sha256(TOKENIZER.read_bytes()).hexdigest()
    vocab = tok.get_vocab_size()
    check_dtype(vocab)
    eos_id = tok.token_to_id(EOS)
    if eos_id is None:
        raise SystemExit("EOS not in tokenizer vocabulary")
    print(f"tokenizer {TOKENIZER.name}  vocab {vocab:,}  eos_id {eos_id}")
    print(f"sha256 {tok_hash}")

    caps = dict(courts.CAPPED)
    per_court = {name: 0 for name, *_ in courts.COURTS}
    dropped = {name: 0 for name, *_ in courts.COURTS}

    writer = ShardWriter(OUTDIR, args.seq_len, args.shard_seqs)
    carry = []
    t0 = time.time()
    n_lines = 0
    batch, batch_court = [], []

    def consume(texts, names):
        nonlocal carry
        for enc, name in zip(tok.encode_batch(texts, add_special_tokens=False), names):
            ids = enc.ids + [eos_id]
            per_court[name] += len(ids)
            carry.extend(ids)
        while len(carry) >= args.seq_len:
            writer.add(carry[:args.seq_len])
            del carry[:args.seq_len]

    with CORPUS.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            n_lines += 1
            if args.limit_docs and i >= args.limit_docs:
                break
            name = courts.court_of(i)
            cap = caps.get(name)
            if cap is not None and per_court[name] >= cap:
                dropped[name] += 1
                continue
            batch.append(json.loads(line)["text"])
            batch_court.append(name)
            if len(batch) >= args.batch:
                consume(batch, batch_court)
                batch, batch_court = [], []
                if writer.n_seqs and writer.n_seqs % 100_000 < args.batch:
                    tot = sum(per_court.values())
                    print(f"   [{time.strftime('%H:%M:%S')}] line {i:,}  "
                          f"{tot/1e6:,.0f}M tokens  {writer.n_seqs:,} seqs  "
                          f"{writer.idx} shards", flush=True)
    if batch:
        consume(batch, batch_court)
    writer.flush()

    if not args.limit_docs:
        courts.check_line_count(n_lines)

    total = sum(per_court.values())
    packed = sum(s["tokens"] for s in writer.shards)
    manifest = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tokenizer": {"file": TOKENIZER.name, "sha256": tok_hash,
                      "vocab_size": vocab, "eos_id": eos_id},
        "seq_len": args.seq_len, "dtype": str(np.dtype(DTYPE)),
        "caps": courts.CAPPED, "target_tokens": courts.TARGET_TOKENS,
        "tokens_per_court": per_court,
        "docs_dropped_by_cap": dropped,
        "total_tokens_tokenized": total,
        "total_tokens_packed": packed,
        "remainder_tokens_discarded": total - packed,
        "sequences": writer.n_seqs,
        "shards": writer.shards,
        "wall_seconds": round(time.time() - t0, 1),
    }
    (OUTDIR / "manifest.json").write_text(json.dumps(manifest, indent=2),
                                          encoding="utf-8")

    print(f"\n=== {OUTDIR} ===")
    for name, *_ in courts.COURTS:
        n = per_court[name]
        flag = " (CAPPED)" if name in caps else ""
        print(f"   {name:16s} {n/1e6:9.1f}M tokens  {100*n/max(total,1):5.1f}%"
              f"  dropped {dropped[name]:,} docs{flag}")
    print(f"   {'TOTAL':16s} {total/1e6:9.1f}M tokens")
    print(f"   packed {packed/1e6:.1f}M in {writer.n_seqs:,} sequences, "
          f"{writer.idx} shards ({sum(s['bytes'] for s in writer.shards)/1e9:.2f} GB)")
    print(f"   discarded remainder: {total-packed} tokens (< one sequence)")
    print(f"   -> {OUTDIR/'manifest.json'}   [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
