"""
Train the project BPE. 16k and 32k, so the vocab size can be chosen on evidence.

THIS IS THE IRREVERSIBLE STEP. Token ids shift if the tokenizer is retrained, which
invalidates every packed shard and every checkpoint built on them. Nothing here
freezes a choice -- it trains both and measures them. Freezing is a separate,
deliberate act: keep one, record its vocab size in `runs/`, and never retrain.

Byte-level BPE (GPT-2 style). Byte-level is not a stylistic choice: it makes the
vocabulary closed over arbitrary bytes, so there is no <unk>, and encode->decode is
lossless on OCR wreckage like 'D~CEMBER' and 'appoi11tcd' that this corpus is full
of. `verify()` asserts that rather than trusting it.

Added tokens (the 100 IPC labels, the 3 ILSI masks, EOS) are matched literally
before pre-tokenization, which is why `<ipc_294(b)>` and `<ipc_376(2)>` survive
with their parentheses intact. That is the property most likely to break silently,
so it is checked explicitly for all 100.

Training sample: the corpus is 14.34 GB and BPE does not need all of it. Documents
are taken by stride to a byte budget, and a disjoint stride is held out for the
chars/token measurement -- measuring compression on text the vocab was fit to would
flatter it.

Usage:
    python src/train_tokenizer.py                    # 16k and 32k
    python src/train_tokenizer.py --vocab 32000      # one size
    python src/train_tokenizer.py --budget-gb 0.5    # faster trial
"""

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import courts

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers
from tokenizers import AddedToken

CORPUS = Path("data/corpus.jsonl")
ILSI = Path("data/ilsi")
OUTDIR = Path("data/tokenizer")

EOS = "<|endoftext|>"
MASKS = ["<SECTION>", "<ACT>", "<ENTITY>"]

# d_model from PLAN.md section 2 (1024, 24 layers, 16 heads -> ~302M non-embedding)
D_MODEL = 1024
NON_EMBED = 302_000_000
CONTEXT = 2048


def specials():
    m = json.loads((ILSI / "label_map.json").read_text(encoding="utf-8"))
    return m["special_tokens"], m["sections"]


def corpus_stream(budget_bytes, stride, offset):
    """Every `stride`-th document from `offset`, until the byte budget is spent.

    Stride rather than a head slice: corpus.jsonl is written court by court, so the
    first N lines are all Supreme Court. A stride samples every court in proportion.
    """
    used = 0
    with CORPUS.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i % stride != offset:
                continue
            t = json.loads(line)["text"]
            used += len(t)
            yield t
            if used >= budget_bytes:
                return


def capped_stream(budget_bytes, chars_per_token, offset=0):
    """Sample proportional to the *capped* mix, not the raw corpus.

    Uncapped, Bombay and Madras are 59% of corpus characters; under the Decision 2
    caps they are ~36% of what actually gets pretrained on. A vocabulary fit to the
    former is fit to a distribution the model never sees, over-weighting two courts'
    boilerplate. Each court gets a share of the budget proportional to its capped
    contribution, sampled by stride across that court's whole line range so the
    sample is not drawn from one era or one registry.
    """
    caps = courts.capped_chars(chars_per_token)
    total = sum(caps.values())
    want = {n: budget_bytes * c / total for n, c in caps.items()}
    used = {n: 0 for n in want}

    # Stride per court, sized so its allocation spans the court's full range.
    stride = {}
    for name, a, b, chars, _ in courts.ranges():
        docs = b - a
        mean_doc = chars / max(docs, 1)
        need_docs = max(int(want[name] / max(mean_doc, 1)), 1)
        stride[name] = max(docs // need_docs, 1)

    with CORPUS.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            name = courts.court_of(i)
            if used[name] >= want[name]:
                continue
            a = next(r[1] for r in courts.ranges() if r[0] == name)
            if (i - a) % stride[name] != offset % stride[name]:
                continue
            t = json.loads(line)["text"]
            used[name] += len(t)
            yield t
    for n in want:
        print(f"      {n:16s} {used[n]/1e6:8.1f}M chars "
              f"({100*used[n]/max(sum(used.values()),1):4.1f}%)", flush=True)


def ilsi_texts(split):
    return [" ".join(json.loads(l)["text"])
            for l in (ILSI / f"{split}.jsonl").open(encoding="utf-8")]


def sec_src_texts():
    return [" ".join(r["text"]) for r in
            (json.loads(l) for l in (ILSI / "sec_src.jsonl").open(encoding="utf-8"))]


def training_corpus(args):
    """Corpus sample + ILSI train facts + section text, per PLAN.md section 5.3."""
    if args.capped:
        yield from capped_stream(int(args.budget_gb * 1e9), args.chars_per_token)
    else:
        yield from corpus_stream(int(args.budget_gb * 1e9), args.stride, 0)
    yield from ilsi_texts("train")
    yield from sec_src_texts()


def train(vocab_size, ipc_tokens, args):
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.post_processor = processors.ByteLevel(trim_offsets=False)

    added = [AddedToken(t, single_word=False, lstrip=False, rstrip=False,
                        normalized=False, special=True)
             for t in [EOS] + MASKS + ipc_tokens]

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=added,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
        min_frequency=2,
    )
    t0 = time.time()
    tok.train_from_iterator(training_corpus(args), trainer=trainer)
    # Re-assert as added tokens: guarantees literal matching ahead of the
    # pre-tokenizer regardless of whether BPE also learned the pieces.
    tok.add_special_tokens(added)
    return tok, time.time() - t0


def token_lengths(tok, texts, batch=500):
    """Token count per text, encoded in chunks.

    encode_batch over a whole split holds every Encoding object -- ids, tokens,
    offsets, attention mask -- resident at once. On 42,750 ILSI facts that was
    enough to get the process OOM-killed silently on this 16 GB machine (exit 0,
    no traceback, output just stops). Only the lengths are needed, so keep those
    and let each batch go.
    """
    lens = []
    for i in range(0, len(texts), batch):
        lens.extend(len(e.ids)
                    for e in tok.encode_batch(texts[i:i + batch],
                                              add_special_tokens=False))
    return lens


def chars_per_token(lens, texts):
    return sum(len(t) for t in texts) / max(sum(lens), 1)


def verify(tok, ipc_tokens, ilsi_sample):
    """Every check that would be expensive to discover after packing."""
    out = {}

    single, multi = 0, []
    for t in ipc_tokens:
        ids = tok.encode(t, add_special_tokens=False).ids
        if len(ids) == 1:
            single += 1
        else:
            multi.append((t, len(ids), tok.encode(t, add_special_tokens=False).tokens))
    out["ipc_single_id"] = single
    out["ipc_split"] = multi

    paren = {}
    for t in ("<ipc_294(b)>", "<ipc_376(2)>"):
        e = tok.encode(t, add_special_tokens=False)
        paren[t] = {"n_ids": len(e.ids), "tokens": e.tokens,
                    "roundtrip_ok": tok.decode(e.ids, skip_special_tokens=False) == t}
    out["parenthesised"] = paren

    masks = {}
    for t in MASKS:
        e = tok.encode(t, add_special_tokens=False)
        masks[t] = {"n_ids": len(e.ids),
                    "roundtrip_ok": tok.decode(e.ids, skip_special_tokens=False) == t}
    out["masks"] = masks

    # In context, not just standalone -- a mask inside a sentence is the real case.
    ctx = "The accused, <ENTITY>, was charged under <SECTION> of the <ACT> in 2011."
    e = tok.encode(ctx, add_special_tokens=False)
    out["mask_in_context"] = {
        "roundtrip_ok": tok.decode(e.ids, skip_special_tokens=False) == ctx,
        "each_mask_single_id": all(
            len(tok.encode(m, add_special_tokens=False).ids) == 1 for m in MASKS),
    }

    eos_id = tok.token_to_id(EOS)
    e = tok.encode(EOS, add_special_tokens=False)
    out["eos"] = {"token": EOS, "id": eos_id, "n_ids": len(e.ids),
                  "in_vocab": eos_id is not None,
                  "roundtrip_ok": tok.decode(e.ids, skip_special_tokens=False) == EOS}

    bad = []
    for t in ilsi_sample:
        e = tok.encode(t, add_special_tokens=False)
        if tok.decode(e.ids, skip_special_tokens=False) != t:
            bad.append(t[:80])
    out["roundtrip_ilsi"] = {"n": len(ilsi_sample), "failures": len(bad),
                             "examples": bad[:3]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, nargs="*", default=[16000, 32000])
    ap.add_argument("--budget-gb", type=float, default=1.5,
                    help="corpus bytes sampled for BPE training")
    ap.add_argument("--stride", type=int, default=7,
                    help="take every Nth doc for training; offset 1 is held out")
    ap.add_argument("--heldout-docs", type=int, default=4000)
    ap.add_argument("--capped", action="store_true",
                    help="sample proportional to the Decision 2 capped mix")
    ap.add_argument("--chars-per-token", type=float, default=4.418,
                    help="measured ratio, to convert token caps to char budgets")
    ap.add_argument("--name", default=None, help="output basename")
    args = ap.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    ipc_tokens, sections = specials()
    print(f"{len(ipc_tokens)} IPC tokens + {len(MASKS)} masks + EOS")

    print(f"held-out corpus sample ({args.heldout_docs:,} docs, disjoint stride)...",
          flush=True)
    held = []
    for t in corpus_stream(10**12, args.stride, 1):
        held.append(t)
        if len(held) >= args.heldout_docs:
            break
    print(f"   {sum(len(t) for t in held)/1e6:.1f}M chars", flush=True)

    ilsi_train = ilsi_texts("train")
    rng = random.Random(11)
    ilsi_rt = rng.sample(ilsi_train, 100)

    report = {"d_model": D_MODEL, "non_embedding_params": NON_EMBED,
              "context": CONTEXT, "heldout_docs": len(held),
              "train_budget_gb": args.budget_gb, "vocabs": {}}

    for v in args.vocab:
        print(f"\n=== training vocab {v:,} ===", flush=True)
        tok, secs = train(v, ipc_tokens, args)
        path = OUTDIR / (f"{args.name}.json" if args.name else f"bpe_{v}.json")
        tok.save(str(path))
        actual = tok.get_vocab_size()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        print(f"   trained in {secs/60:.1f} min -> {path}  (vocab {actual:,})",
              flush=True)
        print(f"   sha256 {digest}", flush=True)

        print("   measuring corpus...", flush=True)
        cpt_c = chars_per_token(token_lengths(tok, held), held)
        print("   measuring ILSI...", flush=True)
        ilsi_lens = token_lengths(tok, ilsi_train)
        cpt_i = chars_per_token(ilsi_lens, ilsi_train)
        over = sum(1 for x in ilsi_lens if x > CONTEXT)
        lens = sorted(ilsi_lens)

        emb = actual * D_MODEL
        report["vocabs"][str(v)] = {
            "vocab_size": actual,
            "path": str(path),
            "sha256": digest,
            "mix": "capped (Decision 2)" if args.capped else "uncapped corpus",
            "train_seconds": round(secs, 1),
            "chars_per_token_corpus": cpt_c,
            "chars_per_token_ilsi": cpt_i,
            "embedding_params": emb,
            "embedding_params_tied_pct_of_300M": 100 * emb / (NON_EMBED + emb),
            "embedding_params_untied": 2 * emb,
            "embedding_params_untied_pct": 100 * 2 * emb / (NON_EMBED + 2 * emb),
            "ilsi_tokens_p50": lens[len(lens) // 2],
            "ilsi_tokens_p90": lens[int(len(lens) * 0.9)],
            "ilsi_over_context": over,
            "ilsi_over_context_pct": 100 * over / len(lens),
            "corpus_tokens_projected_13565M_chars": 13_565.5e6 / cpt_c / 1e6,
            "verify": verify(tok, ipc_tokens, ilsi_rt),
        }
        r = report["vocabs"][str(v)]
        print(f"   chars/token  corpus {cpt_c:.3f}   ILSI {cpt_i:.3f}")
        print(f"   embedding    {emb/1e6:.1f}M tied  ({r['embedding_params_tied_pct_of_300M']:.1f}% of budget)")
        print(f"   ILSI >2048   {over:,}/{len(lens):,} ({r['ilsi_over_context_pct']:.1f}%)")
        vf = r["verify"]
        print(f"   verify       {vf['ipc_single_id']}/100 single-id, "
              f"roundtrip {vf['roundtrip_ilsi']['n']-vf['roundtrip_ilsi']['failures']}"
              f"/{vf['roundtrip_ilsi']['n']}")

    out = OUTDIR / (f"{args.name}_report.json" if args.name else "comparison.json")
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n-> {out}")
    print("\nNOTHING FROZEN. Choose a vocab, record it in runs/, then pack.")


if __name__ == "__main__":
    main()
