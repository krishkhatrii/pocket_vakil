"""
Which end of a long ILSI fact do we cut?

Context is locked at 2048 by the compute envelope and ~24% of train facts exceed
it, so truncation is not optional -- only the choice of end is. Head-only is the
default everyone reaches for and the one most likely to be quietly wrong here:
these are judgment-derived fact sections, and the conduct that determines the IPC
section is not reliably in the opening.

Probe is the TF-IDF baseline, retrained per strategy. It is not the SLM, but it is
the cheap instrument that already has a number to compare against (untruncated
micro 0.4941 / macro 0.2858, `runs/baseline_tfidf.json`), and it reads lexical
evidence -- which is what truncation destroys or preserves.

Two measurements per strategy:
  1. micro / macro F1 on the correspondingly-truncated test split. Train, dev and
     test are all truncated the same way -- truncating test alone would measure a
     train/test mismatch instead of information loss.
  2. Feature survival: taking the untruncated model's top-weighted features per
     label, how often does a long document keep none of them for its own labels.
     F1 says how much is lost; this says whether the loss is the label evidence.

Tokenisation is GPT-2 50k BPE, the same stand-in every token figure in
FINDINGS.md uses (Decision 4 -- re-measure with the project BPE before packing).

Usage:
    python src/truncation_test.py
    python src/truncation_test.py --context 2048 --top-k 50
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import tiktoken
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier

ILSI = Path("data/ilsi")
OUT = Path("runs/truncation_test.json")

ENC = tiktoken.get_encoding("gpt2")


def load(split):
    rows = [json.loads(l) for l in (ILSI / f"{split}.jsonl").open(encoding="utf-8")]
    return [" ".join(r["text"]) for r in rows], [r["labels"] for r in rows]


def binarise(labels, sections):
    """Dedup per row: 4 ILSI docs repeat a label, and csr sums duplicates."""
    index = {s: j for j, s in enumerate(sections)}
    rows, cols = [], []
    for i, labs in enumerate(labels):
        for s in set(labs):
            rows.append(i)
            cols.append(index[s])
    m = csr_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)),
                   shape=(len(labels), len(sections)))
    assert m.max() <= 1
    return m


def encode_long(texts, context):
    """Token ids for documents that might exceed the context; None for the rest.

    Anything under 2*context characters would need <2 chars/token to overflow,
    which English does not do -- so those skip the tokenizer entirely. That is
    ~60% of ILSI at p50 2,700 chars, and tokenising all of it is the slow step.
    """
    ids = [None] * len(texts)
    cand = [i for i, t in enumerate(texts) if len(t) >= 2 * context]
    for chunk in range(0, len(cand), 1000):
        idx = cand[chunk:chunk + 1000]
        for i, toks in zip(idx, ENC.encode_ordinary_batch([texts[i] for i in idx])):
            ids[i] = toks if len(toks) > context else None
    return ids


def truncate(texts, ids, strategy, context):
    """Apply one strategy. Documents inside the context are returned untouched."""
    out = []
    for t, toks in zip(texts, ids):
        if toks is None:
            out.append(t)
        elif strategy == "head":
            out.append(ENC.decode(toks[:context]))
        elif strategy == "tail":
            out.append(ENC.decode(toks[-context:]))
        elif strategy == "head_tail":
            h = context // 2
            out.append(ENC.decode(toks[:h]) + "\n" + ENC.decode(toks[-h:]))
        else:
            raise ValueError(strategy)
    return out


def fit_score(tr_x, Ytr, dv_x, Ydv, te_x, Yte, args):
    """Full pipeline on one text variant. Threshold tuned on dev, never test."""
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=args.min_df,
                          max_features=args.max_features, sublinear_tf=True,
                          strip_accents="unicode", lowercase=True)
    Xtr = vec.fit_transform(tr_x)
    clf = OneVsRestClassifier(
        LogisticRegression(solver="liblinear", dual=True, C=args.C, max_iter=2000),
        n_jobs=-1,
    )
    clf.fit(Xtr, Ytr.toarray())

    Pdv = clf.predict_proba(vec.transform(dv_x))
    best = (0.0, 0.5)
    for t in np.arange(0.05, 0.96, 0.01):
        f = f1_score(Ydv, (Pdv >= t).astype(np.int8), average="micro", zero_division=0)
        if f > best[0]:
            best = (f, float(t))
    thr = best[1]

    pred = (clf.predict_proba(vec.transform(te_x)) >= thr).astype(np.int8)
    return {
        "threshold": thr,
        "dev_micro_f1": float(best[0]),
        "micro_f1": float(f1_score(Yte, pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(Yte, pred, average="macro", zero_division=0)),
        "zero_f1_labels": int((f1_score(Yte, pred, average=None, zero_division=0) == 0).sum()),
        "empty_prediction_rate": float((pred.sum(axis=1) == 0).mean()),
        "vocab": int(Xtr.shape[1]),
    }, vec, clf


def survival(vec, clf, sections, texts, ids, labels, context, top_k):
    """Do a long document's own labels keep any of their top-weighted features?

    Signal, not just score: a strategy can hold F1 up by keeping generic legal
    boilerplate while dropping the specific evidence. This looks only at long
    documents -- short ones are untouched by every strategy.
    """
    feats = np.array(vec.get_feature_names_out())
    top = {}
    for j, est in enumerate(clf.estimators_):
        w = est.coef_[0]
        top[j] = set(feats[np.argsort(w)[-top_k:]])
    index = {s: j for j, s in enumerate(sections)}

    stats = {}
    for name in ("head", "tail", "head_tail"):
        trunc = truncate(texts, ids, name, context)
        lost_pairs = pairs = lost_docs = docs = 0
        for i, toks in enumerate(ids):
            if toks is None:
                continue
            docs += 1
            present = set(vec.build_analyzer()(trunc[i]))
            doc_all_lost = True
            for s in set(labels[i]):
                pairs += 1
                if top[index[s]] & present:
                    doc_all_lost = False
                else:
                    lost_pairs += 1
            lost_docs += doc_all_lost
        stats[name] = {
            "long_docs": docs,
            "label_pairs_losing_all_top_features": lost_pairs,
            "label_pairs": pairs,
            "pair_loss_rate": lost_pairs / max(pairs, 1),
            "docs_losing_all_top_features_for_every_label": lost_docs,
            "doc_loss_rate": lost_docs / max(docs, 1),
        }
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--context", type=int, default=2048)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--max-features", type=int, default=200_000)
    ap.add_argument("--min-df", type=int, default=2)
    ap.add_argument("-C", type=float, default=1.0)
    args = ap.parse_args()

    t0 = time.time()
    sections = json.loads((ILSI / "label_map.json").read_text(encoding="utf-8"))["sections"]
    tr_x, tr_y = load("train")
    dv_x, dv_y = load("dev")
    te_x, te_y = load("test")
    Ytr, Ydv, Yte = (binarise(tr_y, sections), binarise(dv_y, sections).toarray(),
                     binarise(te_y, sections).toarray())

    print("tokenising (GPT-2 50k BPE)...", flush=True)
    tr_i = encode_long(tr_x, args.context)
    dv_i = encode_long(dv_x, args.context)
    te_i = encode_long(te_x, args.context)
    n_long = sum(x is not None for x in tr_i)
    print(f"  train over {args.context} tokens: {n_long:,}/{len(tr_x):,} "
          f"({100*n_long/len(tr_x):.1f}%)   dev {sum(x is not None for x in dv_i):,}"
          f"   test {sum(x is not None for x in te_i):,}   [{time.time()-t0:.0f}s]",
          flush=True)

    results = {}
    print("\nbaseline: untruncated", flush=True)
    base, vec0, clf0 = fit_score(tr_x, Ytr, dv_x, Ydv, te_x, Yte, args)
    results["untruncated"] = base
    print(f"  micro {base['micro_f1']:.4f}  macro {base['macro_f1']:.4f}"
          f"   [{time.time()-t0:.0f}s]", flush=True)

    for name in ("head", "tail", "head_tail"):
        print(f"\nstrategy: {name}", flush=True)
        r, _, _ = fit_score(truncate(tr_x, tr_i, name, args.context), Ytr,
                            truncate(dv_x, dv_i, name, args.context), Ydv,
                            truncate(te_x, te_i, name, args.context), Yte, args)
        results[name] = r
        print(f"  micro {r['micro_f1']:.4f}  macro {r['macro_f1']:.4f}"
              f"   [{time.time()-t0:.0f}s]", flush=True)

    print(f"\nfeature survival on long test docs (top-{args.top_k} per label)...",
          flush=True)
    surv = survival(vec0, clf0, sections, te_x, te_i, te_y, args.context, args.top_k)
    for k, v in surv.items():
        results[k]["feature_survival"] = v

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "context": args.context, "top_k": args.top_k,
        "tokenizer": "gpt2 50k BPE (provisional, FINDINGS Decision 4)",
        "long_train_docs": n_long, "results": results,
        "wall_seconds": round(time.time() - t0, 1),
    }, indent=2), encoding="utf-8")

    print(f"\n{'strategy':14s} {'micro':>7s} {'macro':>7s} {'zeroF1':>7s} "
          f"{'docs losing all feats':>22s}")
    for k in ("untruncated", "head", "tail", "head_tail"):
        r = results[k]
        fs = r.get("feature_survival")
        loss = f"{100*fs['doc_loss_rate']:.1f}%" if fs else "-"
        print(f"{k:14s} {r['micro_f1']:7.4f} {r['macro_f1']:7.4f} "
              f"{r['zero_f1_labels']:7d} {loss:>22s}")
    print(f"\n-> {OUT}   [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
