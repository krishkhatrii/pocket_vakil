"""
TF-IDF + one-vs-rest logistic regression on ILSI. The number the SLM must beat.

Cheap, CPU-only, and deliberately run before any GPU time: a from-scratch 300M
model that cannot beat bag-of-words is a result worth having in week four rather
than week six.

Mirrors the real task -- multi-label over the same 100 IPC sections, scored on the
same test split -- so the comparison is like-for-like. It does not mirror the
model's *interface*: there is no generation and so no parse-failure mode, and no
2048-token context, so the 24.3% of facts that overflow the SLM's window are
whole here. The baseline is therefore mildly advantaged; note that when comparing.

The decision threshold is tuned on dev and never on test.

Usage:
    python src/baseline_tfidf.py
    python src/baseline_tfidf.py --max-features 100000 -C 4.0
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.multiclass import OneVsRestClassifier

ILSI = Path("data/ilsi")
OUT = Path("runs/baseline_tfidf.json")


def load(split):
    """ILSI `text` is a list of sentences; join before vectorising."""
    rows = [json.loads(l) for l in (ILSI / f"{split}.jsonl").open(encoding="utf-8")]
    texts = [" ".join(r["text"]) for r in rows]
    labels = [r["labels"] for r in rows]
    return texts, labels


def binarise(labels, sections):
    """Multi-hot over the fixed 100-section vocabulary from label_map.json.

    Built off label_map rather than off whatever the split happens to contain, so
    column j means the same section everywhere and the model always has 100
    outputs even if a split were to miss one.

    Deduplicated per row, and not defensively: 4 ILSI documents repeat a label in
    their list (train 3, dev 1 -- e.g. [61, 61, 62, 62]). csr_matrix *sums*
    duplicate coordinates, so without the set those rows hold 2s, the target reads
    as multiclass-multioutput instead of a binary indicator, and OneVsRest refuses
    it outright. Anything building the generative target sequence needs the same
    dedup, or it will teach the model to emit a label token twice.
    """
    index = {s: j for j, s in enumerate(sections)}
    rows, cols = [], []
    for i, labs in enumerate(labels):
        for s in sorted(set(labs), key=lambda x: index[x]):
            rows.append(i)
            cols.append(index[s])
    data = np.ones(len(rows), dtype=np.int8)
    m = csr_matrix((data, (rows, cols)), shape=(len(labels), len(sections)))
    assert m.max() <= 1, "binary indicator expected"
    return m


def tune_threshold(y_dev, prob_dev):
    """Sweep one global threshold on dev. Test is never touched here."""
    best = (0.0, 0.5)
    for t in np.arange(0.05, 0.96, 0.01):
        f1 = f1_score(y_dev, (prob_dev >= t).astype(np.int8),
                      average="micro", zero_division=0)
        if f1 > best[0]:
            best = (f1, float(t))
    return best[1], best[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-features", type=int, default=200_000)
    ap.add_argument("--min-df", type=int, default=2)
    ap.add_argument("-C", type=float, default=1.0)
    args = ap.parse_args()

    t0 = time.time()
    sections = json.loads((ILSI / "label_map.json").read_text(encoding="utf-8"))["sections"]
    print(f"{len(sections)} label sections from label_map.json")

    tr_x, tr_y = load("train")
    dv_x, dv_y = load("dev")
    te_x, te_y = load("test")
    print(f"loaded  train {len(tr_x):,}  dev {len(dv_x):,}  test {len(te_x):,}"
          f"   [{time.time()-t0:.0f}s]")

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=args.min_df,
                          max_features=args.max_features, sublinear_tf=True,
                          strip_accents="unicode", lowercase=True)
    Xtr = vec.fit_transform(tr_x)
    Xdv = vec.transform(dv_x)
    Xte = vec.transform(te_x)
    print(f"tfidf   {Xtr.shape[0]:,} x {Xtr.shape[1]:,}   "
          f"nnz/doc {Xtr.nnz/Xtr.shape[0]:.0f}   [{time.time()-t0:.0f}s]")

    Ytr = binarise(tr_y, sections)
    Ydv = binarise(dv_y, sections).toarray()
    Yte = binarise(te_y, sections).toarray()

    # dual=True: n_features (200k) >> n_samples (43k), the regime liblinear's
    # dual solver is built for.
    clf = OneVsRestClassifier(
        LogisticRegression(solver="liblinear", dual=True, C=args.C, max_iter=2000),
        n_jobs=-1,
    )
    clf.fit(Xtr, Ytr.toarray())
    print(f"fit     100 binary classifiers   [{time.time()-t0:.0f}s]")

    Pdv = clf.predict_proba(Xdv)
    thr, dev_micro = tune_threshold(Ydv, Pdv)
    print(f"thresh  {thr:.2f} (dev micro-F1 {dev_micro:.4f})   [{time.time()-t0:.0f}s]")

    Pte = clf.predict_proba(Xte)
    pred = (Pte >= thr).astype(np.int8)

    micro = f1_score(Yte, pred, average="micro", zero_division=0)
    macro = f1_score(Yte, pred, average="macro", zero_division=0)
    per = f1_score(Yte, pred, average=None, zero_division=0)
    zero = [sections[j] for j, f in enumerate(per) if f == 0.0]
    empty = int((pred.sum(axis=1) == 0).sum())

    order = np.argsort(per)
    result = {
        "model": "tfidf(1-2gram) + OvR logistic regression",
        "config": {
            "max_features": args.max_features, "min_df": args.min_df,
            "sublinear_tf": True, "C": args.C, "solver": "liblinear(dual)",
            "vocab_actual": int(Xtr.shape[1]),
        },
        "threshold": thr,
        "threshold_tuned_on": "dev",
        "dev_micro_f1": float(dev_micro),
        "test": {
            "micro_f1": float(micro),
            "macro_f1": float(macro),
            "micro_precision": float(precision_score(Yte, pred, average="micro", zero_division=0)),
            "micro_recall": float(recall_score(Yte, pred, average="micro", zero_division=0)),
            "labels_with_zero_f1": len(zero),
            "zero_f1_sections": zero,
            "empty_predictions": empty,
            "empty_prediction_rate": empty / len(te_x),
        },
        "per_label_f1": {sections[j]: float(per[j]) for j in range(len(sections))},
        "worst_10": [(sections[j], float(per[j])) for j in order[:10]],
        "best_10": [(sections[j], float(per[j])) for j in order[-10:][::-1]],
        "n_train": len(tr_x), "n_dev": len(dv_x), "n_test": len(te_x),
        "wall_seconds": round(time.time() - t0, 1),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"\n=== ILSI test ({len(te_x):,} docs) ===")
    print(f"   micro-F1 {micro:.4f}")
    print(f"   macro-F1 {macro:.4f}")
    print(f"   labels with F1 = 0.0: {len(zero)}/100")
    print(f"   empty predictions: {empty:,} ({100*empty/len(te_x):.2f}%)")
    print(f"   -> {OUT}   [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
