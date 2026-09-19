#!/usr/bin/env python
"""L1 artifact-validity smoke test.

Runs generic non-legal prompts through a checkpoint at batch size 1 (zero
padding, so the left-padding bug cannot contaminate the result) and prints
RAW TOKEN IDS BEFORE DECODING. The decoded string is what will lie to you:
<ipc_*> special tokens can render as empty strings, so a model emitting only
label tokens looks identical to a model emitting nothing.

Usage:
    PYTHONPATH=src python src/smoke_test.py \
        --ckpt checkpoints/final_model.pt \
        --config prod_300m --device cuda --max-new 24

Verdict lines are printed per prompt:
    EMPTY        -> first generated token is EOS. This is a hard FAIL under
                    Level 1 artifact validity.
    LABELS_ONLY  -> generated only <ipc_*> ids. Decodes to something that may
                    look empty to an external harness. Treat as a FAIL risk.
    TEXT         -> produced ordinary vocabulary tokens. Passes.
"""
import argparse
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer

from model import CONFIGS, build_model, EOS_ID

TOKENIZER = Path("data/tokenizer/bpe_32000_final.json")
LABEL_MAP = Path("data/ilsi_ft/label_map.json")

PROMPTS = [
    "The capital of France is",
    "Write two sentences about rain.",
    "List three colours:",
    "Question: What is 2 + 2?\nAnswer:",
    "Once upon a time, in a small village,",
]


def load_label_ids(tok):
    """Return the set of <ipc_*> token ids, best-effort."""
    ids = set()
    if LABEL_MAP.exists():
        try:
            meta = json.loads(LABEL_MAP.read_text(encoding="utf-8"))
            for token in (meta if isinstance(meta, list) else meta.keys()):
                tid = tok.token_to_id(token) if isinstance(token, str) else None
                if tid is not None:
                    ids.add(tid)
        except Exception as exc:
            print(f"[warn] label_map parse failed ({exc}); falling back to vocab scan")
    if not ids:
        vocab = tok.get_vocab()
        ids = {tid for tok_str, tid in vocab.items() if tok_str.startswith("<ipc_")}
    return ids


def generate_one(model, ids, max_new, device, ctx):
    """Greedy decode, batch size 1, no padding, no re-slice games."""
    x = torch.tensor([ids[-ctx:]], dtype=torch.long, device=device)
    out = []
    for _ in range(max_new):
        logits = model(x[:, -ctx:])
        nxt = int(logits[:, -1, :].float().argmax(-1)[0])
        out.append(nxt)
        if nxt == EOS_ID:
            break
        x = torch.cat([x, torch.tensor([[nxt]], device=device)], dim=1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default="prod_300m", choices=list(CONFIGS))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-new", type=int, default=24)
    args = ap.parse_args()

    cfg = CONFIGS[args.config]
    tok = Tokenizer.from_file(str(TOKENIZER))
    label_ids = load_label_ids(tok)
    print(f"tokenizer vocab {tok.get_vocab_size():,} | "
          f"{len(label_ids)} <ipc_*> ids | EOS_ID {EOS_ID}")

    model = build_model(cfg).to(args.device)
    ck = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    state = ck.get("model_state_dict", ck) if isinstance(ck, dict) else ck
    model.load_state_dict(state)
    model.eval()
    step = ck.get("step") if isinstance(ck, dict) else "?"
    print(f"loaded {args.ckpt} (step {step})\n")

    verdicts = []
    with torch.no_grad():
        for prompt in PROMPTS:
            ids = tok.encode(prompt).ids
            gen = generate_one(model, ids, args.max_new, args.device,
                               cfg.max_seq_len)

            body = [t for t in gen if t != EOS_ID]
            if not gen or gen[0] == EOS_ID:
                verdict = "EMPTY"
            elif body and all(t in label_ids for t in body):
                verdict = "LABELS_ONLY"
            else:
                verdict = "TEXT"
            verdicts.append(verdict)

            print(f"--- {verdict}")
            print(f"    prompt      : {prompt!r}")
            print(f"    prompt ids  : {ids[:12]}{' ...' if len(ids) > 12 else ''}")
            print(f"    raw gen ids : {gen}")
            n_lab = sum(1 for t in gen if t in label_ids)
            print(f"    n_gen {len(gen)} | eos_first {gen[:1] == [EOS_ID]} | "
                  f"label_tokens {n_lab}")
            try:
                print(f"    decoded     : {tok.decode(gen)!r}")
            except Exception as exc:
                print(f"    decoded     : <decode raised {exc}>")
            print()

    print("=" * 60)
    for prompt, verdict in zip(PROMPTS, verdicts):
        print(f"  {verdict:<12} {prompt[:44]}")
    hard = sum(v == "EMPTY" for v in verdicts)
    risk = sum(v == "LABELS_ONLY" for v in verdicts)
    print(f"\nEMPTY {hard}/{len(verdicts)}  LABELS_ONLY {risk}/{len(verdicts)}")
    if hard:
        print("=> HARD FAIL RISK under L1 artifact validity.")
    elif risk:
        print("=> FAIL RISK: output may decode to empty in an external harness.")
    else:
        print("=> Passes a batch-1 artifact-validity check.")


if __name__ == "__main__":
    main()
