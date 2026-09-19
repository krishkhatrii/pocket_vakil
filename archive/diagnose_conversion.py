#!/usr/bin/env python
"""Locate the exact point where the converted HF model diverges from OLM.

Strategy:

  1. Single-token input. At position 0 RoPE is the identity (cos=1, sin=0) and
     causal attention is trivial, so a mismatch here rules RoPE and masking
     OUT and points at the embedding, LayerNorm, MLP or output head.

  2. Multi-token input. If step 1 matches but this does not, the fault is in
     RoPE or attention.

  3. Per-block hidden states. Forward hooks on both models, compared block by
     block, so the first divergent layer is named rather than inferred.

Usage:
    python diagnose_conversion.py --ckpt checkpoints/final_model_v2.pt \
        --hf pocket-vakil-332m-ipc
"""
import argparse
import sys

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/final_model_v2.pt")
    ap.add_argument("--hf", default="pocket-vakil-332m-ipc")
    ap.add_argument("--src", default="src")
    args = ap.parse_args()

    sys.path.insert(0, args.src)
    from model import CONFIGS, build_model
    from transformers import AutoModelForCausalLM

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = ck.get("model_state_dict", ck) if isinstance(ck, dict) else ck

    olm = build_model(CONFIGS["prod_300m"])
    olm.load_state_dict(sd)
    olm.eval()

    hf = AutoModelForCausalLM.from_pretrained(args.hf, trust_remote_code=True)
    hf.eval()

    def compare(name, ids):
        with torch.no_grad():
            a = olm(ids).float()
            b = hf(input_ids=ids).logits.float()
        d = (a - b).abs().max().item()
        agree = (a.argmax(-1) == b.argmax(-1)).float().mean().item()
        print(f"  {name:<28} max|d| {d:10.4e}   argmax {agree*100:6.2f}%")
        return d

    print("=" * 66)
    print("STEP 1-2  logits")
    print("=" * 66)
    d1 = compare("1 token  (RoPE = identity)", torch.tensor([[596]]))
    d2 = compare("2 tokens", torch.tensor([[596, 6041]]))
    d8 = compare("8 tokens", torch.tensor([[596, 6041, 372, 605, 11564, 412, 302, 117]]))

    print("\n  reading:")
    if d1 > 1e-3:
        print("    1-token ALREADY differs -> fault is NOT RoPE and NOT masking.")
        print("    Suspect: embedding scaling, LayerNorm, SwiGLU split, output head.")
    elif d2 > 1e-3:
        print("    1 token matches, 2 do not -> fault IS in RoPE or attention.")
    else:
        print("    short inputs match; divergence grows with length.")

    # ---------------------------------------------------------------- hooks
    print("\n" + "=" * 66)
    print("STEP 3  per-block hidden states (8 tokens)")
    print("=" * 66)

    ids = torch.tensor([[596, 6041, 372, 605, 11564, 412, 302, 117]])
    olm_acts, hf_acts = {}, {}

    def grab(store, key):
        def hook(_m, _i, out):
            store[key] = (out[0] if isinstance(out, tuple) else out).detach().float()
        return hook

    handles = []
    try:
        stack = olm.blocks[1].stack
        handles.append(olm.blocks[0].register_forward_hook(grab(olm_acts, -1)))
        for i, blk in enumerate(stack):
            handles.append(blk.register_forward_hook(grab(olm_acts, i)))
    except Exception as exc:
        print(f"  [could not hook OLM: {exc}]")

    try:
        handles.append(hf.model.embed_tokens.register_forward_hook(
            grab(hf_acts, -1)))
        for i, layer in enumerate(hf.model.layers):
            handles.append(layer.register_forward_hook(grab(hf_acts, i)))
    except Exception as exc:
        print(f"  [could not hook HF: {exc}]")

    with torch.no_grad():
        olm(ids)
        hf(input_ids=ids)
    for h in handles:
        h.remove()

    first_bad = None
    for k in sorted(set(olm_acts) & set(hf_acts)):
        a, b = olm_acts[k], hf_acts[k]
        if a.shape != b.shape:
            print(f"  {'embed' if k < 0 else f'block {k:2d}'}  SHAPE "
                  f"{tuple(a.shape)} vs {tuple(b.shape)}")
            continue
        d = (a - b).abs().max().item()
        ratio = (a.norm() / b.norm()).item() if b.norm() > 0 else float("nan")
        flag = ""
        if d > 1e-3 and first_bad is None:
            first_bad = k
            flag = "  <-- FIRST DIVERGENCE"
        if k < 0 or k < 3 or flag or k == len(hf.model.layers) - 1:
            label = "embed" if k < 0 else f"block {k:2d}"
            print(f"  {label:<10} max|d| {d:10.4e}   |olm|/|hf| {ratio:7.4f}{flag}")

    print("\n" + "=" * 66)
    if first_bad is None:
        print("  blocks all match -- fault is in the final norm or lm_head.")
    elif first_bad == -1:
        print("  EMBEDDING already differs. Check whether OLM's Embedding")
        print("  wrapper scales by sqrt(d_model) or applies anything else.")
        emb_ratio = (olm_acts[-1].norm() / hf_acts[-1].norm()).item()
        print(f"  norm ratio = {emb_ratio:.6f}   (sqrt(1024) = 32.0)")
    else:
        print(f"  first divergence at block {first_bad}. Since every block is")
        print("  identical in structure, the fault is inside the block, not")
        print("  specific to that depth -- earlier blocks likely differ below")
        print("  the 1e-3 threshold and it compounds.")
    print("=" * 66)


if __name__ == "__main__":
    main()
