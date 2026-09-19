#!/usr/bin/env python
"""Compare the RoPE module the trained model actually uses against the one in
the converted HF model.

Everything else is cleared: embeddings match to 0.0, a single token matches to
0.0 (which exercises LayerNorm, SwiGLU, out_proj, the residuals and the tied
head), and FlashAttentionwithRoPE.forward is identical to the HF
re-implementation. Divergence appears only at seq_len >= 2, so RoPE at
positions > 0 is the remaining suspect.
"""
import sys

import torch

sys.path.insert(0, "src")
from model import CONFIGS, build_model
from transformers import AutoModelForCausalLM

olm = build_model(CONFIGS["prod_300m"])
olm.eval()
hf = AutoModelForCausalLM.from_pretrained("pocket-vakil-332m-ipc",
                                          trust_remote_code=True)
hf.eval()

attn = olm.blocks[1].stack[0].blocks[0].blocks[0].block.blocks[1]
r_olm = attn.rope
r_hf = hf.model.layers[0].self_attn.rope

print("OLM rope class :", type(r_olm).__module__ + "." + type(r_olm).__name__)
print("HF  rope class :", type(r_hf).__module__ + "." + type(r_hf).__name__)
print()

for name, r in (("olm", r_olm), ("hf", r_hf)):
    print(f"{name:>4}  attrs:", [a for a in
          ("head_dim", "base", "max_seq_len", "rotary_dim", "scaling_type")
          if hasattr(r, a)])
    for a in ("head_dim", "base", "max_seq_len", "rotary_dim", "scaling_type"):
        if hasattr(r, a):
            print(f"        {a} = {getattr(r, a)}")
    for b in ("inv_freq", "emb_sin", "emb_cos"):
        if hasattr(r, b):
            t = getattr(r, b)
            print(f"        {b}: shape {tuple(t.shape)}  first3 "
                  f"{t.flatten()[:3].tolist()}")
    print()

if hasattr(r_olm, "inv_freq") and hasattr(r_hf, "inv_freq"):
    d = (r_olm.inv_freq.float() - r_hf.inv_freq.float()).abs().max().item()
    print(f"inv_freq  max|d| = {d:.4e}")

if hasattr(r_olm, "emb_sin") and hasattr(r_hf, "emb_sin"):
    a, b = r_olm.emb_sin, r_hf.emb_sin
    if a.shape == b.shape:
        print(f"emb_sin   max|d| = {(a-b).abs().max().item():.4e}")
    else:
        print(f"emb_sin   SHAPE MISMATCH {tuple(a.shape)} vs {tuple(b.shape)}")

torch.manual_seed(0)
x = torch.randn(1, 6, 16, 64)
with torch.no_grad():
    a = r_olm(x).float()
    b = r_hf(x).float()

print(f"\nrope output on (1,6,16,64):")
print(f"  max|d| = {(a - b).abs().max().item():.4e}")
for p in range(6):
    d = (a[0, p] - b[0, p]).abs().max().item()
    print(f"  position {p}: max|d| {d:.4e}")
