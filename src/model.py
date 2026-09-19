"""
Pocket Vakil model definition, built from OLM's building blocks.

The architecture is `olm.nn.blocks.LM`, assembled from OLM primitives:

    Embedding -> Repeat(TransformerBlock) x N -> OutputHead

and each `TransformerBlock` is pre-norm with two residual branches:

    Residual(LayerNorm -> FlashAttentionwithRoPE)
    Residual(LayerNorm -> SwiGLUFFN)

Nothing here is hand-rolled. The only project-specific parts are the two size
presets and the parameter accounting used to confirm the compute budget.

Three presets, one class. **Only `embed_dim`, `num_heads` and `num_layers` differ**
-- context is 2048 everywhere, so a small preset exercises the exact sequence
length production uses and its throughput numbers extrapolate honestly.

    TINY        ~4M non-emb   -- CPU correctness checks only
    DRY_RUN     ~55M non-emb  -- single-GPU rehearsal
    PROD_300M   ~299M non-emb -- the production run

No preset assumes a particular GPU. Batch size and throughput are measured on
whatever card is present (see `src/slot1.py probe` and `smoke`), never assumed.

    python src/model.py     # parameter breakdown for every preset
"""

from dataclasses import dataclass, asdict

import torch
from olm.nn.blocks.LM import LM

# bpe_32000_final.json -- vocab includes the 100 IPC label tokens, the three ILSI
# mask tokens (<SECTION>/<ACT>/<ENTITY>) and EOS.
VOCAB_SIZE = 32_000
EOS_ID = 0


@dataclass
class ModelConfig:
    """Every argument OLM's `LM` accepts, plus a name for logging."""

    name: str
    embed_dim: int
    num_heads: int
    num_layers: int
    max_seq_len: int
    vocab_size: int = VOCAB_SIZE
    dropout: float = 0.0          # no dropout: single pass over ~2B tokens
    causal: bool = True
    ff_multiplier: float = 2.625  # SwiGLU parity with a 4x GELU MLP (8/3 ~ 2.667)
    tie_embeddings: bool = True

    @property
    def head_dim(self):
        assert self.embed_dim % self.num_heads == 0, "embed_dim must divide num_heads"
        return self.embed_dim // self.num_heads

    def to_dict(self):
        return asdict(self)


CONTEXT_LEN = 2048          # identical across presets, matches the packed shards

# CPU correctness checks. Too small to learn anything; big enough to prove the
# plumbing (masking, resume, parsing) is right.
TINY = ModelConfig(
    name="tiny",
    embed_dim=128,
    num_heads=4,
    num_layers=2,
    max_seq_len=CONTEXT_LEN,
)

# Single-GPU rehearsal at the production sequence length.
DRY_RUN = ModelConfig(
    name="dry_run",
    embed_dim=512,
    num_heads=8,
    num_layers=8,
    max_seq_len=CONTEXT_LEN,
)

# Production. Measured at 299.2M non-embedding, against the 300M budget in
# PLAN.md section 2 (6ND = 6 * 3e8 * 6e9 ~ 1.08e19 FLOPs).
PROD_300M = ModelConfig(
    name="prod_300m",
    embed_dim=1024,
    num_heads=16,
    num_layers=24,
    max_seq_len=CONTEXT_LEN,
)

CONFIGS = {"tiny": TINY, "dry_run": DRY_RUN, "prod_300m": PROD_300M}


def apply_scaled_init(model: LM, std: float = 0.02) -> LM:
    """Re-initialise embeddings and linears at std=0.02. **This overrides an OLM default.**

    OLM's `Embedding` wraps `nn.Embedding` and keeps torch's default N(0, 1) init.
    With tied embeddings the output projection is that same matrix, so logits come
    out at roughly std * sqrt(embed_dim) -- measured std 19.7 at embed_dim 384,
    giving an initial cross-entropy of ~361 where ln(32000) = 10.37 is correct.

    That is not a cosmetic difference. A model that starts ~350 nats from uniform
    spends its first thousands of steps collapsing logit magnitude instead of
    learning, and is prone to loss spikes while it does. GPT-style models use
    std=0.02 for exactly this reason.

    Applied by default in `build_model`. Pass `scaled_init=False` to reproduce
    OLM's stock behaviour, e.g. to confirm this description still holds.
    """
    import torch.nn as nn

    for module in model.modules():
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    return model


def build_model(cfg: ModelConfig, scaled_init: bool = True) -> LM:
    """Instantiate OLM's LM from a ModelConfig. Random init -- no pretrained weights."""
    model = LM(
        vocab_size=cfg.vocab_size,
        embed_dim=cfg.embed_dim,
        num_heads=cfg.num_heads,
        num_layers=cfg.num_layers,
        max_seq_len=cfg.max_seq_len,
        dropout=cfg.dropout,
        causal=cfg.causal,
        ff_multiplier=cfg.ff_multiplier,
        tie_embeddings=cfg.tie_embeddings,
    )
    if scaled_init:
        apply_scaled_init(model)
    return model


def param_breakdown(model: LM) -> dict:
    """Split embedding from non-embedding.

    Non-embedding is the number the compute budget is written against, and with
    tied embeddings the output projection shares storage with the input table --
    so `parameters()` already counts it once and no adjustment is needed.
    """
    embedding = model.blocks[0].embedding.weight.numel()
    total = sum(p.numel() for p in model.parameters())
    return {
        "embedding": embedding,
        "non_embedding": total - embedding,
        "total": total,
    }


def describe(cfg: ModelConfig) -> dict:
    model = build_model(cfg)
    c = param_breakdown(model)
    print(f"\n=== {cfg.name} ===")
    print(f"  embed_dim {cfg.embed_dim}  num_heads {cfg.num_heads}  "
          f"num_layers {cfg.num_layers}  head_dim {cfg.head_dim}")
    print(f"  max_seq_len {cfg.max_seq_len}  ff_multiplier {cfg.ff_multiplier}  "
          f"tied {cfg.tie_embeddings}  dropout {cfg.dropout}")
    print(f"  non-embedding {c['non_embedding']:>13,}  ({c['non_embedding']/1e6:.1f}M)")
    print(f"  embedding     {c['embedding']:>13,}  ({c['embedding']/1e6:.1f}M, tied)")
    print(f"  TOTAL         {c['total']:>13,}  ({c['total']/1e6:.1f}M)")
    # fp32 weights + grads + Adam m,v = 16 bytes/param
    print(f"  optimiser state (fp32 AdamW): ~{c['total']*16/1e9:.2f} GB")
    return c


if __name__ == "__main__":
    torch.manual_seed(0)
    for cfg in CONFIGS.values():
        describe(cfg)
