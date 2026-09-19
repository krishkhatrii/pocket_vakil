from dataclasses import dataclass, asdict

import torch
from olm.nn.blocks.LM import LM

VOCAB_SIZE = 32_000
EOS_ID = 0


@dataclass
class ModelConfig:

    name: str
    embed_dim: int
    num_heads: int
    num_layers: int
    max_seq_len: int
    vocab_size: int = VOCAB_SIZE
    dropout: float = 0.0          
    causal: bool = True
    ff_multiplier: float = 2.625  
    tie_embeddings: bool = True

    @property
    def head_dim(self):
        assert self.embed_dim % self.num_heads == 0, "embed_dim must divide num_heads"
        return self.embed_dim // self.num_heads

    def to_dict(self):
        return asdict(self)


CONTEXT_LEN = 2048          

TINY = ModelConfig(
    name="tiny",
    embed_dim=128,
    num_heads=4,
    num_layers=2,
    max_seq_len=CONTEXT_LEN,
)

DRY_RUN = ModelConfig(
    name="dry_run",
    embed_dim=512,
    num_heads=8,
    num_layers=8,
    max_seq_len=CONTEXT_LEN,
)

PROD_300M = ModelConfig(
    name="prod_300m",
    embed_dim=1024,
    num_heads=16,
    num_layers=24,
    max_seq_len=CONTEXT_LEN,
)

CONFIGS = {"tiny": TINY, "dry_run": DRY_RUN, "prod_300m": PROD_300M}


def apply_scaled_init(model: LM, std: float = 0.02) -> LM:
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
    print(f"  optimiser state (fp32 AdamW): ~{c['total']*16/1e9:.2f} GB")
    return c


if __name__ == "__main__":
    torch.manual_seed(0)
    for cfg in CONFIGS.values():
        describe(cfg)
