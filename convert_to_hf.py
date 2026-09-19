#!/usr/bin/env python
"""Convert the Pocket Vakil OLM checkpoint into a HuggingFace folder that
loads with AutoModelForCausalLM.from_pretrained() and AutoTokenizer.from_pretrained().

Writes modelling code, config.json, safetensors weights and HF tokenizer files
into ONE output_dir, then verifies the converted model reproduces the original
OLM model's logits.

Three conventions are replicated exactly, because each one loads cleanly but
produces wrong numbers if guessed:

  SwiGLU      OLM: value, gate = x.chunk(2, -1); out = value * silu(gate)
              So up_proj[0:2688] is the VALUE half and [2688:5376] is the GATE
              half -- the opposite of Llama's gate_proj/up_proj ordering. We
              keep OLM's own naming so the mapping is a pure rename.

  RoPE        OLM rotates INTERLEAVED pairs (x0,x1),(x2,x3)... via
              x.view(b,s,h,d//2,2). Llama/GPT-NeoX rotate HALF-SPLIT pairs
              (x_i, x_{i+d/2}). Using the wrong one degrades attention with
              no error raised.

  LayerNorm   OLM upcasts to float32, uses var(unbiased=False), eps=1e-5,
              and parameters named gamma/beta.

Usage:
    python convert_to_hf.py --ckpt checkpoints/final_model_v2.pt \\
        --tokenizer data/tokenizer/bpe_32000_final.json \\
        --out pocket-vakil-332m-ipc
"""
import argparse
import json
import shutil
from pathlib import Path

import torch

# =====================================================================
# configuration_pocket_vakil.py
# =====================================================================
CONFIG_PY = '''
from transformers.configuration_utils import PretrainedConfig


class PocketVakilConfig(PretrainedConfig):
    """Pocket Vakil: pre-norm decoder, MHA + interleaved RoPE, SwiGLU FFN,
    LayerNorm, tied embeddings, biases on all projections."""

    model_type = "pocket_vakil"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=1024,
        num_hidden_layers=24,
        num_attention_heads=16,
        intermediate_size=2688,
        max_position_embeddings=2048,
        layer_norm_eps=1e-5,
        rope_theta=10000,
        attention_bias=True,
        mlp_bias=True,
        initializer_range=0.02,
        use_cache=False,
        pad_token_id=0,
        bos_token_id=0,
        eos_token_id=0,
        tie_word_embeddings=True,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.layer_norm_eps = layer_norm_eps
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.mlp_bias = mlp_bias
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
'''

# =====================================================================
# modeling_pocket_vakil.py
# =====================================================================
MODELING_PY = '''
"""Pocket Vakil modelling code -- a faithful re-implementation of the OLM
architecture the checkpoint was trained under.

Deliberate deviations from Llama, each load-bearing:
  * LayerNorm (gamma/beta, float32 internals) not RMSNorm
  * INTERLEAVED RoPE pairs, not half-split
  * SwiGLU as value * silu(gate) with value FIRST in the fused projection
  * biases on q/k/v/out and on up/down projections
"""
import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel

try:
    from transformers.generation import GenerationMixin
except ImportError:          # very old transformers
    class GenerationMixin:   # type: ignore
        pass

# transformers >= 4.56 expects a dict here; older versions expect a list.
_TIED_KEYS = (
    {"lm_head.weight": "model.embed_tokens.weight"}
    if hasattr(PreTrainedModel, "get_expanded_tied_weights_keys")
    else ["lm_head.weight"]
)

from .configuration_pocket_vakil import PocketVakilConfig


class PocketVakilLayerNorm(nn.Module):
    """OLM's LayerNorm: float32 internals, unbiased=False, gamma/beta."""

    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        in_dtype = x.dtype
        x = x.to(torch.float32)
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return (x * self.gamma + self.beta).to(in_dtype)


class PocketVakilRoPE(nn.Module):
    """Interleaved-pair RoPE, matching olm.nn.embeddings.positional.rope.

    Tables are computed in forward and cached, NOT registered as buffers.
    A non-persistent buffer is absent from the checkpoint, and from_pretrained
    materialises the model from the checkpoint -- so such a buffer comes back
    as uninitialised memory and silently corrupts every position. A single
    token hides it (softmax over one key is 1.0 regardless of q and k), which
    is what makes this failure mode so easy to miss.
    """

    def __init__(self, head_dim, max_seq_len, base=10000):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.base = float(base)
        self._cache_key = None
        self._cache = None

    def _tables(self, device, dtype):
        key = (str(device), dtype)
        if self._cache_key == key:
            return self._cache
        inv_freq = 1.0 / (
            self.base
            ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32,
                             device=device) / self.head_dim)
        )
        t = torch.arange(self.max_seq_len, dtype=torch.float32, device=device)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        sin = freqs.sin().unsqueeze(1).to(dtype)
        cos = freqs.cos().unsqueeze(1).to(dtype)
        self._cache_key = key
        self._cache = (sin, cos)
        return sin, cos

    def forward(self, x, position_ids=None):
        # x: (batch, seq, heads, head_dim)
        b, s, h, d = x.shape
        if position_ids is None:
            pos = torch.arange(s, device=x.device).unsqueeze(0).expand(b, s)
        else:
            pos = position_ids
        emb_sin, emb_cos = self._tables(x.device, x.dtype)
        sin = emb_sin[pos]   # (b, s, 1, d/2)
        cos = emb_cos[pos]

        x = x.view(b, s, h, d // 2, 2)
        x_even, x_odd = x[..., 0], x[..., 1]
        out_even = x_even * cos - x_odd * sin
        out_odd = x_even * sin + x_odd * cos
        return torch.stack((out_even, out_odd), dim=-1).view(b, s, h, d)

class PocketVakilAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        bias = config.attention_bias
        h = config.hidden_size
        self.q_proj = nn.Linear(h, h, bias=bias)
        self.k_proj = nn.Linear(h, h, bias=bias)
        self.v_proj = nn.Linear(h, h, bias=bias)
        self.out_proj = nn.Linear(h, h, bias=bias)
        self.rope = PocketVakilRoPE(
            self.head_dim, config.max_position_embeddings, config.rope_theta
        )

    def forward(self, x, attention_mask=None, position_ids=None):
        b, s, _ = x.shape
        q = self.q_proj(x).view(b, s, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(b, s, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(b, s, self.num_heads, self.head_dim)

        q = self.rope(q, position_ids)
        k = self.rope(k, position_ids)

        q = q.transpose(1, 2)   # (b, heads, s, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if attention_mask is None:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            # attention_mask: (b, s) with 1 for real tokens
            causal = torch.ones(s, s, dtype=torch.bool, device=x.device).tril()
            pad = attention_mask[:, None, None, :].to(torch.bool)
            mask = causal[None, None] & pad
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

        out = out.transpose(1, 2).reshape(b, s, -1)
        return self.out_proj(out)


class PocketVakilMLP(nn.Module):
    """up_proj emits [value | gate]; SwiGLU is value * silu(gate)."""

    def __init__(self, config):
        super().__init__()
        bias = config.mlp_bias
        self.up_proj = nn.Linear(
            config.hidden_size, 2 * config.intermediate_size, bias=bias
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=bias
        )

    def forward(self, x):
        x = self.up_proj(x)
        value, gate = x.chunk(2, dim=-1)
        return self.down_proj(value * F.silu(gate))


class PocketVakilLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_layernorm = PocketVakilLayerNorm(
            config.hidden_size, config.layer_norm_eps)
        self.self_attn = PocketVakilAttention(config)
        self.post_attention_layernorm = PocketVakilLayerNorm(
            config.hidden_size, config.layer_norm_eps)
        self.mlp = PocketVakilMLP(config)

    def forward(self, x, attention_mask=None, position_ids=None):
        x = x + self.self_attn(self.input_layernorm(x), attention_mask, position_ids)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class PocketVakilPreTrainedModel(PreTrainedModel):
    config_class = PocketVakilConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = False
    _no_split_modules = ["PocketVakilLayer"]

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)


class PocketVakilModel(PocketVakilPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [PocketVakilLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = PocketVakilLayerNorm(config.hidden_size, config.layer_norm_eps)
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                inputs_embeds=None, return_dict=True, **kwargs):
        x = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        for layer in self.layers:
            x = layer(x, attention_mask, position_ids)
        x = self.norm(x)
        if not return_dict:
            return (x,)
        return BaseModelOutputWithPast(last_hidden_state=x)


class PocketVakilForCausalLM(PocketVakilPreTrainedModel, GenerationMixin):
    _tied_weights_keys = _TIED_KEYS

    def __init__(self, config):
        super().__init__(config)
        self.model = PocketVakilModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        self.lm_head = value

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                inputs_embeds=None, labels=None, return_dict=True, **kwargs):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask,
                         position_ids=position_ids, inputs_embeds=inputs_embeds,
                         return_dict=True)
        logits = self.lm_head(out.last_hidden_state).float()

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        if not return_dict:
            return (loss, logits) if loss is not None else (logits,)
        return CausalLMOutputWithPast(loss=loss, logits=logits)

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **kwargs):
        # No KV cache: the full prefix is re-run each step. Correct, not fast.
        return {"input_ids": input_ids, "attention_mask": attention_mask}
'''


def build_key_map(n_layers):
    """OLM key -> HF key. Pure rename; no tensor surgery anywhere."""
    m = {"blocks.0.embedding.weight": "model.embed_tokens.weight",
        "blocks.2.blocks.1.weight": "lm_head.weight",
        "blocks.2.blocks.0.gamma": "model.norm.gamma",
        "blocks.2.blocks.0.beta": "model.norm.beta"}
    for i in range(n_layers):
        src = f"blocks.1.stack.{i}.blocks.0"
        dst = f"model.layers.{i}"
        m[f"{src}.blocks.0.block.blocks.0.gamma"] = f"{dst}.input_layernorm.gamma"
        m[f"{src}.blocks.0.block.blocks.0.beta"] = f"{dst}.input_layernorm.beta"
        for p in ("q_proj", "k_proj", "v_proj", "out_proj"):
            for t in ("weight", "bias"):
                m[f"{src}.blocks.0.block.blocks.1.{p}.{t}"] = \
                    f"{dst}.self_attn.{p}.{t}"
        m[f"{src}.blocks.1.block.blocks.0.gamma"] = \
            f"{dst}.post_attention_layernorm.gamma"
        m[f"{src}.blocks.1.block.blocks.0.beta"] = \
            f"{dst}.post_attention_layernorm.beta"
        for p in ("up_proj", "down_proj"):
            for t in ("weight", "bias"):
                m[f"{src}.blocks.1.block.blocks.1.{p}.{t}"] = f"{dst}.mlp.{p}.{t}"
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/final_model_v2.pt")
    ap.add_argument("--tokenizer", default="data/tokenizer/bpe_32000_final.json")
    ap.add_argument("--out", default="pocket-vakil-332m-ipc")
    ap.add_argument("--layers", type=int, default=24)
    ap.add_argument("--skip-verify", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # 1. modelling code -------------------------------------------------
    (out / "configuration_pocket_vakil.py").write_text(
        CONFIG_PY.lstrip(), encoding="utf-8")
    (out / "modeling_pocket_vakil.py").write_text(
        MODELING_PY.lstrip(), encoding="utf-8")
    print(f"[1/5] wrote modelling code -> {out}")

    # 2. load and remap --------------------------------------------------
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = ck.get("model_state_dict", ck) if isinstance(ck, dict) else ck
    key_map = build_key_map(args.layers)

    missing = [k for k in key_map if k not in sd]
    extra = [k for k in sd if k not in key_map]
    if missing or extra:
        print(f"  !! unmapped -- missing {len(missing)}, extra {len(extra)}")
        for k in (missing + extra)[:10]:
            print("     ", k)
        raise SystemExit("key map does not cover the checkpoint; aborting")

    new_sd = {key_map[k]: v.clone() for k, v in sd.items()}
    new_sd["lm_head.weight"] = new_sd["model.embed_tokens.weight"].clone()
    print(f"[2/5] remapped {len(sd)} tensors (+1 tied lm_head)")

    # 3. build and save the model ---------------------------------------
    import sys
    sys.path.insert(0, str(out.resolve().parent))
    spec_dir = out.resolve()
    sys.path.insert(0, str(spec_dir))
    import importlib.util

    def load_mod(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    cfg_mod = load_mod("configuration_pocket_vakil",
                       out / "configuration_pocket_vakil.py")
    # modeling uses a relative import; patch it for this local load only
    src = (out / "modeling_pocket_vakil.py").read_text(encoding="utf-8")
    tmp = out / "_modeling_local.py"
    tmp.write_text(src.replace("from .configuration_pocket_vakil import",
                               "from configuration_pocket_vakil import"),
                   encoding="utf-8")
    mdl_mod = load_mod("_modeling_local", tmp)

    config = cfg_mod.PocketVakilConfig()
    model = mdl_mod.PocketVakilForCausalLM(config)
    incompatible = model.load_state_dict(new_sd, strict=True)
    model.eval()
    print(f"[3/5] state_dict loaded strict=True {incompatible}")

    config.auto_map = {
        "AutoConfig": "configuration_pocket_vakil.PocketVakilConfig",
        "AutoModelForCausalLM": "modeling_pocket_vakil.PocketVakilForCausalLM",
    }
    model.config = config
    model.save_pretrained(out, safe_serialization=True)
    tmp.unlink(missing_ok=True)
    print(f"[4/5] save_pretrained -> {out}")

    # 4. tokenizer -------------------------------------------------------
    from tokenizers import Tokenizer
    from transformers import PreTrainedTokenizerFast

    raw = Tokenizer.from_file(args.tokenizer)
    eos = raw.id_to_token(0)
    print(f"      token id 0 = {eos!r}")
    tok = PreTrainedTokenizerFast(
        tokenizer_object=raw,
        eos_token=eos,
        bos_token=eos,
        unk_token=eos,
        pad_token=eos,          # eos fallback, as required
        model_max_length=2048,
    )
    tok.save_pretrained(out)
    shutil.copy2(args.tokenizer, out / "tokenizer_source.json")
    print(f"[5/5] tokenizer saved (pad_token={tok.pad_token!r})")

    # 5. verify ----------------------------------------------------------
    print("\n" + "=" * 62)
    print("VERIFY: reload with from_pretrained")
    print("=" * 62)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    hf = AutoModelForCausalLM.from_pretrained(out, trust_remote_code=True)
    hf.eval()
    htok = AutoTokenizer.from_pretrained(out, trust_remote_code=True)
    cfg = json.loads((out / "config.json").read_text(encoding="utf-8"))
    print(f"  reload OK | config vocab_size {cfg['vocab_size']} | "
          f"tokenizer vocab {htok.vocab_size} | pad {htok.pad_token!r}")
    if cfg["vocab_size"] != htok.vocab_size:
        print("  !! VOCAB SIZE MISMATCH -- scoring can silently break")

    if args.skip_verify:
        return

    print("\n  comparing logits against the original OLM model...")
    try:
        sys.path.insert(0, "src")
        from model import CONFIGS, build_model
        olm = build_model(CONFIGS["prod_300m"])
        olm.load_state_dict(sd)
        olm.eval()

        ids = torch.tensor([htok.encode(
            "The petitioner submitted that the accused had committed an offence "
            "punishable under the provisions of the Act.")[:64]])
        with torch.no_grad():
            a = olm(ids)
            b = hf(input_ids=ids).logits
        a = a.float()
        b = b.float()
        diff = (a - b).abs().max().item()
        agree = (a.argmax(-1) == b.argmax(-1)).float().mean().item()
        print(f"  max |Δlogit| = {diff:.3e}")
        print(f"  argmax agreement = {agree*100:.2f}%")
        if diff < 1e-3 and agree == 1.0:
            print("\n  PASS -- the converted model matches the original.")
        else:
            print("\n  FAIL -- conversion is NOT faithful. Do not submit this.")
    except Exception as exc:
        print(f"  [could not run OLM comparison: {exc}]")
        print("  Run this on a machine with OLM installed before submitting.")


if __name__ == "__main__":
    main()
