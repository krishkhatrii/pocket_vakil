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
