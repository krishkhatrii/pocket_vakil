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
