from dataclasses import dataclass
from typing import List, Tuple

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import KVCache
from mlx_lm.models.qwen3 import MLP


@dataclass
class DFlashConfig:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    block_size: int
    target_layer_ids: Tuple[int, ...]
    num_target_layers: int
    mask_token_id: int = 0


class DFlashAttention(nn.Module):
    """DFlash cross-attention: Q from noise, K/V from [target_hidden, noise]."""

    def __init__(self, config: DFlashConfig):
        super().__init__()

        dim = config.hidden_size
        self.n_heads = n_heads = config.num_attention_heads
        self.n_kv_heads = n_kv_heads = config.num_key_value_heads

        head_dim = config.head_dim
        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)
        self.q_norm = nn.RMSNorm(head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(head_dim, eps=config.rms_norm_eps)

    def __call__(self, x: mx.array, x_ctx: mx.array, rope: nn.RoPE, cache: KVCache) -> mx.array:
        B, L, _ = x.shape
        S = x_ctx.shape[1]

        c = mx.concatenate([x_ctx, x], axis=1)
        queries, keys, values = self.q_proj(x), self.k_proj(c), self.v_proj(c)

        queries = self.q_norm(queries.reshape(B, L, self.n_heads, -1)).transpose(0, 2, 1, 3)
        keys = self.k_norm(keys.reshape(B, S + L, self.n_kv_heads, -1)).transpose(0, 2, 1, 3)
        values = values.reshape(B, S + L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        queries = rope(queries, offset=cache.offset + S)
        keys = rope(keys, offset=cache.offset)
        keys, values = cache.update_and_fetch(keys, values)

        output = mx.fast.scaled_dot_product_attention(queries, keys, values, scale=self.scale)
        return self.o_proj(output.transpose(0, 2, 1, 3).reshape(B, L, -1))


class DFlashDecoderLayer(nn.Module):
    """Single DFlash transformer layer with pre-norm."""

    def __init__(self, config: DFlashConfig):
        super().__init__()
        self.self_attn = DFlashAttention(config)
        self.mlp = MLP(config.hidden_size, config.intermediate_size)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(self, x: mx.array, x_ctx: mx.array, rope: nn.RoPE, cache: KVCache) -> mx.array:
        r = self.self_attn(self.input_layernorm(x), x_ctx, rope, cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        out = h + r
        return out


class DFlashDraftModel(nn.Module):
    """DFlash draft model: predicts multiple tokens in parallel using target hidden states."""

    def __init__(self, config: DFlashConfig):
        super().__init__()
        self.config = config

        concat_dim = len(config.target_layer_ids) * config.hidden_size
        self.fc = nn.Linear(concat_dim, config.hidden_size, bias=False)
        self.hidden_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.layers = [DFlashDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rope = nn.RoPE(config.head_dim, traditional=False, base=config.rope_theta)

        # Set by bind()
        self.embed_tokens = None
        self.lm_head = None

    def bind(self, target_model: nn.Module) -> "DFlashDraftModel":
        """Bind to target model's embedding and lm_head layers."""
        if hasattr(target_model, "embed_tokens"):
            inner = target_model
        elif hasattr(target_model, "model") and hasattr(target_model.model, "embed_tokens"):
            inner = target_model.model
        elif (hasattr(target_model, "language_model") and
              hasattr(target_model.language_model, "model") and
              hasattr(target_model.language_model.model, "embed_tokens")):
            inner = target_model.language_model.model
        else:
            raise AttributeError(f"Cannot find embed_tokens in {type(target_model).__name__}")
        self.embed_tokens = inner.embed_tokens
        lm = getattr(target_model, "language_model", target_model)
        lm_head = getattr(target_model, "lm_head", None) or getattr(lm, "lm_head", None)
        self.lm_head = lm_head or self.embed_tokens.as_linear
        return self

    def make_cache(self) -> List[KVCache]:
        """Create KV cache for generation."""
        return [KVCache() for _ in self.layers]

    def __call__(self, inputs: mx.array, target_hidden: mx.array, cache: List[KVCache]) -> mx.array:
        """Forward: inputs [B, L] + target_hidden [B, S, hidden*n] -> logits [B, L, vocab]."""
        h = self.embed_tokens(inputs)
        h_ctx = self.hidden_norm(self.fc(target_hidden))

        for layer, c in zip(self.layers, cache):
            h = layer(h, h_ctx, self.rope, c)

        return self.lm_head(self.norm(h))
