import types
from typing import Callable

import torch
from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLTextAttention,
    apply_rotary_pos_emb,
    eager_attention_forward,
)
from transformers.processing_utils import Unpack

from .sa_kv import SA_KV


def Qwen3VLTextAttention_forward(
    self: Qwen3VLTextAttention,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    past_key_values: Cache | None = None,
    cache_position: torch.LongTensor | None = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> tuple[torch.Tensor, torch.Tensor | None]:
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        if hasattr(self, "sa_kv") and self.sa_kv is not None:
            key_states, value_states = self.sa_kv.update_kv(
                key_states=key_states,
                query_states=query_states,
                value_states=value_states,
            )
            past_key_values.key_cache[self.layer_idx] = key_states
            past_key_values.value_cache[self.layer_idx] = value_states

    attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
        self.config._attn_implementation,
        eager_attention_forward,
    )
    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


def apply_flashvlm_attention_patch(
    model,
    budget: int = 4096,
    window_size: int = 8,
    kernel_size: int = 7,
    mix_lambda: float = 0.07,
    retain_ratio: float = 0.1,
    retain_direction: str = "last",
) -> int:
    language_model = getattr(model.model, "language_model", None)
    if language_model is None or not hasattr(language_model, "layers"):
        raise ValueError("Unsupported Qwen3-VL model structure for FlashVLM patching.")

    patched_layers = 0
    for layer in language_model.layers:
        self_attn = layer.self_attn
        self_attn.sa_kv = SA_KV(
            budget=budget,
            window_size=window_size,
            kernel_size=kernel_size,
            mix_lambda=mix_lambda,
            retain_ratio=retain_ratio,
            retain_direction=retain_direction,
        )
        self_attn.forward = types.MethodType(Qwen3VLTextAttention_forward, self_attn)
        patched_layers += 1

    return patched_layers

