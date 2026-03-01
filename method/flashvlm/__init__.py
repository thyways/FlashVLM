"""FlashVLM integration helpers."""

from .modeling_qwen3_vl import apply_flashvlm_attention_patch

__all__ = ["apply_flashvlm_attention_patch"]

