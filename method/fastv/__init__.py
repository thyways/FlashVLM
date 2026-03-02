"""FastV integration helpers."""

from .fastv import compute_token_attention_scores, fastv_compression, select_important_tokens
from .modeling_qwen3_vl import apply_fastv_patch

__all__ = [
    "apply_fastv_patch",
    "fastv_compression",
    "select_important_tokens",
    "compute_token_attention_scores",
]
