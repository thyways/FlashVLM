"""Compatibility wrapper for legacy FastV Qwen3-VL imports."""

from ..modeling_qwen3_vl import Qwen3VLModel_forward, apply_fastv_patch

__all__ = ["Qwen3VLModel_forward", "apply_fastv_patch"]
