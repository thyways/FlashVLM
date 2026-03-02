"""FastV model-specific compatibility exports."""

from .qwen3_vl import Qwen3VLModel_forward, apply_fastv_patch

__all__ = ["Qwen3VLModel_forward", "apply_fastv_patch"]
