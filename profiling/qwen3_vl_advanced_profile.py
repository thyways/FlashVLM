from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from pathlib import Path
import json
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from transformers.hf_argparser import HfArgumentParser

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


@dataclass
class ProfileArguments:
    question: str = field(default="What is happening in this video?")
    video_path: str = field(default="path/to/video.mp4")
    model_path: str = field(default="Qwen/Qwen3-VL-4B-Instruct")
    method: str = field(default="none")
    num_frames: int = field(default=32)
    max_new_tokens: int = field(default=128)
    min_pixels: int = field(default=64 * 28 * 28)
    max_pixels: int = field(default=256 * 28 * 28)
    attn_implementation: str = field(default="flash_attention_2")
    report_path: str = field(default="")
    # 控制 qwen_vl_utils 视频解码后端：auto/decord/torchvision/torchcodec
    qwen_video_reader_backend: str = field(default="auto")
    # 原始视频解码阶段仅用于诊断，默认关闭以降低底层解码库触发崩溃概率
    skip_raw_decode: bool = field(default=True)

    # FlashVLM
    flashvlm_budget: int = field(default=4096)
    flashvlm_window_size: int = field(default=8)
    flashvlm_kernel_size: int = field(default=7)
    flashvlm_mix_lambda: float = field(default=0.07)
    flashvlm_retain_ratio: float = field(default=0.1)
    flashvlm_retain_direction: str = field(default="last")

    # VidCom2
    vidcom2_r_ratio: float = field(default=0.25)

    # FastV
    fastv_k: int = field(default=2)
    fastv_r_ratio: float = field(default=0.5)

    # VisionZip
    visionzip_r_ratio: float = field(default=0.2)
    visionzip_dominant_ratio: float = field(default=0.6)
    visionzip_k_neighbors: int = field(default=5)

    # HoliTom
    holitom_r_ratio: float = field(default=0.15)
    holitom_t: float = field(default=0.8)
    holitom_beta: float = field(default=0.6)
    holitom_d: float = field(default=0.0)
    holitom_k: int = field(default=7)
    holitom_max_window_size: int = field(default=1024)

    # FlashVID
    flashvid_retention_ratio: float = field(default=0.25)
    flashvid_do_segment: bool = field(default=True)
    flashvid_segment_threshold: float = field(default=0.9)
    flashvid_min_segment_num: int = field(default=8)
    flashvid_complementary_segment: bool = field(default=True)
    flashvid_token_selection_method: str = field(default="attn_div")
    flashvid_alpha: float = field(default=0.7)
    flashvid_temporal_threshold: float = field(default=0.8)
    flashvid_expansion: float = field(default=1.25)
    flashvid_pruning_layer: int = field(default=20)
    flashvid_llm_retention_ratio: float = field(default=0.3)


def _sync_cuda_if_needed() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class TimingCollector:
    def __init__(self) -> None:
        self._records: Dict[str, Dict[str, float]] = {}
        self._events: List[Tuple[str, float]] = []

    @contextmanager
    def time(self, label: str):
        _sync_cuda_if_needed()
        start = time.perf_counter()
        try:
            yield
        finally:
            _sync_cuda_if_needed()
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            stat = self._records.setdefault(
                label,
                {"total_ms": 0.0, "calls": 0.0, "min_ms": float("inf"), "max_ms": 0.0},
            )
            stat["total_ms"] += elapsed_ms
            stat["calls"] += 1
            stat["min_ms"] = min(stat["min_ms"], elapsed_ms)
            stat["max_ms"] = max(stat["max_ms"], elapsed_ms)
            self._events.append((label, elapsed_ms))

    def summary(self) -> Dict[str, Dict[str, float]]:
        result: Dict[str, Dict[str, float]] = {}
        for label, stat in self._records.items():
            calls = int(stat["calls"])
            total_ms = stat["total_ms"]
            result[label] = {
                "calls": calls,
                "total_ms": round(total_ms, 3),
                "avg_ms": round(total_ms / max(calls, 1), 3),
                "min_ms": round(stat["min_ms"], 3),
                "max_ms": round(stat["max_ms"], 3),
            }
        return result

    def top(self, n: int = 20) -> List[Tuple[str, Dict[str, float]]]:
        summary = self.summary()
        return sorted(summary.items(), key=lambda item: item[1]["total_ms"], reverse=True)[:n]


def normalize_video_kwargs(video_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(video_kwargs, dict):
        return {}

    normalized = dict(video_kwargs)
    for key in ("fps", "num_frames"):
        value = normalized.get(key)
        if isinstance(value, list):
            normalized[key] = value[0] if len(value) > 0 else None
        elif isinstance(value, tuple):
            normalized[key] = value[0] if len(value) > 0 else None
    return normalized


def _wrap_function(module: Any, fn_name: str, collector: TimingCollector, label: str) -> Optional[Callable[[], None]]:
    if not hasattr(module, fn_name):
        return None
    original = getattr(module, fn_name)
    if not callable(original):
        return None

    def wrapped(*args, **kwargs):
        with collector.time(label):
            return original(*args, **kwargs)

    setattr(module, fn_name, wrapped)

    def restore() -> None:
        setattr(module, fn_name, original)

    return restore


def install_method_component_hooks(method: str, collector: TimingCollector) -> List[Callable[[], None]]:
    hooks: List[Callable[[], None]] = []
    method = method.lower().strip()

    if method == "flashvlm":
        from method.flashvlm import sa_kv, modeling_qwen3_vl

        restore = _wrap_function(
            modeling_qwen3_vl,
            "Qwen3VLTextAttention_forward",
            collector,
            "method.flashvlm.text_attention_forward",
        )
        if restore:
            hooks.append(restore)

        if hasattr(sa_kv.SA_KV, "update_kv"):
            original = sa_kv.SA_KV.update_kv

            def wrapped_update_kv(self, *args, **kwargs):
                with collector.time("method.flashvlm.sa_kv_update"):
                    return original(self, *args, **kwargs)

            sa_kv.SA_KV.update_kv = wrapped_update_kv

            def restore() -> None:
                sa_kv.SA_KV.update_kv = original

            hooks.append(restore)

    elif method == "vidcom2":
        from method.vidcom2 import modeling_qwen3_vl, vidcom2

        for fn_name, label in [
            ("_compute_keep_indices", "method.vidcom2.compute_keep_indices"),
            ("Qwen3VLModel_forward", "method.vidcom2.model_forward"),
        ]:
            restore = _wrap_function(modeling_qwen3_vl, fn_name, collector, label)
            if restore:
                hooks.append(restore)
        for fn_name, label in [
            ("select_low_var_channels", "method.vidcom2.select_low_var_channels"),
            ("compute_gaussian_scores", "method.vidcom2.compute_gaussian_scores"),
            ("compute_scales", "method.vidcom2.compute_scales"),
            ("select_outlier_indices", "method.vidcom2.select_outlier_indices"),
        ]:
            restore = _wrap_function(vidcom2, fn_name, collector, label)
            if restore:
                hooks.append(restore)

    elif method == "fastv":
        from method.fastv import modeling_qwen3_vl

        for fn_name, label in [
            ("Qwen3VLModel_forward", "method.fastv.model_forward"),
            ("_fastv_text_model_forward", "method.fastv.text_model_forward"),
        ]:
            restore = _wrap_function(modeling_qwen3_vl, fn_name, collector, label)
            if restore:
                hooks.append(restore)

    elif method == "visionzip":
        from method.visionzip import modeling_qwen3_vl, visionzip

        for fn_name, label in [
            ("_get_video_features_with_attention", "method.visionzip.get_video_features_with_attention"),
            ("_compress_video_tokens", "method.visionzip.compress_video_tokens"),
            ("Qwen3VLModel_forward", "method.visionzip.model_forward"),
        ]:
            restore = _wrap_function(modeling_qwen3_vl, fn_name, collector, label)
            if restore:
                hooks.append(restore)
        restore = _wrap_function(visionzip, "visionzip_compression", collector, "method.visionzip.visionzip_compression")
        if restore:
            hooks.append(restore)

    elif method == "holitom":
        from method.holitom import holitom, modeling_qwen3_vl

        for fn_name, label in [
            ("_get_video_features_with_attention", "method.holitom.get_video_features_with_attention"),
            ("_compress_video_tokens", "method.holitom.compress_video_tokens"),
            ("Qwen3VLModel_forward", "method.holitom.model_forward"),
        ]:
            restore = _wrap_function(modeling_qwen3_vl, fn_name, collector, label)
            if restore:
                hooks.append(restore)
        for fn_name, label in [
            ("holitom_compression", "method.holitom.holitom_compression"),
            ("holitom_segment_compression", "method.holitom.holitom_segment_compression"),
            ("select_static_windows", "method.holitom.select_static_windows"),
        ]:
            restore = _wrap_function(holitom, fn_name, collector, label)
            if restore:
                hooks.append(restore)

    elif method == "flashvid":
        from method.flashvid import flashvid, modeling_qwen3_vl

        for fn_name, label in [
            ("Qwen3VLModel_forward", "method.flashvid.model_forward"),
            ("Qwen3VLTextModel_forward", "method.flashvid.text_model_forward"),
            ("_get_video_features_with_cls_attention", "method.flashvid.get_video_features_with_cls_attention"),
        ]:
            restore = _wrap_function(modeling_qwen3_vl, fn_name, collector, label)
            if restore:
                hooks.append(restore)
        for fn_name, label in [
            ("flashvid_compression", "method.flashvid.flashvid_compression"),
            ("segment_compression", "method.flashvid.segment_compression"),
            ("spatiotemporal_compression", "method.flashvid.spatiotemporal_compression"),
            ("fastv_prune", "method.flashvid.fastv_prune"),
        ]:
            restore = _wrap_function(flashvid, fn_name, collector, label)
            if restore:
                hooks.append(restore)

    return hooks


def apply_method(model: Qwen3VLForConditionalGeneration, args: ProfileArguments) -> Qwen3VLForConditionalGeneration:
    method = args.method.lower().strip()
    if method in {"", "none", "baseline"}:
        return model

    if method == "flashvlm":
        from method.flashvlm import apply_flashvlm_patch

        apply_flashvlm_patch(
            model,
            budget=args.flashvlm_budget,
            window_size=args.flashvlm_window_size,
            kernel_size=args.flashvlm_kernel_size,
            mix_lambda=args.flashvlm_mix_lambda,
            retain_ratio=args.flashvlm_retain_ratio,
            retain_direction=args.flashvlm_retain_direction,
        )
        return model

    if method == "vidcom2":
        from method.vidcom2 import apply_vidcom2_patch

        apply_vidcom2_patch(model, base_scale=args.vidcom2_r_ratio)
        return model

    if method == "fastv":
        from method.fastv import apply_fastv_patch

        apply_fastv_patch(model, layer_k=args.fastv_k, retention_ratio=args.fastv_r_ratio)
        return model

    if method == "visionzip":
        from method.visionzip import apply_visionzip_patch

        apply_visionzip_patch(
            model,
            retention_ratio=args.visionzip_r_ratio,
            dominant_ratio=args.visionzip_dominant_ratio,
            k_neighbors=args.visionzip_k_neighbors,
        )
        return model

    if method == "holitom":
        from method.holitom import apply_holitom_patch

        apply_holitom_patch(
            model,
            retain_ratio=args.holitom_r_ratio,
            tau=args.holitom_t,
            beta=args.holitom_beta,
            dominant_ratio=args.holitom_d,
            k_neighbors=args.holitom_k,
            max_window_size=args.holitom_max_window_size,
        )
        return model

    if method == "flashvid":
        from method.flashvid import apply_flashvid_patch

        apply_flashvid_patch(
            model,
            retention_ratio=args.flashvid_retention_ratio,
            do_segment=args.flashvid_do_segment,
            segment_threshold=args.flashvid_segment_threshold,
            min_segment_num=args.flashvid_min_segment_num,
            complementary_segment=args.flashvid_complementary_segment,
            token_selection_method=args.flashvid_token_selection_method,
            alpha=args.flashvid_alpha,
            temporal_threshold=args.flashvid_temporal_threshold,
            expansion=args.flashvid_expansion,
            pruning_layer=args.flashvid_pruning_layer,
            llm_retention_ratio=args.flashvid_llm_retention_ratio,
        )
        return model

    raise ValueError(
        f"Unsupported method: {args.method}. "
        "Supported: none, flashvlm, flashvid, vidcom2, fastv, visionzip, holitom"
    )


def install_model_runtime_hooks(
    model: Qwen3VLForConditionalGeneration,
    collector: TimingCollector,
) -> List[Callable[[], None]]:
    restores: List[Callable[[], None]] = []

    # Hook visual encoder runtime.
    visual = model.model.visual
    orig_visual_forward = visual.forward

    def wrapped_visual_forward(*args, **kwargs):
        with collector.time("stage.visual_encode"):
            return orig_visual_forward(*args, **kwargs)

    visual.forward = wrapped_visual_forward

    def restore_visual() -> None:
        visual.forward = orig_visual_forward

    restores.append(restore_visual)

    # Hook top-level forward to split prefill and decoding.
    orig_forward = model.forward

    def wrapped_model_forward(*args, **kwargs):
        input_ids = kwargs.get("input_ids", None)
        inputs_embeds = kwargs.get("inputs_embeds", None)
        past_key_values = kwargs.get("past_key_values", None)
        seq_len = None
        if input_ids is not None:
            seq_len = input_ids.shape[1]
        elif inputs_embeds is not None:
            seq_len = inputs_embeds.shape[1]
        is_decode = bool(seq_len == 1 and past_key_values is not None)
        label = "stage.llm_decode_step" if is_decode else "stage.llm_prefill"
        with collector.time(label):
            return orig_forward(*args, **kwargs)

    model.forward = wrapped_model_forward

    def restore_forward() -> None:
        model.forward = orig_forward

    restores.append(restore_forward)
    return restores


def profile_video_decode(video_path: str, num_frames: int, collector: TimingCollector) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "decoded_frames": 0,
        "sampled_indices": [],
        "status": "not_run",
    }
    try:
        import decord
    except Exception as exc:  # pragma: no cover - runtime dependency
        info["status"] = f"skipped: decord_import_failed: {exc}"
        return info

    with collector.time("stage.video_decode_raw"):
        try:
            vr = decord.VideoReader(video_path)
            total = len(vr)
            if total <= 0:
                info["status"] = "ok_empty_video"
                return info
            frame_count = min(max(num_frames, 1), total)
            indices = torch.linspace(0, total - 1, steps=frame_count, dtype=torch.long).tolist()
            _ = vr.get_batch(indices)
            info["decoded_frames"] = frame_count
            info["sampled_indices"] = indices
            info["status"] = "ok"
        except Exception as exc:
            info["status"] = f"failed: {exc}"
    return info


def _resolve_eos_token_ids(model: Qwen3VLForConditionalGeneration) -> List[int]:
    eos = getattr(model.generation_config, "eos_token_id", None)
    if eos is None:
        eos = model.config.eos_token_id
    if eos is None:
        return []
    if isinstance(eos, (list, tuple)):
        return [int(x) for x in eos]
    return [int(eos)]


@torch.inference_mode()
def _fastv_greedy_generate(
    model: Qwen3VLForConditionalGeneration,
    inputs: Dict[str, torch.Tensor],
    max_new_tokens: int,
) -> torch.LongTensor:
    if max_new_tokens <= 0:
        return inputs["input_ids"]

    model_inputs = dict(inputs)
    model_inputs["use_cache"] = True
    model_inputs["return_dict"] = True

    outputs = model(**model_inputs)
    past_key_values = outputs.past_key_values
    next_token = outputs.logits[:, -1:, :].argmax(dim=-1)
    generated = [inputs["input_ids"], next_token]

    eos_token_ids = _resolve_eos_token_ids(model)
    eos_tensor = (
        torch.tensor(eos_token_ids, device=next_token.device, dtype=next_token.dtype)
        if eos_token_ids
        else None
    )
    if eos_tensor is not None and bool(torch.isin(next_token, eos_tensor).all()):
        return torch.cat(generated, dim=1)

    for _ in range(max_new_tokens - 1):
        if past_key_values is None:
            raise RuntimeError("FastV greedy generation requires KV cache, but past_key_values is None.")
        past_len = int(past_key_values.get_seq_length())
        decode_inputs = {
            "input_ids": next_token,
            "past_key_values": past_key_values,
            "use_cache": True,
            "return_dict": True,
            "cache_position": torch.tensor([past_len], device=next_token.device, dtype=torch.long),
            "attention_mask": torch.ones(
                (next_token.shape[0], past_len + 1),
                device=next_token.device,
                dtype=torch.long,
            ),
        }
        outputs = model(**decode_inputs)
        past_key_values = outputs.past_key_values
        next_token = outputs.logits[:, -1:, :].argmax(dim=-1)
        generated.append(next_token)

        if eos_tensor is not None and bool(torch.isin(next_token, eos_tensor).all()):
            break

    return torch.cat(generated, dim=1)


def run_profile(args: ProfileArguments) -> Dict[str, Any]:
    collector = TimingCollector()
    method = args.method.lower().strip()
    backend = args.qwen_video_reader_backend.strip().lower()

    if backend not in {"auto", "decord", "torchvision", "torchcodec"}:
        raise ValueError(
            f"Unsupported qwen_video_reader_backend={args.qwen_video_reader_backend}, "
            "supported: auto, decord, torchvision, torchcodec"
        )

    if backend == "auto":
        os.environ.pop("FORCE_QWENVL_VIDEO_READER", None)
    else:
        os.environ["FORCE_QWENVL_VIDEO_READER"] = backend

    try:
        from qwen_vl_utils import process_vision_info
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise ImportError("请先安装 qwen-vl-utils: pip install qwen-vl-utils") from exc

    if method == "fastv" and args.attn_implementation != "eager":
        print("[Info] FastV 依赖 attention weights，自动将 attn_implementation 切换为 eager。")
        args.attn_implementation = "eager"

    with collector.time("stage.model_load"):
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            args.model_path,
            torch_dtype="auto",
            device_map="auto",
            attn_implementation=args.attn_implementation,
        )
        processor = AutoProcessor.from_pretrained(args.model_path)

    method_hooks = install_method_component_hooks(method, collector)
    with collector.time("stage.method_patch"):
        model = apply_method(model, args)
    runtime_hooks = install_model_runtime_hooks(model, collector)

    model.eval()

    report: Dict[str, Any] = {
        "args": asdict(args),
        "video_decode_info": {},
        "input_tokens": None,
        "generated_tokens": None,
        "generated_text": None,
    }

    with collector.time("stage.e2e_total"):
        if args.skip_raw_decode:
            report["video_decode_info"] = {"status": "skipped_by_arg", "decoded_frames": 0, "sampled_indices": []}
        else:
            report["video_decode_info"] = profile_video_decode(args.video_path, args.num_frames, collector)

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {
                        "video": args.video_path,
                        "max_pixels": args.max_pixels,
                        "min_pixels": args.min_pixels,
                        "nframes": args.num_frames,
                    },
                    {"type": "text", "text": args.question},
                ],
            },
        ]

        with collector.time("stage.qwen_vl_process_vision_info"):
            images, videos, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
            video_kwargs = normalize_video_kwargs(video_kwargs)

        with collector.time("stage.processor_chat_template"):
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        with collector.time("stage.processor_pack_inputs"):
            inputs = processor(
                text=text,
                images=images,
                videos=videos,
                padding=True,
                return_tensors="pt",
                **video_kwargs,
            )

        with collector.time("stage.inputs_to_device"):
            inputs = inputs.to("cuda")

        input_tokens = int(inputs["input_ids"].shape[1]) if "input_ids" in inputs else None
        report["input_tokens"] = input_tokens

        with collector.time("stage.llm_generate_total"):
            if method == "fastv":
                generated_ids = _fastv_greedy_generate(
                    model=model,
                    inputs=inputs,
                    max_new_tokens=args.max_new_tokens,
                )
            else:
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                )

        with collector.time("stage.processor_decode_text"):
            generated_text = processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )[0]

        if input_tokens is not None:
            generated_tokens = int(generated_ids.shape[1] - input_tokens)
        else:
            generated_tokens = int(generated_ids.shape[1])
        report["generated_tokens"] = generated_tokens
        report["generated_text"] = generated_text

    # Restore monkey patches.
    for restore in runtime_hooks + method_hooks:
        restore()

    summary = collector.summary()
    top = collector.top(40)

    e2e_ms = summary.get("stage.e2e_total", {}).get("total_ms", 0.0)
    for _, stat in top:
        if e2e_ms > 0:
            stat["pct_of_e2e"] = round(100.0 * stat["total_ms"] / e2e_ms, 3)
        else:
            stat["pct_of_e2e"] = 0.0

    report["timing_summary"] = summary
    report["timing_top"] = [{"label": label, **stat} for label, stat in top]
    report["bottleneck"] = report["timing_top"][0] if report["timing_top"] else {}
    report["throughput"] = {
        "generated_tokens": report["generated_tokens"],
        "llm_generate_total_ms": summary.get("stage.llm_generate_total", {}).get("total_ms", 0.0),
        "tokens_per_second": round(
            1000.0 * (report["generated_tokens"] or 0) / max(summary.get("stage.llm_generate_total", {}).get("total_ms", 1e-9), 1e-9),
            3,
        ),
    }
    return report


def pretty_print_report(report: Dict[str, Any]) -> None:
    print("=" * 100)
    print("Advanced Profiling Report")
    print("=" * 100)
    print(f"method              : {report['args']['method']}")
    print(f"model               : {report['args']['model_path']}")
    print(f"video_path          : {report['args']['video_path']}")
    print(f"qwen_reader_backend : {report['args']['qwen_video_reader_backend']}")
    print(f"skip_raw_decode     : {report['args']['skip_raw_decode']}")
    print(f"input_tokens        : {report.get('input_tokens')}")
    print(f"generated_tokens    : {report.get('generated_tokens')}")
    print(f"tokens_per_second   : {report.get('throughput', {}).get('tokens_per_second')}")
    print("-" * 100)
    print("Top Bottlenecks (sorted by total_ms)")
    print("-" * 100)
    header = f"{'label':48s} {'calls':>7s} {'total_ms':>12s} {'avg_ms':>10s} {'pct_e2e':>10s}"
    print(header)
    print("-" * len(header))
    for row in report.get("timing_top", [])[:20]:
        label = row["label"][:48]
        print(
            f"{label:48s} "
            f"{int(row['calls']):7d} "
            f"{row['total_ms']:12.3f} "
            f"{row['avg_ms']:10.3f} "
            f"{row.get('pct_of_e2e', 0.0):10.3f}"
        )
    print("=" * 100)


def main() -> None:
    parser = HfArgumentParser((ProfileArguments,))
    (args,) = parser.parse_args_into_dataclasses(return_remaining_strings=False)

    report = run_profile(args)
    pretty_print_report(report)

    if args.report_path.strip():
        report_path = Path(args.report_path).expanduser()
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        report_path = PROJECT_ROOT / "profiling" / "reports" / f"profile_{args.method}_{timestamp}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[Saved] report -> {report_path}")


if __name__ == "__main__":
    main()
