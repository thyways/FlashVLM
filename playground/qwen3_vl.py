from dataclasses import dataclass, field
from pathlib import Path
import sys

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from transformers.hf_argparser import HfArgumentParser

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

try:
    from qwen_vl_utils import process_vision_info
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise ImportError("请先安装 qwen-vl-utils: pip install qwen-vl-utils") from exc


@dataclass
class InferenceArguments:
    question: str = field(default="What is happening in this video?")
    video_path: str = field(default="path/to/video.mp4")
    model_path: str = field(default="Qwen/Qwen3-VL-4B-Instruct")
    num_frames: int = field(default=32)
    max_new_tokens: int = field(default=2048)
    min_pixels: int = field(default=64 * 28 * 28)
    max_pixels: int = field(default=256 * 28 * 28)
    attn_implementation: str = field(default="flash_attention_2")
    method: str = field(default="none")

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


def parse_args() -> InferenceArguments:
    parser = HfArgumentParser((InferenceArguments,))
    (arguments,) = parser.parse_args_into_dataclasses(return_remaining_strings=False)
    return arguments


def normalize_video_kwargs(video_kwargs: dict) -> dict:
    """
    transformers>=5 对 processor 的 videos kwargs 做了严格类型校验。
    qwen_vl_utils 可能返回 fps=[x]，这里规整为标量以兼容新版接口。
    """
    if not isinstance(video_kwargs, dict):
        return {}

    normalized = dict(video_kwargs)
    for key in ("fps", "num_frames"):
        value = normalized.get(key)
        if isinstance(value, list):
            if len(value) == 0:
                normalized[key] = None
            else:
                normalized[key] = value[0]
        elif isinstance(value, tuple):
            if len(value) == 0:
                normalized[key] = None
            else:
                normalized[key] = value[0]
    return normalized


def load_model(args: InferenceArguments):
    attn_impl = args.attn_implementation
    if args.method.lower() == "fastv" and attn_impl != "eager":
        print("[Info] FastV 需要 attention weights，自动将 attn_implementation 切换为 eager。")
        attn_impl = "eager"

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype="auto",
        device_map="auto",
        attn_implementation=attn_impl,
    )
    processor = AutoProcessor.from_pretrained(args.model_path)
    return model, processor


def apply_method(model: Qwen3VLForConditionalGeneration, args: InferenceArguments):
    method = args.method.lower().strip()
    if method in {"", "none", "baseline"}:
        print("[Method] baseline (no compression patch)")
        return model

    if method == "flashvlm":
        from method.flashvlm import apply_flashvlm_patch

        patched_layers = apply_flashvlm_patch(
            model,
            budget=args.flashvlm_budget,
            window_size=args.flashvlm_window_size,
            kernel_size=args.flashvlm_kernel_size,
            mix_lambda=args.flashvlm_mix_lambda,
            retain_ratio=args.flashvlm_retain_ratio,
            retain_direction=args.flashvlm_retain_direction,
        )
        print(f"[Method] flashvlm, patched_layers={patched_layers}")
        return model

    if method == "vidcom2":
        from method.vidcom2 import apply_vidcom2_patch

        apply_vidcom2_patch(model, base_scale=args.vidcom2_r_ratio)
        print(f"[Method] vidcom2, base_scale={args.vidcom2_r_ratio}")
        return model

    if method == "fastv":
        from method.fastv import apply_fastv_patch

        apply_fastv_patch(
            model,
            layer_k=args.fastv_k,
            retention_ratio=args.fastv_r_ratio,
        )
        print(f"[Method] fastv, layer_k={args.fastv_k}, retention_ratio={args.fastv_r_ratio}")
        return model

    if method == "visionzip":
        from method.visionzip import apply_visionzip_patch

        apply_visionzip_patch(
            model,
            retention_ratio=args.visionzip_r_ratio,
            dominant_ratio=args.visionzip_dominant_ratio,
            k_neighbors=args.visionzip_k_neighbors,
        )
        print(
            "[Method] visionzip, "
            f"retention_ratio={args.visionzip_r_ratio}, "
            f"dominant_ratio={args.visionzip_dominant_ratio}, "
            f"k_neighbors={args.visionzip_k_neighbors}"
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
        print(
            "[Method] holitom, "
            f"retain_ratio={args.holitom_r_ratio}, tau={args.holitom_t}, "
            f"beta={args.holitom_beta}, dominant_ratio={args.holitom_d}, "
            f"k_neighbors={args.holitom_k}, max_window_size={args.holitom_max_window_size}"
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
        print(
            "[Method] flashvid, "
            f"retention_ratio={args.flashvid_retention_ratio}, "
            f"token_selection_method={args.flashvid_token_selection_method}, "
            f"pruning_layer={args.flashvid_pruning_layer}, "
            f"llm_retention_ratio={args.flashvid_llm_retention_ratio}"
        )
        return model

    raise ValueError(
        "Unsupported method: "
        f"{args.method}. "
        "Supported: none, flashvlm, flashvid, vidcom2, fastv, visionzip, holitom"
    )


@torch.no_grad()
def inference(
    model: Qwen3VLForConditionalGeneration,
    processor: AutoProcessor,
    args: InferenceArguments,
) -> str:
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

    images, videos, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
    video_kwargs = normalize_video_kwargs(video_kwargs)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    print("video input:", videos[0].shape)
    num_frames, _, resized_height, resized_width = videos[0].shape
    print("num of video tokens:", int(num_frames / 2 * resized_height / 28 * resized_width / 28))

    inputs = processor(
        text=text,
        images=images,
        videos=videos,
        padding=True,
        return_tensors="pt",
        **video_kwargs,
    )

    inputs = inputs.to("cuda")
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
    )
    generated_text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )[0]
    return generated_text


def main(args: InferenceArguments):
    model, processor = load_model(args)
    model = apply_method(model, args)
    model.eval()

    generated_text = inference(model, processor, args)
    print(f"Generated Answer: {generated_text}")


if __name__ == "__main__":
    main(parse_args())
