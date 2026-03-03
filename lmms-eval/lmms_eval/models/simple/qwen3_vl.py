import re
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple, Union

import decord
import numpy as np
import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMoeForConditionalGeneration,
    StoppingCriteria,
    StoppingCriteriaList,
)

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.imports import optional_import
from lmms_eval.models.model_utils.gen_metrics import log_metrics

process_vision_info, _has_qwen_vl = optional_import("qwen_vl_utils", "process_vision_info")
if not _has_qwen_vl:
    eval_logger.warning("Failed to import qwen_vl_utils; Please install it via `pip install qwen-vl-utils`")


class TTFTStoppingCriteria(StoppingCriteria):
    """Capture time-to-first-token during generation without altering stop behavior."""

    def __init__(self, start_time: float):
        super().__init__()
        self.start_time = start_time
        self.first_token_time: Optional[float] = None

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        if self.first_token_time is None:
            self.first_token_time = time.time()
        return False

    def get_ttft(self, fallback_end_time: float) -> float:
        first_token_time = self.first_token_time if self.first_token_time is not None else fallback_end_time
        return max(first_token_time - self.start_time, 0.0)


@register_model("qwen3_vl")
class Qwen3_VL(lmms):
    """
    Qwen3_VL Model
    "https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct"
    """

    def __init__(
        self,
        pretrained: str = "Qwen/Qwen3-VL-4B-Instruct",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache=True,
        attn_implementation: Optional[str] = None,
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 1605632,
        max_num_frames: int = 32,
        fps: Optional[float] = None,
        system_prompt: Optional[str] = "You are a helpful assistant.",
        interleave_visuals: Optional[bool] = False,
        reasoning_prompt: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        # Do not use kwargs for now
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        # Validate attention implementation
        valid_attn_implementations = [None, "flash_attention_2", "sdpa", "eager"]
        if attn_implementation not in valid_attn_implementations:
            raise ValueError(f"attn_implementation must be one of {valid_attn_implementations}, got {attn_implementation}")

        accelerator = Accelerator()
        self.accelerator = accelerator
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        else:
            self._device = torch.device(device)
            self.device_map = device_map if device_map else device

        # Prepare model loading arguments
        model_kwargs = {
            "dtype": "bfloat16",
            "device_map": self.device_map,
        }

        # Add attention implementation if specified
        if attn_implementation is not None:
            model_kwargs["attn_implementation"] = attn_implementation

        # check whether its an MoE model
        match = re.search(r"A\d+B", pretrained)
        model_fn = Qwen3VLMoeForConditionalGeneration if match else Qwen3VLForConditionalGeneration
        self._model = model_fn.from_pretrained(pretrained, **model_kwargs).eval()

        compressor = os.getenv("COMPRESSOR")
        if compressor == "flashvlm":
            try:
                from method.flashvlm import apply_flashvlm_patch
            except ImportError:
                project_root = Path(__file__).resolve().parents[4]
                if str(project_root) not in sys.path:
                    sys.path.append(str(project_root))
                from method.flashvlm import apply_flashvlm_patch

            budget = int(os.getenv("FLASHVLM_BUDGET", "4096"))
            window_size = int(os.getenv("FLASHVLM_WINDOW_SIZE", "8"))
            kernel_size = int(os.getenv("FLASHVLM_KERNEL_SIZE", "7"))
            mix_lambda = float(os.getenv("FLASHVLM_MIX_LAMBDA", "0.07"))
            retain_ratio = float(os.getenv("FLASHVLM_RETAIN_RATIO", "0.1"))
            retain_direction = os.getenv("FLASHVLM_RETAIN_DIRECTION", "last")

            patched_layers = apply_flashvlm_patch(
                self._model,
                budget=budget,
                window_size=window_size,
                kernel_size=kernel_size,
                mix_lambda=mix_lambda,
                retain_ratio=retain_ratio,
                retain_direction=retain_direction,
            )
            eval_logger.success(
                f"[FlashVLM] Patched {patched_layers} layers "
                f"(budget={budget}, window={window_size}, kernel={kernel_size}, "
                f"mix_lambda={mix_lambda}, retain_ratio={retain_ratio}, direction={retain_direction})."
            )
        elif compressor == "flashvid":
            try:
                from method.flashvid import apply_flashvid_patch
            except ImportError:
                project_root = Path(__file__).resolve().parents[4]
                if str(project_root) not in sys.path:
                    sys.path.append(str(project_root))
                from method.flashvid import apply_flashvid_patch

            def _as_bool(value: str, default: bool) -> bool:
                if value is None:
                    return default
                return value.strip().lower() in {"1", "true", "yes", "on"}

            retention_ratio = float(os.getenv("FLASHVID_RETENTION_RATIO", "0.25"))
            do_segment = _as_bool(os.getenv("FLASHVID_DO_SEGMENT"), True)
            segment_threshold = float(os.getenv("FLASHVID_SEGMENT_THRESHOLD", "0.9"))
            min_segment_num = int(os.getenv("FLASHVID_MIN_SEGMENT_NUM", "8"))
            complementary_segment = _as_bool(os.getenv("FLASHVID_COMPLEMENTARY_SEGMENT"), True)
            token_selection_method = os.getenv("FLASHVID_TOKEN_SELECTION_METHOD", "attn_div")
            alpha = float(os.getenv("FLASHVID_ALPHA", "0.7"))
            temporal_threshold = float(os.getenv("FLASHVID_TEMPORAL_THRESHOLD", "0.8"))
            expansion = float(os.getenv("FLASHVID_EXPANSION", "1.25"))
            pruning_layer = int(os.getenv("FLASHVID_PRUNING_LAYER", "20"))
            llm_retention_ratio = float(os.getenv("FLASHVID_LLM_RETENTION_RATIO", "0.3"))

            apply_flashvid_patch(
                self._model,
                retention_ratio=retention_ratio,
                do_segment=do_segment,
                segment_threshold=segment_threshold,
                min_segment_num=min_segment_num,
                complementary_segment=complementary_segment,
                token_selection_method=token_selection_method,
                alpha=alpha,
                temporal_threshold=temporal_threshold,
                expansion=expansion,
                pruning_layer=pruning_layer,
                llm_retention_ratio=llm_retention_ratio,
            )
            eval_logger.success(
                f"[FlashVID] Patched Qwen3-VL "
                f"(retention_ratio={retention_ratio}, do_segment={do_segment}, "
                f"segment_threshold={segment_threshold}, min_segment_num={min_segment_num}, "
                f"complementary_segment={complementary_segment}, token_selection_method={token_selection_method}, "
                f"alpha={alpha}, temporal_threshold={temporal_threshold}, expansion={expansion}, "
                f"pruning_layer={pruning_layer}, llm_retention_ratio={llm_retention_ratio})."
            )
        elif compressor == "vidcom2":
            try:
                from method.vidcom2 import apply_vidcom2_patch
            except ImportError:
                project_root = Path(__file__).resolve().parents[4]
                if str(project_root) not in sys.path:
                    sys.path.append(str(project_root))
                from method.vidcom2 import apply_vidcom2_patch

            base_scale = float(os.getenv("VIDCOM2_R_RATIO", os.getenv("R_RATIO", "0.25")))

            apply_vidcom2_patch(
                self._model,
                base_scale=base_scale,
            )
            eval_logger.success(
                f"[VidCom2] Patched Qwen3-VL forward "
                f"(base_scale={base_scale})."
            )
        elif compressor == "fastv":
            try:
                from method.fastv import apply_fastv_patch
            except ImportError:
                project_root = Path(__file__).resolve().parents[4]
                if str(project_root) not in sys.path:
                    sys.path.append(str(project_root))
                from method.fastv import apply_fastv_patch

            layer_k = int(os.getenv("FASTV_K", "2"))
            retention_ratio = float(os.getenv("FASTV_R_RATIO", os.getenv("R_RATIO", "0.5")))

            apply_fastv_patch(
                self._model,
                layer_k=layer_k,
                retention_ratio=retention_ratio,
            )
            eval_logger.success(
                f"[FastV] Patched Qwen3-VL forward "
                f"(layer_k={layer_k}, retention_ratio={retention_ratio})."
            )
        elif compressor == "visionzip":
            try:
                from method.visionzip import apply_visionzip_patch
            except ImportError:
                project_root = Path(__file__).resolve().parents[4]
                if str(project_root) not in sys.path:
                    sys.path.append(str(project_root))
                from method.visionzip import apply_visionzip_patch

            retention_ratio = float(os.getenv("VISIONZIP_R_RATIO", os.getenv("R_RATIO", "0.2")))
            dominant_ratio = float(os.getenv("VISIONZIP_DOMINANT_RATIO", "0.6"))
            k_neighbors = int(os.getenv("VISIONZIP_K_NEIGHBORS", "5"))

            apply_visionzip_patch(
                self._model,
                retention_ratio=retention_ratio,
                dominant_ratio=dominant_ratio,
                k_neighbors=k_neighbors,
            )
            eval_logger.success(
                f"[VisionZip] Patched Qwen3-VL forward "
                f"(retention_ratio={retention_ratio}, dominant_ratio={dominant_ratio}, "
                f"k_neighbors={k_neighbors})."
            )
        elif compressor == "holitom":
            try:
                from method.holitom import apply_holitom_patch
            except ImportError:
                project_root = Path(__file__).resolve().parents[4]
                if str(project_root) not in sys.path:
                    sys.path.append(str(project_root))
                from method.holitom import apply_holitom_patch

            retain_ratio = float(os.getenv("HOLITOM_R_RATIO", os.getenv("R_RATIO", "0.15")))
            tau = float(os.getenv("HOLITOM_T", "0.8"))
            beta = float(os.getenv("HOLITOM_BETA", "0.6"))
            dominant_ratio = float(os.getenv("HOLITOM_D", "0.0"))
            k_neighbors = int(os.getenv("HOLITOM_K", "7"))
            max_window_size = int(os.getenv("HOLITOM_MAX_WINDOW_SIZE", "1024"))

            apply_holitom_patch(
                self._model,
                retain_ratio=retain_ratio,
                tau=tau,
                beta=beta,
                dominant_ratio=dominant_ratio,
                k_neighbors=k_neighbors,
                max_window_size=max_window_size,
            )
            eval_logger.success(
                f"[HoliTom] Patched Qwen3-VL forward "
                f"(retain_ratio={retain_ratio}, tau={tau}, beta={beta}, "
                f"dominant_ratio={dominant_ratio}, k_neighbors={k_neighbors}, "
                f"max_window_size={max_window_size})."
            )
        elif compressor is not None:
            eval_logger.warning(
                f"[Warning] Unknown COMPRESSOR value: {compressor}. "
                "Supported values: flashvlm, flashvid, vidcom2, fastv, visionzip, holitom"
            )
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.max_num_frames = max_num_frames
        self.fps = fps

        if reasoning_prompt:
            self.reasoning_prompt = reasoning_prompt.replace("\\n", "\n")
        else:
            self.reasoning_prompt = None
        self.processor = AutoProcessor.from_pretrained(pretrained, max_pixels=max_pixels, min_pixels=min_pixels)
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained)
        self.system_prompt = system_prompt
        self.interleave_visuals = interleave_visuals

        self._config = self.model.config
        self._max_length = kwargs.get("max_length", 2048)
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
            ], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            if accelerator.distributed_type == DistributedType.FSDP:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self._rank = 0
            self._world_size = 1

    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        # returns the model, unwrapping it if using Accelerate
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood is not implemented for Qwen2.5_VL")

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    @staticmethod
    def compute_tpot(total_elapsed_time: float, ttft: float, output_token_lengths: List[int]) -> Tuple[float, float, int]:
        decode_tokens = sum(max(token_len - 1, 0) for token_len in output_token_lengths)
        decode_time = max(total_elapsed_time - ttft, 0.0)
        tpot = decode_time / decode_tokens if decode_tokens > 0 else 0.0
        return tpot, decode_time, decode_tokens

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []
        total_elapsed_time = 0.0
        total_tokens = 0
        total_ttft = 0.0
        ttft_measurements = 0
        total_decode_time = 0.0
        total_decode_tokens = 0

        def _collate(x):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            visual_list = [doc_to_visual[0](self.task_dict[t][s][i]) for t, s, i in zip(task, split, doc_id)]
            gen_kwargs = all_gen_kwargs[0]

            # Set default until or update values from gen_kwargs if present
            until = gen_kwargs.get("until", [self.tokenizer.decode(self.eot_token_id)])

            if isinstance(until, str):
                until = [until]
            elif not isinstance(until, list):
                raise ValueError(f"Expected `gen_kwargs['until']` to be of type Union[str, list], but got {type(until)}")

            # Avoid using '\n\n' as a stopper for Qwen2.5VL to prevent truncation, which can lead to incorrect results
            until = [item for item in until if item != "\n\n"]

            if isinstance(contexts, tuple):
                contexts = list(contexts)

            for i in range(len(contexts)):
                if "<image>" in contexts[i]:
                    contexts[i] = contexts[i].replace("<image>", "")

            batched_messages = []
            for i, context in enumerate(contexts):
                if "<image>" in context:
                    context = context.replace("<image>", "")

                message = [{"role": "system", "content": self.system_prompt}]
                if self.reasoning_prompt:
                    context = context.strip() + self.reasoning_prompt
                    contexts[i] = context

                processed_visuals = []
                if visual_list[i] is not None:
                    for visual in visual_list[i]:
                        if isinstance(visual, str) and visual.endswith((".mp4", ".avi", ".mov")):  # Video file
                            vr = decord.VideoReader(visual)
                            first_frame = vr[0].asnumpy()
                            height, width = first_frame.shape[:2]
                            # max_pixels = height * width
                            processed_visuals.append(
                                {
                                    "type": "video",
                                    "video": visual,
                                    "max_pixels": self.max_pixels,
                                    "min_pixels": self.min_pixels,
                                }
                            )
                        elif isinstance(visual, Image.Image):  # Handle both single and multiple images
                            processed_visuals.append(
                                {
                                    "type": "image",
                                    "image": visual,
                                    "max_pixels": self.max_pixels,
                                    "min_pixels": self.min_pixels,
                                }
                            )

                if self.interleave_visuals is False:
                    message.append(
                        {
                            "role": "user",
                            "content": processed_visuals + [{"type": "text", "text": context}],
                        }
                    )
                else:  # currently support find <image x> in the context
                    image_placeholders = re.findall(r"<image \d+>", context)
                    content_parts = []
                    text_parts = re.split(r"<image \d+>", context)
                    if text_parts[0]:
                        content_parts.append({"type": "text", "text": text_parts[0]})

                    for i, placeholder in enumerate(image_placeholders):
                        img_idx = int(re.search(r"<image (\d+)>", placeholder).group(1)) - 1
                        image_idx = min(img_idx, len(processed_visuals) - 1) if processed_visuals else 0
                        if processed_visuals and image_idx < len(processed_visuals):
                            content_parts.append(processed_visuals[image_idx])
                        if i + 1 < len(text_parts) and text_parts[i + 1]:
                            content_parts.append({"type": "text", "text": text_parts[i + 1]})

                    message.append(
                        {
                            "role": "user",
                            "content": content_parts,
                        }
                    )

                batched_messages.append(message)
            texts = self.processor.apply_chat_template(batched_messages, tokenize=False, add_generation_prompt=True)
            # TODO: refactor code to allow return_video_kwargs and return_video_metadata
            image_inputs, video_inputs = process_vision_info(
                batched_messages,
                return_video_kwargs=False,
                image_patch_size=16,
                return_video_metadata=False,
            )
            if video_inputs is not None:
                total_frames = video_inputs[0].shape[0]
                indices = np.linspace(0, total_frames - 1, self.max_num_frames, dtype=int)
                # Ensure unique indices if linspace produces duplicates for few frames
                indices = np.unique(indices)
                # Append the last frame index if not already included
                if total_frames - 1 not in indices:
                    indices = np.append(indices, total_frames - 1)
                    indices = np.unique(indices)  # Ensure uniqueness again
                video_inputs[0] = video_inputs[0][indices]
            if self.batch_size > 1:
                inputs = self.processor(
                    text=texts,
                    images=image_inputs,
                    videos=video_inputs,
                    do_resize=False,
                    padding=True,
                    padding_side="left",
                    return_tensors="pt",
                )
            else:
                inputs = self.processor(
                    text=texts,
                    images=image_inputs,
                    videos=video_inputs,
                    do_resize=False,
                    return_tensors="pt",
                )
            if self.device_map == "auto":
                inputs = inputs.to("cuda")
            else:
                inputs = inputs.to(self.device)

            # Set default generation kwargs
            default_gen_kwargs = {
                "max_new_tokens": 128,
                "temperature": 0.0,  # Set to 0 for greedy default
                "top_p": None,
                "num_beams": 1,
            }
            # Update with provided kwargs
            current_gen_kwargs = {**default_gen_kwargs, **gen_kwargs}
            pad_token_id = self.tokenizer.pad_token_id

            if current_gen_kwargs["temperature"] > 0:
                current_gen_kwargs["do_sample"] = True
            else:
                current_gen_kwargs["do_sample"] = False
                current_gen_kwargs["temperature"] = None
                current_gen_kwargs["top_p"] = None

            start_time = time.time()
            ttft_stopping_criteria = TTFTStoppingCriteria(start_time=start_time)
            cont = self.model.generate(
                **inputs,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=pad_token_id,
                do_sample=current_gen_kwargs["do_sample"],
                temperature=current_gen_kwargs["temperature"],
                top_p=current_gen_kwargs["top_p"],
                num_beams=current_gen_kwargs["num_beams"],
                max_new_tokens=current_gen_kwargs["max_new_tokens"],
                use_cache=self.use_cache,
                stopping_criteria=StoppingCriteriaList([ttft_stopping_criteria]),
            )
            end_time = time.time()

            generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, cont)]
            answers = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            output_token_lengths = [len(ids) for ids in generated_ids_trimmed]
            total_tokens += sum(output_token_lengths)

            batch_elapsed_time = end_time - start_time
            batch_ttft = ttft_stopping_criteria.get_ttft(end_time)
            _, batch_decode_time, batch_decode_tokens = self.compute_tpot(
                total_elapsed_time=batch_elapsed_time,
                ttft=batch_ttft,
                output_token_lengths=output_token_lengths,
            )
            total_elapsed_time += batch_elapsed_time
            total_ttft += batch_ttft
            ttft_measurements += 1
            total_decode_time += batch_decode_time
            total_decode_tokens += batch_decode_tokens
            for i, ans in enumerate(answers):
                for term in until:
                    if len(term) > 0:
                        ans = ans.split(term)[0]
                answers[i] = ans

            for ans, context in zip(answers, contexts):
                res.append(ans)
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), ans)
                pbar.update(1)

                # eval_logger.debug(f"Question: {context}")
                # eval_logger.debug(f"Model Response: {ans}")
            # reorder this group of results back to original unsorted form
        res = re_ords.get_original(res)

        avg_speed = total_tokens / total_elapsed_time if total_elapsed_time > 0 else 0.0
        avg_ttft = total_ttft / ttft_measurements if ttft_measurements > 0 else 0.0
        avg_tpot = total_decode_time / total_decode_tokens if total_decode_tokens > 0 else 0.0
        log_metrics(
            total_elapsed_time=total_elapsed_time,
            total_gen_tokens=total_tokens,
            avg_speed=avg_speed,
            additional_metrics={"ttft": avg_ttft, "tpot": avg_tpot, "rank": self.rank},
        )

        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")
