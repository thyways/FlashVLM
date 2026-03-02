"""FlashVID adaptation for Qwen3-VL."""

import types
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers.cache_utils import Cache, DynamicCache
from transformers.masking_utils import create_causal_mask
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.processing_utils import Unpack
from transformers.utils import is_torchdynamo_compiling
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLModelOutputWithPast,
    apply_rotary_pos_emb,
    apply_rotary_pos_emb_vision,
    eager_attention_forward,
    repeat_kv,
)

from .flashvid import FlashVidConfig, fastv_prune, flashvid_compression


def _forward_vision_block_with_cls_attention(
    block,
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    attn = block.attn
    seq_length = hidden_states.shape[0]

    normed_states = block.norm1(hidden_states)
    query_states, key_states, _ = (
        attn.qkv(normed_states)
        .reshape(seq_length, 3, attn.num_heads, -1)
        .permute(1, 0, 2, 3)
        .unbind(0)
    )

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)

    query_states = query_states.transpose(0, 1).unsqueeze(0)
    key_states = key_states.transpose(0, 1).unsqueeze(0)

    lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
    cls_attention_list = []
    offset = 0
    for length in lengths:
        q_chunk = query_states[:, :, offset : offset + length, :]
        k_chunk = key_states[:, :, offset : offset + length, :]
        logits = torch.matmul(q_chunk, k_chunk.transpose(-2, -1)) / attn.head_dim**0.5
        weights = F.softmax(logits, dim=-1, dtype=torch.float32)
        cls_attention = weights[0].mean(dim=0).mean(dim=0)
        cls_attention_list.append(cls_attention)
        offset += length

    return torch.cat(cls_attention_list, dim=0)


def _spatial_merge_attention_scores(
    attention_scores: torch.Tensor,
    grid_thw: torch.Tensor,
    spatial_merge_size: int,
) -> torch.Tensor:
    merge_unit = spatial_merge_size**2
    merged_scores_list = []
    offset = 0

    for t, h, w in grid_thw.tolist():
        tokens_per_frame = h * w
        merged_h = h // spatial_merge_size
        merged_w = w // spatial_merge_size
        merged_tokens_per_frame = merged_h * merged_w

        for frame_idx in range(t):
            frame_start = offset + frame_idx * tokens_per_frame
            frame_scores = attention_scores[frame_start : frame_start + tokens_per_frame]
            frame_scores_2d = frame_scores.view(h, w)
            frame_scores_blocks = frame_scores_2d.view(
                merged_h,
                spatial_merge_size,
                merged_w,
                spatial_merge_size,
            )
            frame_scores_blocks = frame_scores_blocks.permute(0, 2, 1, 3)
            frame_scores_blocks = frame_scores_blocks.reshape(merged_tokens_per_frame, merge_unit)
            merged_scores_list.append(frame_scores_blocks.mean(dim=-1))

        offset += t * tokens_per_frame

    return torch.cat(merged_scores_list, dim=0)


def _get_video_features_with_cls_attention(
    visual_model,
    pixel_values: torch.Tensor,
    grid_thw: torch.Tensor,
) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
    hidden_states = visual_model.patch_embed(pixel_values)
    pos_embeds = visual_model.fast_pos_embed_interpolate(grid_thw)
    hidden_states = hidden_states + pos_embeds

    rotary_pos_emb = visual_model.rot_pos_emb(grid_thw)
    seq_len, _ = hidden_states.size()
    hidden_states = hidden_states.reshape(seq_len, -1)
    rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
    emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
    position_embeddings = (emb.cos(), emb.sin())

    cu_seqlens = torch.repeat_interleave(
        grid_thw[:, 1] * grid_thw[:, 2],
        grid_thw[:, 0],
    ).cumsum(dim=0, dtype=torch.int32)
    cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

    deepstack_feature_lists = []
    cls_attention_scores = None
    num_blocks = len(visual_model.blocks)
    for layer_num, blk in enumerate(visual_model.blocks):
        if layer_num == num_blocks - 1:
            cls_attention_scores = _forward_vision_block_with_cls_attention(
                blk,
                hidden_states,
                cu_seqlens,
                position_embeddings,
            )
        hidden_states = blk(
            hidden_states,
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
        )
        if layer_num in visual_model.deepstack_visual_indexes:
            deepstack_feature = visual_model.deepstack_merger_list[
                visual_model.deepstack_visual_indexes.index(layer_num)
            ](hidden_states)
            deepstack_feature_lists.append(deepstack_feature)

    hidden_states = visual_model.merger(hidden_states)
    cls_attention_scores = _spatial_merge_attention_scores(
        cls_attention_scores,
        grid_thw,
        visual_model.spatial_merge_size,
    )
    return hidden_states, deepstack_feature_lists, cls_attention_scores


def Qwen3VLModel_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Union[tuple, Qwen3VLModelOutputWithPast]:
    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if inputs_embeds is None:
        inputs_embeds = self.get_input_embeddings()(input_ids)

    image_mask = None
    video_mask = None
    deepstack_image_embeds = None
    deepstack_video_embeds = None
    video_embeds = None
    cls_attention = None

    if pixel_values is not None:
        image_features = self.get_image_features(pixel_values, image_grid_thw, return_dict=True)
        image_embeds = image_features.pooler_output
        deepstack_image_embeds = image_features.deepstack_features
        image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        image_mask, _ = self.get_placeholder_mask(
            input_ids,
            inputs_embeds=inputs_embeds,
            image_features=image_embeds,
        )
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

    compression_on = (
        pixel_values_videos is not None
        and video_grid_thw is not None
        and (past_key_values is None or past_key_values.get_seq_length() == 0)
        and inputs_embeds.shape[0] == 1
    )

    if pixel_values_videos is not None:
        if compression_on:
            video_embeds_raw, deepstack_video_embeds, cls_attention = _get_video_features_with_cls_attention(
                self.visual,
                pixel_values_videos.type(self.visual.dtype),
                video_grid_thw,
            )
            split_sizes = (video_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
            video_embeds = torch.cat(torch.split(video_embeds_raw, split_sizes), dim=0).to(
                inputs_embeds.device,
                inputs_embeds.dtype,
            )
            cls_attention = cls_attention.to(video_embeds.device)
        else:
            video_features = self.get_video_features(pixel_values_videos, video_grid_thw, return_dict=True)
            video_embeds = video_features.pooler_output
            deepstack_video_embeds = video_features.deepstack_features
            video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)

        _, video_mask = self.get_placeholder_mask(
            input_ids,
            inputs_embeds=inputs_embeds,
            video_features=video_embeds,
        )
        inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

    if position_ids is None:
        attention_mask_tensor = attention_mask if not isinstance(attention_mask, dict) else attention_mask.get(
            "full_attention", None
        )
        if attention_mask_tensor is not None and attention_mask_tensor.ndim == 4:
            attention_mask_tensor = torch.diagonal(attention_mask_tensor[:, 0], dim1=1, dim2=2)
            if attention_mask_tensor.dtype.is_floating_point:
                attention_mask_tensor = attention_mask_tensor / torch.finfo(attention_mask_tensor.dtype).min
                attention_mask_tensor = (1.0 - attention_mask_tensor).int()

        prefill_compiled_stage = is_torchdynamo_compiling() and (
            (input_ids is not None and input_ids.shape[1] != 1)
            or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
        )
        prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
            (cache_position is not None and cache_position[0] == 0)
            or (past_key_values is None or past_key_values.get_seq_length() == 0)
        )

        if (prefill_compiled_stage or prefill_noncompiled_stage) or self.rope_deltas is None:
            position_ids, rope_deltas = self.get_rope_index(
                input_ids,
                image_grid_thw,
                video_grid_thw,
                attention_mask=attention_mask_tensor,
            )
            self.rope_deltas = rope_deltas
        else:
            batch_size, seq_length, _ = inputs_embeds.shape
            delta = (
                (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                if cache_position is not None
                else 0
            )
            position_ids = torch.arange(seq_length, device=inputs_embeds.device)
            position_ids = position_ids.view(1, -1).expand(batch_size, -1)
            if cache_position is not None:
                delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
            position_ids = position_ids.add(delta)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

    flashvid_config: Optional[FlashVidConfig] = getattr(self, "_flashvid_patch_config", None)
    if compression_on and cls_attention is not None and flashvid_config is not None:
        merge_size = self.visual.spatial_merge_size
        split_sizes = (video_grid_thw.prod(-1) // merge_size**2).tolist()
        video_splits = torch.split(video_embeds, split_sizes)
        attn_splits = torch.split(cls_attention, split_sizes)
        deepstack_splits = (
            [list(torch.split(layer_embeds, split_sizes)) for layer_embeds in deepstack_video_embeds]
            if deepstack_video_embeds is not None
            else []
        )

        kept_global_indices = []
        compressed_video_chunks = []
        compressed_deepstack = [[] for _ in deepstack_splits]
        offset = 0

        for video_idx, (grid, feat, attn) in enumerate(zip(video_grid_thw, video_splits, attn_splits)):
            t, h, w = grid.tolist()
            frame_tokens = (h * w) // (merge_size**2)
            flashvid_config.h = h // merge_size
            flashvid_config.w = w // merge_size

            if t <= 0 or frame_tokens <= 0 or feat.numel() == 0 or feat.shape[0] != t * frame_tokens:
                keep_local = torch.arange(feat.shape[0], device=feat.device)
                compressed_feat = feat
            else:
                video_feat_3d = feat.view(t, frame_tokens, -1)
                attn_3d = attn.view(t, frame_tokens)
                compressed_feat, keep_local = flashvid_compression(
                    video_features=video_feat_3d,
                    cls_attention=attn_3d,
                    flashvid_config=flashvid_config,
                )

            kept_global_indices.append(keep_local + offset)
            compressed_video_chunks.append(compressed_feat)
            for layer_idx, layer_splits in enumerate(deepstack_splits):
                compressed_deepstack[layer_idx].append(layer_splits[video_idx][keep_local])
            offset += feat.shape[0]

        kept_global_indices = torch.cat(kept_global_indices, dim=0)
        video_embeds = torch.cat(compressed_video_chunks, dim=0)
        if compressed_deepstack:
            deepstack_video_embeds = [torch.cat(chunks, dim=0) for chunks in compressed_deepstack]

        video_token_positions = video_mask[..., 0][0].nonzero(as_tuple=False).squeeze(-1)
        kept_video_positions = video_token_positions[kept_global_indices]
        all_positions = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)
        non_video_positions = all_positions[~video_mask[..., 0][0]]
        keep_token_indices = torch.cat((non_video_positions, kept_video_positions)).sort().values

        def _prune_attention(attn_tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            if attn_tensor is None:
                return None
            if attn_tensor.dim() == 2:
                return attn_tensor[:, keep_token_indices]
            if attn_tensor.dim() == 4:
                return attn_tensor[:, :, keep_token_indices, :][:, :, :, keep_token_indices]
            return attn_tensor

        inputs_embeds = inputs_embeds[:, keep_token_indices, :]
        if input_ids is not None:
            input_ids = input_ids[:, keep_token_indices]
        attention_mask = (
            {k: _prune_attention(v) for k, v in attention_mask.items()}
            if isinstance(attention_mask, dict)
            else _prune_attention(attention_mask)
        )
        position_ids = position_ids[:, :, keep_token_indices]
        if cache_position is not None:
            cache_position = cache_position[keep_token_indices]

        if image_mask is not None:
            image_mask = image_mask[:, keep_token_indices, :]
        if video_mask is not None:
            video_mask = video_mask[:, keep_token_indices, :]

        inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds.to(inputs_embeds.dtype))

    if flashvid_config is not None and video_mask is not None:
        video_positions = video_mask[..., 0][0].nonzero(as_tuple=False).squeeze(-1)
        if video_positions.numel() > 0:
            flashvid_config.visual_token_start_index = int(video_positions[0].item())
            flashvid_config.visual_token_length = int(video_positions.numel())

    visual_pos_masks = None
    deepstack_visual_embeds = None
    if image_mask is not None and video_mask is not None:
        image_mask_compact = image_mask[..., 0]
        video_mask_compact = video_mask[..., 0]
        visual_pos_masks = image_mask_compact | video_mask_compact
        deepstack_visual_embeds = []
        image_mask_joint = image_mask_compact[visual_pos_masks]
        video_mask_joint = video_mask_compact[visual_pos_masks]
        for img_embed, vid_embed in zip(deepstack_image_embeds, deepstack_video_embeds):
            embed_joint = img_embed.new_zeros(visual_pos_masks.sum(), img_embed.shape[-1]).to(img_embed.device)
            embed_joint[image_mask_joint, :] = img_embed
            embed_joint[video_mask_joint, :] = vid_embed
            deepstack_visual_embeds.append(embed_joint)
    elif image_mask is not None:
        image_mask_compact = image_mask[..., 0]
        visual_pos_masks = image_mask_compact
        deepstack_visual_embeds = deepstack_image_embeds
    elif video_mask is not None:
        video_mask_compact = video_mask[..., 0]
        visual_pos_masks = video_mask_compact
        deepstack_visual_embeds = deepstack_video_embeds

    outputs = self.language_model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        cache_position=cache_position,
        visual_pos_masks=visual_pos_masks,
        deepstack_visual_embeds=deepstack_visual_embeds,
        **kwargs,
    )

    return Qwen3VLModelOutputWithPast(
        last_hidden_state=outputs.last_hidden_state,
        past_key_values=outputs.past_key_values,
        rope_deltas=self.rope_deltas,
    )


def Qwen3VLTextModel_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    visual_pos_masks: Optional[torch.Tensor] = None,
    deepstack_visual_embeds: Optional[List[torch.Tensor]] = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> Union[tuple, BaseModelOutputWithPast]:
    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if use_cache and past_key_values is None and not torch.jit.is_tracing():
        past_key_values = DynamicCache(config=self.config)

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens,
            past_seen_tokens + inputs_embeds.shape[1],
            device=inputs_embeds.device,
        )

    if position_ids is None:
        position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
    elif position_ids.ndim == 2:
        position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

    if position_ids.ndim == 3 and position_ids.shape[0] == 4:
        text_position_ids = position_ids[0]
        position_ids = position_ids[1:]
    else:
        text_position_ids = position_ids[0]

    attention_mask = create_causal_mask(
        config=self.config,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        cache_position=cache_position,
        past_key_values=past_key_values,
        position_ids=text_position_ids,
    )

    hidden_states = inputs_embeds
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    flashvid_config: FlashVidConfig = getattr(self, "_flashvid_patch_config")
    is_prefill = hidden_states.shape[1] > 1
    layer_outputs = None

    for layer_idx, decoder_layer in enumerate(self.layers):
        if is_prefill:
            if layer_idx == flashvid_config.pruning_layer - 1:
                kwargs["output_attentions"] = True
            elif layer_idx == flashvid_config.pruning_layer and layer_outputs is not None:
                kwargs["output_attentions"] = False
                attn = layer_outputs[1]
                if attn is not None:
                    (
                        hidden_states,
                        attention_mask,
                        text_position_ids,
                        cache_position,
                        position_embeddings,
                        keep_indices,
                    ) = fastv_prune(
                        hidden_states=hidden_states,
                        causal_mask=attention_mask,
                        attentions=attn,
                        cache_position=cache_position,
                        position_ids=text_position_ids,
                        position_embeddings=position_embeddings,
                        flashvid_config=flashvid_config,
                        visual_pos_masks=visual_pos_masks,
                    )
                    if visual_pos_masks is not None:
                        old_visual_positions = visual_pos_masks[0].nonzero(as_tuple=False).squeeze(-1)
                        kept_visual_mask = torch.isin(old_visual_positions, keep_indices)
                        visual_pos_masks = visual_pos_masks[:, keep_indices]
                        if deepstack_visual_embeds is not None:
                            deepstack_visual_embeds = [embeds[kept_visual_mask] for embeds in deepstack_visual_embeds]

        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=text_position_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = layer_outputs[0]

        if deepstack_visual_embeds is not None and layer_idx in range(len(deepstack_visual_embeds)):
            hidden_states = self._deepstack_process(
                hidden_states,
                visual_pos_masks,
                deepstack_visual_embeds[layer_idx],
            )

    hidden_states = self.norm(hidden_states)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values,
    )


def Qwen3VLTextDecoderLayer_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    use_cache: Optional[bool] = False,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    hidden_states, attn_weights = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        use_cache=use_cache,
        cache_position=cache_position,
        position_embeddings=position_embeddings,
        **kwargs,
    )
    hidden_states = residual + hidden_states

    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states
    return hidden_states, attn_weights


def Qwen3VLTextAttention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_values: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
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

    attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
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

    if kwargs.get("output_attentions", False) and attn_weights is None:
        last_query = query_states[:, :, -1:, :]
        expanded_key_states = repeat_kv(key_states, self.num_key_value_groups)
        attn_weights = torch.matmul(last_query, expanded_key_states.transpose(2, 3)) / self.head_dim**0.5
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


def apply_flashvid_patch(
    model,
    retention_ratio: float = 0.25,
    do_segment: bool = True,
    segment_threshold: float = 0.9,
    min_segment_num: int = 8,
    complementary_segment: bool = True,
    token_selection_method: str = "attn_div",
    alpha: float = 0.7,
    temporal_threshold: float = 0.8,
    expansion: float = 1.25,
    pruning_layer: int = 20,
    llm_retention_ratio: float = 0.3,
) -> bool:
    qwen3_vl_model = getattr(model, "model", None)
    if qwen3_vl_model is None:
        raise ValueError("Unsupported Qwen3-VL model structure for FlashVID patching.")
    language_model = getattr(qwen3_vl_model, "language_model", None)
    if language_model is None or not hasattr(language_model, "layers"):
        raise ValueError("Unsupported Qwen3-VL language model structure for FlashVID patching.")

    flashvid_config = FlashVidConfig(
        retention_ratio=float(retention_ratio),
        do_segment=bool(do_segment),
        segment_threshold=float(segment_threshold),
        min_segment_num=int(min_segment_num),
        complementary_segment=bool(complementary_segment),
        token_selection_method=str(token_selection_method),
        alpha=float(alpha),
        temporal_threshold=float(temporal_threshold),
        expansion=float(expansion),
        pruning_layer=int(pruning_layer),
        llm_retention_ratio=float(llm_retention_ratio),
    )

    qwen3_vl_model._flashvid_patch_config = flashvid_config
    language_model._flashvid_patch_config = flashvid_config

    qwen3_vl_model.forward = types.MethodType(Qwen3VLModel_forward, qwen3_vl_model)
    language_model.forward = types.MethodType(Qwen3VLTextModel_forward, language_model)

    for layer in language_model.layers:
        layer.forward = types.MethodType(Qwen3VLTextDecoderLayer_forward, layer)
        layer.self_attn.forward = types.MethodType(Qwen3VLTextAttention_forward, layer.self_attn)

    return True
