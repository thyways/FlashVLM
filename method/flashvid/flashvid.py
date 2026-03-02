"""FlashVID core compression utilities."""

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Optional, Tuple

import torch
from torch.nn import functional as F


class TokenSelectionMethod(str, Enum):
    ATTN = "attn"
    DIV = "div"
    ADTS = "attn_div"
    ADTS_V2 = "attn_div_v2"


@dataclass
class FlashVidConfig:
    retention_ratio: float = field(default=0.25)
    alpha: float = field(default=0.7)
    token_selection_method: str = field(default="attn_div")
    temporal_threshold: float = field(default=0.8)
    do_segment: bool = field(default=True)
    segment_threshold: float = field(default=0.9)
    min_segment_num: int = field(default=8)
    complementary_segment: bool = field(default=True)
    num_attn_div_tokens: Optional[int] = field(default=None)
    num_sttm_tokens: Optional[int] = field(default=None)
    visual_token_start_index: Optional[int] = field(default=None)
    visual_token_length: Optional[int] = field(default=None)
    expansion: float = field(default=1.25)
    pruning_layer: int = field(default=20)
    llm_retention_ratio: float = field(default=0.3)
    h: Optional[int] = field(default=None)
    w: Optional[int] = field(default=None)


def pairwise_cosine_distances(features: torch.Tensor) -> torch.Tensor:
    normed_features = features / features.norm(p=2, dim=-1, keepdim=True)
    similarities = torch.bmm(normed_features, normed_features.transpose(-1, -2))
    return 1.0 - similarities


def attn_based_token_selection(
    features: torch.Tensor,
    cls_attention: torch.Tensor,
    num_retained_tokens: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size, num_visual_tokens, feat_dim = features.shape
    k = min(max(num_retained_tokens, 1), num_visual_tokens)
    topk_indices = torch.topk(cls_attention, k=k, dim=-1).indices.sort().values
    selected_features = torch.gather(features, dim=1, index=topk_indices.unsqueeze(-1).expand(-1, -1, feat_dim))
    return selected_features, topk_indices


def div_based_token_selection(
    features: torch.Tensor,
    num_retained_tokens: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    original_features = features
    features = features.float()
    batch_size, num_visual_tokens, feat_dim = features.shape
    k = min(max(num_retained_tokens, 1), num_visual_tokens)

    dist_matrix = pairwise_cosine_distances(features)
    min_dist = torch.topk(dist_matrix, k=2, dim=1, largest=False).values[:, 1, :]

    keep_indices = torch.zeros(batch_size, k, dtype=torch.long, device=features.device)
    keep_indices[:, 0] = torch.argmax(min_dist, dim=-1)

    for idx in range(1, k):
        dist_sub_matrix = torch.gather(
            dist_matrix,
            dim=1,
            index=keep_indices[:, :idx].unsqueeze(-1).expand(-1, -1, num_visual_tokens),
        )
        min_dist = torch.min(dist_sub_matrix, dim=1).values
        min_dist.scatter_(1, keep_indices[:, :idx], -1)
        keep_indices[:, idx] = torch.argmax(min_dist, dim=-1)

    keep_indices = keep_indices.sort().values
    selected_features = torch.gather(
        original_features,
        dim=1,
        index=keep_indices.unsqueeze(-1).expand(-1, -1, feat_dim),
    )
    return selected_features, keep_indices


def attn_div_based_token_selection(
    features: torch.Tensor,
    cls_attention: torch.Tensor,
    num_retained_tokens: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    original_features = features
    features = features.float()
    cls_attention = cls_attention.float() * 1e6
    batch_size, num_visual_tokens, feat_dim = features.shape
    k = min(max(num_retained_tokens, 1), num_visual_tokens)

    dist_matrix = pairwise_cosine_distances(features)
    dist_matrix = dist_matrix * cls_attention.unsqueeze(1)

    keep_indices = torch.zeros(batch_size, k, dtype=torch.long, device=features.device)
    min_dist = torch.topk(dist_matrix, k=2, dim=1, largest=False).values[:, 1, :]
    keep_indices[:, 0] = torch.argmax(min_dist, dim=-1)

    for idx in range(1, k):
        dist_sub_matrix = torch.gather(
            dist_matrix,
            dim=1,
            index=keep_indices[:, :idx].unsqueeze(-1).expand(-1, -1, num_visual_tokens),
        )
        min_dist = torch.min(dist_sub_matrix, dim=1).values
        min_dist.scatter_(1, keep_indices[:, :idx], -1)
        keep_indices[:, idx] = torch.argmax(min_dist, dim=-1)

    keep_indices = keep_indices.sort().values
    selected_features = torch.gather(
        original_features,
        dim=1,
        index=keep_indices.unsqueeze(-1).expand(-1, -1, feat_dim),
    )
    return selected_features, keep_indices


def attn_div_v2_based_token_selection(
    features: torch.Tensor,
    cls_attention: torch.Tensor,
    num_retained_tokens: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    original_features = features
    features = features.float()
    pooled_features = features.mean(1)
    global_cls_attention = cls_attention.float() * 1e6
    batch_size, num_visual_tokens, feat_dim = features.shape
    k = min(max(num_retained_tokens, 1), num_visual_tokens)

    dist_matrix = pairwise_cosine_distances(features)
    calibration_term1 = global_cls_attention.unsqueeze(1)
    local_cls_attention = torch.einsum("b n d, c d -> b c n", features, pooled_features).mean(1)
    calibration_term2 = local_cls_attention.unsqueeze(1)
    dist_matrix = dist_matrix * calibration_term1 * calibration_term2

    keep_indices = torch.zeros(batch_size, k, dtype=torch.long, device=features.device)
    min_dist = torch.topk(dist_matrix, k=2, dim=1, largest=False).values[:, 1, :]
    keep_indices[:, 0] = torch.argmax(min_dist, dim=-1)

    for idx in range(1, k):
        dist_sub_matrix = torch.gather(
            dist_matrix,
            dim=1,
            index=keep_indices[:, :idx].unsqueeze(-1).expand(-1, -1, num_visual_tokens),
        )
        min_dist = torch.min(dist_sub_matrix, dim=1).values
        min_dist.scatter_(1, keep_indices[:, :idx], -1)
        keep_indices[:, idx] = torch.argmax(min_dist, dim=-1)

    keep_indices = keep_indices.sort().values
    selected_features = torch.gather(
        original_features,
        dim=1,
        index=keep_indices.unsqueeze(-1).expand(-1, -1, feat_dim),
    )
    return selected_features, keep_indices


ALL_TOKEN_SELECTION_METHOD = {
    TokenSelectionMethod.ATTN.value: attn_based_token_selection,
    TokenSelectionMethod.DIV.value: div_based_token_selection,
    TokenSelectionMethod.ADTS.value: attn_div_based_token_selection,
    TokenSelectionMethod.ADTS_V2.value: attn_div_v2_based_token_selection,
}


def flashvid_compression(
    video_features: torch.Tensor,
    cls_attention: torch.Tensor,
    flashvid_config: FlashVidConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    num_frames, num_visual_tokens, _ = video_features.shape

    if flashvid_config.do_segment:
        segment_lengths = segment(
            video_features=video_features.mean(1),
            segment_threshold=flashvid_config.segment_threshold,
            min_segment_num=flashvid_config.min_segment_num,
            complementary_segment=flashvid_config.complementary_segment,
        )
    else:
        segment_lengths = torch.tensor([num_frames], dtype=torch.long, device=video_features.device)

    token_budget = math.ceil(num_visual_tokens * flashvid_config.retention_ratio * flashvid_config.expansion)
    token_budget = max(token_budget, 1)
    num_attn_div_tokens = math.ceil(token_budget * flashvid_config.alpha)
    num_sttm_tokens = token_budget - num_attn_div_tokens
    flashvid_config.num_attn_div_tokens = num_attn_div_tokens
    flashvid_config.num_sttm_tokens = num_sttm_tokens

    global_indices = torch.arange(num_frames * num_visual_tokens, dtype=torch.long, device=video_features.device)
    global_indices = global_indices.view(num_frames, num_visual_tokens)

    all_segment_features = []
    all_segment_indices = []
    offset = 0
    for seg_len in segment_lengths.tolist():
        segment_features = video_features[offset : offset + seg_len]
        segment_cls_attention = cls_attention[offset : offset + seg_len]
        segment_global_indices = global_indices[offset : offset + seg_len]
        segment_out, segment_keep = segment_compression(
            segment_features=segment_features,
            segment_global_indices=segment_global_indices,
            cls_attention=segment_cls_attention,
            flashvid_config=flashvid_config,
        )
        all_segment_features.append(segment_out)
        all_segment_indices.append(segment_keep)
        offset += seg_len

    final_tokens = torch.cat(all_segment_features, dim=0)
    final_global_indices = torch.cat(all_segment_indices, dim=0)
    sorted_indices = final_global_indices.argsort()
    sorted_tokens = final_tokens[sorted_indices]
    sorted_global_indices = final_global_indices[sorted_indices]
    flashvid_config.visual_token_length = sorted_tokens.shape[0]
    return sorted_tokens, sorted_global_indices


def segment_compression(
    segment_features: torch.Tensor,
    segment_global_indices: torch.Tensor,
    cls_attention: torch.Tensor,
    flashvid_config: FlashVidConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    num_frames, num_visual_tokens, feat_dim = segment_features.shape

    num_attn_div_tokens = int(flashvid_config.num_attn_div_tokens or 0)
    num_sttm_tokens = int(flashvid_config.num_sttm_tokens or 0)
    num_attn_div_tokens = min(max(num_attn_div_tokens, 0), num_visual_tokens)

    if flashvid_config.alpha > 0 and num_attn_div_tokens > 0:
        method = ALL_TOKEN_SELECTION_METHOD.get(flashvid_config.token_selection_method)
        if method is None:
            raise ValueError(f"Unsupported token_selection_method: {flashvid_config.token_selection_method}")
        additional_kwargs = {"cls_attention": cls_attention} if "attn" in flashvid_config.token_selection_method else {}
        selected_features, selected_indices = method(
            features=segment_features,
            num_retained_tokens=num_attn_div_tokens,
            **additional_kwargs,
        )
        selected_global_indices = segment_global_indices.gather(1, index=selected_indices).view(-1)
    else:
        selected_features = segment_features.new_zeros((num_frames, 0, feat_dim))
        selected_indices = torch.zeros((num_frames, 0), dtype=torch.long, device=segment_features.device)
        selected_global_indices = torch.zeros((0,), dtype=torch.long, device=segment_features.device)

    mask = torch.ones(num_frames, num_visual_tokens, dtype=torch.bool, device=segment_features.device)
    if selected_indices.numel() > 0:
        mask.scatter_(1, selected_indices, False)

    num_other_tokens = num_sttm_tokens * num_frames
    if num_other_tokens > 0 and flashvid_config.temporal_threshold < 1.0:
        if num_frames > 1:
            temp_merged_token_list, temp_merged_indices_list = spatiotemporal_compression(
                video_features=segment_features,
                temporal_threshold=flashvid_config.temporal_threshold,
                token_mask=mask,
                flashvid_config=flashvid_config,
            )
            temp_merged_global_indices_list = [
                segment_global_indices.view(num_frames, -1)[frame_idx][temp_indices]
                for frame_idx, temp_indices in enumerate(temp_merged_indices_list)
            ]
        else:
            temp_merged_token_list = [segment_features[0]]
            temp_merged_global_indices_list = [segment_global_indices[0]]
    else:
        temp_merged_token_list = []
        temp_merged_global_indices_list = []

    all_tokens = [selected_features.view(-1, feat_dim)]
    all_global_indices = [selected_global_indices]

    if num_other_tokens > 0:
        num_current_retained_tokens = sum(len(tokens) for tokens in temp_merged_token_list)
        if num_current_retained_tokens <= 0:
            adaptive_contextual_ratio = 1.0
        else:
            adaptive_contextual_ratio = num_other_tokens / num_current_retained_tokens

        for temp_tokens, temp_global_indices in zip(temp_merged_token_list, temp_merged_global_indices_list):
            num_tokens, _ = temp_tokens.shape
            aggregated_tokens = temp_tokens
            global_token_indices = temp_global_indices
            num_clusters = math.ceil(num_tokens * adaptive_contextual_ratio)
            num_clusters = min(max(num_clusters, 0), num_tokens)
            if num_clusters > 0 and adaptive_contextual_ratio < 1.0:
                cluster_indices, cluster_center_indices = dpc_knn(
                    features=temp_tokens.unsqueeze(0),
                    num_clusters=num_clusters,
                    k=min(num_clusters, 7),
                )
                assigned_one_hot = F.one_hot(cluster_indices[0], num_classes=num_clusters).to(segment_features.dtype)
                aggregated_tokens = torch.einsum("n c, n d -> c d", assigned_one_hot, temp_tokens)
                aggregated_tokens = aggregated_tokens / assigned_one_hot.sum(dim=0).unsqueeze(-1).clamp(min=1)
                global_token_indices = temp_global_indices[cluster_center_indices[0]]
            all_tokens.append(aggregated_tokens)
            all_global_indices.append(global_token_indices)

    segment_final_tokens = torch.cat(all_tokens, dim=0)
    segment_final_global_indices = torch.cat(all_global_indices, dim=0)
    return segment_final_tokens, segment_final_global_indices


def segment(
    video_features: torch.Tensor,
    segment_threshold: float,
    min_segment_num: int,
    complementary_segment: bool = True,
) -> torch.Tensor:
    num_frames, _ = video_features.shape
    normed_video_features = video_features / video_features.norm(p=2, dim=-1, keepdim=True)
    transition_similarities = torch.sum(normed_video_features[:-1] * normed_video_features[1:], dim=-1)
    cut_indices = torch.where(transition_similarities < segment_threshold)[0]
    return additional_segment(
        cut_indices=cut_indices,
        num_frames=num_frames,
        min_segment_num=min_segment_num,
        transition_similarities=transition_similarities,
        segment_threshold=segment_threshold,
        complementary_segment=complementary_segment,
    )


def additional_segment(
    cut_indices: torch.Tensor,
    num_frames: int,
    min_segment_num: int,
    transition_similarities: torch.Tensor,
    segment_threshold: float,
    complementary_segment: bool = True,
) -> torch.Tensor:
    num_segments = cut_indices.numel() + 1
    if num_segments < min_segment_num and complementary_segment:
        num_remaining_cut_indices = min_segment_num - num_segments
        transition_similarities = transition_similarities.clone()
        transition_similarities[transition_similarities < segment_threshold] = 1.0
        k = min(num_remaining_cut_indices, transition_similarities.shape[0])
        if k > 0:
            complementary_cut_indices = torch.topk(transition_similarities, k=k, largest=False).indices
            cut_indices = torch.cat([cut_indices, complementary_cut_indices]).sort().values

    padded_cut_indices = F.pad(cut_indices, (1, 1), value=0)
    padded_cut_indices[0] = -1
    padded_cut_indices[-1] = num_frames - 1
    return torch.diff(padded_cut_indices, n=1, dim=0)


@torch.no_grad()
def dpc_knn(
    features: torch.Tensor,
    num_clusters: int,
    k: int = 7,
    valid_token_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    invalid_token_mask = ~valid_token_mask if valid_token_mask is not None else None
    batch_size, seq_len, feat_dim = features.shape
    num_clusters = min(max(num_clusters, 1), seq_len)
    k = min(max(k, 1), seq_len)

    dists = torch.cdist(features.float(), features.float()) / math.sqrt(feat_dim)
    if valid_token_mask is not None:
        dists = torch.masked_fill(dists, invalid_token_mask.unsqueeze(1).expand(-1, seq_len, -1), dists.max() + 1)

    nearest_dist = torch.topk(dists, k=k, dim=-1, largest=False).values
    density = torch.mean(-(nearest_dist**2), dim=-1).exp()
    density = density + torch.rand_like(density, device=density.device, dtype=density.dtype) * 1e-6
    if valid_token_mask is not None:
        density = torch.masked_fill(density, invalid_token_mask, 0.0)

    mask = density[:, None, :] > density[:, :, None]
    max_dist = dists.view(batch_size, -1).max(dim=-1)[0].view(-1, 1, 1)
    modified_dists = torch.where(mask, dists, max_dist)
    dist, _ = torch.min(modified_dists, dim=-1)

    score = dist * density
    cluster_center_indices = torch.topk(score, k=num_clusters, dim=-1).indices
    dists_to_centers = torch.gather(
        dists,
        dim=-1,
        index=cluster_center_indices.unsqueeze(1).expand(-1, seq_len, -1),
    )
    cluster_indices = torch.argmin(dists_to_centers, dim=-1)
    cluster_indices.scatter_(
        dim=-1,
        index=cluster_center_indices,
        src=torch.arange(num_clusters, device=cluster_indices.device).unsqueeze(0).expand(batch_size, -1),
    )
    return cluster_indices, cluster_center_indices


def spatiotemporal_compression(
    video_features: torch.Tensor,
    temporal_threshold: float,
    token_mask: torch.Tensor,
    flashvid_config: FlashVidConfig,
) -> Tuple[list[torch.Tensor], list[torch.Tensor]]:
    num_frames, num_visual_tokens, _ = video_features.shape
    lower_bound = (int(flashvid_config.num_attn_div_tokens or 0) + int(flashvid_config.num_sttm_tokens or 0)) * num_frames

    normed_video_features = video_features / video_features.norm(p=2, dim=-1, keepdim=True)
    cosine_similarities = torch.einsum("b n d, b m d -> b n m", normed_video_features[1:], normed_video_features[:-1])
    cosine_similarities[~token_mask[1:].unsqueeze(-1).expand(-1, -1, num_visual_tokens)] = -1.0
    cosine_similarities[~token_mask[:-1].unsqueeze(1).expand(-1, num_visual_tokens, -1)] = -1.0

    max_sims, max_sim_indices = torch.max(cosine_similarities, dim=-1)
    padded_max_sims = F.pad(max_sims, (0, 0, 1, 0), value=-1)
    padded_max_sim_indices = F.pad(max_sim_indices, (0, 0, 1, 0), value=-1)

    token_counts = torch.ones(num_frames, num_visual_tokens, device=video_features.device, dtype=video_features.dtype)
    mask = padded_max_sims > temporal_threshold
    retaining_token_mask = ~mask

    if retaining_token_mask.int().sum() < lower_bound:
        flat = padded_max_sims.view(-1)
        k = min((num_frames * num_visual_tokens) - lower_bound, flat.numel())
        if k > 0:
            soft_threshold = flat.topk(k=k).values[-1]
            soft_threshold = max(float(soft_threshold), -1.0 + 1e-6)
            mask = padded_max_sims > soft_threshold
            retaining_token_mask = ~mask

    for frame_idx in range(num_frames - 1, -1, -1):
        frame_features = video_features[frame_idx]
        frame_token_counts = token_counts[frame_idx]
        frame_max_sim_indices = padded_max_sim_indices[frame_idx]

        tokens_to_merge = frame_features[~mask[frame_idx]]
        to_merge_token_counts = frame_token_counts[~mask[frame_idx]]
        if tokens_to_merge.numel() > 0:
            aggregated_tokens = tokens_to_merge / to_merge_token_counts.unsqueeze(-1).to(tokens_to_merge.dtype)
            video_features[frame_idx][~mask[frame_idx]] = aggregated_tokens
            token_counts[frame_idx][~mask[frame_idx]] = 1

        other_tokens = frame_features[mask[frame_idx]]
        if other_tokens.numel() > 0 and frame_idx > 0:
            anchor_token_indices = frame_max_sim_indices[mask[frame_idx]]
            assigned_one_hot = F.one_hot(anchor_token_indices, num_classes=num_visual_tokens).to(video_features.dtype)
            aggregated_tokens = torch.einsum("m n, m d -> n d", assigned_one_hot, other_tokens)
            aggregated_token_counts = assigned_one_hot.sum(dim=0)
            video_features[frame_idx - 1] += aggregated_tokens
            token_counts[frame_idx - 1] += aggregated_token_counts
            token_counts[frame_idx][mask[frame_idx]] = 0

    final_tokens = []
    retained_token_indices = []
    for frame_idx in range(num_frames):
        frame_mask = retaining_token_mask[frame_idx] & token_mask[frame_idx]
        final_tokens.append(video_features[frame_idx][frame_mask])
        retained_token_indices.append(torch.where(frame_mask)[0])
    return final_tokens, retained_token_indices


def fastv_prune(
    hidden_states: torch.Tensor,
    causal_mask: Optional[torch.Tensor],
    attentions: Optional[torch.Tensor],
    cache_position: Optional[torch.Tensor],
    position_ids: Optional[torch.Tensor],
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    flashvid_config: FlashVidConfig,
    visual_pos_masks: Optional[torch.BoolTensor] = None,
):
    batch_size, seq_length, _ = hidden_states.shape
    device = hidden_states.device

    visual_token_start_index = flashvid_config.visual_token_start_index
    visual_token_length = flashvid_config.visual_token_length
    if visual_token_start_index is None:
        visual_token_start_index = 0
    if visual_token_length is None:
        visual_token_length = 0
    visual_token_end_index = visual_token_start_index + visual_token_length

    retention_ratio = flashvid_config.llm_retention_ratio
    num_retained_tokens = math.ceil(visual_token_length * retention_ratio)
    num_retained_tokens = max(num_retained_tokens, 1) if visual_token_length > 0 else 0

    if visual_pos_masks is None:
        visual_pos_masks = torch.zeros((batch_size, seq_length), dtype=torch.bool, device=device)
        visual_pos_masks[:, visual_token_start_index:visual_token_end_index] = True
    non_visual_pos_masks = ~visual_pos_masks

    visual_features = hidden_states[visual_pos_masks, :]
    visual_global_indices = torch.where(visual_pos_masks[0])[0]
    non_visual_global_indices = torch.where(non_visual_pos_masks[0])[0]
    attn = torch.mean(attentions[:, :, -1, :], dim=1)[visual_pos_masks]

    if num_retained_tokens > 0 and visual_features.numel() > 0:
        _, topk_indices = attn_based_token_selection(
            features=visual_features.unsqueeze(0),
            cls_attention=attn.unsqueeze(0),
            num_retained_tokens=num_retained_tokens,
        )
        topk_indices = topk_indices.squeeze(0)
        keep_indices = torch.sort(torch.cat([non_visual_global_indices, visual_global_indices[topk_indices]])).values
    else:
        keep_indices = non_visual_global_indices

    hidden_states = hidden_states[:, keep_indices]
    cache_position = keep_indices if cache_position is None else cache_position[keep_indices]
    position_ids = keep_indices.unsqueeze(0) if position_ids is None else position_ids[..., keep_indices].contiguous()
    position_embeddings = (
        position_embeddings[0][..., keep_indices, :].contiguous(),
        position_embeddings[1][..., keep_indices, :].contiguous(),
    )

    new_seq_length = hidden_states.shape[1]
    if causal_mask is not None:
        causal_mask = causal_mask[:, :, :new_seq_length, :new_seq_length]

    flashvid_config.visual_token_length = num_retained_tokens
    return hidden_states, causal_mask, position_ids, cache_position, position_embeddings, keep_indices
