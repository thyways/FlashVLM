"""VidCom2 core scoring utilities."""

from typing import List, Tuple

import torch
import torch.nn.functional as F


def select_low_var_channels(x: torch.Tensor, ratio: float = 0.5) -> torch.Tensor:
    """Select channels with low variance to emphasize informative outliers."""
    variances = x.var(dim=0, unbiased=False)
    k = int(x.shape[-1] * ratio)
    _, topk_idx = torch.topk(variances, k=k, largest=False)
    return x[:, topk_idx]


def compute_gaussian_scores(x: torch.Tensor, tpf: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute multi-scale Gaussian similarity to video-level and frame-level centers."""
    frames = x.view(-1, tpf, x.shape[-1])
    frames = F.normalize(frames, dim=-1)

    vid_center = frames.mean(dim=(0, 1), keepdim=True)
    frame_center = frames.mean(dim=1, keepdim=True)

    alphas = [2**k for k in range(-3, 2)]
    video_score = _multi_scale_gaussian(frames, vid_center, alphas)
    frame_score = _multi_scale_gaussian(frames, frame_center, alphas)
    return video_score, frame_score


def _multi_scale_gaussian(
    x: torch.Tensor,
    center: torch.Tensor,
    alphas: List[float],
) -> torch.Tensor:
    """Vectorized multi-scale Gaussian kernel similarity."""
    dist_sq = ((x - center) ** 2).sum(dim=-1)
    return sum(torch.exp(-dist_sq / (2 * alpha)) for alpha in alphas)


def compute_scales(scores: torch.Tensor, base: float, temp: float = 0.01) -> torch.Tensor:
    """Generate per-frame keep ratio with temperature-scaled softmax."""
    probs = F.softmax((scores - scores.max()) / temp, dim=0)
    scales = base * (1 + probs - probs.mean())
    return scales.clamp(max=1.0)


def select_outlier_indices(
    scores: torch.Tensor,
    scales: torch.Tensor,
    tpf: int,
) -> List[torch.Tensor]:
    """Select low-similarity (outlier) token indices for each frame."""
    ks = (scales * tpf).round().long().clamp(min=1).tolist()
    batch_indices = []
    for frame_idx, k in enumerate(ks):
        _, idx = torch.topk(scores[frame_idx], k=k, largest=False, sorted=False)
        batch_indices.append(idx.sort().values)
    return batch_indices


def _map_linear_offset(indices: List[torch.Tensor], tpf: int) -> torch.Tensor:
    """Map frame-local indices to flattened sequence indices."""
    offsets = torch.arange(len(indices), device=indices[0].device) * tpf
    return torch.cat([idx + off for idx, off in zip(indices, offsets)])
