"""Frame and trajectory metrics for frozen target correspondence."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch

from .prototype import image_mask_to_patch_mask, l2_normalize


def _normalized_xy_from_mask(mask: np.ndarray) -> Optional[np.ndarray]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    height, width = mask.shape
    return np.asarray([float(xs.mean() / max(width - 1, 1)), float(ys.mean() / max(height - 1, 1))], dtype=np.float32)


def _grid_xy(grid: tuple[int, int], flat_index: int) -> np.ndarray:
    height, width = grid
    y, x = divmod(int(flat_index), width)
    return np.asarray(
        [(x + 0.5) / width, (y + 0.5) / height],
        dtype=np.float32,
    )


def _weighted_centroid(probability: torch.Tensor, grid: tuple[int, int]) -> np.ndarray:
    flat = probability.reshape(-1).float()
    height, width = grid
    yy, xx = torch.meshgrid(
        torch.arange(height, device=flat.device, dtype=flat.dtype),
        torch.arange(width, device=flat.device, dtype=flat.dtype),
        indexing="ij",
    )
    x = ((xx + 0.5) / width).reshape(-1)
    y = ((yy + 0.5) / height).reshape(-1)
    return torch.stack([(flat * x).sum(), (flat * y).sum()]).detach().cpu().numpy().astype(np.float32)


def _safe_float(value: Any) -> float:
    return float(value) if value is not None and np.isfinite(value) else float("nan")


def compute_frame_metrics(
    features: torch.Tensor,
    prototype: torch.Tensor,
    *,
    target_mask: Optional[np.ndarray] = None,
    previous_centroid: Optional[np.ndarray] = None,
    previous_gt_center: Optional[np.ndarray] = None,
    temperature: float = 0.07,
    min_patch_overlap: float = 0.5,
) -> dict[str, Any]:
    """Compute raw cosine metrics for one camera/frame.

    ``target_mask`` is optional by design.  When absent, target-localization
    metrics are NaN; the reference annotation is never reused as a moving GT
    mask.  The raw similarity map is returned separately from visualization
    normalization.
    """

    if features.ndim != 3:
        raise ValueError(f"features must be [Hpatch,Wpatch,D], got {tuple(features.shape)}")
    if prototype.ndim != 1 or prototype.shape[0] != features.shape[-1]:
        raise ValueError(
            f"prototype shape {tuple(prototype.shape)} is incompatible with features {tuple(features.shape)}"
        )
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    grid = (int(features.shape[0]), int(features.shape[1]))
    normalized_features = l2_normalize(features, dim=-1)
    normalized_prototype = l2_normalize(prototype, dim=0)
    similarity = torch.einsum("hwd,d->hw", normalized_features, normalized_prototype).float()
    flat_probability = torch.softmax(similarity.reshape(-1) / float(temperature), dim=0)
    probability = flat_probability.reshape(*grid)
    peak_index = int(torch.argmax(similarity).item())
    peak_xy = _grid_xy(grid, peak_index)
    centroid_xy = _weighted_centroid(probability, grid)
    entropy = float(-(flat_probability * torch.log(flat_probability.clamp_min(1e-12))).sum().item())

    values: dict[str, Any] = {
        "peak_similarity": float(similarity.max().item()),
        "peak_x": float(peak_xy[0]),
        "peak_y": float(peak_xy[1]),
        "centroid_x": float(centroid_xy[0]),
        "centroid_y": float(centroid_xy[1]),
        "heatmap_entropy": entropy,
        "target_mean_similarity": float("nan"),
        "background_mean_similarity": float("nan"),
        "similarity_margin": float("nan"),
        "target_mass": float("nan"),
        "peak_localization_error": float("nan"),
        "centroid_localization_error": float("nan"),
        "gt_target_center_x": float("nan"),
        "gt_target_center_y": float("nan"),
        "temporal_drift": float("nan") if previous_centroid is None else float(np.linalg.norm(centroid_xy - previous_centroid)),
        "gt_target_center_movement": float("nan") if previous_gt_center is None else float("nan"),
        "tracking_residual_drift": float("nan"),
        "raw_similarity_map": similarity.detach().cpu().numpy().astype(np.float32),
        "probability_map": probability.detach().cpu().numpy().astype(np.float32),
        "predicted_centroid": centroid_xy,
    }

    if target_mask is None:
        return values
    target_mask = np.asarray(target_mask).astype(bool)
    if target_mask.ndim != 2:
        raise ValueError(f"target_mask must be 2D, got {target_mask.shape}")
    if target_mask.shape[0] % grid[0] or target_mask.shape[1] % grid[1]:
        raise ValueError(f"target_mask shape {target_mask.shape} is not compatible with patch grid {grid}")
    target_patch_mask = image_mask_to_patch_mask(
        target_mask,
        grid_size=grid,
        min_patch_overlap=min_patch_overlap,
    )
    target_indices = torch.as_tensor(target_patch_mask.reshape(-1), dtype=torch.bool)
    background_indices = ~target_indices
    if not bool(target_indices.any()):
        raise ValueError("Evaluation GT mask covers zero DINO patches")
    target_similarity = similarity.reshape(-1)[target_indices]
    values["target_mean_similarity"] = float(target_similarity.mean().item())
    if bool(background_indices.any()):
        background_similarity = similarity.reshape(-1)[background_indices]
        values["background_mean_similarity"] = float(background_similarity.mean().item())
        values["similarity_margin"] = values["target_mean_similarity"] - values["background_mean_similarity"]
    values["target_mass"] = float(flat_probability[target_indices].sum().item())

    gt_center = _normalized_xy_from_mask(target_mask)
    if gt_center is not None:
        values["gt_target_center_x"] = float(gt_center[0])
        values["gt_target_center_y"] = float(gt_center[1])
        values["peak_localization_error"] = float(np.linalg.norm(peak_xy - gt_center))
        values["centroid_localization_error"] = float(np.linalg.norm(centroid_xy - gt_center))
        if previous_gt_center is not None:
            gt_movement = float(np.linalg.norm(gt_center - previous_gt_center))
            values["gt_target_center_movement"] = gt_movement
            if previous_centroid is not None:
                predicted_movement = centroid_xy - previous_centroid
                actual_movement = gt_center - previous_gt_center
                values["tracking_residual_drift"] = float(np.linalg.norm(predicted_movement - actual_movement))
    return values
