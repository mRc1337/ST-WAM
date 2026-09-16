"""Leakage-safe target prototype and control construction."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def l2_normalize(features: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    return F.normalize(features.float(), dim=dim, eps=eps)


def resize_binary_mask(mask: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    """Resize a source-image mask to the actual DINO input using nearest-neighbor."""

    mask = np.asarray(mask).astype(np.uint8)
    if mask.ndim != 2:
        raise ValueError(f"Target mask must be 2D, got {mask.shape}")
    height, width = image_size
    if mask.shape != (height, width):
        mask = np.asarray(
            Image.fromarray(mask * 255).resize((width, height), Image.Resampling.NEAREST)
        ) > 127
    return mask.astype(bool)


def image_mask_to_patch_mask(
    image_mask: np.ndarray,
    *,
    grid_size: tuple[int, int],
    min_patch_overlap: float = 0.5,
) -> np.ndarray:
    """Map a binary image mask to patches by fractional pixel overlap."""

    if not 0.0 <= min_patch_overlap <= 1.0:
        raise ValueError(f"min_patch_overlap must be in [0,1], got {min_patch_overlap}")
    grid_h, grid_w = grid_size
    mask = np.asarray(image_mask).astype(bool)
    if mask.ndim != 2 or mask.shape[0] % grid_h or mask.shape[1] % grid_w:
        raise ValueError(f"Mask shape {mask.shape} must be divisible by patch grid {grid_size}")
    patch_h = mask.shape[0] // grid_h
    patch_w = mask.shape[1] // grid_w
    overlap = mask.reshape(grid_h, patch_h, grid_w, patch_w).mean(axis=(1, 3))
    return overlap >= float(min_patch_overlap)


def build_target_prototype(
    reference_features: torch.Tensor,
    reference_mask: np.ndarray,
    *,
    image_size: tuple[int, int] | None = None,
    patch_size: int = 16,
    min_patch_overlap: float = 0.5,
    min_target_patches: int = 1,
) -> dict[str, Any]:
    """Build one fixed normalized prototype from one reference frame.

    The returned vector is the only representation used for subsequent
    frames.  This function intentionally has no update/track operation.
    """

    if reference_features.ndim != 3:
        raise ValueError(f"reference_features must be [Hpatch,Wpatch,D], got {tuple(reference_features.shape)}")
    grid_h, grid_w, feature_dim = reference_features.shape
    if image_size is None:
        image_size = (grid_h * int(patch_size), grid_w * int(patch_size))
    resized_mask = resize_binary_mask(np.asarray(reference_mask), image_size)
    patch_mask = image_mask_to_patch_mask(
        resized_mask,
        grid_size=(grid_h, grid_w),
        min_patch_overlap=min_patch_overlap,
    )
    patch_indices = np.flatnonzero(patch_mask.reshape(-1))
    if len(patch_indices) < int(min_target_patches):
        warnings.warn(
            f"Target reference covers only {len(patch_indices)} DINO patches; "
            f"minimum requested is {min_target_patches}.",
            RuntimeWarning,
            stacklevel=2,
        )
    if len(patch_indices) == 0:
        raise ValueError(
            "Reference target mask covers zero DINO patches. "
            "Adjust the annotation or lower min_patch_overlap."
        )
    flat = l2_normalize(reference_features.reshape(-1, feature_dim), dim=-1)
    prototype = l2_normalize(flat[torch.as_tensor(patch_indices, dtype=torch.long)].mean(dim=0), dim=0)
    return {
        "prototype": prototype,
        "patch_mask": patch_mask,
        "patch_indices": patch_indices.tolist(),
        "target_patch_count": int(len(patch_indices)),
        "feature_dim": int(feature_dim),
        "grid_size": [int(grid_h), int(grid_w)],
        "normalized": True,
        "min_patch_overlap": float(min_patch_overlap),
    }


def build_random_prototypes(
    reference_features: torch.Tensor,
    target_patch_mask: np.ndarray,
    *,
    num_prototypes: int = 1,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Build same-size background/random spatial controls.

    Controls are sampled from non-target patches when possible, so a good
    target result cannot be explained by the prototype operation alone.
    """

    if num_prototypes <= 0:
        return []
    grid_h, grid_w, feature_dim = reference_features.shape
    target_indices = np.flatnonzero(np.asarray(target_patch_mask).reshape(-1))
    candidates = np.flatnonzero(~np.asarray(target_patch_mask).reshape(-1))
    if len(candidates) == 0:
        candidates = np.arange(grid_h * grid_w)
    count = max(1, len(target_indices))
    flat = l2_normalize(reference_features.reshape(-1, feature_dim), dim=-1)
    generator = np.random.default_rng(int(seed))
    controls: list[dict[str, Any]] = []
    for index in range(int(num_prototypes)):
        replace = len(candidates) < count
        selected = generator.choice(candidates, size=count, replace=replace)
        vector = l2_normalize(flat[torch.as_tensor(selected, dtype=torch.long)].mean(dim=0), dim=0)
        controls.append(
            {
                "prototype": vector,
                "patch_indices": [int(x) for x in np.asarray(selected).reshape(-1)],
                "target_patch_count": int(count),
                "control_index": index,
                "normalized": True,
                "kind": "random_background_patch",
            }
        )
    return controls
