"""Side-channel visualization of MoT action-to-vision attention.

The functions in this module deliberately operate on the tensors immediately
before the project's SDPA call.  They are not feature-similarity or attention
rollout helpers.  The optional recorder is only instantiated by inference
when the user enables visualization, and it copies no tensors until a
configured layer/denoising step is reached.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Optional, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch


DISPLAY_NOTE = "Color is within-map relative intensity; total attention is not comparable across maps."


def _plain_value(value: Any) -> Any:
    """Convert common tensor/config values into JSON-compatible values."""

    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _plain_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _as_int_tuple(value: Any, *, name: str, length: Optional[int] = None) -> tuple[int, ...]:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a list/tuple, got {type(value).__name__}")
    result = tuple(int(item) for item in value)
    if length is not None and len(result) != length:
        raise ValueError(f"{name} must have length {length}, got {result}")
    return result


@dataclass(frozen=True)
class MotAttentionVisualizationConfig:
    """User-facing settings for one model process.

    ``layer`` and ``denoising_step`` are zero-based indices.  ``last`` is
    resolved using execution order, never by comparing timestep values.
    """

    enabled: bool = False
    output_dir: str = "./mot_attention"
    layer: int | str = "last"
    layers: Any = None
    denoising_step: int | str = "last"
    denoising_steps: Any = None
    heads: Any = "all"
    action_query_range: Any = "all"
    action_query_ranges: Any = None
    call_indices: Any = "all"
    max_saves: int = 1
    display_normalization: str = "percentile"
    display_percentiles: tuple[float, float] = (1.0, 99.0)
    overlay_alpha: float = 0.45
    input_range: tuple[float, float] = (-1.0, 1.0)
    camera_layout: str = "full"
    camera_names: tuple[str, ...] = ()
    camera_sizes: tuple[tuple[int, int], ...] = ()
    camera_regions: tuple[dict[str, Any], ...] = ()

    @classmethod
    def from_mapping(
        cls,
        value: Optional[Mapping[str, Any] | "MotAttentionVisualizationConfig"],
    ) -> "MotAttentionVisualizationConfig":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError(
                "mot_attention_visualization must be a mapping or None, "
                f"got {type(value).__name__}"
            )

        aliases = {
            "mot_layer": "layer",
            "action_denoising_step": "denoising_step",
            "record_calls": "call_indices",
            "image_range": "input_range",
        }
        values = {aliases.get(str(key), str(key)): item for key, item in value.items()}
        known = {
            "enabled",
            "output_dir",
            "layer",
            "layers",
            "denoising_step",
            "denoising_steps",
            "heads",
            "action_query_range",
            "action_query_ranges",
            "call_indices",
            "max_saves",
            "display_normalization",
            "display_percentiles",
            "overlay_alpha",
            "input_range",
            "camera_layout",
            "camera_names",
            "camera_sizes",
            "camera_regions",
        }
        unknown = sorted(set(values) - known)
        if unknown:
            raise ValueError(
                "Unknown MoT attention visualization option(s): "
                f"{unknown}. This prevents silently ignoring a misspelled setting."
            )

        input_range = values.get("input_range", (-1.0, 1.0))
        input_range_tuple = tuple(float(item) for item in input_range)
        if len(input_range_tuple) != 2 or not input_range_tuple[0] < input_range_tuple[1]:
            raise ValueError(f"input_range must be [low, high] with low < high, got {input_range}")

        normalization = str(values.get("display_normalization", "percentile")).strip().lower()
        if normalization not in {"minmax", "percentile", "none"}:
            raise ValueError(
                "display_normalization must be one of minmax, percentile, none; "
                f"got {normalization!r}"
            )
        display_percentiles = tuple(
            float(item) for item in values.get("display_percentiles", (1.0, 99.0))
        )
        if (
            len(display_percentiles) != 2
            or not 0.0 <= display_percentiles[0] < display_percentiles[1] <= 100.0
        ):
            raise ValueError(
                "display_percentiles must be [low, high] with 0 <= low < high <= 100, "
                f"got {display_percentiles}"
            )
        alpha = float(values.get("overlay_alpha", 0.45))
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"overlay_alpha must be in [0, 1], got {alpha}")
        max_saves = int(values.get("max_saves", 1))
        if max_saves < 0:
            raise ValueError(f"max_saves must be non-negative, got {max_saves}")

        camera_sizes_value = values.get("camera_sizes", ()) or ()
        camera_sizes = tuple(
            _as_int_tuple(item, name="camera_sizes entry", length=2)
            for item in camera_sizes_value
        )
        camera_regions_value = values.get("camera_regions", ()) or ()
        if not isinstance(camera_regions_value, (list, tuple)):
            raise ValueError("camera_regions must be a list of region mappings")
        camera_regions = tuple(dict(item) for item in camera_regions_value)
        for region in camera_regions:
            if not isinstance(region, dict) or "name" not in region:
                raise ValueError("each camera_regions entry must be a mapping with a name")

        return cls(
            enabled=bool(values.get("enabled", False)),
            output_dir=str(values.get("output_dir", "./mot_attention")),
            layer=values.get("layer", "last"),
            layers=values.get("layers"),
            denoising_step=values.get("denoising_step", "last"),
            denoising_steps=values.get("denoising_steps"),
            heads=values.get("heads", "all"),
            action_query_range=values.get("action_query_range", "all"),
            action_query_ranges=values.get("action_query_ranges"),
            call_indices=values.get("call_indices", "all"),
            max_saves=max_saves,
            display_normalization=normalization,
            display_percentiles=display_percentiles,
            overlay_alpha=alpha,
            input_range=input_range_tuple,
            camera_layout=str(
                "full"
                if values.get("camera_layout", "full") is None
                else values.get("camera_layout", "full")
            ),
            camera_names=tuple(str(item) for item in (values.get("camera_names", ()) or ())),
            camera_sizes=camera_sizes,
            camera_regions=camera_regions,
        )


def resolve_index(selector: Any, size: int, *, name: str) -> int:
    """Resolve an integer/``last`` selector against an execution dimension."""

    if size <= 0:
        raise ValueError(f"Cannot resolve {name} against an empty dimension")
    if isinstance(selector, str) and selector.strip().lower() == "last":
        return size - 1
    try:
        index = int(selector)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer or 'last', got {selector!r}") from exc
    if index < 0:
        index += size
    if not 0 <= index < size:
        raise ValueError(f"{name}={selector!r} is outside [0, {size})")
    return index


def resolve_indices(selector: Any, size: int, *, name: str) -> tuple[int, ...]:
    """Resolve one or more execution indices while preserving order."""

    if isinstance(selector, str):
        text = selector.strip().lower()
        if text == "all":
            return tuple(range(size))
        if "," in text:
            selector = [item.strip() for item in text.split(",") if item.strip()]
        else:
            selector = [selector]
    elif isinstance(selector, int):
        selector = [selector]
    if not isinstance(selector, (list, tuple)) or not selector:
        raise ValueError(f"{name} must be an index, 'last', 'all', or a non-empty list")
    result = tuple(resolve_index(item, size, name=name) for item in selector)
    if len(set(result)) != len(result):
        raise ValueError(f"{name} resolves to duplicate indices: {result}")
    return result


def resolve_heads(selector: Any, num_heads: int) -> tuple[int, ...]:
    if isinstance(selector, str):
        text = selector.strip().lower()
        if text == "all":
            return tuple(range(num_heads))
        if "," in text:
            selector = [item.strip() for item in text.split(",") if item.strip()]
        else:
            selector = [text]
    if isinstance(selector, int):
        selector = [selector]
    if not isinstance(selector, (list, tuple)):
        raise ValueError(f"heads must be 'all' or a list of indices, got {selector!r}")
    result = tuple(int(item) for item in selector)
    if not result:
        raise ValueError("heads cannot be empty")
    if len(set(result)) != len(result) or any(item < 0 or item >= num_heads for item in result):
        raise ValueError(f"heads={result} must be unique indices in [0, {num_heads})")
    return result


def resolve_query_range(selector: Any, query_count: int) -> tuple[int, int]:
    if query_count <= 0:
        raise ValueError("Action query count must be positive")
    if selector is None or (isinstance(selector, str) and selector.strip().lower() == "all"):
        return 0, query_count
    if isinstance(selector, str):
        match = re.fullmatch(r"\s*(-?\d+)\s*:\s*(-?\d+)\s*", selector)
        if not match:
            raise ValueError(
                "action_query_range must be 'all', [start, end), or a 'start:end' string; "
                f"got {selector!r}"
            )
        selector = [int(match.group(1)), int(match.group(2))]
    if isinstance(selector, int):
        selector = [selector, selector + 1]
    if not isinstance(selector, (list, tuple)) or len(selector) != 2:
        raise ValueError(f"action_query_range must have two values, got {selector!r}")
    start, end = (int(item) for item in selector)
    if start < 0:
        start += query_count
    if end < 0:
        end += query_count
    if not 0 <= start < end <= query_count:
        raise ValueError(
            f"action_query_range={(start, end)} must satisfy 0 <= start < end <= {query_count}"
        )
    return start, end


def resolve_query_ranges(selector: Any, query_count: int) -> tuple[tuple[int, int], ...]:
    """Resolve one or more non-overlapping action-query ranges."""

    if selector is None or isinstance(selector, (str, int)):
        return (resolve_query_range(selector, query_count),)
    if not isinstance(selector, (list, tuple)) or not selector:
        raise ValueError("action_query_ranges must be a non-empty list of ranges")
    if len(selector) == 2 and all(isinstance(item, (int, np.integer)) for item in selector):
        return (resolve_query_range(selector, query_count),)
    ranges = tuple(resolve_query_range(item, query_count) for item in selector)
    ordered = sorted(ranges)
    if len(set(ranges)) != len(ranges):
        raise ValueError(f"action_query_ranges contains duplicates: {ranges}")
    for previous, current in zip(ordered, ordered[1:]):
        if current[0] < previous[1]:
            raise ValueError(f"action_query_ranges overlap: {ranges}")
    return ranges


def recompute_attention_weights(
    q: torch.Tensor,
    k: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    num_heads: int,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Recompute SDPA weights from post-projection Q/K and the real mask.

    ``q`` and ``k`` are the flattened tensors passed to the project's
    ``flash_attention`` wrapper, i.e. ``[B, sequence, heads * head_dim]``.
    The returned tensor is ``[B, heads, query_sequence, key_sequence]``.
    Boolean masks follow PyTorch SDPA semantics: ``True`` means the key is
    visible.  Floating masks are treated as additive logits, also matching
    SDPA.  Softmax is always over the complete key sequence.
    """

    if q.ndim != 3 or k.ndim != 3:
        raise ValueError(f"q and k must be [B,S,H*D], got {tuple(q.shape)} and {tuple(k.shape)}")
    if q.shape[0] != k.shape[0] or q.shape[-1] != k.shape[-1]:
        raise ValueError(f"q/k batch and hidden dimensions must match, got {tuple(q.shape)} and {tuple(k.shape)}")
    if num_heads <= 0 or q.shape[-1] % num_heads != 0:
        raise ValueError(f"num_heads={num_heads} does not divide q hidden size {q.shape[-1]}")
    query_len, key_len = int(q.shape[1]), int(k.shape[1])
    head_dim = int(q.shape[-1]) // int(num_heads)
    if attention_mask.ndim not in {2, 3, 4}:
        raise ValueError(
            "attention_mask must be 2D, 3D, or 4D and broadcastable to [B,H,Q,K], "
            f"got shape {tuple(attention_mask.shape)}"
        )
    if tuple(attention_mask.shape[-2:]) != (query_len, key_len):
        raise ValueError(
            "attention_mask query/key dimensions must match q/k: "
            f"mask={tuple(attention_mask.shape[-2:])}, q/k={(query_len, key_len)}"
        )

    # flash_attention uses F.scaled_dot_product_attention without an explicit
    # scale, whose default is 1/sqrt(head_dim).  Keep this explicit for the
    # side-channel calculation.
    actual_scale = float(1.0 / math.sqrt(head_dim) if scale is None else scale)
    q_heads = q.detach().float().reshape(q.shape[0], query_len, num_heads, head_dim).transpose(1, 2)
    k_heads = k.detach().float().reshape(k.shape[0], key_len, num_heads, head_dim).transpose(1, 2)
    logits = torch.matmul(q_heads, k_heads.transpose(-2, -1)) * actual_scale

    mask = attention_mask.to(device=logits.device)
    if mask.ndim == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.ndim == 3:
        mask = mask.unsqueeze(1)
    if mask.dtype == torch.bool:
        logits = logits.masked_fill(~mask, float("-inf"))
    elif torch.is_floating_point(mask):
        logits = logits + mask.float()
    else:
        raise TypeError(
            "attention_mask must be bool or floating-point to match SDPA, "
            f"got {attention_mask.dtype}"
        )

    weights = torch.softmax(logits, dim=-1)
    # An all-masked row is invalid for the model but should not turn a saved
    # diagnostic into NaNs.  Valid model rows retain exact softmax values.
    return torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)


def _validate_spans(key_spans: Mapping[str, Sequence[int]], key_count: int) -> dict[str, tuple[int, int]]:
    normalized: dict[str, tuple[int, int]] = {}
    for name, span in key_spans.items():
        if len(span) != 2:
            raise ValueError(f"Key span {name!r} must be [start, end), got {span}")
        start, end = int(span[0]), int(span[1])
        if not 0 <= start < end <= key_count:
            raise ValueError(f"Key span {name!r}={(start, end)} is outside [0, {key_count}]")
        normalized[str(name)] = (start, end)
    if not normalized:
        raise ValueError("At least one key span is required")
    ordered = sorted(normalized.items(), key=lambda item: item[1][0])
    previous_end = 0
    for name, (start, end) in ordered:
        if start < previous_end:
            raise ValueError(f"Key spans overlap at {name!r}: {normalized}")
        previous_end = end
    return normalized


def aggregate_attention(
    weights: torch.Tensor,
    key_spans: Mapping[str, Sequence[int]],
    *,
    head_indices: Sequence[int],
    query_range: Sequence[int],
    target_names: Sequence[str] = ("dino", "vae"),
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor]:
    """Aggregate selected heads/queries and retain complete-key mass checks."""

    if weights.ndim != 4:
        raise ValueError(f"weights must be [B,H,Q,K], got {tuple(weights.shape)}")
    start_q, end_q = int(query_range[0]), int(query_range[1])
    if not 0 <= start_q < end_q <= weights.shape[2]:
        raise ValueError(f"Invalid query range {query_range} for {weights.shape[2]} queries")
    spans = _validate_spans(key_spans, int(weights.shape[-1]))
    heads = torch.as_tensor(tuple(int(item) for item in head_indices), device=weights.device)
    if heads.numel() == 0 or torch.any(heads < 0) or torch.any(heads >= weights.shape[1]):
        raise ValueError(f"Invalid head selection {head_indices} for {weights.shape[1]} heads")

    selected = weights.index_select(1, heads)[:, :, start_q:end_q, :]
    scores: dict[str, torch.Tensor] = {}
    for name in target_names:
        if name not in spans:
            raise ValueError(
                f"Visualization requires key span {name!r}; got {sorted(spans)}. "
                "This model/branch layout is not supported for this diagnostic."
            )
        start, end = spans[name]
        scores[name] = selected[..., start:end].mean(dim=(1, 2))

    masses = {
        name: selected[..., start:end].sum(dim=-1).mean(dim=(1, 2))
        for name, (start, end) in spans.items()
    }
    total_mass = torch.stack(tuple(masses.values()), dim=0).sum(dim=0)
    return scores, masses, total_mass


def aggregate_conditional_attention(
    weights: torch.Tensor,
    key_spans: Mapping[str, Sequence[int]],
    *,
    head_indices: Sequence[int],
    query_range: Sequence[int],
    target_names: Sequence[str] = ("dino", "vae"),
    min_branch_mass: float = 1e-8,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Average per-head/query spatial distributions conditioned on a branch.

    Each selected head/query row is normalized over the target branch before
    averaging. Rows carrying effectively no mass for that branch are excluded
    so numerical noise cannot be amplified into a spatial pattern.
    """

    if weights.ndim != 4:
        raise ValueError(f"weights must be [B,H,Q,K], got {tuple(weights.shape)}")
    start_q, end_q = int(query_range[0]), int(query_range[1])
    if not 0 <= start_q < end_q <= weights.shape[2]:
        raise ValueError(f"Invalid query range {query_range} for {weights.shape[2]} queries")
    spans = _validate_spans(key_spans, int(weights.shape[-1]))
    heads = torch.as_tensor(tuple(int(item) for item in head_indices), device=weights.device)
    if heads.numel() == 0 or torch.any(heads < 0) or torch.any(heads >= weights.shape[1]):
        raise ValueError(f"Invalid head selection {head_indices} for {weights.shape[1]} heads")

    selected = weights.index_select(1, heads)[:, :, start_q:end_q, :]
    scores: dict[str, torch.Tensor] = {}
    valid_fractions: dict[str, torch.Tensor] = {}
    for name in target_names:
        if name not in spans:
            raise ValueError(f"Visualization requires key span {name!r}; got {sorted(spans)}")
        start, end = spans[name]
        branch = selected[..., start:end]
        branch_mass = branch.sum(dim=-1, keepdim=True)
        valid = branch_mass > float(min_branch_mass)
        normalized = torch.where(valid, branch / branch_mass.clamp_min(float(min_branch_mass)), 0.0)
        valid_count = valid.sum(dim=(1, 2)).clamp_min(1)
        scores[name] = normalized.sum(dim=(1, 2)) / valid_count
        valid_fractions[name] = valid.float().mean(dim=(1, 2, 3))
    return scores, valid_fractions


def normalize_heatmap(
    values: np.ndarray,
    mode: str = "percentile",
    percentiles: Sequence[float] = (1.0, 99.0),
) -> tuple[np.ndarray, dict[str, Any]]:
    """Normalize only for display, with finite/constant-map handling."""

    array = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(array)
    if not finite.any():
        return np.zeros_like(array, dtype=np.float32), {
            "mode": mode,
            "finite": False,
            "constant": True,
            "min": None,
            "max": None,
        }
    valid = array[finite]
    low = float(np.min(valid))
    high = float(np.max(valid))
    result = np.zeros_like(array, dtype=np.float32)
    if mode == "none":
        result[finite] = array[finite]
        return result, {"mode": mode, "finite": True, "constant": False, "min": low, "max": high}
    if mode == "percentile":
        percentile_low, percentile_high = (float(item) for item in percentiles)
        if not 0.0 <= percentile_low < percentile_high <= 100.0:
            raise ValueError(f"percentiles must satisfy 0 <= low < high <= 100, got {percentiles}")
        low = float(np.percentile(valid, percentile_low))
        high = float(np.percentile(valid, percentile_high))
    if not np.isfinite(low) or not np.isfinite(high) or high - low <= max(1e-12, abs(low) * 1e-12):
        result[finite] = 0.0
        return result, {"mode": mode, "finite": True, "constant": True, "min": low, "max": high}
    result[finite] = np.clip((array[finite] - low) / (high - low), 0.0, 1.0)
    return result, {"mode": mode, "finite": True, "constant": False, "min": low, "max": high}


def _heatmap_rgb(normalized: np.ndarray) -> np.ndarray:
    # A small dependency-free blue -> cyan -> yellow -> red colormap.
    anchors = np.asarray(
        [[8, 24, 88], [25, 120, 210], [60, 205, 180], [250, 225, 60], [180, 20, 30]],
        dtype=np.float32,
    )
    x = np.clip(np.nan_to_num(normalized, nan=0.0), 0.0, 1.0)
    positions = np.linspace(0.0, 1.0, len(anchors))
    channels = [np.interp(x, positions, anchors[:, channel]) for channel in range(3)]
    return np.stack(channels, axis=-1).round().astype(np.uint8)


def _resize_map(values: np.ndarray, size_hw: tuple[int, int]) -> np.ndarray:
    height, width = int(size_hw[0]), int(size_hw[1])
    image = Image.fromarray(np.clip(values * 255.0, 0.0, 255.0).astype(np.uint8), mode="L")
    return np.asarray(image.resize((width, height), resample=Image.BILINEAR), dtype=np.float32) / 255.0


def tensor_to_rgb_array(value: Any, input_range: Sequence[float] = (-1.0, 1.0)) -> np.ndarray:
    """Convert a CHW/HWC tensor or array into uint8 RGB without changing layout."""

    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().float().numpy()
    array = np.asarray(value)
    if array.ndim == 4:
        array = array[0]
    if array.ndim != 3:
        raise ValueError(f"RGB image must be 3D/4D, got shape {array.shape}")
    if array.shape[0] in {1, 3, 4} and array.shape[-1] not in {1, 3, 4}:
        array = np.transpose(array, (1, 2, 0))
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    if array.shape[-1] == 4:
        array = array[..., :3]
    if array.shape[-1] != 3:
        raise ValueError(f"RGB image must have 3 channels, got shape {array.shape}")
    if array.dtype == np.uint8:
        return np.ascontiguousarray(array)
    low, high = float(input_range[0]), float(input_range[1])
    if not low < high:
        raise ValueError(f"input_range must have low < high, got {input_range}")
    array = np.nan_to_num(array.astype(np.float32), nan=low, posinf=high, neginf=low)
    array = np.clip((array - low) / (high - low), 0.0, 1.0)
    return np.round(array * 255.0).astype(np.uint8)


def scores_to_spatial(scores: Any, visual_meta: Mapping[str, Any]) -> np.ndarray:
    """Restore a first-frame score vector using the expert's actual grid metadata."""

    if isinstance(scores, torch.Tensor):
        scores = scores.detach().cpu().float().numpy()
    array = np.asarray(scores, dtype=np.float32)
    if array.ndim == 1:
        array = array[None, :]
    if array.ndim != 2:
        raise ValueError(f"scores must be [B,N] or [N], got shape {array.shape}")

    grid_size = _as_int_tuple(visual_meta.get("grid_size"), name="grid_size", length=3)
    _, grid_h, grid_w = grid_size
    mode = str(visual_meta.get("latent_patch_mode", "flat")).lower()
    views = int(visual_meta.get("latent_num_views", 1)) if mode == "view" else 1
    if views <= 0:
        raise ValueError(f"latent_num_views must be positive, got {views}")
    expected_tokens = int(grid_h) * int(grid_w) * views
    tokens_per_frame = int(visual_meta.get("tokens_per_frame", expected_tokens))
    if tokens_per_frame != expected_tokens or array.shape[1] != expected_tokens:
        raise ValueError(
            "Score count does not match the actual expert token grid: "
            f"scores={array.shape[1]}, expected={expected_tokens}, meta={dict(visual_meta)}"
        )

    if mode == "flat":
        grid = array.reshape(array.shape[0], grid_h, grid_w)
    elif mode == "view":
        per_view = array.reshape(array.shape[0], views, grid_h, grid_w)
        grid = np.concatenate([per_view[:, index] for index in range(views)], axis=-1)
    else:
        raise ValueError(f"Unsupported latent_patch_mode for visualization: {mode!r}")

    patch_size = _as_int_tuple(
        visual_meta.get("latent_patch_size", (1, 1, 1)),
        name="latent_patch_size",
        length=3,
    )
    patch_h, patch_w = int(patch_size[1]), int(patch_size[2])
    original_grid = visual_meta.get("original_grid_size")
    if original_grid is not None:
        original = _as_int_tuple(original_grid, name="original_grid_size", length=3)
        target_h, target_w = int(original[1]), int(original[2])
    else:
        target_h, target_w = grid_h * patch_h, grid_w * patch_w
    if target_h < grid.shape[1] or target_w < grid.shape[2]:
        raise ValueError(
            f"Original spatial grid {(target_h, target_w)} is smaller than token grid {grid.shape[1:]}"
        )

    # Expanding merged token cells before RGB interpolation preserves the
    # rectangular receptive field, including asymmetric patch merging.
    repeat_h = target_h // grid.shape[1]
    repeat_w = target_w // grid.shape[2]
    if repeat_h * grid.shape[1] != target_h or repeat_w * grid.shape[2] != target_w:
        raise ValueError(
            "Actual visual grid is not an integer expansion of the token grid: "
            f"token={grid.shape[1:]}, target={(target_h, target_w)}"
        )
    return np.repeat(np.repeat(grid, repeat_h, axis=1), repeat_w, axis=2)


def camera_grid_slices(
    grid_hw: Sequence[int],
    *,
    layout: str,
    camera_names: Sequence[str],
    camera_sizes: Sequence[Sequence[int]],
    camera_regions: Sequence[Mapping[str, Any]] = (),
) -> dict[str, tuple[slice, slice]]:
    """Return camera regions in a spatial grid using explicit image geometry.

    For horizontal/vertical layouts the boundaries are proportional to the
    actual post-preprocessing camera sizes.  It intentionally refuses to
    infer equal halves when those sizes are absent.  ``camera_regions`` can
    describe non-rectangular layouts such as RoboTwin in model-input pixels.
    """

    height, width = int(grid_hw[0]), int(grid_hw[1])
    names = tuple(str(item) for item in camera_names)
    if len(names) <= 1:
        return {names[0] if names else "camera0": (slice(0, height), slice(0, width))}
    if len(camera_sizes) != len(names):
        raise ValueError(
            "camera_sizes must be provided for multi-camera visualization; "
            f"got {len(camera_sizes)} sizes for {len(names)} cameras"
        )
    layout_key = str(layout).strip().lower()
    result: dict[str, tuple[slice, slice]] = {}
    if layout_key in {"horizontal", "vertical"}:
        axis_lengths = [int(size[1] if layout_key == "horizontal" else size[0]) for size in camera_sizes]
        if any(length <= 0 for length in axis_lengths):
            raise ValueError(f"camera_sizes must be positive, got {camera_sizes}")
        total = sum(axis_lengths)
        boundaries = [0]
        for length in axis_lengths:
            boundaries.append(round(boundaries[-1] + (length / total) * (width if layout_key == "horizontal" else height)))
        if boundaries[-1] != (width if layout_key == "horizontal" else height):
            boundaries[-1] = width if layout_key == "horizontal" else height
        for name, start, end in zip(names, boundaries[:-1], boundaries[1:]):
            if end <= start:
                raise ValueError(f"Camera {name!r} collapsed in grid {grid_hw}; sizes={camera_sizes}")
            if layout_key == "horizontal":
                result[name] = (slice(0, height), slice(start, end))
            else:
                result[name] = (slice(start, end), slice(0, width))
        return result

    if not camera_regions:
        raise ValueError(
            f"camera layout {layout!r} needs explicit camera_regions for multi-camera mapping; "
            "do not assume a one-dimensional token split"
        )
    for region in camera_regions:
        name = str(region["name"])
        x0, y0 = float(region.get("x", 0)), float(region.get("y", 0))
        region_w, region_h = float(region["width"]), float(region["height"])
        result[name] = (
            slice(round(y0 / float(camera_sizes[0][0]) * height), round((y0 + region_h) / float(camera_sizes[0][0]) * height)),
            slice(round(x0 / float(camera_sizes[0][1]) * width), round((x0 + region_w) / float(camera_sizes[0][1]) * width)),
        )
    if set(result) != set(names):
        raise ValueError(f"camera_regions names {sorted(result)} do not match camera_names {sorted(names)}")
    return result


def _safe_file_part(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "item"


def _overlay(rgb: np.ndarray, normalized: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    resized = _resize_map(normalized, rgb.shape[:2])
    heatmap_resized = _heatmap_rgb(resized)
    overlay = (rgb.astype(np.float32) * (1.0 - alpha) + heatmap_resized.astype(np.float32) * alpha).round()
    return np.clip(overlay, 0, 255).astype(np.uint8), heatmap_resized


def _compose_original_observation(
    original_arrays: Mapping[str, np.ndarray],
    config: MotAttentionVisualizationConfig,
) -> Optional[np.ndarray]:
    """Compose raw camera images with the explicit model-input camera layout."""

    names = tuple(config.camera_names)
    if not names:
        if len(original_arrays) == 1:
            return next(iter(original_arrays.values()))
        return None
    if len(names) == 1:
        return original_arrays.get(names[0])
    if len(config.camera_sizes) != len(names):
        return None
    ordered_arrays = list(original_arrays.values())
    images: list[np.ndarray] = []
    for index, (name, size) in enumerate(zip(names, config.camera_sizes)):
        image = original_arrays.get(name)
        if image is None and index < len(ordered_arrays):
            # Some dataset configs rename the cameras (for example front/wrist)
            # while the simulator helper exposes image/wrist_image.  The
            # helper's insertion order is the authoritative camera order.
            image = ordered_arrays[index]
        if image is None:
            return None
        height, width = int(size[0]), int(size[1])
        images.append(np.asarray(Image.fromarray(image).resize((width, height), resample=Image.BILINEAR)))
    layout = str(config.camera_layout).strip().lower()
    if layout == "horizontal":
        return np.concatenate(images, axis=1)
    if layout == "vertical":
        return np.concatenate(images, axis=0)
    return None


def _render_comparison(
    rgb: np.ndarray,
    panels: Sequence[tuple[np.ndarray, str]],
    *,
    observation_label: str,
) -> Image.Image:
    labeled_panels = [(rgb, observation_label), *panels]
    panel_width = max(int(image.shape[1]) for image, _ in labeled_panels)
    panel_height = max(int(image.shape[0]) for image, _ in labeled_panels)
    label_height = 32
    footer_height = 34
    canvas = Image.new(
        "RGB",
        (panel_width * len(labeled_panels), label_height + panel_height + footer_height),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for index, (image, label) in enumerate(labeled_panels):
        image_pil = Image.fromarray(image).resize((panel_width, panel_height), resample=Image.BILINEAR)
        x = index * panel_width
        canvas.paste(image_pil, (x, label_height))
        draw.text((x + 6, 8), label, fill="black", font=font)
    draw.text(
        (8, label_height + panel_height + 8),
        DISPLAY_NOTE + " Conditional maps normalize each valid head/query within its branch.",
        fill="black",
        font=font,
    )
    return canvas


class MotAttentionRecorder:
    """Capture selected layer/denoising-step attention matrices for one call."""

    def __init__(
        self,
        config: MotAttentionVisualizationConfig | Mapping[str, Any] | None,
        *,
        call_index: int,
        num_layers: int,
        num_heads: int,
        already_saved: int = 0,
    ):
        self.config = MotAttentionVisualizationConfig.from_mapping(config)
        self.call_index = int(call_index)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        layer_selector = self.config.layers if self.config.layers is not None else self.config.layer
        self.target_layers = resolve_indices(layer_selector, self.num_layers, name="layers")
        self.target_layer = self.target_layers[0]
        self.head_indices = resolve_heads(self.config.heads, self.num_heads)
        self.target_steps: tuple[int, ...] = ()
        self.target_step: Optional[int] = None
        self.records: dict[tuple[int, int], dict[str, Any]] = {}
        self.record: Optional[dict[str, Any]] = None
        self._saved = False
        self.active = bool(
            self.config.enabled
            and self.config.max_saves > int(already_saved)
            and self._call_is_selected(self.call_index)
        )

    def _call_is_selected(self, call_index: int) -> bool:
        selector = self.config.call_indices
        if selector is None or (isinstance(selector, str) and selector.strip().lower() == "all"):
            return True
        if isinstance(selector, int):
            selector = [selector]
        if isinstance(selector, str):
            selector = [item.strip() for item in selector.split(",") if item.strip()]
        if not isinstance(selector, (list, tuple)):
            raise ValueError(f"call_indices must be 'all' or a list, got {selector!r}")
        return int(call_index) in {int(item) for item in selector}

    def configure_steps(self, num_steps: int) -> None:
        if not self.active:
            return
        step_selector = (
            self.config.denoising_steps
            if self.config.denoising_steps is not None
            else self.config.denoising_step
        )
        self.target_steps = resolve_indices(step_selector, int(num_steps), name="denoising_steps")
        self.target_step = self.target_steps[0]

    def should_capture(self, *, layer_idx: int, step_idx: int) -> bool:
        return bool(
            self.active
            and int(layer_idx) in self.target_layers
            and int(step_idx) in self.target_steps
            and (int(step_idx), int(layer_idx)) not in self.records
        )

    @staticmethod
    def _timestep_value(timestep: Any) -> Any:
        if timestep is None:
            return None
        if isinstance(timestep, torch.Tensor):
            return timestep.detach().float().cpu().reshape(-1).tolist()
        return _plain_value(timestep)

    def capture(
        self,
        *,
        q_action: torch.Tensor,
        k_all: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_idx: int,
        step_idx: int,
        timestep: Any,
        key_spans: Mapping[str, Sequence[int]],
        visual_meta: Mapping[str, Mapping[str, Any]],
        scale: Optional[float] = None,
    ) -> bool:
        if not self.should_capture(layer_idx=layer_idx, step_idx=step_idx):
            return False
        weights = recompute_attention_weights(
            q_action,
            k_all,
            attention_mask,
            num_heads=self.num_heads,
            scale=scale,
        )
        query_selector = (
            self.config.action_query_ranges
            if self.config.action_query_ranges is not None
            else self.config.action_query_range
        )
        query_ranges = resolve_query_ranges(query_selector, int(weights.shape[2]))
        normalized_spans = _validate_spans(key_spans, int(weights.shape[-1]))
        query_groups = []
        for query_range in query_ranges:
            scores, masses, total_mass = aggregate_attention(
                weights,
                normalized_spans,
                head_indices=self.head_indices,
                query_range=query_range,
                target_names=("dino", "vae"),
            )
            conditional_scores, conditional_valid_fractions = aggregate_conditional_attention(
                weights,
                normalized_spans,
                head_indices=self.head_indices,
                query_range=query_range,
                target_names=("dino", "vae"),
            )
            query_groups.append(
                {
                    "query_range": list(query_range),
                    "scores_absolute": {
                        name: value.detach().float().cpu() for name, value in scores.items()
                    },
                    "scores_conditional": {
                        name: value.detach().float().cpu()
                        for name, value in conditional_scores.items()
                    },
                    "conditional_valid_fractions": {
                        name: value.detach().float().cpu()
                        for name, value in conditional_valid_fractions.items()
                    },
                    "key_masses": {
                        name: value.detach().float().cpu() for name, value in masses.items()
                    },
                    "mass_all_keys": total_mass.detach().float().cpu(),
                }
            )
        record = {
            "attention": weights[:, list(self.head_indices), :, :].detach().float().cpu(),
            "attention_mask": attention_mask.detach().cpu().clone(),
            "query_groups": query_groups,
            "key_spans": normalized_spans,
            "visual_meta": {name: _plain_value(meta) for name, meta in visual_meta.items()},
            "layer_idx": int(layer_idx),
            "step_idx": int(step_idx),
            "timestep": self._timestep_value(timestep),
            "head_indices": list(self.head_indices),
            "num_heads": self.num_heads,
            "scale": float(1.0 / math.sqrt(q_action.shape[-1] // self.num_heads) if scale is None else scale),
            "mask_semantics": "bool True means visible; softmax over complete key sequence",
        }
        self.records[(int(step_idx), int(layer_idx))] = record
        if self.record is None:
            self.record = record
        return True

    def save(
        self,
        *,
        input_image: Any,
        original_images: Optional[Mapping[str, Any]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Optional[Path | list[Path]]:
        if not self.active:
            return None
        if not self.records:
            raise RuntimeError(
                "MoT attention visualization was enabled but none of the requested layer/step "
                "pairs executed. Check the resolved inference schedule and layer indices."
            )
        if self._saved:
            raise RuntimeError("This MoT attention recorder has already been saved")

        root = Path(self.config.output_dir).expanduser()
        rgb = tensor_to_rgb_array(input_image, self.config.input_range)
        original_arrays: dict[str, np.ndarray] = {}
        if original_images:
            for name, value in original_images.items():
                try:
                    original_rgb = tensor_to_rgb_array(value, (0.0, 255.0))
                except (TypeError, ValueError):
                    continue
                original_arrays[str(name)] = original_rgb
        original_observation = _compose_original_observation(original_arrays, self.config)
        saved_dirs = [
            self._save_record(
                record,
                root=root,
                rgb=rgb,
                original_arrays=original_arrays,
                original_observation=original_observation,
                metadata=metadata,
            )
            for _, record in sorted(self.records.items())
        ]
        self._saved = True
        return saved_dirs[0] if len(saved_dirs) == 1 else saved_dirs

    def _save_record(
        self,
        record: Mapping[str, Any],
        *,
        root: Path,
        rgb: np.ndarray,
        original_arrays: Mapping[str, np.ndarray],
        original_observation: Optional[np.ndarray],
        metadata: Optional[Mapping[str, Any]],
    ) -> Path:
        record_dir = root / f"call_{self.call_index:06d}" / (
            f"step_{record['step_idx']:04d}_layer_{record['layer_idx']:04d}"
        )
        record_dir.mkdir(parents=True, exist_ok=True)
        asset_dir = record_dir if len(self.records) == 1 else record_dir.parent
        asset_prefix = "" if asset_dir == record_dir else "../"
        rgb_input_path = asset_dir / "rgb_input.png"
        if not rgb_input_path.exists():
            Image.fromarray(rgb).save(rgb_input_path)
        original_files: dict[str, str] = {}
        for name, original_rgb in original_arrays.items():
            filename = f"rgb_original_{_safe_file_part(name)}.png"
            path = asset_dir / filename
            if not path.exists():
                Image.fromarray(original_rgb).save(path)
            original_files[str(name)] = asset_prefix + filename
        if not original_files:
            path = asset_dir / "rgb_original_model_input.png"
            if not path.exists():
                Image.fromarray(rgb).save(path)
            original_files["model_input"] = asset_prefix + "rgb_original_model_input.png"
        if original_observation is not None:
            path = asset_dir / "rgb_original_concatenated.png"
            if not path.exists():
                Image.fromarray(original_observation).save(path)
            original_files["concatenated"] = asset_prefix + "rgb_original_concatenated.png"

        comparison_rgb = original_observation if original_observation is not None else rgb
        npz_payload = {
            "attention_selected_heads_queries": record["attention"].numpy(),
            "attention_mask": record["attention_mask"].numpy(),
        }
        query_group_metadata = []
        comparison_files: dict[str, str] = {}
        for group_index, group in enumerate(record["query_groups"]):
            start, end = (int(item) for item in group["query_range"])
            query_id = f"q{start:03d}_{end:03d}"
            overlays: dict[tuple[str, str], np.ndarray] = {}
            normalized_maps: dict[tuple[str, str], np.ndarray] = {}
            normalization_metadata: dict[str, dict[str, Any]] = {}
            files: dict[str, str] = {}
            for aggregation_name, score_key in (
                ("absolute", "scores_absolute"),
                ("conditional", "scores_conditional"),
            ):
                normalization_metadata[aggregation_name] = {}
                for branch in ("dino", "vae"):
                    scores = group[score_key][branch].numpy()
                    spatial = scores_to_spatial(scores, record["visual_meta"][branch])[0]
                    normalized, norm_meta = normalize_heatmap(
                        spatial,
                        self.config.display_normalization,
                        self.config.display_percentiles,
                    )
                    overlay, _ = _overlay(rgb, normalized, self.config.overlay_alpha)
                    overlays[(aggregation_name, branch)] = overlay
                    normalized_maps[(aggregation_name, branch)] = normalized
                    normalization_metadata[aggregation_name][branch] = norm_meta
                    npz_payload[f"scores_{branch}_{aggregation_name}_{query_id}"] = scores

            mass_dino = group["key_masses"]["dino"].numpy()
            mass_vae = group["key_masses"]["vae"].numpy()
            valid_dino = group["conditional_valid_fractions"]["dino"].numpy()
            valid_vae = group["conditional_valid_fractions"]["vae"].numpy()
            for name, value in group["key_masses"].items():
                npz_payload[f"mass_key_{_safe_file_part(name)}_{query_id}"] = value.numpy()
            npz_payload[f"mass_all_keys_{query_id}"] = group["mass_all_keys"].numpy()
            npz_payload[f"conditional_valid_fraction_dino_{query_id}"] = valid_dino
            npz_payload[f"conditional_valid_fraction_vae_{query_id}"] = valid_vae

            comparison_name = f"comparison_{query_id}.png"
            _render_comparison(
                comparison_rgb,
                [
                    (overlays[("absolute", "dino")], f"DINO absolute | mass={float(mass_dino[0]):.4f}"),
                    (overlays[("absolute", "vae")], f"VAE absolute | mass={float(mass_vae[0]):.4f}"),
                    (
                        overlays[("conditional", "dino")],
                        f"DINO conditional | valid={float(valid_dino[0]):.1%}",
                    ),
                    (
                        overlays[("conditional", "vae")],
                        f"VAE conditional | valid={float(valid_vae[0]):.1%}",
                    ),
                ],
                observation_label=(
                    f"Original observation | action queries {start}:{end}"
                    if original_observation is not None
                    else f"Observation RGB | action queries {start}:{end}"
                ),
            ).save(record_dir / comparison_name)
            files["comparison"] = comparison_name
            comparison_files[query_id] = comparison_name

            # Preserve the v1 names for a single query group.
            if len(record["query_groups"]) == 1:
                for branch in ("dino", "vae"):
                    Image.fromarray(overlays[("absolute", branch)]).save(record_dir / f"overlay_{branch}.png")
                    Image.fromarray(
                        _heatmap_rgb(_resize_map(normalized_maps[("absolute", branch)], rgb.shape[:2]))
                    ).save(record_dir / f"heatmap_{branch}.png")
                    npz_payload[f"scores_{branch}"] = group["scores_absolute"][branch].numpy()
                    npz_payload[f"mass_{branch}"] = group["key_masses"][branch].numpy()
                npz_payload["mass_all_keys"] = group["mass_all_keys"].numpy()
                Image.open(record_dir / comparison_name).save(record_dir / "comparison.png")

            query_group_metadata.append(
                {
                    "id": query_id,
                    "query_range": [start, end],
                    "branch_mass": {
                        "dino": _plain_value(mass_dino),
                        "vae": _plain_value(mass_vae),
                        "all_keys": _plain_value(group["mass_all_keys"]),
                    },
                    "conditional_valid_fraction": {
                        "dino": _plain_value(valid_dino),
                        "vae": _plain_value(valid_vae),
                    },
                    "display_normalization": normalization_metadata,
                    "files": files,
                }
            )

        np.savez_compressed(record_dir / "attention.npz", **npz_payload)
        metadata_payload = {
            "format": "st-wam-mot-action-vision-attention-v2",
            "source": "MoT action-branch self-attention",
            "model_identifier": (metadata or {}).get("model_identifier", "unknown"),
            "config_identifier": (metadata or {}).get("config_identifier", "unknown"),
            "inference_call_index": self.call_index,
            "layer_idx": record["layer_idx"],
            "denoising_step_idx": record["step_idx"],
            "timestep": record["timestep"],
            "configured_selection": {
                "layers": list(self.target_layers),
                "denoising_steps": list(self.target_steps),
                "call_indices": _plain_value(self.config.call_indices),
                "max_saves": self.config.max_saves,
            },
            "heads": record["head_indices"],
            "query_groups": query_group_metadata,
            "aggregation": {
                "absolute": "mean raw complete-key softmax over selected heads and action queries",
                "conditional": (
                    "normalize each valid head/query over the target branch, then average; "
                    "rows with branch mass <= 1e-8 are excluded"
                ),
            },
            "token_ranges": {name: list(span) for name, span in record["key_spans"].items()},
            "visual_token_ranges": {name: list(record["key_spans"][name]) for name in ("dino", "vae")},
            "grid": record["visual_meta"],
            "camera_layout": {
                "layout": self.config.camera_layout,
                "camera_names": list(self.config.camera_names),
                "camera_sizes": [list(item) for item in self.config.camera_sizes],
                "camera_regions": list(self.config.camera_regions),
                "mapping": "full model-input concatenation; spatial grids are restored from expert metadata",
            },
            "image_preprocessing": {
                "display_image": "model input tensor",
                "input_range": list(self.config.input_range),
                "caller_description": (metadata or {}).get(
                    "image_preprocessing_description",
                    "The visualization receives the already-preprocessed model input; no inverse transform was assumed.",
                ),
                "original_rgb_files": original_files,
            },
            "display_normalization": {
                "mode": self.config.display_normalization,
                "percentiles": list(self.config.display_percentiles),
                "overlay_alpha": self.config.overlay_alpha,
                "note": DISPLAY_NOTE,
            },
            "attention_math": {
                "scale": record["scale"],
                "mask_semantics": record["mask_semantics"],
                "mask_shape": list(record["attention_mask"].shape),
                "mask_visible_keys_per_query": record["attention_mask"]
                .to(dtype=torch.bool)
                .sum(dim=-1)
                .tolist(),
                "softmax_dimension": "complete key sequence before visual-branch slicing",
                "main_path": "unchanged torch.nn.functional.scaled_dot_product_attention",
            },
            "files": {
                "rgb_input": asset_prefix + "rgb_input.png",
                "comparisons": comparison_files,
                "raw_attention": "attention.npz",
            },
        }
        metadata_payload.update(_plain_value(metadata or {}))
        (record_dir / "metadata.json").write_text(
            json.dumps(_plain_value(metadata_payload), indent=2, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        return record_dir


__all__ = [
    "DISPLAY_NOTE",
    "MotAttentionRecorder",
    "MotAttentionVisualizationConfig",
    "aggregate_attention",
    "aggregate_conditional_attention",
    "camera_grid_slices",
    "normalize_heatmap",
    "recompute_attention_weights",
    "resolve_indices",
    "resolve_query_range",
    "resolve_query_ranges",
    "scores_to_spatial",
    "tensor_to_rgb_array",
]
