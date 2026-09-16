"""Fixed-scale heatmaps and publication-ready aggregate figures."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


def normalize_similarity_map(similarity: np.ndarray, *, vmin: float = -1.0, vmax: float = 1.0) -> np.ndarray:
    if vmax <= vmin:
        raise ValueError(f"Expected vmax>vmin, got {vmin}, {vmax}")
    return np.clip((np.asarray(similarity, dtype=np.float32) - vmin) / (vmax - vmin), 0.0, 1.0)


def _resized_rgb(rgb: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    height, width = target_hw
    return np.asarray(Image.fromarray(rgb).resize((width, height), Image.Resampling.BILINEAR))


def render_overlay(rgb: np.ndarray, similarity: np.ndarray, *, alpha: float = 0.48, vmin: float = -1.0, vmax: float = 1.0) -> np.ndarray:
    heat = plt.get_cmap("turbo")(normalize_similarity_map(similarity, vmin=vmin, vmax=vmax))[..., :3]
    heat = (heat * 255).astype(np.uint8)
    heat = _resized_rgb(heat, rgb.shape[:2])
    return np.clip((1.0 - alpha) * rgb.astype(np.float32) + alpha * heat.astype(np.float32), 0, 255).astype(np.uint8)


def save_episode_figure(
    output_path: str | Path,
    *,
    episode_id: str,
    task_name: str,
    outcome: str,
    camera_frames: Mapping[str, np.ndarray],
    similarity_maps: Mapping[str, Sequence[np.ndarray]],
    frame_rows: pd.DataFrame,
    stage_indices: Sequence[int] | None = None,
    vmin: float = -1.0,
    vmax: float = 1.0,
) -> None:
    """Save RGB / raw-map / overlay stages for one episode."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    first_camera = next(iter(camera_frames))
    frame_count = len(camera_frames[first_camera])
    if stage_indices is None:
        stage_indices = [0, frame_count // 3, (2 * frame_count) // 3, frame_count - 1]
    stage_indices = [max(0, min(int(index), frame_count - 1)) for index in stage_indices]
    stage_names = ["approach", "pre_grasp", "close_or_mid", "lift_or_terminal"]
    cameras = list(camera_frames)
    fig, axes = plt.subplots(len(cameras) * 3, len(stage_indices), figsize=(3.2 * len(stage_indices), 2.35 * len(cameras) * 3), squeeze=False)
    for camera_index, camera in enumerate(cameras):
        frames = camera_frames[camera]
        maps = similarity_maps[camera]
        for stage_index, frame_index in enumerate(stage_indices):
            row_offset = camera_index * 3
            rgb = frames[frame_index]
            sim = maps[frame_index]
            axes[row_offset, stage_index].imshow(rgb)
            axes[row_offset, stage_index].set_title(f"{stage_names[stage_index] if stage_index < len(stage_names) else frame_index}\n{camera} RGB")
            axes[row_offset + 1, stage_index].imshow(sim, cmap="turbo", vmin=vmin, vmax=vmax)
            axes[row_offset + 1, stage_index].set_title("raw cosine")
            axes[row_offset + 2, stage_index].imshow(render_overlay(rgb, sim, vmin=vmin, vmax=vmax))
            axes[row_offset + 2, stage_index].set_title("fixed-scale overlay")
            for row in range(row_offset, row_offset + 3):
                axes[row, stage_index].axis("off")
    fig.suptitle(f"{task_name} | episode={episode_id} | {outcome}", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_heatmap_video(
    output_path: str | Path,
    *,
    camera_frames: Mapping[str, np.ndarray],
    similarity_maps: Mapping[str, Sequence[np.ndarray]],
    frame_rows: pd.DataFrame,
    fps: int = 12,
    vmin: float = -1.0,
    vmax: float = 1.0,
) -> None:
    """Optional MP4 with RGB, heatmap, overlay, and metric text."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cameras = list(camera_frames)
    writer = imageio.get_writer(str(output_path), fps=fps, codec="libx264", macro_block_size=1)
    try:
        for frame_index in range(len(camera_frames[cameras[0]])):
            panels: list[np.ndarray] = []
            row = frame_rows[frame_rows["frame_index"] == frame_index]
            for camera in cameras:
                rgb = camera_frames[camera][frame_index]
                sim = similarity_maps[camera][frame_index]
                heat = (plt.get_cmap("turbo")(normalize_similarity_map(sim, vmin=vmin, vmax=vmax))[..., :3] * 255).astype(np.uint8)
                heat = _resized_rgb(heat, rgb.shape[:2])
                overlay = render_overlay(rgb, sim, vmin=vmin, vmax=vmax)
                panels.extend([rgb, heat, overlay])
            target_h = max(panel.shape[0] for panel in panels)
            panels = [
                _resized_rgb(panel, (target_h, int(round(panel.shape[1] * target_h / panel.shape[0]))))
                for panel in panels
            ]
            frame = np.concatenate(panels, axis=1)
            # Text is drawn using matplotlib to avoid an additional font/OpenCV dependency.
            fig = plt.figure(figsize=(max(8, frame.shape[1] / 100), max(2.5, frame.shape[0] / 100)), dpi=100)
            axis = fig.add_axes([0, 0, 1, 1])
            axis.imshow(frame)
            axis.axis("off")
            if not row.empty:
                first = row.iloc[0]
                text = (
                    f"task={first['task_name']}  episode={first['episode_id']}  frame={frame_index} "
                    f"progress={first['progress']:.3f}  outcome={first['outcome']}\n"
                    f"TargetMass={first['target_mass']:.4f}  Margin={first['similarity_margin']:.4f} "
                    f"Drift={first['temporal_drift']:.4f}"
                )
                axis.text(0.01, 0.01, text, transform=axis.transAxes, color="white", fontsize=8, va="bottom", family="monospace", bbox={"facecolor": "black", "alpha": 0.6, "pad": 3})
            fig.canvas.draw()
            image = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
            plt.close(fig)
            writer.append_data(image)
    finally:
        writer.close()


def save_aggregate_plots(
    output_dir: str | Path,
    aggregate_rows: pd.DataFrame,
    *,
    metrics: Sequence[str],
    group_prefix: str = "overall",
) -> list[Path]:
    """Save success/failure trajectory plots with bootstrap ribbons."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    if aggregate_rows.empty:
        return paths
    for metric in metrics:
        fig, axis = plt.subplots(figsize=(7.2, 4.3))
        plotted = False
        for outcome, color in (("success", "#1b9e77"), ("failure", "#d95f02")):
            rows = aggregate_rows[aggregate_rows["outcome"] == outcome].sort_values("progress")
            if rows.empty or f"{metric}_mean" not in rows:
                continue
            x = rows["progress"].to_numpy()
            mean = rows[f"{metric}_mean"].to_numpy()
            low = rows[f"{metric}_ci_low"].to_numpy()
            high = rows[f"{metric}_ci_high"].to_numpy()
            if not np.isfinite(mean).any():
                continue
            axis.plot(x, mean, color=color, label=f"{outcome} (n={int(rows['episode_count'].iloc[0])})", linewidth=2)
            if np.isfinite(low).any() and np.isfinite(high).any():
                axis.fill_between(x, low, high, color=color, alpha=0.18)
            plotted = True
        if not plotted:
            plt.close(fig)
            continue
        axis.set_xlabel("normalized trajectory progress")
        axis.set_ylabel(metric.replace("_", " "))
        axis.set_title(f"Success vs failure: {metric} ({group_prefix})")
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
        fig.tight_layout()
        path = output_dir / f"{group_prefix}_{metric}.png"
        fig.savefig(path, dpi=170)
        plt.close(fig)
        paths.append(path)
    return paths


def save_failure_lead_time_plot(output_path: str | Path, precursor: Mapping[str, Any]) -> None:
    values = [
        float(item["lead_time_frames"])
        for item in precursor.get("episodes", [])
        if item.get("outcome") == "failure" and np.isfinite(item.get("lead_time_frames", np.nan))
    ]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(6.2, 4.0))
    if values:
        axis.hist(values, bins=min(10, max(3, len(values))), color="#7570b3", alpha=0.85)
        axis.axvline(float(np.mean(values)), color="black", linestyle="--", label=f"mean={np.mean(values):.2f}")
        axis.legend(frameon=False)
    else:
        axis.text(0.5, 0.5, "No failure episode had a valid pre-failure anomaly", ha="center", va="center")
    axis.set_xlabel("lead time (frames)")
    axis.set_ylabel("failure episodes")
    axis.set_title("DINO anomaly lead time")
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)
