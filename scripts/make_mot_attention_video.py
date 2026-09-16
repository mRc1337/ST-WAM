#!/usr/bin/env python3
"""Encode per-inference MoT attention comparisons into an episode video.

The model computes a new attention map only when it replans. The map is held
for the corresponding executed action steps. When ``--initial-frames`` and
``--episode-length`` are supplied, the output covers the complete episode:
initial wait steps are represented by an explicit no-capture slate and the
last attention record is held only through the terminal step.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


_CALL_RE = re.compile(r"call_(\d+)$")


def _even_frame(frame: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]
    pad_height = height % 2
    pad_width = width % 2
    if pad_height == 0 and pad_width == 0:
        return frame
    return np.pad(frame, ((0, pad_height), (0, pad_width), (0, 0)), mode="edge")


def _call_index(path: Path, metadata: dict[str, Any]) -> int:
    value = metadata.get("inference_call_index")
    if value is not None:
        return int(value)
    match = _CALL_RE.match(path.parent.parent.name)
    if match is None:
        raise ValueError(f"Cannot infer call index from {path}")
    return int(match.group(1))


def _episode_index(metadata: dict[str, Any]) -> int:
    for key in ("episode_index", "episode_idx", "trial_index"):
        if key in metadata:
            return int(metadata[key])
    return 0


def _comparison_path(
    metadata_path: Path,
    metadata: dict[str, Any],
    query_range: tuple[int, int] | None,
) -> Path | None:
    groups = metadata.get("query_groups")
    if isinstance(groups, list):
        for group in groups:
            current_range = tuple(int(item) for item in group.get("query_range", ()))
            if query_range is None or current_range == query_range:
                filename = group.get("files", {}).get("comparison")
                return metadata_path.parent / filename if filename else None
        return None
    if query_range is not None:
        legacy_range = tuple(int(item) for item in metadata.get("query_range", ()))
        if legacy_range and legacy_range != query_range:
            return None
    return metadata_path.parent / "comparison.png"


def collect_records(
    attention_dir: Path,
    episode_index: int | None = None,
    *,
    layer: int | None = None,
    denoising_step: int | None = None,
    query_range: tuple[int, int] | None = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for metadata_path in sorted(attention_dir.glob("call_*/step_*/metadata.json")):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Cannot read metadata: {metadata_path}") from exc
        current_episode = _episode_index(metadata)
        if episode_index is not None and current_episode != episode_index:
            continue
        if layer is not None and int(metadata.get("layer_idx", -1)) != layer:
            continue
        if denoising_step is not None and int(metadata.get("denoising_step_idx", -1)) != denoising_step:
            continue
        comparison_path = _comparison_path(metadata_path, metadata, query_range)
        if comparison_path is None:
            continue
        if not comparison_path.is_file():
            raise FileNotFoundError(f"Missing comparison image for {metadata_path}: {comparison_path}")
        records.append(
            {
                "metadata_path": metadata_path,
                "comparison_path": comparison_path,
                "metadata": metadata,
                "episode_index": current_episode,
                "call_index": _call_index(metadata_path, metadata),
                "environment_step": metadata.get("environment_step"),
            }
        )
    records.sort(key=lambda item: (item["episode_index"], item["call_index"], str(item["metadata_path"])))
    if not records:
        scope = "" if episode_index is None else f" for episode {episode_index}"
        raise SystemExit(f"No attention comparison records found under {attention_dir}{scope}")
    call_keys = [(item["episode_index"], item["call_index"]) for item in records]
    if len(call_keys) != len(set(call_keys)):
        raise ValueError(
            "Multiple attention captures matched the same inference call; provide --layer and "
            "--denoising-step to select one capture."
        )
    return records


def _frame(path: Path, target_size: tuple[int, int] | None) -> tuple[np.ndarray, tuple[int, int]]:
    image = Image.open(path).convert("RGB")
    if target_size is not None and image.size != target_size:
        image = image.resize(target_size, resample=Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.uint8)
    return array, image.size


def _hold_count(
    records: list[dict[str, Any]],
    index: int,
    default: int,
    *,
    episode_length: int | None = None,
) -> int:
    current = records[index].get("environment_step")
    if index + 1 >= len(records):
        if episode_length is not None:
            if current is None:
                raise ValueError(
                    "Exact episode coverage requires environment_step metadata on the final record"
                )
            count = int(episode_length) - int(current)
            if count <= 0:
                raise ValueError(
                    "episode_length must be greater than the final environment_step; "
                    f"got episode_length={episode_length}, final_step={current}"
                )
            return count
        return default
    following = records[index + 1].get("environment_step")
    if current is None or following is None:
        if episode_length is not None:
            raise ValueError(
                "Exact episode coverage requires environment_step metadata on every record"
            )
        return default
    delta = int(following) - int(current)
    if delta <= 0:
        raise ValueError(
            "Attention records must have strictly increasing environment_step values; "
            f"got {current} followed by {following}"
        )
    return delta


def _no_attention_frame(image_size: tuple[int, int], initial_frames: int) -> np.ndarray:
    """Create an explicit slate for environment steps before first inference."""

    image = Image.new("RGB", image_size, (28, 31, 38))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    text = (
        "No MoT attention capture\n"
        "initial wait steps\n"
        f"environment steps 0:{int(initial_frames)}"
    )
    bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=5)
    x = (image.width - (bbox[2] - bbox[0])) // 2
    y = (image.height - (bbox[3] - bbox[1])) // 2
    draw.multiline_text(
        (x, y),
        text,
        fill=(235, 238, 245),
        font=font,
        spacing=5,
        align="center",
    )
    return np.asarray(image, dtype=np.uint8)


def write_video(
    records: list[dict[str, Any]],
    output_path: Path,
    *,
    fps: int,
    hold_frames: int,
    initial_frames: int = 0,
    episode_length: int | None = None,
) -> dict[str, Any]:
    if initial_frames < 0:
        raise ValueError(f"initial_frames must be non-negative, got {initial_frames}")
    if episode_length is not None and episode_length <= 0:
        raise ValueError(f"episode_length must be positive, got {episode_length}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    first, image_size = _frame(records[0]["comparison_path"], None)
    first_environment_step = records[0].get("environment_step")
    if episode_length is not None:
        if first_environment_step is None:
            raise ValueError("Exact episode coverage requires environment_step metadata")
        if int(first_environment_step) != int(initial_frames):
            raise ValueError(
                "initial_frames must equal the first attention record's environment_step; "
                f"got initial_frames={initial_frames}, first_step={first_environment_step}"
            )
    writer = imageio.get_writer(
        str(output_path),
        fps=max(int(fps), 1),
        codec="libx264",
        format="FFMPEG",
        pixelformat="yuv420p",
        macro_block_size=1,
    )
    frame_count = 0
    hold_counts: list[int] = []
    try:
        if initial_frames:
            initial_frame = _no_attention_frame(image_size, initial_frames)
            for _ in range(initial_frames):
                writer.append_data(_even_frame(initial_frame))
                frame_count += 1
        for index, record in enumerate(records):
            if index == 0:
                frame = first
            else:
                frame, _ = _frame(record["comparison_path"], image_size)
            count = _hold_count(
                records,
                index,
                hold_frames,
                episode_length=episode_length,
            )
            hold_counts.append(count)
            for _ in range(count):
                writer.append_data(_even_frame(frame))
                frame_count += 1
    finally:
        writer.close()

    if episode_length is not None and frame_count != int(episode_length):
        raise RuntimeError(
            "Exact episode video length mismatch: "
            f"wrote {frame_count} frames, expected {episode_length}"
        )

    return {
        "format": "st-wam-mot-attention-video-v2",
        "fps": int(fps),
        "frame_count": frame_count,
        "image_size_wh": list(image_size),
        "hold_frames_default": int(hold_frames),
        "initial_wait_frames": int(initial_frames),
        "episode_length_frames": None if episode_length is None else int(episode_length),
        "coverage": (
            "complete_environment_episode"
            if episode_length is not None
            else "attention_records_with_fallback_holds"
        ),
        "initial_frame": "no_attention_capture_slate" if initial_frames else None,
        "records": [
            {
                "episode_index": int(record["episode_index"]),
                "call_index": int(record["call_index"]),
                "environment_step": record["environment_step"],
                "comparison": str(record["comparison_path"]),
                "hold_frames": int(hold_counts[index]),
            }
            for index, record in enumerate(records)
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attention-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument(
        "--hold-frames",
        type=int,
        default=10,
        help="Fallback number of video frames per replan (default: 10).",
    )
    parser.add_argument(
        "--initial-frames",
        type=int,
        default=0,
        help="Initial environment steps before the first attention capture (default: 0).",
    )
    parser.add_argument(
        "--episode-length",
        type=int,
        default=None,
        help=(
            "Exact total environment-step frame count. With --initial-frames, validates and "
            "trims the final hold to complete the episode."
        ),
    )
    parser.add_argument("--episode-index", type=int, default=None)
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--denoising-step", type=int, default=None)
    parser.add_argument(
        "--query-range",
        default=None,
        help="Action-query range such as 0:8; defaults to the first saved range.",
    )
    args = parser.parse_args()
    if args.fps <= 0:
        parser.error("--fps must be positive")
    if args.hold_frames <= 0:
        parser.error("--hold-frames must be positive")
    if args.initial_frames < 0:
        parser.error("--initial-frames must be non-negative")
    if args.episode_length is not None and args.episode_length <= 0:
        parser.error("--episode-length must be positive")

    query_range = None
    if args.query_range is not None:
        match = re.fullmatch(r"\s*(\d+)\s*:\s*(\d+)\s*", args.query_range)
        if match is None or int(match.group(1)) >= int(match.group(2)):
            parser.error("--query-range must be START:END with START < END")
        query_range = (int(match.group(1)), int(match.group(2)))
    records = collect_records(
        args.attention_dir,
        args.episode_index,
        layer=args.layer,
        denoising_step=args.denoising_step,
        query_range=query_range,
    )
    manifest = write_video(
        records,
        args.output,
        fps=args.fps,
        hold_frames=args.hold_frames,
        initial_frames=args.initial_frames,
        episode_length=args.episode_length,
    )
    manifest_path = args.output.with_suffix(".json")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved MoT attention video: {args.output}")
    print(f"Saved video manifest: {manifest_path}")
    print(f"records={len(records)} frames={manifest['frame_count']} fps={manifest['fps']}")


if __name__ == "__main__":
    main()
