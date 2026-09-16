#!/usr/bin/env python3
"""Create a complete episode video from visual-mass-weighted MoT layers."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from make_mot_attention_multilayer_figure5 import (
    _load_row,
    _parse_layers,
    _render_overlay,
)
from make_mot_attention_video import write_video


_CALL_RE = re.compile(r"call_(\d+)$")


def _calls(attention_dir: Path, *, denoising_step: int, first_layer: int) -> tuple[int, ...]:
    result = []
    pattern = f"step_{denoising_step:04d}_layer_{first_layer:04d}/metadata.json"
    for metadata_path in attention_dir.glob(f"call_*/{pattern}"):
        match = _CALL_RE.fullmatch(metadata_path.parent.parent.name)
        if match:
            result.append(int(match.group(1)))
    result.sort()
    if not result:
        raise SystemExit(f"No multilayer attention calls found under {attention_dir}")
    if result != list(range(result[-1] + 1)):
        raise ValueError(f"Expected contiguous calls beginning at zero, got {result}")
    return tuple(result)


def _comparison_frame(
    rgb: np.ndarray,
    dino_overlay: np.ndarray,
    vae_overlay: np.ndarray,
    *,
    call_index: int,
    environment_step: int,
    weights: np.ndarray,
) -> np.ndarray:
    images = [rgb, dino_overlay, vae_overlay]
    labels = ["Model input", "30-layer weighted DINO", "30-layer weighted VAE"]
    height, width = rgb.shape[:2]
    label_height = 30
    footer_height = 32
    canvas = Image.new("RGB", (width * 3, label_height + height + footer_height), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for index, (array, label) in enumerate(zip(images, labels)):
        canvas.paste(Image.fromarray(array), (index * width, label_height))
        draw.text((index * width + 8, 9), label, fill="black", font=font)
    top = np.argsort(weights)[::-1][:3]
    top_text = ", ".join(f"L{int(layer)}={float(weights[layer]):.1%}" for layer in top)
    draw.text(
        (8, label_height + height + 9),
        f"call={call_index} env_step={environment_step} | top layer weights: {top_text}",
        fill="black",
        font=font,
    )
    return np.asarray(canvas, dtype=np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--row", choices=("drawer", "moka_pot"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--denoising-step", type=int, default=9)
    parser.add_argument("--layers", type=_parse_layers, default=_parse_layers("0:30"))
    parser.add_argument("--overlay-alpha", type=float, default=0.45)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--hold-frames", type=int, default=10)
    parser.add_argument("--initial-frames", type=int, default=30)
    parser.add_argument("--episode-length", type=int, required=True)
    args = parser.parse_args()

    attention_dir = args.run_root / args.row / "attention"
    calls = _calls(
        attention_dir,
        denoising_step=args.denoising_step,
        first_layer=args.layers[0],
    )
    frame_dir = args.output.parent / "multilayer_fused_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    records = []
    weight_rows = []
    for call_index in calls:
        data = _load_row(
            args.run_root,
            row=args.row,
            call_index=call_index,
            denoising_step=args.denoising_step,
            layers=args.layers,
        )
        environment_step = data["environment_step"]
        if environment_step is None:
            raise ValueError(f"call {call_index} is missing environment_step")
        dino_overlay, _, _ = _render_overlay(
            data["rgb"], data["fused"]["dino"], args.overlay_alpha
        )
        vae_overlay, _, _ = _render_overlay(
            data["rgb"], data["fused"]["vae"], args.overlay_alpha
        )
        frame = _comparison_frame(
            data["rgb"],
            dino_overlay,
            vae_overlay,
            call_index=call_index,
            environment_step=int(environment_step),
            weights=data["weights"],
        )
        frame_path = frame_dir / f"call_{call_index:06d}.png"
        Image.fromarray(frame).save(frame_path)
        records.append(
            {
                "comparison_path": frame_path,
                "episode_index": 0,
                "call_index": call_index,
                "environment_step": int(environment_step),
            }
        )
        weight_rows.append(data["weights"].astype(np.float32))

    video_manifest = write_video(
        records,
        args.output,
        fps=args.fps,
        hold_frames=args.hold_frames,
        initial_frames=args.initial_frames,
        episode_length=args.episode_length,
    )
    weights_path = args.output.with_name(args.output.stem + "_layer_weights.npz")
    np.savez_compressed(
        weights_path,
        calls=np.asarray(calls, dtype=np.int64),
        layers=np.asarray(args.layers, dtype=np.int64),
        weights=np.stack(weight_rows),
    )
    video_manifest.update(
        {
            "format": "st-wam-mot-multilayer-visual-mass-video-v1",
            "row": args.row,
            "layers": list(args.layers),
            "denoising_step": args.denoising_step,
            "query_range": [0, 32],
            "fusion": {
                "visual_mass": "m_l = mass_dino_l + mass_vae_l",
                "layer_weight": "w_l = m_l / sum_r(m_r)",
                "within_branch_map": "p_l,b(x) = score_l,b(x) / sum_x(score_l,b(x))",
                "fused_map": "F_b(x) = sum_l(w_l * p_l,b(x))",
            },
            "layer_weights": str(weights_path),
        }
    )
    manifest_path = args.output.with_suffix(".json")
    manifest_path.write_text(json.dumps(video_manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Saved complete multilayer attention video: {args.output}")
    print(f"Saved video manifest: {manifest_path}")
    print(f"Saved per-call layer weights: {weights_path}")


if __name__ == "__main__":
    main()
