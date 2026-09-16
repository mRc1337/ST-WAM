#!/usr/bin/env python3
"""Assemble the final two-by-two MoT attention panel.

The representative call indices are supplied after inspecting RGB contact
sheets. This script never opens attention.npz and never uses heatmap values to
choose a frame; it only reads the already-rendered overlay PNG for the chosen
call and records the RGB-based selection in a manifest.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


TASKS = {
    "drawer": {
        "label": "drawer",
        "description": "open the middle drawer of the cabinet",
        "suite": "libero_goal",
        "task_id": 0,
    },
    "moka_pot": {
        "label": "moka pot",
        "description": "turn on the stove and put the moka pot on it",
        "suite": "libero_10",
        "task_id": 2,
    },
}


def _record(
    attention_dir: Path,
    *,
    call_index: int,
    layer: int,
    denoising_step: int,
) -> tuple[Path, dict]:
    record_dir = attention_dir / f"call_{int(call_index):06d}" / (
        f"step_{int(denoising_step):04d}_layer_{int(layer):04d}"
    )
    metadata_path = record_dir / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing selected RGB frame metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if int(metadata.get("layer_idx", -1)) != int(layer):
        raise ValueError(f"Unexpected layer in {metadata_path}: {metadata.get('layer_idx')}")
    if int(metadata.get("denoising_step_idx", -1)) != int(denoising_step):
        raise ValueError(
            f"Unexpected denoising step in {metadata_path}: "
            f"{metadata.get('denoising_step_idx')}"
        )
    groups = metadata.get("query_groups", [])
    if len(groups) != 1 or groups[0].get("query_range") != [0, 32]:
        raise ValueError(
            f"Selected record is not the unified action-query group 0:32: {metadata_path}"
        )
    for filename in (
        "rgb_input.png",
        "rgb_original_concatenated.png",
        "overlay_dino.png",
        "overlay_vae.png",
    ):
        if not (record_dir / filename).is_file():
            raise FileNotFoundError(f"Missing selected figure asset: {record_dir / filename}")
    return record_dir, metadata


def _fit(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    return ImageOps.contain(image.convert("RGB"), size, method=Image.Resampling.BILINEAR)


def _panel(
    overlay: Image.Image,
    *,
    title: str,
    cell_size: tuple[int, int],
) -> Image.Image:
    width, height = cell_size
    label_height = 36
    canvas = Image.new("RGB", (width, height + label_height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 10), title, fill="black", font=ImageFont.load_default())
    fitted = _fit(overlay, (width, height))
    canvas.paste(fitted, ((width - fitted.width) // 2, label_height + (height - fitted.height) // 2))
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drawer-attention-dir", type=Path, required=True)
    parser.add_argument("--moka-attention-dir", type=Path, required=True)
    parser.add_argument("--drawer-call", type=int, required=True)
    parser.add_argument("--moka-call", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=29)
    parser.add_argument("--denoising-step", type=int, default=9)
    args = parser.parse_args()

    selections = {
        "drawer": (args.drawer_attention_dir, args.drawer_call),
        "moka_pot": (args.moka_attention_dir, args.moka_call),
    }
    records: dict[str, tuple[Path, dict]] = {}
    for row, (attention_dir, call_index) in selections.items():
        records[row] = _record(
            attention_dir,
            call_index=call_index,
            layer=args.layer,
            denoising_step=args.denoising_step,
        )

    cell_size = (560, 300)
    row_label_width = 230
    row_label_height = cell_size[1] + 36
    canvas = Image.new(
        "RGB",
        (row_label_width + cell_size[0] * 2, row_label_height * 2 + 52),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text(
        (row_label_width + 10, 14),
        "Action Queries 0:32 -> Current DINO",
        fill="black",
        font=font,
    )
    draw.text(
        (row_label_width + cell_size[0] + 10, 14),
        "Action Queries 0:32 -> Current VAE",
        fill="black",
        font=font,
    )

    manifest_rows = []
    for row_index, row in enumerate(("drawer", "moka_pot")):
        record_dir, metadata = records[row]
        y = 52 + row_index * row_label_height
        task = TASKS[row]
        draw.text((10, y + 12), task["label"], fill="black", font=font)
        draw.text((10, y + 34), task["description"], fill="black", font=font)
        draw.text(
            (10, y + 56),
            f"call={metadata['inference_call_index']} env_step={metadata.get('environment_step')}",
            fill="black",
            font=font,
        )
        dino = Image.open(record_dir / "overlay_dino.png")
        vae = Image.open(record_dir / "overlay_vae.png")
        canvas.paste(
            _panel(dino, title="", cell_size=cell_size),
            (row_label_width, y),
        )
        canvas.paste(
            _panel(vae, title="", cell_size=cell_size),
            (row_label_width + cell_size[0], y),
        )
        manifest_rows.append(
            {
                "row": row,
                "suite": task["suite"],
                "task_id": task["task_id"],
                "task_description": task["description"],
                "call_index": int(metadata["inference_call_index"]),
                "environment_step": metadata.get("environment_step"),
                "record_dir": str(record_dir),
                "rgb_input": str(record_dir / "rgb_input.png"),
                "rgb_original_concatenated": str(record_dir / "rgb_original_concatenated.png"),
                "overlay_dino": str(record_dir / "overlay_dino.png"),
                "overlay_vae": str(record_dir / "overlay_vae.png"),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)
    manifest = {
        "format": "st-wam-mot-figure5-panel-v1",
        "output": str(args.output),
        "selection_basis": (
            "Representative calls were selected from RGB-only contact sheets; "
            "attention values and heatmap appearance were not used for selection."
        ),
        "protocol": {
            "layer": int(args.layer),
            "denoising_step": int(args.denoising_step),
            "heads": "all",
            "action_query_range": [0, 32],
            "display_normalization": "percentile [1,99]",
        },
        "rows": manifest_rows,
    }
    manifest_path = args.output.with_suffix(".json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Saved Figure-5 2x2 panel: {args.output}")
    print(f"Saved RGB-selection manifest: {manifest_path}")


if __name__ == "__main__":
    main()
