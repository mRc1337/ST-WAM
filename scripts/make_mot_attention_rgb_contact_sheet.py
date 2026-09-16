#!/usr/bin/env python3
"""Make a contact sheet used to select representative MoT calls by RGB only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attention-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--cell-width", type=int, default=360)
    args = parser.parse_args()
    if args.columns <= 0 or args.cell_width <= 0:
        parser.error("--columns and --cell-width must be positive")

    records = []
    for metadata_path in sorted(args.attention_dir.glob("call_*/step_*/metadata.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        record_dir = metadata_path.parent
        image_path = record_dir / "rgb_original_concatenated.png"
        if not image_path.is_file():
            image_path = record_dir / "rgb_input.png"
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing RGB asset for {metadata_path}")
        records.append((metadata, image_path))
    if not records:
        raise SystemExit(f"No attention records found under {args.attention_dir}")

    font = ImageFont.load_default()
    label_height = 30
    cell_height = round(args.cell_width * 0.58) + label_height
    rows = (len(records) + args.columns - 1) // args.columns
    canvas = Image.new("RGB", (args.columns * args.cell_width, rows * cell_height), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (metadata, image_path) in enumerate(records):
        image = Image.open(image_path).convert("RGB")
        fitted = ImageOps.contain(image, (args.cell_width, cell_height - label_height), method=Image.Resampling.BILINEAR)
        x = (index % args.columns) * args.cell_width
        y = (index // args.columns) * cell_height
        canvas.paste(fitted, (x + (args.cell_width - fitted.width) // 2, y + label_height))
        draw.text(
            (x + 6, y + 8),
            f"call={metadata.get('inference_call_index')} env_step={metadata.get('environment_step')}",
            fill="black",
            font=font,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)
    print(f"Saved RGB-only contact sheet: {args.output}")
    print(f"records={len(records)}")


if __name__ == "__main__":
    main()
