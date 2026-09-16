#!/usr/bin/env python3
"""Compare one official and one user MoT attention capture.

The comparison is deliberately based on the raw ``attention.npz`` files as
well as PNGs.  It reports whether the model-input RGB image is byte-identical,
compares branch masses, and computes a Pearson correlation for the DINO/VAE
spatial scores when their token layouts match.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _record(root: Path) -> Path:
    records = sorted(root.glob("call_*/step_*/metadata.json"))
    if not records:
        raise FileNotFoundError(f"No attention metadata found under {root}")
    return records[0].parent


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _corr(left: np.ndarray, right: np.ndarray) -> float | None:
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    if left.shape != right.shape or left.size < 2:
        return None
    left_std = float(left.std())
    right_std = float(right.std())
    if left_std == 0.0 or right_std == 0.0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _tile(record_dirs: list[tuple[str, Path]], output: Path) -> None:
    names = ["RGB input", "DINO heatmap", "VAE heatmap", "DINO overlay", "VAE overlay"]
    rows: list[list[Image.Image]] = []
    for label, record in record_dirs:
        paths = [
            record / "rgb_input.png",
            record / "heatmap_dino.png",
            record / "heatmap_vae.png",
            record / "overlay_dino.png",
            record / "overlay_vae.png",
        ]
        images = [Image.open(path).convert("RGB") for path in paths]
        width = max(image.width for image in images)
        height = max(image.height for image in images)
        row = []
        for image in images:
            canvas = Image.new("RGB", (width, height + 28), "white")
            canvas.paste(image, ((width - image.width) // 2, 28 + (height - image.height) // 2))
            draw = ImageDraw.Draw(canvas)
            draw.text((6, 6), names[len(row)], fill="black")
            row.append(canvas)
        row_width = sum(image.width for image in row)
        row_canvas = Image.new("RGB", (row_width, height + 28), "white")
        x = 0
        for image in row:
            row_canvas.paste(image, (x, 0))
            x += image.width
        label_canvas = Image.new("RGB", (120, row_canvas.height), "white")
        ImageDraw.Draw(label_canvas).text((6, 8), label, fill="black")
        rows.append([label_canvas, row_canvas])

    label_width = max(row[0].width for row in rows)
    row_height = rows[0][1].height
    canvas = Image.new("RGB", (label_width + rows[0][1].width, row_height * len(rows)), "white")
    y = 0
    for label_canvas, row_canvas in rows:
        canvas.paste(label_canvas, (0, y))
        canvas.paste(row_canvas, (label_width, y))
        y += row_height
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-dir", type=Path, required=True)
    parser.add_argument("--user-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    official = _record(args.official_dir)
    user = _record(args.user_dir)
    with (official / "metadata.json").open(encoding="utf-8") as stream:
        official_meta = json.load(stream)
    with (user / "metadata.json").open(encoding="utf-8") as stream:
        user_meta = json.load(stream)

    report: dict[str, object] = {
        "official_record": str(official),
        "user_record": str(user),
        "official_metadata": official_meta,
        "user_metadata": user_meta,
        "input_rgb": {
            "same_sha256": _sha256(official / "rgb_input.png") == _sha256(user / "rgb_input.png"),
            "official_sha256": _sha256(official / "rgb_input.png"),
            "user_sha256": _sha256(user / "rgb_input.png"),
        },
    }
    with np.load(official / "attention.npz") as official_npz, np.load(user / "attention.npz") as user_npz:
        branch_report: dict[str, object] = {}
        for branch in ("dino", "vae"):
            left = official_npz[f"scores_{branch}"]
            right = user_npz[f"scores_{branch}"]
            branch_report[branch] = {
                "official_shape": list(left.shape),
                "user_shape": list(right.shape),
                "same_shape": list(left.shape) == list(right.shape),
                "spatial_pearson": _corr(left, right),
                "official_mass_mean": float(np.asarray(official_npz[f"mass_{branch}"]).mean()),
                "user_mass_mean": float(np.asarray(user_npz[f"mass_{branch}"]).mean()),
            }
        report["branches"] = branch_report
        report["mass_all_keys"] = {
            "official": float(np.asarray(official_npz["mass_all_keys"]).mean()),
            "user": float(np.asarray(user_npz["mass_all_keys"]).mean()),
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "comparison.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _tile(
        [("official", official), ("user", user)],
        args.output_dir / "official_vs_user.png",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"saved: {args.output_dir / 'official_vs_user.png'}")


if __name__ == "__main__":
    main()
