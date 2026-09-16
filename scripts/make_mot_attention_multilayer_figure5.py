#!/usr/bin/env python3
"""Fuse MoT attention maps across layers using each layer's visual mass.

For layer ``l``, the shared layer weight is the mean attention mass assigned
to current DINO plus current VAE keys, normalized across all selected layers.
Within each visual branch, the layer map is normalized to a spatial
distribution before applying that shared weight. This separates the question
"how much does this layer attend to vision?" from "where within this branch?".
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from fastwam.models.wan22.mot_attention_visualization import (
    _heatmap_rgb,
    _resize_map,
    normalize_heatmap,
    scores_to_spatial,
)


TASKS = {
    "drawer": ("drawer", "open the middle drawer of the cabinet"),
    "moka_pot": ("moka pot", "turn on the stove and put the moka pot on it"),
}


def _parse_layers(value: str) -> tuple[int, ...]:
    text = value.strip()
    if ":" in text:
        start_text, end_text = text.split(":", 1)
        layers = tuple(range(int(start_text), int(end_text)))
    else:
        layers = tuple(int(item.strip()) for item in text.split(",") if item.strip())
    if not layers or len(set(layers)) != len(layers) or any(layer < 0 for layer in layers):
        raise argparse.ArgumentTypeError(f"invalid layer selection: {value!r}")
    return layers


def _record_dir(
    run_root: Path,
    row: str,
    call_index: int,
    denoising_step: int,
    layer: int,
) -> Path:
    return (
        run_root
        / row
        / "attention"
        / f"call_{call_index:06d}"
        / f"step_{denoising_step:04d}_layer_{layer:04d}"
    )


def _load_row(
    run_root: Path,
    *,
    row: str,
    call_index: int,
    denoising_step: int,
    layers: tuple[int, ...],
) -> dict[str, Any]:
    layer_records: list[dict[str, Any]] = []
    rgb: np.ndarray | None = None
    environment_step: int | None = None

    for layer in layers:
        record_dir = _record_dir(run_root, row, call_index, denoising_step, layer)
        metadata_path = record_dir / "metadata.json"
        npz_path = record_dir / "attention.npz"
        if not metadata_path.is_file() or not npz_path.is_file():
            raise FileNotFoundError(f"missing layer {layer} record under {record_dir}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        groups = metadata.get("query_groups", [])
        if len(groups) != 1 or groups[0].get("query_range") != [0, 32]:
            raise ValueError(f"expected exactly one action-query range [0,32): {metadata_path}")
        if int(metadata["layer_idx"]) != layer:
            raise ValueError(f"layer mismatch in {metadata_path}")
        if environment_step is None:
            environment_step = metadata.get("environment_step")

        rgb_path = (record_dir / metadata["files"]["rgb_input"]).resolve()
        current_rgb = np.asarray(Image.open(rgb_path).convert("RGB"))
        if rgb is None:
            rgb = current_rgb
        elif not np.array_equal(rgb, current_rgb):
            raise ValueError(f"RGB input differs between layer records: {rgb_path}")

        with np.load(npz_path) as payload:
            branch_scores: dict[str, np.ndarray] = {}
            branch_masses: dict[str, float] = {}
            branch_distributions: dict[str, np.ndarray] = {}
            for branch in ("dino", "vae"):
                scores = np.asarray(
                    payload[f"scores_{branch}_absolute_q000_032"], dtype=np.float64
                )
                spatial = scores_to_spatial(scores, metadata["grid"][branch])[0].astype(
                    np.float64
                )
                mass = float(np.asarray(payload[f"mass_key_{branch}_q000_032"]).reshape(-1)[0])
                spatial_sum = float(spatial.sum())
                if not np.isclose(spatial_sum, mass, rtol=1e-5, atol=1e-7):
                    raise ValueError(
                        f"{row} layer {layer} {branch}: score sum {spatial_sum} != mass {mass}"
                    )
                branch_scores[branch] = spatial
                branch_masses[branch] = mass
                branch_distributions[branch] = (
                    spatial / spatial_sum if spatial_sum > 1e-12 else np.zeros_like(spatial)
                )

        layer_records.append(
            {
                "layer": layer,
                "record_dir": record_dir,
                "visual_mass": branch_masses["dino"] + branch_masses["vae"],
                "branch_masses": branch_masses,
                "branch_scores": branch_scores,
                "branch_distributions": branch_distributions,
            }
        )

    assert rgb is not None
    visual_masses = np.asarray([item["visual_mass"] for item in layer_records], dtype=np.float64)
    if not np.isfinite(visual_masses).all() or float(visual_masses.sum()) <= 0.0:
        raise ValueError(f"invalid visual masses for {row}: {visual_masses}")
    weights = visual_masses / visual_masses.sum()
    fused: dict[str, np.ndarray] = {}
    for branch in ("dino", "vae"):
        fused[branch] = sum(
            float(weight) * item["branch_distributions"][branch]
            for weight, item in zip(weights, layer_records)
        )

    return {
        "rgb": rgb,
        "environment_step": environment_step,
        "layers": layer_records,
        "visual_masses": visual_masses,
        "weights": weights,
        "fused": fused,
    }


def _render_overlay(rgb: np.ndarray, fused: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray, dict]:
    normalized, normalization = normalize_heatmap(fused, "percentile", (1.0, 99.0))
    heatmap = _heatmap_rgb(_resize_map(normalized, rgb.shape[:2]))
    overlay = np.clip(
        np.round(rgb.astype(np.float32) * (1.0 - alpha) + heatmap.astype(np.float32) * alpha),
        0,
        255,
    ).astype(np.uint8)
    return overlay, heatmap, normalization


def _fit(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    return ImageOps.contain(image.convert("RGB"), size, method=Image.Resampling.BILINEAR)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--drawer-call", type=int, default=9)
    parser.add_argument("--moka-call", type=int, default=23)
    parser.add_argument("--denoising-step", type=int, default=9)
    parser.add_argument("--layers", type=_parse_layers, default=_parse_layers("0:30"))
    parser.add_argument("--overlay-alpha", type=float, default=0.45)
    args = parser.parse_args()
    if not 0.0 <= args.overlay_alpha <= 1.0:
        parser.error("--overlay-alpha must be in [0,1]")

    calls = {"drawer": args.drawer_call, "moka_pot": args.moka_call}
    output_dir = args.output.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: dict[str, dict[str, Any]] = {}
    for row, call_index in calls.items():
        row_data = _load_row(
            args.run_root,
            row=row,
            call_index=call_index,
            denoising_step=args.denoising_step,
            layers=args.layers,
        )
        row_outputs = {}
        for branch in ("dino", "vae"):
            overlay, heatmap, normalization = _render_overlay(
                row_data["rgb"], row_data["fused"][branch], args.overlay_alpha
            )
            overlay_path = output_dir / f"{row}_multilayer_overlay_{branch}.png"
            heatmap_path = output_dir / f"{row}_multilayer_heatmap_{branch}.png"
            Image.fromarray(overlay).save(overlay_path)
            Image.fromarray(heatmap).save(heatmap_path)
            row_outputs[branch] = {
                "overlay": str(overlay_path),
                "heatmap": str(heatmap_path),
                "normalization": normalization,
                "overlay_array": overlay,
            }
        rows[row] = {**row_data, "outputs": row_outputs}

    cell_size = (560, 300)
    row_label_width = 230
    row_height = cell_size[1] + 36
    canvas = Image.new("RGB", (row_label_width + 2 * cell_size[0], 52 + 2 * row_height), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((row_label_width + 10, 14), "Mass-weighted layers -> Current DINO", fill="black", font=font)
    draw.text((row_label_width + cell_size[0] + 10, 14), "Mass-weighted layers -> Current VAE", fill="black", font=font)
    for row_index, row in enumerate(("drawer", "moka_pot")):
        label, description = TASKS[row]
        data = rows[row]
        y = 52 + row_index * row_height
        draw.text((10, y + 12), label, fill="black", font=font)
        draw.text((10, y + 34), description, fill="black", font=font)
        draw.text(
            (10, y + 56),
            f"call={calls[row]} env_step={data['environment_step']} layers={args.layers[0]}:{args.layers[-1]}",
            fill="black",
            font=font,
        )
        for column, branch in enumerate(("dino", "vae")):
            panel = _fit(Image.fromarray(data["outputs"][branch]["overlay_array"]), cell_size)
            canvas.paste(
                panel,
                (
                    row_label_width + column * cell_size[0] + (cell_size[0] - panel.width) // 2,
                    y + 36 + (cell_size[1] - panel.height) // 2,
                ),
            )
    canvas.save(args.output)

    manifest_rows = []
    npz_payload: dict[str, np.ndarray] = {"layers": np.asarray(args.layers, dtype=np.int64)}
    for row, data in rows.items():
        npz_payload[f"{row}_visual_mass"] = data["visual_masses"].astype(np.float32)
        npz_payload[f"{row}_layer_weight"] = data["weights"].astype(np.float32)
        for branch in ("dino", "vae"):
            npz_payload[f"{row}_fused_{branch}"] = data["fused"][branch].astype(np.float32)
        manifest_rows.append(
            {
                "row": row,
                "call_index": calls[row],
                "environment_step": data["environment_step"],
                "layers": list(args.layers),
                "visual_mass": data["visual_masses"].tolist(),
                "normalized_layer_weight": data["weights"].tolist(),
                "branch_mass": {
                    branch: [item["branch_masses"][branch] for item in data["layers"]]
                    for branch in ("dino", "vae")
                },
                "source_records": [str(item["record_dir"]) for item in data["layers"]],
                "outputs": {
                    branch: {
                        key: value
                        for key, value in data["outputs"][branch].items()
                        if key != "overlay_array"
                    }
                    for branch in ("dino", "vae")
                },
            }
        )

    npz_path = args.output.with_suffix(".npz")
    np.savez_compressed(npz_path, **npz_payload)
    manifest = {
        "format": "st-wam-mot-multilayer-visual-mass-fusion-v1",
        "output": str(args.output),
        "raw_fusion": str(npz_path),
        "formula": {
            "visual_mass": "m_l = mass_dino_l + mass_vae_l",
            "layer_weight": "w_l = m_l / sum_r(m_r)",
            "within_branch_map": "p_l,b(x) = score_l,b(x) / sum_x(score_l,b(x))",
            "fused_map": "F_b(x) = sum_l(w_l * p_l,b(x))",
        },
        "attention_source": "absolute attention after softmax over the complete key sequence",
        "display": {
            "normalization": "independent percentile [1,99] after layer fusion",
            "overlay_alpha": args.overlay_alpha,
            "warning": "Color intensity is comparable only within each rendered map.",
        },
        "rows": manifest_rows,
    }
    manifest_path = args.output.with_suffix(".json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Saved multilayer fusion panel: {args.output}")
    print(f"Saved manifest: {manifest_path}")
    print(f"Saved raw fusion arrays: {npz_path}")


if __name__ == "__main__":
    main()
