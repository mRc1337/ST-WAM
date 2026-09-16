#!/usr/bin/env python3
"""Deterministic, resumable held-out evaluation for one ST-WAM checkpoint.

The full validation set is evaluated with the same per-window diffusion noise
for every checkpoint.  Expensive action diffusion is evaluated on a fixed,
evenly-spaced subset from every held-out episode.  Results are appended to
JSONL so an interrupted job can be restarted with the identical command.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fastwam.trainer import Wan22Trainer  # noqa: E402
from fastwam.utils import misc  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--stats", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-loss-samples",
        type=int,
        default=0,
        help="0 evaluates all validation windows; otherwise use an evenly spaced subset.",
    )
    parser.add_argument(
        "--action-samples-per-episode",
        type=int,
        default=10,
        help="Expensive action-inference windows per held-out episode; 0 disables it.",
    )
    parser.add_argument("--inference-steps", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true", help="Build only the validation dataset and index plan.")
    return parser.parse_args()


def require_file(path: Path, label: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path


def set_sample_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evenly_spaced(start: int, stop: int, count: int) -> list[int]:
    size = stop - start
    if count <= 0 or size <= 0:
        return []
    if count >= size:
        return list(range(start, stop))
    return sorted({int(round(x)) for x in np.linspace(start, stop - 1, count)})


def episode_ranges(dataset: Any) -> list[tuple[int, int, int]]:
    bounds = dataset.lerobot_dataset.episode_data_index
    starts = [int(x) for x in bounds["from"]]
    stops = [int(x) for x in bounds["to"]]
    selected = dataset.lerobot_dataset.selected_episodes
    source_episode_ids = [x for values in selected.values() for x in values]
    if len(source_episode_ids) != len(starts):
        source_episode_ids = list(range(len(starts)))
    return list(zip(source_episode_ids, starts, stops))


def locate_episode(index: int, ranges: list[tuple[int, int, int]]) -> tuple[int, int]:
    for episode_id, start, stop in ranges:
        if start <= index < stop:
            return episode_id, index - start
    raise IndexError(index)


def load_records(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    records: dict[tuple[str, int], dict[str, Any]] = {}
    if not path.exists():
        return records
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            records[(str(record["record_type"]), int(record["dataset_index"]))] = record
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            print(f"Ignoring malformed JSONL line {line_no}: {path}", file=sys.stderr)
    return records


def append_record(handle: Any, records: dict, record: dict[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    handle.flush()
    os.fsync(handle.fileno())
    records[(record["record_type"], record["dataset_index"])] = record


def action_metrics(dataset: Any, sample: dict, pred_action: torch.Tensor, gt_action: torch.Tensor) -> dict:
    proxy = SimpleNamespace(val_dataset=dataset)
    return Wan22Trainer._compute_eval_action_metrics(proxy, sample, pred_action, gt_action)


@torch.inference_mode()
def infer_action(
    model: Any, sample: dict, inference_steps: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    video0 = sample["video"][0]
    prompt = sample["prompt"][0]
    gt_action = sample["action"][0]
    proprio = sample["proprio"][0, 0]
    kwargs: dict[str, Any] = {
        "input_image": video0[:, 0].unsqueeze(0),
        "action_horizon": sample["action_horizon"],
        "proprio": proprio,
        "num_inference_steps": inference_steps,
        "seed": seed,
        "tiled": False,
    }
    if sample.get("context") is not None:
        kwargs.update(
            prompt=None,
            context=sample["context"][0],
            context_mask=sample["context_mask"][0],
        )
    else:
        kwargs["prompt"] = prompt
    for key in ("history_dino_latents", "history_vae_latents"):
        if sample.get(key) is not None:
            kwargs[key] = sample[key][0]
    if sample.get("history_video") is not None:
        kwargs["history_video"] = sample["history_video"]
    if sample.get("semantic_image") is not None:
        kwargs["semantic_image"] = sample["semantic_image"][0]
    if getattr(model, "semantic_history_encoder", None) is not None:
        kwargs["semantic_prompt"] = prompt
    prediction = model.infer_action(**kwargs)
    return prediction["action"], gt_action


def metric_summary(records: list[dict[str, Any]], keys: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {"count": len(records)}
    for key in keys:
        values = [float(r[key]) for r in records if key in r and math.isfinite(float(r[key]))]
        if values:
            result[key] = {
                "mean": statistics.fmean(values),
                "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
                "count": len(values),
            }
    return result


def write_summary(
    path: Path,
    args: argparse.Namespace,
    dataset_len: int,
    loss_indices: list[int],
    action_indices: list[int],
    records: dict[tuple[str, int], dict[str, Any]],
) -> None:
    loss_records = [records[("loss", i)] for i in loss_indices if ("loss", i) in records]
    action_records = [records[("action", i)] for i in action_indices if ("action", i) in records]
    per_episode: dict[str, Any] = {}
    episode_ids = sorted({int(r["episode_id"]) for r in loss_records + action_records})
    keys = ["loss_total", "loss_video", "loss_dino", "loss_action", "action_l1", "action_l2"]
    for episode_id in episode_ids:
        selected = [r for r in loss_records + action_records if int(r["episode_id"]) == episode_id]
        per_episode[str(episode_id)] = metric_summary(selected, keys)
    payload = {
        "complete": len(loss_records) == len(loss_indices) and len(action_records) == len(action_indices),
        "run_config": str(args.run_config),
        "checkpoint": str(args.checkpoint),
        "stats": str(args.stats),
        "seed": args.seed,
        "dataset_windows": dataset_len,
        "expected_loss_samples": len(loss_indices),
        "completed_loss_samples": len(loss_records),
        "expected_action_samples": len(action_indices),
        "completed_action_samples": len(action_records),
        "inference_steps": args.inference_steps,
        "global": metric_summary(loss_records + action_records, keys),
        "per_episode": per_episode,
        "notes": (
            "loss_* are deterministic fixed-noise diffusion objectives (already loss-weighted by the model); "
            "action_l1/l2 are denormalized action metrics on a fixed subset. PSNR/SSIM are not computed."
        ),
    }
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(path)


def main() -> None:
    args = parse_args()
    if args.max_loss_samples < 0 or args.action_samples_per_episode < 0:
        raise ValueError("sample counts must be >= 0")
    if args.inference_steps <= 0:
        raise ValueError("--inference-steps must be positive")
    args.run_config = require_file(args.run_config, "run config")
    args.checkpoint = require_file(args.checkpoint, "checkpoint")
    args.stats = require_file(args.stats, "dataset stats")
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records_path = args.output_dir / "per_window.jsonl"
    summary_path = args.output_dir / "summary.json"
    spec_path = args.output_dir / "run_spec.json"
    run_spec = {
        "run_config": str(args.run_config),
        "checkpoint": str(args.checkpoint),
        "stats": str(args.stats),
        "seed": args.seed,
        "max_loss_samples": args.max_loss_samples,
        "action_samples_per_episode": args.action_samples_per_episode,
        "inference_steps": args.inference_steps,
    }
    if spec_path.exists():
        existing_spec = json.loads(spec_path.read_text(encoding="utf-8"))
        if existing_spec != run_spec:
            raise RuntimeError(
                f"Output directory belongs to a different evaluation spec: {spec_path}\n"
                f"existing={existing_spec}\nrequested={run_spec}"
            )
    else:
        spec_path.write_text(json.dumps(run_spec, indent=2), encoding="utf-8")
    if summary_path.exists():
        try:
            if json.loads(summary_path.read_text(encoding="utf-8")).get("complete"):
                print(f"Already complete: {summary_path}")
                return
        except (json.JSONDecodeError, OSError):
            pass

    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError(f"CUDA is unavailable, cannot use {args.device}")
    cfg = OmegaConf.load(args.run_config)
    cfg.data.val.pretrained_norm_stats = str(args.stats)
    runtime_dir = args.output_dir / ".runtime"
    runtime_dir.mkdir(exist_ok=True)
    misc.register_work_dir(str(runtime_dir))

    print(f"Building validation dataset from {args.run_config}", flush=True)
    dataset = instantiate(cfg.data.val)
    ranges = episode_ranges(dataset)
    all_indices = list(range(len(dataset)))
    if args.max_loss_samples and args.max_loss_samples < len(dataset):
        loss_indices = evenly_spaced(0, len(dataset), args.max_loss_samples)
    else:
        loss_indices = all_indices
    action_indices = sorted(
        index
        for _, start, stop in ranges
        for index in evenly_spaced(start, stop, args.action_samples_per_episode)
    )
    if args.dry_run:
        print(
            f"Dry run OK: dataset_windows={len(dataset)}, episodes={len(ranges)}, "
            f"loss_samples={len(loss_indices)}, action_samples={len(action_indices)}"
        )
        return
    records = load_records(records_path)
    write_summary(summary_path, args, len(dataset), loss_indices, action_indices, records)

    pending_loss = [i for i in loss_indices if ("loss", i) not in records]
    pending_action = [i for i in action_indices if ("action", i) not in records]
    if not pending_loss and not pending_action:
        print(f"All records already exist; finalized {summary_path}")
        return

    precision = str(cfg.get("mixed_precision", "bf16")).lower()
    dtype = {"no": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[precision]
    print(f"Loading model on {args.device}: {args.checkpoint}", flush=True)
    model = instantiate(cfg.model, model_dtype=dtype, device=args.device)
    model.load_checkpoint(str(args.checkpoint))
    model = model.to(args.device).eval()

    with records_path.open("a", encoding="utf-8") as handle:
        for position, index in enumerate(pending_loss, 1):
            episode_id, episode_offset = locate_episode(index, ranges)
            set_sample_seed(args.seed * 1_000_000 + index)
            sample = Wan22Trainer._to_batched_eval_sample(dataset[index])
            with torch.inference_mode():
                loss, loss_dict = model.training_loss(sample)
            record = {
                "record_type": "loss",
                "dataset_index": index,
                "episode_id": episode_id,
                "episode_offset": episode_offset,
                "loss_total": float(loss.float().item()),
                **{str(k): float(v) for k, v in loss_dict.items()},
            }
            append_record(handle, records, record)
            if position == 1 or position % 25 == 0 or position == len(pending_loss):
                print(f"loss {position}/{len(pending_loss)} (dataset index {index})", flush=True)
                write_summary(summary_path, args, len(dataset), loss_indices, action_indices, records)

        for position, index in enumerate(pending_action, 1):
            episode_id, episode_offset = locate_episode(index, ranges)
            sample_seed = args.seed * 1_000_000 + index
            set_sample_seed(sample_seed)
            sample = Wan22Trainer._to_batched_eval_sample(dataset[index])
            pred_action, gt_action = infer_action(model, sample, args.inference_steps, sample_seed)
            metrics = action_metrics(dataset, sample, pred_action, gt_action)
            record = {
                "record_type": "action",
                "dataset_index": index,
                "episode_id": episode_id,
                "episode_offset": episode_offset,
                **{str(k): float(v) for k, v in metrics.items()},
            }
            append_record(handle, records, record)
            if position == 1 or position % 10 == 0 or position == len(pending_action):
                print(f"action {position}/{len(pending_action)} (dataset index {index})", flush=True)
                write_summary(summary_path, args, len(dataset), loss_indices, action_indices, records)

    write_summary(summary_path, args, len(dataset), loss_indices, action_indices, records)
    print(f"Complete: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
