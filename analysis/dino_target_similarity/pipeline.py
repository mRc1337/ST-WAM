"""End-to-end runner for DINOv3 target correspondence analysis."""

from __future__ import annotations

import json
import logging
import os
import re
import warnings
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd
import torch
import yaml

from .features import DINOFeatureExtractor
from .io import (
    EpisodeRecord,
    discover_episodes,
    get_evaluation_gt_mask,
    get_reference_mask,
    get_rollout_gt_mask,
    load_annotations,
    load_rollout_frames,
    parse_episode_path,
    resolve_episode_annotation,
)
from .prototype import build_random_prototypes, build_target_prototype, image_mask_to_patch_mask, resize_binary_mask
from .metrics import compute_frame_metrics
from .statistics import (
    CORE_AGGREGATE_METRICS,
    aggregate_trajectory,
    compute_effects_and_auc,
    episode_summaries,
    failure_precursors,
)
from .visualize import (
    normalize_similarity_map,
    save_aggregate_plots,
    save_episode_figure,
    save_failure_lead_time_plot,
    save_heatmap_video,
)

LOGGER = logging.getLogger(__name__)


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        return str(Path(os.path.expandvars(value)).expanduser())
    if isinstance(value, list):
        return [_expand(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    return value


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Analysis config must be a mapping, got {type(payload)}")
    payload = _expand(payload)
    payload.setdefault("analysis", {})
    payload["analysis"].setdefault("output_dir", "outputs/dino_target_similarity")
    return payload


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True), encoding="utf-8")


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "episode"


def _episode_cache_name(record: EpisodeRecord) -> str:
    return f"{_slug(record.episode_id)}__task-{_slug(record.task_id or 'unknown')}.pt"


def _as_int_or_none(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _detect_grasp_close(auxiliary: Mapping[str, Any], annotation: Optional[Mapping[str, Any]], config: Mapping[str, Any]) -> Optional[int]:
    if annotation is not None and annotation.get("grasp_close_frame") is not None:
        return int(annotation["grasp_close_frame"])
    if auxiliary.get("grasp_close_frame") is not None:
        return _as_int_or_none(np.asarray(auxiliary["grasp_close_frame"]).reshape(-1)[0])
    values = auxiliary.get("gripper_state", auxiliary.get("gripper"))
    if values is None:
        return None
    values = np.asarray(values)
    if values.ndim > 1:
        values = values[..., -1]
    values = values.reshape(-1).astype(np.float64)
    if len(values) == 0:
        return None
    threshold = float(config.get("gripper_close_threshold", 0.5))
    direction = str(config.get("gripper_close_direction", "high")).lower()
    closed = values >= threshold if direction in {"high", "positive", "close"} else values <= threshold
    transitions = np.flatnonzero(closed & ~np.concatenate(([False], closed[:-1])))
    return int(transitions[0]) if len(transitions) else (int(np.flatnonzero(closed)[0]) if closed.any() else None)


def _failure_frame(record: EpisodeRecord, auxiliary: Mapping[str, Any], annotation: Optional[Mapping[str, Any]], frame_count: int) -> tuple[int, str]:
    for source in (annotation or {}, record.metadata, auxiliary):
        for key in ("failure_frame", "failure_event_frame", "terminal_frame"):
            if source.get(key) is not None:
                return int(source[key]), "metadata"
    return max(frame_count - 1, 0), "terminal_proxy"


def _merge_embedded_rollout_metadata(record: EpisodeRecord, auxiliary: Mapping[str, Any]) -> EpisodeRecord:
    raw = auxiliary.get("metadata_json")
    if raw is None:
        return record
    try:
        value = np.asarray(raw).reshape(-1)[0]
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        embedded = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError, IndexError):
        return record
    if not isinstance(embedded, Mapping):
        return record
    record.metadata.update(dict(embedded))
    if embedded.get("episode_id"):
        record.episode_id = str(embedded["episode_id"])
    if record.success is None:
        record.success = embedded.get("success")
    if record.task_id is None and embedded.get("task_id") is not None:
        record.task_id = str(embedded["task_id"])
    if embedded.get("task_name"):
        record.task_name = str(embedded["task_name"])
    if embedded.get("shift_condition"):
        record.shift_condition = str(embedded["shift_condition"])
    if embedded.get("failure_reason"):
        record.failure_reason = str(embedded["failure_reason"])
    if embedded.get("failure_class"):
        record.failure_class = str(embedded["failure_class"])
    if embedded.get("grasp_relevant_failure"):
        record.grasp_relevant_failure = str(embedded["grasp_relevant_failure"])
    return record


def _model_image_size(feature_map: torch.Tensor, patch_size: int) -> tuple[int, int]:
    return int(feature_map.shape[0]) * patch_size, int(feature_map.shape[1]) * patch_size


def _load_records(config: Mapping[str, Any]) -> list[EpisodeRecord]:
    input_cfg = config.get("input", {})
    episode_ids = input_cfg.get("episode_ids")
    task_ids = input_cfg.get("task_ids")
    return discover_episodes(
        input_cfg["path"],
        metadata_path=input_cfg.get("metadata_path"),
        max_episodes=input_cfg.get("max_episodes"),
        episode_ids=episode_ids,
        task_ids=task_ids,
        include_success=bool(input_cfg.get("include_success", True)),
        include_failure=bool(input_cfg.get("include_failure", True)),
    )


def _record_manifest(record: EpisodeRecord, *, cache_path: Path, frame_count: int, camera_shapes: Mapping[str, Any], auxiliary_keys: list[str]) -> dict[str, Any]:
    return {
        "path": str(record.path),
        "episode_id": record.episode_id,
        "task_id": record.task_id,
        "task_name": record.task_name,
        "success": record.success,
        "outcome": "success" if record.success is True else "failure" if record.success is False else "unknown",
        "shift_condition": record.shift_condition,
        "failure_reason": record.failure_reason,
        "failure_class": record.failure_class,
        "grasp_relevant_failure": record.grasp_relevant_failure,
        "seed": record.seed,
        "feature_cache": str(cache_path),
        "frame_count": int(frame_count),
        "camera_shapes": {key: list(value) for key, value in camera_shapes.items()},
        "auxiliary_keys": auxiliary_keys,
        "metadata": record.metadata,
    }


def extract_features(config: Mapping[str, Any], run_dir: Path) -> dict[str, Any]:
    """Encode all selected episodes with one frozen encoder instance."""

    root = Path(config.get("repo_root", Path(__file__).resolve().parents[2]))
    model_cfg = config["model"]
    analysis_cfg = config.get("analysis", {})
    input_cfg = config.get("input", {})
    camera_names = [str(x) for x in input_cfg.get("camera_names", ["front", "wrist"])]
    extractor = DINOFeatureExtractor(
        model_cfg,
        repo_root=root,
        resolution_mode=str(analysis_cfg.get("resolution_mode", "native")),
        device=str(analysis_cfg.get("device", "auto")),
        dtype=str(analysis_cfg.get("dtype", "auto")),
        view_mode=str(analysis_cfg.get("view_mode", "current_concat")),
    )
    records = _load_records(config)
    feature_dir = run_dir / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    cache_dtype = str(analysis_cfg.get("feature_cache_dtype", "float16")).lower()
    for index, record in enumerate(records):
        LOGGER.info("Extracting DINO features %d/%d: %s", index + 1, len(records), record.path)
        camera_frames, auxiliary = load_rollout_frames(
            record,
            video_layout=str(input_cfg.get("video_layout", "side_by_side")),
            camera_names=camera_names,
        )
        record = _merge_embedded_rollout_metadata(record, auxiliary)
        features = extractor.encode_cameras(camera_frames)
        if cache_dtype in {"float16", "fp16"}:
            features = {key: value.half() for key, value in features.items()}
        elif cache_dtype in {"bfloat16", "bf16"}:
            features = {key: value.bfloat16() for key, value in features.items()}
        elif cache_dtype not in {"float32", "fp32"}:
            raise ValueError(f"Unsupported feature_cache_dtype={cache_dtype!r}")
        cache_path = feature_dir / _episode_cache_name(record)
        payload = {
            "features": features,
            "metadata": extractor.metadata(camera_names),
            "episode": {
                "episode_id": record.episode_id,
                "task_id": record.task_id,
                "path": str(record.path),
            },
            "auxiliary": {key: value for key, value in auxiliary.items() if isinstance(value, (np.ndarray, int, float, str, bool))},
        }
        torch.save(payload, cache_path)
        entries.append(
            _record_manifest(
                record,
                cache_path=cache_path,
                frame_count=len(next(iter(camera_frames.values()))),
                camera_shapes={key: list(value.shape) for key, value in camera_frames.items()},
                auxiliary_keys=sorted(auxiliary),
            )
        )
    manifest = {
        "stage": "features",
        "encoder": extractor.metadata(camera_names),
        "episodes": entries,
        "input": input_cfg,
    }
    _write_json(run_dir / "run_manifest.json", manifest)
    return manifest


def _load_feature_manifest(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "run_manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Feature manifest does not exist: {path}; run --stage extract first")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not payload.get("episodes"):
        raise ValueError(f"Feature manifest has no episodes: {path}")
    return payload


def _scalar_metric_row(metric_values: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in metric_values.items()
        if not isinstance(value, (np.ndarray, torch.Tensor, list, tuple, dict))
    }


def analyze_features(config: Mapping[str, Any], run_dir: Path, manifest: Optional[Mapping[str, Any]] = None) -> pd.DataFrame:
    """Create frame metrics, prototypes, maps, plots, statistics, and summary."""

    manifest = dict(manifest or _load_feature_manifest(run_dir))
    input_cfg = config.get("input", {})
    analysis_cfg = config.get("analysis", {})
    camera_names = [str(x) for x in input_cfg.get("camera_names", ["front", "wrist"])]
    annotations = load_annotations(input_cfg.get("annotation_path"))
    patch_size = int(config["model"].get("patch_size", 16))
    min_patch_overlap = float(analysis_cfg.get("min_patch_overlap", 0.5))
    min_target_patches = int(analysis_cfg.get("min_target_patches", 1))
    temperature = float(analysis_cfg.get("similarity_temperature", 0.07))
    random_count = int(analysis_cfg.get("random_prototypes", 1))
    random_seed = int(analysis_cfg.get("random_seed", 17))
    reference_mode = str(analysis_cfg.get("reference_mode", "episode_reference"))
    if reference_mode not in {"episode_reference", "task_exemplar"}:
        raise ValueError(f"Unsupported reference_mode={reference_mode!r}")
    manifest_by_episode = {str(item["episode_id"]): item for item in manifest["episodes"]}
    frame_rows: list[dict[str, Any]] = []
    reference_checks: list[dict[str, Any]] = []
    missing_gt_frames = 0

    prototype_dir = run_dir / "prototypes"
    map_dir = run_dir / "heatmaps"
    figure_dir = run_dir / "figures" / "episodes"
    (run_dir / "videos").mkdir(parents=True, exist_ok=True)
    prototype_dir.mkdir(parents=True, exist_ok=True)
    map_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    for episode_entry in manifest["episodes"]:
        record = parse_episode_path(episode_entry["path"], episode_entry.get("metadata", {}))
        camera_frames, auxiliary = load_rollout_frames(
            record,
            video_layout=str(input_cfg.get("video_layout", "side_by_side")),
            camera_names=camera_names,
        )
        record = _merge_embedded_rollout_metadata(record, auxiliary)
        cache_path = Path(episode_entry["feature_cache"])
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        cache_metadata = cache.get("metadata", {})
        manifest_encoder = manifest.get("encoder", {})
        for metadata_key in (
            "model_name",
            "checkpoint_identifier",
            "input_resolution",
            "patch_size",
            "patch_grid",
            "feature_dim",
            "view_mode",
        ):
            if cache_metadata.get(metadata_key) != manifest_encoder.get(metadata_key):
                raise ValueError(
                    f"Feature cache metadata mismatch for {cache_path}: key={metadata_key!r}, "
                    f"cache={cache_metadata.get(metadata_key)!r}, manifest={manifest_encoder.get(metadata_key)!r}"
                )
        features_by_camera: Mapping[str, torch.Tensor] = cache["features"]
        annotation = resolve_episode_annotation(annotations, record)
        if annotation is None:
            if "gt_masks" in auxiliary:
                # A simulator-recorded target mask provides both reference and
                # future evaluation masks. Keep a minimal explicit reference
                # contract; no future mask is inferred from RGB.
                annotation = {
                    "reference_frame": 0,
                    "target_name": "simulator_obj_of_interest",
                    "annotation_source": "raw_rollout_gt_masks",
                }
            else:
                raise ValueError(
                    f"No target annotation for episode={record.episode_id}, task={record.task_id}. "
                    "Provide annotation_path or a raw rollout NPZ with gt_masks; "
                    "the pipeline will not infer a target from future RGB frames."
                )
        frame_count = len(next(iter(camera_frames.values())))
        grasp_close = _detect_grasp_close(auxiliary, annotation, analysis_cfg)
        failure_frame, failure_source = _failure_frame(record, auxiliary, annotation, frame_count)
        episode_maps: dict[str, list[np.ndarray]] = {}
        episode_metric_rows: dict[str, list[dict[str, Any]]] = {}
        for camera, features in features_by_camera.items():
            camera_annotation = annotation
            if isinstance(annotation.get("cameras"), Mapping):
                camera_annotation = annotation["cameras"].get(camera)
            if not isinstance(camera_annotation, Mapping):
                warnings.warn(
                    f"No target annotation for camera={camera}, episode={record.episode_id}; skipping this view.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue
            reference_frame = int(camera_annotation.get("reference_frame", annotation.get("reference_frame", 0)))
            if not 0 <= reference_frame < frame_count:
                raise ValueError(f"reference_frame={reference_frame} outside episode length {frame_count}")
            source_hw = tuple(int(x) for x in camera_frames[camera].shape[1:3])
            feature_image_size = _model_image_size(features[0], patch_size)
            reference_mask_source = get_rollout_gt_mask(
                auxiliary,
                camera=camera,
                camera_names=camera_names,
                frame_idx=reference_frame,
                source_hw=source_hw,
            )
            reference_mask_source_kind = "rollout_gt_mask" if reference_mask_source is not None else "annotation"
            if reference_mask_source is None:
                reference_mask_source = get_reference_mask(
                    camera_annotation,
                    camera=camera,
                    frame_idx=reference_frame,
                    source_hw=source_hw,
                    episode=record,
                )
            prototype_features = features
            prototype_frame = reference_frame
            prototype_source_episode = record.episode_id
            prototype_source_mask = reference_mask_source
            if reference_mode == "task_exemplar":
                exemplar_id = (
                    camera_annotation.get("exemplar_episode_id")
                    or annotation.get("exemplar_episode_id")
                    or analysis_cfg.get("exemplar_episode_by_task", {}).get(str(record.task_id))
                )
                if exemplar_id is None:
                    raise ValueError(
                        f"reference_mode=task_exemplar requires exemplar_episode_id for task={record.task_id}, episode={record.episode_id}"
                    )
                exemplar_entry = manifest_by_episode.get(str(exemplar_id))
                if exemplar_entry is None:
                    raise ValueError(f"Exemplar episode {exemplar_id!r} is not present in feature manifest")
                exemplar_record = parse_episode_path(exemplar_entry["path"], exemplar_entry.get("metadata", {}))
                exemplar_frames, exemplar_auxiliary = load_rollout_frames(
                    exemplar_record,
                    video_layout=str(input_cfg.get("video_layout", "side_by_side")),
                    camera_names=camera_names,
                )
                exemplar_cache = torch.load(Path(exemplar_entry["feature_cache"]), map_location="cpu", weights_only=False)
                exemplar_features = exemplar_cache["features"].get(camera)
                exemplar_annotation = resolve_episode_annotation(annotations, exemplar_record)
                if exemplar_annotation is None and "gt_masks" in exemplar_auxiliary:
                    exemplar_annotation = {
                        "reference_frame": 0,
                        "target_name": "simulator_obj_of_interest",
                        "annotation_source": "raw_rollout_gt_masks",
                    }
                if exemplar_features is None or exemplar_annotation is None:
                    raise ValueError(f"Exemplar {exemplar_id!r} lacks features or target annotation for camera={camera}")
                exemplar_camera_annotation = exemplar_annotation
                if isinstance(exemplar_annotation.get("cameras"), Mapping):
                    exemplar_camera_annotation = exemplar_annotation["cameras"].get(camera)
                if not isinstance(exemplar_camera_annotation, Mapping):
                    raise ValueError(f"Exemplar {exemplar_id!r} lacks target annotation for camera={camera}")
                exemplar_reference_frame = int(exemplar_camera_annotation.get("reference_frame", exemplar_annotation.get("reference_frame", 0)))
                exemplar_source_hw = tuple(int(x) for x in exemplar_frames[camera].shape[1:3])
                prototype_source_mask = get_rollout_gt_mask(
                    exemplar_auxiliary,
                    camera=camera,
                    camera_names=camera_names,
                    frame_idx=exemplar_reference_frame,
                    source_hw=exemplar_source_hw,
                )
                if prototype_source_mask is None:
                    prototype_source_mask = get_reference_mask(
                        exemplar_camera_annotation,
                        camera=camera,
                        frame_idx=exemplar_reference_frame,
                        source_hw=exemplar_source_hw,
                        episode=exemplar_record,
                    )
                prototype_features = exemplar_features
                prototype_frame = exemplar_reference_frame
                prototype_source_episode = exemplar_record.episode_id

            # A simulator mask can be empty in one view at the nominal frame
            # (for example, before the object enters the wrist camera), or in
            # every frame when that view never sees the object. For raw GT
            # rollouts, use the first frame that covers at least one DINO
            # patch as the episode reference. If no such frame exists, omit
            # only that camera; the other view still provides valid evidence.
            if reference_mode == "episode_reference" and reference_mask_source_kind == "rollout_gt_mask":
                reference_frame_candidates = [reference_frame] + [
                    frame_idx for frame_idx in range(frame_count) if frame_idx != reference_frame
                ]
                selected_reference = None
                for candidate_frame in reference_frame_candidates:
                    candidate_mask = get_rollout_gt_mask(
                        auxiliary,
                        camera=camera,
                        camera_names=camera_names,
                        frame_idx=candidate_frame,
                        source_hw=source_hw,
                    )
                    candidate_mask_model = resize_binary_mask(candidate_mask, feature_image_size)
                    candidate_patch_mask = image_mask_to_patch_mask(
                        candidate_mask_model,
                        grid_size=(int(features.shape[1]), int(features.shape[2])),
                        min_patch_overlap=min_patch_overlap,
                    )
                    if bool(candidate_patch_mask.any()):
                        selected_reference = (candidate_frame, candidate_mask)
                        break
                if selected_reference is None:
                    LOGGER.warning(
                        "Skipping camera=%s for episode=%s: simulator target mask covers no DINO patches in any frame.",
                        camera,
                        record.episode_id,
                    )
                    continue
                reference_frame, reference_mask_source = selected_reference
                prototype_frame = reference_frame
                prototype_source_mask = reference_mask_source

            prototype_mask_model = resize_binary_mask(
                prototype_source_mask,
                _model_image_size(prototype_features[prototype_frame], patch_size),
            )
            target = build_target_prototype(
                prototype_features[prototype_frame].float(),
                prototype_mask_model,
                image_size=_model_image_size(prototype_features[prototype_frame], patch_size),
                patch_size=patch_size,
                min_patch_overlap=min_patch_overlap,
                min_target_patches=min_target_patches,
            )
            controls = build_random_prototypes(
                prototype_features[prototype_frame].float(),
                target["patch_mask"],
                num_prototypes=random_count,
                seed=random_seed + sum((index + 1) * ord(char) for index, char in enumerate(record.episode_id + camera)),
            )
            prototype_path = prototype_dir / f"{_slug(record.episode_id)}__{_slug(camera)}.pt"
            torch.save(
                {
                    "prototype": target["prototype"].float(),
                    "metadata": {
                        "episode_id": record.episode_id,
                        "camera": camera,
                        "reference_frame": reference_frame,
                        "reference_mask_source_shape": list(source_hw),
                        "reference_mask_model_shape": list(feature_image_size),
                        "target_patch_count": target["target_patch_count"],
                        "reference_mode": reference_mode,
                        "prototype_source_episode": prototype_source_episode,
                        "prototype_source_frame": prototype_frame,
                        "leakage_guard": "prototype fixed at reference frame; later GT masks are evaluation-only",
                    },
                    "random_controls": controls,
                },
                prototype_path,
            )
            maps = []
            rows_for_camera: list[dict[str, Any]] = []
            previous_centroid = None
            previous_gt_center = None
            for frame_index in range(frame_count):
                evaluation_mask = get_rollout_gt_mask(
                    auxiliary,
                    camera=camera,
                    camera_names=camera_names,
                    frame_idx=frame_index,
                    source_hw=source_hw,
                )
                evaluation_mask_source = "rollout_gt_mask" if evaluation_mask is not None else None
                if evaluation_mask is None:
                    evaluation_mask = get_evaluation_gt_mask(
                        camera_annotation,
                        camera=camera,
                        frame_idx=frame_index,
                        source_hw=source_hw,
                        episode=record,
                    )
                    if evaluation_mask is not None:
                        evaluation_mask_source = "annotation"
                # The reference mask is valid at t_ref only. It is not a
                # tracking mask and is never reused for t>t_ref.
                if evaluation_mask is None and frame_index == reference_frame:
                    evaluation_mask = reference_mask_source
                    evaluation_mask_source = reference_mask_source_kind
                evaluation_mask_model = None
                if evaluation_mask is not None:
                    candidate_mask_model = resize_binary_mask(evaluation_mask, feature_image_size)
                    candidate_patch_mask = image_mask_to_patch_mask(
                        candidate_mask_model,
                        grid_size=(int(features.shape[1]), int(features.shape[2])),
                        min_patch_overlap=min_patch_overlap,
                    )
                    if bool(candidate_patch_mask.any()):
                        evaluation_mask_model = candidate_mask_model
                    else:
                        # A visible object can still be too thin/small to
                        # cover half of any DINO patch. Keep the raw map, but
                        # mark target-dependent metrics missing rather than
                        # aborting the whole trajectory. Reference-frame
                        # prototype construction remains strict above.
                        evaluation_mask_source = f"{evaluation_mask_source or 'unknown'}_zero_patch"
                metric = compute_frame_metrics(
                    features[frame_index].float(),
                    target["prototype"].float(),
                    target_mask=evaluation_mask_model,
                    previous_centroid=previous_centroid,
                    previous_gt_center=previous_gt_center,
                    temperature=temperature,
                    min_patch_overlap=min_patch_overlap,
                )
                maps.append(metric["raw_similarity_map"])
                previous_centroid = metric["predicted_centroid"]
                if np.isfinite(metric["gt_target_center_x"]):
                    previous_gt_center = np.asarray([metric["gt_target_center_x"], metric["gt_target_center_y"]], dtype=np.float32)
                if evaluation_mask_model is None:
                    missing_gt_frames += 1
                row = _scalar_metric_row(metric)
                row.update(
                    {
                        "episode_id": record.episode_id,
                        "task_id": record.task_id or "unknown",
                        "task_name": record.task_name,
                        "camera": camera,
                        "frame_index": frame_index,
                        "progress": frame_index / max(frame_count - 1, 1),
                        "timestamp": float(np.asarray(auxiliary["timestamps"])[frame_index]) if "timestamps" in auxiliary and len(np.asarray(auxiliary["timestamps"])) > frame_index else float(frame_index),
                        "outcome": "success" if record.success is True else "failure" if record.success is False else "unknown",
                        "success": record.success,
                        "failure_reason": record.failure_reason,
                        "failure_class": record.failure_class,
                        "grasp_relevant_failure": record.grasp_relevant_failure,
                        "failure_frame": failure_frame,
                        "failure_frame_source": failure_source,
                        "shift_condition": record.shift_condition,
                        "seed": record.seed,
                        "reference_frame": reference_frame,
                        "reference_mode": reference_mode,
                        "reference_mask_source": reference_mask_source_kind,
                        "prototype_source_episode": prototype_source_episode,
                        "prototype_source_frame": prototype_frame,
                        "is_reference_frame": frame_index == reference_frame,
                        "has_evaluation_gt_mask": evaluation_mask_model is not None,
                        "evaluation_gt_mask_source": evaluation_mask_source or "missing",
                        "grasp_close_frame": grasp_close,
                        "target_patch_count": target["target_patch_count"],
                        "prototype_kind": "target",
                    }
                )
                for array_key, column in (("actions", "action_json"), ("states", "state_json")):
                    if array_key in auxiliary and frame_index < len(np.asarray(auxiliary[array_key])):
                        row[column] = json.dumps(np.asarray(auxiliary[array_key])[frame_index].tolist())
                if "gripper_state" in auxiliary and frame_index < len(np.asarray(auxiliary["gripper_state"])):
                    value = np.asarray(auxiliary["gripper_state"])[frame_index]
                    row["gripper_state"] = float(np.asarray(value).reshape(-1)[-1])
                rows_for_camera.append(row)

                for control_index, control in enumerate(controls):
                    control_metric = compute_frame_metrics(
                        features[frame_index].float(),
                        control["prototype"].float(),
                        target_mask=evaluation_mask_model,
                        temperature=temperature,
                        min_patch_overlap=min_patch_overlap,
                    )
                    for key in ("target_mass", "similarity_margin", "peak_localization_error", "centroid_localization_error", "peak_similarity"):
                        row[f"random_{control_index}_{key}"] = control_metric[key]
            episode_maps[camera] = maps
            episode_metric_rows[camera] = rows_for_camera

            reference_row = rows_for_camera[reference_frame]
            reference_checks.append(
                {
                    "episode_id": record.episode_id,
                    "camera": camera,
                    "reference_frame": reference_frame,
                    "target_patch_count": target["target_patch_count"],
                    "reference_target_mean_similarity": reference_row["target_mean_similarity"],
                    "reference_background_mean_similarity": reference_row["background_mean_similarity"],
                    "reference_margin": reference_row["similarity_margin"],
                    "reference_target_mass": reference_row["target_mass"],
                }
            )

        for camera, rows_for_camera in episode_metric_rows.items():
            frame_rows.extend(rows_for_camera)
        heatmap_vmin = float(analysis_cfg.get("heatmap_vmin", -1.0))
        heatmap_vmax = float(analysis_cfg.get("heatmap_vmax", 1.0))
        raw_maps = {camera: np.stack(maps, axis=0).astype(np.float32) for camera, maps in episode_maps.items()}
        normalized_maps = {
            f"normalized_{camera}": normalize_similarity_map(
                values,
                vmin=heatmap_vmin,
                vmax=heatmap_vmax,
            ).astype(np.float32)
            for camera, values in raw_maps.items()
        }
        # Camera keys remain backward-compatible raw cosine maps. The explicit
        # normalized_* keys make the fixed visualization transform reusable
        # without recomputing it or confusing visualization values with raw
        # metrics.
        np.savez_compressed(
            map_dir / f"{_slug(record.episode_id)}.npz",
            **raw_maps,
            **normalized_maps,
            normalization_vmin=np.asarray(heatmap_vmin, dtype=np.float32),
            normalization_vmax=np.asarray(heatmap_vmax, dtype=np.float32),
            normalization_strategy=np.asarray("fixed_cosine_range"),
        )
        if bool(analysis_cfg.get("save_heatmaps", True)):
            selected_ids = analysis_cfg.get("heatmap_episode_ids")
            if selected_ids is None or record.episode_id in {str(x) for x in selected_ids}:
                save_episode_figure(
                    figure_dir / f"{_slug(record.episode_id)}.png",
                    episode_id=record.episode_id,
                    task_name=record.task_name,
                    outcome="success" if record.success is True else "failure" if record.success is False else "unknown",
                    camera_frames={key: camera_frames[key] for key in episode_maps},
                    similarity_maps=episode_maps,
                    frame_rows=pd.DataFrame([row for rows in episode_metric_rows.values() for row in rows]),
                    vmin=heatmap_vmin,
                    vmax=heatmap_vmax,
                )
                if bool(analysis_cfg.get("save_video", False)):
                    save_heatmap_video(
                        run_dir / "videos" / f"{_slug(record.episode_id)}.mp4",
                        camera_frames={key: camera_frames[key] for key in episode_maps},
                        similarity_maps=episode_maps,
                        frame_rows=pd.DataFrame([row for rows in episode_metric_rows.values() for row in rows]),
                        fps=int(analysis_cfg.get("video_fps", 12)),
                        vmin=heatmap_vmin,
                        vmax=heatmap_vmax,
                    )

    frame_df = pd.DataFrame(frame_rows)
    if frame_df.empty:
        raise ValueError("No frame metrics were produced; check annotations and camera names")
    frame_df.sort_values(["episode_id", "camera", "frame_index"], inplace=True)
    frame_df.reset_index(drop=True, inplace=True)
    # Keep frame-level data in the explicitly named artifact. The required
    # episode_metrics.csv is written below after one-row-per-episode reduction.
    frame_df.to_csv(run_dir / "frame_metrics.csv", index=False)
    try:
        frame_df.to_parquet(run_dir / "frame_metrics.parquet", index=False)
    except Exception as exc:
        warnings.warn(f"Parquet output unavailable ({exc}); writing CSV fallback", RuntimeWarning, stacklevel=2)
        frame_df.to_csv(run_dir / "frame_metrics.csv", index=False)

    all_aggregate_rows: list[dict[str, Any]] = []
    aggregate_dir = run_dir / "plots"
    cameras_in_df = sorted(str(x) for x in frame_df["camera"].unique())
    task_ids = sorted(str(x) for x in frame_df["task_id"].unique())
    shifts = sorted(str(x) for x in frame_df["shift_condition"].unique())
    for camera in cameras_in_df:
        groups = [("overall", None, None)]
        groups.extend((f"task_{task_id}", task_id, None) for task_id in task_ids)
        groups.extend((f"shift_{_slug(shift)}", None, shift) for shift in shifts)
        for prefix, task_id, shift in groups:
            rows = aggregate_trajectory(
                frame_df,
                metrics=CORE_AGGREGATE_METRICS,
                progress_points=int(analysis_cfg.get("progress_points", 100)),
                n_bootstrap=int(analysis_cfg.get("bootstrap_samples", 2000)),
                seed=int(analysis_cfg.get("random_seed", 17)),
                camera=camera,
                task_id=task_id,
                shift_condition=shift,
            )
            if not rows:
                continue
            aggregate_df = pd.DataFrame(rows)
            aggregate_df["camera"] = camera
            aggregate_df["group"] = prefix
            all_aggregate_rows.extend(aggregate_df.to_dict(orient="records"))
            save_aggregate_plots(
                aggregate_dir / camera,
                aggregate_df,
                metrics=CORE_AGGREGATE_METRICS,
                group_prefix=f"{prefix}",
            )
    aggregate_df_all = pd.DataFrame(all_aggregate_rows)
    if not aggregate_df_all.empty:
        aggregate_df_all.to_csv(run_dir / "aggregate_metrics.csv", index=False)

    summary_by_camera: dict[str, Any] = {}
    precursor_by_camera: dict[str, Any] = {}
    episode_summary_tables: list[pd.DataFrame] = []
    for camera in cameras_in_df:
        summaries = episode_summaries(frame_df, camera=camera, pregrasp_only=False)
        summaries.to_csv(run_dir / f"episode_summaries_{camera}.csv", index=False)
        episode_summary_tables.append(summaries)
        summary_by_camera[camera] = compute_effects_and_auc(summaries)
        precursor_by_camera[camera] = failure_precursors(
            frame_df,
            camera=camera,
            success_percentile=float(analysis_cfg.get("anomaly_success_percentile", 5.0)),
            persistence_frames=int(analysis_cfg.get("anomaly_persistence_frames", 1)),
        )
        save_failure_lead_time_plot(run_dir / "plots" / camera / "failure_lead_time.png", precursor_by_camera[camera])
    if episode_summary_tables:
        pd.concat(episode_summary_tables, ignore_index=True).to_csv(run_dir / "episode_metrics.csv", index=False)

    statistics = {
        "encoder": manifest.get("encoder", {}),
        "episode_count": int(frame_df["episode_id"].nunique()),
        "frame_count": int(len(frame_df)),
        "success_count": int(frame_df.loc[frame_df["outcome"] == "success", "episode_id"].nunique()),
        "failure_count": int(frame_df.loc[frame_df["outcome"] == "failure", "episode_id"].nunique()),
        "failure_class_counts": {
            str(key): int(value)
            for key, value in frame_df.loc[frame_df["outcome"] == "failure"]
            .drop_duplicates("episode_id")["failure_class"]
            .value_counts(dropna=False)
            .items()
        },
        "grasp_relevant_failure_counts": {
            str(key): int(value)
            for key, value in frame_df.loc[frame_df["outcome"] == "failure"]
            .drop_duplicates("episode_id")["grasp_relevant_failure"]
            .value_counts(dropna=False)
            .items()
        },
        "camera_count": len(cameras_in_df),
        "cameras": cameras_in_df,
        "frames_without_evaluation_gt_mask": int(missing_gt_frames),
        "frames_with_future_evaluation_gt_mask": int(
            (frame_df["has_evaluation_gt_mask"] & ~frame_df["is_reference_frame"]).sum()
        ),
        "reference_self_similarity": reference_checks,
        "random_control_reference": frame_df[
            frame_df["is_reference_frame"]
        ][
            [
                column
                for column in [
                    "episode_id",
                    "camera",
                    "target_mass",
                    "similarity_margin",
                    "peak_localization_error",
                    "random_0_target_mass",
                    "random_0_similarity_margin",
                    "random_0_peak_localization_error",
                ]
                if column in frame_df.columns
            ]
        ].to_dict(orient="records"),
        "by_camera": summary_by_camera,
        "failure_precursors": precursor_by_camera,
        "random_control_columns": [column for column in frame_df.columns if column.startswith("random_")],
        "metric_scope_note": "Reference mask is used only at reference_frame; target/background/localization metrics are NaN after that unless frame-specific GT masks are supplied.",
        "heatmap_visualization_normalization": {
            "strategy": "fixed_cosine_range",
            "vmin": float(analysis_cfg.get("heatmap_vmin", -1.0)),
            "vmax": float(analysis_cfg.get("heatmap_vmax", 1.0)),
            "raw_maps_are_preserved_under_camera_keys": True,
            "normalized_maps_are_saved_under": "normalized_<camera>",
        },
    }
    _write_json(run_dir / "statistics.json", statistics)
    _write_summary(run_dir / "summary.md", config, statistics, frame_df)
    return frame_df


def _write_summary(path: Path, config: Mapping[str, Any], statistics: Mapping[str, Any], frame_df: pd.DataFrame) -> None:
    encoder = statistics.get("encoder", {})
    lines = [
        "# DINO target correspondence analysis",
        "",
        "## Representation",
        "",
        f"- model: `{encoder.get('model_name')}`",
        f"- checkpoint: `{encoder.get('checkpoint')}`",
        f"- checkpoint identifier: `{encoder.get('checkpoint_identifier')}`",
        f"- input resolution: `{encoder.get('input_resolution')}` ({encoder.get('resolution_mode')})",
        f"- patch size/grid: `{encoder.get('patch_size')}` / `{encoder.get('patch_grid')}`",
        f"- feature dimension: `{encoder.get('feature_dim')}`",
        f"- view mode: `{encoder.get('view_mode')}`",
        f"- preprocessing: `{encoder.get('normalization')}`",
        "",
        "## Data",
        "",
        f"- episodes: {statistics.get('episode_count')} (success={statistics.get('success_count')}, failure={statistics.get('failure_count')})",
        f"- cameras analyzed: `{statistics.get('cameras')}`",
        f"- frames without frame-specific evaluation GT: {statistics.get('frames_without_evaluation_gt_mask')}",
        f"- failure classes: `{statistics.get('failure_class_counts', {})}`; grasp-relevance: `{statistics.get('grasp_relevant_failure_counts', {})}`",
        "- success/failure labels are read from rollout metadata; no policy intervention is performed.",
        "",
        "## Automatic observations",
        "",
        f"- Reference-frame target/background margin values: `{[round(float(item['reference_margin']), 4) for item in statistics.get('reference_self_similarity', []) if item.get('reference_margin') is not None]}`.",
        f"- Random-prototype comparison columns: `{statistics.get('random_control_columns', [])}`.",
        f"- Frame-specific GT availability after the reference frame: `{statistics.get('frames_with_future_evaluation_gt_mask', 0)}` rows; missing rows remain NaN.",
        "- Failure-precursor results are only interpretable when frame-specific GT or a non-GT confidence metric is available before the physical failure event.",
        "",
        "## Leakage guard and interpretation",
        "",
        "The prototype is computed once from the annotated reference frame and is never updated. The reference mask is used as an evaluation mask only at the reference frame; later target metrics require frame-specific GT masks. Missing GT is reported as NaN rather than filled with the reference mask.",
        "",
        "## Primary evidence",
        "",
        "See `statistics.json`, `frame_metrics.parquet`, `aggregate_metrics.csv`, `figures/episodes/`, and `plots/`. Bootstrap intervals are episode-level after normalized-progress interpolation.",
        "",
        "## Caveats",
        "",
        "- Historical evaluator MP4s do not contain actions, robot state, segmentation, or physical failure timestamps; their terminal frame is only a failure-event proxy.",
        "- The historical MP4 RGB contains the evaluator's small `image`/`wrist_image` text overlay; use the optional raw NPZ recorder for clean RGB in the next run.",
        "- Manual/annotation quality and small targets relative to a 16-pixel patch can dominate correspondence metrics.",
        "- DINO correspondence association does not establish that DINO causes success; this is an offline diagnostic.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_from_existing_run(config: Mapping[str, Any], run_dir: Path) -> None:
    path = run_dir / "frame_metrics.parquet"
    if not path.exists():
        path = run_dir / "frame_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(f"No frame metrics found in {run_dir}")
    frame_df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    analysis_cfg = config.get("analysis", {})
    for camera in sorted(frame_df["camera"].unique()):
        for prefix, task_id, shift in [("overall", None, None)] + [
            (f"task_{task}", str(task), None) for task in sorted(frame_df["task_id"].astype(str).unique())
        ] + [
            (f"shift_{_slug(shift)}", None, shift) for shift in sorted(frame_df["shift_condition"].unique())
        ]:
            rows = aggregate_trajectory(
                frame_df,
                metrics=CORE_AGGREGATE_METRICS,
                progress_points=int(analysis_cfg.get("progress_points", 100)),
                n_bootstrap=int(analysis_cfg.get("bootstrap_samples", 2000)),
                seed=int(analysis_cfg.get("random_seed", 17)),
                camera=str(camera),
                task_id=task_id,
                shift_condition=shift,
            )
            if rows:
                save_aggregate_plots(run_dir / "plots" / str(camera), pd.DataFrame(rows), metrics=CORE_AGGREGATE_METRICS, group_prefix=prefix)
