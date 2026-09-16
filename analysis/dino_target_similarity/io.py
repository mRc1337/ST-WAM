"""Input and annotation adapters for DINO target-correspondence analysis.

The evaluator historically saved side-by-side MP4s, while newer experiments
may save a lossless/raw rollout record.  This module deliberately keeps the
adapter small and explicit so that a missing state or GT mask is represented
as missing data instead of being inferred from the policy output.
"""

from __future__ import annotations

import json
import glob
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Optional

import numpy as np
from PIL import Image


_EPISODE_RE = re.compile(r"episode=(?P<episode>[^-]+)")
_TASK_TRIAL_EPISODE_RE = re.compile(r"(?:^|--)(?P<episode>task\d+_trial\d+)(?:--|$)", re.IGNORECASE)
_SUCCESS_RE = re.compile(r"success=(?P<success>True|False|true|false|1|0)")
_TASK_ID_RE = re.compile(r"task(?P<task_id>\d+)(?:[_-]|$)", re.IGNORECASE)


@dataclass
class EpisodeRecord:
    """Metadata and source path for one rollout episode."""

    path: Path
    episode_id: str
    task_id: Optional[str]
    task_name: str
    success: Optional[bool]
    shift_condition: str = "clean"
    failure_reason: str = "unknown"
    failure_class: str = "unknown"
    grasp_relevant_failure: str = "unknown"
    seed: Optional[int] = None
    metadata: dict[str, Any] = field(default_factory=dict)


def _parse_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "success"}:
        return True
    if text in {"false", "0", "no", "failure"}:
        return False
    return None


def _clean_task_name(text: str) -> str:
    text = Path(text).stem
    # Evaluator filenames use task=... and replace spaces with underscores.
    return text.replace("_", " ").strip()


def parse_episode_path(path: str | Path, metadata: Optional[Mapping[str, Any]] = None) -> EpisodeRecord:
    """Parse evaluator-style metadata, with conservative filename fallbacks."""

    path = Path(path)
    supplied = dict(metadata or {})
    name = path.name
    episode_match = _EPISODE_RE.search(name)
    task_trial_match = _TASK_TRIAL_EPISODE_RE.search(name)
    episode_id = str(
        supplied.get("episode_id")
        or (episode_match.group("episode") if episode_match else task_trial_match.group("episode") if task_trial_match else path.stem)
    )
    success_match = _SUCCESS_RE.search(name)
    success = _parse_bool(supplied.get("success"))
    if success is None and success_match:
        success = _parse_bool(success_match.group("success"))

    task_match = re.search(r"(?:^|--)task=(?P<task>.+?)(?:\.mp4|\.npz|\.json)$", name, re.IGNORECASE)
    task_name = str(supplied.get("task_name") or (task_match.group("task") if task_match else path.stem))
    task_name = _clean_task_name(task_name)
    task_id_match = _TASK_ID_RE.search(episode_id)
    task_id = supplied.get("task_id")
    if task_id is None and task_id_match:
        task_id = task_id_match.group("task_id")
    shift = str(supplied.get("shift_condition") or supplied.get("visual_shift_condition") or "clean")
    failure_reason = str(supplied.get("failure_reason") or supplied.get("raw_failure_reason") or "unknown")
    failure_class = str(supplied.get("failure_class") or supplied.get("failure_type") or "unknown")
    grasp_relevant_failure = str(supplied.get("grasp_relevant_failure") or "unknown")
    seed = supplied.get("seed", supplied.get("episode_seed"))
    return EpisodeRecord(
        path=path,
        episode_id=episode_id,
        task_id=None if task_id is None else str(task_id),
        task_name=task_name,
        success=success,
        shift_condition=shift,
        failure_reason=failure_reason,
        failure_class=failure_class,
        grasp_relevant_failure=grasp_relevant_failure,
        seed=None if seed is None else int(seed),
        metadata=supplied,
    )


def load_episode_metadata(path: str | Path | None) -> dict[str, dict[str, Any]]:
    """Load optional metadata keyed by episode id, filename, or path.

    Accepted JSON forms are a list of records, ``{"episodes": [...]}``, or a
    mapping from an episode id to a metadata object.
    """

    if path is None:
        return {}
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Episode metadata file does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "episodes" in payload:
        payload = payload["episodes"]
    records: list[tuple[str, dict[str, Any]]] = []
    if isinstance(payload, list):
        for item in payload:
            if not isinstance(item, dict):
                raise TypeError(f"Episode metadata entries must be objects, got {type(item)}")
            key = str(item.get("episode_id") or item.get("id") or item.get("path") or "")
            if key:
                records.append((key, dict(item)))
    elif isinstance(payload, dict):
        for key, item in payload.items():
            if not isinstance(item, dict):
                raise TypeError(f"Metadata for {key!r} must be an object")
            merged = dict(item)
            merged.setdefault("episode_id", key)
            records.append((str(key), merged))
    else:
        raise TypeError(f"Unsupported episode metadata JSON type: {type(payload)}")

    result: dict[str, dict[str, Any]] = {}
    for key, item in records:
        result[key] = item
        if "path" in item:
            result[str(item["path"])] = item
            result[Path(str(item["path"])).name] = item
        if "episode_id" in item:
            result[str(item["episode_id"])] = item
    return result


def discover_episodes(
    input_path: str | Path,
    *,
    metadata_path: str | Path | None = None,
    max_episodes: Optional[int] = None,
    episode_ids: Optional[Iterable[str]] = None,
    task_ids: Optional[Iterable[str]] = None,
    include_success: bool = True,
    include_failure: bool = True,
) -> list[EpisodeRecord]:
    """Discover and deterministically order video/raw rollout episodes."""

    input_path = Path(input_path)
    if any(char in str(input_path) for char in "*?[]"):
        paths = sorted(Path(path) for path in glob.glob(str(input_path), recursive=True))
    elif input_path.is_dir():
        paths = sorted(p for p in input_path.rglob("*") if p.suffix.lower() in {".mp4", ".npz"})
    elif input_path.exists():
        paths = [input_path]
    else:
        # Path.glob handles relative recursive patterns; Path().glob does not
        # accept an absolute pattern, handled above.
        paths = sorted(Path(path) for path in glob.glob(str(input_path), recursive=True))
    paths = [p for p in paths if p.suffix.lower() in {".mp4", ".npz"}]
    if not paths:
        raise FileNotFoundError(f"No rollout files found at {input_path}")

    metadata = load_episode_metadata(metadata_path)
    requested_episode_ids = None if episode_ids is None else {str(x) for x in episode_ids}
    requested_task_ids = None if task_ids is None else {str(x) for x in task_ids}
    episodes: list[EpisodeRecord] = []
    for path in paths:
        supplied = metadata.get(str(path), metadata.get(path.name, {}))
        record = parse_episode_path(path, supplied)
        if record.episode_id in metadata:
            record = parse_episode_path(path, {**record.metadata, **metadata[record.episode_id]})
        if requested_episode_ids is not None and record.episode_id not in requested_episode_ids:
            continue
        if requested_task_ids is not None and str(record.task_id) not in requested_task_ids:
            continue
        if record.success is True and not include_success:
            continue
        if record.success is False and not include_failure:
            continue
        episodes.append(record)
    episodes.sort(key=lambda item: (item.task_id or "", item.episode_id, str(item.path)))
    if max_episodes is not None:
        episodes = episodes[: int(max_episodes)]
    if not episodes:
        raise ValueError("Rollout filters selected zero episodes.")
    return episodes


def _as_rgb(frame: np.ndarray) -> np.ndarray:
    frame = np.asarray(frame)
    if frame.ndim == 2:
        frame = np.repeat(frame[..., None], 3, axis=-1)
    if frame.ndim != 3:
        raise ValueError(f"Expected RGB frame [H,W,C], got {frame.shape}")
    if frame.shape[-1] == 4:
        frame = frame[..., :3]
    if frame.shape[-1] != 3:
        raise ValueError(f"Expected 3 or 4 channels, got {frame.shape}")
    if frame.dtype != np.uint8:
        frame = np.nan_to_num(frame)
        if np.issubdtype(frame.dtype, np.floating) and float(np.nanmax(frame, initial=0.0)) <= 1.0:
            frame = frame * 255.0
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(frame)


def split_camera_frame(frame: np.ndarray, layout: str, camera_names: list[str]) -> dict[str, np.ndarray]:
    """Split one raw frame without combining camera features."""

    frame = _as_rgb(frame)
    if layout in {"single", "raw"}:
        if len(camera_names) != 1:
            raise ValueError("single/raw video layout requires exactly one camera name")
        return {camera_names[0]: frame}
    if layout in {"side_by_side", "horizontal"}:
        if frame.shape[1] % len(camera_names) != 0:
            raise ValueError(
                f"Frame width {frame.shape[1]} is not divisible by {len(camera_names)} cameras"
            )
        width = frame.shape[1] // len(camera_names)
        return {name: frame[:, index * width : (index + 1) * width] for index, name in enumerate(camera_names)}
    if layout in {"vertical", "stacked"}:
        if frame.shape[0] % len(camera_names) != 0:
            raise ValueError(
                f"Frame height {frame.shape[0]} is not divisible by {len(camera_names)} cameras"
            )
        height = frame.shape[0] // len(camera_names)
        return {name: frame[index * height : (index + 1) * height] for index, name in enumerate(camera_names)}
    raise ValueError(f"Unsupported video_layout={layout!r}")


def iter_video_frames(path: str | Path) -> Iterator[np.ndarray]:
    import imageio.v2 as imageio

    reader = imageio.get_reader(str(path))
    try:
        for frame in reader:
            yield _as_rgb(frame)
    finally:
        reader.close()


def load_rollout_frames(
    record: EpisodeRecord,
    *,
    video_layout: str,
    camera_names: list[str],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Load RGB frames and optional per-frame arrays from MP4/NPZ records.

    Returns ``(camera_frames, auxiliary)``.  Camera arrays are ``[T,H,W,3]``;
    auxiliary values may include ``actions``, ``states``, ``gripper_state``,
    ``timestamps``, ``failure_frame`` and ``gt_masks`` when present.
    """

    suffix = record.path.suffix.lower()
    if suffix == ".npz":
        payload = np.load(record.path, allow_pickle=True)
        arrays = {key: payload[key] for key in payload.files}
        if "frames" in arrays:
            frames = np.asarray(arrays["frames"])
            if frames.ndim == 5:
                camera_frames = {
                    name: np.stack([_as_rgb(frame) for frame in frames[:, index]], axis=0)
                    for index, name in enumerate(camera_names)
                }
            elif frames.ndim == 4:
                camera_frames = {camera_names[0]: np.stack([_as_rgb(frame) for frame in frames], axis=0)}
            else:
                raise ValueError(f"NPZ frames must be [T,H,W,C] or [T,V,H,W,C], got {frames.shape}")
        else:
            camera_frames = {}
            for name in camera_names:
                for key in (name, f"images_{name}", f"camera_{name}"):
                    if key in arrays:
                        camera_frames[name] = np.stack([_as_rgb(frame) for frame in arrays[key]], axis=0)
                        break
            if len(camera_frames) != len(camera_names):
                raise ValueError(f"NPZ record lacks frames or all camera arrays for {camera_names}: {record.path}")
        camera_keys = set(camera_frames) | {f"images_{name}" for name in camera_names} | {f"camera_{name}" for name in camera_names}
        auxiliary = {key: value for key, value in arrays.items() if key not in camera_keys and key != "frames"}
        return camera_frames, auxiliary

    frames_by_camera: dict[str, list[np.ndarray]] = {name: [] for name in camera_names}
    for frame in iter_video_frames(record.path):
        split = split_camera_frame(frame, video_layout, camera_names)
        for name, image in split.items():
            frames_by_camera[name].append(image)
    if not frames_by_camera[camera_names[0]]:
        raise ValueError(f"Rollout video contains no frames: {record.path}")
    return {name: np.stack(images, axis=0) for name, images in frames_by_camera.items()}, {}


def _load_mask_file(path: str | Path, target_hw: tuple[int, int]) -> np.ndarray:
    image = np.asarray(Image.open(path).convert("L"))
    image = np.asarray(Image.fromarray(image).resize((target_hw[1], target_hw[0]), Image.Resampling.NEAREST))
    return image > 127


def _format_path(template: str | Path, *, annotation: Mapping[str, Any], episode: EpisodeRecord, frame_idx: int) -> Path:
    values = {
        "episode_id": episode.episode_id,
        "task_id": episode.task_id or "",
        "frame": frame_idx,
        "frame_idx": frame_idx,
    }
    values.update({key: value for key, value in annotation.items() if isinstance(key, str)})
    return Path(str(template).format(**values))


def _bbox_mask(bbox: Iterable[float], target_hw: tuple[int, int]) -> np.ndarray:
    values = list(bbox)
    if len(values) != 4:
        raise ValueError(f"bbox must be [x0,y0,x1,y1], got {values}")
    x0, y0, x1, y1 = [int(round(float(value))) for value in values]
    height, width = target_hw
    x0, x1 = max(0, min(x0, width)), max(0, min(x1, width))
    y0, y1 = max(0, min(y0, height)), max(0, min(y1, height))
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"bbox has empty intersection with image {target_hw}: {values}")
    mask = np.zeros((height, width), dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def _select_annotation_entry(payload: Any, episode: EpisodeRecord) -> Optional[dict[str, Any]]:
    if payload is None:
        return None
    entries = payload.get("episodes", payload) if isinstance(payload, dict) else payload
    if isinstance(entries, dict):
        candidates = []
        for key in (episode.episode_id, episode.path.name, str(episode.path), str(episode.task_id or "")):
            if key and key in entries:
                candidates.append(entries[key])
        # A task-level mapping may use task_3 or task3.
        for key in (f"task_{episode.task_id}", f"task{episode.task_id}"):
            if episode.task_id and key in entries:
                candidates.append(entries[key])
        if not candidates:
            return None
        entry = candidates[0]
        return dict(entry) if isinstance(entry, dict) else None
    if isinstance(entries, list):
        exact = []
        task_level = []
        for item in entries:
            if not isinstance(item, dict):
                continue
            ids = {str(item.get(key)) for key in ("episode_id", "id", "path", "filename") if item.get(key) is not None}
            if episode.episode_id in ids or episode.path.name in ids or str(episode.path) in ids:
                exact.append(item)
            elif episode.task_id is not None and str(item.get("task_id")) == episode.task_id:
                task_level.append(item)
        if exact:
            return dict(exact[0])
        if task_level:
            return dict(task_level[0])
    return None


def load_annotations(path: str | Path | None) -> Any:
    if path is None:
        return None
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Annotation file does not exist: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_episode_annotation(payload: Any, episode: EpisodeRecord) -> Optional[dict[str, Any]]:
    return _select_annotation_entry(payload, episode)


def get_reference_mask(
    annotation: Mapping[str, Any],
    *,
    camera: str,
    frame_idx: int,
    source_hw: tuple[int, int],
    episode: EpisodeRecord | None = None,
) -> np.ndarray:
    """Read only the reference target annotation used for prototype creation."""

    camera_entry: Mapping[str, Any] = annotation
    cameras = annotation.get("cameras")
    if isinstance(cameras, Mapping):
        camera_entry = cameras.get(camera, {})
    if camera_entry.get("camera") not in {None, camera}:
        raise ValueError(f"Annotation camera={camera_entry.get('camera')!r} does not match requested {camera!r}")
    reference_frame = int(camera_entry.get("reference_frame", annotation.get("reference_frame", 0)))
    if frame_idx != reference_frame:
        raise ValueError("get_reference_mask may only be used at reference_frame; use get_evaluation_gt_mask later")
    if camera_entry.get("mask_path") is not None:
        path = _format_path(
            camera_entry["mask_path"],
            annotation=annotation,
            episode=episode or EpisodeRecord(Path("."), "", None, ""),
            frame_idx=frame_idx,
        )
        return _load_mask_file(path, source_hw)
    if camera_entry.get("bbox") is not None:
        return _bbox_mask(camera_entry["bbox"], source_hw)
    raise ValueError(f"Annotation has neither bbox nor mask_path for camera={camera}")


def get_rollout_gt_mask(
    auxiliary: Mapping[str, Any],
    *,
    camera: str,
    camera_names: list[str],
    frame_idx: int,
    source_hw: tuple[int, int],
) -> Optional[np.ndarray]:
    """Read a simulator-provided target mask from a raw rollout NPZ.

    The evaluator stores ``gt_masks`` as ``[T,V,H,W]`` in the same camera
    order as ``camera_names``.  This is a frame-specific evaluation signal;
    it is never used to update a prototype.  Returning ``None`` means that
    the source did not provide simulator masks, not that the reference mask
    should be reused.
    """

    raw = auxiliary.get("gt_masks")
    if raw is None:
        return None
    masks = np.asarray(raw)
    if masks.ndim != 4:
        raise ValueError(f"Rollout gt_masks must have shape [T,V,H,W], got {masks.shape}")
    if camera not in camera_names:
        raise ValueError(f"Camera {camera!r} is not present in rollout camera_names={camera_names}")
    camera_index = camera_names.index(camera)
    if camera_index >= masks.shape[1]:
        raise ValueError(
            f"Rollout gt_masks has {masks.shape[1]} views, cannot read camera index {camera_index} ({camera})"
        )
    if not 0 <= int(frame_idx) < masks.shape[0]:
        raise IndexError(f"frame_idx={frame_idx} outside rollout gt_masks length {masks.shape[0]}")
    mask = np.asarray(masks[int(frame_idx), camera_index]).astype(bool)
    if mask.ndim != 2:
        raise ValueError(f"Rollout gt_masks[{frame_idx},{camera_index}] must be 2D, got {mask.shape}")
    if mask.shape != tuple(source_hw):
        mask = np.asarray(
            Image.fromarray(mask.astype(np.uint8) * 255).resize(
                (int(source_hw[1]), int(source_hw[0])), Image.Resampling.NEAREST
            )
        ) > 127
    return mask


def get_evaluation_gt_mask(
    annotation: Optional[Mapping[str, Any]],
    *,
    camera: str,
    frame_idx: int,
    source_hw: tuple[int, int],
    episode: EpisodeRecord,
) -> Optional[np.ndarray]:
    """Get a frame-specific GT mask, never falling back to the reference mask."""

    if annotation is None:
        return None
    camera_entry: Mapping[str, Any] = annotation
    cameras = annotation.get("cameras")
    if isinstance(cameras, Mapping):
        camera_entry = cameras.get(camera, {})

    paths = camera_entry.get("evaluation_gt_masks", camera_entry.get("gt_masks"))
    if isinstance(paths, Mapping):
        value = paths.get(str(frame_idx), paths.get(frame_idx))
        if value is not None:
            return _load_mask_file(_format_path(value, annotation=annotation, episode=episode, frame_idx=frame_idx), source_hw)
    elif isinstance(paths, list) and frame_idx < len(paths) and paths[frame_idx] is not None:
        return _load_mask_file(_format_path(paths[frame_idx], annotation=annotation, episode=episode, frame_idx=frame_idx), source_hw)

    template = camera_entry.get("evaluation_gt_mask", camera_entry.get("gt_mask_path"))
    if template is not None:
        path = _format_path(template, annotation=annotation, episode=episode, frame_idx=frame_idx)
        if path.exists():
            return _load_mask_file(path, source_hw)

    bboxes = camera_entry.get("evaluation_gt_bboxes", camera_entry.get("gt_bboxes"))
    if isinstance(bboxes, Mapping):
        bbox = bboxes.get(str(frame_idx), bboxes.get(frame_idx))
        if bbox is not None:
            return _bbox_mask(bbox, source_hw)
    elif isinstance(bboxes, list) and frame_idx < len(bboxes) and bboxes[frame_idx] is not None:
        return _bbox_mask(bboxes[frame_idx], source_hw)
    return None
