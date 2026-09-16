import json
import inspect
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

import hydra
import numpy as np
import torch
from accelerate import PartialState
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from tqdm import tqdm

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
# LIBERO init-state files are pickled with older PyTorch behavior.
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

# try:
#     import rootutils

#     rootutils.setup_root(__file__, indicator=".python-version", pythonpath=True)
# except ModuleNotFoundError:
project_root = Path(__file__).resolve().parents[2]
project_src = project_root / "src"
libero_root = Path(os.environ.get("LIBERO_ROOT", project_root.parent / "LIBERO"))
for path in (libero_root, project_src, project_root):
    path_str = str(path)
    if path.exists() and path_str not in sys.path:
        sys.path.insert(0, path_str)

from experiments.libero.libero_utils import (
    LIBERO_ENV_RESOLUTION,
    as_uint8_rgb_image,
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    invert_gripper_action,
    quat2axisangle,
    save_prediction_video,
    save_rollout_video,
)
from fastwam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor
from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from fastwam.utils.pytorch_utils import set_global_seed
from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from libero.libero import benchmark
from experiments.libero.action_ensembler import ActionEnsembler

OmegaConf.register_new_resolver("eval", eval)
OmegaConf.register_new_resolver("max", lambda x: max(x))
OmegaConf.register_new_resolver("split", lambda s, idx: s.split("/")[int(idx)])

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _normalize_mixed_precision(mixed_precision: str) -> str:
    key = str(mixed_precision).strip().lower()
    if key not in {"no", "fp16", "bf16"}:
        raise ValueError(
            f"Unsupported mixed_precision: {mixed_precision}. "
            "Expected one of: ['no', 'fp16', 'bf16']."
        )
    return key


def _mixed_precision_to_model_dtype(mixed_precision: str) -> torch.dtype:
    precision = _normalize_mixed_precision(mixed_precision)
    if precision == "no":
        return torch.float32
    if precision == "fp16":
        return torch.float16
    return torch.bfloat16


def _resolve_eval_device(cfg: DictConfig) -> str:
    eval_device = cfg.EVALUATION.get("device")
    if eval_device is not None:
        return str(eval_device)
    return "cuda" if torch.cuda.is_available() else "cpu"


def _resolve_dataset_stats_path(cfg: DictConfig) -> Path:
    explicit = cfg.EVALUATION.get("dataset_stats_path")
    candidates: list[Path] = []

    if explicit is not None:
        candidates.append(Path(os.path.expanduser(os.path.expandvars(str(explicit)))))

    ckpt = Path(os.path.expanduser(os.path.expandvars(str(cfg.ckpt))))
    for parent in list(ckpt.parents)[:4]:
        candidates.append(parent / "dataset_stats.json")

    seen = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            return resolved

    msg = (
        "Failed to locate dataset_stats.json. Tried explicit "
        "EVALUATION.dataset_stats_path and checkpoint parent directories. "
        "Please pass EVALUATION.dataset_stats_path=/path/to/dataset_stats.json."
    )
    raise FileNotFoundError(msg)


def _load_model_checkpoint(model: torch.nn.Module, ckpt: str) -> None:
    model.load_checkpoint(ckpt)
    logging.info("Loaded checkpoint via model.load_checkpoint: %s", ckpt)
    return

    # deprecated legacy checkpoint loading
    payload = torch.load(ckpt, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Legacy checkpoint payload must be dict, got: {type(payload)}")

    if "mot" in payload and hasattr(model, "mot"):
        missing, unexpected = model.mot.load_state_dict(payload["mot"], strict=False)
        logging.warning(
            "Loaded fallback `mot` state_dict with strict=False. Missing=%d Unexpected=%d",
            len(missing),
            len(unexpected),
        )
        return

    state_dict = None
    for key in ("model_state_dict", "state_dict", "model"):
        value = payload.get(key)
        if isinstance(value, dict):
            state_dict = value
            break
    if state_dict is None and all(torch.is_tensor(v) for v in payload.values()):
        state_dict = payload
    if state_dict is None:
        raise ValueError(f"Cannot parse legacy checkpoint keys from: {ckpt}")

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    logging.warning(
        "Loaded fallback model state_dict with strict=False. Missing=%d Unexpected=%d",
        len(missing),
        len(unexpected),
    )


def _filter_call_kwargs(fn, kwargs: dict[str, Any]) -> dict[str, Any]:
    signature = inspect.signature(fn)
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return kwargs
    return {key: value for key, value in kwargs.items() if key in signature.parameters}


def _center_crop_resize(image: np.ndarray, width: int, height: int) -> np.ndarray:
    image = as_uint8_rgb_image(image)
    pil_image = Image.fromarray(image)
    src_w, src_h = pil_image.size
    scale = max(width / src_w, height / src_h)
    resized = pil_image.resize((round(src_w * scale), round(src_h * scale)), resample=Image.BILINEAR)
    rw, rh = resized.size
    left = max((rw - width) // 2, 0)
    top = max((rh - height) // 2, 0)
    cropped = resized.crop((left, top, left + width, top + height))
    return np.asarray(cropped, dtype=np.uint8)


def _rotate_observation_images_180(cfg: DictConfig) -> bool:
    return bool(cfg.EVALUATION.get("rotate_observation_images_180", True))


def _normalize_proprio(
    proprio: np.ndarray,
    processor: FastWAMProcessor,
) -> torch.Tensor:
    state_meta = processor.shape_meta["state"]
    if len(state_meta) != 1:
        raise ValueError(
            "LIBERO eval currently expects a single merged state key in shape_meta['state']."
        )
    state_key = state_meta[0]["key"]

    state_batch = {"state": {state_key: torch.as_tensor(proprio, dtype=torch.float32).unsqueeze(0)}}
    state_batch = processor.action_state_transform(state_batch)
    state_batch = processor.normalizer.forward(state_batch)
    return state_batch["state"][state_key]


def _obs_to_model_image(
    obs: dict,
    cfg: DictConfig,
    processor: FastWAMProcessor,
    width: int,
    height: int,
    device: str,
    dtype: torch.dtype,
):
    imgs = get_libero_image(
        obs,
        rotate_180=_rotate_observation_images_180(cfg),
    )
    image_meta = processor.shape_meta["images"]
    if len(image_meta) < int(processor.num_output_cameras):
        raise ValueError(
            f"shape_meta.images has {len(image_meta)} entries, "
            f"but num_output_cameras={processor.num_output_cameras}."
        )

    def _meta_to_hw(meta: dict, camera_idx: int) -> tuple[int, int]:
        shape = meta["shape"]
        if len(shape) != 3:
            raise ValueError(f"shape_meta.images[{camera_idx}].shape must be [C,H,W], got {shape}")
        return int(shape[1]), int(shape[2])

    concatenation = cfg.data.train.get("concat_multi_camera", "horizontal")
    num_cameras = processor.num_output_cameras
    if num_cameras == 1:
        primary_h, primary_w = _meta_to_hw(image_meta[0], camera_idx=0)
        rgb = _center_crop_resize(imgs["image"], width=primary_w, height=primary_h)
    elif num_cameras == 2:
        primary_h, primary_w = _meta_to_hw(image_meta[0], camera_idx=0)
        wrist_h, wrist_w = _meta_to_hw(image_meta[1], camera_idx=1)
        primary = _center_crop_resize(imgs["image"], width=primary_w, height=primary_h)
        wrist = _center_crop_resize(imgs["wrist_image"], width=wrist_w, height=wrist_h)
        if concatenation == "horizontal":
            rgb = np.concatenate([primary, wrist], axis=1)
        elif concatenation == "vertical":
            rgb = np.concatenate([primary, wrist], axis=0)
        else:
            raise ValueError(f"Invalid concat_multi_camera: {concatenation}")
    else:
        raise ValueError(f"LIBERO eval currently supports num_output_cameras in [1, 2], got {num_cameras}.")

    actual_h, actual_w = int(rgb.shape[0]), int(rgb.shape[1])
    expected_h, expected_w = int(height), int(width)
    image_shapes = [meta["shape"] for meta in image_meta]
    assert actual_h == expected_h and actual_w == expected_w, (
        "Input image size mismatch after per-camera resize + concat: "
        f"got (H,W)=({actual_h},{actual_w}), expected (H,W)=({expected_h},{expected_w}) "
        f"from data.train.video_size={[expected_h, expected_w]}; "
        f"shape_meta.images={image_shapes}, concat_multi_camera={concatenation}."
    )

    x = torch.tensor(rgb).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=dtype)
    x = x * (2.0 / 255.0) - 1.0

    return x, imgs


def _obs_to_model_input(
    obs: dict,
    cfg: DictConfig,
    processor: FastWAMProcessor,
    width: int,
    height: int,
    device: str,
    dtype: torch.dtype,
):
    x, imgs = _obs_to_model_image(
        obs,
        cfg=cfg,
        processor=processor,
        width=width,
        height=height,
        device=device,
        dtype=dtype,
    )
    proprio = _normalize_proprio(_extract_sim_state(obs), processor)

    return x, proprio, imgs


def _extract_sim_state(obs: dict) -> np.ndarray:
    """Build simulator state from current observation.

    This is used as proprio input for model inference.
    """
    state = np.concatenate(
        (
            obs["robot0_eef_pos"],
            quat2axisangle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"],
        )
    ).astype(np.float32)
    return state


def _denormalize_action(action: torch.Tensor, processor: FastWAMProcessor) -> np.ndarray:
    if action.ndim == 2:
        action = action.unsqueeze(0)
    if action.ndim != 3:
        raise ValueError(f"Expected action tensor [B, T, D], got {tuple(action.shape)}")

    action_meta = processor.shape_meta["action"]
    if len(action_meta) != 1:
        raise ValueError(
            "LIBERO eval currently expects a single merged action key in shape_meta['action']."
        )

    action_key = action_meta[0]["key"]
    normalizer = processor.normalizer.normalizers["action"][action_key]
    action = action.to(dtype=torch.float32, device="cpu")
    denorm = normalizer.backward(action)
    return denorm.numpy()


def _postprocess_gripper_action(action: np.ndarray, cfg: DictConfig) -> np.ndarray:
    """Convert the dataset gripper convention to LIBERO simulator actions."""
    mode = str(cfg.EVALUATION.get("gripper_action_mode", "legacy_rlds")).strip().lower()
    if mode == "legacy_rlds":
        # Legacy Fast-WAM data stores gripper openness in [0, 1]. Convert it
        # to the LIBERO convention: -1=open and +1=close.
        action[..., -1] = action[..., -1] * 2 - 1
        action = invert_gripper_action(action)
    elif mode == "libero_raw":
        # LeRobot v3 LIBERO keeps source.action_raw unchanged, including the
        # native -1=open and +1=close gripper command.
        pass
    else:
        raise ValueError(
            f"Unsupported EVALUATION.gripper_action_mode={mode!r}. "
            "Expected one of: ['legacy_rlds', 'libero_raw']."
        )

    if bool(cfg.EVALUATION.get("binarize_gripper", False)):
        action[..., -1] = np.sign(action[..., -1])
    return action


def _analysis_rollout_enabled(cfg: DictConfig) -> bool:
    analysis_cfg = cfg.get("analysis", {})
    # Recording simulator masks necessarily requires the raw rollout record.
    return bool(
        analysis_cfg.get("record_rollout", False)
        or analysis_cfg.get("record_gt_masks", False)
    )


def _analysis_gt_masks_enabled(cfg: DictConfig) -> bool:
    return bool(cfg.get("analysis", {}).get("record_gt_masks", False))


def _rotate_analysis_mask(mask: np.ndarray, cfg: DictConfig) -> np.ndarray:
    """Apply the same camera-orientation transform as the RGB recorder."""

    mask = np.asarray(mask).astype(bool)
    if _rotate_observation_images_180(cfg):
        mask = mask[::-1, ::-1]
    return np.ascontiguousarray(mask)


def _analysis_target_instances(env: Any, cfg: DictConfig) -> list[str]:
    """Resolve the manipulated object(s) to keep in simulator GT masks.

    LIBERO's ``get_segmentation_of_interest`` intentionally combines every
    BDDL ``obj_of_interest`` instance, which commonly includes both the
    manipulated object and its receptacle.  Target correspondence should not
    silently treat the receptacle as target, so selection is explicit.
    """

    analysis_cfg = cfg.get("analysis", {})
    configured = analysis_cfg.get("target_instances")
    if configured is None:
        by_task = analysis_cfg.get("target_instances_by_task")
        task_id = cfg.get("EVALUATION", {}).get("task_id")
        if by_task is not None and task_id is not None:
            configured = by_task.get(str(task_id), by_task.get(int(task_id)))
    if configured is None:
        available = [str(item) for item in getattr(env, "obj_of_interest", [])]
        raise RuntimeError(
            "analysis.record_gt_masks=true requires analysis.target_instances "
            "(or target_instances_by_task) because LIBERO obj_of_interest may "
            f"include a receptacle. Available instances: {available}"
        )
    if isinstance(configured, str):
        configured = [configured]
    target_instances = [str(item) for item in configured]
    if not target_instances:
        raise ValueError("analysis.target_instances must contain at least one instance name")
    available = {str(item) for item in getattr(env, "obj_of_interest", [])}
    missing = [item for item in target_instances if item not in available]
    if missing:
        raise ValueError(
            f"analysis.target_instances={target_instances} contains unknown instance(s) {missing}; "
            f"available obj_of_interest={sorted(available)}"
        )
    instance_to_id = {str(key): int(value) for key, value in getattr(env, "instance_to_id", {}).items()}
    missing_ids = [item for item in target_instances if item not in instance_to_id]
    if missing_ids:
        raise RuntimeError(
            f"LIBERO segmentation mapping has no ID for target instance(s) {missing_ids}; "
            f"mapping keys={sorted(instance_to_id)}"
        )
    return target_instances


def _extract_analysis_gt_masks(
    obs: dict[str, Any],
    env: Any,
    cfg: DictConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Extract target-only simulator masks for the two LIBERO cameras.

    LIBERO's ``SegmentationRenderEnv`` can combine all BDDL
    ``obj_of_interest`` instances, including a destination receptacle.  We
    instead select the explicitly configured manipulated instance IDs, so
    background / robot / receptacle pixels cannot be mistaken for the target.
    Missing segmentation is an error when explicitly requested; silently
    writing an incomplete GT trajectory would invalidate the study.
    """

    if not hasattr(env, "instance_to_id"):
        raise RuntimeError(
            "analysis.record_gt_masks=true requires LIBERO SegmentationRenderEnv; "
            "the active environment does not expose instance_to_id segmentation mapping."
        )

    target_instances = _analysis_target_instances(env, cfg)
    instance_to_id = {str(key): int(value) for key, value in getattr(env, "instance_to_id", {}).items()}
    segmentation_keys = {
        "front": "agentview_segmentation_instance",
        "wrist": "robot0_eye_in_hand_segmentation_instance",
    }
    masks: list[np.ndarray] = []
    for camera, key in segmentation_keys.items():
        if key not in obs:
            raise RuntimeError(
                f"analysis.record_gt_masks=true but observation key {key!r} is missing. "
                f"Available keys include: {sorted(str(item) for item in obs if 'segmentation' in str(item))}"
            )
        segmentation = np.asarray(obs[key])
        if segmentation.ndim == 3 and segmentation.shape[-1] == 1:
            segmentation = segmentation[..., 0]
        if segmentation.ndim != 2:
            raise ValueError(
                f"Expected {key} to have shape [H,W] or [H,W,1], got {segmentation.shape}"
            )
        target_mask = np.zeros(segmentation.shape, dtype=bool)
        for instance_name in target_instances:
            target_mask |= segmentation == instance_to_id[instance_name]
        target_mask = _rotate_analysis_mask(target_mask, cfg)
        if not bool(target_mask.any()):
            logging.warning("Simulator target mask is empty for camera=%s at the current frame.", camera)
        masks.append(target_mask)

    metadata = {
        "source": "libero.SegmentationRenderEnv.instance_to_id",
        "camera_names": ["front", "wrist"],
        "target_instances": target_instances,
        "target_instance_selection": "analysis.target_instances",
        "available_obj_of_interest": [str(item) for item in getattr(env, "obj_of_interest", [])],
        "segmentation_id_mapping": {
            str(key): str(value)
            for key, value in getattr(env, "segmentation_id_mapping", {}).items()
        },
        "orientation_transform": "rotate_180" if _rotate_observation_images_180(cfg) else "none",
    }
    return np.stack(masks, axis=0).astype(np.uint8), metadata


def _analysis_rollout_dir(cfg: DictConfig, fallback: Path) -> Path:
    analysis_cfg = cfg.get("analysis", {})
    configured = analysis_cfg.get("rollout_dir", None)
    return Path(configured) if configured is not None else fallback / "analysis_rollouts"


def _save_analysis_rollout(
    output_dir: Path,
    *,
    task_id: int,
    episode_idx: int,
    task_description: str,
    success: bool,
    rollout: dict[str, Any],
) -> Path:
    """Persist optional raw RGB/state/action data for offline analysis.

    This is deliberately opt-in and does not alter action inference.  Frames
    are saved without the text labels used by the legacy replay-video writer.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    safe_task = "_".join(str(task_description).lower().split())[:80]
    path = output_dir / f"task{int(task_id)}_trial{int(episode_idx)}--success={bool(success)}--task={safe_task}.npz"
    frames = np.stack(
        [np.stack(rollout["image"], axis=0), np.stack(rollout["wrist_image"], axis=0)],
        axis=1,
    )
    metadata = {
        "task_id": int(task_id),
        "episode_id": f"task{int(task_id)}_trial{int(episode_idx)}",
        "task_name": str(task_description),
        "success": bool(success),
        "failure_reason": "not_applicable" if success else "unknown",
        "failure_class": "success" if success else "unknown",
        "grasp_relevant_failure": "not_applicable" if success else "unknown",
        "camera_names": ["front", "wrist"],
        "source": "experiments/libero/eval_libero_single.py",
        "failure_frame": int(max(len(rollout["timestamps"]) - 1, 0)),
        "failure_frame_source": "terminal_proxy",
    }
    payload: dict[str, Any] = {
        "frames": frames,
        "actions": np.asarray(rollout["actions"], dtype=np.float32),
        "states": np.asarray(rollout["states"], dtype=np.float32),
        "timestamps": np.asarray(rollout["timestamps"], dtype=np.int64),
        "gripper_state": np.asarray(rollout["states"], dtype=np.float32)[:, -1],
    }
    if rollout.get("gt_masks"):
        gt_masks = np.stack(rollout["gt_masks"], axis=0).astype(np.uint8)
        if gt_masks.ndim != 4 or gt_masks.shape[1] != 2:
            raise ValueError(f"Expected recorded gt_masks [T,2,H,W], got {gt_masks.shape}")
        payload["gt_masks"] = gt_masks
        metadata["gt_mask_source"] = rollout.get("gt_mask_metadata", {}).get(
            "source", "libero.SegmentationRenderEnv"
        )
        metadata["gt_mask_metadata"] = rollout.get("gt_mask_metadata", {})
    payload["metadata_json"] = np.asarray(json.dumps(metadata), dtype=np.str_)
    np.savez_compressed(path, **payload)
    return path


def _get_num_video_frames(cfg: DictConfig) -> int:
    return (int(cfg.data.train.num_frames) - 1) // int(cfg.data.train.action_video_freq_ratio) + 1


def _validate_visualize_future_video_cfg(cfg: DictConfig) -> None:
    if not bool(cfg.EVALUATION.get("visualize_future_video", False)):
        return

    action_conditioned = cfg.model.video_dit_config.get("action_conditioned", None)
    if action_conditioned is not False:
        raise ValueError(
            "EVALUATION.visualize_future_video=true requires "
            "model.video_dit_config.action_conditioned=false."
        )


def _validate_expected_task_description(task: Any, cfg: DictConfig) -> None:
    expected = cfg.EVALUATION.get("expected_task_description", None)
    if expected is None:
        return
    expected = " ".join(str(expected).strip().lower().split())
    actual = " ".join(str(task.language).strip().lower().split())
    if actual != expected:
        raise ValueError(
            "Resolved LIBERO task does not match EVALUATION.expected_task_description: "
            f"suite={cfg.EVALUATION.task_suite_name!r}, task_id={cfg.EVALUATION.task_id}, "
            f"expected={expected!r}, actual={actual!r}. Check LIBERO_ROOT/PYTHONPATH."
        )


def _select_predicted_future_frames(pred_video: list[Image.Image], cfg: DictConfig) -> list[Image.Image]:
    if len(pred_video) == 0:
        raise ValueError("`infer_joint` returned an empty predicted video.")

    replan_steps = int(cfg.EVALUATION.get("replan_steps", 5))
    action_video_freq_ratio = int(cfg.data.train.action_video_freq_ratio)
    num_future_frames = replan_steps // action_video_freq_ratio
    keep_frames = 1 + num_future_frames
    return list(pred_video[:keep_frames])


def _get_future_frame_capture_steps(cfg: DictConfig) -> list[int]:
    replan_steps = int(cfg.EVALUATION.get("replan_steps", 5))
    action_video_freq_ratio = int(cfg.data.train.action_video_freq_ratio)
    num_future_frames = replan_steps // action_video_freq_ratio
    return [step_idx * action_video_freq_ratio for step_idx in range(num_future_frames + 1)]


def _model_uses_history_intent(model: torch.nn.Module) -> bool:
    if getattr(model, "intent_encoder", None) is not None:
        return True
    if getattr(model, "semantic_history_encoder", None) is None:
        return False
    semantic_config = getattr(model, "semantic_history_config", None) or {}
    return bool(semantic_config.get("use_history", True))


def _get_history_frame_offsets(model: torch.nn.Module) -> list[int]:
    if getattr(model, "semantic_history_encoder", None) is not None:
        semantic_config = getattr(model, "semantic_history_config", None) or {}
        offsets = semantic_config.get("history_offsets", [-24, -16, -8, -1])
    else:
        intent_config = getattr(model, "intent_config", None) or {}
        offsets = intent_config.get(
            "history_offsets",
            intent_config.get("history_dino_frame_offsets", [-8, -4, 0]),
        )
    return [int(offset) for offset in offsets]


def _select_history_video(history_frames: list[torch.Tensor], offsets: list[int]) -> torch.Tensor:
    if len(history_frames) == 0:
        raise ValueError("Cannot build history video from an empty frame buffer.")
    selected = []
    latest_idx = len(history_frames) - 1
    for offset in offsets:
        frame_idx = max(0, min(latest_idx, latest_idx + int(offset)))
        selected.append(history_frames[frame_idx])
    return torch.stack(selected, dim=0)


def _frame_to_rgb_array(frame: Any) -> np.ndarray:
    if isinstance(frame, dict):
        images = []
        for value in frame.values():
            value_array = np.array(value) if isinstance(value, Image.Image) else np.array(value, copy=True)
            images.append(value_array)
        return np.concatenate(images, axis=1)
    if isinstance(frame, Image.Image):
        return np.array(frame.convert("RGB"))
    return np.array(frame, copy=True)


def _compute_clip_mean_psnr(
    gt_frames: list[Any],
    pred_frames: list[Any],
    eps: float = 1e-8,
) -> Optional[float]:
    if len(gt_frames) == 0 or len(pred_frames) == 0:
        return None
    assert len(gt_frames) == len(pred_frames), (
        "GT/pred frame count mismatch for PSNR: "
        f"len(gt_frames)={len(gt_frames)} len(pred_frames)={len(pred_frames)}. "
        "This indicates temporal misalignment in future-video capture."
    )
    num_frames = len(gt_frames)

    frame_psnr_values = []
    for gt_frame, pred_frame in zip(gt_frames[:num_frames], pred_frames[:num_frames]):
        gt_image = _frame_to_rgb_array(gt_frame)
        pred_image = _frame_to_rgb_array(pred_frame)
        target_h, target_w = pred_image.shape[:2]
        if gt_image.shape[:2] != (target_h, target_w):
            gt_image = np.array(
                Image.fromarray(gt_image).resize((target_w, target_h), resample=Image.BILINEAR)
            )

        gt_f32 = gt_image.astype(np.float32)
        pred_f32 = pred_image.astype(np.float32)
        mse = float(np.mean((pred_f32 - gt_f32) ** 2))
        psnr = 10.0 * np.log10((255.0 * 255.0) / max(mse, eps))
        frame_psnr_values.append(float(psnr))

    if len(frame_psnr_values) == 0:
        return None
    return float(np.mean(frame_psnr_values))


def _predict_action_chunk(
    obs: dict,
    task_description: str,
    model: torch.nn.Module,
    processor: FastWAMProcessor,
    cfg: DictConfig,
    *,
    action_horizon: int,
    input_w: int,
    input_h: int,
    model_device: str,
    history_video: Optional[torch.Tensor] = None,
    episode_idx: Optional[int] = None,
    environment_step: Optional[int] = None,
) -> tuple[np.ndarray, dict, Optional[list[Image.Image]]]:
    num_inference_steps_cfg = cfg.EVALUATION.get("num_inference_steps", None)
    if num_inference_steps_cfg is None:
        num_inference_steps = int(cfg.get("eval_num_inference_steps", 20))
    else:
        num_inference_steps = int(num_inference_steps_cfg)
    prompt_template = DEFAULT_PROMPT
    prompt = prompt_template.format(task=task_description)

    image, proprio, imgs = _obs_to_model_input(
        obs,
        cfg=cfg,
        processor=processor,
        width=input_w,
        height=input_h,
        device=model_device,
        dtype=model.torch_dtype,
    )

    infer_kwargs = {
        "prompt": prompt,
        "input_image": image,
        "action_horizon": action_horizon,
        "negative_prompt": str(cfg.EVALUATION.get("negative_prompt", "")),
        "text_cfg_scale": float(cfg.EVALUATION.get("text_cfg_scale", 1.0)),
        "num_inference_steps": num_inference_steps,
        "proprio": proprio,
        "sigma_shift": (
            None
            if cfg.EVALUATION.get("sigma_shift") is None
            else float(cfg.EVALUATION.get("sigma_shift"))
        ),
        "seed": None if cfg.get("seed") is None else int(cfg.seed),
        "rand_device": str(cfg.EVALUATION.get("rand_device", "cpu")),
        "tiled": bool(cfg.EVALUATION.get("tiled", False)),
    }
    if history_video is not None:
        infer_kwargs["history_video"] = history_video
    mot_attention_cfg = cfg.EVALUATION.get("mot_attention_visualization", None)
    if mot_attention_cfg is not None and bool(mot_attention_cfg.get("enabled", False)):
        mot_attention_cfg = OmegaConf.to_container(mot_attention_cfg, resolve=True)
        if not isinstance(mot_attention_cfg, dict):
            raise ValueError("EVALUATION.mot_attention_visualization must resolve to a mapping")
        image_meta = list(processor.shape_meta.get("images", []))
        camera_names = [str(meta["key"]) for meta in image_meta[: int(processor.num_output_cameras)]]
        camera_sizes = [
            [int(meta["shape"][1]), int(meta["shape"][2])]
            for meta in image_meta[: int(processor.num_output_cameras)]
        ]
        configured_camera_layout = mot_attention_cfg.get("camera_layout")
        if configured_camera_layout is None or str(configured_camera_layout).strip().lower() in {
            "",
            "auto",
        }:
            configured_camera_layout = cfg.data.train.get("concat_multi_camera", "full")
        mot_attention_cfg["camera_layout"] = str(configured_camera_layout)
        if not mot_attention_cfg.get("camera_names"):
            mot_attention_cfg["camera_names"] = camera_names
        if not mot_attention_cfg.get("camera_sizes"):
            mot_attention_cfg["camera_sizes"] = camera_sizes
        infer_kwargs["mot_attention_visualization"] = mot_attention_cfg
        infer_kwargs["mot_attention_original_images"] = imgs
        infer_kwargs["mot_attention_metadata"] = {
            "config_identifier": str(cfg.model.get("_target_", "unknown")),
            "checkpoint": str(cfg.ckpt),
            "checkpoint_sha256": (
                None
                if cfg.EVALUATION.get("checkpoint_sha256") is None
                else str(cfg.EVALUATION.get("checkpoint_sha256"))
            ),
            "task_suite": str(cfg.EVALUATION.task_suite_name),
            "task_id": int(cfg.EVALUATION.task_id),
            "task_description": str(task_description),
            "episode_index": None if episode_idx is None else int(episode_idx),
            "environment_step": (
                None if environment_step is None else int(environment_step)
            ),
            "replan_steps": int(cfg.EVALUATION.get("replan_steps", 5)),
            "input_size_hw": [int(input_h), int(input_w)],
            "image_preprocessing_description": (
                "LIBERO get_libero_image orientation transform, per-camera center-crop/resize, "
                "configured camera concatenation, then float RGB scaling to [-1, 1]."
            ),
        }
    visualize_future_video = bool(cfg.EVALUATION.get("visualize_future_video", False))
    predicted_future_frames = None
    if visualize_future_video:
        infer_kwargs["num_video_frames"] = _get_num_video_frames(cfg)
    elif "num_video_frames" in inspect.signature(model.infer_action).parameters:
        infer_kwargs["num_video_frames"] = _get_num_video_frames(cfg)

    with torch.no_grad():
        if visualize_future_video:
            pred = model.infer_joint(**_filter_call_kwargs(model.infer_joint, infer_kwargs))
            predicted_future_frames = _select_predicted_future_frames(pred["video"], cfg)
        else:
            pred = model.infer_action(**_filter_call_kwargs(model.infer_action, infer_kwargs))
    action = pred["action"]  # [T, D]

    action = _denormalize_action(action, processor)[0]  # [T, D]

    action = _postprocess_gripper_action(action, cfg)
    return action, imgs, predicted_future_frames


def _get_max_steps(task_suite_name: str) -> int:
    suite_steps = {
        "libero_spatial": 400,
        "libero_object": 400,
        "libero_goal": 400,
        "libero_10": 700,
        "libero_90": 700,
        # LIBERO-PRO keeps the LIBERO-Object episode horizon for each of its
        # five single-perturbation suites.
        "libero_object_env": 400,
        "libero_object_swap": 400,
        "libero_object_object": 400,
        "libero_object_lan": 400,
        "libero_object_task": 400,
    }
    if task_suite_name not in suite_steps:
        raise ValueError(f"Unknown task suite: {task_suite_name}")
    return suite_steps[task_suite_name]


def run_single_episode(
    env,
    initial_state,
    task_description: str,
    model: torch.nn.Module,
    processor: FastWAMProcessor,
    cfg: DictConfig,
    episode_idx: int,
    *,
    action_horizon: int,
    input_w: int,
    input_h: int,
    model_device: str,
) -> tuple[bool, list, list[dict[str, Any]], Optional[float], Optional[dict[str, Any]]]:
    max_steps = _get_max_steps(cfg.EVALUATION.task_suite_name)
    replan_steps = int(cfg.EVALUATION.get("replan_steps", 5))
    num_steps_wait = int(cfg.EVALUATION.get("num_steps_wait", 5))
    use_action_ensembler = bool(cfg.EVALUATION.get("use_action_ensembler", False))
    visualize_future_video = bool(cfg.EVALUATION.get("visualize_future_video", False))
    record_rollout = _analysis_rollout_enabled(cfg)
    record_gt_masks = _analysis_gt_masks_enabled(cfg)
    capture_steps = set(_get_future_frame_capture_steps(cfg)[1:])

    env.reset()
    obs = env.set_init_state(initial_state)
    if use_action_ensembler:
        ensembler = ActionEnsembler()
        ensembler.reset()

    replay_images = []
    predicted_future_video_clips: list[dict[str, Any]] = []
    episode_future_clip_psnr: list[float] = []
    pending_actions: list[list[float]] = []
    current_predicted_future_clip: Optional[dict[str, Any]] = None
    current_replan_step = 0
    current_replan_idx = -1
    use_history_intent = _model_uses_history_intent(model)
    history_offsets = _get_history_frame_offsets(model) if use_history_intent else []
    max_history_frames = 1 + max([max(0, -int(offset)) for offset in history_offsets], default=0)
    history_frames: list[torch.Tensor] = []
    analysis_rollout: Optional[dict[str, Any]] = None
    if record_rollout:
        analysis_rollout = {
            "image": [],
            "wrist_image": [],
            "actions": [],
            "states": [],
            "timestamps": [],
        }
        if record_gt_masks:
            analysis_rollout["gt_masks"] = []
            analysis_rollout["gt_mask_metadata"] = {}

    t = 0
    done = False
    pbar = tqdm(total=max_steps + num_steps_wait, desc=f"Episode {episode_idx + 1}")
    while t < max_steps + num_steps_wait:
        pbar.update(1)
        if t < num_steps_wait:
            obs, _, done, _ = env.step(get_libero_dummy_action())
            t += 1
            continue

        if use_history_intent:
            history_image, _ = _obs_to_model_image(
                obs,
                cfg=cfg,
                processor=processor,
                width=input_w,
                height=input_h,
                device=model_device,
                dtype=model.torch_dtype,
            )
            history_frames.append(history_image.squeeze(0).detach().clone())
            if len(history_frames) > max_history_frames:
                del history_frames[: len(history_frames) - max_history_frames]

        if len(pending_actions) == 0:
            history_video = (
                _select_history_video(history_frames, history_offsets)
                if use_history_intent
                else None
            )
            action_chunk, imgs, predicted_future_frames = _predict_action_chunk(
                obs=obs,
                task_description=task_description,
                model=model,
                processor=processor,
                cfg=cfg,
                action_horizon=action_horizon,
                input_w=input_w,
                input_h=input_h,
                model_device=model_device,
                history_video=history_video,
                episode_idx=episode_idx,
                environment_step=t,
            )
            if predicted_future_frames is not None:
                current_replan_idx += 1
                current_predicted_future_clip = {
                    "replan_idx": current_replan_idx,
                    "gt_frames": [imgs.copy()],
                    "pred_frames": predicted_future_frames,
                }
            else:
                current_predicted_future_clip = None
            current_replan_step = 0
            if use_action_ensembler:
                ensembler.add_actions(action_chunk, t)
                pending_actions = [ensembler.get_action(ts).tolist() for ts in range(t, t + replan_steps)]
            else:
                pending_actions = action_chunk[:replan_steps].tolist()
            replay_images.append(imgs.copy())
        else:
            imgs = get_libero_image(
                obs,
                rotate_180=_rotate_observation_images_180(cfg),
            )
            replay_images.append(imgs.copy())

        if analysis_rollout is not None:
            # This is the observation immediately before the action that is
            # actually sent to LIBERO. It is aligned one-to-one with actions.
            analysis_rollout["image"].append(np.asarray(imgs["image"], dtype=np.uint8).copy())
            analysis_rollout["wrist_image"].append(np.asarray(imgs["wrist_image"], dtype=np.uint8).copy())
            analysis_rollout["actions"].append(np.asarray(pending_actions[0], dtype=np.float32).copy())
            analysis_rollout["states"].append(_extract_sim_state(obs).copy())
            analysis_rollout["timestamps"].append(np.asarray(t, dtype=np.int64))
            if record_gt_masks:
                gt_masks, gt_mask_metadata = _extract_analysis_gt_masks(obs, env, cfg)
                analysis_rollout["gt_masks"].append(gt_masks)
                analysis_rollout["gt_mask_metadata"] = gt_mask_metadata

        obs, _, done, _ = env.step(pending_actions.pop(0))
        if visualize_future_video and current_predicted_future_clip is not None:
            current_replan_step += 1
            if current_replan_step in capture_steps:
                current_predicted_future_clip["gt_frames"].append(
                    get_libero_image(
                        obs,
                        rotate_180=_rotate_observation_images_180(cfg),
                    )
                )
            if done or len(pending_actions) == 0:
                expected_frame_count = 1 + sum(
                    1 for capture_step in capture_steps if capture_step <= current_replan_step
                )
                gt_len = len(current_predicted_future_clip["gt_frames"])
                pred_len = len(current_predicted_future_clip["pred_frames"])
                assert gt_len == expected_frame_count, (
                    "GT future frames do not match expected capture count: "
                    f"gt_len={gt_len} expected={expected_frame_count} "
                    f"episode={episode_idx} replan={current_predicted_future_clip['replan_idx']} "
                    f"current_replan_step={current_replan_step} capture_steps={sorted(capture_steps)}."
                )
                assert pred_len >= expected_frame_count, (
                    "Predicted future frames shorter than expected capture count: "
                    f"pred_len={pred_len} expected={expected_frame_count} "
                    f"episode={episode_idx} replan={current_predicted_future_clip['replan_idx']}."
                )
                if pred_len != expected_frame_count:
                    logging.info(
                        "Align predicted clip length to executed steps: "
                        "episode=%s replan=%s done=%s expected=%s pred_full=%s",
                        episode_idx,
                        current_predicted_future_clip["replan_idx"],
                        done,
                        expected_frame_count,
                        pred_len,
                    )
                current_predicted_future_clip["pred_frames"] = current_predicted_future_clip["pred_frames"][
                    :expected_frame_count
                ]
                assert len(current_predicted_future_clip["gt_frames"]) == len(
                    current_predicted_future_clip["pred_frames"]
                ), (
                    "GT/pred frame count mismatch after alignment: "
                    f"len(gt_frames)={len(current_predicted_future_clip['gt_frames'])} "
                    f"len(pred_frames)={len(current_predicted_future_clip['pred_frames'])} "
                    f"episode={episode_idx} replan={current_predicted_future_clip['replan_idx']}."
                )
                clip_psnr = _compute_clip_mean_psnr(
                    current_predicted_future_clip["gt_frames"],
                    current_predicted_future_clip["pred_frames"],
                )
                if clip_psnr is not None:
                    episode_future_clip_psnr.append(clip_psnr)
                predicted_future_video_clips.append(current_predicted_future_clip)
                current_predicted_future_clip = None
        if done:
            break
        t += 1
    pbar.close()

    episode_mean_psnr = (
        float(np.mean(episode_future_clip_psnr)) if len(episode_future_clip_psnr) > 0 else None
    )
    return bool(done), replay_images, predicted_future_video_clips, episode_mean_psnr, analysis_rollout


def run_single_task(
    task,
    initial_states,
    model: torch.nn.Module,
    processor: FastWAMProcessor,
    cfg: DictConfig,
    video_dir: Path,
    predicted_video_dir: Path,
    *,
    action_horizon: int,
    input_w: int,
    input_h: int,
    model_device: str,
) -> dict:
    env_resolution = int(cfg.EVALUATION.get("env_resolution", LIBERO_ENV_RESOLUTION))
    if env_resolution <= 0:
        raise ValueError(f"EVALUATION.env_resolution must be positive, got {env_resolution}.")
    env, task_description = get_libero_env(
        task,
        env_resolution,
        cfg.get("seed"),
        record_gt_masks=_analysis_gt_masks_enabled(cfg),
    )
    visualize_future_video = bool(cfg.EVALUATION.get("visualize_future_video", False))
    results = {
        "successes": 0,
        "failure_episodes": [],
        "success_episodes": [],
        "task_description": task_description,
    }
    record_rollout = _analysis_rollout_enabled(cfg)
    if record_rollout:
        results["analysis_rollout_paths"] = []
    if visualize_future_video:
        results["episode_future_video_psnr"] = []
        results["future_video_psnr_mean"] = None

    for trial_idx in range(int(cfg.EVALUATION.num_trials)):
        success, replay_images, predicted_future_video_clips, episode_mean_psnr, analysis_rollout = run_single_episode(
            env=env,
            initial_state=initial_states[trial_idx],
            task_description=task_description,
            model=model,
            processor=processor,
            cfg=cfg,
            episode_idx=trial_idx,
            action_horizon=action_horizon,
            input_w=input_w,
            input_h=input_h,
            model_device=model_device,
        )
        if success:
            results["successes"] += 1
            results["success_episodes"].append(trial_idx)
        else:
            results["failure_episodes"].append(trial_idx)
        if analysis_rollout is not None:
            rollout_path = _save_analysis_rollout(
                _analysis_rollout_dir(cfg, video_dir.parent),
                task_id=int(cfg.EVALUATION.task_id),
                episode_idx=trial_idx,
                task_description=task_description,
                success=success,
                rollout=analysis_rollout,
            )
            results["analysis_rollout_paths"].append(str(rollout_path))
        if visualize_future_video:
            results["episode_future_video_psnr"].append(episode_mean_psnr)

        save_rollout_video(
            video_dir,
            replay_images,
            f"task{cfg.EVALUATION.task_id}_trial{trial_idx}",
            success=success,
            task_description=task_description,
        )
        if visualize_future_video:
            if len(predicted_future_video_clips) == 0:
                logging.warning(
                    "No predicted future frames collected for task %s trial %s.",
                    cfg.EVALUATION.task_id,
                    trial_idx,
                )
            else:
                all_gt_frames = []
                all_pred_frames = []
                for clip in predicted_future_video_clips:
                    all_gt_frames.extend(clip["gt_frames"])
                    all_pred_frames.extend(clip["pred_frames"])
                    save_prediction_video(
                        predicted_video_dir,
                        clip["gt_frames"],
                        clip["pred_frames"],
                        f"task{cfg.EVALUATION.task_id}_trial{trial_idx}",
                        clip["replan_idx"],
                        success=success,
                        task_description=task_description,
                    )
                save_prediction_video(
                    predicted_video_dir,
                    all_gt_frames,
                    all_pred_frames,
                    f"task{cfg.EVALUATION.task_id}_trial{trial_idx}",
                    "all",
                    success=success,
                    task_description=task_description,
                )

    if visualize_future_video:
        valid_episode_psnr = [x for x in results["episode_future_video_psnr"] if x is not None]
        if len(valid_episode_psnr) > 0:
            results["future_video_psnr_mean"] = float(np.mean(valid_episode_psnr))
    return results


@hydra.main(version_base="1.3", config_path="../../configs", config_name="sim_libero.yaml")
def eval_single_process(cfg: DictConfig):
    start_time = time.time()
    partial_state = PartialState()
    partial_state.config = cfg

    if cfg.get("seed") is not None:
        set_global_seed(int(cfg.seed), get_worker_init_fn=False)

    if cfg.ckpt is None:
        raise ValueError("cfg.ckpt must not be None.")
    _validate_visualize_future_video_cfg(cfg)

    env_num = int(cfg.EVALUATION.get("env_num", 1))
    if env_num != 1:
        raise ValueError(
            "Only env_num=1 is supported in eval_libero_single.py. "
            "Use run_libero_manager/run_libero_parallel_test.sh for multi-GPU task parallelism."
        )

    model_device = _resolve_eval_device(cfg)
    model_dtype = _mixed_precision_to_model_dtype(cfg.get("mixed_precision", "bf16"))
    model = instantiate(cfg.model, model_dtype=model_dtype, device=model_device)
    _load_model_checkpoint(model, str(cfg.ckpt))
    model = model.to(model_device).eval()

    dataset_stats_path = _resolve_dataset_stats_path(cfg)
    dataset_stats = load_dataset_stats_from_json(str(dataset_stats_path))
    processor: FastWAMProcessor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(dataset_stats)
    logging.info("Using dataset stats: %s", dataset_stats_path)
    logging.info(
        "Evaluation data convention: gripper_action_mode=%s, "
        "rotate_observation_images_180=%s, env_resolution=%s",
        cfg.EVALUATION.get("gripper_action_mode", "legacy_rlds"),
        _rotate_observation_images_180(cfg),
        cfg.EVALUATION.get("env_resolution", LIBERO_ENV_RESOLUTION),
    )

    action_horizon_cfg = cfg.EVALUATION.get("action_horizon", None)
    if action_horizon_cfg is None:
        action_horizon = int(cfg.data.train.num_frames) - 1
    else:
        action_horizon = int(action_horizon_cfg)
    if action_horizon <= 0:
        raise ValueError(f"EVALUATION.action_horizon must be positive, got {action_horizon}")

    video_size = cfg.data.train.get("video_size", [224, 224])
    if len(video_size) != 2:
        raise ValueError(f"data.train.video_size must be [H, W], got {video_size}")
    input_h = int(video_size[0])
    input_w = int(video_size[1])
    concat_multi_camera = cfg.data.train.get("concat_multi_camera", None)
    shape_meta_images = [meta["shape"] for meta in processor.shape_meta["images"]]

    local_log_dir = Path(cfg.EVALUATION.output_dir)
    local_log_dir.mkdir(parents=True, exist_ok=True)
    video_dir = local_log_dir / cfg.EVALUATION.task_suite_name / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    predicted_video_dir = local_log_dir / cfg.EVALUATION.task_suite_name / "predicted_videos"
    if bool(cfg.EVALUATION.get("visualize_future_video", False)):
        predicted_video_dir.mkdir(parents=True, exist_ok=True)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.EVALUATION.task_suite_name]()
    task = task_suite.get_task(cfg.EVALUATION.task_id)
    _validate_expected_task_description(task, cfg)
    initial_states = task_suite.get_task_init_states(cfg.EVALUATION.task_id)

    while len(initial_states) < int(cfg.EVALUATION.num_trials):
        initial_states.extend(initial_states[: (int(cfg.EVALUATION.num_trials) - len(initial_states))])

    results = {
        "task_suite": cfg.EVALUATION.task_suite_name,
        "task_id": cfg.EVALUATION.task_id,
        "task_description": None,
        "successes": 0,
        "total_episodes": int(cfg.EVALUATION.num_trials),
        "gpu_id": int(cfg.gpu_id),
        "success_episodes": [],
        "failure_episodes": [],
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "duration": 0,
    }

    logging.info("Running LIBERO evaluation with env_num=1")
    task_results = run_single_task(
        task=task,
        initial_states=initial_states,
        model=model,
        processor=processor,
        cfg=cfg,
        video_dir=video_dir,
        predicted_video_dir=predicted_video_dir,
        action_horizon=action_horizon,
        input_w=input_w,
        input_h=input_h,
        model_device=model_device,
    )
    results.update(task_results)

    results["duration"] = time.time() - start_time
    output_dir = Path(cfg.EVALUATION.output_dir) / cfg.EVALUATION.task_suite_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"gpu{cfg.gpu_id}_task{cfg.EVALUATION.task_id}_results.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, cls=NumpyEncoder)

    print(
        f"Task {cfg.EVALUATION.task_id} completed: "
        f"{results['successes']}/{cfg.EVALUATION.num_trials} successes"
    )
    if results.get("future_video_psnr_mean") is not None:
        print(f"Task {cfg.EVALUATION.task_id} future-video PSNR mean: {results['future_video_psnr_mean']:.4f}")
    print(f"Time taken: {results['duration']:.2f} seconds")
    return results


if __name__ == "__main__":
    eval_single_process()
