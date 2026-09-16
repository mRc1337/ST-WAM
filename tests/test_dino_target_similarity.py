import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from analysis.dino_target_similarity.features import resolve_model_resolution
from analysis.dino_target_similarity.io import (
    EpisodeRecord,
    get_evaluation_gt_mask,
    get_reference_mask,
    get_rollout_gt_mask,
    parse_episode_path,
    split_camera_frame,
)
from analysis.dino_target_similarity.metrics import compute_frame_metrics
from analysis.dino_target_similarity.pipeline import analyze_features
from analysis.dino_target_similarity.prototype import build_random_prototypes, build_target_prototype
from analysis.dino_target_similarity.visualize import normalize_similarity_map
from experiments.libero.eval_libero_single import _analysis_target_instances


def _synthetic_features() -> tuple[torch.Tensor, np.ndarray]:
    rng = np.random.default_rng(4)
    features = torch.from_numpy(rng.normal(size=(4, 4, 8)).astype(np.float32))
    target_vector = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    features[1:3, 1:3] = target_vector
    mask = np.zeros((64, 64), dtype=bool)
    mask[16:48, 16:48] = True
    return features, mask


def test_native_and_high_resolution_grid_are_patch_aligned():
    cfg = {"input_resolution": [224, 448]}
    assert resolve_model_resolution(cfg, "native") == (224, 448)
    assert resolve_model_resolution({**cfg, "high_resolution": [448, 896]}, "high_resolution") == (448, 896)


def test_prototype_is_normalized_and_reference_similarity_localizes_target():
    features, mask = _synthetic_features()
    target = build_target_prototype(features, mask, image_size=(64, 64), patch_size=16, min_patch_overlap=0.5)
    assert target["normalized"] is True
    assert torch.isclose(torch.linalg.vector_norm(target["prototype"]), torch.tensor(1.0), atol=1e-6)
    metrics = compute_frame_metrics(features, target["prototype"], target_mask=mask, temperature=0.07)
    assert -1.0 <= metrics["raw_similarity_map"].min() <= 1.0
    assert metrics["target_mean_similarity"] > metrics["background_mean_similarity"]
    assert metrics["target_mass"] > 0.5
    assert metrics["peak_localization_error"] < 0.3


def test_random_prototype_is_a_real_control():
    features, mask = _synthetic_features()
    target = build_target_prototype(features, mask, image_size=(64, 64))
    controls = build_random_prototypes(features, target["patch_mask"], num_prototypes=1, seed=9)
    assert len(controls) == 1
    target_metrics = compute_frame_metrics(features, target["prototype"], target_mask=mask)
    random_metrics = compute_frame_metrics(features, controls[0]["prototype"], target_mask=mask)
    assert random_metrics["target_mass"] <= target_metrics["target_mass"]


def test_visualization_normalization_is_fixed_and_separate_from_raw_metrics():
    raw = np.asarray([[-1.0, 0.0, 1.0, 2.0]], dtype=np.float32)
    normalized = normalize_similarity_map(raw, vmin=0.0, vmax=1.0)
    np.testing.assert_allclose(normalized, [[0.0, 0.0, 1.0, 1.0]])
    np.testing.assert_allclose(raw, [[-1.0, 0.0, 1.0, 2.0]])


def test_reference_mask_is_not_reused_as_future_gt():
    episode = EpisodeRecord(Path("episode.mp4"), "episode", "1", "task", None)
    annotation = {"reference_frame": 0, "bbox": [8, 8, 24, 24]}
    ref = get_reference_mask(annotation, camera="front", frame_idx=0, source_hw=(32, 32), episode=episode)
    assert ref.sum() == 16 * 16
    assert get_evaluation_gt_mask(annotation, camera="front", frame_idx=1, source_hw=(32, 32), episode=episode) is None


def test_simulator_rollout_masks_are_frame_specific_and_camera_separated():
    masks = np.zeros((3, 2, 8, 8), dtype=np.uint8)
    masks[0, 0, 1:3, 2:4] = 1
    masks[1, 1, 4:6, 5:7] = 1
    auxiliary = {"gt_masks": masks}
    front = get_rollout_gt_mask(
        auxiliary,
        camera="front",
        camera_names=["front", "wrist"],
        frame_idx=0,
        source_hw=(8, 8),
    )
    wrist = get_rollout_gt_mask(
        auxiliary,
        camera="wrist",
        camera_names=["front", "wrist"],
        frame_idx=1,
        source_hw=(8, 8),
    )
    assert front is not None and wrist is not None
    assert int(front.sum()) == 4
    assert int(wrist.sum()) == 4
    assert not np.array_equal(front, wrist)


def test_frame_order_metadata_and_multiview_split_are_preserved():
    record = parse_episode_path(
        "2026--episode=task9_trial46--success=False--task=pick_up_the_orange_juice.mp4"
    )
    assert record.episode_id == "task9_trial46"
    assert record.task_id == "9"
    assert record.success is False
    raw_record = parse_episode_path("task9_trial46--success=False--task=pick_up_the_orange_juice.npz")
    assert raw_record.episode_id == "task9_trial46"
    frame = np.zeros((8, 16, 3), dtype=np.uint8)
    frame[:, :8] = 10
    frame[:, 8:] = 20
    split = split_camera_frame(frame, "side_by_side", ["front", "wrist"])
    assert split["front"].mean() == 10
    assert split["wrist"].mean() == 20


def test_raw_rollout_masks_drive_pipeline_without_reference_leakage(tmp_path):
    """Exercise raw NPZ loading, prototype creation, and future-mask scope."""

    frame_count = 3
    image_hw = (8, 8)
    camera_names = ["front", "wrist"]
    frames = np.zeros((frame_count, len(camera_names), *image_hw, 3), dtype=np.uint8)
    gt_masks = np.zeros((frame_count, len(camera_names), *image_hw), dtype=np.uint8)
    # Each camera's target moves to a different patch after the reference.
    gt_masks[0, 0, :4, :4] = 1
    gt_masks[1:, 0, 4:, 4:] = 1
    gt_masks[2, 0, 4:, 4:] = 1
    gt_masks[0, 1, :4, 4:] = 1
    gt_masks[1:, 1, 4:, :4] = 1

    feature_dim = 4
    background = torch.tensor([0.0, 1.0, 0.0, 0.0])
    front_target = torch.tensor([1.0, 0.0, 0.0, 0.0])
    wrist_target = torch.tensor([0.0, 0.0, 1.0, 0.0])
    features = {
        camera: torch.zeros((frame_count, 2, 2, feature_dim), dtype=torch.float32)
        for camera in camera_names
    }
    for camera in camera_names:
        features[camera][:] = background
    features["front"][0, 0, 0] = front_target
    features["front"][1:, 1, 1] = front_target
    features["wrist"][0, 0, 1] = wrist_target
    features["wrist"][1:, 1, 0] = wrist_target

    rollout_path = tmp_path / "task1_trial0--success=True--task=synthetic.npz"
    np.savez_compressed(rollout_path, frames=frames, gt_masks=gt_masks, timestamps=np.arange(frame_count))

    encoder_metadata = {
        "model_name": "synthetic",
        "checkpoint_identifier": "synthetic",
        "input_resolution": [8, 16],
        "patch_size": 4,
        "patch_grid": [2, 2],
        "feature_dim": feature_dim,
        "view_mode": "current_concat",
    }
    feature_path = tmp_path / "synthetic_features.pt"
    torch.save({"features": features, "metadata": encoder_metadata}, feature_path)
    manifest = {
        "encoder": encoder_metadata,
        "episodes": [
            {
                "path": str(rollout_path),
                "episode_id": "task1_trial0",
                "task_id": "1",
                "task_name": "synthetic",
                "success": True,
                "shift_condition": "clean",
                "failure_reason": "not_applicable",
                "failure_class": "success",
                "grasp_relevant_failure": "not_applicable",
                "feature_cache": str(feature_path),
                "metadata": {},
            }
        ],
    }
    config = {
        "model": {"patch_size": 4},
        "input": {"path": str(rollout_path), "video_layout": "side_by_side", "camera_names": camera_names},
        "analysis": {
            "reference_mode": "episode_reference",
            "min_patch_overlap": 0.5,
            "min_target_patches": 1,
            "random_prototypes": 0,
            "progress_points": 3,
            "bootstrap_samples": 0,
            "save_heatmaps": False,
        },
    }

    run_dir = tmp_path / "raw_mask_run"
    frame_df = analyze_features(config, run_dir, manifest=manifest)
    assert set(frame_df["camera"]) == set(camera_names)
    assert frame_df["has_evaluation_gt_mask"].all()
    assert set(frame_df["evaluation_gt_mask_source"]) == {"rollout_gt_mask"}

    front_rows = frame_df[frame_df["camera"] == "front"].sort_values("frame_index")
    wrist_rows = frame_df[frame_df["camera"] == "wrist"].sort_values("frame_index")
    assert front_rows["gt_target_center_x"].iloc[0] < front_rows["gt_target_center_x"].iloc[1]
    assert wrist_rows["gt_target_center_x"].iloc[0] > wrist_rows["gt_target_center_x"].iloc[1]
    assert front_rows["target_mass"].iloc[1] > 0.9
    assert wrist_rows["target_mass"].iloc[1] > 0.9

    saved = torch.load(run_dir / "prototypes" / "task1_trial0__front.pt", map_location="cpu", weights_only=False)
    expected = build_target_prototype(
        features["front"][0],
        gt_masks[0, 0].astype(bool),
        image_size=image_hw,
        patch_size=4,
    )
    assert saved["metadata"]["prototype_source_frame"] == 0
    assert saved["metadata"]["leakage_guard"].startswith("prototype fixed at reference frame")
    assert torch.allclose(saved["prototype"], expected["prototype"], atol=1e-6)

    # A reference-only annotation must not turn into a moving GT mask.
    annotation_path = tmp_path / "reference_only.json"
    annotation_path.write_text(
        '{"task1_trial1": {"reference_frame": 0, "cameras": '
        '{"front": {"reference_frame": 0, "bbox": [0, 0, 4, 4]}, '
        '"wrist": {"reference_frame": 0, "bbox": [4, 0, 8, 4]}}}}',
        encoding="utf-8",
    )
    no_gt_path = tmp_path / "task1_trial1--success=True--task=synthetic.npz"
    np.savez_compressed(no_gt_path, frames=frames, timestamps=np.arange(frame_count))
    no_gt_manifest = {
        **manifest,
        "episodes": [{**manifest["episodes"][0], "path": str(no_gt_path), "episode_id": "task1_trial1"}],
    }
    no_gt_config = {**config, "input": {**config["input"], "path": str(no_gt_path), "annotation_path": str(annotation_path)}}
    no_gt_df = analyze_features(no_gt_config, tmp_path / "reference_only_run", manifest=no_gt_manifest)
    future = no_gt_df[no_gt_df["frame_index"] > 0]
    assert not future["has_evaluation_gt_mask"].any()
    assert future["target_mass"].isna().all()


def test_task_exemplar_mode_uses_fixed_exemplar_prototype(tmp_path):
    image_hw = (8, 8)
    camera_names = ["front"]
    feature_dim = 4
    target_vector = torch.tensor([1.0, 0.0, 0.0, 0.0])
    background = torch.tensor([0.0, 1.0, 0.0, 0.0])
    feature_paths = {}
    rollout_paths = {}
    annotations = {}
    for episode_id, success in (("task1_trial0", True), ("task1_trial1", False)):
        frames = np.zeros((2, *image_hw, 3), dtype=np.uint8)
        gt_masks = np.zeros((2, 1, *image_hw), dtype=np.uint8)
        gt_masks[:, 0, :4, :4] = 1
        rollout_path = tmp_path / f"{episode_id}--success={success}--task=synthetic.npz"
        np.savez_compressed(rollout_path, frames=frames, gt_masks=gt_masks, timestamps=np.arange(2))
        rollout_paths[episode_id] = rollout_path

        features = torch.empty((2, 2, 2, feature_dim), dtype=torch.float32)
        features[:] = background
        features[:, 0, 0] = target_vector
        feature_path = tmp_path / f"{episode_id}.pt"
        torch.save(
            {
                "features": {"front": features},
                "metadata": {
                    "model_name": "synthetic",
                    "checkpoint_identifier": "synthetic",
                    "input_resolution": [8, 8],
                    "patch_size": 4,
                    "patch_grid": [2, 2],
                    "feature_dim": feature_dim,
                    "view_mode": "independent",
                },
            },
            feature_path,
        )
        feature_paths[episode_id] = feature_path
        annotations[episode_id] = {
            "reference_frame": 0,
            "exemplar_episode_id": "task1_trial0",
            "cameras": {"front": {"reference_frame": 0, "exemplar_episode_id": "task1_trial0"}},
        }

    manifest = {
        "encoder": {
            "model_name": "synthetic",
            "checkpoint_identifier": "synthetic",
            "input_resolution": [8, 8],
            "patch_size": 4,
            "patch_grid": [2, 2],
            "feature_dim": feature_dim,
            "view_mode": "independent",
        },
        "episodes": [
            {
                "path": str(rollout_paths[episode_id]),
                "episode_id": episode_id,
                "task_id": "1",
                "task_name": "synthetic",
                "success": success,
                "shift_condition": "clean",
                "failure_reason": "unknown",
                "failure_class": "unknown",
                "grasp_relevant_failure": "unknown",
                "feature_cache": str(feature_paths[episode_id]),
                "metadata": {},
            }
            for episode_id, success in (("task1_trial0", True), ("task1_trial1", False))
        ],
    }
    annotation_path = tmp_path / "exemplars.json"
    annotation_path.write_text(json.dumps(annotations), encoding="utf-8")
    config = {
        "model": {"patch_size": 4},
        "input": {
            "path": str(tmp_path / "*.npz"),
            "video_layout": "raw",
            "camera_names": camera_names,
            "annotation_path": str(annotation_path),
        },
        "analysis": {
            "reference_mode": "task_exemplar",
            "min_patch_overlap": 0.5,
            "min_target_patches": 1,
            "random_prototypes": 0,
            "progress_points": 2,
            "bootstrap_samples": 0,
            "save_heatmaps": False,
        },
    }
    run_dir = tmp_path / "task_exemplar_run"
    analyze_features(config, run_dir, manifest=manifest)
    saved = torch.load(run_dir / "prototypes" / "task1_trial1__front.pt", map_location="cpu", weights_only=False)
    assert saved["metadata"]["reference_mode"] == "task_exemplar"
    assert saved["metadata"]["prototype_source_episode"] == "task1_trial0"
    assert saved["metadata"]["prototype_source_frame"] == 0


def test_simulator_target_selection_does_not_include_receptacle_by_default():
    env = SimpleNamespace(
        obj_of_interest=["bbq_sauce_1", "basket_1"],
        instance_to_id={"bbq_sauce_1": 1, "basket_1": 2},
    )
    config = {"analysis": {"target_instances": ["bbq_sauce_1"]}, "EVALUATION": {"task_id": 3}}
    assert _analysis_target_instances(env, config) == ["bbq_sauce_1"]
    with pytest.raises(RuntimeError, match="requires analysis.target_instances"):
        _analysis_target_instances(env, {"analysis": {}, "EVALUATION": {"task_id": 3}})
