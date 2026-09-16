import json

import numpy as np
import torch

from fastwam.models.wan22.mot_attention_visualization import (
    MotAttentionRecorder,
    aggregate_attention,
    aggregate_conditional_attention,
    camera_grid_slices,
    normalize_heatmap,
    recompute_attention_weights,
    scores_to_spatial,
)


def test_recompute_softmaxes_complete_key_sequence_before_slicing():
    torch.manual_seed(7)
    q = torch.randn(1, 2, 6)
    k = torch.randn(1, 5, 6)
    mask = torch.tensor([[True, True, False, True, False], [True, False, True, True, True]])

    actual = recompute_attention_weights(q, k, mask, num_heads=2)
    q_heads = q.reshape(1, 2, 2, 3).transpose(1, 2).float()
    k_heads = k.reshape(1, 5, 2, 3).transpose(1, 2).float()
    logits = torch.matmul(q_heads, k_heads.transpose(-2, -1)) / (3.0**0.5)
    expected = torch.softmax(logits.masked_fill(~mask[None, None], float("-inf")), dim=-1)
    expected = expected.masked_fill(~mask[None, None], 0.0)

    torch.testing.assert_close(actual, expected)
    scores, masses, total = aggregate_attention(
        actual,
        {"vae": (0, 2), "dino": (2, 4), "action": (4, 5)},
        head_indices=[0, 1],
        query_range=(0, 2),
    )
    assert scores["vae"].shape == (1, 2)
    torch.testing.assert_close(total, torch.ones_like(total))
    torch.testing.assert_close(
        masses["vae"] + masses["dino"] + masses["action"],
        torch.ones_like(total),
    )


def test_masked_positions_are_zero_and_other_key_mass_is_retained():
    q = torch.ones(1, 1, 4)
    k = torch.ones(1, 4, 4)
    mask = torch.tensor([[True, False, True, False]])
    weights = recompute_attention_weights(q, k, mask, num_heads=2)
    assert torch.count_nonzero(weights[..., 1]).item() == 0
    assert torch.count_nonzero(weights[..., 3]).item() == 0
    torch.testing.assert_close(weights.sum(dim=-1), torch.ones(1, 2, 1))


def test_head_query_aggregation_and_spatial_grid_mapping():
    # Each key is already a recognizable value, so the resulting map exposes
    # ordering errors immediately.
    weights = torch.zeros(1, 2, 2, 8)
    weights[:, :, :, :8] = 1.0 / 8.0
    scores, masses, _ = aggregate_attention(
        weights,
        {"vae": (0, 4), "dino": (4, 8)},
        head_indices=[1],
        query_range=(1, 2),
    )
    assert scores["vae"].shape == (1, 4)
    assert scores["dino"].shape == (1, 4)
    torch.testing.assert_close(masses["vae"], torch.full((1,), 0.5))

    spatial = scores_to_spatial(
        torch.arange(8, dtype=torch.float32).view(1, 8),
        {
            "grid_size": (1, 2, 4),
            "tokens_per_frame": 8,
            "latent_patch_size": (1, 1, 1),
            "latent_patch_mode": "flat",
            "latent_num_views": 1,
        },
    )
    np.testing.assert_array_equal(spatial[0], np.arange(8, dtype=np.float32).reshape(2, 4))


def test_conditional_attention_normalizes_each_head_query_before_averaging():
    weights = torch.tensor(
        [[[[0.90, 0.10, 0.00], [0.00, 0.00, 1.00]], [[0.09, 0.01, 0.90], [0.00, 0.00, 1.00]]]],
        dtype=torch.float32,
    )
    scores, valid = aggregate_conditional_attention(
        weights,
        {"dino": (0, 2), "action": (2, 3)},
        head_indices=[0, 1],
        query_range=(0, 2),
        target_names=("dino",),
    )
    torch.testing.assert_close(scores["dino"], torch.tensor([[0.9, 0.1]]))
    torch.testing.assert_close(valid["dino"], torch.tensor([0.5]))


def test_view_grid_restores_left_right_camera_order_without_halving_guess():
    # The model's view-aware sequence is [view0 grid, view1 grid].
    scores = torch.arange(8, dtype=torch.float32).view(1, 8)
    spatial = scores_to_spatial(
        scores,
        {
            "grid_size": (1, 2, 2),
            "original_grid_size": (1, 2, 4),
            "tokens_per_frame": 8,
            "latent_patch_size": (1, 1, 1),
            "latent_patch_mode": "view",
            "latent_num_views": 2,
        },
    )
    expected = np.array([[0, 1, 4, 5], [2, 3, 6, 7]], dtype=np.float32)
    np.testing.assert_array_equal(spatial[0], expected)

    slices = camera_grid_slices(
        (2, 6),
        layout="horizontal",
        camera_names=["left", "right"],
        camera_sizes=[(224, 320), (224, 160)],
    )
    np.testing.assert_array_equal(spatial[0][:, slices["left"][1]], expected[:, :4])
    np.testing.assert_array_equal(spatial[0][:, slices["right"][1]], expected[:, 4:])


def test_vertical_camera_geometry_uses_real_camera_heights():
    slices = camera_grid_slices(
        (6, 2),
        layout="vertical",
        camera_names=["top", "bottom"],
        camera_sizes=[(100, 224), (200, 224)],
    )
    assert slices["top"] == (slice(0, 2), slice(0, 2))
    assert slices["bottom"] == (slice(2, 6), slice(0, 2))


def test_constant_heatmap_normalization_is_finite():
    normalized, metadata = normalize_heatmap(np.full((3, 3), 2.0, dtype=np.float32))
    assert np.isfinite(normalized).all()
    assert np.all(normalized == 0.0)
    assert metadata["constant"] is True


def test_disabled_recorder_does_not_create_output(tmp_path):
    recorder = MotAttentionRecorder(
        {
            "enabled": False,
            "output_dir": str(tmp_path / "should-not-exist"),
        },
        call_index=0,
        num_layers=2,
        num_heads=2,
    )
    recorder.configure_steps(3)
    assert recorder.active is False
    assert recorder.save(input_image=torch.zeros(1, 3, 4, 4)) is None
    assert not (tmp_path / "should-not-exist").exists()


def test_enabled_recorder_saves_raw_scores_mass_and_comparison(tmp_path):
    recorder = MotAttentionRecorder(
        {
            "enabled": True,
            "output_dir": str(tmp_path),
            "layer": "last",
            "denoising_step": "last",
            "heads": "all",
            "action_query_range": "all",
            "max_saves": 1,
            "camera_layout": "horizontal",
            "camera_names": ["left", "right"],
            "camera_sizes": [[4, 4], [4, 4]],
        },
        call_index=3,
        num_layers=1,
        num_heads=1,
    )
    recorder.configure_steps(1)
    q = torch.tensor([[[1.0, 0.0]]])
    k = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]])
    recorder.capture(
        q_action=q,
        k_all=k,
        attention_mask=torch.ones(1, 3, dtype=torch.bool),
        layer_idx=0,
        step_idx=0,
        timestep=torch.tensor([0.75]),
        key_spans={"vae": (0, 1), "dino": (1, 2), "action": (2, 3)},
        visual_meta={
            "vae": {
                "grid_size": (1, 1, 1),
                "tokens_per_frame": 1,
                "latent_patch_size": (1, 1, 1),
            },
            "dino": {
                "grid_size": (1, 1, 1),
                "tokens_per_frame": 1,
                "latent_patch_size": (1, 1, 1),
                "latent_patch_mode": "flat",
                "latent_num_views": 1,
            },
        },
    )
    record_dir = recorder.save(
        input_image=torch.zeros(1, 3, 4, 8),
        original_images={
            "left": np.full((2, 2, 3), 32, dtype=np.uint8),
            "right": np.full((2, 2, 3), 224, dtype=np.uint8),
        },
    )
    assert record_dir is not None
    assert (record_dir / "comparison.png").exists()
    assert (record_dir / "rgb_input.png").exists()
    assert (record_dir / "rgb_original_left.png").exists()
    assert (record_dir / "rgb_original_right.png").exists()
    assert (record_dir / "rgb_original_concatenated.png").exists()
    with np.load(record_dir / "attention.npz") as payload:
        assert payload["scores_dino"].shape == (1, 1)
        assert payload["scores_vae"].shape == (1, 1)
        assert payload["attention_mask"].shape == (1, 3)
        np.testing.assert_allclose(payload["mass_all_keys"], 1.0, atol=1e-6)
    metadata = json.loads((record_dir / "metadata.json").read_text())
    assert metadata["denoising_step_idx"] == 0
    assert metadata["display_normalization"]["note"]


def test_recorder_scans_layers_steps_and_query_groups(tmp_path):
    recorder = MotAttentionRecorder(
        {
            "enabled": True,
            "output_dir": str(tmp_path),
            "layers": [0, 2],
            "denoising_steps": [0, 1],
            "heads": "all",
            "action_query_ranges": ["0:1", "1:2"],
            "max_saves": 1,
            "display_normalization": "percentile",
        },
        call_index=4,
        num_layers=3,
        num_heads=1,
    )
    recorder.configure_steps(2)
    q = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    k = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.0]]])
    for step in range(2):
        for layer in range(3):
            recorder.capture(
                q_action=q,
                k_all=k,
                attention_mask=torch.ones(2, 4, dtype=torch.bool),
                layer_idx=layer,
                step_idx=step,
                timestep=torch.tensor([float(step)]),
                key_spans={"vae": (0, 1), "dino": (1, 3), "action": (3, 4)},
                visual_meta={
                    "vae": {"grid_size": (1, 1, 1), "tokens_per_frame": 1},
                    "dino": {
                        "grid_size": (1, 1, 2),
                        "tokens_per_frame": 2,
                        "latent_patch_mode": "flat",
                    },
                },
            )
    saved = recorder.save(input_image=torch.zeros(1, 3, 4, 4))
    assert isinstance(saved, list)
    assert len(saved) == 4
    for record_dir in saved:
        assert (record_dir / "comparison_q000_001.png").is_file()
        assert (record_dir / "comparison_q001_002.png").is_file()
        with np.load(record_dir / "attention.npz") as payload:
            assert "scores_dino_absolute_q000_001" in payload
            assert "scores_dino_conditional_q000_001" in payload
        metadata = json.loads((record_dir / "metadata.json").read_text())
        assert metadata["format"] == "st-wam-mot-action-vision-attention-v2"
        assert len(metadata["query_groups"]) == 2
