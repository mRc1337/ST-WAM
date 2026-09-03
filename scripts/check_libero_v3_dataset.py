#!/usr/bin/env python3
"""Validate the local four-suite LeRobot v3 LIBERO training dataset."""

import argparse
import json
import os
from pathlib import Path

from fastwam.datasets.lerobot.base_lerobot_dataset import BaseLerobotDataset


SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_100")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(
            os.environ.get(
                "LIBERO_DATA_ROOT",
                "/mnt/data/embodied_datasets/public_datasets_staging/lerobot_v3_0/libero",
            )
        ),
    )
    args = parser.parse_args()

    dataset_dirs = [str(args.root / suite) for suite in SUITES]
    shape_meta = {
        "images": [
            {"key": "front", "raw_shape": [3, 128, 128], "shape": [3, 224, 224]},
            {"key": "wrist", "raw_shape": [3, 128, 128], "shape": [3, 224, 224]},
        ],
        "action": [{"key": "default", "raw_shape": 7, "shape": 7}],
        "state": [{"key": "default", "raw_shape": 8, "shape": 8}],
    }
    dataset = BaseLerobotDataset(
        dataset_dirs=dataset_dirs,
        shape_meta=shape_meta,
        action_size=32,
        obs_size=33,
        val_set_proportion=0.0,
        is_training_set=True,
        task_indices=list(range(10)),
    )

    selected_episodes = {
        Path(dataset_dir).name: len(indices)
        for dataset_dir, indices in dataset.selected_episodes.items()
    }
    if selected_episodes != {suite: 500 for suite in SUITES}:
        raise AssertionError(f"Unexpected episode selection: {selected_episodes}")
    if dataset.multi_dataset.num_episodes != 2_000:
        raise AssertionError(
            f"Expected 2000 episodes, got {dataset.multi_dataset.num_episodes}"
        )

    samples = []
    for subdataset in dataset.multi_dataset._datasets:
        sample = subdataset[0]
        samples.append(
            {
                "suite": subdataset.root.name,
                "task": sample["task"],
                "front_shape": list(sample["observation.images.front"].shape),
                "wrist_shape": list(sample["observation.images.wrist"].shape),
                "state_shape": list(sample["observation.state"].shape),
                "action_shape": list(sample["action"].shape),
            }
        )

    print(
        json.dumps(
            {
                "root": str(args.root),
                "total_frames": len(dataset),
                "total_episodes": dataset.multi_dataset.num_episodes,
                "selected_episodes": selected_episodes,
                "samples": samples,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
