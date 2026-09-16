#!/usr/bin/env python3
"""Download an official ST-WAM LIBERO checkpoint from ModelScope.

The released repository contains several ST-WAM snapshots.  This helper
keeps the ModelScope path below a shared checkpoint root so the resulting
paths can be passed directly to the LIBERO evaluator.

Example:

    python scripts/download_official_stwam.py \
      --variant base \
      --root /mnt/data/embodied_datasets/ST-WAM/checkpoints/modelscope/Semantic_Temporal_World_Action_Model
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


MODEL_ID = "THU4Spiderman/Semantic_Temporal_World_Action_Model"
VARIANT_PATHS = {
    "base": "libero/ST-WAM/st-wam-base-step021700",
    "extra": "libero/ST-WAM/st-wam-extra-step021700",
}
FILES = ("config.yaml", "dataset_stats.json", "model.pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=sorted(VARIANT_PATHS), default="base")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(
            "/mnt/data/embodied_datasets/ST-WAM/checkpoints/modelscope/"
            "Semantic_Temporal_World_Action_Model"
        ),
        help="Local root that will contain the ModelScope directory structure.",
    )
    parser.add_argument("--revision", default="master")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="ModelScope cache directory (default: <root-parent>/.cache).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Parallel range-download workers for the large model file.",
    )
    parser.add_argument(
        "--intra-cloud",
        action="store_true",
        help="Use ModelScope intra-cloud acceleration when available.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be positive")

    # These are read by ModelScope at import time.  The internal OSS endpoint
    # is not reachable from every runtime, so public CDN download is the safe
    # default; callers can opt in explicitly.
    os.environ.setdefault("MODELSCOPE_DOWNLOAD_PARALLELS", str(args.workers))
    os.environ.setdefault(
        "INTRA_CLOUD_ACCELERATION", "true" if args.intra_cloud else "false"
    )

    from modelscope.hub.api import HubApi
    from modelscope.hub.file_download import model_file_download

    args.root.mkdir(parents=True, exist_ok=True)
    cache_dir = args.cache_dir or args.root.parent / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    prefix = VARIANT_PATHS[args.variant]

    # Obtain sizes first.  This makes interrupted downloads distinguishable
    # from complete files and lets a rerun skip verified files.
    api = HubApi()
    metadata = {
        item["Path"]: int(item["Size"])
        for item in api.get_model_files(
            MODEL_ID, revision=args.revision, recursive=True
        )
        if item.get("Type") == "blob" and item.get("Path", "").startswith(prefix + "/")
    }

    for name in FILES:
        remote_path = f"{prefix}/{name}"
        destination = args.root / remote_path
        expected_size = metadata.get(remote_path)
        if expected_size is None:
            raise FileNotFoundError(
                f"ModelScope file metadata not found for {MODEL_ID}:{remote_path}"
            )
        if destination.is_file() and destination.stat().st_size == expected_size:
            print(f"skip complete: {destination} ({expected_size} bytes)", flush=True)
            continue

        print(
            f"download: {MODEL_ID}:{remote_path} -> {destination} "
            f"({expected_size} bytes)",
            flush=True,
        )
        saved = model_file_download(
            MODEL_ID,
            remote_path,
            revision=args.revision,
            cache_dir=str(cache_dir),
            local_dir=str(args.root),
        )
        saved_path = Path(saved)
        if not saved_path.is_file() or saved_path.stat().st_size != expected_size:
            raise RuntimeError(
                f"Downloaded file size mismatch: {saved_path} "
                f"got={saved_path.stat().st_size if saved_path.exists() else None} "
                f"expected={expected_size}"
            )
        print(f"saved: {saved_path}", flush=True)

    checkpoint = args.root / f"{prefix}/model.pt"
    print(f"OFFICIAL_CKPT={checkpoint}")
    print(f"OFFICIAL_CONFIG={args.root / f'{prefix}/config.yaml'}")
    print(f"OFFICIAL_STATS={args.root / f'{prefix}/dataset_stats.json'}")


if __name__ == "__main__":
    main()
