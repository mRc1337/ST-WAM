#!/usr/bin/env python
"""Run frozen-DINO target-correspondence analysis.

Examples:
  python scripts/analysis/run_dino_target_analysis.py --stage all --config configs/analysis/dino_target_similarity.yaml
  python scripts/analysis/run_dino_target_analysis.py --stage extract --run-dir outputs/dino_target_similarity/pilot
  python scripts/analysis/run_dino_target_analysis.py --stage plot --run-dir outputs/dino_target_similarity/pilot
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from analysis.dino_target_similarity.pipeline import (  # noqa: E402
    analyze_features,
    extract_features,
    load_config,
    plot_from_existing_run,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/analysis/dino_target_similarity.yaml")
    parser.add_argument("--stage", choices=["extract", "analyze", "plot", "all"], default="all")
    parser.add_argument("--run-dir", default=None, help="Output run directory; defaults to analysis.output_dir/run_name")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--episode-id", action="append", default=None)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper()), format="%(asctime)s %(levelname)s %(message)s")
    config_path = (REPO_ROOT / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)
    config = load_config(config_path)
    config.setdefault("repo_root", str(REPO_ROOT))
    if not Path(str(config["repo_root"])).is_absolute():
        config["repo_root"] = str((REPO_ROOT / str(config["repo_root"])).resolve())
    if args.max_episodes is not None:
        config.setdefault("input", {})["max_episodes"] = args.max_episodes
    if args.episode_id:
        config.setdefault("input", {})["episode_ids"] = args.episode_id

    analysis_cfg = config.setdefault("analysis", {})
    run_name = args.run_name or str(analysis_cfg.get("run_name", "pilot"))
    if args.run_dir is not None:
        run_dir = Path(args.run_dir)
    else:
        output_root = Path(str(analysis_cfg.get("output_dir", "outputs/dino_target_similarity")))
        if not output_root.is_absolute():
            output_root = REPO_ROOT / output_root
        run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(config_path, run_dir / "config.yaml")

    manifest = None
    if args.stage in {"extract", "all"}:
        manifest = extract_features(config, run_dir)
        print(f"Feature extraction complete: {run_dir / 'run_manifest.json'}")
    if args.stage in {"analyze", "all"}:
        frame_df = analyze_features(config, run_dir, manifest=manifest)
        print(f"Analysis complete: {run_dir / 'summary.md'} ({len(frame_df)} frame-camera rows)")
    if args.stage == "plot":
        plot_from_existing_run(config, run_dir)
        print(f"Plots refreshed: {run_dir / 'plots'}")


if __name__ == "__main__":
    main()
