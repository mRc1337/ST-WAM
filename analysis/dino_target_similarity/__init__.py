"""DINOv3 target-correspondence analysis for ST-WAM rollouts."""

from .metrics import compute_frame_metrics
from .prototype import build_target_prototype

__all__ = ["build_target_prototype", "compute_frame_metrics"]
