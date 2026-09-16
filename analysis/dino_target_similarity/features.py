"""Frozen DINOv3 feature extraction using ST-WAM's actual encoder wrapper."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F

from fastwam.models.wan22.dino_encoder import DinoVideoEncoder


def _resolve_path(path: str | Path, root: Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else root / path


def checkpoint_identifier(path: str | Path, root: Path) -> str:
    """Return a reproducible checkpoint identifier without requiring a cache hit."""

    resolved = _resolve_path(path, root)
    if not resolved.exists():
        return f"missing:{resolved}"
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def resolve_model_resolution(model_cfg: Mapping[str, Any], resolution_mode: str = "native") -> tuple[int, int]:
    native = tuple(int(x) for x in model_cfg.get("input_resolution", [224, 224]))
    if len(native) != 2:
        raise ValueError(f"model.input_resolution must have two values, got {native}")
    if resolution_mode == "native":
        return native
    if resolution_mode == "high_resolution":
        configured = model_cfg.get("high_resolution")
        if configured is None:
            # Preserve the current two-camera aspect ratio and patch alignment.
            configured = [native[0] * 2, native[1] * 2]
        high = tuple(int(x) for x in configured)
        if len(high) != 2:
            raise ValueError(f"model.high_resolution must have two values, got {high}")
        return high
    raise ValueError(f"Unsupported resolution_mode={resolution_mode!r}")


def _dtype_from_name(name: str, device: torch.device) -> torch.dtype:
    name = str(name).lower()
    if device.type == "cpu":
        return torch.float32
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    return torch.float32


class DINOFeatureExtractor:
    """Encode camera frames in batches and retain a metadata-rich contract."""

    def __init__(
        self,
        model_cfg: Mapping[str, Any],
        *,
        repo_root: str | Path,
        resolution_mode: str = "native",
        device: str = "auto",
        dtype: str = "auto",
        view_mode: str = "current_concat",
    ) -> None:
        self.repo_root = Path(repo_root)
        self.model_cfg = dict(model_cfg)
        self.resolution_mode = resolution_mode
        self.view_mode = view_mode
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("DINO analysis requested CUDA, but torch.cuda.is_available() is false")
        self.resolution = resolve_model_resolution(model_cfg, resolution_mode)
        patch_size = int(model_cfg.get("patch_size", 16))
        if any(size % patch_size for size in self.resolution):
            raise ValueError(
                f"DINO resolution {self.resolution} must be divisible by patch_size={patch_size}"
            )
        if view_mode not in {"current_concat", "independent"}:
            raise ValueError(f"Unsupported view_mode={view_mode!r}")
        if view_mode == "independent":
            if len(self.resolution) != 2:
                raise AssertionError("invalid resolution")
            # An independent view uses a square native/high-resolution frame.
            if self.resolution[1] != self.resolution[0] * 2:
                self.view_resolution = self.resolution
            else:
                self.view_resolution = (self.resolution[0], self.resolution[0])
        else:
            self.view_resolution = None
        dtype_name = "bf16" if dtype == "auto" and self.device.type == "cuda" else dtype
        self.dtype = _dtype_from_name(dtype_name, self.device)
        model_path = _resolve_path(str(model_cfg["model_path"]), self.repo_root)
        if not model_path.is_file():
            raise FileNotFoundError(
                f"Configured ST-WAM DINO checkpoint is missing: {model_path}. "
                "Refusing to silently fall back to another DINO model."
            )
        encoder_resolution = self.resolution if view_mode == "current_concat" else self.view_resolution
        self.encoder = DinoVideoEncoder(
            model_name=str(model_cfg.get("model_name", "dinov3-vits16-timm")),
            model_path=str(model_path),
            input_resolution=encoder_resolution,
            patch_size=patch_size,
            feature_dim=int(model_cfg.get("feature_dim", 384)),
            use_cls_token=bool(model_cfg.get("use_cls_token", False)),
            normalize_features=bool(model_cfg.get("normalize_features", False)),
            latent_spatial_pool=tuple(model_cfg.get("latent_spatial_pool", [1, 1])),
            encode_microbatch_size=int(model_cfg.get("encode_microbatch_size", 16)),
        ).to(device=self.device)
        self.encoder.load_backbone(device=self.device, dtype=self.dtype)
        self.encoder.eval()
        self.model_path = model_path
        self.patch_size = patch_size
        self.feature_dim = int(model_cfg.get("feature_dim", 384))
        self.grid_size = tuple(int(x) // patch_size for x in encoder_resolution)
        self.checkpoint_id = checkpoint_identifier(model_path, self.repo_root)

    def metadata(self, camera_names: list[str]) -> dict[str, Any]:
        return {
            "model_name": str(self.model_cfg.get("model_name", "dinov3-vits16-timm")),
            "checkpoint": str(self.model_path),
            "checkpoint_identifier": self.checkpoint_id,
            "input_resolution": list(self.encoder.input_resolution),
            "resolution_mode": self.resolution_mode,
            "view_mode": self.view_mode,
            "patch_size": self.patch_size,
            "patch_grid": list(self.grid_size),
            "feature_dim": self.feature_dim,
            "camera_names": list(camera_names),
            "normalization": "uint8 -> [-1,1] -> ImageNet mean/std inside DinoVideoEncoder",
            "feature_output": "x_norm_patchtokens (CLS/register tokens excluded)",
            "encoder_normalize_features": bool(self.model_cfg.get("normalize_features", False)),
        }

    @staticmethod
    def _resize_camera_batch(frames: np.ndarray, size: tuple[int, int]) -> torch.Tensor:
        if frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError(f"Camera frames must be [T,H,W,3], got {frames.shape}")
        tensor = torch.from_numpy(np.ascontiguousarray(frames)).permute(0, 3, 1, 2).float()
        tensor = tensor / 127.5 - 1.0
        if tuple(tensor.shape[-2:]) != tuple(size):
            tensor = F.interpolate(tensor, size=size, mode="bilinear", align_corners=False, antialias=True)
        return tensor

    def _encode(self, tensor: torch.Tensor) -> torch.Tensor:
        microbatch = int(self.encoder.encode_microbatch_size)
        if microbatch <= 0:
            microbatch = len(tensor)
        outputs: list[torch.Tensor] = []
        with torch.inference_mode():
            for start in range(0, len(tensor), microbatch):
                batch = tensor[start : start + microbatch].to(device=self.device, dtype=self.dtype)
                outputs.append(self.encoder.encode_frames(batch).float().cpu())
        features = torch.cat(outputs, dim=0)
        expected = self.grid_size[0] * self.grid_size[1]
        if features.ndim != 3 or features.shape[1] != expected or features.shape[2] != self.feature_dim:
            raise ValueError(
                f"DINO feature shape mismatch: got {tuple(features.shape)}, "
                f"expected [T,{expected},{self.feature_dim}] for grid={self.grid_size}"
            )
        return features.reshape(len(tensor), self.grid_size[0], self.grid_size[1], self.feature_dim)

    def encode_cameras(self, camera_frames: Mapping[str, np.ndarray]) -> dict[str, torch.Tensor]:
        """Return per-camera ``[T,Hpatch,Wpatch,D]`` patch tokens.

        ``current_concat`` is the exact current ST-WAM representation: each
        camera is resized to its 224×224 input, concatenated horizontally, and
        passed through the same DINO encoder configured for 224×448.  The
        resulting spatial tokens are split back into views only after DINO.
        This preserves cross-view attention and avoids averaging views.
        """

        names = list(camera_frames)
        if not names:
            raise ValueError("At least one camera is required")
        lengths = {int(np.asarray(camera_frames[name]).shape[0]) for name in names}
        if len(lengths) != 1:
            raise ValueError(f"Camera frame counts differ: {lengths}")
        if self.view_mode == "current_concat":
            if len(names) == 1:
                combined = self._resize_camera_batch(camera_frames[names[0]], self.resolution)
                encoded = self._encode(combined)
                return {names[0]: encoded}
            per_camera_width = self.resolution[1] // len(names)
            if self.resolution[1] % len(names) != 0:
                raise ValueError(f"Concat width {self.resolution[1]} is not divisible by cameras {names}")
            resized = [
                self._resize_camera_batch(camera_frames[name], (self.resolution[0], per_camera_width))
                for name in names
            ]
            combined = torch.cat(resized, dim=-1)
            encoded = self._encode(combined)
            patch_width = self.grid_size[1] // len(names)
            if self.grid_size[1] % len(names) != 0:
                raise ValueError(f"Patch grid width {self.grid_size[1]} is not divisible by cameras {names}")
            return {
                name: encoded[:, :, index * patch_width : (index + 1) * patch_width]
                for index, name in enumerate(names)
            }

        outputs: dict[str, torch.Tensor] = {}
        for name in names:
            tensor = self._resize_camera_batch(camera_frames[name], self.view_resolution)
            outputs[name] = self._encode(tensor)
        return outputs
