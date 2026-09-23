"""Loaders for calibration artifacts consumed unchanged from the existing
camera_2d_lidar_calibration package (PLAN.md Sec 4.1).

cam_lidar_2d_icp.py itself is not modified beyond the single additive
correspondence dump (PLAN.md Sec 4.2, Phase 2); these loaders only read its
existing outputs.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import config


def load_transform(path: str | Path) -> np.ndarray:
    """Load the 3x3 SE(2) transform T2: LiDAR frame L -> camera frame R
    (lidar_to_camera_2d.npy). [x_R, y_R]^T = R2 [x_L, y_L]^T + t2.
    """
    T2 = np.load(path)
    if T2.shape != (3, 3):
        raise ValueError(f"Expected a 3x3 SE(2) transform in {path}, got shape {T2.shape}")
    return T2


def load_calibration_result(path: str | Path) -> dict:
    """Load calibration_result.json (transform, residuals, sanity checks)."""
    with open(path) as f:
        return json.load(f)


@dataclass
class RectifiedIntrinsics:
    K: np.ndarray               # (3, 3) rectified intrinsics; distortion is exactly zero
    image_wh: tuple[int, int]   # (W, H) pixels
    source: str                  # provenance string, for the export summary JSON


def load_intrinsics_from_manifest(manifest_path: str | Path) -> RectifiedIntrinsics:
    """Load the ZED SDK rectified LEFT-camera intrinsics from a capture
    session's session_manifest.json (calibration.rectified.left).

    PLAN.md Sec 4.3: the canonical K is the ZED SDK rectified left-camera
    intrinsics, recorded once per capture session. Non-zero rectified
    distortion means the frames are not rectified, which is a hard error
    rather than something silently ignored.
    """
    manifest_path = Path(manifest_path)
    with open(manifest_path) as f:
        manifest = json.load(f)

    left = manifest["calibration"]["rectified"]["left"]
    if any(float(d) != 0.0 for d in left["disto"]):
        raise ValueError(
            f"{manifest_path}: calibration.rectified.left.disto is non-zero. "
            "This pipeline assumes rectified (zero-distortion) images "
            "(PLAN.md Sec 6.2) -- do not project through this K without "
            "resolving that first."
        )

    K = np.array([
        [left["fx"], 0.0, left["cx"]],
        [0.0, left["fy"], left["cy"]],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    image_wh = tuple(int(v) for v in left["image_size"])

    return RectifiedIntrinsics(K=K, image_wh=image_wh, source=str(manifest_path))


def check_intrinsics_match_calibration(
    intrinsics: RectifiedIntrinsics,
    calibration_result: dict,
    tolerance_px: float = 1e-6,
) -> None:
    """Refuse to project with a K that differs from the one the SE(2)
    calibration was solved with (e.g. a manifest from another session).
    """
    calibration_K = np.array(calibration_result["camera_intrinsics"], dtype=np.float64)
    if not np.allclose(calibration_K, intrinsics.K, atol=tolerance_px):
        raise ValueError(
            f"K from {intrinsics.source} does not match the K recorded in the "
            "calibration result (camera_intrinsics_source="
            f"{calibration_result.get('camera_intrinsics_source')!r}). Use the "
            "calibration and the images from the same capture session."
        )
