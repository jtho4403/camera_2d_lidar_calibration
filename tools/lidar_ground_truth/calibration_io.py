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


def load_transform(path: str | Path = config.DEFAULT_TRANSFORM_PATH) -> np.ndarray:
    """Load the 3x3 SE(2) transform T2: LiDAR frame L -> camera frame R
    (lidar_to_camera_2d.npy). [x_R, y_R]^T = R2 [x_L, y_L]^T + t2.
    """
    T2 = np.load(path)
    if T2.shape != (3, 3):
        raise ValueError(f"Expected a 3x3 SE(2) transform in {path}, got shape {T2.shape}")
    return T2


def load_calibration_result(
    path: str | Path = config.DEFAULT_CALIBRATION_RESULT_PATH,
) -> dict:
    """Load calibration_result.json (transform, residuals, sanity checks)."""
    with open(path) as f:
        return json.load(f)


@dataclass
class RectifiedIntrinsics:
    K: np.ndarray               # (3, 3) rectified intrinsics; distortion is exactly zero
    image_wh: tuple[int, int]   # (W, H) pixels
    source: str                  # provenance string, for the export summary JSON


def load_intrinsics_from_metadata(metadata_path: str | Path) -> RectifiedIntrinsics:
    """Load the ZED SDK's per-capture rectified LEFT-camera intrinsics.

    PLAN.md Sec 4.3: the canonical K is the ZED SDK rectified left-camera
    intrinsics recorded per-capture in metadata_pose_NN.json, not the
    hardcoded camera_k constant in cam_lidar_2d_icp.py (that constant is
    retained there only as a validation reference, unmodified). Every
    metadata_pose_NN.json checked in this repo (data_2026-06-28) has
    left_camera.disto entirely zero, i.e. images are already rectified;
    distortion is treated as exactly zero here and non-zero disto is a hard
    error rather than something silently ignored.
    """
    metadata_path = Path(metadata_path)
    with open(metadata_path) as f:
        meta = json.load(f)

    left = meta["camera_info"]["left_camera"]
    if any(float(d) != 0.0 for d in left.get("disto", [])):
        raise ValueError(
            f"{metadata_path}: left_camera.disto is non-zero. This pipeline "
            "assumes rectified (zero-distortion) images (PLAN.md Sec 6.2) -- "
            "do not project through this K without resolving that first."
        )

    K = np.array([
        [left["fx"], 0.0, left["cx"]],
        [0.0, left["fy"], left["cy"]],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    w_str, h_str = meta["camera_info"]["resolution"].split("x")
    image_wh = (int(w_str), int(h_str))

    return RectifiedIntrinsics(K=K, image_wh=image_wh, source=str(metadata_path))


def load_intrinsics_from_calibration_result(
    calibration_result_path: str | Path = config.DEFAULT_CALIBRATION_RESULT_PATH,
) -> RectifiedIntrinsics:
    """Fallback K source: the hardcoded intrinsics recorded in
    calibration_result.json, for data with no per-pose metadata_pose_NN.json
    (e.g. examples/). See load_intrinsics_from_metadata's docstring for why
    per-pose metadata is preferred when available -- the two sources differ
    by well under a pixel in fx/fy/cx/cy for the checked data_2026-06-28
    session, so this fallback is not a meaningful source of error, but it is
    not the canonical source either.
    """
    result = load_calibration_result(calibration_result_path)
    K = np.array(result["camera_intrinsics"], dtype=np.float64)
    dist = np.array(result["camera_distortion"], dtype=np.float64)
    if np.any(dist != 0.0):
        raise ValueError(
            f"{calibration_result_path}: camera_distortion is non-zero. "
            "This pipeline assumes rectified (zero-distortion) images."
        )
    return RectifiedIntrinsics(
        K=K, image_wh=config.FALLBACK_IMAGE_WH, source=str(calibration_result_path)
    )
