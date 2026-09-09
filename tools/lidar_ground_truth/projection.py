"""Sparse LiDAR -> rectified-camera-pixel projection core (PLAN.md Sec 1.2, 6.2).

    d_gt = x_R                        ground-truth depth (camera-frame Z)
    u    = c_x - f_x * y_R / x_R      image column
    v    = c_y - f_y * delta_z / x_R  image row

where [x_R, y_R]^T = R2 [x_L, y_L]^T + t2 (the existing SE(2) calibration,
PLAN.md Sec 4.1) and delta_z is the measured rig constant (PLAN.md Sec 3,
rig.py). d_gt and u depend only on the SE(2) transform T2; delta_z, pitch and
roll affect only v (PLAN.md Sec 1.2). Never let a vertical parameter leak
into the depth or column computation -- that asymmetry is the reason this
whole approach is defensible.

Rectified images, rectified K, distortion exactly zero. Do not use
cv2.projectPoints here -- it adds no value for a flat-plane LiDAR return and
invites extrinsics/distortion confusion (PLAN.md hard constraints).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ProjectionResult:
    u: np.ndarray                      # image column, pixels (float; may be out of [0, W) where keep is False)
    v: np.ndarray                      # image row, pixels (float; may be out of [0, H) where keep is False)
    depth_m: np.ndarray                # camera-frame Z (frame R, PLAN.md Sec 1.1), NOT radial range
    lidar_radial_range_m: np.ndarray   # hypot(x_L, y_L); provenance only -- never compare to depth_m
    keep: np.ndarray                   # bool: in front of the camera (x_R > z_min_m) AND inside image bounds


def project_scan(
    xy_lidar: np.ndarray,
    T2: np.ndarray,
    delta_z_m: float,
    K: np.ndarray,
    image_wh: tuple[int, int],
    z_min_m: float = 0.05,
) -> ProjectionResult:
    """Project 2D LiDAR returns into rectified camera pixels.

    xy_lidar  : (N, 2) float, LiDAR frame L [m], x forward / y left
                (PLAN.md Sec 1.1).
    T2        : (3, 3) SE(2) homogeneous transform, LiDAR frame L -> camera
                "robot" frame R (lidar_to_camera_2d.npy).
    delta_z_m : scan-plane height above the rectified left camera optical
                centre, expressed in frame R [m]. Positive = LiDAR higher
                than the camera (rig.py).
    K         : (3, 3) rectified intrinsics; distortion must be exactly zero.
    image_wh  : (W, H) in pixels.
    z_min_m   : minimum camera-frame depth to keep. Guards the x_R divide --
                returns behind (or at) the camera would otherwise project to
                plausible-looking garbage pixels rather than an obvious
                failure.
    """
    xy_lidar = np.asarray(xy_lidar, dtype=np.float64)
    if xy_lidar.ndim != 2 or xy_lidar.shape[1] != 2:
        raise ValueError(f"Expected an Nx2 LiDAR point array, got shape {xy_lidar.shape}")

    R2, t2 = T2[:2, :2], T2[:2, 2]
    xy_r = xy_lidar @ R2.T + t2
    x_r, y_r = xy_r[:, 0], xy_r[:, 1]

    in_front = x_r > z_min_m
    x_r_safe = np.where(in_front, x_r, 1.0)  # avoid a real divide-by-zero/negative for rejected points

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    u = cx - fx * y_r / x_r_safe
    v = cy - fy * delta_z_m / x_r_safe
    depth_m = x_r  # camera-frame Z, NOT radial range

    W, H = image_wh
    keep = in_front & (u >= 0) & (u < W) & (v >= 0) & (v < H)

    lidar_radial_range_m = np.hypot(xy_lidar[:, 0], xy_lidar[:, 1])

    return ProjectionResult(
        u=u,
        v=v,
        depth_m=depth_m,
        lidar_radial_range_m=lidar_radial_range_m,
        keep=keep,
    )
