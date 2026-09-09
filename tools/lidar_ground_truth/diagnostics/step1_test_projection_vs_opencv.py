#!/usr/bin/env python3
"""Phase 1.5 Step 1: unit-test projection.py's pixel formula against
cv2.projectPoints, to exonerate or convict the NEW projection code before
looking anywhere else.

Method: generate random points in camera "robot" frame R (x_R, y_R, z_R),
including points off-axis near the image edges. Convert to OpenCV frame C
via the fixed rotation already used throughout this repo
(gui.py::ImageVisInterface.done_callback: X_C = -y_R, Y_C = -z_R, Z_C = x_R),
project with cv2.projectPoints (identity extrinsics, rectified K, dist=0) to
get a reference (u, v). Separately, call the SHIPPED project_scan() with
T2 = identity and delta_z_m = z_R (looped per point, since project_scan
takes one scalar delta_z per call) to get project_scan's own (u, v). Assert
agreement to 1e-6 px.

This is read-only / additive: it does not touch cam_lidar_2d_icp.py, gui.py,
icp_2d.py, or tools/lidar_characterisation.

Run from the repository root:
    python tools/lidar_ground_truth/diagnostics/step1_test_projection_vs_opencv.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # tools/lidar_ground_truth/

import cv2
import numpy as np

import projection

RNG_SEED = 0
N_POINTS = 500
TOL_PX = 1e-6


def cam_robot_to_opencv(x_r, y_r, z_r):
    """R -> C, per gui.py::ImageVisInterface.done_callback's rot_cam_to_robot
    (this is its inverse direction: R frame -> OpenCV C frame).

    rot_cam_to_robot maps C -> R via X_R = Z_C, Y_R = -X_C, Z_R = -Y_C
    (the matrix product rot_zn90 @ rot_xn90 applied to a C-frame point).
    We need the inverse (R -> C) for this test since project_scan/K expect
    C-frame pinhole geometry: solving that 3-equation system gives
    X_C = -y_R, Y_C = -z_R, Z_C = x_R -- exactly the mapping already stated
    in gui.py's own inline comments and in PLAN.md Sec 1.1.
    """
    return -y_r, -z_r, x_r


def build_test_points(n: int, rng: np.random.Generator) -> np.ndarray:
    """(N, 3) points in frame R: x_R (depth) in [0.2, 8] m, y_R/z_R spanning
    well beyond the image edges at typical depths so off-axis/edge cases are
    exercised, not just near-centre points.
    """
    x_r = rng.uniform(0.2, 8.0, size=n)
    # At x_r ~ 0.2 m and fx ~ 522, +-0.5 rad-ish lateral offsets already
    # leave the frame; scale y/z offset by depth so a healthy fraction of
    # points land inside the image (and the rest legitimately fall outside,
    # exercising project_scan's keep mask / cv2's behaviour together).
    y_r = rng.uniform(-1.2, 1.2, size=n) * (x_r / 3.0 + 0.2)
    z_r = rng.uniform(-0.8, 0.8, size=n) * (x_r / 3.0 + 0.2)
    return np.stack([x_r, y_r, z_r], axis=1)


def project_with_opencv(points_r: np.ndarray, K: np.ndarray) -> np.ndarray:
    x_c, y_c, z_c = cam_robot_to_opencv(points_r[:, 0], points_r[:, 1], points_r[:, 2])
    points_c = np.stack([x_c, y_c, z_c], axis=1).astype(np.float64)
    rvec = np.zeros((3, 1), dtype=np.float64)
    tvec = np.zeros((3, 1), dtype=np.float64)
    dist = np.zeros((5, 1), dtype=np.float64)
    uv, _ = cv2.projectPoints(points_c, rvec, tvec, K, dist)
    return uv.reshape(-1, 2)


def project_with_project_scan(points_r: np.ndarray, K: np.ndarray, image_wh) -> np.ndarray:
    """Loop per point: project_scan takes one scalar delta_z per call, and
    here every point has its own z_R.
    """
    identity_T2 = np.eye(3, dtype=np.float64)
    out = np.full((len(points_r), 2), np.nan, dtype=np.float64)
    for i, (x_r, y_r, z_r) in enumerate(points_r):
        result = projection.project_scan(
            xy_lidar=np.array([[x_r, y_r]], dtype=np.float64),
            T2=identity_T2,
            delta_z_m=float(z_r),
            K=K,
            image_wh=image_wh,
            z_min_m=1e-6,
        )
        out[i, 0] = result.u[0]
        out[i, 1] = result.v[0]
    return out


def main() -> None:
    K = np.array([
        [522.00367955, 0.0, 636.71609061],
        [0.0, 522.00367955, 355.69182042],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    # Deliberately oversized "image" so points near/outside the real 1280x720
    # frame are still returned (keep=True) by project_scan and can be
    # compared -- this test is about the pixel FORMULA, not the frame-bounds
    # clipping (which is separately correct/trivial).
    image_wh = (100000, 100000)

    rng = np.random.default_rng(RNG_SEED)
    points_r = build_test_points(N_POINTS, rng)

    uv_opencv = project_with_opencv(points_r, K)
    uv_project_scan = project_with_project_scan(points_r, K, image_wh)

    diff = np.abs(uv_opencv - uv_project_scan)
    max_diff = float(diff.max())
    mean_diff = float(diff.mean())

    print(f"N points tested: {N_POINTS}")
    print(f"max |u,v| difference vs cv2.projectPoints: {max_diff:.3e} px")
    print(f"mean |u,v| difference vs cv2.projectPoints: {mean_diff:.3e} px")

    passed = max_diff < TOL_PX
    print(f"PASSED (< {TOL_PX:.0e} px)" if passed else f"FAILED (>= {TOL_PX:.0e} px)")

    if not passed:
        worst_idx = np.argmax(diff.max(axis=1))
        print("Worst offending point (frame R x_R, y_R, z_R):", points_r[worst_idx])
        print("  cv2.projectPoints (u, v):     ", uv_opencv[worst_idx])
        print("  project_scan       (u, v):    ", uv_project_scan[worst_idx])
        sys.exit(1)


if __name__ == "__main__":
    main()
