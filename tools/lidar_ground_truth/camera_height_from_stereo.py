#!/usr/bin/env python3
"""Estimate the rectified left camera's height above the floor from the
session's rectified stereo snapshots (rig metrology input, PLAN.md Sec 3.1).

For each capture, the rectified <ID>_left.png / <ID>_right.png pair is
matched with OpenCV SGBM over a band of image rows that sees only floor
(plus the occasional board base, rejected by RANSAC), triangulated with the
session's rectified K and stereo baseline, and a plane is fitted. The
camera height is the distance from the optical centre to that plane; the
plane normal also gives the camera's tilt relative to the floor.

This is independent of the ZED SDK's own depth (PLAN.md Sec 9 item 6) and of
the LiDAR. Combined with the LiDAR scan-plane height above the floor it
gives delta_z = z_scan_plane_above_floor - camera_height_above_floor.

Run from the repository root:
    python tools/lidar_ground_truth/camera_height_from_stereo.py \\
        --captures-dir data/<session>/captures \\
        --out results/lidar_ground_truth/<session>/camera_height_from_stereo.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np


def floor_plane(
    left: np.ndarray,
    right: np.ndarray,
    row_band: tuple[int, int],
    fx: float, fy: float, cx: float, cy: float,
    baseline_m: float,
    max_depth_m: float,
    inlier_threshold_m: float,
    rng: np.random.Generator,
) -> dict:
    top, bottom = row_band
    matcher = cv2.StereoSGBM_create(
        minDisparity=0, numDisparities=128, blockSize=7,
        P1=8 * 3 * 49, P2=32 * 3 * 49, uniquenessRatio=10,
        speckleWindowSize=100, speckleRange=2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    # Rectified pairs are row-aligned, so matching a row band is exact.
    disparity = matcher.compute(left[top:bottom], right[top:bottom]).astype(np.float32) / 16.0

    v, u = np.mgrid[top:bottom, 0:left.shape[1]]
    valid = disparity > 2.0
    z = fx * baseline_m / disparity[valid]
    points = np.c_[(u[valid] - cx) * z / fx, (v[valid] - cy) * z / fy, z]
    points = points[points[:, 2] < max_depth_m]

    best_count, best_inliers = 0, None
    for _ in range(400):
        sample = points[rng.choice(len(points), 3, replace=False)]
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        if np.linalg.norm(normal) < 1e-12:
            continue
        normal /= np.linalg.norm(normal)
        inliers = np.abs((points - sample[0]) @ normal) < inlier_threshold_m
        if inliers.sum() > best_count:
            best_count, best_inliers = int(inliers.sum()), inliers

    floor = points[best_inliers]
    centroid = floor.mean(axis=0)
    _, _, vh = np.linalg.svd(floor - centroid, full_matrices=False)
    normal = vh[2]
    if normal[1] < 0:  # OpenCV frame: +y is down, towards the floor
        normal = -normal

    return {
        "camera_height_m": float(normal @ centroid),
        "floor_normal_camera_cv": normal.tolist(),
        # Camera tilt relative to the floor, OpenCV frame (x right, y down, z forward).
        "tilt_about_x_deg": float(math.degrees(math.atan2(normal[2], normal[1]))),
        "tilt_about_z_deg": float(math.degrees(math.atan2(-normal[0], normal[1]))),
        "inliers": best_count,
        "points": int(len(points)),
        "plane_rms_m": float(np.std((floor - centroid) @ normal)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--captures-dir", required=True, help="data/<session>/captures")
    parser.add_argument("--out", required=True, help="output JSON path")
    parser.add_argument(
        "--row-band", nargs=2, type=int, default=[440, 650],
        help="image rows [top, bottom) that see only floor (default excludes the robot body at the bottom)",
    )
    parser.add_argument("--max-depth-m", type=float, default=4.0)
    parser.add_argument("--inlier-threshold-m", type=float, default=0.01)
    args = parser.parse_args()

    captures_dir = Path(args.captures_dir)
    manifest = json.loads((captures_dir / "session_manifest.json").read_text())
    calibration = manifest["calibration"]
    left_k = calibration["rectified"]["left"]
    baseline_m = float(calibration["baseline_m"])

    rng = np.random.default_rng(0)
    per_capture = []
    for capture_dir in sorted(p for p in captures_dir.iterdir() if p.is_dir()):
        capture_id = capture_dir.name
        left = cv2.imread(str(capture_dir / f"{capture_id}_left.png"), cv2.IMREAD_GRAYSCALE)
        right = cv2.imread(str(capture_dir / f"{capture_id}_right.png"), cv2.IMREAD_GRAYSCALE)
        result = floor_plane(
            left, right, tuple(args.row_band),
            left_k["fx"], left_k["fy"], left_k["cx"], left_k["cy"],
            baseline_m, args.max_depth_m, args.inlier_threshold_m, rng,
        )
        result["capture_id"] = capture_id
        per_capture.append(result)
        print(
            f"{capture_id}: camera height {result['camera_height_m'] * 1000:.1f} mm, "
            f"tilt about x {result['tilt_about_x_deg']:+.3f} deg, about z {result['tilt_about_z_deg']:+.3f} deg, "
            f"plane rms {result['plane_rms_m'] * 1000:.1f} mm ({result['inliers']}/{result['points']} inliers)"
        )

    heights = np.array([r["camera_height_m"] for r in per_capture])
    summary = {
        "captures_dir": str(captures_dir),
        "baseline_m": baseline_m,
        "rectified_left_K": [left_k["fx"], left_k["fy"], left_k["cx"], left_k["cy"]],
        "row_band": args.row_band,
        "camera_height_m_median": float(np.median(heights)),
        "camera_height_m_std": float(np.std(heights)),
        "camera_height_m_range": [float(heights.min()), float(heights.max())],
        "tilt_about_x_deg_median": float(np.median([r["tilt_about_x_deg"] for r in per_capture])),
        "tilt_about_z_deg_median": float(np.median([r["tilt_about_z_deg"] for r in per_capture])),
        "per_capture": per_capture,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(
        f"camera height above floor: median {summary['camera_height_m_median'] * 1000:.1f} mm "
        f"(std {summary['camera_height_m_std'] * 1000:.1f} mm over {len(per_capture)} captures)"
    )
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
