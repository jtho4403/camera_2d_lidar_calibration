#!/usr/bin/env python3
"""Phase 1.5 Step 5: definitively rule out (or confirm) unrectified images.

metadata_pose_NN.json's disto=0 says what the ZED SDK reported, not
necessarily what image the (missing) capture script actually saved. With
fx ~ 522 at 1280 px width the horizontal FoV is ~101 deg; if the saved
images were actually unrectified, straight lines in the scene would bow
noticeably (tens of px) near the edges.

Definitive test: run cv2.findChessboardCorners on the calibration images,
group the detected corners into rows (the checkerboard grid gives 4 rows of
6 points each), fit a straight line to each row, and report the maximum
perpendicular deviation from that line. Rectified: sub-pixel. Unrectified at
this FoV: many pixels of bow.

Read-only against cam_lidar_2d_icp.py / gui.py / icp_2d.py /
tools/lidar_characterisation -- reimplements only their published,
unmodified checkerboard-detection parameters.

Run from the repository root:
    python tools/lidar_ground_truth/diagnostics/step5_rectification_check.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # tools/lidar_ground_truth/

import cv2
import numpy as np

import config

SESSION_DIR = config.REPO_ROOT / "data" / "data_2026-06-28"
OUT_DIR = config.REPO_ROOT / "results" / "lidar_ground_truth" / "diagnostics" / "step5_rectification_check"

CHECKERBOARD_WIDTH = 6
CHECKERBOARD_HEIGHT = 4
CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def max_perpendicular_deviation(points_2d: np.ndarray) -> float:
    """Fit a straight (total-least-squares) line to points_2d (N, 2) and
    return the maximum absolute perpendicular deviation, in the same units
    as points_2d (pixels here).
    """
    centroid = points_2d.mean(axis=0)
    centred = points_2d - centroid
    _, _, vh = np.linalg.svd(centred, full_matrices=False)
    normal = vh[1]  # second singular vector = direction perpendicular to the fitted line
    deviations = centred @ normal
    return float(np.max(np.abs(deviations)))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    image_dir = SESSION_DIR / "images"

    per_pose = []
    overall_max = 0.0
    for image_path in sorted(image_dir.glob("*.png")):
        pose_id = image_path.stem
        image_bgr = cv2.imread(str(image_path))
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (CHECKERBOARD_WIDTH, CHECKERBOARD_HEIGHT), None)
        if not ret:
            print(f"{pose_id}: checkerboard NOT detected, skipping")
            continue
        corners2 = cv2.cornerSubPix(gray, corners, (3, 3), (-1, -1), CRITERIA)
        grid = corners2.reshape(CHECKERBOARD_HEIGHT, CHECKERBOARD_WIDTH, 2)

        row_max_devs = [max_perpendicular_deviation(grid[r]) for r in range(CHECKERBOARD_HEIGHT)]
        col_max_devs = [max_perpendicular_deviation(grid[:, c]) for c in range(CHECKERBOARD_WIDTH)]
        pose_max = max(row_max_devs + col_max_devs)
        overall_max = max(overall_max, pose_max)

        per_pose.append({
            "pose_id": pose_id,
            "row_max_deviation_px": row_max_devs,
            "col_max_deviation_px": col_max_devs,
            "pose_max_deviation_px": pose_max,
        })
        print(f"{pose_id}: row max deviations (px) = {[round(d, 3) for d in row_max_devs]}, "
              f"col max deviations (px) = {[round(d, 3) for d in col_max_devs]}, "
              f"pose max = {pose_max:.3f} px")

    print()
    print(f"Overall max perpendicular deviation across all poses/rows/cols: {overall_max:.3f} px")
    verdict = ("RECTIFIED (sub-pixel bow) -- unrectified-image hypothesis REJECTED"
               if overall_max < 1.0 else
               "NOT sub-pixel -- unrectified-image hypothesis cannot be ruled out")
    print(verdict)

    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump({"poses": per_pose, "overall_max_px": overall_max, "verdict": verdict}, f, indent=2)
    print(f"Saved {OUT_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
