#!/usr/bin/env python3
"""Phase 1.5 Step 2: round-trip the calibration's OWN checkerboard geometry
through project_scan, entirely independent of the LiDAR and of T2.

ImageVisInterface (gui.py) derives the checkerboard wall line in camera
"robot" frame R from solvePnP's rvec/tvec alone -- no LiDAR data, no T2
involved (gui.py:191-224, board_origin / tf_robot_board_to_robot). This
script REPLICATES that computation (read-only: gui.py is not imported or
modified) but keeps the z_R component that gui.py drops (it only needs x,y
for the 2D ICP line), then projects every checkerboard OBJECT point through
project_scan (T2 = identity, delta_z_m = that point's own z_R) and compares
against the SAME points' actual detected pixel corners (corners2 -- exact
sub-pixel accuracy, from the exact same cv2.findChessboardCorners +
solvePnP pipeline cam_lidar_2d_icp.py::main() runs).

If this round-trip lands correctly (sub-pixel residual): project_scan and
the R<->C frame convention are right, and the fault (if the Phase 1.5
horizontal misalignment is real) is upstream in T2 or the LiDAR data
convention -- NOT in the new projection code.

If it does NOT land correctly: the fault is in the pixel formula or frame
convention themselves (though Step 1 already found project_scan agrees with
cv2.projectPoints to 1e-13 px, so a Step 2 failure would point at THIS
script's replica of the R-frame geometry, or expose something Step 1's
synthetic test didn't cover).

Read-only against cam_lidar_2d_icp.py / gui.py / icp_2d.py / tools/lidar_characterisation --
this script only reimplements their published, unmodified math to test it
independently.

Run from the repository root:
    python tools/lidar_ground_truth/diagnostics/step2_board_roundtrip.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # tools/lidar_ground_truth/

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import calibration_io
import config
import projection

REPO_ROOT = config.REPO_ROOT
SESSION_DIR = REPO_ROOT / "data" / "data_2026-06-28"
OUT_DIR = REPO_ROOT / "results" / "lidar_ground_truth" / "diagnostics" / "step2_board_roundtrip"

# Exactly cam_lidar_2d_icp.py::main()'s checkerboard parameters.
CHECKERBOARD_WIDTH = 6
CHECKERBOARD_HEIGHT = 4
CHECKERBOARD_SIZE = 0.037
CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def checkerboard_object_points() -> np.ndarray:
    pts = np.zeros((CHECKERBOARD_WIDTH * CHECKERBOARD_HEIGHT, 3), np.float32)
    pts[:, :2] = np.mgrid[0:CHECKERBOARD_WIDTH, 0:CHECKERBOARD_HEIGHT].T.reshape(-1, 2) * CHECKERBOARD_SIZE
    return pts


def rot_cam_to_robot_matrix() -> np.ndarray:
    """Exact replica of gui.py::ImageVisInterface.done_callback's
    rot_cam_to_robot -- copied, not imported (gui.py is off-limits to
    import-and-run since it opens a Tk window on construction).
    """
    rot_rod_zn90 = np.array([[0], [0], [-np.pi / 2]])
    rot_zn90, _ = cv2.Rodrigues(rot_rod_zn90)
    rot_rod_xn90 = np.array([[-np.pi / 2], [0], [0]])
    rot_xn90, _ = cv2.Rodrigues(rot_rod_xn90)
    return rot_zn90 @ rot_xn90


def detect_checkerboard(image_bgr: np.ndarray):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (CHECKERBOARD_WIDTH, CHECKERBOARD_HEIGHT), None)
    if not ret:
        return None, None, None
    corners2 = cv2.cornerSubPix(gray, corners, (3, 3), (-1, -1), CRITERIA)
    return ret, corners, corners2


def board_points_in_robot_frame(rvec, tvec, object_points, rot_cam_to_robot) -> np.ndarray:
    """Replica of gui.py's board_origin / board_x_direction / line_points
    machinery, generalised to ALL object points (not just the wall-line
    samples) and keeping z_R (gui.py drops it -- it only needs the 2D line
    for ICP).
    """
    tf_cam_to_robot = np.eye(4)
    tf_cam_to_robot[0:3, 0:3] = rot_cam_to_robot

    rotation, _ = cv2.Rodrigues(rvec)
    tf_board_to_cam = np.eye(4)
    tf_board_to_cam[0:3, 0:3] = rotation
    tf_board_to_cam[0:3, 3:4] = tvec

    tf_robot_board_to_robot = tf_cam_to_robot @ tf_board_to_cam

    object_points_h = np.hstack(
        [object_points, np.ones((len(object_points), 1), dtype=np.float64)]
    )
    points_robot_h = (tf_robot_board_to_robot @ object_points_h.T).T
    return points_robot_h[:, :3]  # (N, 3): x_R, y_R, z_R


def project_points_robot_frame(points_r: np.ndarray, K: np.ndarray, image_wh) -> np.ndarray:
    """Loop-per-point call into the SHIPPED project_scan (T2 = identity,
    delta_z_m = each point's own z_R) -- exactly Step 1's approach, so this
    is testing the real code path, not a hand-rolled formula.
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
        out[i] = result.u[0], result.v[0]
    return out


def make_overlay(image_bgr, corners2, uv_roundtrip, pose_id, out_path):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(image_rgb)
    c2 = corners2.reshape(-1, 2)
    ax.scatter(c2[:, 0], c2[:, 1], s=60, facecolors="none", edgecolors="lime",
               linewidths=1.5, label="detected corners2 (findChessboardCorners)")
    ax.scatter(uv_roundtrip[:, 0], uv_roundtrip[:, 1], s=20, c="red", marker="x",
               label="round-trip via project_scan(R-frame board geometry)")
    ax.set_title(f"{pose_id}: board-geometry round-trip vs detected corners")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim(0, image_bgr.shape[1])
    ax.set_ylim(image_bgr.shape[0], 0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rot_cam_to_robot = rot_cam_to_robot_matrix()
    object_points = checkerboard_object_points()

    image_dir = SESSION_DIR / "images"
    metadata_dir = SESSION_DIR / "additional_image_data"

    pose_reports = []
    for image_path in sorted(image_dir.glob("*.png")):
        pose_id = image_path.stem.replace("left_", "")
        image_bgr = cv2.imread(str(image_path))
        h, w = image_bgr.shape[:2]

        metadata_path = metadata_dir / f"metadata_{pose_id}.json"
        intrinsics = calibration_io.load_intrinsics_from_metadata(metadata_path)

        ret, corners, corners2 = detect_checkerboard(image_bgr)
        if not ret:
            print(f"{pose_id}: checkerboard NOT detected, skipping")
            continue

        ret_pnp, rvec, tvec = cv2.solvePnP(object_points, corners2, intrinsics.K, np.zeros((1, 5)))

        points_r = board_points_in_robot_frame(rvec, tvec, object_points, rot_cam_to_robot)
        uv_roundtrip = project_points_robot_frame(points_r, intrinsics.K, (w, h))

        corners2_flat = corners2.reshape(-1, 2)
        residual_px = np.linalg.norm(uv_roundtrip - corners2_flat, axis=1)

        make_overlay(image_bgr, corners2, uv_roundtrip, pose_id,
                     OUT_DIR / f"{pose_id}_roundtrip.png")

        report = {
            "pose_id": pose_id,
            "n_corners": int(len(corners2_flat)),
            "residual_px_mean": float(residual_px.mean()),
            "residual_px_max": float(residual_px.max()),
            "residual_px_median": float(np.median(residual_px)),
        }
        pose_reports.append(report)
        print(
            f"{pose_id}: N={report['n_corners']} "
            f"mean_residual={report['residual_px_mean']:.4f}px "
            f"median={report['residual_px_median']:.4f}px "
            f"max={report['residual_px_max']:.4f}px"
        )

    overall_mean = float(np.mean([r["residual_px_mean"] for r in pose_reports]))
    overall_max = float(np.max([r["residual_px_max"] for r in pose_reports]))
    print()
    print(f"Overall across {len(pose_reports)} poses: mean of per-pose means="
          f"{overall_mean:.4f}px, max residual={overall_max:.4f}px")
    verdict = "SUB-PIXEL: projection.py / R<->C frame convention are correct" \
        if overall_max < 1.0 else \
        "NOT sub-pixel: investigate projection.py / frame convention further"
    print(verdict)

    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump({"poses": pose_reports, "overall_mean_px": overall_mean,
                    "overall_max_px": overall_max, "verdict": verdict}, f, indent=2)
    print(f"Saved {OUT_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
