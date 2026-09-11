"""Shared helper: reconstruct per-pose board-plane correspondences from
on-disk artifacts + the EXISTING solved T2, for Tests A and B (Phase 1.6).

## Why a reconstruction, not the real correspondence dump

`cam_lidar_2d_icp.py` now has the additive `calibration_correspondences.npz`
dump authorised and added per PLAN.md Sec 4.2 (see the diff in
`cam_lidar_2d_icp.py::main()` around the `pose_rvecs`/`pose_tvecs` lists and
the `np.savez("calibration_correspondences.npz", ...)` call -- purely
additive, no existing behaviour changed). But producing an actual `.npz`
requires a live interactive run through 7 poses x 2 Tk GUIs
(`ImageVisInterface`, `SelectPointsInterface`), and this environment has no
display and no human to drive them. Faking that interaction was explicitly
ruled out.

So this module reconstructs an equivalent correspondence set algorithmically
from data already on disk, using nothing but:

  - the already-curated, already-on-disk raw scans (`data/.../lasers/*.pcd`)
  - the EXISTING, unmodified, already-solved `T2` (`lidar_to_camera_2d.npy`)
  - the EXISTING, unmodified `calibration_result.json`'s own per-pose TLS
    camera-line fit (`fitted_line_diagnostics`) -- literally the reference
    line the real ICP fit against
  - a fresh, fully deterministic (no GUI) `cv2.findChessboardCorners` +
    `solvePnP` re-detection per pose (reused from `step2_board_roundtrip.py`)

A LiDAR point is counted as "struck the board" if, after transforming by the
EXISTING T2, it falls within `LINE_DIST_THRESHOLD_M` of the pose's own TLS
camera line AND within `LINE_ALONG_EXTENT_M` of the line's centroid along
the line's own direction (the second bound excludes a coincidentally
similarly-oriented surface elsewhere in the room that happens to sit at a
similar perpendicular offset). `LINE_DIST_THRESHOLD_M` matches the real
staged ICP's own final-stage (tightest) distance threshold, so this should
recover very nearly the point set the real fit actually used.

**This is a reconstruction, not the ground truth human selection.** Every
downstream result built on it (Tests A and B) should be read with that
caveat, and is reported as such in report.md.
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # tools/lidar_ground_truth/
sys.path.insert(0, str(Path(__file__).resolve().parent))       # diagnostics/ (sibling imports)

import cv2
import numpy as np

import calibration_io
import config
import scan_io
from step2_board_roundtrip import (
    CHECKERBOARD_SIZE,
    checkerboard_object_points,
    detect_checkerboard,
)

SESSION_DIR = config.REPO_ROOT / "data" / "data_2026-06-28"
CALIBRATION_RESULT_PATH = config.DEFAULT_CALIBRATION_RESULT_PATH

# Matches the real staged ICP's final (tightest) stage threshold
# (cam_lidar_2d_icp.py::main()'s icp_stages, last entry: 0.10 m).
LINE_DIST_THRESHOLD_M = 0.10
# Excludes a coincidentally similarly-oriented surface elsewhere in the
# room; generous relative to the ~0.4 m synthetic camera-line span
# (gui.py's line_start=-0.1/line_end=0.3) so it does not itself bias which
# real board-return points are kept.
LINE_ALONG_EXTENT_M = 0.5


@dataclass
class PoseCorrespondences:
    pose_id: str
    xy_lidar: np.ndarray          # (N, 2) LiDAR-frame points that struck the board (frame L)
    camera_line_point: np.ndarray  # (2,) frame R
    camera_line_direction: np.ndarray  # (2,) frame R, unit
    camera_line_normal: np.ndarray     # (2,) frame R, unit
    K: np.ndarray
    image_wh: tuple[int, int]
    board_normal_C: np.ndarray    # (3,) OpenCV camera frame, canonicalised to point toward the camera
    board_point_C: np.ndarray     # (3,) tvec, OpenCV camera frame
    obliquity_deg: float          # pose's camera-line angle minus 90 deg (0 = frontoparallel)


def pose_id_from_filename(path: Path) -> str:
    match = re.search(r"pose_(\d+)", path.stem)
    return f"pose_{int(match.group(1)):02d}"


def load_camera_lines() -> dict:
    with open(CALIBRATION_RESULT_PATH) as f:
        result = json.load(f)
    out = {}
    for p in result["residual_validation"]["per_pose"]:
        diag = p["fitted_line_diagnostics"]
        out[p["pose"]] = {
            "point": np.array(diag["camera_line_point"], dtype=np.float64),
            "direction": np.array(diag["camera_line_direction"], dtype=np.float64),
            "normal": np.array(diag["camera_line_normal"], dtype=np.float64),
        }
    return out


def solve_board_plane(image_bgr: np.ndarray, K: np.ndarray):
    """solvePnP -> (n, p0) in OpenCV camera frame C, n canonicalised to
    point FROM the board TOWARD the camera origin (so PLAN.md Sec 5.2's
    "offset along -n" moves away from the camera, onto the wall behind the
    board, as intended).
    """
    ret, corners, corners2 = detect_checkerboard(image_bgr)
    if not ret:
        return None
    object_points = checkerboard_object_points()
    ret_pnp, rvec, tvec = cv2.solvePnP(object_points, corners2, K, np.zeros((1, 5)))
    R_board, _ = cv2.Rodrigues(rvec)
    n = R_board[:, 2]
    p0 = tvec.reshape(3)
    if np.dot(n, -p0) < 0:
        n = -n
    return n, p0


def build_all_pose_correspondences() -> list[PoseCorrespondences]:
    T2 = calibration_io.load_transform()
    R2, t2 = T2[:2, :2], T2[:2, 2]
    camera_lines = load_camera_lines()

    image_dir = SESSION_DIR / "images"
    laser_dir = SESSION_DIR / "lasers"
    metadata_dir = SESSION_DIR / "additional_image_data"
    pairs = scan_io.load_images_and_scans_from_folders(image_dir, laser_dir)

    out = []
    for image_path, laser_path in pairs:
        pose_id = pose_id_from_filename(image_path)
        image_bgr = cv2.imread(str(image_path))
        h, w = image_bgr.shape[:2]

        intrinsics = calibration_io.load_intrinsics_from_metadata(
            metadata_dir / f"metadata_{pose_id}.json"
        )

        xy_valid, _, _ = scan_io.load_valid_xy(laser_path)
        xy_r = xy_valid @ R2.T + t2

        line = camera_lines[pose_id]
        perp_dist = np.abs((xy_r - line["point"]) @ line["normal"])
        along = (xy_r - line["point"]) @ line["direction"]
        mask = (perp_dist < LINE_DIST_THRESHOLD_M) & (np.abs(along) < LINE_ALONG_EXTENT_M)

        board = solve_board_plane(image_bgr, intrinsics.K)
        if board is None:
            print(f"{pose_id}: checkerboard not detected, skipping")
            continue
        n, p0 = board

        angle_deg = float(np.degrees(np.arctan2(line["direction"][1], line["direction"][0])))
        obliquity_deg = angle_deg - 90.0

        out.append(PoseCorrespondences(
            pose_id=pose_id,
            xy_lidar=xy_valid[mask],
            camera_line_point=line["point"],
            camera_line_direction=line["direction"],
            camera_line_normal=line["normal"],
            K=intrinsics.K,
            image_wh=(w, h),
            board_normal_C=n,
            board_point_C=p0,
            obliquity_deg=obliquity_deg,
        ))
    return out


if __name__ == "__main__":
    for pc in build_all_pose_correspondences():
        print(f"{pc.pose_id}: N={len(pc.xy_lidar)} obliquity={pc.obliquity_deg:+.2f}deg "
              f"board_normal_C={pc.board_normal_C} board_point_C={pc.board_point_C}")
