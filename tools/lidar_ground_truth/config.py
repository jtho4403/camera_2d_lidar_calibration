"""Explicit, reproducible config constants for the LiDAR ground-truth pipeline.

Mirrors tools/lidar_characterisation/config.py: all thresholds live here (not
buried in analysis code) so they are easy to audit and easy to dump into
output reports.
"""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data"
RESULTS_ROOT = REPO_ROOT / "results" / "lidar_ground_truth"
CONFIG_ROOT = REPO_ROOT / "config"

# --- Invalid-point filtering (scan_io.py) ---
# Retargeted from tools/lidar_characterisation/pcd_io.py (LDROBOT STL-19P
# convention: exact (0, 0, 0) marks a dropped return) to the RPLIDAR S3.
# The S3's own no-return encoding has NOT been confirmed against a real
# captured PCD -- no S3 data exists in this repo yet at Phase 0/1 time (see
# README.md "Open questions"). scan_io.load_valid_xy therefore checks every
# convention a ROS LaserScan -> PointCloud path commonly produces (exact
# zero, NaN/Inf, out of datasheet range), so it is safe against whichever
# one the real S3 driver turns out to use. Reconfirm empirically in Phase 3
# against a real captured S3 PCD before trusting silently-passing filtering.
INVALID_POINT_TOL_M = 1e-9

# RPLIDAR S3 (model S3M1-R2) datasheet range: 0.05-40 m at 70% reflectivity
# (PLAN.md Sec 0). Returns outside this band are treated as invalid.
RANGE_MIN_M = 0.05
RANGE_MAX_M = 40.0

# --- Projection (projection.py, PLAN.md Sec 6.2) ---
# Guard against dividing by a near-zero/negative camera-frame depth.
Z_MIN_M = 0.05

# --- Occlusion / parallax filter (filters.py, PLAN.md Sec 6.3 item 3) ---
# Reject LiDAR returns within this many bearing-order samples of a range
# discontinuity |r[j+1] - r[j]| > threshold, and reject returns whose
# projected-u depth ordering flips by more than this same magnitude across
# a single neighbouring pair (a real occlusion boundary, not wall curvature
# noise).
RANGE_DISCONTINUITY_THRESHOLD_M = 0.15
RANGE_DISCONTINUITY_NEIGHBOUR_COUNT = 2

# --- Rig parameters ---
DEFAULT_RIG_PATH = CONFIG_ROOT / "rig_template.json"

# --- Existing calibration artifacts (repo root, PLAN.md Sec 4.1) ---
DEFAULT_TRANSFORM_PATH = REPO_ROOT / "lidar_to_camera_2d.npy"
DEFAULT_CALIBRATION_RESULT_PATH = REPO_ROOT / "calibration_result.json"

# Image resolution asserted by cam_lidar_2d_icp.py::main() for every
# calibration image; used as a fallback image size when no per-pose
# metadata_pose_NN.json is available (e.g. examples/).
FALLBACK_IMAGE_WH = (1280, 720)
