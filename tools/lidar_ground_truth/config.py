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
# RPLIDAR S3 PCDs extracted from /scan (data_2026-09-23) contain no exact
# (0, 0) points and no NaN/Inf: no-return bearings are simply absent, so the
# per-scan point count varies (2822-3068 per revolution). scan_io.load_valid_xy
# still checks every convention a ROS LaserScan -> PointCloud path commonly
# produces (exact zero, NaN/Inf, out of datasheet range), so a different
# extraction path cannot silently pass invalid returns through.
INVALID_POINT_TOL_M = 1e-9

# RPLIDAR S3 (model S3M1-R2) datasheet range: 0.05-40 m at 70% reflectivity
# (PLAN.md Sec 0). Returns outside this band are treated as invalid.
RANGE_MIN_M = 0.05
RANGE_MAX_M = 40.0

# --- Projection (projection.py, PLAN.md Sec 6.2) ---
# Guard against dividing by a near-zero/negative camera-frame depth.
Z_MIN_M = 0.05

# --- Declared depth-model operating range (EXPERIMENT_DESIGN_v3.md Sec 6) ---
# Every depth-model engine's max-disparity setting (128 px FULL tier, 64 px
# HALF tier) is built for this range; ground-truth points outside it are
# excluded from the export (export_scene.py, filters.reject_outside_operating_range),
# not merely flagged. Distinct from Z_MIN_M above, which is only a numerical
# divide-by-zero guard inside project_scan and is not the declared range.
OPERATING_RANGE_Z_MIN_M = 0.5
OPERATING_RANGE_Z_MAX_M = 8.0

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

# Calibration artifacts (lidar_to_camera_2d.npy, calibration_result.json)
# and rectified intrinsics (session_manifest.json) are per-session inputs
# passed on the command line; there are deliberately no defaults, so a run
# can never pick up another session's calibration by accident.

# --- Uncertainty (uncertainty.py, PLAN.md Sec 6.4) ---
BOOTSTRAP_RESAMPLES = 200
BOOTSTRAP_SEED = 0
MONTE_CARLO_SAMPLES = 200
MONTE_CARLO_SEED = 0

# --- Board-plane validation acceptance thresholds (PLAN.md Sec 5.6) ---
# Stated before the tests are run. The LiDAR noise floor is the per-bearing
# range spread of the staged burst (staging_manifest.json), so the bias
# threshold is read from the session rather than fixed here.
ACCEPT_BOOTSTRAP_YAW_STD_DEG = 0.3
# A trend (slope of the residual vs distance or vs board yaw) fails when it is
# significant at this many standard errors.
ACCEPT_TREND_SIGNIFICANCE_SIGMA = 2.0
