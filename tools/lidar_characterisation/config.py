"""Explicit, reproducible config constants for LiDAR characterisation.

All thresholds live here (not buried in analysis code) so they are easy to
audit and are also dumped into the output report.

HISTORICAL: every value below describes the OLD rig's LDROBOT STL-19P and its
sessions, which now live in data/_archive/. Inputs are read from there and
outputs are written to results/_archive/ so this study can never be mistaken
for, or mixed into, RPLIDAR S3 results. Retargeting to the S3 (PLAN.md
Phase 5) needs new S3 accuracy captures and new datasheet bands.
"""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data" / "_archive"
RESULTS_ROOT = REPO_ROOT / "results" / "_archive" / "lidar_characterisation"

# --- Invalid-point filtering ---
# The STL-19P raw PCDs encode a dropped/no-return reading as an exact
# (0, 0, 0) point. Anything within this tolerance of the origin is treated
# as invalid and discarded before any fitting.
INVALID_POINT_TOL_M = 1e-9

# --- RANSAC dominant-line (wall) extraction ---
RANSAC_DIST_THRESHOLD_M = 0.03      # inlier perpendicular-distance threshold
RANSAC_MIN_INLIERS = 30             # minimum inlier count to accept a line as "a wall"
RANSAC_MIN_SPATIAL_EXTENT_M = 0.25  # minimum along-line inlier span (m)
RANSAC_MIN_ANGULAR_EXTENT_DEG = 5.0 # minimum angular span subtended by inliers (deg)
RANSAC_N_ITERS = 3000
RANSAC_REFINE_ROUNDS = 3            # re-fit-and-reselect-inliers rounds after initial RANSAC vote
RANSAC_SEED = 42

# --- Planarity (Step 3) ---
# Polynomial degrees fit to the residual-vs-position trend, compared against
# a flat (degree-0 / zero-mean) baseline to detect systematic structure.
PLANARITY_POLY_DEGREES = (1, 2, 3)

# --- Datasheet comparison (LDROBOT STL-19P, 80% reflectivity white target) ---
# (min_range_m, max_range_m, precision_std_mm)
DATASHEET_PRECISION_BANDS_MM = [
    (0.5, 2.0, 4.0),
    (2.0, 12.0, 15.0),
]
# Accuracy (bias) bands are recorded for reference only. This study does NOT
# assess accuracy (no independently measured true distances) — see report.
DATASHEET_ACCURACY_BANDS_MM = [
    (0.5, 2.0, 20.0),
    (2.0, 12.0, 30.0),
]

# --- Datasets ---
DATA_SESSIONS = ["data_2026-06-28", "data_2026-07-10"]
POSES = [f"pose_{i:02d}" for i in range(1, 8)]

# --- Accuracy capture (2026-09-02): flat corridor end wall at a rangefinder-
# measured true distance per pose. Raw scans live directly under
# data/<session>/extracted_pcd/pose_NN/ (no "raw_lasers/" subdir, unlike the
# earlier precision-only sessions above).
ACCURACY_SESSION = "data_2026-09-02_accuracy"
ACCURACY_TRUE_DISTANCES_M = {
    "pose_01": 0.501,
    "pose_02": 1.000,
    "pose_03": 1.501,
    "pose_04": 2.000,
    "pose_05": 2.501,
    "pose_06": 3.001,
    "pose_07": 3.501,
    "pose_08": 4.001,
    "pose_09": 4.500,
    "pose_10": 5.000,
    "pose_11": 5.501,
}

# Maximum number of candidate wall segments searched for per scan (iterative
# RANSAC with inlier removal) before giving up on finding more.
ACCURACY_MAX_CANDIDATES = 8

# RANSAC minimum-inlier override for the accuracy capture only (does NOT
# change RANSAC_MIN_INLIERS used by the earlier <1m precision-only study).
# At long range the target wall subtends a much narrower angle from the
# sensor than in the close-range study, so fewer fixed-angular-resolution
# beams land on it; verified against raw scans for poses at 4.0-5.5m true
# distance, where the target wall is clearly present (a clean, smoothly
# r(angle)-varying planar cluster, spatial extent >1.2m, angular extent
# 13-19deg) but only carries ~21-28 inlier points under the shared 0.03m
# perpendicular-distance threshold -- below RANSAC_MIN_INLIERS=30. This
# lower floor was chosen with margin below that observed range (21-28).
ACCURACY_RANSAC_MIN_INLIERS = 15

# Target-wall selection tolerance gate: a candidate is accepted as "the
# target wall" only if its perpendicular distance to the sensor origin is
# within max(ACCURACY_TOLERANCE_MIN_M, ACCURACY_TOLERANCE_FRAC * true_distance)
# of that pose's known true distance.
ACCURACY_TOLERANCE_MIN_M = 0.05
ACCURACY_TOLERANCE_FRAC = 0.05

# Measurement-uncertainty caveat band for bias interpretation: rangefinder
# spec (+/-2mm) plus placement/wall-flatness slop. Biases within this band
# are not distinguishable from measurement noise, not necessarily "accurate".
ACCURACY_MEASUREMENT_UNCERTAINTY_MM = 10.0

# A scan's target_bearing_deg deviating from its pose's median bearing by
# more than this is flagged as an actual wrong-surface selection (distinct
# from "ambiguous_selection", which only flags that >1 candidate qualified
# for the tolerance gate -- most such scans still pick the correct one).
ACCURACY_BEARING_OUTLIER_THRESHOLD_DEG = 10.0
