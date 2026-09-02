"""Explicit, reproducible config constants for LiDAR characterisation.

All thresholds live here (not buried in analysis code) so they are easy to
audit and are also dumped into the output report.
"""

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
