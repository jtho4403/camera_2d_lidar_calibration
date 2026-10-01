"""Explicit, reproducible config constants for the depth-model evaluation harness.

Mirrors tools/lidar_ground_truth/config.py and tools/lidar_characterisation/config.py:
all thresholds and defaults live here, not buried in analysis code.
"""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data"
RESULTS_ROOT = REPO_ROOT / "results" / "depth_eval"
GROUND_TRUTH_ROOT = REPO_ROOT / "results" / "lidar_ground_truth"

# --- Depth-bucket edges (metres), EXPERIMENT_DESIGN_v3.md Sec 6 declared range ---
# Aggregate-only metrics hide exactly the kind of distance-dependent structure this
# project's own ground-truth validation found in itself (REPORT.md Sec 5) -- bucketed
# reporting is mandatory, not optional, for the same reason: a bias or an accuracy
# falloff concentrated in one range band must not be averaged away by abundant
# near-range points. Buckets are half-open [lo, hi) except the last, which is closed.
DEPTH_BUCKETS_M = [
    (0.5, 2.0, "near"),
    (2.0, 4.0, "mid"),
    (4.0, 8.0, "far"),
]

# --- Metric guards ---
# abs_rel/sq_rel/log_rms divide by (or take the log of) the ground-truth depth;
# GT is already filtered to >= 0.5 m (tools/lidar_ground_truth), so this is a
# numerical-safety floor, not an expected code path.
GT_DEPTH_EPS_M = 1e-3

# Threshold-accuracy deltas (a1/a2/a3), DEPTH_METRICS.md Standard Depth Metrics.
ACCURACY_DELTAS = [1.25, 1.25 ** 2, 1.25 ** 3]

# --- Bilinear sampling of a predictor's dense depth map at sparse GT (u, v) ---
# A sample is marked invalid (excluded from accuracy metrics, counted against
# coverage) if ANY of its 4 bilinear neighbours is invalid -- interpolating across
# a real prediction/no-prediction boundary would silently blend a valid depth with
# a hole and misrepresent the model's own coverage.
BILINEAR_ANY_NEIGHBOUR_INVALID_REJECTS_SAMPLE = True

# --- ZED SDK backend defaults (predictors/zed_sdk.py) ---
# MASTER_PLAN.md 4.2 "Lock SDK depth settings" is an explicitly open decision; these
# are defaults, not a resolved answer. Rationale: SDK out-of-the-box confidence/
# texture thresholds (100 = no filtering) and fill_mode=False, so the evaluation
# measures the mode's genuine, un-interpolated prediction quality -- coverage
# (fraction of GT points with a real prediction) is tracked and reported as its own
# metric rather than papered over by hole-filling. depth_minimum/maximum_distance
# match this project's declared operating range (EXPERIMENT_DESIGN_v3.md Sec 6) so
# the SDK's own search range matches what's being evaluated.
ZED_SDK_CONFIDENCE_THRESHOLD = 100
ZED_SDK_TEXTURE_CONFIDENCE_THRESHOLD = 100
ZED_SDK_ENABLE_FILL_MODE = False
ZED_SDK_DEPTH_MINIMUM_DISTANCE_M = 0.5
ZED_SDK_DEPTH_MAXIMUM_DISTANCE_M = 8.0

# Which frame, by index, to grab from a capture's SVO2 burst for evaluation.
# Scenes A/B/C are captured with the rig static throughout a burst (EXPERIMENT_DESIGN_v3.md
# Sec 3), so any in-burst frame is equally valid; deliberately NOT the burst mean used for
# Scene A's calibration images (tools/extract_burst_mean_images.py) -- averaging ~150 frames
# is not representative of a single real-time inference call, which is what is being
# evaluated here. A mid-burst index avoids both the first frame (closest to the settle/
# record boundary) and SVO2 end-of-file edge effects.
EVAL_FRAME_INDEX = 75

K_MATCH_TOLERANCE_PX = 0.5
