"""Model-independent quality filters for projected LiDAR ground truth
(PLAN.md Sec 6.3).

Filter 1 (invalid returns) is applied upstream by scan_io.load_valid_xy.
Filter 2 (behind camera / out of image bounds) is applied upstream by
projection.project_scan's `keep` mask. This module adds:

  3. Occlusion / parallax rejection -- PLAN.md Sec 6.3 item 3.
  4. High incidence angle -- PLAN.md Sec 6.3 item 4. Deferred: it needs the
     RPLIDAR S3 accuracy-vs-incidence-angle characterisation from PLAN.md
     Phase 5 (tools/lidar_characterisation), which does not exist yet.
     reject_high_incidence_angle raises NotImplementedError rather than
     silently no-op filtering, so a caller cannot mistake "not yet wired up"
     for "nothing to reject".
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

import config


@dataclass
class FilterReport:
    keep: np.ndarray                       # combined bool mask, indexed like the input scan
    rejected_counts: dict[str, int] = field(default_factory=dict)


def reject_range_discontinuity(
    xy_lidar_bearing_ordered: np.ndarray,
    threshold_m: float = config.RANGE_DISCONTINUITY_THRESHOLD_M,
    neighbour_count: int = config.RANGE_DISCONTINUITY_NEIGHBOUR_COUNT,
) -> np.ndarray:
    """Reject returns within `neighbour_count` bearing-order samples of a
    range discontinuity |r[j+1] - r[j]| > threshold_m.

    xy_lidar_bearing_ordered must already be in the scan's native bearing
    order (as read off disk) -- this filter has no meaning after points are
    reordered or a subset is taken out of order.

    Background returns near a depth discontinuity can straddle a foreground
    object's silhouette once projected, corrupting ground truth exactly at
    object boundaries -- where depth models are worst -- so rejecting them
    here matters more than the point count suggests (PLAN.md Sec 6.3).
    """
    xy = np.asarray(xy_lidar_bearing_ordered, dtype=np.float64)
    n = len(xy)
    range_m = np.hypot(xy[:, 0], xy[:, 1])

    reject = np.zeros(n, dtype=bool)
    if n > 1:
        diffs = np.abs(np.diff(range_m))
        discontinuity_at = np.flatnonzero(diffs > threshold_m)  # jump between sample idx, idx+1
        for idx in discontinuity_at:
            lo = max(0, idx - neighbour_count + 1)
            hi = min(n, idx + 1 + neighbour_count)
            reject[lo:hi] = True

    return ~reject


def reject_depth_order_inconsistent_with_u_order(
    u: np.ndarray,
    depth_m: np.ndarray,
    jump_threshold_m: float = config.RANGE_DISCONTINUITY_THRESHOLD_M,
) -> np.ndarray:
    """Reject points whose projected-column ordering is locally inconsistent
    with their depth ordering, at a magnitude consistent with a real
    occlusion boundary (not smooth wall curvature).

    A background point that projects -- due to the camera/LiDAR lateral
    offset -- into a column position already occupied by a nearer
    foreground point's silhouette is exactly the parallax failure mode
    PLAN.md Sec 6.3 warns about. Detected as a local sign flip in
    depth-vs-u, sorted by u, gated on both adjacent depth steps exceeding
    jump_threshold_m so ordinary smooth depth variation across a flat wall
    (mm-scale, driven by viewing-angle curvature) is never flagged.
    """
    u = np.asarray(u, dtype=np.float64)
    depth_m = np.asarray(depth_m, dtype=np.float64)
    n = len(u)
    keep = np.ones(n, dtype=bool)
    if n < 3:
        return keep

    order = np.argsort(u)
    depth_sorted = depth_m[order]
    d1 = np.diff(depth_sorted)

    keep_sorted = np.ones(n, dtype=bool)
    if n > 2:
        sign_flip = np.sign(d1[:-1]) * np.sign(d1[1:]) < 0
        big_enough = (np.abs(d1[:-1]) > jump_threshold_m) & (np.abs(d1[1:]) > jump_threshold_m)
        flips = np.flatnonzero(sign_flip & big_enough)
        keep_sorted[flips + 1] = False  # the middle point of the flip is the outlier

    keep[order] = keep_sorted
    return keep


def reject_outside_operating_range(
    depth_m: np.ndarray,
    z_min_m: float = config.OPERATING_RANGE_Z_MIN_M,
    z_max_m: float = config.OPERATING_RANGE_Z_MAX_M,
) -> np.ndarray:
    """Reject points outside the declared depth-model operating range
    (EXPERIMENT_DESIGN_v3.md Sec 6: Z_min=0.5 m, Z_max=8 m).

    Not part of apply_filters below: that function's occlusion/parallax
    filters are shared with overlay_diagnostic.py's Phase 1 sanity checks,
    which deliberately inspect the whole scan including out-of-range
    background to catch sign/frame-convention errors. Ground-truth export
    (export_scene.py) is the one place this scope restriction belongs, so
    callers apply it explicitly on top of apply_filters's result.
    """
    depth_m = np.asarray(depth_m, dtype=np.float64)
    return (depth_m >= z_min_m) & (depth_m <= z_max_m)


def reject_high_incidence_angle(*_args, **_kwargs):
    raise NotImplementedError(
        "High-incidence-angle rejection needs the RPLIDAR S3 accuracy-vs-"
        "incidence-angle characterisation from PLAN.md Phase 5 "
        "(tools/lidar_characterisation), which has not been produced yet. "
        "Do not call this until that data exists."
    )


def apply_filters(
    xy_lidar_bearing_ordered: np.ndarray,
    project_keep: np.ndarray,
    u: np.ndarray,
    depth_m: np.ndarray,
) -> FilterReport:
    """Compose the model-independent filters and report counts by reason.

    xy_lidar_bearing_ordered, project_keep, u, depth_m must all be indexed
    identically (same point, same order) -- project_keep, u and depth_m
    must come from projection.project_scan called on this exact
    xy_lidar_bearing_ordered array.

    Filter 4 (incidence angle) is not applied here -- see
    reject_high_incidence_angle's docstring.
    """
    n = len(xy_lidar_bearing_ordered)
    project_keep = np.asarray(project_keep, dtype=bool)

    discontinuity_keep = reject_range_discontinuity(xy_lidar_bearing_ordered)
    after_discontinuity = project_keep & discontinuity_keep

    order_keep = np.ones(n, dtype=bool)
    idx = np.flatnonzero(after_discontinuity)
    if len(idx) >= 3:
        order_keep[idx] = reject_depth_order_inconsistent_with_u_order(
            u[idx], depth_m[idx]
        )

    combined = after_discontinuity & order_keep

    rejected_counts = {
        "total": n,
        "behind_camera_or_out_of_bounds": int(np.count_nonzero(~project_keep)),
        "range_discontinuity": int(np.count_nonzero(project_keep & ~discontinuity_keep)),
        "depth_order_inconsistent": int(
            np.count_nonzero(after_discontinuity & ~order_keep)
        ),
        "surviving": int(np.count_nonzero(combined)),
    }

    return FilterReport(keep=combined, rejected_counts=rejected_counts)
