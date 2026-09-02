"""Generic 2D RANSAC line fitting used to find the dominant wall in a scan.

Deliberately board-agnostic: it just finds the straight segment with the most
inliers subject to minimum-count / minimum-extent thresholds. No assumptions
about a checkerboard, expected distance, or expected orientation.
"""
from dataclasses import dataclass

import numpy as np

import config


@dataclass
class LineFit:
    point: np.ndarray        # a point on the line (centroid of inliers)
    direction: np.ndarray    # unit vector along the line
    normal: np.ndarray       # unit normal to the line
    inlier_mask: np.ndarray  # bool mask into the input points array
    distance_to_origin: float  # perpendicular distance from sensor origin (0,0) to the line
    residuals: np.ndarray    # signed perpendicular residuals of inliers about the line
    angular_extent_deg: float
    spatial_extent_m: float
    along_line_positions: np.ndarray  # inlier projections onto `direction`, centred
    angles_deg: np.ndarray            # atan2(y, x) of inliers, in degrees


def _fit_tls_line(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Total-least-squares line fit via PCA. Returns (point, direction, normal)."""
    centroid = points.mean(axis=0)
    centered = points - centroid
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    direction = vt[0]
    direction = direction / np.linalg.norm(direction)
    normal = np.array([-direction[1], direction[0]])
    return centroid, direction, normal


def fit_dominant_wall(points_xy: np.ndarray, rng: np.random.Generator | None = None) -> LineFit | None:
    """Find the single dominant straight-line ("wall") segment in points_xy.

    Returns None if no line meets the configured minimum-inlier /
    minimum-extent thresholds.
    """
    n = len(points_xy)
    if n < config.RANSAC_MIN_INLIERS:
        return None
    if rng is None:
        rng = np.random.default_rng(config.RANSAC_SEED)

    # Vectorized RANSAC: sample all candidate two-point lines at once, then
    # score every candidate against every point via broadcasting. A plain
    # per-iteration Python loop is far too slow at this data volume (many
    # thousands of scans x thousands of iterations).
    n_iters = config.RANSAC_N_ITERS
    idx1 = rng.integers(0, n, size=n_iters)
    idx2 = rng.integers(0, n, size=n_iters)
    same = idx1 == idx2
    while np.any(same):
        idx2[same] = rng.integers(0, n, size=int(same.sum()))
        same = idx1 == idx2

    p1 = points_xy[idx1]                       # (n_iters, 2)
    p2 = points_xy[idx2]                       # (n_iters, 2)
    seg = p2 - p1
    norm = np.linalg.norm(seg, axis=1)
    valid = norm > 1e-9
    norm_safe = np.where(valid, norm, 1.0)
    direction = seg / norm_safe[:, None]
    normal = np.stack([-direction[:, 1], direction[:, 0]], axis=1)  # (n_iters, 2)

    diff = points_xy[None, :, :] - p1[:, None, :]           # (n_iters, n, 2)
    dists = np.abs(np.einsum("ijk,ik->ij", diff, normal))   # (n_iters, n)
    inlier_masks = dists < config.RANSAC_DIST_THRESHOLD_M
    counts = inlier_masks.sum(axis=1)
    counts = np.where(valid, counts, -1)

    best_iter = int(np.argmax(counts))
    best_count = int(counts[best_iter])
    best_mask = inlier_masks[best_iter]

    if best_count < config.RANSAC_MIN_INLIERS:
        return None

    # Refine: alternate TLS refit on current inliers / inlier reselection.
    mask = best_mask
    point = direction_ = normal_ = None
    for _ in range(config.RANSAC_REFINE_ROUNDS):
        inlier_pts = points_xy[mask]
        if len(inlier_pts) < 2:
            return None
        point, direction_, normal_ = _fit_tls_line(inlier_pts)
        dists = np.abs((points_xy - point) @ normal_)
        mask = dists < config.RANSAC_DIST_THRESHOLD_M

    n_inliers = int(mask.sum())
    if n_inliers < config.RANSAC_MIN_INLIERS:
        return None

    inlier_pts = points_xy[mask]
    residuals = (inlier_pts - point) @ normal_
    along = (inlier_pts - point) @ direction_
    spatial_extent = float(along.max() - along.min())
    raw_angles_deg = np.degrees(np.arctan2(inlier_pts[:, 1], inlier_pts[:, 0]))
    # angular extent robust to wraparound at +-180 deg
    sorted_ang = np.sort(raw_angles_deg)
    gaps = np.diff(sorted_ang, append=sorted_ang[0] + 360.0)
    angular_extent = float(360.0 - gaps.max())
    # Unwrap angles around their circular mean so a wall straddling the
    # +-180 deg discontinuity doesn't get split into two far-apart clusters
    # by a naive "angle - mean(angle)" downstream (a wall subtends < 180 deg
    # as seen from any interior point, so this unwrap is always well-defined).
    circ_mean_deg = float(np.degrees(np.arctan2(
        np.mean(np.sin(np.radians(raw_angles_deg))),
        np.mean(np.cos(np.radians(raw_angles_deg))),
    )))
    angles_deg = circ_mean_deg + (((raw_angles_deg - circ_mean_deg + 180.0) % 360.0) - 180.0)

    if spatial_extent < config.RANSAC_MIN_SPATIAL_EXTENT_M:
        return None
    if angular_extent < config.RANSAC_MIN_ANGULAR_EXTENT_DEG:
        return None

    distance_to_origin = float(abs(point @ normal_))

    return LineFit(
        point=point,
        direction=direction_,
        normal=normal_,
        inlier_mask=mask,
        distance_to_origin=distance_to_origin,
        residuals=residuals,
        angular_extent_deg=angular_extent,
        spatial_extent_m=spatial_extent,
        along_line_positions=along - along.mean(),
        angles_deg=angles_deg,
    )
