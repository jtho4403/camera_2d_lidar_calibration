"""Bilinear sampling of a predictor's dense depth map at sparse, sub-pixel
ground-truth (u, v) locations. Shared by every predictor backend -- GT pixel
coordinates are floats (the projection equations in
tools/lidar_ground_truth/projection.py are continuous), so nearest-pixel
rounding would throw away real sub-pixel information the ground truth
actually has.
"""
from __future__ import annotations

import numpy as np

import config


def bilinear_sample_depth(depth_map: np.ndarray, u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sample depth_map (H, W, metres, invalid entries non-finite or <= 0) at
    float pixel coordinates (u, v).

    Returns (values, valid): values[i] is the bilinearly interpolated depth
    at (u[i], v[i]) where valid[i] is True, and undefined (not NaN-guaranteed)
    where valid[i] is False -- callers must gate on valid, not on values alone.

    A sample is valid only if (u, v) is inside the image and all 4 bilinear
    neighbours are finite and positive (config.BILINEAR_ANY_NEIGHBOUR_INVALID_REJECTS_SAMPLE):
    interpolating across a real prediction/hole boundary would silently
    blend a genuine depth with a missing one.
    """
    h, w = depth_map.shape
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)

    u0 = np.floor(u).astype(np.int64)
    v0 = np.floor(v).astype(np.int64)
    u1, v1 = u0 + 1, v0 + 1

    in_bounds = (u0 >= 0) & (u1 <= w - 1) & (v0 >= 0) & (v1 <= h - 1)

    u0c, u1c = np.clip(u0, 0, w - 1), np.clip(u1, 0, w - 1)
    v0c, v1c = np.clip(v0, 0, h - 1), np.clip(v1, 0, h - 1)

    d00 = depth_map[v0c, u0c]
    d01 = depth_map[v0c, u1c]
    d10 = depth_map[v1c, u0c]
    d11 = depth_map[v1c, u1c]

    neighbours_valid = np.isfinite(d00) & (d00 > 0) & np.isfinite(d01) & (d01 > 0) \
        & np.isfinite(d10) & (d10 > 0) & np.isfinite(d11) & (d11 > 0)

    wu = u - u0
    wv = v - v0
    interpolated = (
        d00 * (1 - wu) * (1 - wv) + d01 * wu * (1 - wv)
        + d10 * (1 - wu) * wv + d11 * wu * wv
    )

    valid = in_bounds & (neighbours_valid if config.BILINEAR_ANY_NEIGHBOUR_INVALID_REJECTS_SAMPLE else True)
    return interpolated, valid
