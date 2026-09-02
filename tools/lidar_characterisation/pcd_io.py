"""Minimal ASCII PCD reader for 2D LiDAR scans (x y z fields, z == 0).

Self-contained (no dependency on other tools/ scripts) per task constraints.
"""
from pathlib import Path

import numpy as np

import config


def read_ascii_pcd_xyz(path: Path) -> np.ndarray:
    """Read an ASCII PCD's x,y,z data block into an (N, 3) float array."""
    with open(path, "r") as f:
        lines = f.readlines()

    data_start = None
    for i, line in enumerate(lines):
        if line.strip().lower() == "data ascii":
            data_start = i + 1
            break
    if data_start is None:
        raise ValueError(f"No 'DATA ascii' header found in {path}")

    pts = []
    for line in lines[data_start:]:
        parts = line.split()
        if len(parts) >= 3:
            pts.append((float(parts[0]), float(parts[1]), float(parts[2])))
    return np.asarray(pts, dtype=float)


def load_valid_xy(path: Path) -> tuple[np.ndarray, int, int]:
    """Load a scan, drop the z column, and filter out (0,0) invalid returns.

    Returns (valid_xy, n_total, n_valid).
    """
    pts = read_ascii_pcd_xyz(path)
    xy = pts[:, :2]
    n_total = len(xy)
    invalid = (np.abs(xy[:, 0]) < config.INVALID_POINT_TOL_M) & (
        np.abs(xy[:, 1]) < config.INVALID_POINT_TOL_M
    )
    valid_xy = xy[~invalid]
    return valid_xy, n_total, len(valid_xy)
