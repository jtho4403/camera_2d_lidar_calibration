"""PCD reading and invalid-return filtering for the LiDAR ground-truth pipeline.

Reuses the reader / invalid-return-filter pattern from
tools/lidar_characterisation/pcd_io.py, for the RPLIDAR S3 -- see config.py's
INVALID_POINT_TOL_M / RANGE_MIN_M / RANGE_MAX_M comments. The extracted S3
PCDs simply omit no-return bearings; this filter still checks every
convention a common ROS conversion path could produce.

Every point handled here is in frame L (PLAN.md Sec 1.1): right-handed,
x forward, y left, z up, origin at the scanner rotation centre, z == 0 for
every return.
"""
from __future__ import annotations

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


def load_valid_xy(path: str | Path) -> tuple[np.ndarray, int, int]:
    """Load a scan, drop the z column, and filter out invalid returns.

    A return is invalid if it is non-finite (NaN/Inf), or its range
    hypot(x, y) falls outside [config.RANGE_MIN_M, config.RANGE_MAX_M] -- the
    RPLIDAR S3 datasheet range band. Since RANGE_MIN_M (0.05 m) is well above
    config.INVALID_POINT_TOL_M, this range check also subsumes an exact
    (0, 0) dropped-return convention, should an extraction path produce one.

    Point order is preserved (a prefix/subset of the on-disk order), which
    matters because filters.py's occlusion/parallax rejection needs the
    scan's native bearing order.

    Returns (valid_xy [frame L, metres], n_total, n_valid).
    """
    pts = read_ascii_pcd_xyz(Path(path))
    xy = pts[:, :2]
    n_total = len(xy)

    finite = np.isfinite(xy).all(axis=1)
    range_m = np.where(finite, np.hypot(xy[:, 0], xy[:, 1]), np.inf)
    valid = finite & (range_m >= config.RANGE_MIN_M) & (range_m <= config.RANGE_MAX_M)

    valid_xy = xy[valid]
    return valid_xy, n_total, int(valid_xy.shape[0])


def load_images_and_scans_from_folders(
    image_dir: str | Path, laser_dir: str | Path,
) -> list[tuple[Path, Path]]:
    """Pair image and scan files by sorted filename order.

    Mirrors cam_lidar_2d_icp.py's load_images_from_folder /
    load_clouds_from_folder pairing convention (CLAUDE.md "Architecture"):
    folder contents must already be curated 1:1 -- there is no timestamp or
    ID matching, only sorted-order pairing within each folder.
    """
    image_paths = sorted(p for p in Path(image_dir).iterdir() if p.is_file())
    laser_paths = sorted(p for p in Path(laser_dir).iterdir() if p.is_file())
    if len(image_paths) != len(laser_paths):
        raise ValueError(
            f"Image/laser count mismatch: {len(image_paths)} images in "
            f"{image_dir}, {len(laser_paths)} lasers in {laser_dir}."
        )
    return list(zip(image_paths, laser_paths))
