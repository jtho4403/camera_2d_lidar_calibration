"""Stage a capture session for cam_lidar_2d_icp.py.

For every capture <ID> in <session>/captures/ this writes, pairing by the
shared capture ID:

  <session>/images/<ID>_left.png  -- copy of the rectified left snapshot
  <session>/lasers/<ID>.pcd       -- per-bearing median of the LiDAR scans
                                     recorded inside that capture's camera
                                     recording window (PLAN.md Sec 6.7)
  <session>/staging_manifest.json -- provenance: window, scan files used,
                                     per-bearing spread, bins kept

Only scans whose timestamp (from the extracted_pcd file name,
cloud_<epoch ns>.pcd) falls inside [first_frame_ts_ns, last_frame_ts_ns] of
<ID>_metadata.json are used, so each staged scan is provably associated with
its image. The scan's angular grid is fixed per revolution, so bearings are
binned on that grid and a bin is kept only when it has a return in at least
--min-return-fraction of the in-window scans.
"""
from pathlib import Path
import argparse
import json
import shutil

import numpy as np


def read_ascii_pcd_xy(path: Path) -> np.ndarray:
    with path.open("r") as f:
        lines = f.readlines()

    data_start = None
    for i, line in enumerate(lines):
        if line.strip().lower() == "data ascii":
            data_start = i + 1
            break

    if data_start is None:
        raise ValueError(f"No DATA ascii header found in {path}")

    pts = []
    for line in lines[data_start:]:
        parts = line.strip().split()
        if len(parts) >= 2:
            pts.append([float(parts[0]), float(parts[1])])

    return np.asarray(pts, dtype=float)


def write_ascii_pcd_xy(path: Path, xy: np.ndarray) -> None:
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        "FIELDS x y z\n"
        "SIZE 4 4 4\n"
        "TYPE F F F\n"
        "COUNT 1 1 1\n"
        f"WIDTH {len(xy)}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {len(xy)}\n"
        "DATA ascii\n"
    )
    with path.open("w") as f:
        f.write(header)
        for x, y in xy:
            f.write(f"{x:.10g} {y:.10g} 0\n")


def scan_timestamp_ns(path: Path) -> int:
    # extracted_pcd naming convention: cloud_<epoch ns>.pcd
    return int(path.stem.split("_", 1)[1])


def burst_median(scans: list[np.ndarray], min_return_fraction: float) -> tuple[np.ndarray, dict]:
    bearings = [np.arctan2(xy[:, 1], xy[:, 0]) for xy in scans]
    ranges = [np.hypot(xy[:, 0], xy[:, 1]) for xy in scans]

    # Angular grid step, from consecutive returns within each scan.
    steps = np.concatenate([np.diff(b) for b in bearings])
    step = float(np.median(steps[(steps > 0) & (steps < np.radians(1.0))]))
    reference = float(min(b.min() for b in bearings))

    per_bin = {}
    for b, r in zip(bearings, ranges):
        for k, rr in zip(np.round((b - reference) / step).astype(int), r):
            per_bin.setdefault(int(k), []).append(rr)

    minimum_returns = min_return_fraction * len(scans)
    kept = sorted(k for k, v in per_bin.items() if len(v) >= minimum_returns)

    median_range = np.array([np.median(per_bin[k]) for k in kept])
    spread = np.array([np.std(per_bin[k]) for k in kept])
    bearing = reference + np.array(kept) * step

    xy = np.c_[median_range * np.cos(bearing), median_range * np.sin(bearing)]
    stats = {
        "angular_step_deg": float(np.degrees(step)),
        "bins_seen": len(per_bin),
        "bins_kept": len(kept),
        "median_per_bearing_range_std_m": float(np.median(spread)),
        "p95_per_bearing_range_std_m": float(np.percentile(spread, 95)),
    }
    return xy, stats


def main():
    parser = argparse.ArgumentParser(
        description="Stage images/ and lasers/ for cam_lidar_2d_icp.py from a capture session."
    )
    parser.add_argument("session", help="Session root, e.g. data/data_2026-09-23.")
    parser.add_argument(
        "--min-return-fraction",
        type=float,
        default=0.5,
        help="Keep a bearing only if it returned in at least this fraction of in-window scans.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing files in images/ and lasers/.",
    )
    args = parser.parse_args()

    session = Path(args.session)
    image_dir = session / "images"
    laser_dir = session / "lasers"

    for out in (image_dir, laser_dir):
        if out.exists() and any(out.iterdir()) and not args.overwrite:
            raise FileExistsError(f"{out} is not empty; pass --overwrite to replace it.")
        out.mkdir(parents=True, exist_ok=True)

    capture_dirs = sorted(p for p in (session / "captures").iterdir() if p.is_dir())

    staged = []
    for capture_dir in capture_dirs:
        capture_id = capture_dir.name
        metadata = json.loads((capture_dir / f"{capture_id}_metadata.json").read_text())
        first_ns = int(metadata["timing"]["first_frame_ts_ns"])
        last_ns = int(metadata["timing"]["last_frame_ts_ns"])

        scan_files = [
            p for p in sorted((session / "extracted_pcd" / capture_id).glob("*.pcd"))
            if first_ns <= scan_timestamp_ns(p) <= last_ns
        ]
        if not scan_files:
            raise RuntimeError(f"{capture_id}: no scans inside the camera window.")

        xy, stats = burst_median(
            [read_ascii_pcd_xy(p) for p in scan_files],
            args.min_return_fraction,
        )

        shutil.copy2(capture_dir / f"{capture_id}_left.png", image_dir / f"{capture_id}_left.png")
        write_ascii_pcd_xy(laser_dir / f"{capture_id}.pcd", xy)

        record = {
            "capture_id": capture_id,
            "image": str(image_dir / f"{capture_id}_left.png"),
            "image_source": str(capture_dir / f"{capture_id}_left.png"),
            "laser": str(laser_dir / f"{capture_id}.pcd"),
            "camera_window_ns": [first_ns, last_ns],
            "scan_count": len(scan_files),
            "first_scan": scan_files[0].name,
            "last_scan": scan_files[-1].name,
            **stats,
        }
        staged.append(record)
        print(
            f"{capture_id}: {len(scan_files)} in-window scans -> {stats['bins_kept']} bearings, "
            f"median per-bearing std {stats['median_per_bearing_range_std_m'] * 1000:.2f} mm"
        )

    manifest = {
        "session": str(session),
        "laser_method": "per-bearing median of in-window scans",
        "min_return_fraction": args.min_return_fraction,
        "captures": staged,
    }
    (session / "staging_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Staged {len(staged)} captures; wrote {session / 'staging_manifest.json'}")


if __name__ == "__main__":
    main()
