"""Stage a capture session for cam_lidar_2d_icp.py.

For every capture <ID> found under <session>/captures/ this writes, pairing
by the shared capture ID:

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

Capture dirs are found by recursively searching <session>/captures/ for any
directory <D> containing <D>_metadata.json, so both a flat layout
(captures/<ID>/) and a scene-nested layout (captures/<scene>/<ID>/, e.g.
data_2026-09-24's captures/scene_A/A01/) work unmodified; the corresponding
extracted_pcd/ location is derived from the same path, relative to
captures/. Use --scenes to stage only the capture dirs whose immediate
parent (the scene-group directory name, e.g. "scene_A") is in the given
list -- capture IDs are assumed unique across scenes (they are, by prefix:
A../B../C..), so images/ and lasers/ accumulate across separate invocations
and staging_manifest.json is merged (keyed by capture_id), not overwritten.
"""
from pathlib import Path
import argparse
import json
import shutil

import numpy as np


def find_capture_dirs(captures_root: Path) -> list[Path]:
    """Recursively find capture directories under captures_root.

    A directory <D> is a capture dir iff it contains <D.name>_metadata.json.
    Matches both captures/<ID>/ (flat) and captures/<scene>/<ID>/ (nested).
    """
    dirs = {
        metadata_file.parent
        for metadata_file in captures_root.rglob("*_metadata.json")
        if metadata_file.name == f"{metadata_file.parent.name}_metadata.json"
    }
    return sorted(dirs)


def scene_group(capture_dir: Path, captures_root: Path) -> str | None:
    """The capture's scene-group directory name (e.g. "scene_A"), or None
    for a flat layout where the capture dir is a direct child of captures/.
    """
    parts = capture_dir.relative_to(captures_root).parts
    return parts[0] if len(parts) > 1 else None


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
    parser.add_argument("session", help="Session root, e.g. data/data_2026-09-24.")
    parser.add_argument(
        "--scenes",
        nargs="*",
        default=None,
        help=(
            "Stage only captures under these scene-group directory names "
            "(e.g. --scenes scene_A), for a scene-nested session. Default: "
            "stage every capture dir found under captures/."
        ),
    )
    parser.add_argument(
        "--min-return-fraction",
        type=float,
        default=0.5,
        help="Keep a bearing only if it returned in at least this fraction of in-window scans.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an already-staged capture's image/laser/manifest entry.",
    )
    args = parser.parse_args()

    session = Path(args.session)
    image_dir = session / "images"
    laser_dir = session / "lasers"
    captures_root = session / "captures"
    image_dir.mkdir(parents=True, exist_ok=True)
    laser_dir.mkdir(parents=True, exist_ok=True)

    capture_dirs = find_capture_dirs(captures_root)
    if args.scenes is not None:
        capture_dirs = [
            d for d in capture_dirs if scene_group(d, captures_root) in args.scenes
        ]
        if not capture_dirs:
            raise ValueError(f"--scenes {args.scenes} matched no capture dirs under {captures_root}")

    staging_manifest_path = session / "staging_manifest.json"
    if staging_manifest_path.exists():
        manifest = json.loads(staging_manifest_path.read_text())
        already_staged = {c["capture_id"]: c for c in manifest["captures"]}
    else:
        manifest = {
            "session": str(session),
            "laser_method": "per-bearing median of in-window scans",
            "min_return_fraction": args.min_return_fraction,
            "captures": [],
        }
        already_staged = {}

    for capture_dir in capture_dirs:
        capture_id = capture_dir.name
        if capture_id in already_staged and not args.overwrite:
            print(f"{capture_id}: already staged, skipping (pass --overwrite to redo it).")
            continue

        metadata = json.loads((capture_dir / f"{capture_id}_metadata.json").read_text())
        first_ns = int(metadata["timing"]["first_frame_ts_ns"])
        last_ns = int(metadata["timing"]["last_frame_ts_ns"])

        extracted_pcd_dir = session / "extracted_pcd" / capture_dir.relative_to(captures_root)
        scan_files = [
            p for p in sorted(extracted_pcd_dir.glob("*.pcd"))
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

        already_staged[capture_id] = {
            "capture_id": capture_id,
            "scene_group": scene_group(capture_dir, captures_root),
            "image": str(image_dir / f"{capture_id}_left.png"),
            "image_source": str(capture_dir / f"{capture_id}_left.png"),
            "laser": str(laser_dir / f"{capture_id}.pcd"),
            "camera_window_ns": [first_ns, last_ns],
            "scan_count": len(scan_files),
            "first_scan": scan_files[0].name,
            "last_scan": scan_files[-1].name,
            **stats,
        }
        print(
            f"{capture_id}: {len(scan_files)} in-window scans -> {stats['bins_kept']} bearings, "
            f"median per-bearing std {stats['median_per_bearing_range_std_m'] * 1000:.2f} mm"
        )

    manifest["captures"] = [already_staged[k] for k in sorted(already_staged)]
    staging_manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Staged {len(manifest['captures'])} captures total; wrote {staging_manifest_path}")


if __name__ == "__main__":
    main()
