#!/usr/bin/env python3
"""Phase 1 diagnostic (PLAN.md Sec 5.5, 8): project existing curated LiDAR
scans into their paired calibration images using the existing SE(2)
calibration plus a rig delta_z, to visually confirm the projection core.

Verification gate (PLAN.md Sec 8, Phase 1): projected points must land on
the checkerboard, in a near-horizontal line, at the predicted row. This
catches sign errors and frame-convention mistakes immediately.

Headless. Requires no new data capture -- it runs against the existing
image/laser pairs already used by cam_lidar_2d_icp.py, which are read
identically (same folder-pairing convention) but never through that file.

Run from the repository root:
    python tools/lidar_ground_truth/overlay_diagnostic.py \\
        --image-dir data/<session>/images \\
        --laser-dir data/<session>/lasers \\
        --camera-manifest data/<session>/captures/session_manifest.json \\
        --transform results/calibration/<session>/lidar_to_camera_2d.npy \\
        --calibration-result results/calibration/<session>/calibration_result.json \\
        --out-dir results/lidar_ground_truth/<session>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import calibration_io
import config
import filters
import projection
import rig as rig_module
import scan_io


def pose_id_from_filename(path: Path) -> str:
    """Capture ID of a staged pair: the laser file stem (e.g. A09.pcd -> A09)."""
    return path.stem


def make_overlay_plot(
    image_bgr: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    depth_m: np.ndarray,
    predicted_row_px: float | None,
    pose_id: str,
    out_path: Path,
) -> None:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(image_rgb)
    if len(u):
        sc = ax.scatter(u, v, c=depth_m, cmap="viridis", s=12, edgecolors="black", linewidths=0.3)
        fig.colorbar(sc, ax=ax, label="camera-frame depth d_gt (m)")
    if predicted_row_px is not None:
        ax.axhline(
            predicted_row_px, color="red", ls="--", lw=1.0,
            label=f"predicted row at median depth ({predicted_row_px:.1f} px)",
        )
        ax.legend(loc="upper right", fontsize=8)
    ax.set_title(f"{pose_id}: projected LiDAR returns surviving filters (N={len(u)})")
    ax.set_xlim(0, image_bgr.shape[1])
    ax.set_ylim(image_bgr.shape[0], 0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def process_pose(
    image_path: Path,
    laser_path: Path,
    T2: np.ndarray,
    rig_cfg: rig_module.RigConfig,
    intrinsics: calibration_io.RectifiedIntrinsics,
    out_dir: Path,
) -> dict:
    pose_id = pose_id_from_filename(laser_path)
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise ValueError(f"Could not read image: {image_path}")
    h, w = image_bgr.shape[:2]
    if (w, h) != intrinsics.image_wh:
        raise ValueError(
            f"{image_path} is {w}x{h} but {intrinsics.source} is for "
            f"{intrinsics.image_wh[0]}x{intrinsics.image_wh[1]} images."
        )

    xy_valid, n_total, n_valid = scan_io.load_valid_xy(laser_path)
    result = projection.project_scan(
        xy_valid, T2, rig_cfg.delta_z_m, intrinsics.K, (w, h), z_min_m=config.Z_MIN_M,
    )
    report = filters.apply_filters(xy_valid, result.keep, result.u, result.depth_m)

    kept = report.keep
    n_kept = int(np.count_nonzero(kept))
    median_depth = float(np.median(result.depth_m[kept])) if n_kept else float("nan")

    predicted_row_px = None
    if n_kept and np.isfinite(median_depth) and median_depth > 0:
        predicted_row_px = float(
            intrinsics.K[1, 2] - intrinsics.K[1, 1] * rig_cfg.delta_z_m / median_depth
        )

    make_overlay_plot(
        image_bgr, result.u[kept], result.v[kept], result.depth_m[kept],
        predicted_row_px, pose_id, out_dir / f"{pose_id}_overlay.png",
    )

    return {
        "pose_id": pose_id,
        "image_file": image_path.name,
        "laser_file": laser_path.name,
        "n_total_returns": n_total,
        "n_valid_returns": n_valid,
        "n_in_frame": int(np.count_nonzero(result.keep)),
        "n_after_filters": n_kept,
        "filter_rejected_counts": report.rejected_counts,
        "K_source": intrinsics.source,
        "median_depth_m": median_depth,
        "predicted_row_px": predicted_row_px,
        "image_height_px": h,
        "overlay_png": str((out_dir / f"{pose_id}_overlay.png").resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--laser-dir", required=True)
    parser.add_argument(
        "--camera-manifest", required=True,
        help="session_manifest.json of the capture session (rectified K, PLAN.md Sec 4.3)",
    )
    parser.add_argument("--transform", required=True, help="lidar_to_camera_2d.npy")
    parser.add_argument(
        "--calibration-result", required=True,
        help="calibration_result.json from the same calibration run",
    )
    parser.add_argument("--rig", default=str(config.DEFAULT_RIG_PATH))
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    T2 = calibration_io.load_transform(args.transform)
    intrinsics = calibration_io.load_intrinsics_from_manifest(args.camera_manifest)
    calibration_io.check_intrinsics_match_calibration(
        intrinsics, calibration_io.load_calibration_result(args.calibration_result),
    )
    rig_cfg = rig_module.load_rig(args.rig, allow_placeholder=True)
    rig_module.warn_if_placeholder(rig_cfg)

    out_dir = Path(args.out_dir) / "overlay"
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = scan_io.load_images_and_scans_from_folders(args.image_dir, args.laser_dir)
    print(f"{len(pairs)} image/laser pairs found in {args.image_dir} / {args.laser_dir}")

    pose_records = []
    for image_path, laser_path in pairs:
        record = process_pose(
            image_path, laser_path, T2, rig_cfg, intrinsics, out_dir,
        )
        pose_records.append(record)
        print(
            f"{record['pose_id']}: scan={record['laser_file']} "
            f"n_total={record['n_total_returns']} n_valid={record['n_valid_returns']} "
            f"n_in_frame={record['n_in_frame']} n_after_filters={record['n_after_filters']} "
            f"K_source={record['K_source']} median_depth_m={record['median_depth_m']:.3f} "
            f"predicted_row_px={record['predicted_row_px']}"
        )
        print(f"    filter counts: {record['filter_rejected_counts']}")

    summary_path = out_dir.parent / "overlay_summary.json"
    with open(summary_path, "w") as f:
        json.dump(
            {
                "image_dir": str(args.image_dir),
                "laser_dir": str(args.laser_dir),
                "camera_manifest": args.camera_manifest,
                "calibration_result_path": str(args.calibration_result),
                "transform_path": str(args.transform),
                "rig_path": str(args.rig),
                "rig_is_placeholder": rig_cfg.is_placeholder,
                "delta_z_m_used": rig_cfg.delta_z_m,
                "poses": pose_records,
            },
            f,
            indent=2,
        )
    print(f"Saved {summary_path}")
    print(f"Saved overlays under {out_dir}")


if __name__ == "__main__":
    main()
