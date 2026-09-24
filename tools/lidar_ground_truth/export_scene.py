#!/usr/bin/env python3
"""Export sparse LiDAR-referenced ground-truth pixels for a capture session
(PLAN.md Sec 6.5).

For each capture, the staged burst-median scan (lasers/<ID>.pcd) is filtered
(scan_io), projected into the rectified left image (projection), filtered for
occlusion/parallax (filters), and given per-point uncertainty by Monte Carlo
over the bootstrap T2 ensemble and the range-noise model (uncertainty).
Captures are grouped by the `scene` field of their <ID>_metadata.json.

Outputs under --out-dir, per scene:
  <scene_id>_ground_truth.csv          one row per surviving point
  <scene_id>_ground_truth_summary.json provenance (input hashes, K, rig,
                                       bootstrap T2 covariance, counts by
                                       rejection reason, validation status)
  overlays/<ID>_ground_truth.png       image with points coloured by depth

Range-noise model: without an RPLIDAR S3 accuracy characterisation
(PLAN.md Phase 5) no bias correction is applied, and the per-bearing range
spread of the capture's scan burst (staging_manifest.json) is used as the
range sigma -- a precision term only. Every row records this in
uncertainty_source.

Run from the repository root:
    python tools/lidar_ground_truth/export_scene.py \\
        --session data/<session> \\
        --camera-manifest data/<session>/captures/<scene>/session_manifest.json \\  # scene-nested sessions only
        --transform results/calibration/<run>/lidar_to_camera_2d.npy \\
        --calibration-result results/calibration/<run>/calibration_result.json \\
        --correspondences results/calibration/<run>/calibration_correspondences.npz \\
        --bootstrap-ensemble results/lidar_ground_truth/<session>/board_plane_validation/bootstrap_T2_ensemble.npz \\
        --board-plane-validation results/lidar_ground_truth/<session>/board_plane_validation/board_plane_validation.json \\
        --rig config/rig_template.json \\
        --out-dir results/lidar_ground_truth/<session>/export
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
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
import uncertainty

CSV_FIELDS = [
    "scene_id", "capture_id", "source_scan_file", "source_image_file",
    "x_lidar_m", "y_lidar_m", "lidar_radial_range_m",
    "pixel_u", "pixel_v", "camera_depth_m",
    "depth_std_m", "u_std_px", "row_uncertainty_px",
    "lidar_bias_correction_mm", "lidar_sigma_mm", "uncertainty_source",
]


def find_capture_metadata_dir(captures_root: Path, capture_id: str) -> Path:
    """Locate <capture_id>'s metadata dir under captures_root, whether the
    layout is flat (captures/<ID>/) or scene-nested (captures/<scene>/<ID>/,
    e.g. data_2026-09-24's captures/scene_A/A01/).
    """
    matches = sorted(captures_root.glob(f"**/{capture_id}/{capture_id}_metadata.json"))
    if not matches:
        raise FileNotFoundError(f"No {capture_id}_metadata.json found under {captures_root}")
    if len(matches) > 1:
        raise ValueError(f"Multiple {capture_id}_metadata.json found under {captures_root}: {matches}")
    return matches[0].parent


def scene_group(metadata_dir: Path, captures_root: Path) -> str | None:
    """The capture's scene-group directory name (e.g. "scene_A"), or None
    for a flat layout where the capture dir is a direct child of captures/.
    """
    parts = metadata_dir.relative_to(captures_root).parts
    return parts[0] if len(parts) > 1 else None


def resolve_scene_name(metadata: dict, metadata_dir: Path, captures_root: Path) -> str:
    """The capture's scene name for grouping.

    Prefers the scene-group directory name over metadata['scene'] when the
    session is scene-nested: data_2026-09-24's capture metadata has
    "scene": "calibration" hardcoded on *every* capture regardless of which
    of scene_A/scene_B/scene_C it actually belongs to (a capture-side bug,
    confirmed by inspecting the raw files -- not something to silently
    "fix" by rewriting captured data). The directory layout is the
    unambiguous signal: one session_manifest.json per scene, so the
    scene-group directory a capture lives under is reliable even when the
    field inside its metadata isn't. Flat-layout sessions have no such
    grouping to fall back on, so metadata['scene'] is trusted there.
    """
    group = scene_group(metadata_dir, captures_root)
    return group if group is not None else metadata["scene"]


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def overlay_png(image_path: Path, u, v, depth, title: str, out_path: Path) -> None:
    image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(image)
    if len(u):
        sc = ax.scatter(u, v, c=depth, cmap="viridis", s=6, linewidths=0)
        fig.colorbar(sc, ax=ax, label="camera-frame depth d_gt (m)")
    ax.set_xlim(0, image.shape[1])
    ax.set_ylim(image.shape[0], 0)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--session", required=True, help="data/<session> with captures/, images/, lasers/, staging_manifest.json")
    parser.add_argument(
        "--camera-manifest",
        default=None,
        help=(
            "session_manifest.json with the rectified K (PLAN.md Sec 4.3). Default: "
            "<session>/captures/session_manifest.json (flat-layout sessions). A "
            "scene-nested session (e.g. data_2026-09-24) has one manifest per scene "
            "(<session>/captures/<scene>/session_manifest.json) and must pass this "
            "explicitly."
        ),
    )
    parser.add_argument("--capture-ids", nargs="*", default=None, help="default: every staged capture")
    parser.add_argument("--transform", required=True)
    parser.add_argument("--calibration-result", required=True)
    parser.add_argument("--correspondences", required=True, help="for the bootstrap T2 ensemble")
    parser.add_argument("--bootstrap-ensemble", required=True, help="ensemble cache (.npz); built here if missing or stale")
    parser.add_argument("--bootstrap-resamples", type=int, default=config.BOOTSTRAP_RESAMPLES)
    parser.add_argument("--board-plane-validation", default=None, help="validate_board_plane.py JSON, recorded in the summary")
    parser.add_argument("--rig", default=str(config.DEFAULT_RIG_PATH))
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    session = Path(args.session)
    manifest_path = Path(args.camera_manifest) if args.camera_manifest else session / "captures" / "session_manifest.json"
    staging_path = session / "staging_manifest.json"
    out_dir = Path(args.out_dir)
    (out_dir / "overlays").mkdir(parents=True, exist_ok=True)

    T2 = calibration_io.load_transform(args.transform)
    calibration_result = calibration_io.load_calibration_result(args.calibration_result)
    intrinsics = calibration_io.load_intrinsics_from_manifest(manifest_path)
    calibration_io.check_intrinsics_match_calibration(intrinsics, calibration_result)
    rig_cfg = rig_module.load_rig(args.rig)
    correspondences = uncertainty.load_correspondences(args.correspondences)
    ensemble = uncertainty.load_or_build_ensemble(
        Path(args.bootstrap_ensemble), args.correspondences, correspondences,
        calibration_result, args.bootstrap_resamples, config.BOOTSTRAP_SEED,
    )
    staged = {c["capture_id"]: c for c in json.loads(staging_path.read_text())["captures"]}
    capture_ids = args.capture_ids or sorted(staged)

    validation = None
    if args.board_plane_validation is not None:
        validation_json = json.loads(Path(args.board_plane_validation).read_text())
        validation = {
            "path": args.board_plane_validation,
            "sha256": sha256(args.board_plane_validation),
            "all_passed": validation_json["all_passed"],
            "acceptance": validation_json["acceptance"],
            "hold_out_leave_one_pose_out": validation_json["hold_out_leave_one_pose_out"],
        }

    captures_root = session / "captures"
    scenes: dict[str, list[dict]] = {}
    for capture_id in capture_ids:
        metadata_dir = find_capture_metadata_dir(captures_root, capture_id)
        metadata = json.loads((metadata_dir / f"{capture_id}_metadata.json").read_text())
        scene_id = f"{session.name}_{resolve_scene_name(metadata, metadata_dir, captures_root)}"
        image_path = session / "images" / f"{capture_id}_left.png"
        scan_path = session / "lasers" / f"{capture_id}.pcd"

        xy, n_total, n_valid = scan_io.load_valid_xy(scan_path)
        result = projection.project_scan(xy, T2, rig_cfg.delta_z_m, intrinsics.K, intrinsics.image_wh, z_min_m=config.Z_MIN_M)
        report = filters.apply_filters(xy, result.keep, result.u, result.depth_m)
        in_operating_range = filters.reject_outside_operating_range(result.depth_m)
        keep = report.keep & in_operating_range
        filtered_out_counts = dict(report.rejected_counts)
        filtered_out_counts["outside_operating_range_0.5_8m"] = int(np.count_nonzero(report.keep & ~in_operating_range))
        filtered_out_counts["surviving"] = int(np.count_nonzero(keep))

        range_sigma_m = float(staged[capture_id]["median_per_bearing_range_std_m"])
        mc = uncertainty.monte_carlo_point_uncertainty(
            xy[keep], ensemble, range_sigma_m, rig_cfg.delta_z_m, rig_cfg.delta_z_tolerance_m,
            rig_cfg.attitude_tolerance_deg, intrinsics.K, intrinsics.image_wh,
        )
        uncertainty_source = (
            f"bootstrap T2 ({len(ensemble)} resamples) + burst per-bearing range spread "
            f"({range_sigma_m * 1000:.2f} mm, precision only) + rig tolerances; "
            "no RPLIDAR S3 accuracy characterisation, no range bias correction"
        )
        rows = []
        for k, index in enumerate(np.flatnonzero(keep)):
            rows.append({
                "scene_id": scene_id,
                "capture_id": capture_id,
                "source_scan_file": str(scan_path),
                "source_image_file": str(image_path),
                "x_lidar_m": xy[index, 0],
                "y_lidar_m": xy[index, 1],
                "lidar_radial_range_m": result.lidar_radial_range_m[index],
                "pixel_u": result.u[index],
                "pixel_v": result.v[index],
                "camera_depth_m": result.depth_m[index],
                "depth_std_m": mc["depth_std_m"][k],
                "u_std_px": mc["u_std_px"][k],
                "row_uncertainty_px": mc["row_uncertainty_px"][k],
                "lidar_bias_correction_mm": 0.0,
                "lidar_sigma_mm": range_sigma_m * 1000,
                "uncertainty_source": uncertainty_source,
            })

        overlay_path = out_dir / "overlays" / f"{capture_id}_ground_truth.png"
        overlay_png(image_path, result.u[keep], result.v[keep], result.depth_m[keep],
                    f"{scene_id} / {capture_id}: {len(rows)} ground-truth pixels", overlay_path)

        scenes.setdefault(scene_id, []).append({
            "capture_id": capture_id,
            "rows": rows,
            "record": {
                "capture_id": capture_id,
                "image": str(image_path),
                "image_sha256": sha256(image_path),
                "scan": str(scan_path),
                "scan_sha256": sha256(scan_path),
                "scan_method": "per-bearing median of in-window scans (staging_manifest.json)",
                "n_total_returns": n_total,
                "n_valid_returns": n_valid,
                "filtered_out_counts": filtered_out_counts,
                "n_ground_truth_points": len(rows),
                "depth_range_m": [float(result.depth_m[keep].min()), float(result.depth_m[keep].max())] if rows else None,
                "median_depth_std_m": float(np.median(mc["depth_std_m"])) if rows else None,
                "median_u_std_px": float(np.median(mc["u_std_px"])) if rows else None,
                "median_row_uncertainty_px": float(np.median(mc["row_uncertainty_px"])) if rows else None,
                "overlay_png": str(overlay_path),
            },
        })
        print(
            f"{capture_id} ({scene_id}): {len(rows)} ground-truth pixels "
            f"(of {n_total} returns; {filtered_out_counts}), median depth std "
            f"{scenes[scene_id][-1]['record']['median_depth_std_m'] * 1000:.2f} mm, "
            f"median row uncertainty {scenes[scene_id][-1]['record']['median_row_uncertainty_px']:.2f} px"
        )

    provenance_files = {
        "transform": args.transform,
        "calibration_result": args.calibration_result,
        "correspondences": args.correspondences,
        "bootstrap_ensemble": args.bootstrap_ensemble,
        "rig": args.rig,
        "camera_manifest": str(manifest_path),
        "staging_manifest": str(staging_path),
    }
    for scene_id, captures in scenes.items():
        csv_path = out_dir / f"{scene_id}_ground_truth.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()
            for capture in captures:
                writer.writerows(capture["rows"])

        summary = {
            "scene_id": scene_id,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "inputs": {name: {"path": path, "sha256": sha256(path)} for name, path in provenance_files.items()},
            "K": intrinsics.K.tolist(),
            "K_source": intrinsics.source,
            "image_wh": list(intrinsics.image_wh),
            "T2": T2.tolist(),
            "T2_bootstrap": uncertainty.ensemble_summary(ensemble),
            "rig": {
                "rig_id": rig_cfg.rig_id,
                "measurement_status": rig_cfg.measurement_status,
                "measured_on": rig_cfg.measured_on,
                "delta_z_m": rig_cfg.delta_z_m,
                "delta_z_tolerance_m": rig_cfg.delta_z_tolerance_m,
                "pitch_deg": rig_cfg.pitch_deg,
                "roll_deg": rig_cfg.roll_deg,
                "attitude_tolerance_deg": rig_cfg.attitude_tolerance_deg,
            },
            "board_plane_validation": validation,
            "filters_not_applied": ["high incidence angle (needs RPLIDAR S3 characterisation, PLAN.md Phase 5)"],
            "range_noise_model": "burst per-bearing range spread (precision only); no bias correction",
            "depth_convention": "camera_depth_m is camera-frame Z in the rectified left frame, not radial range",
            "captures": [capture["record"] for capture in captures],
            "ground_truth_points": int(sum(len(capture["rows"]) for capture in captures)),
        }
        summary_path = out_dir / f"{scene_id}_ground_truth_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))
        print(f"Saved {csv_path} ({summary['ground_truth_points']} points) and {summary_path}")


if __name__ == "__main__":
    main()
