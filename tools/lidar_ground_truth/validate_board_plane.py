#!/usr/bin/env python3
"""Board-plane validation of the projected LiDAR ground truth (PLAN.md Sec 5).

For every calibration pose, the LiDAR returns selected on the board are
projected through PLAN.md Sec 1.2 to (u, v, d_gt), and each pixel ray is
intersected with the board plane that solvePnP placed from the image alone
(offset by the rig's board_standoff_m to the surface the LiDAR strikes):

    e = d_gt - d_pnp,   d_pnp = (n . p0) / (n . K^-1 [u, v, 1])

Reported in-sample (the calibration's own T2) and in hold-out form
(leave-one-pose-out: T2 re-solved without the pose being evaluated, PLAN.md
Sec 5.3), with trends against distance and board yaw, the bootstrap T2
spread, and pass/fail against the PLAN.md Sec 5.6 thresholds in config.py.

This validates d_gt and u within the board's angular window only. It cannot
validate v (depth on a vertical plane is row-independent) or bearings the
board never covered (PLAN.md Sec 5.4).

Run from the repository root:
    python tools/lidar_ground_truth/validate_board_plane.py \\
        --correspondences results/calibration/<run>/calibration_correspondences.npz \\
        --transform results/calibration/<run>/lidar_to_camera_2d.npy \\
        --calibration-result results/calibration/<run>/calibration_result.json \\
        --camera-manifest data/<session>/captures/session_manifest.json \\
        --staging-manifest data/<session>/staging_manifest.json \\
        --rig config/rig_template.json \\
        --out-dir results/lidar_ground_truth/<session>/board_plane_validation
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import calibration_io
import config
import projection
import rig as rig_module
import uncertainty


def board_plane_residuals(
    xy_lidar: np.ndarray,
    T2: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    rig_cfg: rig_module.RigConfig,
    intrinsics: calibration_io.RectifiedIntrinsics,
) -> dict:
    result = projection.project_scan(
        xy_lidar, T2, rig_cfg.delta_z_m, intrinsics.K, intrinsics.image_wh, z_min_m=config.Z_MIN_M,
    )
    keep = result.keep
    u, v, d_gt = result.u[keep], result.v[keep], result.depth_m[keep]

    R_board, _ = cv2.Rodrigues(rvec)
    normal = R_board[:, 2]
    p0 = np.asarray(tvec, dtype=np.float64).ravel()
    if normal @ p0 < 0:  # orient the normal away from the camera
        normal = -normal
    p0 = p0 + rig_cfg.board_standoff_m * normal  # surface the LiDAR strikes

    K = intrinsics.K
    rays = np.c_[(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)]
    d_pnp = (normal @ p0) / (rays @ normal)

    points_cv = rays * d_gt[:, None]  # camera OpenCV frame, Z = d_gt
    return {
        "u": u,
        "v": v,
        "d_gt": d_gt,
        "d_pnp": d_pnp,
        "e": d_gt - d_pnp,
        "point_to_plane_m": (points_cv - p0) @ normal,
        "n_projected": int(np.count_nonzero(keep)),
        "n_total": int(len(xy_lidar)),
    }


def board_yaw_deg(rvec: np.ndarray) -> float:
    """Horizontal angle between the board normal and the optical axis."""
    R_board, _ = cv2.Rodrigues(rvec)
    normal = R_board[:, 2]
    if normal[2] < 0:
        normal = -normal
    return float(math.degrees(math.atan2(normal[0], normal[2])))


def stats(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    return {
        "count": int(values.size),
        "mean_m": float(values.mean()),
        "std_m": float(values.std()),
        "rmse_m": float(np.sqrt(np.mean(values ** 2))),
        "p95_abs_m": float(np.percentile(np.abs(values), 95)),
    }


def pose_level_trend(x: np.ndarray, y: np.ndarray) -> dict:
    """Least-squares slope of per-pose mean residual against x, with its
    standard error (pose-level, so points within a pose are not treated as
    independent)."""
    A = np.c_[x, np.ones_like(x)]
    coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    residual = y - A @ coef
    dof = max(len(x) - 2, 1)
    cov = np.linalg.inv(A.T @ A) * (residual @ residual) / dof
    slope_se = float(math.sqrt(cov[0, 0]))
    return {
        "slope": float(coef[0]),
        "slope_standard_error": slope_se,
        "slope_significance_sigma": float(abs(coef[0]) / slope_se) if slope_se > 0 else float("inf"),
        "intercept": float(coef[1]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--correspondences", required=True)
    parser.add_argument("--transform", required=True)
    parser.add_argument("--calibration-result", required=True)
    parser.add_argument("--camera-manifest", required=True)
    parser.add_argument("--staging-manifest", required=True, help="per-bearing burst spread = LiDAR noise floor")
    parser.add_argument("--rig", default=str(config.DEFAULT_RIG_PATH))
    parser.add_argument("--bootstrap-resamples", type=int, default=config.BOOTSTRAP_RESAMPLES)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    T2 = calibration_io.load_transform(args.transform)
    calibration_result = calibration_io.load_calibration_result(args.calibration_result)
    intrinsics = calibration_io.load_intrinsics_from_manifest(args.camera_manifest)
    calibration_io.check_intrinsics_match_calibration(intrinsics, calibration_result)
    rig_cfg = rig_module.load_rig(args.rig)
    correspondences = uncertainty.load_correspondences(args.correspondences)
    pose_count = len(correspondences.pose_ids)

    staging = json.loads(Path(args.staging_manifest).read_text())
    noise_floor_m = float(np.median([c["median_per_bearing_range_std_m"] for c in staging["captures"]]))

    rows = []
    per_pose = []
    in_sample_all, hold_out_all = [], []
    for index, pose_id in enumerate(correspondences.pose_ids):
        others = [j for j in range(pose_count) if j != index]
        T2_hold_out = uncertainty.solve_T2(correspondences, others, calibration_result)

        args_common = (correspondences.lidar_points[index], correspondences.rvecs[index], correspondences.tvecs[index])
        in_sample = board_plane_residuals(args_common[0], T2, *args_common[1:], rig_cfg, intrinsics)
        hold_out = board_plane_residuals(args_common[0], T2_hold_out, *args_common[1:], rig_cfg, intrinsics)
        in_sample_all.append(in_sample["e"])
        hold_out_all.append(hold_out["e"])

        yaw = board_yaw_deg(correspondences.rvecs[index])
        record = {
            "pose_id": pose_id,
            "board_yaw_deg": yaw,
            "mean_distance_m": float(np.mean(hold_out["d_pnp"])),
            "n_points": hold_out["n_projected"],
            "in_sample": stats(in_sample["e"]),
            "hold_out": stats(hold_out["e"]),
            "hold_out_point_to_plane": stats(hold_out["point_to_plane_m"]),
            "hold_out_T2": dict(zip(("t_x_m", "t_y_m", "yaw_deg"), uncertainty.T2_parameters(T2_hold_out))),
        }
        per_pose.append(record)
        print(
            f"{pose_id}: distance {record['mean_distance_m']:.2f} m, board yaw {yaw:+.1f} deg | "
            f"in-sample mean e {record['in_sample']['mean_m'] * 1000:+.1f} mm | "
            f"hold-out mean e {record['hold_out']['mean_m'] * 1000:+.1f} mm, RMSE {record['hold_out']['rmse_m'] * 1000:.1f} mm"
        )
        for k in range(len(hold_out["e"])):
            rows.append({
                "pose_id": pose_id,
                "u_px": hold_out["u"][k],
                "v_px": hold_out["v"][k],
                "d_gt_hold_out_m": hold_out["d_gt"][k],
                "d_pnp_m": hold_out["d_pnp"][k],
                "e_hold_out_m": hold_out["e"][k],
                "point_to_plane_hold_out_m": hold_out["point_to_plane_m"][k],
            })

    in_sample_all = np.concatenate(in_sample_all)
    hold_out_all = np.concatenate(hold_out_all)

    distance = np.array([p["mean_distance_m"] for p in per_pose])
    yaw = np.array([p["board_yaw_deg"] for p in per_pose])
    pose_bias = np.array([p["hold_out"]["mean_m"] for p in per_pose])
    trend_distance = pose_level_trend(distance, pose_bias)
    trend_yaw = pose_level_trend(yaw, pose_bias)

    ensemble = uncertainty.load_or_build_ensemble(
        out_dir / "bootstrap_T2_ensemble.npz", args.correspondences, correspondences,
        calibration_result, args.bootstrap_resamples, config.BOOTSTRAP_SEED,
    )
    bootstrap = uncertainty.ensemble_summary(ensemble)

    hold_out_stats = stats(hold_out_all)
    mean_abs_pose_bias = float(np.mean(np.abs(pose_bias)))
    acceptance = {
        "hold_out_overall_bias_below_noise_floor": {
            "value_m": abs(hold_out_stats["mean_m"]), "threshold_m": noise_floor_m,
            "passed": abs(hold_out_stats["mean_m"]) < noise_floor_m,
        },
        "hold_out_mean_abs_pose_bias_below_noise_floor": {
            "value_m": mean_abs_pose_bias, "threshold_m": noise_floor_m,
            "passed": mean_abs_pose_bias < noise_floor_m,
        },
        "no_trend_vs_distance": {
            "significance_sigma": trend_distance["slope_significance_sigma"],
            "threshold_sigma": config.ACCEPT_TREND_SIGNIFICANCE_SIGMA,
            "passed": trend_distance["slope_significance_sigma"] < config.ACCEPT_TREND_SIGNIFICANCE_SIGMA,
        },
        "no_trend_vs_board_yaw": {
            "significance_sigma": trend_yaw["slope_significance_sigma"],
            "threshold_sigma": config.ACCEPT_TREND_SIGNIFICANCE_SIGMA,
            "passed": trend_yaw["slope_significance_sigma"] < config.ACCEPT_TREND_SIGNIFICANCE_SIGMA,
        },
        "bootstrap_yaw_std_below_threshold": {
            "value_deg": bootstrap["std"]["yaw_deg"], "threshold_deg": config.ACCEPT_BOOTSTRAP_YAW_STD_DEG,
            "passed": bootstrap["std"]["yaw_deg"] < config.ACCEPT_BOOTSTRAP_YAW_STD_DEG,
        },
    }

    summary = {
        "inputs": {
            "correspondences": args.correspondences,
            "transform": args.transform,
            "calibration_result": args.calibration_result,
            "camera_manifest": args.camera_manifest,
            "staging_manifest": args.staging_manifest,
            "rig": args.rig,
            "rig_measurement_status": rig_cfg.measurement_status,
            "delta_z_m": rig_cfg.delta_z_m,
            "board_standoff_m": rig_cfg.board_standoff_m,
        },
        "lidar_noise_floor_m": noise_floor_m,
        "in_sample": stats(in_sample_all),
        "hold_out_leave_one_pose_out": hold_out_stats,
        "trend_vs_distance_per_m": trend_distance,
        "trend_vs_board_yaw_per_deg": trend_yaw,
        "bootstrap_T2": bootstrap,
        "acceptance": acceptance,
        "all_passed": all(check["passed"] for check in acceptance.values()),
        "per_pose": per_pose,
        "limitations": (
            "Validates d_gt and u only within the board's angular window; cannot validate the image "
            "row v or bearings the board never covered (PLAN.md Sec 5.4)."
        ),
    }
    (out_dir / "board_plane_validation.json").write_text(json.dumps(summary, indent=2))
    with open(out_dir / "board_plane_residuals.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, x, label, trend in (
        (axes[0], distance, "mean board distance (m)", trend_distance),
        (axes[1], yaw, "board yaw (deg)", trend_yaw),
    ):
        ax.axhline(0.0, color="gray", lw=0.8)
        ax.scatter(x, pose_bias * 1000, color="tab:blue", zorder=3)
        for xi, bi, pose_id in zip(x, pose_bias, correspondences.pose_ids):
            ax.annotate(pose_id, (xi, bi * 1000), fontsize=7, xytext=(3, 3), textcoords="offset points")
        xs = np.linspace(x.min(), x.max(), 2)
        ax.plot(xs, (trend["slope"] * xs + trend["intercept"]) * 1000, color="tab:red",
                label=f"slope {trend['slope'] * 1000:+.2f} mm/unit ({trend['slope_significance_sigma']:.1f} sigma)")
        ax.set_xlabel(label)
        ax.set_ylabel("hold-out mean e = d_gt - d_pnp (mm)")
        ax.legend(fontsize=8)
    fig.suptitle("Board-plane hold-out residual per pose")
    fig.tight_layout()
    fig.savefig(out_dir / "board_plane_residual_trends.png", dpi=150)
    plt.close(fig)

    print()
    print(f"In-sample:  mean {summary['in_sample']['mean_m'] * 1000:+.2f} mm, RMSE {summary['in_sample']['rmse_m'] * 1000:.2f} mm")
    print(f"Hold-out:   mean {hold_out_stats['mean_m'] * 1000:+.2f} mm, std {hold_out_stats['std_m'] * 1000:.2f} mm, "
          f"RMSE {hold_out_stats['rmse_m'] * 1000:.2f} mm, p95|e| {hold_out_stats['p95_abs_m'] * 1000:.2f} mm")
    print(f"Trend vs distance: {trend_distance['slope'] * 1000:+.2f} mm/m ({trend_distance['slope_significance_sigma']:.1f} sigma); "
          f"vs board yaw: {trend_yaw['slope'] * 1000:+.3f} mm/deg ({trend_yaw['slope_significance_sigma']:.1f} sigma)")
    print(f"Bootstrap T2: sigma t_x {bootstrap['std']['t_x_m'] * 1000:.2f} mm, sigma t_y {bootstrap['std']['t_y_m'] * 1000:.2f} mm, "
          f"sigma yaw {bootstrap['std']['yaw_deg']:.3f} deg")
    for name, check in acceptance.items():
        print(f"  {'PASS' if check['passed'] else 'FAIL'}  {name}")
    print(f"Saved {out_dir / 'board_plane_validation.json'}")


if __name__ == "__main__":
    main()
