#!/usr/bin/env python3
"""Phase 1.6 Test A: regress the board-plane depth residual e = d_gt - d_pnp
against camera-frame lateral position y_R, pooled and per pose.

This is PLAN.md Sec 5's board-plane validation (an independent geometric
reference: the checkerboard plane, known from solvePnP alone, no LiDAR, no
T2) with the diagnostic regression Phase 1.6 asked for. It uses the
board-struck correspondence reconstruction in board_correspondences.py (see
that module's docstring for why it is a reconstruction, not the true
human-curated correspondence set, and why that is a reasonable stand-in
here) -- N is in the hundreds, not four.

Signature table (PLAN.md Sec 2 Jacobians):
    straight line crossing zero at y_R = t_y, slope = -delta_psi  -> yaw error
    constant offset, no slope                                    -> t_x error
    slope on oblique poses only, ~zero on frontal poses           -> t_y error

Read-only against cam_lidar_2d_icp.py / gui.py / icp_2d.py /
tools/lidar_characterisation (the one authorised additive change --
the calibration_correspondences.npz dump -- is already in
cam_lidar_2d_icp.py; this script does not depend on it existing on disk,
since producing it needs a live interactive run this environment cannot
perform).

Run from the repository root:
    python tools/lidar_ground_truth/diagnostics/testA_board_plane_lateral_regression.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # tools/lidar_ground_truth/
sys.path.insert(0, str(Path(__file__).resolve().parent))       # diagnostics/

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

import calibration_io
import config
import projection
import rig as rig_module
from board_correspondences import build_all_pose_correspondences

OUT_DIR = config.REPO_ROOT / "results" / "lidar_ground_truth" / "diagnostics" / "testA_board_plane_regression"


def compute_residuals(pc, T2, delta_z_m, board_standoff_m):
    R2, t2 = T2[:2, :2], T2[:2, 2]
    xy_r = pc.xy_lidar @ R2.T + t2
    y_r = xy_r[:, 1]

    result = projection.project_scan(
        pc.xy_lidar, T2, delta_z_m, pc.K, pc.image_wh, z_min_m=config.Z_MIN_M,
    )
    u, v, d_gt, keep = result.u, result.v, result.depth_m, result.keep

    fx, fy, cx, cy = pc.K[0, 0], pc.K[1, 1], pc.K[0, 2], pc.K[1, 2]
    m = np.stack([(u - cx) / fx, (v - cy) / fy, np.ones_like(u)], axis=1)

    p_wall = pc.board_point_C - board_standoff_m * pc.board_normal_C
    n = pc.board_normal_C
    denom = m @ n
    d_pnp = (n @ p_wall) / denom

    e = d_gt - d_pnp
    return y_r[keep], e[keep], d_gt[keep]


def fit_line_with_ci(x, y):
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    n = len(x)
    t_crit = stats.t.ppf(0.975, df=max(n - 2, 1))
    slope_ci = t_crit * std_err
    return {
        "slope": float(slope), "slope_ci95": float(slope_ci),
        "intercept": float(intercept), "r2": float(r_value ** 2),
        "p_value": float(p_value), "n": int(n),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    T2 = calibration_io.load_transform()
    rig_cfg = rig_module.load_rig(config.DEFAULT_RIG_PATH, allow_placeholder=True)
    rig_module.warn_if_placeholder(rig_cfg)

    t_y = T2[1, 2]
    print(f"T2 t_y = {t_y:+.4f} m (signature-table prediction: yaw error -> "
          f"regression crosses zero near y_R = t_y)")

    pcs = build_all_pose_correspondences()
    print(f"\nPer-pose board-struck point counts: "
          f"{[(pc.pose_id, len(pc.xy_lidar)) for pc in pcs]}")

    all_y, all_e, per_pose = [], [], {}
    for pc in pcs:
        y_r, e, d_gt = compute_residuals(pc, T2, rig_cfg.delta_z_m, rig_cfg.board_standoff_m)
        per_pose[pc.pose_id] = {"y_r": y_r, "e": e, "obliquity_deg": pc.obliquity_deg}
        all_y.append(y_r)
        all_e.append(e)

    all_y = np.concatenate(all_y)
    all_e = np.concatenate(all_e)

    pooled_fit = fit_line_with_ci(all_y, all_e)
    print()
    print("=" * 90)
    print(f"POOLED fit (N={pooled_fit['n']}): e = {pooled_fit['slope']:.4f} "
          f"(+-{pooled_fit['slope_ci95']:.4f}) * y_R + {pooled_fit['intercept']:.4f}, "
          f"R^2={pooled_fit['r2']:.4f}, p={pooled_fit['p_value']:.2e}")
    zero_crossing = -pooled_fit['intercept'] / pooled_fit['slope'] if pooled_fit['slope'] else float('nan')
    print(f"  zero-crossing y_R = {zero_crossing:.4f} m  (t_y = {t_y:+.4f} m)")
    implied_dpsi_deg = float(np.degrees(-pooled_fit['slope']))
    print(f"  implied delta_psi if this is a pure yaw error: {implied_dpsi_deg:+.4f} deg")
    print("=" * 90)

    per_pose_fits = {}
    print()
    print(f"{'pose':10s} {'obliquity':>10s} {'N':>5s} {'slope':>10s} {'slope_CI95':>11s} "
          f"{'intercept':>10s} {'R2':>7s}")
    for pose_id, d in per_pose.items():
        fit = fit_line_with_ci(d["y_r"], d["e"])
        per_pose_fits[pose_id] = {**fit, "obliquity_deg": d["obliquity_deg"]}
        print(f"{pose_id:10s} {d['obliquity_deg']:>+9.2f}d {fit['n']:>5d} "
              f"{fit['slope']:>10.4f} {fit['slope_ci95']:>11.4f} "
              f"{fit['intercept']:>10.4f} {fit['r2']:>7.4f}")

    # --- Sensitivity to the two placeholder rig parameters ---
    print()
    print("Sensitivity of the POOLED slope to placeholder parameters:")
    for standoff in [0.0, 0.005, 0.01, 0.02]:
        y_r_all, e_all = [], []
        for pc in pcs:
            y_r, e, _ = compute_residuals(pc, T2, rig_cfg.delta_z_m, standoff)
            y_r_all.append(y_r); e_all.append(e)
        f = fit_line_with_ci(np.concatenate(y_r_all), np.concatenate(e_all))
        print(f"  board_standoff_m={standoff:.3f}: slope={f['slope']:+.4f} R2={f['r2']:.4f}")
    for dz in [0.0, 0.02, 0.045, 0.08]:
        y_r_all, e_all = [], []
        for pc in pcs:
            y_r, e, _ = compute_residuals(pc, T2, dz, rig_cfg.board_standoff_m)
            y_r_all.append(y_r); e_all.append(e)
        f = fit_line_with_ci(np.concatenate(y_r_all), np.concatenate(e_all))
        print(f"  delta_z_m={dz:.3f}: slope={f['slope']:+.4f} R2={f['r2']:.4f}")

    # --- Plots ---
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(per_pose)))
    for (pose_id, d), c in zip(per_pose.items(), colors):
        ax.scatter(d["y_r"], d["e"] * 1000, s=10, color=c, alpha=0.6, label=pose_id)
    xs = np.linspace(all_y.min(), all_y.max(), 100)
    ax.plot(xs, (pooled_fit["slope"] * xs + pooled_fit["intercept"]) * 1000,
            "k-", lw=2, label=f"pooled fit (slope={pooled_fit['slope']:.3f})")
    ax.axhline(0, c="gray", lw=0.8)
    ax.axvline(t_y, c="gray", lw=0.8, ls=":", label=f"t_y={t_y:.3f}")
    ax.set_xlabel("y_R (m)")
    ax.set_ylabel("e = d_gt - d_pnp (mm)")
    ax.set_title("Test A: board-plane depth residual vs lateral position")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "e_vs_yR.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    obliquities = [f["obliquity_deg"] for f in per_pose_fits.values()]
    slopes = [f["slope"] for f in per_pose_fits.values()]
    errs = [f["slope_ci95"] for f in per_pose_fits.values()]
    ax.errorbar(obliquities, slopes, yerr=errs, fmt="o", capsize=3)
    for pose_id, f in per_pose_fits.items():
        ax.annotate(pose_id, (f["obliquity_deg"], f["slope"]), fontsize=7,
                    textcoords="offset points", xytext=(5, 5))
    ax.axhline(0, c="gray", lw=0.8)
    ax.set_xlabel("pose obliquity (deg from frontoparallel)")
    ax.set_ylabel("per-pose slope (de/dy_R)")
    ax.set_title("Test A: slope vs pose obliquity")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "slope_vs_obliquity.png", dpi=150)
    plt.close(fig)

    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump({
            "t_y": float(t_y),
            "pooled_fit": pooled_fit,
            "zero_crossing_y_R": zero_crossing,
            "implied_delta_psi_deg": implied_dpsi_deg,
            "per_pose_fits": per_pose_fits,
        }, f, indent=2)
    print(f"\nSaved {OUT_DIR}")


if __name__ == "__main__":
    main()
