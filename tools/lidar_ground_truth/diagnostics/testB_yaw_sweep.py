#!/usr/bin/env python3
"""Phase 1.6 Test B: sweep the ICP objective in yaw, re-optimising
translation only at each value, and plot the resulting cost.

Uses the SAME board-struck correspondence reconstruction as Test A
(board_correspondences.py -- see its docstring for the reconstruction
methodology and its caveats) and the SAME orthogonal point-to-line cost
already established as the repo's primary residual metric
(cam_lidar_2d_icp.py::evaluate_alignment_residuals / point_to_infinite_line_distances).

For a fixed candidate yaw psi, the optimal SHARED translation t (one t
across all poses at once, matching how icp_2d.icp_per_line solves one
shared rigid transform per iteration across per-pose isolated
correspondences) minimising sum of squared orthogonal distances to each
pose's own camera line has a closed form (ordinary 2D linear least
squares): for point i belonging to pose p(i), source point s_i (frame L),
camera-line point/normal (l_p, n_p),

    residual_i(t) = n_p . (R(psi) s_i + t - l_p)
                  = [n_p . (R(psi) s_i - l_p)]  +  n_p . t
                  =: c_i(psi)                   +  n_p . t

so t*(psi) = argmin_t sum_i (c_i(psi) + n_p(i) . t)^2, an ordinary least
squares problem in t (2 unknowns), solved directly via
numpy.linalg.lstsq -- no re-implementation of icp_2d's iterative search,
and icp_2d itself is not imported or modified.

This does NOT re-select which points belong to which pose as psi varies
(the correspondence membership is fixed once, from board_correspondences.py,
at the baseline solved T2) -- see report.md for why that is the right
simplification for "does the objective have a flat valley or a sharp
minimum" rather than re-litigating membership at every sweep step.

Read-only against cam_lidar_2d_icp.py / gui.py / icp_2d.py /
tools/lidar_characterisation.

Run from the repository root:
    python tools/lidar_ground_truth/diagnostics/testB_yaw_sweep.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # tools/lidar_ground_truth/
sys.path.insert(0, str(Path(__file__).resolve().parent))       # diagnostics/

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import calibration_io
import config
from board_correspondences import build_all_pose_correspondences

OUT_DIR = config.REPO_ROOT / "results" / "lidar_ground_truth" / "diagnostics" / "testB_yaw_sweep"


def rotation(psi_rad: float) -> np.ndarray:
    c, s = math.cos(psi_rad), math.sin(psi_rad)
    return np.array([[c, -s], [s, c]])


def cost_at_yaw(psi_rad: float, pcs) -> tuple[float, np.ndarray]:
    """Return (RMSE in metres, optimal shared t) at fixed yaw psi_rad."""
    R = rotation(psi_rad)

    N_rows = []
    c_vals = []
    for pc in pcs:
        s = pc.xy_lidar  # (n_i, 2), frame L
        n = pc.camera_line_normal  # (2,)
        l = pc.camera_line_point   # (2,)
        rotated = s @ R.T  # (n_i, 2)
        c_i = (rotated - l) @ n  # (n_i,)
        N_rows.append(np.tile(n, (len(s), 1)))
        c_vals.append(c_i)

    N_mat = np.concatenate(N_rows, axis=0)  # (total, 2)
    c_vec = np.concatenate(c_vals)          # (total,)

    t_opt, *_ = np.linalg.lstsq(N_mat, -c_vec, rcond=None)
    residuals = c_vec + N_mat @ t_opt
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    return rmse, t_opt


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    T2 = calibration_io.load_transform()
    psi0 = math.atan2(T2[1, 0], T2[0, 0])
    print(f"Solved yaw psi0 = {math.degrees(psi0):+.4f} deg")

    pcs = build_all_pose_correspondences()
    total_points = sum(len(pc.xy_lidar) for pc in pcs)
    print(f"Total board-struck points across {len(pcs)} poses: {total_points}")

    offsets_deg = np.arange(-6.0, 6.001, 0.25)
    costs_mm = []
    t_opts = []
    for off_deg in offsets_deg:
        psi = psi0 + math.radians(off_deg)
        rmse, t_opt = cost_at_yaw(psi, pcs)
        costs_mm.append(rmse * 1000.0)
        t_opts.append(t_opt)

    costs_mm = np.array(costs_mm)
    t_opts = np.array(t_opts)

    baseline_idx = int(np.argmin(np.abs(offsets_deg)))
    print()
    print(f"{'yaw offset (deg)':>18} {'abs yaw (deg)':>15} {'RMSE (mm)':>10} {'t_opt':>20}")
    for off, cost, t in zip(offsets_deg, costs_mm, t_opts):
        marker = " <-- baseline" if abs(off) < 1e-9 else ""
        print(f"{off:>18.2f} {math.degrees(psi0)+off:>15.3f} {cost:>10.3f} "
              f"[{t[0]:+.4f},{t[1]:+.4f}]{marker}")

    min_idx = int(np.argmin(costs_mm))
    print()
    print(f"Baseline (offset=0) RMSE: {costs_mm[baseline_idx]:.3f} mm")
    print(f"Global minimum over sweep: offset={offsets_deg[min_idx]:+.2f} deg, "
          f"RMSE={costs_mm[min_idx]:.3f} mm")

    # Two-number comparison requested: cost at psi_solved vs psi_solved + 3.2 deg
    target_offset = 3.2
    closest_idx = int(np.argmin(np.abs(offsets_deg - target_offset)))
    print()
    print(f"Cost at psi_solved            : {costs_mm[baseline_idx]:.3f} mm")
    print(f"Cost at psi_solved + {offsets_deg[closest_idx]:+.2f} deg : {costs_mm[closest_idx]:.3f} mm "
          f"(closest sweep sample to +3.2 deg)")
    print(f"Ratio: {costs_mm[closest_idx] / costs_mm[baseline_idx]:.3f}x")

    # Flatness characterisation: RMSE within +/-10% of the minimum, how wide (deg)?
    tol = 1.10 * costs_mm[min_idx]
    within_tol = offsets_deg[costs_mm <= tol]
    flat_width_deg = float(within_tol.max() - within_tol.min()) if len(within_tol) else 0.0
    print()
    print(f"Width of the offset range within 10% of the minimum RMSE: {flat_width_deg:.2f} deg")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(offsets_deg, costs_mm, "o-", ms=3)
    ax.axvline(0, c="gray", ls=":", label="solved yaw")
    ax.axvline(target_offset, c="red", ls="--", label="+3.2 deg (cancels observed Delta_u bias)")
    ax.set_xlabel("yaw offset from solved value (deg)")
    ax.set_ylabel("RMSE (mm), translation re-optimised at each yaw")
    ax.set_title("Test B: ICP objective vs yaw (translation re-optimised per point)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "cost_vs_yaw.png", dpi=150)
    plt.close(fig)

    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump({
            "psi0_deg": math.degrees(psi0),
            "offsets_deg": offsets_deg.tolist(),
            "costs_mm": costs_mm.tolist(),
            "baseline_cost_mm": float(costs_mm[baseline_idx]),
            "min_cost_mm": float(costs_mm[min_idx]),
            "min_offset_deg": float(offsets_deg[min_idx]),
            "cost_at_plus_3p2_deg": float(costs_mm[closest_idx]),
            "flat_width_deg_within_10pct": flat_width_deg,
        }, f, indent=2)
    print(f"\nSaved {OUT_DIR}")


if __name__ == "__main__":
    main()
