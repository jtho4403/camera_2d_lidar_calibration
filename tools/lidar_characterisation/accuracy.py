#!/usr/bin/env python3
"""LiDAR (LDROBOT STL-19P) range ACCURACY (bias vs true distance) + extended
precision-vs-distance characterisation.

Headless, fully algorithmic. Extends tools/lidar_characterisation/characterise.py
(reuses pcd_io, ransac_line, planarity's datasheet helpers, and characterise's
CSV/markdown/plot helpers) rather than duplicating it. Does NOT modify the
existing calibration pipeline or the earlier precision-only characterisation.

Target-wall selection is done OBJECTIVELY BY KNOWN DISTANCE, not by assumed
bearing: for each scan we search for multiple candidate wall segments (not
just the single dominant one) and pick whichever candidate's perpendicular
distance from the sensor origin falls within a tolerance gate of that pose's
rangefinder-measured true distance. Scans with no such candidate are flagged
"target not found" and excluded from aggregate stats.

Run from the repository root:
    python tools/lidar_characterisation/accuracy.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config
import datasheet
from pcd_io import load_valid_xy
from ransac_line import LineFit, find_candidate_walls

# Reuse existing CSV/markdown/plot helpers rather than duplicating them.
from characterise import (
    REPO_ROOT,
    write_csv,
    markdown_table,
    make_precision_vs_distance_plot,
)

DATA_DIR = REPO_ROOT / "data" / config.ACCURACY_SESSION / "extracted_pcd"
RESULTS_ROOT = REPO_ROOT / "results" / "lidar_characterisation" / "accuracy"
PLOTS_ROOT = RESULTS_ROOT / "plots"

CANDIDATE_COLORS = ["tab:blue", "tab:green", "tab:orange", "tab:purple",
                     "tab:brown", "tab:pink", "tab:olive", "tab:cyan"]


def target_bearing_deg(fit: LineFit) -> float:
    """Bearing (deg, sensor-native atan2(y,x) frame) of the perpendicular foot
    from the sensor origin onto the fitted line — i.e. the direction "straight
    at" the wall, independent of which inlier points happened to be sampled.
    """
    signed_d = float(fit.point @ fit.normal)
    foot = signed_d * fit.normal
    return float(np.degrees(np.arctan2(foot[1], foot[0])))


def circular_mean_std_deg(angles_deg: np.ndarray) -> tuple[float, float]:
    """Circular mean and STD (deg), robust to a +-180 deg wraparound."""
    mean = float(np.degrees(np.arctan2(
        np.mean(np.sin(np.radians(angles_deg))),
        np.mean(np.cos(np.radians(angles_deg))),
    )))
    unwrapped = mean + (((angles_deg - mean + 180.0) % 360.0) - 180.0)
    std = float(unwrapped.std(ddof=1)) if len(unwrapped) > 1 else float("nan")
    return mean, std


def select_target_candidate(candidates, true_distance_m: float, tolerance_m: float):
    """Among candidates within the tolerance gate, return the one closest to
    true_distance_m, its distance error, and how many candidates qualified
    (or (None, None, 0) if none did).

    A qualifying-count > 1 flags scans where a second real surface in the
    room happened to sit within the tolerance band at roughly the same
    distance as the true target (an inherent ambiguity of selecting by
    distance alone) -- see report.md for a concrete case (pose_05).
    """
    in_tol = [(c, abs(c.distance_to_origin - true_distance_m)) for c in candidates]
    in_tol = [(c, diff) for c, diff in in_tol if diff <= tolerance_m]
    if not in_tol:
        return None, None, 0
    best, best_diff = min(in_tol, key=lambda cd: cd[1])
    return best, best_diff, len(in_tol)


def process_pose_scans(pose_dir: Path, true_distance_m: float):
    """Run multi-candidate RANSAC + target-by-distance selection on every raw
    scan in a pose directory.

    Returns (records, accepted, debug_scan) where:
      - records: per-scan result dicts (inventory + selection outcome)
      - accepted: list of (path, xy, candidates, selected_fit) for scans where
        the target wall was found
      - debug_scan: (path, xy, candidates, selected_fit_or_None) for the FIRST
        scan processed, kept even if nothing was accepted, so a selection
        diagnostic plot can always be produced for the pose.
    """
    tolerance_m = max(config.ACCURACY_TOLERANCE_MIN_M,
                       config.ACCURACY_TOLERANCE_FRAC * true_distance_m)
    scan_paths = sorted(pose_dir.glob("*.pcd"))
    rng = np.random.default_rng(config.RANSAC_SEED)
    records = []
    accepted = []
    debug_scan = None

    for path in scan_paths:
        xy, n_total, n_valid = load_valid_xy(path)
        candidates = (find_candidate_walls(xy, rng=rng, max_candidates=config.ACCURACY_MAX_CANDIDATES,
                                            min_inliers=config.ACCURACY_RANSAC_MIN_INLIERS)
                      if n_valid >= config.ACCURACY_RANSAC_MIN_INLIERS else [])
        selected, diff_m, n_in_tolerance = select_target_candidate(candidates, true_distance_m, tolerance_m)

        rec = {
            "pose": pose_dir.name,
            "scan_file": path.name,
            "n_total_points": n_total,
            "n_valid_points": n_valid,
            "dropout_fraction": (n_total - n_valid) / n_total if n_total else float("nan"),
            "n_candidates_found": len(candidates),
            "n_candidates_in_tolerance": n_in_tolerance,
            "ambiguous_selection": n_in_tolerance > 1,
            "target_found": selected is not None,
        }
        if selected is not None:
            rec.update({
                "selected_distance_m": selected.distance_to_origin,
                "distance_error_from_true_m": diff_m,
                "target_bearing_deg": target_bearing_deg(selected),
                "n_inliers": int(selected.inlier_mask.sum()),
            })
            accepted.append((path, xy, candidates, selected))
        else:
            rec.update({
                "selected_distance_m": "",
                "distance_error_from_true_m": "",
                "target_bearing_deg": "",
                "n_inliers": "",
            })
        records.append(rec)

        if debug_scan is None:
            debug_scan = (path, xy, candidates, selected)

    return records, accepted, tolerance_m, debug_scan


def make_selection_plot(pose: str, xy: np.ndarray, candidates, selected, true_distance_m: float,
                         tolerance_m: float, out_path: Path):
    """Full 360 deg scan + every candidate segment + the selected target wall
    (if any) highlighted, so target selection is human-verifiable by eye.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(xy[:, 0], xy[:, 1], s=3, c="lightgray", label="all valid points", zorder=1)

    for i, c in enumerate(candidates):
        is_selected = selected is not None and c is selected
        color = "black" if is_selected else CANDIDATE_COLORS[i % len(CANDIDATE_COLORS)]
        inl = xy[c.inlier_mask]
        ax.scatter(inl[:, 0], inl[:, 1], s=8,
                   c=color, alpha=(1.0 if is_selected else 0.6),
                   zorder=3 if is_selected else 2)
        along = np.array([-1.0, 1.0]) * max(c.spatial_extent_m, 0.3) * 0.6
        line_pts = c.point + np.outer(along, c.direction)
        lw = 3.0 if is_selected else 1.3
        label = (f"SELECTED target (d={c.distance_to_origin:.3f} m)" if is_selected
                 else f"candidate {i} (d={c.distance_to_origin:.3f} m)")
        ax.plot(line_pts[:, 0], line_pts[:, 1], c=color, lw=lw, label=label, zorder=3 if is_selected else 2)

    # Tolerance-gate reference: annulus of acceptable distances around origin.
    theta = np.linspace(0, 2 * np.pi, 200)
    for r, ls in [(true_distance_m - tolerance_m, ":"), (true_distance_m, "--"), (true_distance_m + tolerance_m, ":")]:
        if r > 0:
            ax.plot(r * np.cos(theta), r * np.sin(theta), c="tab:red", lw=0.8, ls=ls, alpha=0.6)

    ax.scatter([0], [0], marker="^", c="black", s=80, label="sensor origin", zorder=4)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    status = "target FOUND" if selected is not None else "target NOT FOUND"
    ax.set_title(f"{pose} — target selection ({status})\n"
                 f"true={true_distance_m:.3f} m, tolerance=+-{tolerance_m*1000:.0f} mm, "
                 f"{len(candidates)} candidate(s) found")
    ax.legend(loc="best", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def make_bias_vs_distance_plot(pose_summary_rows, out_path: Path):
    fig, ax = plt.subplots(figsize=(8, 6))

    for lo, hi, spec_mm in config.DATASHEET_ACCURACY_BANDS_MM:
        ax.fill_between([lo, hi], [-spec_mm, -spec_mm], [spec_mm, spec_mm],
                        color="tab:red", alpha=0.08)
        ax.plot([lo, hi], [spec_mm, spec_mm], "r--", lw=1.2)
        ax.plot([lo, hi], [-spec_mm, -spec_mm], "r--", lw=1.2)
        ax.annotate(f"+-{spec_mm:.0f} mm datasheet", (hi, spec_mm), textcoords="offset points",
                    xytext=(-5, 5), ha="right", fontsize=7, color="tab:red")

    u = config.ACCURACY_MEASUREMENT_UNCERTAINTY_MM
    xs_all = [r["true_distance_m"] for r in pose_summary_rows]
    ax.fill_between([min(xs_all) - 0.1, max(xs_all) + 0.1], [-u, -u], [u, u],
                    color="gray", alpha=0.25, label=f"+-{u:.0f} mm measurement-uncertainty caveat")

    ax.axhline(0, c="k", lw=1.0)
    xs = [r["true_distance_m"] for r in pose_summary_rows]
    ys = [r["bias_mm"] for r in pose_summary_rows]
    yerr = [r["across_scan_sigma_mm"] for r in pose_summary_rows]
    ax.errorbar(xs, ys, yerr=yerr, fmt="o", c="tab:blue", capsize=3, label="measured bias (+-1 sigma across-scan)")

    ax.set_xlabel("true distance (m)")
    ax.set_ylabel("bias = measured_mean - true (mm)")
    ax.set_title("Range accuracy: bias vs true distance")
    ax.legend(fontsize=7, loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def make_measured_vs_true_plot(pose_summary_rows, out_path: Path):
    fig, ax = plt.subplots(figsize=(7, 7))
    xs = np.array([r["true_distance_m"] for r in pose_summary_rows])
    ys = np.array([r["measured_distance_mean_m"] for r in pose_summary_rows])

    lo, hi = xs.min() - 0.2, xs.max() + 0.2
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.2, label="y = x (ideal)")

    slope, intercept = np.polyfit(xs, ys, 1)
    xs_fit = np.array([lo, hi])
    ax.plot(xs_fit, slope * xs_fit + intercept, c="tab:red", lw=1.5,
            label=f"linear fit: slope={slope:.4f}, intercept={intercept*1000:.1f} mm")

    ax.scatter(xs, ys, c="tab:blue", s=50, zorder=3, label="measured mean")
    ax.set_xlabel("true distance (m)")
    ax.set_ylabel("measured mean distance (m)")
    ax.set_title("Measured vs true distance")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return float(slope), float(intercept)


def analyse_pose(pose: str, true_distance_m: float):
    pose_dir = DATA_DIR / pose
    records, accepted, tolerance_m, debug_scan = process_pose_scans(pose_dir, true_distance_m)
    n_raw = len(records)
    n_accepted = len(accepted)
    n_flagged = n_raw - n_accepted
    n_ambiguous = sum(1 for r in records if r["ambiguous_selection"])

    plots_dir = PLOTS_ROOT
    plots_dir.mkdir(parents=True, exist_ok=True)
    if accepted:
        distances = np.array([f.distance_to_origin for _, _, _, f in accepted])
        median_dist = float(np.median(distances))
        rep_idx = int(np.argmin(np.abs(distances - median_dist)))
        rep_path, rep_xy, rep_candidates, rep_selected = accepted[rep_idx]
        make_selection_plot(pose, rep_xy, rep_candidates, rep_selected, true_distance_m, tolerance_m,
                            plots_dir / f"{pose}_selection.png")
    elif debug_scan is not None:
        dbg_path, dbg_xy, dbg_candidates, dbg_selected = debug_scan
        make_selection_plot(pose, dbg_xy, dbg_candidates, dbg_selected, true_distance_m, tolerance_m,
                            plots_dir / f"{pose}_selection.png")

    inv_n_total = np.array([r["n_total_points"] for r in records], dtype=float)
    inv_n_valid = np.array([r["n_valid_points"] for r in records], dtype=float)
    inventory = {
        "pose": pose,
        "n_raw_scans": n_raw,
        "mean_n_total_points": float(inv_n_total.mean()) if n_raw else float("nan"),
        "mean_n_valid_points": float(inv_n_valid.mean()) if n_raw else float("nan"),
        "mean_dropout_fraction": float(np.mean([r["dropout_fraction"] for r in records])) if n_raw else float("nan"),
    }

    if not accepted:
        summary = {
            "pose": pose,
            "true_distance_m": true_distance_m,
            "tolerance_m": tolerance_m,
            "n_raw_scans": n_raw,
            "n_accepted_scans": 0,
            "n_flagged_target_not_found": n_flagged,
            "n_ambiguous_scans": n_ambiguous,
            "note": "target wall not found in ANY scan for this pose",
        }
        return records, inventory, summary

    distances = np.array([f.distance_to_origin for _, _, _, f in accepted])
    resid_stds = np.array([f.residuals.std(ddof=1) for _, _, _, f in accepted])
    bearings = np.array([target_bearing_deg(f) for _, _, _, f in accepted])

    # Flag scans whose selected bearing deviates from this pose's own median
    # bearing -- a much more precise "actually got the wrong surface" signal
    # than `ambiguous_selection` alone (most ambiguous scans still pick the
    # correct candidate; this catches the ones that didn't).
    median_bearing = float(np.median(bearings))
    outlier_thresh = config.ACCURACY_BEARING_OUTLIER_THRESHOLD_DEG
    is_outlier = np.abs(bearings - median_bearing) > outlier_thresh
    accepted_files = {path.name for path, _, _, _ in accepted}
    outlier_files = {accepted[i][0].name for i in range(len(accepted)) if is_outlier[i]}
    for rec in records:
        if rec["scan_file"] in accepted_files:
            rec["bearing_outlier"] = rec["scan_file"] in outlier_files
    n_bearing_outliers = int(is_outlier.sum())

    measured_mean_m = float(distances.mean())
    measured_std_mm = float(distances.std(ddof=1) * 1000.0) if len(distances) > 1 else float("nan")
    bias_mm = (measured_mean_m - true_distance_m) * 1000.0
    within_scan_sigma_mm = float(resid_stds.mean() * 1000.0)
    across_scan_sigma_mm = measured_std_mm

    if n_bearing_outliers:
        clean = ~is_outlier
        measured_mean_m_excl = float(distances[clean].mean())
        bias_mm_excl = (measured_mean_m_excl - true_distance_m) * 1000.0
    else:
        measured_mean_m_excl = measured_mean_m
        bias_mm_excl = bias_mm

    bearing_mean_deg, bearing_std_deg = circular_mean_std_deg(bearings)

    precision_spec_mm = datasheet.precision_band_mm(measured_mean_m)
    accuracy_spec_mm = datasheet.accuracy_band_mm(true_distance_m)

    within_uncertainty = abs(bias_mm) <= config.ACCURACY_MEASUREMENT_UNCERTAINTY_MM
    if accuracy_spec_mm is None:
        accuracy_status = "no datasheet spec at this distance"
    elif abs(bias_mm) > accuracy_spec_mm:
        accuracy_status = "beyond datasheet accuracy band"
    else:
        accuracy_status = "within datasheet accuracy band"

    summary = {
        "pose": pose,
        "true_distance_m": true_distance_m,
        "tolerance_m": tolerance_m,
        "n_raw_scans": n_raw,
        "n_accepted_scans": n_accepted,
        "n_flagged_target_not_found": n_flagged,
        "n_ambiguous_scans": n_ambiguous,
        "n_bearing_outlier_scans": n_bearing_outliers,
        "measured_distance_mean_m": measured_mean_m,
        "measured_distance_std_mm": measured_std_mm,
        "bias_mm": bias_mm,
        "measured_distance_mean_m_excl_bearing_outliers": measured_mean_m_excl,
        "bias_mm_excl_bearing_outliers": bias_mm_excl,
        "within_measurement_uncertainty_band": within_uncertainty,
        "accuracy_status_vs_datasheet": accuracy_status,
        "datasheet_accuracy_spec_mm": accuracy_spec_mm,
        "within_scan_sigma_mm": within_scan_sigma_mm,
        "across_scan_sigma_mm": across_scan_sigma_mm,
        "datasheet_precision_spec_mm": precision_spec_mm,
        "precision_status_vs_datasheet": datasheet.classify(within_scan_sigma_mm, precision_spec_mm),
        "target_bearing_mean_deg": bearing_mean_deg,
        "target_bearing_std_deg": bearing_std_deg,
        "mean_distance_m": measured_mean_m,  # alias for make_precision_vs_distance_plot() reuse
        "session": config.ACCURACY_SESSION,   # alias for make_precision_vs_distance_plot() reuse
    }
    return records, inventory, summary


def main():
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    PLOTS_ROOT.mkdir(parents=True, exist_ok=True)

    all_records = []
    inventory_rows = []
    pose_summary_rows = []

    poses = sorted(config.ACCURACY_TRUE_DISTANCES_M.keys())
    for pose in poses:
        true_distance_m = config.ACCURACY_TRUE_DISTANCES_M[pose]
        pose_dir = DATA_DIR / pose
        if not pose_dir.is_dir():
            print(f"WARNING: expected pose dir not found, skipping: {pose_dir}")
            continue
        records, inventory, summary = analyse_pose(pose, true_distance_m)
        all_records.extend(records)
        inventory_rows.append(inventory)
        pose_summary_rows.append(summary)

    write_csv(all_records, RESULTS_ROOT / "per_scan_details.csv")
    write_csv(pose_summary_rows, RESULTS_ROOT / "summary.csv")

    valid_rows = [r for r in pose_summary_rows if "measured_distance_mean_m" in r]
    slope, intercept = (float("nan"), float("nan"))
    if valid_rows:
        make_bias_vs_distance_plot(valid_rows, RESULTS_ROOT / "bias_vs_distance.png")
        slope, intercept = make_measured_vs_true_plot(valid_rows, RESULTS_ROOT / "measured_vs_true.png")
        make_precision_vs_distance_plot(valid_rows, RESULTS_ROOT / "precision_vs_distance.png")


    print("Wrote:")
    for p in sorted(RESULTS_ROOT.rglob("*")):
        if p.is_file():
            print(" -", p.relative_to(REPO_ROOT))


if __name__ == "__main__":
    main()
