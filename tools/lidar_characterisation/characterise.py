#!/usr/bin/env python3
"""LiDAR (LDROBOT STL-19P) range precision + planarity characterisation.

Headless, fully algorithmic (RANSAC dominant-wall line extraction — no GUI,
no board-specific logic). Reads only from data/<session>/raw_lasers/extracted_pcd/.
Writes all outputs under results/lidar_characterisation/.

Does NOT assess range accuracy (bias) — no independently measured true
distances are available for this capture. See report.md for details.

Run from the repository root:
    python tools/lidar_characterisation/characterise.py

No pandas/tabulate dependency (not declared/installed for this repo) —
CSV via the stdlib csv module, tables via a small manual markdown formatter.
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config
import datasheet
from pcd_io import load_valid_xy
from planarity import analyse_trend
from ransac_line import fit_dominant_wall

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data"
RESULTS_ROOT = REPO_ROOT / "results" / "lidar_characterisation"
PLOTS_ROOT = RESULTS_ROOT / "plots"


def process_pose_scans(pose_dir: Path):
    """Run RANSAC wall-fitting on every raw scan in a pose directory.

    Returns a list of per-scan record dicts and a parallel list of
    (path, xy, LineFit) tuples for accepted scans (for plotting/planarity).
    """
    scan_paths = sorted(pose_dir.glob("*.pcd"))
    records = []
    accepted = []
    rng = np.random.default_rng(config.RANSAC_SEED)
    for path in scan_paths:
        xy, n_total, n_valid = load_valid_xy(path)
        fit = fit_dominant_wall(xy, rng=rng) if n_valid >= config.RANSAC_MIN_INLIERS else None
        rec = {
            "scan_file": path.name,
            "n_total_points": n_total,
            "n_valid_points": n_valid,
            "accepted": fit is not None,
        }
        if fit is not None:
            rec.update({
                "n_inliers": int(fit.inlier_mask.sum()),
                "distance_m": fit.distance_to_origin,
                "residual_std_m": float(fit.residuals.std(ddof=1)),
                "spatial_extent_m": fit.spatial_extent_m,
                "angular_extent_deg": fit.angular_extent_deg,
            })
            accepted.append((path, xy, fit))
        else:
            rec.update({
                "n_inliers": 0,
                "distance_m": "",
                "residual_std_m": "",
                "spatial_extent_m": "",
                "angular_extent_deg": "",
            })
        records.append(rec)
    return records, accepted


def make_wall_fit_plot(session: str, pose: str, xy: np.ndarray, fit, distance_m: float, out_path: Path):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(xy[:, 0], xy[:, 1], s=4, c="lightgray", label="all scan points")
    inl = xy[fit.inlier_mask]
    ax.scatter(inl[:, 0], inl[:, 1], s=6, c="tab:red", label="wall inliers")
    along = np.array([-1.0, 1.0]) * max(fit.spatial_extent_m, 0.3) * 0.6
    line_pts = fit.point + np.outer(along, fit.direction)
    ax.plot(line_pts[:, 0], line_pts[:, 1], c="tab:blue", lw=2, label="fitted wall line")
    ax.scatter([0], [0], marker="^", c="black", s=60, label="sensor origin")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"{session} / {pose} — dominant wall fit (distance={distance_m:.3f} m)")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def make_planarity_plot(session: str, pose: str, angles_rel, residuals_mm, angle_result,
                         along_pos, along_result, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    ax.scatter(angles_rel, residuals_mm, s=4, alpha=0.4, c="tab:gray")
    xs = np.linspace(angles_rel.min(), angles_rel.max(), 200)
    trend = np.polyval(angle_result.coeffs_best, xs) * 1000.0
    ax.plot(xs, trend, c="tab:red", lw=2,
            label=f"deg-{angle_result.best_degree} fit (R2={angle_result.r2_by_degree[angle_result.best_degree]:.3f})")
    ax.axhline(0, c="k", lw=0.8)
    ax.set_xlabel("angle relative to wall centre (deg)")
    ax.set_ylabel("perpendicular residual (mm)")
    ax.set_title("residual vs scan angle")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.scatter(along_pos, residuals_mm, s=4, alpha=0.4, c="tab:gray")
    xs = np.linspace(along_pos.min(), along_pos.max(), 200)
    trend = np.polyval(along_result.coeffs_best, xs) * 1000.0
    ax.plot(xs, trend, c="tab:red", lw=2,
            label=f"deg-{along_result.best_degree} fit (R2={along_result.r2_by_degree[along_result.best_degree]:.3f})")
    ax.axhline(0, c="k", lw=0.8)
    ax.set_xlabel("along-wall position, centred (m)")
    ax.set_ylabel("perpendicular residual (mm)")
    ax.set_title("residual vs along-wall position")
    ax.legend(fontsize=8)

    fig.suptitle(f"{session} / {pose} — planarity / systematic structure")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def make_precision_vs_distance_plot(pose_summary_rows, out_path: Path):
    fig, ax = plt.subplots(figsize=(8, 6))
    for lo, hi, spec_mm in config.DATASHEET_PRECISION_BANDS_MM:
        ax.plot([lo, hi], [spec_mm, spec_mm], "k--", lw=1.5)
        ax.annotate(f"datasheet {spec_mm:.0f} mm", (hi, spec_mm), textcoords="offset points",
                    xytext=(-5, 5), ha="right", fontsize=8)

    markers = {"data_2026-06-28": "o", "data_2026-07-10": "s"}
    sessions = sorted({r["session"] for r in pose_summary_rows})
    for session in sessions:
        rows = [r for r in pose_summary_rows if r["session"] == session]
        dist = [r["mean_distance_m"] for r in rows]
        within = [r["within_scan_sigma_mm"] for r in rows]
        across = [r["across_scan_sigma_mm"] for r in rows]
        m = markers.get(session, "o")
        ax.scatter(dist, within, marker=m, label=f"{session} (within-scan)", s=60)
        ax.scatter(dist, across, marker=m, facecolors="none",
                   edgecolors="tab:red", label=f"{session} (across-scan)", s=60)

    ax.set_xlabel("mean wall distance (m)")
    ax.set_ylabel("measured precision sigma (mm)")
    ax.set_title("Measured precision vs datasheet spec")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def analyse_session(session: str):
    session_dir = DATA_ROOT / session / "raw_lasers" / "extracted_pcd"
    per_scan_rows = []
    pose_summary_rows = []
    plots_dir = PLOTS_ROOT / session
    plots_dir.mkdir(parents=True, exist_ok=True)

    for pose in config.POSES:
        pose_dir = session_dir / pose
        if not pose_dir.is_dir():
            continue
        records, accepted = process_pose_scans(pose_dir)
        for rec in records:
            row = {"session": session, "pose": pose}
            row.update(rec)
            per_scan_rows.append(row)

        if not accepted:
            pose_summary_rows.append({
                "session": session, "pose": pose,
                "n_raw_scans": len(records), "n_accepted_scans": 0,
                "note": "no scan met RANSAC min-inlier/min-extent thresholds",
            })
            continue

        distances = np.array([f.distance_to_origin for _, _, f in accepted])
        resid_stds = np.array([f.residuals.std(ddof=1) for _, _, f in accepted])
        n_inliers = np.array([f.inlier_mask.sum() for _, _, f in accepted])
        mean_distance = float(distances.mean())
        within_scan_sigma_mm = float(resid_stds.mean() * 1000.0)
        within_scan_sigma_mm_median = float(np.median(resid_stds) * 1000.0)
        across_scan_sigma_mm = float(distances.std(ddof=1) * 1000.0) if len(distances) > 1 else float("nan")

        # --- Planarity: aggregate residuals across all accepted scans ---
        all_angles_rel = []
        all_along = []
        all_resid = []
        for _, _, fit in accepted:
            all_angles_rel.append(fit.angles_deg - fit.angles_deg.mean())
            all_along.append(fit.along_line_positions)
            all_resid.append(fit.residuals)
        all_angles_rel = np.concatenate(all_angles_rel)
        all_along = np.concatenate(all_along)
        all_resid = np.concatenate(all_resid)

        angle_result = analyse_trend(all_angles_rel, all_resid)
        along_result = analyse_trend(all_along, all_resid)

        # representative scan (closest to median distance) for the wall-fit diagnostic plot
        median_dist = float(np.median(distances))
        rep_idx = int(np.argmin(np.abs(distances - median_dist)))
        rep_path, rep_xy, rep_fit = accepted[rep_idx]
        make_wall_fit_plot(session, pose, rep_xy, rep_fit, rep_fit.distance_to_origin,
                            plots_dir / f"{pose}_wall_fit.png")
        make_planarity_plot(session, pose, all_angles_rel, all_resid * 1000.0, angle_result,
                             all_along, along_result,
                             plots_dir / f"{pose}_planarity.png")

        spec_mm = datasheet.precision_band_mm(mean_distance)
        across_status = (datasheet.classify(across_scan_sigma_mm, spec_mm)
                         if not np.isnan(across_scan_sigma_mm) else "n/a (only one accepted scan)")
        pose_summary_rows.append({
            "session": session,
            "pose": pose,
            "n_raw_scans": len(records),
            "n_accepted_scans": len(accepted),
            "mean_inliers_per_scan": float(n_inliers.mean()),
            "mean_distance_m": mean_distance,
            "within_scan_sigma_mm": within_scan_sigma_mm,
            "within_scan_sigma_mm_median": within_scan_sigma_mm_median,
            "across_scan_sigma_mm": across_scan_sigma_mm,
            "planarity_r2_linear": angle_result.r2_by_degree.get(1),
            "planarity_r2_quad": angle_result.r2_by_degree.get(2),
            "planarity_r2_cubic": angle_result.r2_by_degree.get(3),
            "planarity_max_dev_mm_vs_angle": angle_result.max_abs_deviation_mm,
            "planarity_max_dev_mm_vs_position": along_result.max_abs_deviation_mm,
            "datasheet_precision_spec_mm": spec_mm,
            "within_scan_status": datasheet.classify(within_scan_sigma_mm, spec_mm),
            "across_scan_status": across_status,
            "representative_scan_file": rep_path.name,
        })

    return per_scan_rows, pose_summary_rows


def write_csv(rows, out_path: Path):
    if not rows:
        out_path.write_text("")
        return
    fieldnames = []
    for r in rows:
        for k in r.keys():
            if k not in fieldnames:
                fieldnames.append(k)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _fmt(v, floatfmt=".3f"):
    if isinstance(v, float):
        if np.isnan(v):
            return "nan"
        return format(v, floatfmt)
    if v is None:
        return ""
    return str(v)


def markdown_table(rows, columns, floatfmt=".3f") -> str:
    present_cols = [c for c in columns if any(c in r for r in rows)]
    header = "| " + " | ".join(present_cols) + " |"
    sep = "| " + " | ".join(["---"] * len(present_cols)) + " |"
    body_lines = []
    for r in rows:
        body_lines.append("| " + " | ".join(_fmt(r.get(c, ""), floatfmt) for c in present_cols) + " |")
    return "\n".join([header, sep] + body_lines)


def write_report(pose_summary_rows, out_path: Path):
    lines = []
    lines.append("# LiDAR (STL-19P) Range Precision & Planarity Characterisation\n")
    lines.append("Headless, algorithmic RANSAC wall-segmentation of raw 2D LiDAR scans. "
                 "No GUI, no board-specific logic — the dominant straight planar surface "
                 "(largest RANSAC inlier set meeting the thresholds below) in each scan is "
                 "treated as \"the wall\".\n")
    lines.append("**Scope**: range PRECISION (noise) and PLANARITY only. "
                 "Range ACCURACY (bias vs. true distance) is NOT assessed — no independently "
                 "measured true distances were available for this capture. Deferred to a later "
                 "capture with a laser distance measurer.\n")

    lines.append("## Config constants used\n")
    lines.append(f"- RANSAC inlier distance threshold: {config.RANSAC_DIST_THRESHOLD_M*1000:.0f} mm")
    lines.append(f"- RANSAC minimum inliers: {config.RANSAC_MIN_INLIERS}")
    lines.append(f"- RANSAC minimum spatial extent: {config.RANSAC_MIN_SPATIAL_EXTENT_M} m")
    lines.append(f"- RANSAC minimum angular extent: {config.RANSAC_MIN_ANGULAR_EXTENT_DEG} deg")
    lines.append(f"- RANSAC iterations / refine rounds: {config.RANSAC_N_ITERS} / {config.RANSAC_REFINE_ROUNDS}")
    lines.append(f"- Invalid-point tolerance (dropped (0,0) returns): {config.INVALID_POINT_TOL_M} m")
    lines.append(f"- Datasheet precision bands (mm): {config.DATASHEET_PRECISION_BANDS_MM}\n")

    lines.append("## Per-pose results\n")
    cols = ["session", "pose", "n_raw_scans", "n_accepted_scans", "mean_distance_m",
            "within_scan_sigma_mm", "across_scan_sigma_mm", "datasheet_precision_spec_mm",
            "within_scan_status", "across_scan_status"]
    lines.append(markdown_table(pose_summary_rows, cols))
    lines.append("")

    lines.append("\n## Planarity (systematic structure) per pose\n")
    cols2 = ["session", "pose", "planarity_r2_linear", "planarity_r2_quad", "planarity_r2_cubic",
             "planarity_max_dev_mm_vs_angle", "planarity_max_dev_mm_vs_position"]
    lines.append(markdown_table(pose_summary_rows, cols2))
    lines.append("\nR^2 values are variance-explained of a degree-d polynomial fit to "
                 "(residual vs angle) / (residual vs along-wall position) relative to a flat "
                 "(mean-only) baseline, aggregated over all accepted scans per pose. Higher R^2 "
                 "indicates more systematic (non-random) structure in the residuals rather than "
                 "pure noise scatter.\n")

    lines.append("## Notes\n")
    lines.append("- `within_scan_sigma_mm`: mean, over all accepted scans in the pose, of the "
                 "STD of perpendicular residuals of inlier points about that scan's fitted wall "
                 "line (spatial noise across beams within one sweep).\n")
    lines.append("- `across_scan_sigma_mm`: STD, across accepted scans of the same static pose, "
                 "of each scan's fitted wall distance (temporal repeatability of the aggregate "
                 "distance estimate). Because this is a STD of an average over many inliers per "
                 "scan, it is expected to be smaller than single-beam noise and is not directly "
                 "the same quantity as `within_scan_sigma_mm`.\n")
    lines.append("- Status classification: within spec if measured < 0.9x datasheet value, "
                 "\"at spec\" if within 0.9x-1.1x, \"worse than spec\" if > 1.1x.\n")
    lines.append("- Scans that failed the RANSAC min-inlier/min-extent thresholds were excluded "
                 "and are counted in `n_raw_scans - n_accepted_scans`; see `per_scan_details.csv` "
                 "for the full per-scan breakdown.\n")
    lines.append("- CAVEAT on `planarity_max_dev_mm_vs_position`: for several poses this value is "
                 "driven by a small number of sparse points at the extreme ends of the along-wall "
                 "domain (visible as thin vertical outlier stripes in the `*_planarity.png` plots), "
                 "which are plausibly RANSAC inliers picked up near a room corner rather than the "
                 "main wall, combined with degree-3 polynomial extrapolation past the dense core. "
                 "This inflates the reported max deviation beyond what the dense central part of the "
                 "wall actually shows. Treat `planarity_r2_*` and the vs-angle columns, together with "
                 "visual inspection of the diagnostic plots, as more reliable planarity indicators "
                 "than this column in isolation. Not silently corrected — flagged here per instruction.\n")

    out_path.write_text("\n".join(lines))


def main():
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    PLOTS_ROOT.mkdir(parents=True, exist_ok=True)

    all_per_scan = []
    all_pose_summary = []
    for session in config.DATA_SESSIONS:
        per_scan_rows, pose_summary_rows = analyse_session(session)
        all_per_scan.extend(per_scan_rows)
        all_pose_summary.extend(pose_summary_rows)

    write_csv(all_per_scan, RESULTS_ROOT / "per_scan_details.csv")
    write_csv(all_pose_summary, RESULTS_ROOT / "summary.csv")

    valid_summary = [r for r in all_pose_summary if "mean_distance_m" in r]
    if valid_summary:
        make_precision_vs_distance_plot(valid_summary, RESULTS_ROOT / "precision_vs_distance.png")

    write_report(all_pose_summary, RESULTS_ROOT / "report.md")

    print("Wrote:")
    for p in sorted(RESULTS_ROOT.rglob("*")):
        if p.is_file():
            print(" -", p.relative_to(REPO_ROOT))


if __name__ == "__main__":
    main()
