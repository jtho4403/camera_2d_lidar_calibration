#!/usr/bin/env python3
"""Phase 1.5 Step 3: quantify the horizontal (column) misalignment reported
on visual inspection of the Phase 1 overlays.

For each pose in data/data_2026-06-28/:
  1. Detect LiDAR range discontinuities (candidate vertical-edge features,
     e.g. cabinet/box/chair-post silhouette boundaries) in the bearing-
     ordered scan, and note each discontinuity's projected (u, v, depth).
  2. Detect vertical image edges via a Sobel-x gradient, restricted to a row
     band around each candidate's projected v, and match the nearest strong
     edge within a search window of the candidate's projected u.
  3. Record signed Delta_u = u_projected - u_image_edge, camera_depth_m,
     u_projected, pose_id for every match.
  4. Merge in any hand-annotated corrections from --manual-edges (a JSON
     file with the same schema as the auto-generated
     <out>/manual_edges_template.json -- entries there are ADDED to, not
     silently replacing, the automatic matches, and are tagged
     source="manual" in every output so they're distinguishable).
  5. Plot Delta_u vs Z, Delta_u vs (u - c_x), Delta_u vs pose id.
  6. Fit the signature-table models (constant / 1/Z / linear in (u-c_x) /
     quadratic in (u-c_x) / the "2*(c_x-u)" handedness-flip line) and report
     which best explains the data, plus the pose_01-vs-pose_03 comparison.

Read-only against cam_lidar_2d_icp.py / gui.py / icp_2d.py /
tools/lidar_characterisation. Uses the existing (unmodified)
lidar_to_camera_2d.npy / calibration_result.json and this package's own
Phase 1 modules only.

Run from the repository root:
    python tools/lidar_ground_truth/diagnostics/step3_horizontal_offset.py
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # tools/lidar_ground_truth/

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

import calibration_io
import config
import projection
import rig as rig_module
import scan_io

REPO_ROOT = config.REPO_ROOT
SESSION_DIR = REPO_ROOT / "data" / "data_2026-06-28"
OUT_DIR = REPO_ROOT / "results" / "lidar_ground_truth" / "diagnostics" / "step3_horizontal_offset"

RANGE_JUMP_THRESHOLD_M = 0.10   # candidate LiDAR discontinuity, deliberately looser than filters.py's 0.15 to surface more features for matching
DEDUPE_MIN_SEP_PX = 8.0         # merge candidates whose projected u are this close (thin-object double-edge spikes)
MAX_CANDIDATES_PER_POSE = 8
EDGE_SEARCH_RADIUS_PX = 150.0   # generous vs the ~70 px eyeballed shift
EDGE_ROW_BAND_PX = 20.0
MIN_CONFIDENCE = 3.0            # peak / median gradient in the search window


@dataclass
class EdgeMatch:
    pose_id: str
    source: str                 # "auto" or "manual"
    u_lidar_px: float
    v_lidar_px: float
    depth_m: float
    u_image_edge_px: float
    delta_u_px: float
    confidence: float
    jump_m: float | None = None
    note: str = ""


def pose_id_from_filename(path: Path) -> str:
    match = re.search(r"pose_(\d+)", path.stem)
    return f"pose_{int(match.group(1)):02d}"


def detect_lidar_discontinuities(xy_bearing_ordered, u, v, depth, keep, jump_threshold_m):
    range_m = np.hypot(xy_bearing_ordered[:, 0], xy_bearing_ordered[:, 1])
    n = len(range_m)
    candidates = []
    if n < 2:
        return candidates
    diffs = np.abs(np.diff(range_m))
    for idx in np.flatnonzero(diffs > jump_threshold_m):
        i_a, i_b = idx, idx + 1
        i_near, i_far = (i_a, i_b) if range_m[i_a] < range_m[i_b] else (i_b, i_a)
        for i, side in ((i_near, "near"), (i_far, "far")):
            if keep[i]:
                candidates.append({
                    "lidar_index": int(i),
                    "u_lidar": float(u[i]),
                    "v_lidar": float(v[i]),
                    "depth_m": float(depth[i]),
                    "jump_m": float(diffs[idx]),
                    "side": side,
                })
                break
    return candidates


def dedupe_by_u(candidates, min_sep_px):
    for_sort = sorted(candidates, key=lambda c: -c["jump_m"])
    kept = []
    for c in for_sort:
        if all(abs(c["u_lidar"] - k["u_lidar"]) > min_sep_px for k in kept):
            kept.append(c)
    return sorted(kept, key=lambda c: -c["jump_m"])[:MAX_CANDIDATES_PER_POSE]


def match_to_image_edge(grad_mag, u_lidar, v_lidar, search_radius_px, row_band_px,
                         min_prominence_factor=2.5):
    """Match the NEAREST sufficiently-strong vertical-edge peak to u_lidar,
    rather than the single strongest edge anywhere in the search window.

    A naive "argmax gradient in a wide window" is easily pulled onto a
    strong but unrelated edge elsewhere in a cluttered scene (chair frame,
    cables, a person) well outside the search radius' intended tolerance for
    the true corresponding edge's own position noise. Restricting to local
    peaks above an image-wide strength floor, then picking the nearest one
    to u_lidar, is a much closer match to "the edge this LiDAR discontinuity
    actually corresponds to".
    """
    H, W = grad_mag.shape
    row_lo = int(max(0, v_lidar - row_band_px))
    row_hi = int(min(H, v_lidar + row_band_px))
    if row_hi <= row_lo:
        return None
    col_profile = grad_mag[row_lo:row_hi, :].sum(axis=0)

    # Absolute strength floor from the FULL row band (not just the search
    # window), so the threshold doesn't shift depending on how much
    # unrelated texture happens to fall inside a given candidate's window.
    floor = float(np.median(col_profile)) * min_prominence_factor + 1e-6

    peak_idx, props = find_peaks(col_profile, height=floor, distance=5)
    if len(peak_idx) == 0:
        return None

    u_lo = u_lidar - search_radius_px
    u_hi = u_lidar + search_radius_px
    in_window = (peak_idx >= u_lo) & (peak_idx <= u_hi)
    if not np.any(in_window):
        return None

    candidate_idx = peak_idx[in_window]
    candidate_heights = props["peak_heights"][in_window]
    nearest = int(np.argmin(np.abs(candidate_idx - u_lidar)))
    u_edge = float(candidate_idx[nearest])
    peak_height = float(candidate_heights[nearest])

    return {"u_image_edge": u_edge, "confidence": peak_height / floor}


def process_pose(image_path, laser_path, metadata_path, T2, rig_cfg) -> list[EdgeMatch]:
    pose_id = pose_id_from_filename(image_path)
    image_bgr = cv2.imread(str(image_path))
    h, w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    grad_mag = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))

    intrinsics = calibration_io.load_intrinsics_from_metadata(metadata_path)
    xy_valid, _, _ = scan_io.load_valid_xy(laser_path)
    result = projection.project_scan(
        xy_valid, T2, rig_cfg.delta_z_m, intrinsics.K, (w, h), z_min_m=config.Z_MIN_M,
    )

    raw_candidates = detect_lidar_discontinuities(
        xy_valid, result.u, result.v, result.depth_m, result.keep, RANGE_JUMP_THRESHOLD_M
    )
    candidates = dedupe_by_u(raw_candidates, DEDUPE_MIN_SEP_PX)

    matches = []
    for c in candidates:
        edge = match_to_image_edge(
            grad_mag, c["u_lidar"], c["v_lidar"], EDGE_SEARCH_RADIUS_PX, EDGE_ROW_BAND_PX
        )
        if edge is None:
            continue
        delta_u = c["u_lidar"] - edge["u_image_edge"]
        matches.append(EdgeMatch(
            pose_id=pose_id,
            source="auto",
            u_lidar_px=c["u_lidar"],
            v_lidar_px=c["v_lidar"],
            depth_m=c["depth_m"],
            u_image_edge_px=edge["u_image_edge"],
            delta_u_px=delta_u,
            confidence=edge["confidence"],
            jump_m=c["jump_m"],
            note="low_confidence" if edge["confidence"] < MIN_CONFIDENCE else "",
        ))

    make_pose_overlay(image_bgr, matches, intrinsics.K[0, 2], pose_id,
                       OUT_DIR / f"{pose_id}_edges.png")
    return matches


def make_pose_overlay(image_bgr, matches, cx, pose_id, out_path):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.imshow(image_rgb)
    for m in matches:
        color = "red" if m.source == "auto" else "cyan"
        ax.axvline(m.u_lidar_px, color=color, lw=1.0, ls="--", alpha=0.7)
        ax.axvline(m.u_image_edge_px, color="lime", lw=1.0, ls="-", alpha=0.7)
        ax.plot([m.u_lidar_px, m.u_image_edge_px], [m.v_lidar_px, m.v_lidar_px],
                color="yellow", lw=1.5)
        ax.annotate(f"{m.delta_u_px:+.0f}px", (m.u_lidar_px, m.v_lidar_px),
                    textcoords="offset points", xytext=(0, -12), color="white",
                    fontsize=8, ha="center",
                    bbox=dict(boxstyle="round,pad=0.1", fc="black", alpha=0.5))
    ax.axvline(cx, color="white", lw=1.0, ls=":", alpha=0.6, label="c_x")
    ax.set_title(
        f"{pose_id}: LiDAR-projected edge (red dashed) vs matched image edge "
        f"(green solid), N={len(matches)}"
    )
    ax.set_xlim(0, image_bgr.shape[1])
    ax.set_ylim(image_bgr.shape[0], 0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def load_manual_edges(path: Path | None) -> list[EdgeMatch]:
    if path is None or not Path(path).is_file():
        return []
    raw = json.loads(Path(path).read_text())
    out = []
    for pose_id, entries in raw.items():
        for e in entries:
            out.append(EdgeMatch(
                pose_id=pose_id,
                source="manual",
                u_lidar_px=float(e["u_lidar_px"]),
                v_lidar_px=float(e.get("v_lidar_px", float("nan"))),
                depth_m=float(e["depth_m"]),
                u_image_edge_px=float(e["u_image_edge_px"]),
                delta_u_px=float(e["u_lidar_px"]) - float(e["u_image_edge_px"]),
                confidence=float("inf"),
                note=e.get("note", "manual"),
            ))
    return out


def write_manual_template(matches: list[EdgeMatch], out_path: Path) -> None:
    """Pre-populate an editable manual-override template with the automatic
    matches, grouped by pose, so hand corrections can be added/edited.
    """
    template: dict[str, list[dict]] = {}
    for m in matches:
        template.setdefault(m.pose_id, []).append({
            "u_lidar_px": round(m.u_lidar_px, 2),
            "v_lidar_px": round(m.v_lidar_px, 2),
            "depth_m": round(m.depth_m, 4),
            "u_image_edge_px": round(m.u_image_edge_px, 2),
            "note": f"auto (confidence={m.confidence:.1f}); edit u_image_edge_px and set note to override",
        })
    out_path.write_text(json.dumps(template, indent=2))


def fit_models(delta_u, depth_m, u_minus_cx):
    models = {}

    # Constant (yaw signature)
    c = float(np.mean(delta_u))
    resid = delta_u - c
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((delta_u - np.mean(delta_u)) ** 2)) or 1e-12
    models["constant"] = {"c_px": c, "r2": 1 - ss_res / ss_tot}

    # 1/Z only, through origin (lateral translation signature)
    inv_z = 1.0 / depth_m
    A = float(np.sum(delta_u * inv_z) / np.sum(inv_z ** 2))
    resid = delta_u - A * inv_z
    ss_res = float(np.sum(resid ** 2))
    models["inv_z_no_intercept"] = {"A_px_m": A, "r2": 1 - ss_res / ss_tot}

    # constant + 1/Z (combined / confound check)
    X = np.stack([np.ones_like(inv_z), inv_z], axis=1)
    coeffs, *_ = np.linalg.lstsq(X, delta_u, rcond=None)
    resid = delta_u - X @ coeffs
    ss_res = float(np.sum(resid ** 2))
    models["constant_plus_inv_z"] = {
        "c_px": float(coeffs[0]), "A_px_m": float(coeffs[1]),
        "r2": 1 - ss_res / ss_tot,
    }

    # linear in (u - cx): fx-scale-error / handedness-flip signature
    X = np.stack([np.ones_like(u_minus_cx), u_minus_cx], axis=1)
    coeffs, *_ = np.linalg.lstsq(X, delta_u, rcond=None)
    resid = delta_u - X @ coeffs
    ss_res = float(np.sum(resid ** 2))
    models["linear_u_minus_cx"] = {
        "intercept_px": float(coeffs[0]), "slope": float(coeffs[1]),
        "r2": 1 - ss_res / ss_tot,
    }

    # quadratic in (u - cx): unrectified-image signature
    X = np.stack([np.ones_like(u_minus_cx), u_minus_cx, u_minus_cx ** 2], axis=1)
    coeffs, *_ = np.linalg.lstsq(X, delta_u, rcond=None)
    resid = delta_u - X @ coeffs
    ss_res = float(np.sum(resid ** 2))
    models["quadratic_u_minus_cx"] = {
        "intercept_px": float(coeffs[0]), "linear_coeff": float(coeffs[1]),
        "quad_coeff": float(coeffs[2]), "r2": 1 - ss_res / ss_tot,
    }

    # exact handedness-flip line: delta_u = -2*(u - cx), i.e. slope fixed at -2, no free params
    predicted = -2.0 * u_minus_cx
    resid = delta_u - predicted
    ss_res = float(np.sum(resid ** 2))
    models["handedness_flip_fixed_slope_-2"] = {"r2": 1 - ss_res / ss_tot}

    return models


def make_plots(all_matches: list[EdgeMatch]):
    delta_u = np.array([m.delta_u_px for m in all_matches])
    depth_m = np.array([m.depth_m for m in all_matches])
    u_lidar = np.array([m.u_lidar_px for m in all_matches])
    pose_ids = [m.pose_id for m in all_matches]

    K = calibration_io.load_intrinsics_from_metadata(
        SESSION_DIR / "additional_image_data" / "metadata_pose_01.json"
    ).K
    cx = K[0, 2]
    u_minus_cx = u_lidar - cx

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(depth_m, delta_u, c="tab:blue")
    ax.axhline(0, c="k", lw=0.8)
    ax.set_xlabel("camera-frame depth Z (m)")
    ax.set_ylabel("delta_u = u_lidar_projected - u_image_edge (px)")
    ax.set_title("Horizontal offset vs depth")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "delta_u_vs_Z.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(u_minus_cx, delta_u, c="tab:green")
    ax.axhline(0, c="k", lw=0.8)
    ax.axvline(0, c="k", lw=0.8)
    ax.set_xlabel("u_lidar_projected - c_x (px)")
    ax.set_ylabel("delta_u (px)")
    ax.set_title("Horizontal offset vs off-axis column")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "delta_u_vs_u_minus_cx.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    unique_poses = sorted(set(pose_ids))
    for i, p in enumerate(unique_poses):
        vals = [d for d, pid in zip(delta_u, pose_ids) if pid == p]
        ax.scatter([i] * len(vals), vals, c="tab:purple")
        if vals:
            ax.scatter([i], [np.mean(vals)], c="black", marker="_", s=200)
    ax.set_xticks(range(len(unique_poses)))
    ax.set_xticklabels(unique_poses, rotation=45)
    ax.axhline(0, c="k", lw=0.8)
    ax.set_ylabel("delta_u (px)")
    ax.set_title("Horizontal offset by pose (black bar = per-pose mean)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "delta_u_vs_pose.png", dpi=150)
    plt.close(fig)

    return delta_u, depth_m, u_minus_cx, pose_ids


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manual-edges", default=None)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    T2 = calibration_io.load_transform()
    rig_cfg = rig_module.load_rig(config.DEFAULT_RIG_PATH, allow_placeholder=True)
    rig_module.warn_if_placeholder(rig_cfg)

    image_dir = SESSION_DIR / "images"
    laser_dir = SESSION_DIR / "lasers"
    metadata_dir = SESSION_DIR / "additional_image_data"
    pairs = scan_io.load_images_and_scans_from_folders(image_dir, laser_dir)

    all_matches: list[EdgeMatch] = []
    for image_path, laser_path in pairs:
        pose_id = pose_id_from_filename(image_path)
        metadata_path = metadata_dir / f"metadata_{pose_id}.json"
        matches = process_pose(image_path, laser_path, metadata_path, T2, rig_cfg)
        all_matches.extend(matches)
        print(f"{pose_id}: {len(matches)} auto-matched edges, "
              f"delta_u = {[round(m.delta_u_px, 1) for m in matches]}")

    write_manual_template(all_matches, OUT_DIR / "manual_edges_template.json")

    manual_matches = load_manual_edges(Path(args.manual_edges) if args.manual_edges else None)
    if manual_matches:
        print(f"Loaded {len(manual_matches)} manual correspondences from {args.manual_edges}")
        all_matches.extend(manual_matches)

    with open(OUT_DIR / "edges.json", "w") as f:
        json.dump([asdict(m) for m in all_matches], f, indent=2)

    delta_u, depth_m, u_minus_cx, pose_ids = make_plots(all_matches)
    models = fit_models(delta_u, depth_m, u_minus_cx)

    print()
    print("=" * 80)
    print(f"N matched edges (all poses, auto+manual): {len(all_matches)}")
    print(f"delta_u: mean={delta_u.mean():+.2f}px std={delta_u.std():.2f}px "
          f"min={delta_u.min():+.2f}px max={delta_u.max():+.2f}px")
    print()
    print("Model fits (R^2, higher is better fit):")
    for name, m in models.items():
        print(f"  {name}: {m}")
    print("=" * 80)

    pose01 = [m.delta_u_px for m in all_matches if m.pose_id == "pose_01"]
    pose03 = [m.delta_u_px for m in all_matches if m.pose_id == "pose_03"]
    pose01_mean = float(np.mean(pose01)) if pose01 else float("nan")
    pose03_mean = float(np.mean(pose03)) if pose03 else float("nan")
    ratio = pose03_mean / pose01_mean if pose01_mean else float("nan")
    print()
    print(f"pose_01 (median depth ~0.71m): N={len(pose01)}, mean delta_u={pose01_mean:+.2f}px")
    print(f"pose_03 (median depth ~1.64m): N={len(pose03)}, mean delta_u={pose03_mean:+.2f}px")
    print(f"ratio pose_03/pose_01 = {ratio:.3f} "
          f"(expect ~0.43 if lateral-translation error, ~1.0 if yaw error)")

    summary = {
        "n_matches": len(all_matches),
        "delta_u_mean_px": float(delta_u.mean()),
        "delta_u_std_px": float(delta_u.std()),
        "models": models,
        "pose_01_mean_delta_u_px": pose01_mean,
        "pose_03_mean_delta_u_px": pose03_mean,
        "pose03_over_pose01_ratio": ratio,
    }
    with open(OUT_DIR / "fit_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {OUT_DIR}")


if __name__ == "__main__":
    main()
