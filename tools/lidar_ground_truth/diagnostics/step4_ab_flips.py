#!/usr/bin/env python3
"""Phase 1.5 Step 4: cheap A/B probes on T2.

For a fixed set of (u_lidar_baseline_source_point, u_image_edge) correspondences,
recompute u under each T2 variant and report mean |Delta_u|. This is a probe,
not a re-fit: T2 is mutated by literal component negation, never re-solved.
Nothing here is committed back to any calibration artifact.

Uses the 4-point hand-verified correspondence set (pose_01 left/right,
pose_03 left/right box edges) from the Step 3 report -- identified by direct
visual inspection + Sobel-peak confirmation, small N but high per-point
confidence. The Step 3 52-point AUTOMATIC matched set is deliberately not
reused here: Step 3 found its global model fits all sit at R^2~0 (dominated
by mismatch noise, and it does not persist each match's underlying (x_L,
y_L) LiDAR point needed to recompute u exactly under a modified T2), so it
would not add trustworthy signal to an A/B probe.

Read-only against cam_lidar_2d_icp.py / gui.py / icp_2d.py /
tools/lidar_characterisation. T2 variants are constructed in memory only.

Run from the repository root:
    python tools/lidar_ground_truth/diagnostics/step4_ab_flips.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # tools/lidar_ground_truth/

import numpy as np

import calibration_io
import config
import projection

OUT_DIR = config.REPO_ROOT / "results" / "lidar_ground_truth" / "diagnostics" / "step4_ab_flips"

# Hand-verified correspondences (see report.md "Step 3: hand-verified case
# study"): for each, the ORIGINAL LiDAR-frame point (x_L, y_L) that produced
# the projected edge (so u can be recomputed exactly under a modified T2),
# and the matched image-edge column u_image_edge_px (frozen from the
# baseline analysis -- NOT re-matched per variant).
HAND_VERIFIED = [
    # pose_01 left box edge: idx 427, xL=0.860, yL=0.356 -> baseline u=448.38, camera depth=0.719m
    {"pose_id": "pose_01", "side": "left", "xL": 0.860, "yL": 0.356,
     "u_image_edge_px": 429.0, "depth_m": 0.719},
    # pose_01 right box edge: idx 20, xL=0.873, yL=-0.230 -> baseline u=874.55, camera depth=0.717m
    {"pose_id": "pose_01", "side": "right", "xL": 0.873, "yL": -0.230,
     "u_image_edge_px": 842.0, "depth_m": 0.717},
    # pose_03 left box edge: idx 435, xL=1.62895, yL=0.376971 -> baseline u=545.10, camera depth=1.488m
    {"pose_id": "pose_03", "side": "left", "xL": 1.62895, "yL": 0.376971,
     "u_image_edge_px": 491.5, "depth_m": 1.488},
    # pose_03 right box edge: idx 11, xL=1.643866, yL=-0.200103 -> baseline u=747.56, camera depth=1.488m
    {"pose_id": "pose_03", "side": "right", "xL": 1.643866, "yL": -0.200103,
     "u_image_edge_px": 737.0, "depth_m": 1.488},
]


def yaw_of(R2: np.ndarray) -> float:
    return math.atan2(R2[1, 0], R2[0, 0])


def rotation_from_yaw(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s], [s, c]])


def build_variants(T2: np.ndarray) -> dict[str, np.ndarray]:
    variants = {}
    variants["baseline"] = T2.copy()

    v = T2.copy()
    v[1, 2] = -v[1, 2]
    variants["t2_ty_negated"] = v

    v = T2.copy()
    v[0, 2] = -v[0, 2]
    variants["t2_tx_negated"] = v

    variants["y_L_handedness_flip"] = T2.copy()  # applied to input points, not T2 itself

    v = T2.copy()
    yaw = yaw_of(T2[:2, :2])
    v[:2, :2] = rotation_from_yaw(-yaw)
    variants["yaw_negated"] = v

    return variants


def recompute_u(xy_lidar: np.ndarray, T2: np.ndarray, K: np.ndarray) -> np.ndarray:
    result = projection.project_scan(
        xy_lidar, T2, delta_z_m=0.0, K=K, image_wh=(1_000_000, 1_000_000), z_min_m=1e-6,
    )
    return result.u


def evaluate_hand_verified(T2_variants, K) -> dict:
    out = {}
    for name, T2 in T2_variants.items():
        deltas = []
        for pt in HAND_VERIFIED:
            xL, yL = pt["xL"], pt["yL"]
            if name == "y_L_handedness_flip":
                yL = -yL
            xy = np.array([[xL, yL]], dtype=np.float64)
            u = recompute_u(xy, T2, K)[0]
            deltas.append(u - pt["u_image_edge_px"])
        deltas = np.array(deltas)
        out[name] = {
            "mean_abs_delta_u_px": float(np.mean(np.abs(deltas))),
            "mean_signed_delta_u_px": float(np.mean(deltas)),
            "per_point_delta_u_px": [round(float(d), 1) for d in deltas],
        }
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    T2 = calibration_io.load_transform()
    variants = build_variants(T2)

    K_pose01 = calibration_io.load_intrinsics_from_metadata(
        config.REPO_ROOT / "data/data_2026-06-28/additional_image_data/metadata_pose_01.json"
    ).K

    results = evaluate_hand_verified(variants, K_pose01)

    print("=" * 90)
    print("Step 4 A/B flips -- hand-verified 4-point correspondence set "
          "(pose_01 L/R, pose_03 L/R box edges)")
    print("=" * 90)
    for name, r in results.items():
        print(f"{name:24s}  mean|Delta_u|={r['mean_abs_delta_u_px']:7.2f}px  "
              f"mean(signed)={r['mean_signed_delta_u_px']:+7.2f}px  "
              f"per-point={r['per_point_delta_u_px']}")

    # Explicit t_y-seed-sign-flip magnitude check (task Step 4 note): a sign
    # error on the y_left=0.06 seed would leave the solution ~0.12 m off in
    # t_y. Compare that PREDICTED Delta_u (fx * 0.12 / depth) against the
    # observed one for each hand-verified point.
    fx = K_pose01[0, 0]
    print()
    print("t_y seed-sign-flip hypothesis check (predicted vs observed |Delta_u|):")
    baseline_deltas = results["baseline"]["per_point_delta_u_px"]
    for pt, observed in zip(HAND_VERIFIED, baseline_deltas):
        predicted = fx * 0.12 / pt["depth_m"]
        print(f"  {pt['pose_id']} {pt['side']}: depth={pt['depth_m']:.3f}m, "
              f"predicted |Delta_u| if t_y seed sign-flipped = {predicted:.1f}px, "
              f"observed baseline Delta_u = {observed:+.1f}px")

    with open(OUT_DIR / "ab_flip_results.json", "w") as f:
        json.dump({"hand_verified_points": HAND_VERIFIED, "results": results}, f, indent=2)
    print(f"\nSaved {OUT_DIR / 'ab_flip_results.json'}")


if __name__ == "__main__":
    main()
