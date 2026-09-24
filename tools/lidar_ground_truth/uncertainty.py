"""T2 uncertainty by bootstrap, and per-point Monte Carlo propagation
(PLAN.md Sec 2.2, 6.4).

T2 is re-solved with the calibration package's own staged ICP
(cam_lidar_2d_icp.run_staged_icp), from the per-pose correspondences in
calibration_correspondences.npz and the initial transform and stage settings
recorded in calibration_result.json, so every re-solve matches the original
calibration run exactly apart from which poses it sees.

Pixel/depth uncertainty is propagated by sampling (PLAN.md Sec 2.2
implementation note), never by hand-coded Jacobians: each sample draws a T2
from the bootstrap ensemble and a range perturbation from the range-noise
model, and is pushed through projection.project_scan.
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import config
import projection

sys.path.insert(0, str(config.REPO_ROOT / "camera_2d_lidar_calibration"))
import cam_lidar_2d_icp  # noqa: E402


@dataclass
class Correspondences:
    pose_ids: list[str]
    lidar_points: list[np.ndarray]        # per pose, (N, 2) frame L
    camera_line_points: list[np.ndarray]  # per pose, (M, 2) frame R
    rvecs: list[np.ndarray]
    tvecs: list[np.ndarray]


def load_correspondences(npz_path: str | Path) -> Correspondences:
    data = np.load(npz_path)
    count = int(data["pose_count"])
    return Correspondences(
        pose_ids=[str(pose_id) for pose_id in data["pose_ids"]],
        lidar_points=[data[f"pose_{i:02d}_lidar_points"][:, :2] for i in range(count)],
        camera_line_points=[data[f"pose_{i:02d}_camera_line_points"] for i in range(count)],
        rvecs=[data[f"pose_{i:02d}_rvec"] for i in range(count)],
        tvecs=[data[f"pose_{i:02d}_tvec"] for i in range(count)],
    )


def solve_T2(
    correspondences: Correspondences,
    pose_indices: list[int] | np.ndarray,
    calibration_result: dict,
) -> np.ndarray:
    """Re-solve T2 on a subset (or resample) of poses, exactly as the
    calibration run did: same initial transform, same ICP stages.
    """
    initial_tf = np.array(calibration_result["initial_transform"]["transform_matrix_3x3"])
    stages = [
        {
            "distance_threshold_m": stage["distance_threshold_m"],
            "max_iterations": stage["max_iterations"],
            "point_pairs_threshold": stage["point_pairs_threshold"],
        }
        for stage in calibration_result["staged_icp"]["stages"]
    ]
    with contextlib.redirect_stdout(io.StringIO()):
        T2, _, _ = cam_lidar_2d_icp.run_staged_icp(
            camera_lines=[correspondences.camera_line_points[i] for i in pose_indices],
            raw_lidar_lines=[correspondences.lidar_points[i] for i in pose_indices],
            initial_tf=initial_tf,
            stages=stages,
        )
    return T2


def T2_parameters(T2: np.ndarray) -> tuple[float, float, float]:
    """(t_x [m], t_y [m], yaw [deg]) of an SE(2) transform."""
    return float(T2[0, 2]), float(T2[1, 2]), float(math.degrees(math.atan2(T2[1, 0], T2[0, 0])))


def bootstrap_T2(
    correspondences: Correspondences,
    calibration_result: dict,
    resamples: int = config.BOOTSTRAP_RESAMPLES,
    seed: int = config.BOOTSTRAP_SEED,
) -> np.ndarray:
    """(resamples, 3, 3) ensemble of T2, each solved on poses drawn with
    replacement (PLAN.md Sec 6.4)."""
    rng = np.random.default_rng(seed)
    count = len(correspondences.pose_ids)
    ensemble = []
    for index in range(resamples):
        ensemble.append(solve_T2(correspondences, rng.integers(0, count, count), calibration_result))
        if (index + 1) % 25 == 0:
            print(f"  bootstrap {index + 1}/{resamples}")
    return np.array(ensemble)


def ensemble_summary(ensemble: np.ndarray) -> dict:
    params = np.array([T2_parameters(T2) for T2 in ensemble])
    # Yaw is near +-180 deg on some rigs; express spread relative to the mean.
    yaw = params[:, 2]
    yaw = (yaw - yaw[0] + 180.0) % 360.0 - 180.0 + yaw[0]
    params[:, 2] = yaw
    return {
        "resamples": int(len(ensemble)),
        "mean": {"t_x_m": float(params[:, 0].mean()), "t_y_m": float(params[:, 1].mean()), "yaw_deg": float(yaw.mean())},
        "std": {"t_x_m": float(params[:, 0].std()), "t_y_m": float(params[:, 1].std()), "yaw_deg": float(yaw.std())},
        "percentile_2_5": {"t_x_m": float(np.percentile(params[:, 0], 2.5)), "t_y_m": float(np.percentile(params[:, 1], 2.5)), "yaw_deg": float(np.percentile(yaw, 2.5))},
        "percentile_97_5": {"t_x_m": float(np.percentile(params[:, 0], 97.5)), "t_y_m": float(np.percentile(params[:, 1], 97.5)), "yaw_deg": float(np.percentile(yaw, 97.5))},
        "covariance_tx_ty_yawdeg": np.cov(params.T).tolist(),
    }


def load_or_build_ensemble(
    cache_path: Path,
    correspondences_path: str | Path,
    correspondences: Correspondences,
    calibration_result: dict,
    resamples: int,
    seed: int,
) -> np.ndarray:
    """Reuse a cached ensemble only if it was built from the same
    correspondences file *contents* (not just the same path -- a re-solve
    into the same --out-dir overwrites calibration_correspondences.npz in
    place, so a path-only key would give a false cache hit) with the same
    settings."""
    digest = hashlib.sha256(Path(correspondences_path).read_bytes()).hexdigest()
    key = {"correspondences_sha256": digest, "resamples": resamples, "seed": seed}
    if cache_path.is_file():
        cached = np.load(cache_path, allow_pickle=False)
        if json.loads(str(cached["key"])) == key:
            print(f"Reusing bootstrap ensemble from {cache_path}")
            return cached["ensemble"]
    print(f"Building bootstrap ensemble ({resamples} resamples)...")
    ensemble = bootstrap_T2(correspondences, calibration_result, resamples, seed)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, ensemble=ensemble, key=json.dumps(key))
    return ensemble


def monte_carlo_point_uncertainty(
    xy_lidar: np.ndarray,
    ensemble: np.ndarray,
    range_sigma_m: float,
    delta_z_m: float,
    delta_z_tolerance_m: float,
    attitude_tolerance_deg: float,
    K: np.ndarray,
    image_wh: tuple[int, int],
    samples: int = config.MONTE_CARLO_SAMPLES,
    seed: int = config.MONTE_CARLO_SEED,
) -> dict:
    """Per-point depth/column/row uncertainty (PLAN.md Sec 6.4).

    Each sample uses one bootstrap T2 and a radial range perturbation
    N(0, range_sigma_m). row_uncertainty_px folds in the rig tolerances:
    sqrt(mc_v_std^2 + (f_y dz_tol / Z)^2 + (f_y theta_tol)^2 + (phi_tol (u - c_x))^2).
    """
    rng = np.random.default_rng(seed)
    radial = np.hypot(xy_lidar[:, 0], xy_lidar[:, 1])
    unit = xy_lidar / np.where(radial > 0, radial, 1.0)[:, None]

    u_samples, v_samples, depth_samples = [], [], []
    for index in rng.integers(0, len(ensemble), samples):
        perturbed = xy_lidar + unit * rng.normal(0.0, range_sigma_m, size=(len(xy_lidar), 1))
        result = projection.project_scan(perturbed, ensemble[index], delta_z_m, K, image_wh, z_min_m=config.Z_MIN_M)
        u_samples.append(result.u)
        v_samples.append(result.v)
        depth_samples.append(result.depth_m)

    u_samples, v_samples, depth_samples = map(np.array, (u_samples, v_samples, depth_samples))
    fy, cx = K[1, 1], K[0, 2]
    depth = depth_samples.mean(axis=0)
    u = u_samples.mean(axis=0)
    attitude_tol_rad = math.radians(attitude_tolerance_deg)
    row_uncertainty_px = np.sqrt(
        v_samples.std(axis=0) ** 2
        + (fy * delta_z_tolerance_m / np.where(depth > config.Z_MIN_M, depth, np.inf)) ** 2
        + (fy * attitude_tol_rad) ** 2
        + (attitude_tol_rad * (u - cx)) ** 2
    )
    return {
        "depth_std_m": depth_samples.std(axis=0),
        "u_std_px": u_samples.std(axis=0),
        "row_uncertainty_px": row_uncertainty_px,
    }
