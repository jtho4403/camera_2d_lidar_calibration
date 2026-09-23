"""
cam_lidar_2d_icp.py
"""
# import rclpy
# from rclpy.executors import ExternalShutdownException
# from rclpy.node import Node

import os
import argparse
from datetime import datetime
import json
import csv
import math

import cv2
# import cv_bridge
import open3d as o3d

import numpy as np
from numpy import linalg as la
from scipy.spatial.transform import Rotation

import tkinter as tk
from tkinter import ttk
from pathlib import Path

import icp_2d

import matplotlib.pyplot as plt
import matplotlib.backends.backend_tkagg as tkagg 
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D
from sklearn import linear_model
from sklearn.neighbors import NearestNeighbors

from gui import SelectPointsInterface, ImageVisInterface

home = Path.home()

def load_images_from_folder(folder):
    images = []
    print("Reading images from directory: " + folder)    
    for filename in sorted(os.listdir(folder)):
        img = cv2.imread(os.path.join(folder,filename))
        print(os.path.join(folder,filename)) # printing file names to verify the order of them in the list
        if img is not None:
            # image = cv2.rotate(img, cv2.ROTATE_180)
            # images.append(image)
            images.append(img)
    return images

def load_clouds_from_folder(folder):
    clouds = []
    print("Reading point clouds from directory: " + folder)    
    for filename in sorted(os.listdir(folder)):
        pcd = o3d.io.read_point_cloud(os.path.join(folder,filename))
        print(os.path.join(folder,filename)) # printing file names to verify the order of them in the list
        # print(pcd) 
        if len(pcd.points) > 0:
            clouds.append(pcd)
    return clouds

def load_rectified_left_intrinsics(manifest_path):
    """
    Load the ZED SDK rectified LEFT-camera intrinsics from a capture
    session's session_manifest.json (calibration.rectified.left).

    The rectified K is SDK-version and resolution dependent, so it is read
    per capture session instead of being hardcoded. Non-zero rectified
    distortion means the frames are not rectified, which is a hard error.

    Returns (K 3x3, dist 1x5 zeros, (width, height)).
    """
    with open(manifest_path) as f:
        manifest = json.load(f)

    left = manifest["calibration"]["rectified"]["left"]

    if any(float(d) != 0.0 for d in left["disto"]):
        raise ValueError(
            f"{manifest_path}: calibration.rectified.left.disto is non-zero; "
            "the frames are not rectified."
        )

    camera_k = np.array([
        [left["fx"], 0.0, left["cx"]],
        [0.0, left["fy"], left["cy"]],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    camera_dist = np.zeros((1, 5), dtype=np.float64)
    image_wh = tuple(int(v) for v in left["image_size"])

    return camera_k, camera_dist, image_wh


def draw(img, corners, imgpts):
    # Draw a 3D axis at the OpenCV origin of a checkerboard, for introspection
    corner = tuple(corners[0].ravel().astype("int32"))
    imgpts = imgpts.astype("int32")
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (0,0,255), 2)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 2)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (255,0,0), 2)
    return img

def transform_lines(lines: list[np.ndarray], tf: np.ndarray) -> list[np.ndarray]:
    """
    Apply one 2D homogeneous rigid transform to a list of Nx2 point arrays.
    """
    transformed_lines = []

    for line in lines:
        line = np.asarray(line, dtype=np.float64)

        if line.ndim != 2 or line.shape[1] != 2:
            raise ValueError(f"Expected an Nx2 line array, got shape {line.shape}")

        homogeneous = np.hstack(
            (line, np.ones((line.shape[0], 1), dtype=np.float64))
        )
        transformed = (tf @ homogeneous.T).T[:, :2]
        transformed_lines.append(transformed)

    return transformed_lines


def compose_icp_history(transformation_history: list[np.ndarray]) -> np.ndarray:
    """
    Compose incremental [R | t] ICP updates in the same order in which
    they were applied to the points.

    If:
        p1 = T1 p0
        p2 = T2 p1
    then:
        p2 = T2 T1 p0

    Therefore each new update is left-multiplied.
    """
    total = np.eye(3, dtype=np.float64)

    for incremental in transformation_history:
        incremental_h = np.eye(3, dtype=np.float64)
        incremental_h[:2, :] = incremental
        total = incremental_h @ total

    return total


def split_stacked_lines(
    stacked_points: np.ndarray,
    line_lengths: list[int],
) -> list[np.ndarray]:
    """
    Reconstruct the individual pose lines returned by icp_per_line(),
    which currently returns one vertically stacked array.
    """
    if sum(line_lengths) != len(stacked_points):
        raise ValueError(
            "Cannot split ICP output: total line lengths do not equal "
            f"stacked point count ({sum(line_lengths)} != {len(stacked_points)})."
        )

    split_indices = np.cumsum(line_lengths)[:-1]
    return [
        part.copy()
        for part in np.split(stacked_points, split_indices)
    ]


def nearest_line_distances(
    reference_line: np.ndarray,
    query_line: np.ndarray,
) -> np.ndarray:
    """
    Return distance from each query point to the nearest sampled point
    in the reference line array.

    This is a sampled-segment distance, not a pure point-to-infinite-line
    distance. It is retained for ICP correspondence eligibility and as a
    secondary diagnostic because it is affected by line endpoint extent
    and point-sampling density.
    """
    neighbour_model = NearestNeighbors(
        n_neighbors=1,
        algorithm="kd_tree",
    ).fit(reference_line)

    distances, _ = neighbour_model.kneighbors(query_line)
    return distances[:, 0]


def fit_line_tls(
    points: np.ndarray,
    robust: bool = False,
    max_iterations: int = 10,
    mad_scale: float = 3.5,
    minimum_points: int = 3,
) -> dict:
    """
    Fit an infinite 2D line using total least squares (orthogonal PCA).

    The line is represented by:
      point     — one point on the line, using the fitted centroid
      direction — unit vector along the line
      normal    — unit vector perpendicular to the line

    When robust=True, iterative MAD-based trimming is applied. This is
    useful for fitting the selected LiDAR points if a few edge or
    background points were included accidentally.
    """
    points = np.asarray(points, dtype=np.float64)

    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(
            f"Expected an Nx2 point array, got shape {points.shape}"
        )

    if len(points) < minimum_points:
        raise ValueError(
            f"At least {minimum_points} points are required to fit a line; "
            f"received {len(points)}."
        )

    working_mask = np.ones(len(points), dtype=bool)

    for _ in range(max_iterations if robust else 1):
        working_points = points[working_mask]

        if len(working_points) < minimum_points:
            break

        centroid = working_points.mean(axis=0)
        centred = working_points - centroid

        # Principal right-singular vector gives the TLS line direction.
        _, _, vh = np.linalg.svd(centred, full_matrices=False)
        direction = vh[0]
        direction = direction / np.linalg.norm(direction)

        # Give the line direction a deterministic sign so repeat runs do
        # not arbitrarily report directions differing by 180 degrees.
        dominant_axis = int(np.argmax(np.abs(direction)))
        if direction[dominant_axis] < 0:
            direction = -direction

        normal = np.array(
            [-direction[1], direction[0]],
            dtype=np.float64,
        )

        if not robust:
            break

        signed_distances = (points - centroid) @ normal
        absolute_distances = np.abs(signed_distances)

        median_distance = np.median(absolute_distances)
        mad = np.median(
            np.abs(absolute_distances - median_distance)
        )

        # Robust sigma estimate. If the data are essentially perfectly
        # linear, stop instead of constructing an almost-zero threshold.
        robust_sigma = 1.4826 * mad
        if robust_sigma < 1e-9:
            break

        new_mask = absolute_distances <= (
            median_distance + mad_scale * robust_sigma
        )

        if np.count_nonzero(new_mask) < minimum_points:
            break

        if np.array_equal(new_mask, working_mask):
            break

        working_mask = new_mask

    # Refit once using the final retained point set.
    working_points = points[working_mask]
    centroid = working_points.mean(axis=0)
    centred = working_points - centroid

    _, _, vh = np.linalg.svd(centred, full_matrices=False)
    direction = vh[0]
    direction = direction / np.linalg.norm(direction)

    dominant_axis = int(np.argmax(np.abs(direction)))
    if direction[dominant_axis] < 0:
        direction = -direction

    normal = np.array(
        [-direction[1], direction[0]],
        dtype=np.float64,
    )

    signed_distances = (points - centroid) @ normal

    return {
        "point": centroid,
        "direction": direction,
        "normal": normal,
        "inlier_mask": working_mask,
        "inlier_count": int(np.count_nonzero(working_mask)),
        "total_count": int(len(points)),
        "signed_distances": signed_distances,
        "absolute_distances": np.abs(signed_distances),
    }


def point_to_infinite_line_distances(
    line_points: np.ndarray,
    query_points: np.ndarray,
) -> np.ndarray:
    """
    Return perpendicular distances from query points to the infinite
    total-least-squares line fitted through line_points.

    Unlike nearest sampled-point distance, this metric is unaffected by
    the synthetic reference line's start, end, or sampling spacing.
    """
    fitted_line = fit_line_tls(
        line_points,
        robust=False,
    )

    query_points = np.asarray(query_points, dtype=np.float64)
    signed_distances = (
        query_points - fitted_line["point"]
    ) @ fitted_line["normal"]

    return np.abs(signed_distances)


def acute_line_angle_difference_deg(
    direction_a: np.ndarray,
    direction_b: np.ndarray,
) -> float:
    """
    Return the acute orientation difference between two undirected lines.

    A line direction and its negative represent the same geometric line,
    so the result is constrained to 0–90 degrees.
    """
    direction_a = np.asarray(direction_a, dtype=np.float64)
    direction_b = np.asarray(direction_b, dtype=np.float64)

    direction_a = direction_a / np.linalg.norm(direction_a)
    direction_b = direction_b / np.linalg.norm(direction_b)

    cosine = float(
        np.clip(
            np.abs(np.dot(direction_a, direction_b)),
            0.0,
            1.0,
        )
    )

    return float(np.degrees(np.arccos(cosine)))


def fitted_line_alignment_diagnostics(
    camera_line: np.ndarray,
    transformed_lidar_line: np.ndarray,
) -> dict:
    """
    Compare the fitted infinite camera and transformed-LiDAR lines.

    The camera line uses standard TLS because it is synthetically
    generated and already perfectly linear.

    The LiDAR line uses robust TLS to reduce the influence of occasional
    manually selected edge/background points.
    """
    camera_fit = fit_line_tls(
        camera_line,
        robust=False,
    )
    lidar_fit = fit_line_tls(
        transformed_lidar_line,
        robust=True,
    )

    angular_difference_deg = acute_line_angle_difference_deg(
        camera_fit["direction"],
        lidar_fit["direction"],
    )

    # Signed perpendicular displacement of the LiDAR fitted-line centroid
    # relative to the camera fitted line.
    signed_normal_offset_m = float(
        (
            lidar_fit["point"] - camera_fit["point"]
        ) @ camera_fit["normal"]
    )

    return {
        "camera_line_point": camera_fit["point"].tolist(),
        "camera_line_direction": camera_fit["direction"].tolist(),
        "camera_line_normal": camera_fit["normal"].tolist(),
        "lidar_line_point": lidar_fit["point"].tolist(),
        "lidar_line_direction": lidar_fit["direction"].tolist(),
        "lidar_robust_fit_inlier_count": lidar_fit["inlier_count"],
        "lidar_robust_fit_total_count": lidar_fit["total_count"],
        "lidar_robust_fit_inlier_fraction": float(
            lidar_fit["inlier_count"]
            / max(lidar_fit["total_count"], 1)
        ),
        "angular_difference_deg": angular_difference_deg,
        "signed_normal_offset_m": signed_normal_offset_m,
        "absolute_normal_offset_m": abs(signed_normal_offset_m),
    }


def distance_statistics(distances: np.ndarray) -> dict:
    """
    Return JSON-serialisable residual statistics.
    """
    distances = np.asarray(distances, dtype=np.float64)

    if distances.size == 0:
        return {
            "count": 0,
            "mean_abs_m": None,
            "rmse_m": None,
            "median_m": None,
            "p95_m": None,
            "max_m": None,
        }

    return {
        "count": int(distances.size),
        "mean_abs_m": float(np.mean(np.abs(distances))),
        "rmse_m": float(np.sqrt(np.mean(np.square(distances)))),
        "median_m": float(np.median(distances)),
        "p95_m": float(np.percentile(distances, 95)),
        "max_m": float(np.max(distances)),
    }


def evaluate_alignment_residuals(
    camera_lines: list[np.ndarray],
    transformed_lidar_lines: list[np.ndarray],
    acceptance_threshold_m: float,
) -> dict:
    """
    Evaluate post-alignment quality independently for each pose.

    Primary metric:
      perpendicular distance from each transformed LiDAR point to the
      infinite TLS line fitted through the corresponding camera line.

    Secondary metric:
      nearest sampled camera-segment point distance, retained to show the
      effect of finite synthetic line length and sampling.

    LiDAR pose i is always evaluated only against camera pose i.
    """
    if len(camera_lines) != len(transformed_lidar_lines):
        raise ValueError(
            "Camera/LiDAR pose count mismatch during residual evaluation."
        )

    per_pose = []

    all_orthogonal_distances = []
    all_accepted_orthogonal_distances = []

    all_segment_distances = []
    all_accepted_segment_distances = []

    print()
    print("=" * 96)
    print("POST-ALIGNMENT RESIDUAL VALIDATION")
    print(
        "Primary metric: orthogonal transformed-LiDAR point distance "
        "to corresponding infinite camera line"
    )
    print(
        "Secondary metric: nearest sampled camera-segment point distance"
    )
    print(f"Acceptance threshold: {acceptance_threshold_m:.3f} m")
    print("=" * 96)

    for pose_index, (camera_line, lidar_line) in enumerate(
        zip(camera_lines, transformed_lidar_lines),
        start=1,
    ):
        camera_line = np.asarray(camera_line, dtype=np.float64)
        lidar_line = np.asarray(lidar_line, dtype=np.float64)

        # Primary metric: endpoint-independent perpendicular distance.
        orthogonal_distances = point_to_infinite_line_distances(
            camera_line,
            lidar_line,
        )
        orthogonal_accepted_mask = (
            orthogonal_distances <= acceptance_threshold_m
        )
        accepted_orthogonal_distances = orthogonal_distances[
            orthogonal_accepted_mask
        ]

        # Secondary metric: old sampled-segment nearest-neighbour result.
        segment_distances = nearest_line_distances(
            camera_line,
            lidar_line,
        )
        segment_accepted_mask = (
            segment_distances <= acceptance_threshold_m
        )
        accepted_segment_distances = segment_distances[
            segment_accepted_mask
        ]

        orthogonal_total_count = int(orthogonal_distances.size)
        orthogonal_accepted_count = int(
            np.count_nonzero(orthogonal_accepted_mask)
        )
        orthogonal_accepted_fraction = (
            orthogonal_accepted_count / orthogonal_total_count
            if orthogonal_total_count > 0
            else 0.0
        )

        segment_total_count = int(segment_distances.size)
        segment_accepted_count = int(
            np.count_nonzero(segment_accepted_mask)
        )
        segment_accepted_fraction = (
            segment_accepted_count / segment_total_count
            if segment_total_count > 0
            else 0.0
        )

        orthogonal_all_stats = distance_statistics(
            orthogonal_distances
        )
        orthogonal_accepted_stats = distance_statistics(
            accepted_orthogonal_distances
        )

        segment_all_stats = distance_statistics(
            segment_distances
        )
        segment_accepted_stats = distance_statistics(
            accepted_segment_distances
        )

        line_diagnostics = fitted_line_alignment_diagnostics(
            camera_line,
            lidar_line,
        )

        pose_result = {
            "pose": f"pose_{pose_index:02d}",
            "primary_metric": (
                "orthogonal_distance_to_infinite_camera_line"
            ),
            "acceptance_threshold_m": float(
                acceptance_threshold_m
            ),
            "orthogonal_point_to_line": {
                "total_point_count": orthogonal_total_count,
                "accepted_point_count": (
                    orthogonal_accepted_count
                ),
                "accepted_fraction": float(
                    orthogonal_accepted_fraction
                ),
                "all_points": orthogonal_all_stats,
                "accepted_points": (
                    orthogonal_accepted_stats
                ),
            },
            "sampled_segment_secondary": {
                "total_point_count": segment_total_count,
                "accepted_point_count": segment_accepted_count,
                "accepted_fraction": float(
                    segment_accepted_fraction
                ),
                "all_points": segment_all_stats,
                "accepted_points": segment_accepted_stats,
            },
            "fitted_line_diagnostics": line_diagnostics,
        }
        per_pose.append(pose_result)

        all_orthogonal_distances.append(
            orthogonal_distances
        )
        if accepted_orthogonal_distances.size:
            all_accepted_orthogonal_distances.append(
                accepted_orthogonal_distances
            )

        all_segment_distances.append(segment_distances)
        if accepted_segment_distances.size:
            all_accepted_segment_distances.append(
                accepted_segment_distances
            )

        print(
            f"Pose {pose_index:02d} | "
            f"orthogonal: "
            f"N={orthogonal_total_count:3d}, "
            f"accepted={orthogonal_accepted_count:3d}/"
            f"{orthogonal_total_count:3d} "
            f"({orthogonal_accepted_fraction * 100:5.1f}%), "
            f"RMSE={orthogonal_all_stats['rmse_m']:.4f} m, "
            f"median={orthogonal_all_stats['median_m']:.4f} m, "
            f"p95={orthogonal_all_stats['p95_m']:.4f} m | "
            f"line angle={line_diagnostics['angular_difference_deg']:.3f} deg, "
            f"normal offset="
            f"{line_diagnostics['signed_normal_offset_m']:+.4f} m"
        )

        print(
            f"        sampled segment: "
            f"accepted={segment_accepted_count:3d}/"
            f"{segment_total_count:3d} "
            f"({segment_accepted_fraction * 100:5.1f}%), "
            f"RMSE={segment_all_stats['rmse_m']:.4f} m, "
            f"median={segment_all_stats['median_m']:.4f} m, "
            f"p95={segment_all_stats['p95_m']:.4f} m"
        )

    overall_orthogonal_distances = (
        np.concatenate(all_orthogonal_distances)
        if all_orthogonal_distances
        else np.array([], dtype=np.float64)
    )
    overall_accepted_orthogonal = (
        np.concatenate(
            all_accepted_orthogonal_distances
        )
        if all_accepted_orthogonal_distances
        else np.array([], dtype=np.float64)
    )

    overall_segment_distances = (
        np.concatenate(all_segment_distances)
        if all_segment_distances
        else np.array([], dtype=np.float64)
    )
    overall_accepted_segment = (
        np.concatenate(
            all_accepted_segment_distances
        )
        if all_accepted_segment_distances
        else np.array([], dtype=np.float64)
    )

    orthogonal_total_count = int(
        overall_orthogonal_distances.size
    )
    orthogonal_accepted_count = int(
        overall_accepted_orthogonal.size
    )
    orthogonal_accepted_fraction = (
        orthogonal_accepted_count / orthogonal_total_count
        if orthogonal_total_count > 0
        else 0.0
    )

    segment_total_count = int(
        overall_segment_distances.size
    )
    segment_accepted_count = int(
        overall_accepted_segment.size
    )
    segment_accepted_fraction = (
        segment_accepted_count / segment_total_count
        if segment_total_count > 0
        else 0.0
    )

    overall = {
        "primary_metric": (
            "orthogonal_distance_to_infinite_camera_line"
        ),
        "acceptance_threshold_m": float(
            acceptance_threshold_m
        ),
        "orthogonal_point_to_line": {
            "total_point_count": orthogonal_total_count,
            "accepted_point_count": (
                orthogonal_accepted_count
            ),
            "accepted_fraction": float(
                orthogonal_accepted_fraction
            ),
            "all_points": distance_statistics(
                overall_orthogonal_distances
            ),
            "accepted_points": distance_statistics(
                overall_accepted_orthogonal
            ),
        },
        "sampled_segment_secondary": {
            "total_point_count": segment_total_count,
            "accepted_point_count": segment_accepted_count,
            "accepted_fraction": float(
                segment_accepted_fraction
            ),
            "all_points": distance_statistics(
                overall_segment_distances
            ),
            "accepted_points": distance_statistics(
                overall_accepted_segment
            ),
        },
    }

    orthogonal_stats = overall[
        "orthogonal_point_to_line"
    ]["all_points"]
    segment_stats = overall[
        "sampled_segment_secondary"
    ]["all_points"]

    print("-" * 96)
    print(
        "Overall orthogonal point-to-line: "
        f"N={orthogonal_total_count}, "
        f"accepted={orthogonal_accepted_count}/"
        f"{orthogonal_total_count} "
        f"({orthogonal_accepted_fraction * 100:.1f}%), "
        f"RMSE={orthogonal_stats['rmse_m']:.4f} m, "
        f"mean={orthogonal_stats['mean_abs_m']:.4f} m, "
        f"median={orthogonal_stats['median_m']:.4f} m, "
        f"p95={orthogonal_stats['p95_m']:.4f} m, "
        f"max={orthogonal_stats['max_m']:.4f} m"
    )
    print(
        "Overall sampled-segment secondary: "
        f"accepted={segment_accepted_count}/"
        f"{segment_total_count} "
        f"({segment_accepted_fraction * 100:.1f}%), "
        f"RMSE={segment_stats['rmse_m']:.4f} m, "
        f"mean={segment_stats['mean_abs_m']:.4f} m, "
        f"median={segment_stats['median_m']:.4f} m, "
        f"p95={segment_stats['p95_m']:.4f} m, "
        f"max={segment_stats['max_m']:.4f} m"
    )
    print("=" * 96)

    return {
        "metric_methodology": {
            "primary": (
                "Orthogonal distance from each transformed LiDAR "
                "point to the infinite total-least-squares camera line "
                "for the corresponding pose."
            ),
            "secondary": (
                "Nearest sampled camera-segment point distance, "
                "retained to expose endpoint and sampling effects."
            ),
            "line_fit_diagnostic": (
                "Camera and transformed LiDAR lines are fitted with "
                "total least squares; robust MAD trimming is used for "
                "the LiDAR diagnostic fit."
            ),
        },
        "per_pose": per_pose,
        "overall": overall,
    }


def count_threshold_correspondences(
    camera_lines: list[np.ndarray],
    lidar_lines: list[np.ndarray],
    threshold_m: float,
) -> tuple[int, int]:
    """
    Count same-pose nearest-neighbour correspondences below a threshold.
    """
    accepted = 0
    total = 0

    for camera_line, lidar_line in zip(camera_lines, lidar_lines):
        distances = nearest_line_distances(camera_line, lidar_line)
        accepted += int(np.count_nonzero(distances <= threshold_m))
        total += int(distances.size)

    return accepted, total


def run_staged_icp(
    camera_lines: list[np.ndarray],
    raw_lidar_lines: list[np.ndarray],
    initial_tf: np.ndarray,
    stages: list[dict],
) -> tuple[np.ndarray, list[np.ndarray], list[dict]]:
    """
    Run coarse-to-fine ICP.

    Each stage begins from the result of the previous stage:
      0.30 m capture
      0.15 m refinement
      0.10 m final refinement

    Pose correspondences remain isolated inside icp_per_line().
    """
    total_tf = initial_tf.copy()
    working_lines = transform_lines(raw_lidar_lines, initial_tf)
    stage_results = []

    print()
    print("=" * 80)
    print("STAGED PER-POSE 2D ICP")
    print("=" * 80)
    print("Initial transform:")
    print(initial_tf)

    for stage_number, stage in enumerate(stages, start=1):
        threshold = float(stage["distance_threshold_m"])
        max_iterations = int(stage.get("max_iterations", 150))
        point_pairs_threshold = int(stage.get("point_pairs_threshold", 20))

        print()
        print("-" * 80)
        print(
            f"Stage {stage_number}/{len(stages)}: "
            f"distance threshold={threshold:.3f} m, "
            f"max iterations={max_iterations}"
        )
        print("-" * 80)

        input_lines = [line.copy() for line in working_lines]
        line_lengths = [len(line) for line in input_lines]

        accepted_before, total_before = count_threshold_correspondences(
            camera_lines,
            input_lines,
            threshold,
        )

        print(
            f"Eligible same-pose correspondences before stage: "
            f"{accepted_before}/{total_before} "
            f"({100.0 * accepted_before / max(total_before, 1):.1f}%)"
        )

        transformation_history, stacked_aligned_points = icp_2d.icp_per_line(
            camera_lines,
            input_lines,
            max_iterations=max_iterations,
            distance_threshold=threshold,
            convergence_translation_threshold=1e-7,
            convergence_rotation_threshold=1e-7,
            point_pairs_threshold=point_pairs_threshold,
            verbose=True,
        )

        stage_tf = compose_icp_history(transformation_history)
        total_tf = stage_tf @ total_tf

        working_lines = split_stacked_lines(
            stacked_aligned_points,
            line_lengths,
        )

        accepted_after, total_after = count_threshold_correspondences(
            camera_lines,
            working_lines,
            threshold,
        )

        stage_yaw_deg = math.degrees(
            math.atan2(stage_tf[1, 0], stage_tf[0, 0])
        )

        stage_record = {
            "stage": stage_number,
            "distance_threshold_m": threshold,
            "max_iterations": max_iterations,
            "point_pairs_threshold": point_pairs_threshold,
            "iteration_count": len(transformation_history),
            "eligible_before": accepted_before,
            "total_before": total_before,
            "eligible_after": accepted_after,
            "total_after": total_after,
            "incremental_transform_3x3": stage_tf.tolist(),
            "incremental_yaw_deg": float(stage_yaw_deg),
            "cumulative_transform_3x3": total_tf.tolist(),
        }
        stage_results.append(stage_record)

        print(
            f"Stage {stage_number} complete: "
            f"iterations={len(transformation_history)}, "
            f"eligible after={accepted_after}/{total_after}"
        )
        print("Incremental transform:")
        print(stage_tf)
        print("Cumulative transform:")
        print(total_tf)

        if len(transformation_history) == 0:
            print(
                "WARNING: this stage performed zero updates. "
                "Inspect its correspondence count and threshold."
            )

    print()
    print("=" * 80)
    print("FINAL STAGED ICP TRANSFORM")
    print(total_tf)
    print("=" * 80)

    return total_tf, working_lines, stage_results


def wrap_angle_deg(angle_deg: float) -> float:
    """
    Wrap an angle difference to [-180, 180) degrees.
    """
    return float((angle_deg + 180.0) % 360.0 - 180.0)


def rigid_transform_summary(tf: np.ndarray) -> dict:
    """
    Summarise a 2D rigid transform.
    """
    rotation = tf[:2, :2]
    translation = tf[:2, 2]

    yaw_rad = math.atan2(rotation[1, 0], rotation[0, 0])
    yaw_deg = math.degrees(yaw_rad)

    # For p_camera = R * p_lidar + t:
    # camera origin expressed in LiDAR coordinates is -R^T t.
    camera_origin_in_lidar = -rotation.T @ translation

    return {
        "yaw_deg": float(yaw_deg),
        "translation_x_m": float(translation[0]),
        "translation_y_m": float(translation[1]),
        "translation_magnitude_m": float(np.linalg.norm(translation)),
        "camera_origin_in_lidar_x_m": float(camera_origin_in_lidar[0]),
        "camera_origin_in_lidar_y_m": float(camera_origin_in_lidar[1]),
        "camera_origin_in_lidar_magnitude_m": float(
            np.linalg.norm(camera_origin_in_lidar)
        ),
    }


def validate_against_manual_rig_geometry(
    estimated_tf: np.ndarray,
    measured_camera_origin_lidar: np.ndarray,
    measured_yaw_deg: float,
) -> dict:
    """
    Compare the calibrated transform against the rough manual mounting
    measurement supplied on the command line (camera centre in the LiDAR
    frame, x forward / y left, and relative yaw).

    This is only a sanity check because housing measurements are not
    measurements of the exact optical/reference origins.
    """
    measured_camera_origin_lidar = np.asarray(
        measured_camera_origin_lidar,
        dtype=np.float64,
    )

    transform_summary = rigid_transform_summary(estimated_tf)

    estimated_camera_origin_lidar = np.array(
        [
            transform_summary["camera_origin_in_lidar_x_m"],
            transform_summary["camera_origin_in_lidar_y_m"],
        ],
        dtype=np.float64,
    )

    position_error_vector = (
        estimated_camera_origin_lidar
        - measured_camera_origin_lidar
    )
    position_error_m = float(np.linalg.norm(position_error_vector))
    yaw_error_deg = wrap_angle_deg(
        transform_summary["yaw_deg"] - measured_yaw_deg
    )

    # Deliberately loose, diagnostic-only tolerances for manual
    # housing measurements.
    position_tolerance_m = 0.075
    yaw_tolerance_deg = 5.0

    position_pass = position_error_m <= position_tolerance_m
    yaw_pass = abs(yaw_error_deg) <= yaw_tolerance_deg
    overall_pass = position_pass and yaw_pass

    result = {
        "purpose": (
            "Diagnostic sanity check only; manual housing measurements "
            "are not exact sensor optical-origin measurements."
        ),
        "measured_camera_origin_in_lidar_m": {
            "x_forward": float(measured_camera_origin_lidar[0]),
            "y_left": float(measured_camera_origin_lidar[1]),
        },
        "measured_yaw_deg": measured_yaw_deg,
        "estimated_transform_summary": transform_summary,
        "estimated_camera_origin_in_lidar_m": {
            "x_forward": float(estimated_camera_origin_lidar[0]),
            "y_left": float(estimated_camera_origin_lidar[1]),
        },
        "position_error_vector_m": {
            "x": float(position_error_vector[0]),
            "y": float(position_error_vector[1]),
        },
        "position_error_magnitude_m": position_error_m,
        "yaw_error_deg": yaw_error_deg,
        "diagnostic_tolerances": {
            "position_m": position_tolerance_m,
            "yaw_deg": yaw_tolerance_deg,
        },
        "position_check_passed": position_pass,
        "yaw_check_passed": yaw_pass,
        "overall_sanity_check_passed": overall_pass,
    }

    print()
    print("=" * 80)
    print("PHYSICAL RIG GEOMETRY SANITY CHECK")
    print("=" * 80)
    print(
        "Measured camera origin in LiDAR frame : "
        f"x={measured_camera_origin_lidar[0]:+.4f} m, "
        f"y={measured_camera_origin_lidar[1]:+.4f} m"
    )
    print(
        "Estimated camera origin in LiDAR frame: "
        f"x={estimated_camera_origin_lidar[0]:+.4f} m, "
        f"y={estimated_camera_origin_lidar[1]:+.4f} m"
    )
    print(
        "Position discrepancy                  : "
        f"{position_error_m:.4f} m"
    )
    print(
        "Measured yaw                          : "
        f"{measured_yaw_deg:+.3f} deg"
    )
    print(
        "Estimated yaw                         : "
        f"{transform_summary['yaw_deg']:+.3f} deg"
    )
    print(
        "Yaw discrepancy                       : "
        f"{yaw_error_deg:+.3f} deg"
    )
    print(
        "Diagnostic sanity check               : "
        f"{'PASS' if overall_pass else 'REVIEW'}"
    )
    print("=" * 80)

    return result


def run_threshold_sensitivity(
    camera_lines: list[np.ndarray],
    raw_lidar_lines: list[np.ndarray],
    initial_tf: np.ndarray,
    thresholds_m: list[float],
    reference_tf: np.ndarray,
) -> list[dict]:
    """
    Independently rerun ICP from the same initial transform at several
    fixed thresholds.

    This is separate from staged ICP. It tests whether the final estimate
    is excessively dependent on one chosen threshold.
    """
    reference_summary = rigid_transform_summary(reference_tf)
    results = []

    print()
    print("=" * 80)
    print("FIXED-THRESHOLD SENSITIVITY TEST")
    print("=" * 80)

    for threshold in thresholds_m:
        initial_lines = transform_lines(raw_lidar_lines, initial_tf)
        line_lengths = [len(line) for line in initial_lines]

        history, stacked_points = icp_2d.icp_per_line(
            camera_lines,
            [line.copy() for line in initial_lines],
            max_iterations=300,
            distance_threshold=threshold,
            convergence_translation_threshold=1e-7,
            convergence_rotation_threshold=1e-7,
            point_pairs_threshold=20,
            verbose=False,
        )

        incremental_tf = compose_icp_history(history)
        test_tf = incremental_tf @ initial_tf

        transformed_lines = split_stacked_lines(
            stacked_points,
            line_lengths,
        )

        residual_result = evaluate_alignment_residuals(
            camera_lines,
            transformed_lines,
            acceptance_threshold_m=0.10,
        )

        summary = rigid_transform_summary(test_tf)

        translation_difference = np.array(
            [
                summary["translation_x_m"]
                - reference_summary["translation_x_m"],
                summary["translation_y_m"]
                - reference_summary["translation_y_m"],
            ]
        )

        result = {
            "distance_threshold_m": float(threshold),
            "iteration_count": len(history),
            "transform_matrix_3x3": test_tf.tolist(),
            "transform_summary": summary,
            "difference_from_staged_result": {
                "translation_difference_x_m": float(
                    translation_difference[0]
                ),
                "translation_difference_y_m": float(
                    translation_difference[1]
                ),
                "translation_difference_magnitude_m": float(
                    np.linalg.norm(translation_difference)
                ),
                "yaw_difference_deg": wrap_angle_deg(
                    summary["yaw_deg"]
                    - reference_summary["yaw_deg"]
                ),
            },
            "overall_residuals": residual_result["overall"],
        }
        results.append(result)

        delta = result["difference_from_staged_result"]

        orthogonal_overall = residual_result[
            "overall"
        ]["orthogonal_point_to_line"]

        orthogonal_stats = orthogonal_overall[
            "all_points"
        ]

        print(
            f"Threshold {threshold:.2f} m: "
            f"iterations={len(history):3d}, "
            f"tx={summary['translation_x_m']:+.4f} m, "
            f"ty={summary['translation_y_m']:+.4f} m, "
            f"yaw={summary['yaw_deg']:+.3f} deg, "
            f"orthogonal RMSE="
            f"{orthogonal_stats['rmse_m']:.4f} m, "
            f"orthogonal median="
            f"{orthogonal_stats['median_m']:.4f} m, "
            f"|delta t|="
            f"{delta['translation_difference_magnitude_m']:.4f} m, "
            f"delta yaw="
            f"{delta['yaw_difference_deg']:+.3f} deg"
        )

    print("=" * 80)
    return results


def save_residual_summary_csv(
    path: str,
    residual_results: dict,
) -> None:
    """
    Save primary orthogonal point-to-line metrics and secondary
    sampled-segment metrics for each pose and overall.
    """
    fieldnames = [
        "pose",
        "acceptance_threshold_m",

        "point_count",

        "orthogonal_accepted_count",
        "orthogonal_accepted_fraction",
        "orthogonal_mean_abs_m",
        "orthogonal_rmse_m",
        "orthogonal_median_m",
        "orthogonal_p95_m",
        "orthogonal_max_m",

        "segment_accepted_count",
        "segment_accepted_fraction",
        "segment_mean_abs_m",
        "segment_rmse_m",
        "segment_median_m",
        "segment_p95_m",
        "segment_max_m",

        "fitted_line_angle_difference_deg",
        "fitted_line_signed_normal_offset_m",
        "fitted_line_absolute_normal_offset_m",
        "lidar_robust_fit_inlier_count",
        "lidar_robust_fit_total_count",
        "lidar_robust_fit_inlier_fraction",
    ]

    rows = []

    for pose_result in residual_results["per_pose"]:
        orthogonal = pose_result[
            "orthogonal_point_to_line"
        ]
        orthogonal_stats = orthogonal["all_points"]

        segment = pose_result[
            "sampled_segment_secondary"
        ]
        segment_stats = segment["all_points"]

        diagnostics = pose_result[
            "fitted_line_diagnostics"
        ]

        rows.append({
            "pose": pose_result["pose"],
            "acceptance_threshold_m": (
                pose_result["acceptance_threshold_m"]
            ),

            "point_count": orthogonal[
                "total_point_count"
            ],

            "orthogonal_accepted_count": orthogonal[
                "accepted_point_count"
            ],
            "orthogonal_accepted_fraction": orthogonal[
                "accepted_fraction"
            ],
            "orthogonal_mean_abs_m": orthogonal_stats[
                "mean_abs_m"
            ],
            "orthogonal_rmse_m": orthogonal_stats[
                "rmse_m"
            ],
            "orthogonal_median_m": orthogonal_stats[
                "median_m"
            ],
            "orthogonal_p95_m": orthogonal_stats[
                "p95_m"
            ],
            "orthogonal_max_m": orthogonal_stats[
                "max_m"
            ],

            "segment_accepted_count": segment[
                "accepted_point_count"
            ],
            "segment_accepted_fraction": segment[
                "accepted_fraction"
            ],
            "segment_mean_abs_m": segment_stats[
                "mean_abs_m"
            ],
            "segment_rmse_m": segment_stats[
                "rmse_m"
            ],
            "segment_median_m": segment_stats[
                "median_m"
            ],
            "segment_p95_m": segment_stats[
                "p95_m"
            ],
            "segment_max_m": segment_stats[
                "max_m"
            ],

            "fitted_line_angle_difference_deg": diagnostics[
                "angular_difference_deg"
            ],
            "fitted_line_signed_normal_offset_m": diagnostics[
                "signed_normal_offset_m"
            ],
            "fitted_line_absolute_normal_offset_m": diagnostics[
                "absolute_normal_offset_m"
            ],
            "lidar_robust_fit_inlier_count": diagnostics[
                "lidar_robust_fit_inlier_count"
            ],
            "lidar_robust_fit_total_count": diagnostics[
                "lidar_robust_fit_total_count"
            ],
            "lidar_robust_fit_inlier_fraction": diagnostics[
                "lidar_robust_fit_inlier_fraction"
            ],
        })

    overall = residual_results["overall"]

    overall_orthogonal = overall[
        "orthogonal_point_to_line"
    ]
    overall_orthogonal_stats = overall_orthogonal[
        "all_points"
    ]

    overall_segment = overall[
        "sampled_segment_secondary"
    ]
    overall_segment_stats = overall_segment[
        "all_points"
    ]

    rows.append({
        "pose": "overall",
        "acceptance_threshold_m": overall[
            "acceptance_threshold_m"
        ],

        "point_count": overall_orthogonal[
            "total_point_count"
        ],

        "orthogonal_accepted_count": overall_orthogonal[
            "accepted_point_count"
        ],
        "orthogonal_accepted_fraction": overall_orthogonal[
            "accepted_fraction"
        ],
        "orthogonal_mean_abs_m": overall_orthogonal_stats[
            "mean_abs_m"
        ],
        "orthogonal_rmse_m": overall_orthogonal_stats[
            "rmse_m"
        ],
        "orthogonal_median_m": overall_orthogonal_stats[
            "median_m"
        ],
        "orthogonal_p95_m": overall_orthogonal_stats[
            "p95_m"
        ],
        "orthogonal_max_m": overall_orthogonal_stats[
            "max_m"
        ],

        "segment_accepted_count": overall_segment[
            "accepted_point_count"
        ],
        "segment_accepted_fraction": overall_segment[
            "accepted_fraction"
        ],
        "segment_mean_abs_m": overall_segment_stats[
            "mean_abs_m"
        ],
        "segment_rmse_m": overall_segment_stats[
            "rmse_m"
        ],
        "segment_median_m": overall_segment_stats[
            "median_m"
        ],
        "segment_p95_m": overall_segment_stats[
            "p95_m"
        ],
        "segment_max_m": overall_segment_stats[
            "max_m"
        ],

        "fitted_line_angle_difference_deg": None,
        "fitted_line_signed_normal_offset_m": None,
        "fitted_line_absolute_normal_offset_m": None,
        "lidar_robust_fit_inlier_count": None,
        "lidar_robust_fit_total_count": None,
        "lidar_robust_fit_inlier_fraction": None,
    })

    with open(path, "w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)

def main():
  parser = argparse.ArgumentParser(description="Calibrate image and laser extrinsics from a collection of checkerboard images and laser scans.")
  parser.add_argument("image_dir", help="Image directory.")
  parser.add_argument("laser_dir", help="Laser directory.")
  parser.add_argument(
      "--camera-manifest",
      required=True,
      help="Capture session's session_manifest.json; the rectified LEFT "
           "intrinsics (calibration.rectified.left) are read from it.",
  )
  parser.add_argument(
      "--init-camera-origin-in-lidar",
      nargs=2,
      type=float,
      required=True,
      metavar=("X_FORWARD_M", "Y_LEFT_M"),
      help="Rough manual measurement of the camera centre in the LiDAR "
           "frame. Used as the ICP initial transform and as the rig "
           "sanity-check reference.",
  )
  parser.add_argument(
      "--init-yaw-deg",
      type=float,
      required=True,
      help="Rough relative yaw of the LiDAR frame to the camera robot "
           "frame, in degrees. Used with --init-camera-origin-in-lidar.",
  )
  parser.add_argument(
      "--out-dir",
      default=None,
      help="Output directory. Defaults to results/calibration/<session>, "
           "where <session> is the parent folder name of image_dir.",
  )
  args = parser.parse_args()

  image_dir = args.image_dir
  laser_dir = args.laser_dir

  out_dir = Path(
      args.out_dir
      if args.out_dir is not None
      else Path("results") / "calibration" / Path(image_dir).resolve().parent.name
  )
  out_dir.mkdir(parents=True, exist_ok=True)
  print("Writing outputs to: " + str(out_dir))

  # Load images and pcb clouds from file, after they are extracted from a rosbag and selected for calibration
  # They have to have one to one correspondences - that is usually true when they are ordered in each folder correctly
  # The load functions will have print outs for order verification
  images = load_images_from_folder(image_dir)
  lasers = load_clouds_from_folder(laser_dir)

  assert len(images) == len(lasers), "Images and lasers length mismatch!"

  # Rectified LEFT intrinsics for this capture session, from the ZED SDK.
  # Distortion is zero because ZED SDK rectified images are used.
  camera_k, camera_dist, expected_image_wh = load_rectified_left_intrinsics(
      args.camera_manifest
  )
  print("Rectified LEFT intrinsics from " + args.camera_manifest + ":")
  print(camera_k)

  # termination criteria, for aligning checkerboard corners onto an image
  criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

  # Checkerboard shape, example: 7*10 in checkerboard blocks, 20 mm in checkerboard block size
  # TODO: Change the checkerboard parameters into those that match with the actual board you are using
  checkerboard_height = 3
  checkerboard_width = 6 
  checkerboard_size = 0.050
  checkerboard_points = np.zeros((checkerboard_width*checkerboard_height, 3), np.float32)
  checkerboard_points[:, :2] = np.mgrid[0:checkerboard_width, 0:checkerboard_height].T.reshape(-1,2)*checkerboard_size

  # Extent of the camera-derived line along the board X axis, measured from
  # the first inner corner. The printed pattern spans
  # [-checkerboard_size, checkerboard_width * checkerboard_size]; the margin
  # must exceed the board's white border so every LiDAR return on the board
  # has a line point beside it (a line shorter than the board drags the
  # outermost returns toward the line endpoint during point-to-point ICP).
  camera_line_margin = 0.10
  camera_line_start = -checkerboard_size - camera_line_margin
  camera_line_end = checkerboard_width * checkerboard_size + camera_line_margin

  # A predefined 3D axis for visualisation, axis length 10 cm
  axis = np.float32([[0.1,0,0], [0,0.1,0], [0,0,0.1]]).reshape(-1,3)

  # Collected/extracted camera and LiDAR points in 2D - supposed to be aligned with each other
  camera_points = []
  laser_points = []

  # Per-pose solvePnP rvec/tvec, retained (in addition to the loop-local use
  # above) only for the correspondence dump below -- PLAN.md Sec 4.2.
  pose_rvecs = []
  pose_tvecs = []

  # Now knowing that list 'images' and 'lasers' are of the same length, loop through them at the same time
  # to extract the corresponding 2D line points for alignment
  for i, (this_image, this_laser) in enumerate(zip(images, lasers)):
    # print(i) 
  
    # Extract the checkerboard from the image and extract a line that represent the checkerboard in 2D
    # But first! Undistortion - without undistortion, the projection of linear structure into the 3D space will be distorted too
    # Distortion effect means that a straight line in 3D is not a straight line in the camera view
    # The attempt to extract a straight line on the checkerboard from the camera view will therefore be affected
    h, w = this_image.shape[:2]

    if (w, h) != expected_image_wh:
      raise ValueError(
        f"Expected images of size {expected_image_wh[0]}x{expected_image_wh[1]} "
        f"(from {args.camera_manifest}), but got {w}x{h}. "
        "Use rectified LEFT images from the same capture session as the manifest."
      )

    # For repository smoke test/example data, avoid applying possibly mismatched
    # hardcoded distortion parameters before checkerboard detection.
    new_camera_k = camera_k.copy()
    new_camera_dist = camera_dist.copy()
    # new_camera_dist = np.zeros((1, 5))
    undistorted_image = this_image.copy()

    gray = cv2.cvtColor(undistorted_image, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (checkerboard_width, checkerboard_height), None)

    if ret == True:
      corners2 = cv2.cornerSubPix(gray, corners, (3,3), (-1,-1), criteria)
      # Find the rotation and translation vectors (pose) between the board and the camera.
      ret_pnp, rvecs, tvecs = cv2.solvePnP(checkerboard_points, corners2, new_camera_k, new_camera_dist)
      pose_rvecs.append(rvecs.copy())
      pose_tvecs.append(tvecs.copy())

      projected, _ = cv2.projectPoints(checkerboard_points, rvecs, tvecs, new_camera_k, new_camera_dist)
      err = np.linalg.norm(projected.reshape(-1, 2) - corners2.reshape(-1, 2), axis=1)
      print(f"Image {i}: solvePnP={ret_pnp}, reprojection mean={err.mean():.3f}px, max={err.max():.3f}px")

      # draw checkerboard corners for introspection
      cv2.drawChessboardCorners(undistorted_image, (checkerboard_width, checkerboard_height), corners2, ret)
      # project 3D axis to image plane for introspection
      imgpts, jacobian = cv2.projectPoints(axis, rvecs, tvecs, new_camera_k, new_camera_dist)
      undistorted_image = draw(undistorted_image, corners2, imgpts)

      # # Visualisation - OpenCV
      # cv2.imshow('Image with checkerboard axis', undistorted_image) # Note the direction of the z axis
      # cv2.waitKey(500)

      # Visualisation - Interactive and extract horizontal line
      # This is where a horizontal line is actually estimated based on the checkerboard pose
      visualise_camera_interface = ImageVisInterface(
          rvecs, tvecs, undistorted_image, camera_points,
          line_start=camera_line_start, line_end=camera_line_end,
      )
      confirmed, camera_points = visualise_camera_interface.run()
      if confirmed == False:
        print("Something wrong with this image?")
        return
      # else:
      #   print(camera_points) # For introspection only
    else: 
      print("Unable to extract a checkerboard - are you using the correct checkerboard parameters?")
      return

    # Extract the line that correspond to the checkerboard from the LiDAR scan
    # Using an interactive interface
    this_laser_points = np.asarray(this_laser.points)
    select_points_interface = SelectPointsInterface(this_laser_points, laser_points)
    laser_points = select_points_interface.run()
    # print(laser_points)

  # Concatenate the list of arrays into long arrays
  all_lidar_points = np.vstack(laser_points)
  all_camera_points = np.vstack(camera_points)

  # ------------------------------------------------------------------
  # Additive correspondence dump (PLAN.md Sec 4.2). Per pose: the selected
  # LiDAR points (Nx2), the checkerboard-derived wall line points (Mx2),
  # the solvePnP rvec/tvec, and the pose id. Purely additive: no existing
  # output is altered and nothing above this point (or below, besides the
  # np.savez call itself) is changed by its presence. Ragged per-pose point
  # counts are stored as indexed keys rather than a single object-dtype
  # array, so the file stays a plain, portable .npz.
  # ------------------------------------------------------------------
  correspondence_payload = {
      "pose_count": np.array(len(laser_points)),
  }
  for pose_index in range(len(laser_points)):
    correspondence_payload[f"pose_{pose_index:02d}_lidar_points"] = laser_points[pose_index]
    correspondence_payload[f"pose_{pose_index:02d}_camera_line_points"] = camera_points[pose_index]
    correspondence_payload[f"pose_{pose_index:02d}_rvec"] = pose_rvecs[pose_index]
    correspondence_payload[f"pose_{pose_index:02d}_tvec"] = pose_tvecs[pose_index]

  np.savez(out_dir / "calibration_correspondences.npz", **correspondence_payload)
  print("Saved calibration_correspondences.npz")

  # Introspection - are the two sets of points, camera_points and laser_points, looking reasonable with each other?
  print("Saving pre-alignment diagnostic plot...")

  fig, ax = plt.subplots()
  ax.scatter(all_lidar_points[:,0], all_lidar_points[:,1], c='blue', label='All 2D LiDAR Points')
  ax.scatter(all_camera_points[:,0], all_camera_points[:,1], c='green', label='All 2D Camera Points')
  ax.legend()
  ax.set_xlabel('x')
  ax.set_ylabel('y')
  ax.set_title('Checkerboard in Image (Green) and LiDAR (Blue)')
  ax.set_aspect('equal', adjustable='box')
  fig.savefig(out_dir / "checkerboard_lidar_pre_alignment.png", dpi=200, bbox_inches="tight")
  plt.close(fig)

  print("Saved checkerboard_lidar_pre_alignment.png")

  # ------------------------------------------------------------------
  # Measurement-derived initial LiDAR-to-camera estimate
  # ------------------------------------------------------------------
  #
  # Rough manual camera centre position expressed in the LiDAR frame
  # (x forward, y left) and relative yaw, from the command line.
  #
  # For:
  #   p_camera = R_lidar_to_camera @ p_lidar + t_lidar_to_camera
  #
  # the camera origin in the LiDAR frame is -R^T t, so:
  #   t_lidar_to_camera = -R @ camera_origin_in_lidar
  #
  measured_camera_origin_in_lidar = np.array(
      args.init_camera_origin_in_lidar,
      dtype=np.float64,
  )
  measured_yaw_deg = args.init_yaw_deg

  initial_yaw_rad = math.radians(measured_yaw_deg)
  initial_rotation = np.array([
      [math.cos(initial_yaw_rad), -math.sin(initial_yaw_rad)],
      [math.sin(initial_yaw_rad), math.cos(initial_yaw_rad)],
  ], dtype=np.float64)
  initial_translation = (
      -initial_rotation @ measured_camera_origin_in_lidar
  )

  initial_tf = np.eye(3, dtype=np.float64)
  initial_tf[:2, :2] = initial_rotation
  initial_tf[:2, 2] = initial_translation

  print()
  print("Measurement-derived initial LiDAR-to-camera transform:")
  print(initial_tf)

  # ------------------------------------------------------------------
  # Coarse-to-fine staged ICP
  # ------------------------------------------------------------------
  #
  # 0.30 m: broad initial capture range
  # 0.15 m: intermediate refinement
  # 0.10 m: final tighter refinement
  #
  icp_stages = [
      {
          "distance_threshold_m": 0.30,
          "max_iterations": 150,
          "point_pairs_threshold": 20,
      },
      {
          "distance_threshold_m": 0.15,
          "max_iterations": 150,
          "point_pairs_threshold": 20,
      },
      {
          "distance_threshold_m": 0.10,
          "max_iterations": 150,
          "point_pairs_threshold": 20,
      },
  ]

  tf_total, transformed_lidar_lines, staged_icp_results = run_staged_icp(
      camera_lines=camera_points,
      raw_lidar_lines=laser_points,
      initial_tf=initial_tf,
      stages=icp_stages,
  )

  print()
  print("The final result of staged 2D ICP:")
  print(tf_total)

  transform_summary = rigid_transform_summary(tf_total)

  print(
      "Transform summary: "
      f"tx={transform_summary['translation_x_m']:+.6f} m, "
      f"ty={transform_summary['translation_y_m']:+.6f} m, "
      f"yaw={transform_summary['yaw_deg']:+.6f} deg"
  )

  # ------------------------------------------------------------------
  # Quantitative post-alignment residual validation
  # ------------------------------------------------------------------
  final_residual_threshold_m = 0.10

  residual_results = evaluate_alignment_residuals(
      camera_lines=camera_points,
      transformed_lidar_lines=transformed_lidar_lines,
      acceptance_threshold_m=final_residual_threshold_m,
  )

  save_residual_summary_csv(
      out_dir / "calibration_residual_summary.csv",
      residual_results,
  )
  print("Saved calibration_residual_summary.csv")

  # ------------------------------------------------------------------
  # Physical mounting sanity check
  # ------------------------------------------------------------------
  physical_geometry_validation = (
      validate_against_manual_rig_geometry(
          tf_total,
          measured_camera_origin_in_lidar,
          measured_yaw_deg,
      )
  )

  # ------------------------------------------------------------------
  # Independent fixed-threshold sensitivity test
  # ------------------------------------------------------------------
  #
  # Each run starts again from the same physical initial transform.
  # These are not chained stages.
  #
  threshold_sensitivity_results = run_threshold_sensitivity(
      camera_lines=camera_points,
      raw_lidar_lines=laser_points,
      initial_tf=initial_tf,
      thresholds_m=[0.20, 0.25, 0.30, 0.35],
      reference_tf=tf_total,
  )  

  # Save calibration result for downstream evaluation.
  result_payload = {
      "created_utc": datetime.utcnow().isoformat() + "Z",
      "method": "camera_2d_lidar_staged_custom_icp",
      "transform_direction": (
          "lidar_2d_to_camera_ground_plane_2d"
      ),
      "transform_matrix_3x3": tf_total.tolist(),
      "transform_summary": transform_summary,

      "camera_intrinsics": camera_k.tolist(),
      "camera_distortion": camera_dist.tolist(),
      "camera_intrinsics_source": str(args.camera_manifest),
      "image_size_wh": list(expected_image_wh),

      "checkerboard_width_inner_corners": checkerboard_width,
      "checkerboard_height_inner_corners": checkerboard_height,
      "checkerboard_square_size_m": checkerboard_size,
      "camera_line_extent_along_board_x_m": [
          camera_line_start,
          camera_line_end,
      ],

      "image_dir": str(image_dir),
      "laser_dir": str(laser_dir),

      "image_count": len(images),
      "laser_count": len(lasers),

      "initial_transform": {
          "source": "rough_manual_rig_measurement",
          "manual_camera_origin_in_lidar_m": {
              "x_forward": float(measured_camera_origin_in_lidar[0]),
              "y_left": float(measured_camera_origin_in_lidar[1]),
          },
          "manual_relative_yaw_deg": measured_yaw_deg,
          "transform_matrix_3x3": initial_tf.tolist(),
      },

      "staged_icp": {
          "stages": staged_icp_results,
          "final_residual_acceptance_threshold_m": (
              final_residual_threshold_m
          ),
      },

      "residual_validation": residual_results,

      "physical_geometry_validation": (
          physical_geometry_validation
      ),

      "threshold_sensitivity": (
          threshold_sensitivity_results
      ),

      "notes": (
          "Rectified left image stream; intrinsics are the ZED SDK "
          "rectified left values from camera_intrinsics_source. "
          "Distortion coefficients are zero for the rectified SDK "
          "output. ICP nearest-neighbour searches are isolated per "
          "pose. The primary reported validation residual is the "
          "orthogonal distance from each transformed LiDAR point to "
          "the corresponding infinite camera-derived TLS line. "
          "Nearest sampled camera-segment distances are retained as "
          "a secondary diagnostic for endpoint and sampling effects. "
          "Manual rig measurements are diagnostic references rather "
          "than surveyed ground truth."
      ),
  }

  np.save(out_dir / "lidar_to_camera_2d.npy", tf_total)

  with open(out_dir / "calibration_result.json", "w") as f:
      json.dump(result_payload, f, indent=2)

  print("Saved lidar_to_camera_2d.npy")
  print("Saved calibration_result.json")

  # Introspection - the end result of the alignment
  print("Saving aligned point cloud diagnostic plot...")

  fig, ax = plt.subplots()

  transformed = np.vstack(transformed_lidar_lines)

  ax.scatter(all_camera_points[:,0], all_camera_points[:,1], c='green', label='All 2D Camera Points')
  # ax.scatter(transformed[:,0], transformed[:,1], c='red', label='All 2D LiDAR Points, Open3D')
  # ax.scatter(points[:,0], points[:,1], c='blue', label='All 2D LiDAR Points, custom function')
  ax.scatter(transformed[:,0], transformed[:,1], c='blue', label='All 2D LiDAR Points, staged custom ICP')
  ax.legend()
  ax.set_xlabel('x')
  ax.set_ylabel('y')
  ax.set_title('Detected Wall - Aligned')
  ax.set_aspect('equal', adjustable='box')
  fig.savefig(out_dir / "aligned_point_clouds.png", dpi=200, bbox_inches="tight")
  plt.close(fig)

  print("Saved aligned_point_clouds.png")

if __name__ == '__main__':
    main()

