"""ZED SDK depth-mode predictor: opens a capture's raw .svo2 and retrieves a
dense depth map from one of the SDK's proprietary depth modes (NEURAL,
NEURAL_PLUS, NEURAL_LIGHT, QUALITY, ULTRA, PERFORMANCE).

Reads directly from .svo2 rather than any staged image: SVO2 stores raw
sensor data plus factory calibration (MASTER_PLAN.md 1.24), so this is the
one place a fresh, single, non-burst-averaged frame can be pulled for every
scene uniformly. The staged calibration images (data/<session>/images/) are
not used here -- Scene A's are a ~150-frame temporal mean (tools/
extract_burst_mean_images.py), deliberately unrepresentative of a single
real-time inference call, which is what this harness evaluates.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyzed.sl as sl

import config


@dataclass
class ZedPrediction:
    depth_m: np.ndarray       # (H, W) float32; invalid entries are non-finite or <= 0
    frame_index: int
    extracted_K: dict         # for the caller's own K-consistency check against GT.K


DEPTH_MODE_BY_NAME = {
    "NEURAL": sl.DEPTH_MODE.NEURAL,
    "NEURAL_PLUS": sl.DEPTH_MODE.NEURAL_PLUS,
    "NEURAL_LIGHT": sl.DEPTH_MODE.NEURAL_LIGHT,
    "QUALITY": sl.DEPTH_MODE.QUALITY,
    "ULTRA": sl.DEPTH_MODE.ULTRA,
    "PERFORMANCE": sl.DEPTH_MODE.PERFORMANCE,
}


def predict(svo_path, depth_mode_name: str, frame_index: int = config.EVAL_FRAME_INDEX) -> ZedPrediction:
    if depth_mode_name not in DEPTH_MODE_BY_NAME:
        raise ValueError(f"Unknown ZED depth mode {depth_mode_name!r}; choose from {sorted(DEPTH_MODE_BY_NAME)}")

    init = sl.InitParameters()
    init.set_from_svo_file(str(svo_path))
    init.svo_real_time_mode = False
    init.depth_mode = DEPTH_MODE_BY_NAME[depth_mode_name]
    init.camera_disable_self_calib = True
    # SDK default coordinate_units is MILLIMETER, not METER -- must be explicit, since
    # it also governs the units depth_minimum/maximum_distance below are interpreted in,
    # and ground truth (camera_depth_m) and config.py's range constants are both metres.
    init.coordinate_units = sl.UNIT.METER
    init.depth_minimum_distance = config.ZED_SDK_DEPTH_MINIMUM_DISTANCE_M
    init.depth_maximum_distance = config.ZED_SDK_DEPTH_MAXIMUM_DISTANCE_M

    cam = sl.Camera()
    status = cam.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"{svo_path}: cam.open() failed for depth_mode={depth_mode_name}: {status}")

    try:
        left_cam = cam.get_camera_information().camera_configuration.calibration_parameters.left_cam
        extracted_K = {
            "fx": float(left_cam.fx), "fy": float(left_cam.fy),
            "cx": float(left_cam.cx), "cy": float(left_cam.cy),
            "image_size": (int(left_cam.image_size.width), int(left_cam.image_size.height)),
        }

        runtime = sl.RuntimeParameters()
        runtime.confidence_threshold = config.ZED_SDK_CONFIDENCE_THRESHOLD
        runtime.texture_confidence_threshold = config.ZED_SDK_TEXTURE_CONFIDENCE_THRESHOLD
        runtime.enable_fill_mode = config.ZED_SDK_ENABLE_FILL_MODE

        depth_mat = sl.Mat()
        for i in range(frame_index + 1):
            grab_status = cam.grab(runtime)
            if grab_status == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
                raise RuntimeError(
                    f"{svo_path}: reached end of SVO2 at frame {i}, before the "
                    f"configured eval frame index {frame_index} (config.EVAL_FRAME_INDEX)"
                )
            if grab_status != sl.ERROR_CODE.SUCCESS:
                raise RuntimeError(f"{svo_path}: grab() failed at frame {i}: {grab_status}")
        cam.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)
        depth_m = depth_mat.numpy().astype(np.float32)
    finally:
        cam.close()

    return ZedPrediction(depth_m=depth_m, frame_index=frame_index, extracted_K=extracted_K)


def check_K_matches(extracted_K: dict, gt_K: np.ndarray, tolerance_px: float = config.K_MATCH_TOLERANCE_PX) -> None:
    """Refuse to evaluate with a K that differs from the one the ground truth
    was derived with -- same paranoia, same reasoning, as
    tools/lidar_ground_truth/calibration_io.check_intrinsics_match_calibration.
    """
    delta = max(
        abs(extracted_K["fx"] - gt_K[0, 0]), abs(extracted_K["fy"] - gt_K[1, 1]),
        abs(extracted_K["cx"] - gt_K[0, 2]), abs(extracted_K["cy"] - gt_K[1, 2]),
    )
    if delta > tolerance_px:
        raise ValueError(
            f"ZED SDK's live-extracted K {extracted_K} differs from the ground truth's "
            f"K {gt_K.tolist()} by {delta:.4f} px (tolerance {tolerance_px} px) -- "
            "refusing to sample a depth map against a mismatched K."
        )
