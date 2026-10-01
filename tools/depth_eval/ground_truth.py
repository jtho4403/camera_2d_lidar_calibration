"""Load exported sparse ground truth (tools/lidar_ground_truth/export_scene.py
output) for the depth-model evaluation harness.

Deliberately thin: the export CSVs are already clean and complete (every
coordinate, uncertainty, and provenance field the harness needs), so this
module only groups rows by capture_id and resolves each capture's svo2 source
-- no reshaping of the ground truth itself.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import config


@dataclass
class CaptureGroundTruth:
    scene_id: str
    capture_id: str
    svo_path: Path
    pixel_u: np.ndarray
    pixel_v: np.ndarray
    camera_depth_m: np.ndarray
    depth_std_m: np.ndarray


@dataclass
class SceneGroundTruth:
    scene_id: str
    K: np.ndarray            # (3, 3) rectified left intrinsics, the frame GT u/v are in
    image_wh: tuple[int, int]
    fx_px: float
    captures: list[CaptureGroundTruth]


def _resolve_svo_path(session: Path, capture_id: str) -> Path:
    matches = sorted((session / "captures").glob(f"**/{capture_id}/{capture_id}.svo2"))
    if not matches:
        raise FileNotFoundError(f"No {capture_id}.svo2 found under {session / 'captures'}")
    if len(matches) > 1:
        raise ValueError(f"Multiple {capture_id}.svo2 found under {session / 'captures'}: {matches}")
    return matches[0]


def load_scene(session: str | Path, export_dir: str | Path, scene_name: str) -> SceneGroundTruth:
    """Load one scene's exported ground truth (e.g. scene_name="scene_B" for
    data_2026-09-24_scene_B_ground_truth.csv / _summary.json)."""
    session = Path(session)
    export_dir = Path(export_dir)
    scene_id = f"{session.name}_{scene_name}"

    csv_path = export_dir / f"{scene_id}_ground_truth.csv"
    summary_path = export_dir / f"{scene_id}_ground_truth_summary.json"
    df = pd.read_csv(csv_path)
    summary = json.loads(summary_path.read_text())

    K = np.array(summary["K"], dtype=np.float64)
    image_wh = tuple(int(v) for v in summary["image_wh"])

    captures = []
    for capture_id, group in df.groupby("capture_id"):
        captures.append(CaptureGroundTruth(
            scene_id=scene_id,
            capture_id=str(capture_id),
            svo_path=_resolve_svo_path(session, str(capture_id)),
            pixel_u=group["pixel_u"].to_numpy(dtype=np.float64),
            pixel_v=group["pixel_v"].to_numpy(dtype=np.float64),
            camera_depth_m=group["camera_depth_m"].to_numpy(dtype=np.float64),
            depth_std_m=group["depth_std_m"].to_numpy(dtype=np.float64),
        ))
    captures.sort(key=lambda c: c.capture_id)

    return SceneGroundTruth(scene_id=scene_id, K=K, image_wh=image_wh, fx_px=float(K[0, 0]), captures=captures)
