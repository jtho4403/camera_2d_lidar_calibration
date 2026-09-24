#!/usr/bin/env python3
"""Extract Fix-1 burst-mean rectified-left images from a scene's SVO2 captures.

EXPERIMENT_DESIGN_v3.md Sec 4, "Downstream requirement: burst averaging (Fix 1)":
with only 18 checkerboard corners, single-frame PnP gives a scan-line plane
error of roughly 12 mm RMS (25 mm p95) at a far frontal pose; averaging over
the burst (~150 frames) cuts random corner noise by roughly sqrt(N) --
predicted 5.5 mm -> 0.5 mm at 1.5 m frontal. This script replaces the single
QC-snapshot image staged by stage_calibration_session.py
(<session>/images/<ID>_left.png) with a temporal-mean image computed from
every frame in <ID>.svo2, written to that same path so cam_lidar_2d_icp.py
needs no changes downstream.

Requires pyzed and a working NVIDIA GPU/driver. Any SDK version >= the
capture session's is fine (MASTER_PLAN.md 1.24): SVO2 stores raw frames plus
factory calibration, and rectification happens here, at extraction time --
this script cross-checks its extracted rectified K against the session's
session_manifest.json and refuses to proceed on a mismatch, the same
paranoia tools/lidar_ground_truth/calibration_io.py applies downstream.

Only Scene A needs this: it is the only scene whose images feed solvePnP
corner localisation. Scenes B/C keep the single-frame staging from
stage_calibration_session.py (no checkerboard geometry is extracted from
their images).

Run from the repository root, inside an environment with pyzed installed:
    python tools/extract_burst_mean_images.py data/<session> --scenes scene_A
"""
from __future__ import annotations

from pathlib import Path
import argparse
import json

import cv2
import numpy as np
import pyzed.sl as sl


def find_capture_dirs(captures_root: Path) -> list[Path]:
    """Recursively find capture directories under captures_root (flat or
    scene-nested layout), mirroring tools/stage_calibration_session.py.
    """
    dirs = {
        metadata_file.parent
        for metadata_file in captures_root.rglob("*_metadata.json")
        if metadata_file.name == f"{metadata_file.parent.name}_metadata.json"
    }
    return sorted(dirs)


def scene_group(capture_dir: Path, captures_root: Path) -> str | None:
    parts = capture_dir.relative_to(captures_root).parts
    return parts[0] if len(parts) > 1 else None


def load_manifest_K(manifest_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text())
    left = manifest["calibration"]["rectified"]["left"]
    return {
        "fx": float(left["fx"]), "fy": float(left["fy"]),
        "cx": float(left["cx"]), "cy": float(left["cy"]),
        "image_size": tuple(int(v) for v in left["image_size"]),
    }


def extract_burst_mean(svo_path: Path, min_frames: int) -> tuple[np.ndarray, dict]:
    """Open an SVO2 recording, grab every frame, and return the per-pixel
    temporal mean of the rectified left view (BGR, uint8) plus provenance.
    """
    init = sl.InitParameters()
    init.set_from_svo_file(str(svo_path))
    init.svo_real_time_mode = False
    init.depth_mode = sl.DEPTH_MODE.NONE
    init.camera_disable_self_calib = True  # match capture-time init_parameters

    cam = sl.Camera()
    status = cam.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"{svo_path}: cam.open() failed: {status}")

    try:
        left_cam = cam.get_camera_information().camera_configuration.calibration_parameters.left_cam
        extracted_K = {
            "fx": float(left_cam.fx), "fy": float(left_cam.fy),
            "cx": float(left_cam.cx), "cy": float(left_cam.cy),
            "image_size": (int(left_cam.image_size.width), int(left_cam.image_size.height)),
        }
        distortion_nonzero = any(float(d) != 0.0 for d in left_cam.disto)

        image = sl.Mat()
        runtime = sl.RuntimeParameters()
        accumulator = None
        n_frames = 0
        while True:
            grab_status = cam.grab(runtime)
            if grab_status == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
                break
            if grab_status != sl.ERROR_CODE.SUCCESS:
                raise RuntimeError(f"{svo_path}: grab() failed at frame {n_frames}: {grab_status}")
            cam.retrieve_image(image, sl.VIEW.LEFT)
            frame_bgr = image.numpy()[:, :, :3].astype(np.float64)  # BGRA -> BGR
            accumulator = frame_bgr if accumulator is None else accumulator + frame_bgr
            n_frames += 1
    finally:
        cam.close()

    if n_frames < min_frames:
        raise RuntimeError(f"{svo_path}: only {n_frames} frames grabbed, need >= {min_frames}")

    mean_image = np.clip(np.round(accumulator / n_frames), 0, 255).astype(np.uint8)
    provenance = {
        "n_frames": n_frames,
        "extracted_rectified_K": extracted_K,
        "extracted_distortion_nonzero": distortion_nonzero,
        "extraction_sdk_version": str(sl.Camera().get_sdk_version()),
    }
    return mean_image, provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("session", help="Session root, e.g. data/data_2026-09-24.")
    parser.add_argument("--scenes", nargs="*", default=None, help="Stage only these scene-group dirs (e.g. --scenes scene_A).")
    parser.add_argument("--min-frames", type=int, default=100, help="EXPERIMENT_DESIGN_v3.md Fix 1: >= 100 frames.")
    parser.add_argument(
        "--camera-manifest", default=None,
        help="session_manifest.json to cross-check the extracted rectified K against. "
             "Default: <session>/captures/session_manifest.json (flat layout); scene-nested "
             "sessions must pass this explicitly (or rely on --scenes matching a single scene "
             "and its per-scene manifest being auto-resolved).",
    )
    parser.add_argument("--k-tolerance-px", type=float, default=0.5)
    parser.add_argument("--overwrite", action="store_true", help="Replace an already-extracted image.")
    args = parser.parse_args()

    session = Path(args.session)
    captures_root = session / "captures"
    image_dir = session / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    capture_dirs = find_capture_dirs(captures_root)
    if args.scenes is not None:
        capture_dirs = [d for d in capture_dirs if scene_group(d, captures_root) in args.scenes]
        if not capture_dirs:
            raise ValueError(f"--scenes {args.scenes} matched no capture dirs under {captures_root}")

    if args.camera_manifest:
        manifest_K = load_manifest_K(Path(args.camera_manifest))
    elif args.scenes is not None and len(args.scenes) == 1 and (captures_root / args.scenes[0] / "session_manifest.json").exists():
        manifest_K = load_manifest_K(captures_root / args.scenes[0] / "session_manifest.json")
    else:
        manifest_K = load_manifest_K(captures_root / "session_manifest.json")

    records = []
    for capture_dir in capture_dirs:
        capture_id = capture_dir.name
        out_path = image_dir / f"{capture_id}_left.png"
        if out_path.exists() and not args.overwrite:
            print(f"{capture_id}: {out_path} already exists, skipping (pass --overwrite to redo it).")
            continue

        svo_path = capture_dir / f"{capture_id}.svo2"
        mean_image, provenance = extract_burst_mean(svo_path, args.min_frames)

        extracted = provenance["extracted_rectified_K"]
        if provenance["extracted_distortion_nonzero"]:
            raise ValueError(f"{capture_id}: extracted rectified left distortion is non-zero; refusing to use it.")
        k_delta = max(
            abs(extracted["fx"] - manifest_K["fx"]), abs(extracted["fy"] - manifest_K["fy"]),
            abs(extracted["cx"] - manifest_K["cx"]), abs(extracted["cy"] - manifest_K["cy"]),
        )
        if k_delta > args.k_tolerance_px:
            raise ValueError(
                f"{capture_id}: extracted rectified K {extracted} differs from "
                f"session_manifest.json's {manifest_K} by {k_delta:.4f} px "
                f"(tolerance {args.k_tolerance_px} px) -- refusing to use a mismatched K."
            )
        if tuple(extracted["image_size"]) != tuple(manifest_K["image_size"]):
            raise ValueError(f"{capture_id}: extracted image size {extracted['image_size']} != manifest {manifest_K['image_size']}")

        cv2.imwrite(str(out_path), mean_image)

        record = {
            "capture_id": capture_id,
            "svo_source": str(svo_path),
            "image": str(out_path),
            **provenance,
        }
        records.append(record)
        print(f"{capture_id}: averaged {provenance['n_frames']} frames -> {out_path} (K matched manifest within {k_delta:.4f} px)")

    manifest_path = session / "burst_mean_manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())["captures"]
        existing = {r["capture_id"]: r for r in existing}
    else:
        existing = {}
    for r in records:
        existing[r["capture_id"]] = r
    manifest_path.write_text(json.dumps({
        "session": str(session),
        "min_frames": args.min_frames,
        "k_tolerance_px": args.k_tolerance_px,
        "captures": [existing[k] for k in sorted(existing)],
    }, indent=2))
    print(f"Extracted {len(records)} burst-mean images; wrote {manifest_path}")


if __name__ == "__main__":
    main()
