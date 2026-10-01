#!/usr/bin/env python3
"""Evaluate a depth predictor against this project's sparse LiDAR-referenced
ground truth (tools/lidar_ground_truth/export_scene.py output), using the
metric set in docs/DEPTH_METRICS.md (metrics.py).

Only the ZED SDK depth-mode backend is implemented so far (the immediate
priority: evaluable now, without the TensorRT/Jetson build work that gates
the open-source candidate models). Every ground-truth point in scope is
sampled (bilinearly, sampling.py) from the predictor's dense depth map;
metrics are reported overall, per scene, per depth bucket, and per capture --
never only as one aggregate number, since both this project's own ground
truth (REPORT.md Sec 5) and depth models in general are known to vary with
distance.

Run from the repository root:
    python tools/depth_eval/evaluate.py \\
        --session data/data_2026-09-24 \\
        --scenes scene_B scene_C \\
        --backend zed_sdk --depth-mode NEURAL \\
        --out-dir results/depth_eval/data_2026-09-24/zed_sdk_NEURAL
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import config
import ground_truth
import metrics as metrics_module
import sampling
import zed_sdk_predictor


def evaluate_capture(capture: ground_truth.CaptureGroundTruth, gt_K: np.ndarray, args) -> dict:
    if args.backend == "zed_sdk":
        prediction = zed_sdk_predictor.predict(capture.svo_path, args.depth_mode, frame_index=args.frame_index)
        zed_sdk_predictor.check_K_matches(prediction.extracted_K, gt_K)
        depth_map = prediction.depth_m
        provenance = {"backend": "zed_sdk", "depth_mode": args.depth_mode, "frame_index": prediction.frame_index}
    else:
        raise ValueError(f"Unknown backend {args.backend!r}")

    pred_depth, sample_valid = sampling.bilinear_sample_depth(depth_map, capture.pixel_u, capture.pixel_v)
    # also reject predictions outside the declared operating range, the same filter
    # tools/lidar_ground_truth/export_scene.py already applies to the ground-truth side
    in_range = (pred_depth >= config.ZED_SDK_DEPTH_MINIMUM_DISTANCE_M) & (pred_depth <= config.ZED_SDK_DEPTH_MAXIMUM_DISTANCE_M)
    valid = sample_valid & in_range

    buckets = metrics_module.assign_buckets(capture.camera_depth_m)

    return {
        "capture_id": capture.capture_id,
        "scene_id": capture.scene_id,
        "provenance": provenance,
        "gt_depth_m": capture.camera_depth_m,
        "pred_depth_m": pred_depth,
        "valid": valid,
        "bucket": buckets,
    }


def summarise(groups: list[dict], fx_px: float, baseline_m: float | None, label: str) -> dict:
    gt = np.concatenate([g["gt_depth_m"] for g in groups])
    pred = np.concatenate([g["pred_depth_m"] for g in groups])
    valid = np.concatenate([g["valid"] for g in groups])
    result = metrics_module.compute_metrics(gt, pred, valid, fx_px=fx_px, baseline_m=baseline_m)
    return {"label": label, **result.as_dict()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--session", required=True, help="data/<session>")
    parser.add_argument(
        "--export-dir", default=None,
        help="Ground-truth export dir. Default: results/lidar_ground_truth/<session>/export",
    )
    parser.add_argument("--scenes", nargs="+", required=True, help="e.g. scene_B scene_C")
    parser.add_argument("--backend", choices=["zed_sdk"], default="zed_sdk")
    parser.add_argument("--depth-mode", choices=sorted(zed_sdk_predictor.DEPTH_MODE_BY_NAME), required=True)
    parser.add_argument("--frame-index", type=int, default=config.EVAL_FRAME_INDEX)
    parser.add_argument("--baseline-m", type=float, default=None, help="For disparity-domain metrics; e.g. session_manifest.json's calibration.baseline_m")
    parser.add_argument("--capture-ids", nargs="*", default=None, help="default: every capture in the chosen scenes")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    session = Path(args.session)
    export_dir = Path(args.export_dir) if args.export_dir else config.GROUND_TRUTH_ROOT / session.name / "export"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scene_results: dict[str, list[dict]] = {}
    per_capture_summaries = []
    gt_K = None
    fx_px = None
    for scene_name in args.scenes:
        scene_gt = ground_truth.load_scene(session, export_dir, scene_name)
        if gt_K is None:
            gt_K = scene_gt.K
            fx_px = scene_gt.fx_px
        elif not np.allclose(gt_K, scene_gt.K, atol=1e-6):
            raise ValueError(f"{scene_name}'s K differs from an earlier scene's -- refusing to pool scenes with different intrinsics")

        captures = scene_gt.captures
        if args.capture_ids is not None:
            captures = [c for c in captures if c.capture_id in args.capture_ids]

        groups = []
        for capture in captures:
            result = evaluate_capture(capture, gt_K, args)
            groups.append(result)
            cap_summary = summarise([result], fx_px, args.baseline_m, result["capture_id"])
            per_capture_summaries.append(cap_summary)
            print(
                f"{result['capture_id']} ({scene_name}): n={cap_summary['n_gt_points']} "
                f"coverage={cap_summary['coverage']*100:.1f}% abs_rel={cap_summary['abs_rel']:.4f} "
                f"rms={cap_summary['rms']:.3f}m a1={cap_summary['a1']*100:.1f}%"
            )
        scene_results[scene_name] = groups

    all_groups = [g for groups in scene_results.values() for g in groups]
    overall = summarise(all_groups, fx_px, args.baseline_m, "overall")
    per_scene = [summarise(groups, fx_px, args.baseline_m, name) for name, groups in scene_results.items()]

    per_bucket = []
    for lo, hi, bucket_name in config.DEPTH_BUCKETS_M:
        bucket_groups = []
        for g in all_groups:
            mask = g["bucket"] == bucket_name
            if not np.any(mask):
                continue
            bucket_groups.append({
                "gt_depth_m": g["gt_depth_m"][mask], "pred_depth_m": g["pred_depth_m"][mask], "valid": g["valid"][mask],
            })
        if bucket_groups:
            per_bucket.append(summarise(bucket_groups, fx_px, args.baseline_m, bucket_name))

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "session": str(session),
        "export_dir": str(export_dir),
        "scenes": args.scenes,
        "backend": args.backend,
        "depth_mode": args.depth_mode,
        "frame_index": args.frame_index,
        "baseline_m": args.baseline_m,
        "K": gt_K.tolist(),
        "overall": overall,
        "per_scene": per_scene,
        "per_bucket": per_bucket,
        "per_capture": per_capture_summaries,
    }
    summary_path = out_dir / f"{args.backend}_{args.depth_mode}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print()
    print(f"Overall: n={overall['n_gt_points']} coverage={overall['coverage']*100:.1f}% "
          f"abs_rel={overall['abs_rel']:.4f} sq_rel={overall['sq_rel']:.4f} rms={overall['rms']:.3f}m "
          f"log_rms={overall['log_rms']:.4f} a1={overall['a1']*100:.1f}% a2={overall['a2']*100:.1f}% a3={overall['a3']*100:.1f}% "
          f"bias={overall['bias_m']*1000:+.1f}mm")
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
