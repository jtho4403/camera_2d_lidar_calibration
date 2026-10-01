"""Depth-accuracy metrics (docs/DEPTH_METRICS.md).

Implements the seven Standard Depth Metrics plus three additions beyond the
document, decided and justified in docs/DEPTH_METRICS.md's "Harness additions"
section:

  - bias_rel / bias_m: signed error (both relative and metric-unit), to
    characterise systematic over/under-estimation separately from the
    unsigned abs_rel/rms -- directly relevant given this project's own ground
    truth carries a known, unresolved distance-correlated bias risk
    (REPORT.md Sec 5): a model evaluation that only reports unsigned error
    cannot distinguish "the model is biased" from "the ground truth is."
  - coverage: fraction of ground-truth points for which the predictor
    produced a usable (finite, in-range) depth value. Reported and compared
    across models, never silently dropped -- a model with high accuracy but
    low coverage (e.g. from aggressive confidence filtering) is answering a
    different, easier question than one with full coverage, and the two are
    not comparable without this number alongside them.
  - Stereo disparity-domain metrics (EPE, bad3, D1-all): optional, computed
    when focal length and baseline are supplied. Included because every
    current and planned candidate model is a stereo-matching network
    (docs/MASTER_PLAN.md, docs/DEPTH_MODEL_BUILD_SPEC.md both frame the
    problem in disparity/max-disparity terms), so error in the domain these
    models actually optimise is directly relevant alongside depth-domain
    error.

Image-synthesis metrics (PSNR/SSIM/photo_rmse, DEPTH_METRICS.md's third
table) are deliberately not implemented here: they measure reconstructed-image
quality, not depth-map accuracy, and have no role in this evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

import config


@dataclass
class MetricResult:
    n_gt_points: int
    n_valid: int
    coverage: float  # n_valid / n_gt_points

    abs_rel: float
    sq_rel: float
    rms: float
    log_rms: float
    a1: float
    a2: float
    a3: float

    bias_rel: float
    bias_m: float

    epe_px: float | None = None
    bad3: float | None = None
    d1_all: float | None = None

    def as_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


def compute_metrics(
    gt_depth_m: np.ndarray,
    pred_depth_m: np.ndarray,
    valid_mask: np.ndarray,
    fx_px: float | None = None,
    baseline_m: float | None = None,
) -> MetricResult:
    """Compute the metric set over one group of ground-truth points (a capture,
    a scene, a depth bucket, or the whole dataset -- the caller decides the
    grouping by what it passes in).

    gt_depth_m, pred_depth_m, valid_mask must be the same length and aligned
    index-for-index: valid_mask[i] is True iff the predictor produced a
    usable depth value at ground-truth point i (already resolved by the
    predictor/sampling layer, e.g. tools/depth_eval/predictors/zed_sdk.py's
    bilinear sampling -- see config.py's any-neighbour-invalid rule).
    Accuracy metrics are computed only over valid_mask; coverage is computed
    over every point passed in, so a caller must pass the full group (not a
    pre-filtered valid-only subset) for coverage to mean anything.
    """
    gt_depth_m = np.asarray(gt_depth_m, dtype=np.float64)
    pred_depth_m = np.asarray(pred_depth_m, dtype=np.float64)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    n_gt = len(gt_depth_m)
    n_valid = int(np.count_nonzero(valid_mask))

    if n_valid == 0:
        return MetricResult(
            n_gt_points=n_gt, n_valid=0, coverage=0.0,
            abs_rel=float("nan"), sq_rel=float("nan"), rms=float("nan"), log_rms=float("nan"),
            a1=float("nan"), a2=float("nan"), a3=float("nan"),
            bias_rel=float("nan"), bias_m=float("nan"),
        )

    gt = gt_depth_m[valid_mask]
    pred = pred_depth_m[valid_mask]
    gt_safe = np.maximum(gt, config.GT_DEPTH_EPS_M)

    diff = pred - gt
    abs_diff = np.abs(diff)

    abs_rel = float(np.mean(abs_diff / gt_safe))
    sq_rel = float(np.mean((abs_diff ** 2) / gt_safe))
    rms = float(np.sqrt(np.mean(diff ** 2)))
    log_rms = float(np.sqrt(np.mean((np.log(gt_safe) - np.log(np.maximum(pred, config.GT_DEPTH_EPS_M))) ** 2)))

    ratio = np.maximum(pred / gt_safe, gt_safe / np.maximum(pred, config.GT_DEPTH_EPS_M))
    a1, a2, a3 = (float(np.mean(ratio < d)) for d in config.ACCURACY_DELTAS)

    bias_rel = float(np.mean(diff / gt_safe))
    bias_m = float(np.mean(diff))

    epe_px = bad3 = d1_all = None
    if fx_px is not None and baseline_m is not None:
        disp_gt = fx_px * baseline_m / gt_safe
        disp_pred = fx_px * baseline_m / np.maximum(pred, config.GT_DEPTH_EPS_M)
        disp_diff = np.abs(disp_pred - disp_gt)
        epe_px = float(np.mean(disp_diff))
        bad3 = float(np.mean(disp_diff > 3.0))
        d1_all = float(np.mean((disp_diff > 3.0) & (disp_diff / np.maximum(disp_gt, config.GT_DEPTH_EPS_M) > 0.05)))

    return MetricResult(
        n_gt_points=n_gt, n_valid=n_valid, coverage=n_valid / n_gt,
        abs_rel=abs_rel, sq_rel=sq_rel, rms=rms, log_rms=log_rms,
        a1=a1, a2=a2, a3=a3, bias_rel=bias_rel, bias_m=bias_m,
        epe_px=epe_px, bad3=bad3, d1_all=d1_all,
    )


def assign_buckets(depth_m: np.ndarray) -> np.ndarray:
    """Label each GT depth with its bucket name from config.DEPTH_BUCKETS_M,
    or "" if it falls outside every bucket (should not happen for GT already
    filtered to the declared operating range, but not assumed)."""
    depth_m = np.asarray(depth_m, dtype=np.float64)
    labels = np.full(depth_m.shape, "", dtype=object)
    for lo, hi, name in config.DEPTH_BUCKETS_M:
        is_last = (lo, hi, name) == config.DEPTH_BUCKETS_M[-1]
        in_bucket = (depth_m >= lo) & ((depth_m <= hi) if is_last else (depth_m < hi))
        labels[in_bucket] = name
    return labels
