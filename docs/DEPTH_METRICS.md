# Evaluation Metrics

This section describes the evaluation metrics used to assess the performance of depth estimation and stereo matching algorithms. During evaluation, we may only evaluate areas that have non-zero ground truth or within a specific value range to avoid bias from abnormal ground truth distribution.

## Standard Depth Metrics

Our evaluation follows standard metrics used in depth estimation literature:

| Metric | Description | Formula | Better | Meaning |
|---|---|---|---|---|
| `abs_rel` | Absolute Relative Error | $\left(\frac{1}{N}\right)\sum\frac{\lvert d_i - \hat{d}_i \rvert}{\hat{d}_i}$ | Lower | Measures average relative depth error, normalised by true depth. Less sensitive to errors in far regions. |
| `sq_rel` | Squared Relative Error | $\left(\frac{1}{N}\right)\sum\frac{\lvert d_i - \hat{d}_i \rvert^2}{\hat{d}_i}$ | Lower | Emphasises larger depth errors by squaring the difference. Particularly sensitive to outliers. |
| `rms` | Root Mean Squared Error | $\sqrt{\left(\frac{1}{N}\right)\sum\lvert d_i - \hat{d}_i \rvert^2}$ | Lower | Measures average magnitude of depth errors in metric units (e.g., metres). |
| `log_rms` | Log Root Mean Squared Error | $\sqrt{\left(\frac{1}{N}\right)\sum\lvert \log(d_i) - \log(\hat{d}_i) \rvert^2}$ | Lower | Measures errors in logarithmic space, more sensitive to close regions. |
| `a1` | Threshold Accuracy ($\delta < 1.25$) | % of $\max\left(\frac{d_i}{\hat{d}_i}, \frac{\hat{d}_i}{d_i}\right) < 1.25$ | Higher | Percentage of pixels where relative error is within 25%. |
| `a2` | Threshold Accuracy ($\delta < 1.25^2$) | % of $\max\left(\frac{d_i}{\hat{d}_i}, \frac{\hat{d}_i}{d_i}\right) < 1.25^2$ | Higher | Percentage of pixels where relative error is within 56.25%. |
| `a3` | Threshold Accuracy ($\delta < 1.25^3$) | % of $\max\left(\frac{d_i}{\hat{d}_i}, \frac{\hat{d}_i}{d_i}\right) < 1.25^3$ | Higher | Percentage of pixels where relative error is within 95.31%. |

Where $d_i$ is the predicted depth and $\hat{d}_i$ is the ground truth depth.

## Stereo-Specific Metrics

For evaluating stereo matching algorithms, we include:

| Metric | Description | Formula | Better | Meaning |
|---|---|---|---|---|
| `EPE-all` | End-Point Error | $\left(\frac{1}{N}\right)\sum\lvert disp_i - \widehat{disp}_i \rvert$ | Lower | Average absolute disparity error in pixels. |
| `>3px Error (bad3)` | Absolute Disparity Error Rate | % of pixels where $\lvert disp_i - \widehat{disp}_i \rvert > 3$ | Lower | Percentage of pixels whose disparity error exceeds 3 pixels. |
| `D1-all` | Disparity Error Rate | % where $\lvert disp_i - \widehat{disp}_i \rvert > 3$ AND $> 5\%$ | Lower | Standard error metric for KITTI Stereo benchmark. |

## Image Synthesis Metrics

For evaluating image reconstruction quality:

| Metric | Description | Formula | Better | Meaning |
|---|---|---|---|---|
| `PSNR` | Peak Signal-to-Noise Ratio | $20\log_{10}\left(\frac{MAX_i}{\sqrt{MSE}}\right)$ | Higher | Measures ratio between maximum signal power and noise. Values above 30dB indicate good quality. |
| `SSIM` | Structural Similarity Index | Complex formula considering luminance, contrast, structure | Higher | Measures perceived similarity (0-1, 1=perfect). More aligned with human perception than PSNR. |
| `photo_rmse` | Photometric RMSE | $\sqrt{\left(\frac{1}{N}\right)\sum\lvert I_i - \hat{I}_i \rvert^2}$ | Lower | Root mean squared error between pixel values in reconstructed and ground truth images. |

## Harness additions (2026-10-01)

`tools/depth_eval/metrics.py` implements the seven Standard Depth Metrics above exactly as
specified, plus three additions, each justified by something specific to this project rather
than added by default:

| Metric | Why added |
|---|---|
| `bias_rel`, `bias_m` | Signed error (relative and metric-unit), alongside the unsigned metrics above. This project's own ground truth carries a known, unresolved distance-correlated bias risk (`REPORT.md` Sec 5) — an evaluation that only reports unsigned error (`abs_rel`, `rms`, ...) cannot distinguish "the model is biased" from "the ground truth is." Signed bias is the diagnostic that can tell them apart when read alongside the known GT bias direction/magnitude. |
| `coverage` | Fraction of ground-truth points for which the predictor produced a usable (finite, in-range) depth value, computed and reported separately from the accuracy metrics (which are computed only over the covered subset). A model with high accuracy but low coverage (e.g. from aggressive confidence filtering) is answering a different, easier question than one with full coverage — the two are not comparable on accuracy numbers alone without coverage reported alongside. |
| `epe_px`, `bad3`, `d1_all` | The Stereo-Specific Metrics above, computed in the disparity domain (`disparity = f_x · baseline / depth`, using the session's rectified `f_x` and `baseline_m`) rather than depth. Every current and planned candidate model (`docs/MASTER_PLAN.md`, `docs/DEPTH_MODEL_BUILD_SPEC.md`) is a stereo-matching network specified and built in disparity/max-disparity terms — error in the domain these models actually optimise is directly relevant alongside depth-domain error, not redundant with it. |

**Not implemented**: the Image Synthesis Metrics table (`PSNR`, `SSIM`, `photo_rmse`) measures
reconstructed-image quality, not depth-map accuracy, and has no role in this evaluation.

**Mandatory reporting structure, not a new metric**: every run reports the full metric set
overall, per scene, per depth bucket (`tools/depth_eval/config.py`'s `DEPTH_BUCKETS_M`: near
0.5–2 m, mid 2–4 m, far 4–8 m), and per capture — never as one aggregate number across the full
0.5–8 m range. This mirrors `REPORT.md` Sec 5's own finding that this project's ground truth has
distance-correlated structure that an aggregate number would hide, and depth models are
independently known to vary strongly with range.

## Reference

All data/information/formulas in this document were extracted from the following source:

```bibtex
@online{Wang2026DepthMetrics,
  title = {In-{{Air}} vs {{Underwater}} - {{Stereo Depth Estimation Comparison}}},
  author = {Wang, Yiran},
  url = {https://u7079256.github.io/In-Air-VS-Underwater/#metrics},
  urldate = {2026-05-22},
  file = {/Users/jackthompson/Zotero/storage/MEZ9KZEL/In-Air-VS-Underwater.html}
}
```
