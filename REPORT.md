# Ground-Truth Calibration and Validation Report — `data_2026-09-24`

**Status: ACCEPTED for depth-model evaluation, with explicitly scoped claims.**
This document lays out the calibration and validation results that support that decision,
quantifies the one open limitation, and defines what can and cannot be claimed from the
resulting ground truth.

---

## 1. Executive summary

The camera–LiDAR extrinsic transform `T2` was solved from Scene A (24 checkerboard poses,
0.6–1.5 m) using burst-mean-averaged images (150 frames per pose) and validated by
leave-one-pose-out hold-out residuals, a 200-resample bootstrap, and an independent visual
check on Scene B (off-board, out to ~5 m). Sparse LiDAR-referenced ground truth was exported
for all three scenes, filtered to the declared 0.5–8 m operating range (§7, §9): 19,629 points
(Scene A), 4,245 points (Scene B), 7,983 points (Scene C).

Four of five automated acceptance checks pass cleanly. The fifth — a statistically significant
(3.6σ) residual bias that grows with either distance or elapsed capture time, the two being
almost perfectly confounded in this session's capture order (r = 0.959) — does not. This report
treats that finding as a **known, quantified, unresolved risk to absolute-accuracy claims**,
not as grounds to discard the dataset. §5 characterises it in detail; §8 defines the scope of
valid use given it.

**Decision**: accept this ground truth for depth-model evaluation. Prioritise relative
model-to-model comparisons (low risk from a shared systematic bias). Report absolute accuracy
numbers alongside the quantified bias-risk band in §5.4, not in isolation. Revisit if either the
pending RPLIDAR S3 characterisation or a future thermal-drift diagnostic resolves the cause.

---

## 2. Dataset and pipeline overview

| | |
|---|---|
| Session | `data_2026-09-24` (ROSbot XL + ZED 2i + RPLIDAR S3 rig) |
| Scene A — calibration | 24 poses, plain checkerboard (6×3 corners, 50 mm — measured, confirmed exact), 0.6–1.5 m |
| Scene B — verification | 5 arrangements (B01–B05), 3 flat plates each, no checkerboard, out to ~5 m |
| Scene C — evaluation | 10 captures across 8 designed scenes (C1–C8, some with rig-height/tilt repeats), no checkerboard |
| Pipeline | `cam_lidar_2d_icp.py` (extrinsic solve) → `validate_board_plane.py` (Phase 2) → `overlay_diagnostic.py` (Phase 1, off-board check) → `export_scene.py` (Phase 4, GT export) |

Two pipeline-code fixes were required before this session's data could run through the
pipeline at all (scene-nested `captures/scene_A|B|C/` layout vs. the pipeline's prior flat-session
assumption; a capture-metadata bug where every capture's `scene` field read `"calibration"`
regardless of actual scene) — both are code/workaround fixes, not data quality issues, and are
recorded in the repository's commit history and internal notes.

---

## 3. Scene A: extrinsic calibration (`T2`) results

### Figure 1 — Before and after staged ICP

| Before alignment | After alignment |
|---|---|
| ![pre-alignment](results/calibration/data_2026-09-24/checkerboard_lidar_pre_alignment.png) | ![aligned](results/calibration/data_2026-09-24/aligned_point_clouds.png) |

24 poses' camera-derived board lines (green) and LiDAR wall/board points (blue), all poses
overlaid in a single frame. Left: raw point sets before any transform. Right: after the
3-stage coarse-to-fine ICP solve (0.30 m → 0.15 m → 0.10 m thresholds) — camera lines and LiDAR
points lie on top of each other across all 24 poses simultaneously, the qualitative signature of
a well-converged single rigid-body fit.

### Table 1 — Headline calibration metrics

| Metric | Value | Direction | Note |
|---|---|---|---|
| Solved yaw | 178.884° | — | ROSbot rig: LiDAR frame is ~180° rotated from camera |
| Solved translation | `t_x` = +22.2 mm, `t_y` = −63.9 mm | — | camera origin in LiDAR frame |
| Cross-session yaw repeatability | 178.863°/178.884°, prior-session 178.87° | — | independent solves agree to ≤0.03° |
| Overall residual RMSE (orthogonal point-to-line, N=4923) | **6.81 mm** | ↓ lower is better | |
| Overall residual median | 3.86 mm | ↓ lower is better | |
| Overall residual p95 | 14.23 mm | ↓ lower is better | |
| Overall residual max | 36.16 mm | ↓ lower is better | single worst point of 4923 |
| Points accepted at 0.10 m threshold | 4923 / 4923 (100%) | ↑ higher is better | |
| Threshold-sensitivity sweep (0.20/0.25/0.30/0.35 m, refit from scratch) | Δt = 0.000 m, Δyaw = 0.000° at every threshold | — | solution is not threshold-fragile |
| Mean checkerboard reprojection error (24 poses) | 0.05–0.18 px | ↓ lower is better | solvePnP quality, all well sub-pixel |

**Fix 1 (burst-mean averaging) was applied and tested**: images were re-extracted as the
temporal mean of ~150 SVO2 frames per pose (rather than a single QC snapshot) and the
calibration re-solved. The result changed by <0.1 mm / 0.02° and RMSE by <0.1 mm — a genuine
negative result: single-frame corner-detection noise was **not** the dominant source of residual
scatter in this session. This is relevant to §5: it rules out one candidate explanation for the
distance-trend bias below.

---

## 4. Board-plane validation (Phase 2)

### Table 2 — Acceptance checks

| Check | Result | Value | Threshold | Direction |
|---|---|---|---|---|
| Hold-out overall bias below LiDAR noise floor | **PASS** | 0.10 mm | 1.73 mm | ↓ lower is better |
| Hold-out mean absolute per-pose bias below noise floor | **FAIL** | 4.90 mm | 1.73 mm | ↓ lower is better |
| No trend vs. distance | **FAIL** | 3.55σ | 2.0σ | ↓ lower is better |
| No trend vs. board yaw (incidence) | **PASS** | 0.03σ | 2.0σ | ↓ lower is better |
| Bootstrap yaw std. below threshold | **PASS** | 0.163° | 0.3° | ↓ lower is better |

**3 of 5 pass.** The two failures are not independent — both trace to the same distance/time-
correlated bias characterised in §5. Read together with Table 1's clean residuals and Table 3's
zero net bias, the picture is: `T2` is **well-determined in orientation and unbiased in aggregate**,
with a **directional residual that grows with distance (or session time)**, not a gross
miscalibration.

### Table 3 — Bootstrap `T2` uncertainty (200 resamples)

| Parameter | σ | Direction |
|---|---|---|
| `t_x` | 1.29 mm | ↓ tighter is better |
| `t_y` | 2.66 mm | ↓ tighter is better |
| yaw | 0.163° | ↓ tighter is better |

All tight relative to the 0.3° acceptance threshold — the transform's parameter uncertainty
itself is not the concern; the concern is the systematic component described next.

---

## 5. The distance-trend bias: characterisation

### 5.1 What was found

### Figure 2 — Hold-out residual vs. distance and vs. board yaw

![board plane residual trends](results/lidar_ground_truth/data_2026-09-24/board_plane_validation/board_plane_residual_trends.png)

Left: hold-out residual (`d_gt − d_pnp`, LiDAR-derived depth minus independent camera+board
depth) plotted against each pose's mean board distance — a significant positive slope
(+13.31 mm/m, 3.55σ). Right: the identical residual against board yaw (incidence angle) —
flat, 0.03σ, no relationship.

### 5.2 Hypotheses tested and ruled out

| Candidate cause | Status | Evidence |
|---|---|---|
| Checkerboard print scale error | **Ruled out** | Measured 300 mm / 6 squares = exactly 50.0 mm, matches code |
| Single-frame PnP / corner-detection noise | **Ruled out** | Burst-mean re-solve (150 frames/pose) changed the result by <0.1 mm |
| Pure multiplicative scale error (board size or camera `f_x`) | **Disfavoured** | Fitted trend has a significant non-zero intercept (−12.11 mm, t=−3.07); a pure scale error predicts a line through the origin |
| Camera attitude (roll/pitch) drift over the session | **Ruled out** | ZED IMU drift over 54 min: 0.08–0.18° (within the IMU's own ±0.5° noise floor); correlation with bias only r≈0.10–0.15 |
| Residual `T2` yaw miscalibration | **Disfavoured** | Would imprint bearing/yaw-dependent structure; `no_trend_vs_board_yaw` passed cleanly at 0.03σ |
| LiDAR range-dependent bias (offset + gain) | **Open, plausible** | Functional form (non-zero intercept + slope) matches the classic LiDAR error model; only the pending S3 characterisation can bound this directly |
| Camera thermal/optical drift over the session | **Open, plausible — leading candidate** | See 5.3; `camera_disable_self_calib: true` means factory calibration was held fixed all session regardless of actual thermal state |

### 5.3 Distance vs. session time: a genuine confound, and what the regression shows

Scene A was captured as five ascending-distance blocks over ~54 minutes — distance and elapsed
session time are correlated at **r = 0.959** in this dataset. This cannot be resolved
post-hoc; it is a property of the capture order.

| Model | R² | Coefficient (per-unit) | Significance |
|---|---|---|---|
| bias ~ distance | 0.364 | +13.31 mm/m | t = +3.55 |
| bias ~ elapsed time | **0.504** | +0.276 mm/min | t = +4.73 |
| bias ~ distance + elapsed time (joint) | 0.578 | distance: −21.1 mm/m (unstable); time: +0.632 mm/min | distance t=−1.92 (n.s.); time t=+3.26 |

Elapsed time alone explains *more* variance than distance alone, and when both are regressed
jointly, distance's coefficient loses significance and flips sign (a classic symptom of severe
collinearity, condition number 464) while time's holds up. **This is genuine evidence, not proof**
— 24 points along a near-1-D axis cannot cleanly separate two variables correlated at 0.959 —
but it shifts the balance of likelihood toward a time-correlated drift mechanism (most plausibly
camera thermal drift, untested) being at least as responsible as a purely range-dependent LiDAR
effect, which was the working assumption before this analysis.

### 5.4 Quantified risk to absolute-accuracy claims

The fitted trend (`bias(d) = −12.11 + 13.31·d` mm) is fit only over 0.6–1.5 m. Extrapolating it
— **explicitly unvalidated beyond 1.5 m** — against the depths the exported ground truth
actually spans gives:

| Scene | Depth | Implied bias if linear (mm) | Stated point-wise uncertainty (mm) | Ratio |
|---|---|---|---|---|
| A | 3.88 m (median) | +39.5 | 8.7 | 4.5× |
| A | 7.38 m (max, now capped at ≤8 m — §9) | +86.1 | 8.5 | 10.1× |
| B | 3.55 m (median) | +35.1 | 7.4 | 4.7× |
| B | 6.42 m (max) | +73.4 | 8.9 | 8.3× |
| C | 2.29 m (median) | +18.3 | 7.0 | 2.6× |
| C | 8.00 m (max, now capped at ≤8 m — §9) | +94.3 | 8.5 | 11.1× |

This table is the central number for §8's scoping decision: **at the depths Scenes B and C
actually cover, the possible unmodelled bias is several times larger than the stated per-point
uncertainty** — *if* the effect is real and linear that far out, which is not established. The
exported `depth_std_m` column contains zero contribution from this term by design (every row's
`uncertainty_source` states "no range bias correction" explicitly).

### 5.5 A supplementary, low-confidence check

Scene B's B01 plates (1.3–2.5 m) were matched against design-nominal target distances (no
rangefinder measurements were actually recorded in this session's metadata — a separate,
unrelated data-capture gap): diffs of +16 mm, +46 mm, −161 mm, no clear pattern. B05
(3.5–5 m, the long-range case that would actually speak to this question) could not be reliably
interpreted — plate/background clusters could not be cleanly separated, and the one candidate
region showed a continuous depth gradient consistent with plate-yaw geometry rather than a
usable distance. **This check is inconclusive and should not be weighted in the decision.**

---

## 6. Scene B: off-board visual verification (Phase 1)

### Figure 3 — Projection holds up off the calibration board, at both short and long range

| B01 (short range, 1.3–3.5 m) | B05 (long range, up to ~5 m) |
|---|---|
| ![B01](results/lidar_ground_truth/data_2026-09-24/overlay/B01_overlay.png) | ![B05](results/lidar_ground_truth/data_2026-09-24/overlay/B05_overlay.png) |

Projected LiDAR returns (coloured by depth) land at the plates' base edges and track the
predicted-row line (red dashed) closely across the full image width, at both distances. This
validates the **row (`v`) / vertical geometry** off-board and at range — a different axis from
the in-plane distance bias in §5, but independent, positive evidence that the projection
generalises beyond the calibration board.

---

## 7. Ground-truth export: Scenes A, B, C

### Figure 4 — Example exported ground truth, one per scene

| Scene A (A09, on-board) | Scene C (C01, evaluation) |
|---|---|
| ![A09](results/lidar_ground_truth/data_2026-09-24/export/overlays/A09_ground_truth.png) | ![C01](results/lidar_ground_truth/data_2026-09-24/export/overlays/C01_ground_truth.png) |

### Table 4 — Export headline numbers (post range-filter, §9)

| Metric | Scene A | Scene B | Scene C | Direction |
|---|---|---|---|---|
| Ground-truth points | 19,629 | 4,245 | 7,983 | ↑ more is generally better |
| Captures | 24 | 5 | 10 | — |
| Points per capture (mean) | 818 | 849 | 798 | ↑ higher is better |
| Depth range | 0.50–7.38 m | 1.05–6.42 m | 1.01–8.00 m | now hard-capped to [0.5, 8] m |
| Points excluded by the 0.5–8 m operating-range filter | 768 (1.1% of valid returns) | 0 (0%) | 444 (1.5% of valid returns) | filter now enforced — 0 points remain out of range in every scene |
| Vertical (row) coverage | 46.3 px | 19.7 px | 21.4 px | out of 720 px image height — see §9 |
| Horizontal (column) coverage | 0.2–1279.5 px | 0.3–1279.5 px | 0.9–1279.7 px | full-width in all scenes |
| `depth_std_m` (stated, precision-only) — median / p95 | 5.42 / 11.13 mm | 4.10 / 11.11 mm | 3.87 / 9.04 mm | ↓ lower is better |
| Points surviving all filters (of all valid LiDAR returns) | 28.0% | 28.7% | 27.5% | ~69% rejected for being outside camera FOV (structural, expected), <2% by occlusion/parallax quality filters, 1.1–1.5% by the operating-range filter |

**Structural limitation (expected, not new information)**: as understood going into this
project, the ground truth is a thin horizontal strip (2.7–6.4% of frame height) since it comes
from a single 2D LiDAR scan plane — it supports evaluating depth accuracy along that strip, not
full-frame.

---

## 8. Decision: accepted, with scoped claims

| Use case | Risk from §5's bias | Recommendation |
|---|---|---|
| Relative ranking / trade-off comparison between depth models | **Low** — a shared, roughly-consistent bias largely cancels out of a ranking | Primary basis for conclusions |
| Absolute accuracy numbers at Scene A range (≤1.5 m) | **Low** — this is the validated range; residual RMSE 6.8 mm | Report directly |
| Absolute accuracy numbers at Scene B/C range (2–10 m) | **Real, quantified, direction- and magnitude-uncertain** (Table in §5.4) | Report the number **alongside** the bias-risk band from §5.4; do not present in isolation |
| Row/vertical accuracy claims | Independently validated off-board (§6) | Low additional risk beyond the general sparse-GT scope |

This report's position: the dataset is accepted for depth-model evaluation on this basis, not
on the basis that the open question is resolved. The distinction matters for how results should
be written up — headline accuracy numbers at Scene C ranges should carry the §5.4 caveat
explicitly, not as a footnote.

---

## 9. Outstanding limitations and follow-up work

1. **Root cause of the distance/time-correlated bias remains open.** Leading candidate:
   camera thermal drift (untested — would need a controlled thermal-soak comparison of `K` with
   self-calibration temporarily enabled). RPLIDAR S3 characterisation (pending, separate
   session) will bound the LiDAR-side contribution but, per §5.3's regression evidence, may not
   be the dominant cause.
2. ~~Declared operating range (0.5–8 m) not enforced or flagged in the exported CSV.~~
   **Resolved 2026-09-25**: `filters.reject_outside_operating_range` now excludes points outside
   [0.5, 8] m from the export (`export_scene.py`, applied on top of the existing occlusion/parallax
   filters; `config.OPERATING_RANGE_Z_MIN_M`/`OPERATING_RANGE_Z_MAX_M`, kept distinct from the
   numerical `Z_MIN_M` divide-by-zero guard used inside `projection.project_scan`). Scoped to
   `export_scene.py` only — `overlay_diagnostic.py`'s Phase 1 sanity checks (Figure 3, §6) still
   inspect the whole scan including out-of-range background, deliberately, since restricting that
   diagnostic's scope could hide real sign/frame-convention errors. Re-running the export dropped
   768 points from Scene A and 444 from Scene C (§7's Table 4 reflects the corrected counts); Scene
   B was unaffected (already fully in-range). This changed the exact figures in §5.4, §7 and the
   executive summary above — all now reflect the post-filter export.
3. **The `board_plane_validation` pass/fail status lives only in the scene-level summary JSON,
   not per-row in the exported CSV.** A consumer of just the CSV would not see §5's caveat.
   Consider surfacing it more prominently in any downstream accuracy-harness documentation.
4. **For any future recapture** (not required for the current decision): randomising Scene A's
   pose order relative to distance would decorrelate time from distance and let §5.3's regression
   attribute the effect cleanly for the first time — a low-cost design change, not a hardware or
   equipment change.
5. Scene A's ChArUco repeat capture (designed as an independent second `T2` solve for
   cross-validation) was not captured this session; the pipeline also does not yet support ChArUco
   detection. Not blocking, but the cross-session yaw agreement in Table 1 is currently the only
   repeatability evidence available.

---

## Appendix: artifact index

| Artifact | Path |
|---|---|
| Solved transform | `results/calibration/data_2026-09-24/lidar_to_camera_2d.npy` |
| Calibration result (full) | `results/calibration/data_2026-09-24/calibration_result.json` |
| Calibration correspondences | `results/calibration/data_2026-09-24/calibration_correspondences.npz` |
| Board-plane validation | `results/lidar_ground_truth/data_2026-09-24/board_plane_validation/board_plane_validation.json` |
| Bootstrap ensemble | `results/lidar_ground_truth/data_2026-09-24/board_plane_validation/bootstrap_T2_ensemble.npz` |
| Phase 1 overlay diagnostics (all 39 captures) | `results/lidar_ground_truth/data_2026-09-24/overlay/` |
| Ground-truth CSVs + summaries + overlays | `results/lidar_ground_truth/data_2026-09-24/export/` |
| Burst-mean extraction provenance | `data/data_2026-09-24/burst_mean_manifest.json` |
| Staging provenance (all 39 captures) | `data/data_2026-09-24/staging_manifest.json` |
