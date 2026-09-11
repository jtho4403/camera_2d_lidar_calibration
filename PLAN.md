# LiDAR-Referenced Ground-Truth Pixel Pipeline — Revised Plan

**Status:** supersedes the previous `PLAN.md` in full.
**Scope:** produce sparse, LiDAR-referenced ground-truth depth pixels for static real
scenes, with a quantified uncertainty budget and an independent correctness proof.
Depth-model inference and benchmarking are explicitly out of scope; this plan stops at
the exported ground-truth dataset and defines the interface the evaluation harness will
consume.

---

## 0. Context and scope decisions

The research goal is to recommend a lightweight depth-estimation model that predicts
well and runs in real time on constrained hardware (NVIDIA Jetson Orin Nano, on a
Husarion ROSbot XL). Evaluating candidate models on real camera input requires ground
truth. The strategy is sparse ground truth: a subset of image pixels whose true depth is
known from a rigidly co-mounted LiDAR.

**Hardware (changed from the previous plan — all prior sensor-specific results are
void):**

| Role | Device |
|---|---|
| Compute / hub / sensor rig | Husarion ROSbot XL (NVIDIA Jetson Orin Nano) |
| LiDAR | SLAMTEC RPLIDAR S3 (2D, 360°, dToF) |
| Camera | Stereolabs ZED 2i |

RPLIDAR S3 datasheet figures relevant to this plan (model S3M1-R2): accuracy ±30 mm,
resolution 10 mm, angular resolution 0.1125° typ., scan rate 10 Hz typ., sample rate
32 kHz, range 0.05–40 m at 70 % reflectivity. Scan-field flatness is specified as
0°–1.5°.

**Scope decisions, confirmed:**

1. **Capture protocol stays simple.** Data is captured at multiple distances, lateral
   offsets and yaw angles relative to a checkerboard target mounted on a flat vertical
   wall. There is no pan/tilt of the target and no pan/tilt or height adjustment of the
   sensors.
2. **The existing SE(2) calibration is retained unmodified.** `cam_lidar_2d_icp.py`
   already solves a 3×3 `[[R,t],[0,1]]` transform between the LiDAR frame and the
   camera's horizontal plane. This plan consumes that artifact and does not replace it.
3. **The vertical offset is supplied by mechanical metrology, not solved.** Justified in
   §2 and §3; the consequence is bounded and validated rather than assumed away.
4. **Deliverable boundary.** Exported ground-truth dataset plus its validation report.
   No model inference harness.

---

## 1. The reduction that makes this tractable

This section is the mathematical core of the plan. Everything downstream follows from
it.

### 1.1 Frames

| Symbol | Frame | Convention |
|---|---|---|
| `L` | LiDAR scan frame | right-handed, x forward, y left, z up; origin at the scanner rotation centre; every return has z = 0 |
| `R` | Camera "robot" frame | right-handed, x forward, y left, z up; origin at the **rectified left** camera optical centre |
| `C` | Camera OpenCV frame | x right, y down, z forward; same origin as `R` |

`R → C` is the fixed pure rotation already encoded by `rot_cam_to_robot` in `gui.py`:
`X_C = −y_R`, `Y_C = −z_R`, `Z_C = x_R`.

### 1.2 The key algebraic result

The existing calibration gives `T2` (3×3, SE(2)), mapping a LiDAR point into the
camera's horizontal plane:

```
[x_R]        [x_L]
[y_R]  = R2  [y_L]  +  t2
```

Under the zero relative roll/pitch assumption the inter-sensor rotation is a pure yaw
about the vertical axis. A yaw rotation applied to `(x_L, y_L, 0)` **cannot produce a
non-zero z component**. Therefore, in frame `R`, every LiDAR return sits at a single
constant height:

```
z_R = Δz     (constant for all returns, all poses, all scenes)
```

where `Δz` is the height of the LiDAR scan plane above the rectified left camera optical
centre (positive when the LiDAR is above the camera).

Substituting into the pinhole model, on **rectified** images (so distortion is exactly
zero):

```
d_gt = x_R                          ground-truth depth (camera-frame Z)
u    = c_x − f_x · y_R / x_R        image column
v    = c_y − f_y · Δz  / x_R        image row
```

Read what this says, because it determines the entire risk profile of the experiment:

- **`d_gt` depends only on `T2`.** `Δz`, pitch and roll do not appear.
- **`u` depends only on `T2`.** Same.
- **`Δz`, pitch and roll enter in exactly one place: the image row `v`.**

The parameters that the simplified capture protocol cannot observe therefore do not
contaminate the ground-truth depth value or its image column. They determine only which
row a correct depth is attributed to. §3 bounds that error; §5 neutralises it.

### 1.3 Row geometry sanity check

`v − c_y = −f_y Δz / x_R`. The scan line sits off the image centre in the near field and
converges monotonically to the horizon row `c_y` as range increases. With
`f_y ≈ 530 px` and `Δz = 45 mm`, the offset is ≈ 24 px at 1 m and ≈ 5 px at 5 m. The
scan line is inside the image over the whole working range, and far-field ground truth
concentrates in a narrow band of rows. Recompute this with the measured `f_y` and `Δz`
during Phase 0 and record the result.

---

## 2. Sensitivity and error budget

### 2.1 What each error source does

| Error source | Affects `d_gt` | Affects `u` | Affects `v` |
|---|---|---|---|
| LiDAR range error `σ_r` | yes | yes | negligible |
| `T2` translation error `σ_tx`, `σ_ty` | yes | yes | no |
| `T2` yaw error `σ_ψ` | yes | yes | no |
| `Δz` error | **no** | **no** | `f_y δ(Δz) / Z`, shrinks with range |
| relative pitch `θ` | **no** | **no** | `≈ f_y θ`, constant with range |
| relative roll `φ` | **no** | **no** | `≈ φ (u − c_x)`, tilts the line |
| rectified intrinsics error | no | yes | yes |

Numerically, with `f_y ≈ 530 px`: 5 mm of `Δz` error is 2.7 px at 1 m and 0.9 px at 3 m;
1° of pitch is ≈ 9 px at all ranges; 1° of roll is ≈ 11 px across a half image width.
Pitch and roll dominate `v`, and unlike `Δz` their effect does not shrink with range.

### 2.2 Analytic depth budget

With `x_R = cos ψ · x_L − sin ψ · y_L + t_x` and `y_R = sin ψ · x_L + cos ψ · y_L + t_y`,
the exact partial derivatives are:

```
∂x_R/∂ψ = −(y_R − t_y)
∂y_R/∂ψ =  (x_R − t_x)
```

giving the first-order depth budget

```
σ_d² ≈ σ_r,x² + σ_tx² + (y_R − t_y)² · σ_ψ²
```

The yaw term is easy to overlook and grows with how far off-axis a point is: 0.2° of
yaw error at 1.5 m lateral offset is 5 mm of depth error.

**Implementation note:** do not hand-code the pixel Jacobians. Propagate uncertainty by
Monte Carlo over the bootstrap ensemble of `T2` (§4.2) plus sampled range noise. It is
simpler, harder to get wrong, and produces the full joint distribution of
`(u, v, d_gt)` directly.

---

## 3. Rig parameters: metrology, not guesswork

`Δz`, pitch and roll are supplied as measured rig constants with stated tolerances. A
tape-measure estimate is not acceptable; a documented metrology chain is.

### 3.1 `Δz` chain

```
Δz = (h_lidar_mount + z_scan_plane_above_lidar_base)
   − (h_camera_mount + z_optical_centre_above_camera_datum)
```

- `z_scan_plane_above_lidar_base`: from the RPLIDAR S3 mechanical dimensions drawing in
  the datasheet.
- `z_optical_centre_above_camera_datum`: from the Stereolabs ZED 2i mechanical drawing
  and the SDK's reported left optical centre relative to the housing datum.
- `h_lidar_mount`, `h_camera_mount`: measured with calipers against marked datum lines on
  the ROSbot XL mounting plates.

Target tolerance ±3–5 mm. Record every term, its source, and its tolerance.

### 3.2 Pitch and roll

Measure with a digital inclinometer (±0.1–0.2° typical) on the LiDAR base plate and on
the camera mounting face, in both the pitch and roll axes, and record the readings and
the instrument specification. If both sensors bolt to a common machined flat plate, note
that roll and pitch are near-zero by construction and the inclinometer is the evidence.

This converts "assume zero roll and pitch" into "measured, bounded at ±0.2°, worth
±2 px of row error", which is defensible.

### 3.3 Rig parameter file

All of the above lives in a single versioned JSON file, `config/rig_<rig_id>.json`,
referenced by hash from every exported dataset:

```json
{
  "rig_id": "rosbotxl_zed2i_rplidars3_v1",
  "measured_on": "YYYY-MM-DD",
  "delta_z_m": 0.045,
  "delta_z_tolerance_m": 0.004,
  "delta_z_chain": [
    {"term": "h_lidar_mount", "value_m": 0.0, "source": "calipers", "tol_m": 0.001},
    {"term": "z_scan_plane_above_lidar_base", "value_m": 0.0, "source": "S3 datasheet Fig 4-1", "tol_m": 0.001},
    {"term": "h_camera_mount", "value_m": 0.0, "source": "calipers", "tol_m": 0.001},
    {"term": "z_optical_centre_above_camera_datum", "value_m": 0.0, "source": "ZED 2i mechanical drawing", "tol_m": 0.002}
  ],
  "pitch_deg": 0.0,
  "roll_deg": 0.0,
  "attitude_tolerance_deg": 0.2,
  "attitude_instrument": "<make/model>, spec ±0.1 deg",
  "board_standoff_m": 0.005,
  "notes": ""
}
```

`board_standoff_m` is the thickness of the checkerboard plus its mounting, i.e. the
offset between the printed board plane and the wall plane the LiDAR actually strikes. It
is required by the §5 validation and is a real systematic if left unrecorded.

---

## 4. Calibration artifacts consumed

### 4.1 Existing, unchanged

- `lidar_to_camera_2d.npy` — 3×3 SE(2) transform, LiDAR → camera robot frame.
- `calibration_result.json` — transform, residuals, sanity checks.

`cam_lidar_2d_icp.py` is **not** to be modified beyond the single additive change below.

### 4.2 One additive change to `cam_lidar_2d_icp.py`

The uncertainty budget in §2 requires the spread of `T2`, which requires re-solving on
resampled subsets of poses. That requires the per-pose correspondences, which currently
live only inside `main()` and are discarded.

**Change:** after correspondence collection and before the ICP call, dump the per-pose
correspondence set to `calibration_correspondences.npz` alongside the existing outputs:
per pose, the selected LiDAR points (Nx2), the checkerboard-derived wall line points
(Mx2), the `solvePnP` `rvec`/`tvec`, and the pose id. No behavioural change; no existing
output altered. This is the only edit to the existing calibration package.

### 4.3 Intrinsics: resolve the existing mismatch

The repository hardcodes `camera_k` in `cam_lidar_2d_icp.py`, which does not match the
ZED SDK's per-capture intrinsics in `metadata_pose_NN.json`. Resolve as follows and
document it:

- **Canonical `K` = the ZED SDK rectified left-camera intrinsics**, with `dist = 0`.
- Run the SE(2) calibration on **rectified** images so the solved transform lands in the
  rectified left frame — the same frame every candidate depth model predicts in.
- The repository's own `cam_intrinsic.py` checkerboard calibration is retained as a
  **validation** of the factory calibration (report the difference in `f_x`, `f_y`,
  `c_x`, `c_y` and the reprojection RMS), not as the source of truth.

### 4.4 Calibration capture geometry

#### What actually determines yaw observability

For a point-to-line fit, yaw observability comes entirely from the **diversity of line
normal directions** in the camera frame. This has three consequences that the
`data_2026-06-28` session got wrong:

1. **Frontal poses at different distances are near-duplicates.** They share one normal.
   They constrain `t_x` and contribute essentially nothing to yaw. Three of the seven poses
   were frontal (line angles 91.4°, 90.7°, 86.5°), so the effective N for yaw was four.
2. **Lateral offset with a frontal board contributes nothing either.** Translating sideways
   without rotating the board leaves the normal unchanged.
3. **Only board yaw rotates the normal.** It is the single variable that buys yaw
   observability.

#### The second lever: angular extent

The residual's sensitivity to a yaw error scales as `d · sin α`, where `α` is a return's
bearing off boresight. Measured pooled RMS of `d·sin α` for `data_2026-06-28` was 188.5 mm,
and the correspondence set spanned only about ±14° of bearing.

Two consequences:

- A small board centred in front of the sensor is close to the worst possible geometry for
  constraining yaw, regardless of how many poses are taken.
- Any bearing-dependent error **outside** ±14° is untested by construction. This is the
  structural gap the Phase 1.5/1.6 diagnostics could not close, and it is a capture-design
  problem rather than a code problem.

#### Revised protocol

| Parameter | `data_2026-06-28` | Revised |
|---|---|---|
| Total poses | 7 | 20–30 |
| Frontal poses | 3 | 2–3, as a `t_x` anchor only |
| Board yaw range | ±32° | ±45°, in ~15° steps, both signs |
| Board position | mostly centred | deliberately off-centre for at least half the poses |
| Bearing coverage of correspondences | ±14° | ±40° or better |
| Board size | as used | as large as practical |
| Distance range | 0.71–1.65 m | 0.8–4 m |
| Scene | cluttered bedroom | controlled, see below |

Specifics:

- **Yaw diversity is the priority.** Spend poses on board orientation, not on distance
  repeats. The user's own instinct here was correct: on-centre-line poses at varied yaw are
  worth more than frontal poses at varied distance.
- **Push the board off-axis.** For at least half the poses, position so the board sits well
  to the side of the camera's boresight rather than centred. This raises `d · sin α` and
  extends bearing coverage in the same move.
- **Use the largest board that remains fully in frame** at the working distances. Angular
  extent is a direct multiplier on yaw sensitivity.
- **Controlled scene.** Plain walls, minimal clutter at scan height, no cables or chair legs
  crossing the scan plane. The `data_2026-06-28` session's clutter is a plausible
  contributor to correspondence contamination and made every downstream diagnostic harder
  than it needed to be.
- **Burst capture** per pose, per §6.7: ≥100 frames and ≥100 scans, 3–5 s settle after any
  movement.
- **Hold out a third of the poses** from the fit, for §5.3.

#### Hardware note

The RPLIDAR S3 samples at 0.1125° versus the STL-19P's ~0.71°, a factor of about six. The
per-point column-localisation noise floor drops from roughly ±7 px to roughly ±1 px at
`f_x ≈ 522`. Several of the diagnostic ambiguities in the Phase 1.5/1.6 work were driven by
that floor and will not recur.

### 4.5 Verification scene (new, mandatory)

The root cause of the Phase 1.5/1.6 impasse is that **the calibration has no acceptance test
that probes outside the target region.** Tests A and B both live on the board, so they can
only ever confirm that `T2` is optimal where it was fitted. A capture session that does not
fix this will reproduce the same impasse with better data.

Capture one scene, in the same session and without moving the rig relative to its calibrated
state, designed so off-board alignment is unambiguous:

- **3–4 isolated vertical objects** (poles, boxes, stands) at well-separated bearings,
  targeting roughly −30°, −10°, +10°, +30°.
- **High contrast against the background**: light objects on a dark background or vice
  versa, so the image silhouette is locatable to ±2 px rather than ±30 px.
- **Large depth discontinuities**, ≥0.5 m clear space behind each object, so the LiDAR range
  discontinuity is unambiguous.
- **Nothing else at scan height.** No cables, no chair legs, no furniture edges.
- Measure each object's true distance with the laser rangefinder.

This scene is **never used for fitting**. It is the acceptance test.

It converts the off-board measurement from ±30 px on N=4 to roughly ±3 px on dozens of
points, which is what makes the question decidable at all.

---

## 5. Correctness proof: checkerboard-plane validation

This is what makes the method defensible, and it needs no new capture, no tilting, and
no use of ZED depth.

### 5.1 Principle

For every calibration pose, `cv2.solvePnP` places the checkerboard plane in the camera
frame from geometry alone, to millimetre accuracy, entirely independently of the LiDAR
and of `T2`. That plane is an independent depth reference.

### 5.2 Procedure

For each pose `i`:

1. Take the LiDAR returns that struck the wall/board (already curated by the existing
   `SelectPointsInterface` flow).
2. Project them through §1.2 to obtain `(u, v, d_gt)`.
3. Form the board plane in the OpenCV camera frame from the pose's `rvec`/`tvec`:
   normal `n = R_board[:, 2]`, point on plane `p_0 = tvec`. Offset the plane by
   `board_standoff_m` along `−n` to obtain the **wall** plane that the LiDAR actually
   strikes.
4. Back-project the pixel: `m = K⁻¹ [u, v, 1]ᵀ`, normalised so `m_z = 1`.
5. Intersect with the plane: `d_pnp = (n · p_0) / (n · m)`.
6. Residual `e_i = d_gt − d_pnp`. Also record the point-to-plane distance
   `n · (p_C − p_0)`.

### 5.3 What to report

- Distribution of `e` overall: mean (bias), standard deviation, RMSE, 95th percentile.
- `e` versus distance, and `e` versus pose yaw angle. Both should be flat and centred on
  zero. Systematic drift with distance indicates a scale or intrinsics problem; drift
  with yaw indicates a `T2` yaw error.
- **Hold-out form:** solve `T2` on a random subset of poses, evaluate `e` on the held-out
  poses. Repeat over folds. This is the honest number and is the one to quote as the
  ground-truth depth uncertainty in every downstream depth-model result.
- Optional absolute anchor: a laser rangefinder reading to the wall at two or three
  poses, compared against projected `d_gt`. Independent of everything, ±1 mm.

### 5.4 What this does and does not validate — state this explicitly

It validates `d_gt` and `u`, which is exactly where `T2` lives. It **cannot** validate
`v`, because depth on a vertical plane is row-independent. The `v` chain is covered
separately by the §3 metrology, the §3.2 inclinometer readings, and the §6.4
row-sensitivity mechanism. Write this into the thesis rather than leaving it to be
found. Extend the "what this does not validate" list.** The board-plane residual validates
`d_gt` and `u` *within the board's angular window*, currently ±14°. It cannot detect a
bearing-dependent error outside that window, and it cannot validate `v` at all. Both
limitations must be stated explicitly.

### 5.5 Qualitative check

Overlay projected points on the calibration images. They must sit on the board, in a
near-horizontal line, at the predicted row. This catches sign errors and frame-convention
mistakes immediately and should be the first thing run. Upgrade from qualitative to quantitative.** Replace the overlay eyeball with the §4.5 verification-scene test: for each isolated object, signed `Δu` between the projected LiDAR range discontinuity and the image silhouette, reported per bearing with mean, standard error and a fit against bearing angle.

### 5.6 New acceptance criteria

State pass thresholds *before* running the tests, so the
outcome is not negotiated after the fact. Suggested starting points, to be confirmed against
the S3 characterisation:

| Test | Threshold |
|---|---|
| Hold-out board-plane depth residual | mean \|bias\| < LiDAR noise floor; no trend vs distance or vs pose yaw |
| Verification-scene `Δu`, per bearing | \|mean\| < 5 px at every bearing |
| Verification-scene `Δu` vs bearing angle | slope not significantly different from zero |
| Bootstrap `σ_ψ` | < 0.3° |
| Rangefinder absolute anchor | agreement within the S3's characterised bias band |

Any failure sends you back to capture, not to code.

---

## 6. Pipeline design

### 6.1 New package `tools/lidar_ground_truth/`

Placed alongside `tools/lidar_characterisation/`, following its precedent: a headless,
standalone consumer of calibration artifacts, not part of the interactive GUI flow.

| Module | Responsibility |
|---|---|
| `config.py` | paths, thresholds, constants; mirrors `tools/lidar_characterisation/config.py` |
| `scan_io.py` | `.pcd` reading and dropped-return filtering; reuse the pattern in `tools/lidar_characterisation/pcd_io.py`, retargeted to the RPLIDAR S3 invalid-return convention |
| `rig.py` | load and validate `config/rig_<id>.json`; expose `delta_z_m`, tolerances, attitude |
| `calibration_io.py` | load `lidar_to_camera_2d.npy`, `calibration_result.json`, `calibration_correspondences.npz`, rectified `K` |
| `projection.py` | §6.2 core |
| `filters.py` | §6.3 occlusion and quality filters |
| `uncertainty.py` | §6.4 bootstrap + Monte Carlo propagation |
| `validate_board_plane.py` | §5 CLI; the correctness proof |
| `export_scene.py` | §6.5 CLI; per-scene ground-truth export |
| `README.md` | frames, conventions, the §1.2 derivation, capture protocol, how to re-run |

### 6.2 Projection core

```python
def project_scan(xy_lidar, T2, delta_z, K, image_wh, z_min=0.05):
    """Project 2D LiDAR returns into rectified camera pixels.

    xy_lidar : (N, 2) float, LiDAR-frame points [m], x forward / y left
    T2       : (3, 3) SE(2), LiDAR -> camera robot frame
    delta_z  : float, scan-plane height above optical centre [m], +ve = LiDAR higher
    K        : (3, 3) rectified intrinsics; distortion is exactly zero
    image_wh : (W, H)

    Returns u, v (float pixel coords), depth (camera-frame Z), and the keep mask.
    """
    R2, t2 = T2[:2, :2], T2[:2, 2]
    xy_r = xy_lidar @ R2.T + t2
    x_r, y_r = xy_r[:, 0], xy_r[:, 1]

    keep = x_r > z_min                      # in front of the camera; guard the divide
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    u = cx - fx * y_r / np.where(keep, x_r, 1.0)
    v = cy - fy * delta_z / np.where(keep, x_r, 1.0)
    depth = x_r                             # camera-frame Z, NOT radial range

    W, H = image_wh
    keep &= (u >= 0) & (u < W) & (v >= 0) & (v < H)
    return u, v, depth, keep
```

Non-negotiable details:

- **Rectified images, rectified `K`, `dist = 0`.** The formula above is then exact and
  there is no distortion model to argue about. Do not use `cv2.projectPoints`; it adds
  no value here and invites extrinsics/distortion confusion.
- **Guard `x_R > z_min` before dividing.** Returns behind the camera otherwise project to
  plausible-looking garbage pixels.
- **`depth` is camera-frame Z, not radial range.** Retain `lidar_radial_range_m =
  hypot(x_L, y_L)` per point for provenance and uncertainty lookup, and never compare it
  to a model prediction. The two differ by the point's off-axis angle.
- **Verify the sign of `Δz` on the first overlay.** If the scan line appears on the wrong
  side of `c_y`, the sign is flipped.
- **Verify the LaserScan handedness empirically.** The S3 datasheet specifies a
  left-handed frame with clockwise-increasing angle; ROS REP-103 is right-handed with y
  left. Place an object at a known bearing and confirm the driver's convention before
  trusting anything. Record the result in the README.

### 6.3 Filters

Applied in `filters.py`, all model-independent:

1. **Invalid returns** — outside `[range_min, range_max]`, NaN/Inf, exact `(0,0)`.
2. **Behind camera / out of bounds** — from `project_scan`.
3. **Occlusion / parallax rejection.** The lateral component of the camera–LiDAR offset
   means background returns near a depth discontinuity can project onto foreground image
   regions. This corrupts ground truth precisely at object boundaries, where depth models
   are worst, so it inflates measured error where it matters most. Reject returns within
   `k` samples of a range discontinuity `|r_{j+1} − r_j| > τ` in bearing order, and reject
   returns whose projected `u` ordering is inconsistent with their depth ordering. Record
   counts by rejection reason.
4. **High incidence angle** — reject where the LiDAR characterisation (§7) shows bias
   growing unacceptably. Estimate local surface incidence from the scan's local gradient.

### 6.4 Uncertainty

- **`T2` covariance by bootstrap.** Resample calibration poses with replacement from
  `calibration_correspondences.npz`, re-run `icp_2d.icp_per_line`, collect the ensemble
  of `(t_x, t_y, ψ)`. Report mean, covariance, and percentile intervals. A few hundred
  resamples is ample.
- **Per-point Monte Carlo.** For each exported point, sample from the bootstrap `T2`
  ensemble and from the range-noise model, push through `project_scan`, and record
  `depth_std_m`, `u_std_px`, `row_uncertainty_px`.
- **`row_uncertainty_px`** additionally folds in the `Δz` tolerance and the attitude
  tolerance from `rig.py`: `sqrt((f_y δΔz / Z)² + (f_y δθ)² + (δφ (u − c_x))²)` combined
  with the Monte Carlo term.
- **Range noise model** from the §7 characterisation: `bias_mm(r)` applied as a
  correction, `sigma_mm(r)` as the sampled noise. Carry forward the caveat that the
  characterisation covers a specific target geometry and material set, and that applying
  it to arbitrary scene surfaces is an extrapolation.
- **Row-sensitivity helper.** Export `row_uncertainty_px` per point. Provide
  `flag_row_sensitive(prediction, u, v, row_uncertainty_px, threshold)` as a helper for
  the future evaluation harness: it samples the model's own prediction in a vertical
  strip of `±row_uncertainty_px` and flags points where the spread exceeds a threshold.
  This deliberately lives at evaluation time, not export time, so the exported ground
  truth stays model-independent. Headline metrics are then reported both with and
  without flagged points; if the two agree, row uncertainty demonstrably does not drive
  the conclusions.

### 6.5 Export format

Under `results/lidar_ground_truth/<scene_id>/`, mirroring the existing JSON + CSV + plots
convention:

**`<scene_id>_ground_truth.csv`**, one row per surviving point:

```
scene_id, capture_id, source_scan_file, source_image_file,
x_lidar_m, y_lidar_m, lidar_radial_range_m,
pixel_u, pixel_v, camera_depth_m,
depth_std_m, u_std_px, row_uncertainty_px,
lidar_bias_correction_mm, lidar_sigma_mm, uncertainty_source
```

**`<scene_id>_ground_truth_summary.json`** — provenance: reference image path,
calibration artifact paths with hashes and timestamps, `rig_<id>.json` hash, `K` source
and values, bootstrap `T2` covariance, per-capture point counts, and filtered-out counts
by reason. Follow the self-documenting style of the existing `result_payload`.

**Diagnostic PNG** — reference image with points overlaid, coloured by depth, analogous
to the existing `aligned_point_clouds.png`.

### 6.6 Capture tooling

The only capture path that currently exists is the interactive checkerboard calibration
flow, and the script that reportedly captured arbitrary scenes (`phase1/validate_zed.py`)
is absent from the repository. A minimal replacement is a hard prerequisite for §6.5 and
is independent of everything above.

`tools/capture/capture_scene.py`, kept deliberately thin:

- Records a **lossless** SVO2 file for the capture window. Lossy H.264/H.265 alters the
  image the depth model will see and is a direct confound in a depth-accuracy study.
- Simultaneously dumps a rectified left PNG, a rectified right PNG, and the ZED depth
  `.npy` at the capture midpoint, for immediate quality control.
- Writes `metadata_<capture_id>.json`: rectified intrinsics, baseline, resolution, lens
  variant, camera serial, SDK version, timestamps, lighting notes, `rig_id`.
- Publishes a `/capture_event` marker message so the association is recorded inside the
  LiDAR bag.

LiDAR side: `ros2 bag record /scan` (MCAP) on the Jetson, extracted to `.pcd` per capture
using the existing extraction path. Record `/scan` rather than a derived point cloud: it
is the raw sensor output with `angle_min`, `angle_increment` and `range_min/max`, with no
lossy conversion.

### 6.7 Synchronisation

Both sensors hang off the same Jetson, so there is a single system clock and no
cross-machine offset. Verify the RPLIDAR is on the Jetson's USB rather than the ROSbot's
microcontroller board.

Because the rig is rigid and stationary and the scene is static, temporal alignment is
irrelevant to correctness; what is required is **association**, not synchronisation.
Protocol:

- Settle delay of 3–5 s after any rig movement; discard the first frames and scans.
  Vibration is the one thing that breaks the static assumption.
- Capture a **burst** at each position (target ≥ 100 frames and ≥ 100 scans, ~10 s at
  10 Hz). This makes synchronisation a non-issue by construction and yields frame-to-frame
  precision statistics for free.
- Aggregate the scan burst per bearing: median range, plus per-bearing standard deviation
  and return rate. Export all three.
- Write `capture_id` into both streams plus timestamps, so the pairing is provable after
  the fact.
- State in the thesis that this is a deliberate static-scene simplification and that hard
  synchronisation becomes mandatory for the dynamic case.

---

## 7. LiDAR characterisation (parallel workstream)

The existing `tools/lidar_characterisation/` toolkit is retained and retargeted from the
LDROBOT STL-19P to the RPLIDAR S3. `datasheet.py`'s spec bands must be replaced with the
S3 figures in §0. The accuracy study (bias and precision versus rangefinder-measured true
distance) is re-run on the S3, and its `summary.csv` becomes the range-noise model
consumed by §6.4. Captured concurrently with the calibration and scene data.

This plan assumes the outcome shows the S3 is defensibly accurate and precise for the
working range. If it does not, §6.4's uncertainty output will make that visible in the
exported dataset rather than hiding it.

---

## 8. Implementation phases

Each phase is independently verifiable. Phases 1 and 2 require **no new data capture** —
they run against the existing `data/data_2026-06-28/` and `data/data_2026-07-10/`
sessions.

| Phase | Deliverable | Verification gate |
|---|---|---|
| **0** | `config/rig_<id>.json` populated; §1.3 row-geometry check recomputed; LaserScan handedness confirmed | rig file validates; scan line predicted inside the image across the working range |
| **1** | `scan_io.py`, `rig.py`, `calibration_io.py`, `projection.py`, `filters.py`; §5.5 overlay tool | projected points land on the board in existing calibration images, in a near-horizontal line at the predicted row |
| **2** | `cam_lidar_2d_icp.py` correspondence dump; `validate_board_plane.py`; `uncertainty.py` | hold-out residual `e` reported with mean, std, RMSE; flat versus distance and versus yaw; bootstrap `T2` covariance produced |
| **2b** | §4.5 verification scene captured and analysed | `Δu` within threshold at every bearing; no significant slope vs bearing |
| **3** | `tools/capture/capture_scene.py` + bag extraction path | one static scene captured end to end; SVO2 lossless; `capture_event` present in the bag |
| **4** | `export_scene.py`, CSV + JSON + PNG | one scene exported; counts by rejection reason reported; uncertainty fields populated and non-degenerate |
| **5** | `datasheet.py` retargeted to S3; accuracy study re-run | S3 `summary.csv` produced and consumed by `uncertainty.py` |

Phase 2's hold-out residual is the single most important number this plan produces. If it
is not small, flat and unbiased, nothing downstream is trustworthy and the cause must be
found before proceeding.

Phase 2b is the gate that Phase 1.5/1.6 revealed to be missing. Phase 2's board-plane
residual on its own is not sufficient evidence that the calibration is correct, because it
is structurally blind to the region where the problem was observed.

---

## 9. Assumptions and limitations to carry into the thesis

State these explicitly rather than leaving them to be discovered:

1. **Observability.** Yaw-only, wall-parallel capture makes the vertical offset and the
   relative roll and pitch unobservable from the calibration data (Zhang & Pless, IROS
   2004, for the underlying point-on-plane constraint and its degenerate configurations).
   They are therefore supplied by direct mechanical measurement and inclinometer readings.
   This is sound because, per §1.2, those parameters affect only the image row and not the
   ground-truth depth value or column; the row sensitivity is bounded by §3 metrology and
   neutralised by §6.4. This is a deliberate, understood scope decision.
2. **Coverage.** A single-plane LiDAR yields one line of pixels per image. At 0.1125°
   angular resolution the returns are spaced comparably to the image pixel pitch, giving
   on the order of several hundred ground-truth points per capture, all in a single row
   band that converges toward the horizon with range. The correct framing is *sparse
   horizontal-slice depth validation*. Claims about vertical position dependence are out
   of reach; claims about error versus true depth, texture and material are not.
3. **Scan-field flatness.** The S3 specifies 0°–1.5°. A constant plane tilt is absorbed
   into the calibrated attitude; what remains is bearing-dependent residual non-flatness.
   Quantify it as residual versus bearing angle from the §5 validation rather than
   assuming it away.
4. **Depth convention.** `d_gt` is camera-frame Z in the rectified left frame. Each
   candidate model's own convention must be confirmed separately: some predict Z, some
   predict radial range, and monocular models predict scale-ambiguous depth requiring
   per-frame scale and shift alignment before any metric is meaningful.
5. **Range-noise extrapolation.** The characterisation covers a specific target geometry
   and material set. Applying it as a per-point uncertainty model for arbitrary scene
   surfaces, materials and incidence angles is an extrapolation the underlying study does
   not itself validate.
6. **ZED depth is never validation.** It may be used as a coarse sanity check on units and
   frame conventions only. It is the thing being measured.
