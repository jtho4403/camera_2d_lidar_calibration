# LiDAR-to-Camera Ground-Truth Projection — Method and Open Questions

**Status: pipelines executed, sparse ground truth generated. Quantitative accuracy/precision
validation has *not* yet been performed.** This document explains what was built and how the
underlying method works, shows the key outputs, and lays out the validation questions I want
to discuss before drawing any conclusions about whether the result is trustworthy.

---

## 1. What this is for

I've run the camera–LiDAR calibration and ground-truth projection pipelines
(`camera_2d_lidar_calibration/`, `tools/lidar_ground_truth/`) end to end on a new capture
session (three scenes: a calibration scene with a checkerboard, a verification scene with flat
plates, and a set of realistic evaluation scenes). The pipeline runs cleanly and produces a
solved sensor-to-sensor transform and sparse LiDAR-referenced depth values projected onto the
camera images.

What I have **not** done yet is decide whether the result is *good* — accurate, precise, and
trustworthy enough to use as ground truth for evaluating depth-estimation models. Before I go
build my own metrics and thresholds, I want to walk through the method with you, show you what
came out, and get your read on how to approach that validation properly. §7 is the actual list
of questions.

---

## 2. The pipeline, in outline

Two rigidly mounted sensors on a mobile robot — a stereo camera (used monocularly here) and a
2D planar LiDAR — need a known spatial relationship before a LiDAR return can be drawn on a
camera image. Getting there is two stages:

1. **Extrinsic calibration** — solve the fixed rigid-body transform between the LiDAR's
   coordinate frame and the camera's, using a checkerboard seen by both sensors simultaneously
   from many different positions.
2. **Projection / ground-truth export** — apply that transform to every LiDAR return in a
   scene, convert each one into a camera pixel coordinate plus a depth value, and keep only the
   returns that survive some basic geometric sanity filters.

Both stages are described below with the actual mathematics, since that's the part I want your
opinion informed by, not just the headline pictures.

---

## 3. Stage 1: solving the camera–LiDAR extrinsic transform

### 3.1 What's actually unknown

The LiDAR is a 2D planar scanner: every return it produces lies in a single horizontal plane at
a fixed height, reported as `(x, y)` in the scanner's own frame. The camera sees the world in
3D. Because both sensors are bolted to the same rigid chassis, the transform between them is
**constant** — solve it once, and it holds for every future capture (until the physical mount
is disturbed).

Because the LiDAR only ever samples one height, and the two sensors are assumed to share the
same roll and pitch (both mounted flat on the same rigid rover — a fair assumption for a
wheeled platform, not so much for the working assumption I'll flag in §4), the *unknown* part
of that transform reduces from a full 6-DOF rigid body transform down to a planar one: 2D
translation `(t_x, t_y)` plus a single rotation `(yaw)`. That's what gets solved. The remaining
vertical offset between the two sensors is measured separately, by hand, as a fixed rig
parameter — it isn't part of this fit at all (more on why that split is convenient in §4).

### 3.2 The method: checkerboard correspondence, then registration

For each of several checkerboard positions:

1. **Camera side.** `cv2.findChessboardCorners` + `solvePnP` recovers the checkerboard's pose
   (rotation + translation) relative to the camera, using the camera's known intrinsics and the
   board's known, physically measured geometry. From that pose, a line is derived along the
   board's own horizontal axis, at the height the LiDAR's scan plane is expected to intersect
   it — this line is the camera's independent claim of "where the wall/board surface is."
2. **LiDAR side.** The raw 2D scan for the same pose is inspected, and the returns that struck
   the board are identified.
3. **Registration.** Across all poses simultaneously, an Iterative Closest Point (ICP)
   procedure solves for the single rigid transform `(t_x, t_y, yaw)` that best aligns every
   pose's LiDAR points onto that pose's camera-derived line, minimizing the summed
   point-to-line distance. It runs coarse-to-fine — an initial pass with a loose correspondence
   distance, then progressively tighter passes — so that a rough initial guess doesn't lock in
   bad point-to-line correspondences before the fit has had a chance to improve.

The reason many poses are used together, rather than solving pose-by-pose, is observability: a
single checkerboard position can't separate a yaw error from a lateral translation error (both
shift the projected points sideways in similar ways at a single fixed distance). Positions at
different distances and different bearings behave differently under a yaw error versus a
translation error, so combining many poses is what actually pins down all three parameters
uniquely rather than trading them off against each other.

### Figure 1 — Before and after registration

| Before | After |
|---|---|
| ![pre-alignment](results/calibration/data_2026-09-24/checkerboard_lidar_pre_alignment.png) | ![aligned](results/calibration/data_2026-09-24/aligned_point_clouds.png) |

Each colour is one sensor's points, with all 24 checkerboard poses overlaid in a single frame
(so the "V" and zig-zag shapes are 24 separate short line segments, not one continuous line).
Left: the camera-derived lines (green) and LiDAR points (blue) before any transform is applied
— two separate, offset point clouds. Right: after the registration solve — the two sets sit on
top of each other across all 24 poses at once. The pipeline reports that this converged (no
errors, the fit completed), and visually the alignment looks tight and consistent. I have not
yet turned "looks tight" into a number.

Input images for this solve were the temporal mean of ~150 video frames per checkerboard pose,
rather than a single snapshot — intended to reduce random corner-detection noise before it
reaches the fit.

---

## 4. Stage 2: projecting LiDAR returns into the camera frame

### 4.1 Frames

Three coordinate frames are in play:

| Frame | Convention | Origin |
|---|---|---|
| `L` — LiDAR | right-handed, x forward, y left, z up | the scanner's rotation centre |
| `R` — camera "robot" frame | same convention, x forward/y left/z up | the camera's optical centre |
| `C` — camera OpenCV frame | x right, y down, z forward | same origin as `R`, just rotated |

Stage 1 solves `T2`: the transform from `L` to `R`. `R` and `C` share an origin and differ only
by a fixed rotation (an axis-convention flip, not something that's solved for — it's just
bookkeeping between the two conventions).

### 4.2 The projection equations

A LiDAR return `(x_L, y_L)` is first moved into the camera's horizontal plane:

```
[x_R, y_R]^T = R2 · [x_L, y_L]^T + t2       (T2 = [R2, t2], solved in Stage 1)
```

Every LiDAR return sits at the same fixed height in frame `R` — call it `Δz`, the vertical
offset between the LiDAR's scan plane and the camera's optical centre, measured separately by
hand (rig metrology), not solved by the calibration above. Given that and the standard pinhole
camera model, a LiDAR return converts into a ground-truth depth value and a pixel location as:

```
d_gt = x_R                              (ground-truth depth: camera-frame forward distance)
u    = c_x − f_x · y_R / x_R            (pixel column)
v    = c_y − f_y · Δz / x_R             (pixel row)
```

where `(f_x, f_y, c_x, c_y)` are the camera's known rectified intrinsics.

**The point worth discussing**: `d_gt` and `u` depend *only* on `T2` (the part that was
actually solved against real correspondences). `Δz` — the hand-measured rig parameter, and the
one quantity in this whole pipeline that rests on a physical tape-measure/drawing chain rather
than a fitted result — enters *only* into `v`, the row. If `Δz` (or the flat-mounting,
zero-relative-roll/pitch assumption from §3.1) turns out to be slightly wrong, the row where a
point is drawn shifts, but the depth value and column position it carries do not. That's the
structural argument for why this method could be trustworthy even though one input is a manual
measurement rather than a fitted quantity — I'd like your view on whether that argument actually
holds up, or whether it's a more fragile decomposition than it looks.

### Figure 2 — the row equation, visualised

![B01 predicted row](results/lidar_ground_truth/data_2026-09-24/overlay/B01_overlay.png)

This is one capture from the verification scene (flat plates at various distances, no
checkerboard — never used to fit anything above). The dots are projected LiDAR returns coloured
by `d_gt`; the red dashed line is the row `v` the equation above predicts, evaluated at that
capture's median depth. This is a single-image illustration of the equations, not a validation
result — I'm showing it to explain the math, not to claim anything about accuracy yet.

### 4.3 Filtering

The camera sees a narrower forward field of view than the LiDAR's full sweep, so most raw
returns simply aren't in frame at all — those are dropped first. Two further geometric filters
are applied: one rejects points sitting near a large range discontinuity in the raw scan (a
likely occlusion boundary, where the camera and LiDAR — physically a few centimetres apart —
may not agree on exactly which surface is visible), and one rejects points whose projected
column ordering is locally inconsistent with their depth ordering (the parallax signature of
the same problem). Both are geometric/structural checks, independent of any particular depth
model.

---

## 5. What the sparse ground truth looks like across the three scenes

### Figure 3 — Exported ground truth, across scenes

| Scene A (calibration, on-board) | Scene B (verification plates) |
|---|---|
| ![A09](results/lidar_ground_truth/data_2026-09-24/export/overlays/A09_ground_truth.png) | ![B01](results/lidar_ground_truth/data_2026-09-24/export/overlays/B01_ground_truth.png) |

| Scene C (evaluation — general room) | Scene C (evaluation — corridor/bins) |
|---|---|
| ![C01](results/lidar_ground_truth/data_2026-09-24/export/overlays/C01_ground_truth.png) | ![C08](results/lidar_ground_truth/data_2026-09-24/export/overlays/C08_ground_truth.png) |

Points are coloured by `d_gt`. A few descriptive (not accuracy) observations:

- The points consistently land on visible surfaces at plausible depths — nothing lands "in the
  air" or somewhere obviously wrong.
- They form a thin horizontal band across the middle of every image, not a full-frame
  scattering. That's structural: a single 2D LiDAR scan plane only ever samples one horizontal
  slice of the scene, so the resulting ground truth can only ever support evaluating depth
  accuracy along that slice, never the full frame. That scope was understood and accepted
  before this capture session — it's a property of using a 2D LiDAR at all, not something this
  run got wrong.
- Roughly 800–850 points survive per capture on average, fairly consistently across all three
  scenes and across near and far arrangements.

None of this tells me whether a given point's depth is accurate to 5 mm or 50 mm — only that
the pipeline is producing plausible, well-behaved output at a glance.

---

## 6. What has — and hasn't — been checked

**Done:**
- The calibration pipeline runs to completion and reports convergence.
- The registration looks visually tight across all 24 calibration poses (Figure 1).
- The projected points land in visually sensible places across all three scenes, including two
  scenes the calibration board was never in (Figure 3).
- The row-prediction equation visibly tracks where points land in at least one example capture
  (Figure 2).

**Not done:**
- No quantitative error metric of any kind — no residual statistics, no accuracy or precision
  number, nothing that could be called "the ground truth is accurate to ±X mm."
- No hold-out or cross-validation scheme — I haven't checked whether the fit generalises to
  data it wasn't fit on.
- No uncertainty quantification — I don't yet have a defensible number for how much to trust
  any individual point, or how that trust should vary with distance, angle, or anything else.
- No check for systematic bias (e.g., does accuracy degrade with distance, with viewing angle,
  across the session, or in some other structured way rather than randomly).
- No comparison against any independent reference measurement.

That last set of gaps is exactly what I want your input on.

---

## 7. Questions for discussion

1. Is there an established methodology — in the depth-estimation, SLAM, or sensor-fusion
   literature, or in how benchmark datasets like KITTI/nuScenes/DDAD validated their own
   LiDAR-camera calibration — for validating a LiDAR-derived sparse ground truth against camera
   imagery, especially when there's no independent, higher-precision reference sensor available?
2. How should I separate **precision** (repeatability / random noise) from **accuracy**
   (systematic bias) for this kind of result, given the LiDAR itself doesn't yet have its own
   independent accuracy characterisation?
3. What's the minimum set of metrics you'd want to see before trusting this? Residual
   statistics from the fit itself? A hold-out or leave-one-out scheme on the calibration poses?
   Something structural, like checking for trends against distance or viewing angle that
   shouldn't be there if the geometry is right?
4. Is it valid to validate using data from the same calibration scene/poses used to fit the
   transform, or do I need genuinely independent data — e.g. the verification scene, or a target
   at an independently measured distance — to make a credible accuracy claim?
5. Should uncertainty be reported per point, or only as an aggregate figure for the whole
   dataset? And what would that uncertainty need to account for, beyond obvious sensor noise?
6. What would you consider a "good" result here — is there a standard threshold or rule of
   thumb (e.g. relative to the LiDAR's own datasheet spec, or relative to the depth differences
   I'd actually need to distinguish between candidate models) that would tell me this is fit for
   purpose?
7. Are there failure modes specific to this camera + 2D-LiDAR cross-calibration approach (as
   opposed to a full 3D LiDAR) that I should be specifically testing for, rather than discovering
   later?
8. Given the ground truth is inherently a thin horizontal strip rather than full-frame (§5),
   how should that scope limitation shape the validation plan itself — should I be validating
   differently than I would for a dense, full-frame ground truth?

---

## Appendix: what was generated

| Artifact | Path |
|---|---|
| Solved transform (`T2`) | `results/calibration/data_2026-09-24/lidar_to_camera_2d.npy` |
| Full calibration output | `results/calibration/data_2026-09-24/calibration_result.json` |
| Registration figures | `results/calibration/data_2026-09-24/*.png` |
| Sparse ground-truth CSVs + per-capture overlays | `results/lidar_ground_truth/data_2026-09-24/export/` |
| Row-equation illustration overlays (all 39 captures) | `results/lidar_ground_truth/data_2026-09-24/overlay/` |
