# Experiment and Scene Design — Capture Session (v3)

Supersedes v2. This is a **targeted revision, not a redesign**: the scene structure, the
definitions, and Scenes C and S3 characterisation are unchanged. Every changed passage is
marked **[v3]**.

### Changelog v2 → v3

| # | Change | Reason |
|---|---|---|
| 1 | **Target board replaced** with the as-built A3 board (50 mm checkers, 6 × 3 inner corners) on a 600 × 300 × 6 mm MDF panel, plus a ChArUco twin | The first Scene A capture failed: 25.4 mm checkers gave 4–7 px per checker beyond 1.8 m, below the ~12 px detection floor. Only 6 of 32 images were usable. |
| 2 | **Scene A range compressed from 0.8–3.0 m to 0.6–1.5 m** (Fix 2) | The calibration solves a range-independent rigid transform. Separating yaw from lateral translation needs spread in 1/Z, which is *larger* at short range (1.000 for 0.6–1.5 m vs 0.917 for 0.8–3.0 m). Far poses also degrade the PnP plane normal. |
| 3 | **Bearing now scales with distance** in Scene A | At short range, high bearing pushes the board out of frame. Lateral offset (which conditions yaw) is dominated by far poses anyway. |
| 4 | **Scene A captured twice**, plain board then ChArUco board | Two independent solutions for `T2`; their spread bounds real calibration accuracy. |
| 5 | **Burst averaging mandated downstream** (Fix 1) | With only 18 corners, single-frame PnP plane error reaches ~12 mm RMS at far frontal poses. Averaging ~100 frames reduces this roughly twelve-fold. |
| 6 | **Scene B extended to long range** (new B5) | Scene A no longer covers range beyond 1.5 m. Long-range validation of the projection now lives entirely in Scene B. |
| 7 | **Scan plane height corrected** to ≤ 174.8 mm (measured) | v2 assumed 20–25 cm. |
| 8 | **Operating depth range declared**: `Z_min` = 0.5 m, `Z_max` = 8 m | Sets max disparity for the depth models; evaluation scenes must respect it. |

---

## 1. Definitions — read this first

Every angle in this document is measured **in the horizontal plane, viewed from above**.

**Sign convention: positive = counterclockwise viewed from above = toward the camera's
LEFT = toward the LEFT side of the image.** This matches ROS REP-103, so it is consistent
with the `/scan` data.

```
                    BIRD'S-EYE VIEW (looking down)

                         centreline (bearing 0)
                                 |
                    +bearing     |     -bearing
                  (camera left)  |  (camera right)
                       \         |         /
                        \        |        /
                         \       |       /
                          \      |      /
                           \     |     /
                            \    |    /
                             \   |   /
                              \  |  /
                               \ | /
                            [  ROSbot  ]  <- FIXED. Never rotates, never moves
                              camera +       during Scenes A and B.
                              LiDAR
```

### Bearing (β) — where the target sits

The angle from the centreline to the target's centre, measured at the camera.

- `β = 0` means the target is dead ahead, appearing at image column ≈ `c_x` ≈ 637.
- `β = +30°` means the target is 30° to the camera's left; it appears left of image centre.
- **You achieve bearing by moving the target sideways, never by rotating the ROSbot.**

### Board yaw (ψ) — which way the board faces

Rotation of the board about its own vertical axis.

- **`ψ = 0` means the board faces the camera directly**: its surface normal points straight
  back along the line of sight from board to camera.
- `ψ ≠ 0` means the board is turned so its face is oblique to that line of sight.
- **Positive ψ brings the board's left edge (as you see it in the image) closer to the
  camera**, and pushes its right edge further away.

### They are independent

A board at `β = +35°` can be at `ψ = 0` (facing the camera) or `ψ = +20°` (turned further).
Bearing says *where it is*; yaw says *which way it points*.

### Two derived quantities that explain the whole pose table

- **Board normal angle = β + ψ.** This is what determines whether the calibration is well
  conditioned. Diversity here is what makes the camera-to-LiDAR yaw observable. The v3 table
  spans ±75°.
- **Incidence angle = |ψ|.** The angle between the LiDAR beam and the board's surface
  normal. `ψ = 0` is head-on. Beyond roughly 60° the returns degrade, so the table caps `ψ`
  at 45°.

### How to set these physically

1. Mark the point on the floor directly below the camera. Run a tape measure forward from it
   along the boresight. That tape is the **centreline**.
2. The pose table gives **forward** and **lateral** in metres. Measure forward along the
   centreline, then lateral at right angles (positive = camera's left). Mark with tape and
   write the pose id on it.
3. Run a string from the camera's floor mark to the board's floor mark. That string is the
   **line of sight** for that pose. Setting the board perpendicular to the string gives
   `ψ = 0`. Then rotate the board by `ψ` using a protractor or digital angle finder laid on
   the floor at the mark.
4. Before recording, check the QC snapshot from `zed_capture.py`: the board must be fully in
   frame with visible margin.

---

## 2. Target boards [v3 — replaced]

### As-built specification

Two boards, identical in geometry, differing only in whether ArUco markers occupy the white
squares.

| Property | Plain board (primary) | ChArUco board (repeat) |
|---|---|---|
| Print | A3, 420 × 297 mm, landscape | same |
| Checker size | **50 mm nominal — measure it, see below** | same |
| Squares | 7 × 4 (columns × rows) | same |
| Inner corners | **6 × 3 = 18** | up to 18 ChArUco corners |
| Pattern area | 350 × 200 mm | same |
| Print margins | 35 mm left/right, 48.5 mm top/bottom | same |
| Markers | — | 37 mm, 14 markers, `DICT_4X4_50` is sufficient |
| Backing | 6 mm MDF, 600 × 300 mm, spray-adhesive bonded | same |

6 × 3 is even × odd, so the board has a unique orientation with no 180° ambiguity.

### Detection at the Scene A range

| Distance | px per checker | px per ChArUco marker |
|---|---|---|
| 0.6 m | 43.5 | 32.2 |
| 1.0 m | 26.1 | 19.3 |
| 1.2 m | 21.7 | 16.1 |
| **1.5 m** | **17.4** | **12.9** |

Checkers clear the ~12 px floor with margin everywhere. ChArUco markers are solid to 1.2 m
and **marginal at 1.5 m**; expect some far-pose ChArUco detections to fail. That is
acceptable — that sequence simply solves from fewer poses.

### Scan-plane intersection

The LiDAR scan plane sits **at or below 174.8 mm** (ROSbot 133.5 mm + RPLIDAR S3 41.3 mm;
the emission plane is slightly lower). Including the S3's 0–1.5° scan-field flatness, the
beam lies between roughly 128 mm and 222 mm at 1.5 m. The 300 mm panel standing on the floor
is intersected comfortably at every pose.

With the print centred on the panel, the pattern spans roughly 50–250 mm above the floor, so
**the scan plane strikes the printed paper**. The PnP plane and the LiDAR strike surface are
therefore the same physical surface, and `board_standoff_m = 0` within the print. The bare
MDF either side of the print sits ~0.15 mm behind the paper surface, which is negligible;
returns there are valid constraints on the same plane.

### Mandatory checks after mounting — before capture

1. **Measure the actual checker pitch.** Steel rule across 6 squares (nominal 300 mm), divide
   by 6. Print scaling errors of 1–2% are common and propagate directly as a scale error on
   every distance the calibration produces. **Use the measured pitch everywhere, never the
   nominal.** Record it in `config/rig_<id>.json`.
2. **Check flatness.** Straight edge across both diagonals of the 600 mm panel; record the
   worst gap. The entire method assumes a plane.
3. **Quiet zone.** The side margins are 35 mm, i.e. 0.7 of a square, below the one-square
   guideline, with brown MDF beyond. This is a risk, not a predicted failure. Test on the
   first three poses (see §4). If detection is unreliable, bond white paper strips onto the
   MDF along both side edges of the print to extend the white border.
4. **Update the board configuration in both tools** to 6 × 3 inner corners and the measured
   pitch: the board definition in `cam_lidar_2d_icp.py`, and the `BOARD CONFIG` block in
   `check_corners.py`, which still holds the old 16 × 11 board.

### Stand

The panel must reach the floor, and **nothing may sit in front of it at scan height**. Use
feet glued to the *back* of the panel, or a small easel behind it. Anything behind the panel
is occluded by it and cannot produce a spurious return.

### Rig setup

- **Fix the ROSbot and move the target.** Do not reposition the robot between poses. Every
  rig move risks disturbing the mount and invalidating the extrinsics.
- **Nothing else at scan height** anywhere in the forward view: cables, chair legs,
  furniture feet.
- Before starting, record the metrology (`Δz` chain, inclinometer pitch and roll) into
  `config/rig_<id>.json`, photograph the mount from three angles, and then do not touch it.
- Run `python3 zed_capture.py --level` and confirm roll and pitch read near zero.

---

## 3. The four scenes: what each one is for

| Scene | Question it answers | Checkerboard? | Fits `T2`? | Required? |
|---|---|---|---|---|
| **A. Calibration** | What is the camera-to-LiDAR transform? | **Yes — it is the entire scene** | Yes | Yes |
| **B. Verification** | Is that transform correct *away from the board*, including at long range? | **No** (except optional B4) | **Never** | Yes |
| **C. Evaluation** | Sparse ground truth on realistic scenes | **No** | No | Yes |
| **S3 characterisation** | How accurate is the LiDAR itself? | **No, never** | No | Yes, but decoupled |

**Why B exists.** Every test that lives on the calibration board can only confirm that `T2`
is optimal *where it was fitted*. The `data_2026-06-28` diagnostics burned three rounds on an
apparent horizontal offset that could neither be confirmed nor refuted, because no
measurement existed outside the board's angular window. Scene B is that measurement.

**[v3] B now also carries all long-range validation.** Separating two requirements that v2
conflated:

- **Observability** (can the parameters be estimated?) needs spread in 1/Z, which short range
  provides best. This is Scene A's job.
- **Validation** (is the projection demonstrably correct at the ranges we evaluate at?) needs
  coverage out to the evaluation range. This is Scene B's job, and needs no checkerboard.

**Why B and C are separate.** B is designed for *measurability*: isolated objects, crisp
silhouettes, huge depth gaps, so an edge is locatable to ±2 px. C is designed for
*representativeness*: clutter, thin structures, textureless regions. Merging them gives a
scene that does neither job.

**Rig movement rules, which differ per scene:**

- **Scenes A and B:** the ROSbot stays completely still. Move the targets.
- **Scene C:** you may move the whole ROSbot, including raising, lowering or tilting it. What
  must never change is the **mount**, i.e. the camera's position relative to the LiDAR.
- **S3 characterisation:** move the robot freely. Fully decoupled from `T2`.

`zed_capture.py` reads the ZED's IMU at every capture and warns if roll or pitch has drifted
more than 1° from the session's first reading.

---

## 4. Scene A — calibration [v3 — revised]

**Contents: one checkerboard, and nothing else.** Plain wall behind it. Nothing at scan
height anywhere in the LiDAR's forward view except the board itself.

### Why 0.6–1.5 m (Fix 2)

The calibration solves a rigid transform — ψ, `t_x`, `t_y` — which has no range-dependent
term. Once correctly estimated it is as valid at 8 m as at 0.6 m. What needs range diversity
is separating yaw from lateral translation, because the column error decomposes as

```
Δu = −f_x · δψ  −  f_x · δt_y / Z
```

The yaw term is constant with range; the translation term falls as 1/Z. They separate only
if 1/Z varies across the pose set:

| Range | 1/Z spread |
|---|---|
| 0.8 – 3.0 m (v2) | 0.917 |
| **0.6 – 1.5 m (v3)** | **1.000** |
| 1.0 – 5.0 m | 0.800 |

Short range carries *more* of this information, not less. Far poses additionally degrade the
`solvePnP` plane normal, because the perspective cue that fixes a planar target's orientation
weakens as it shrinks in the image. Long-range validation moves to Scene B (§5).

### Bearing scales with distance

At short range a large bearing pushes the board out of frame; at 1.5 m the board can reach
±35°. This costs nothing, because the lateral offset `y_R = d · sin β` that conditions yaw is
dominated by the far poses: 0.15 m at 0.6 m and 15°, versus 0.86 m at 1.5 m and 35°. Every
pose below clears the frame by at least 160 px.

### Pose list (24 poses)

Distances are camera-to-board-centre. **Forward** and **lateral** are what you measure on the
floor; lateral positive = camera's left.

Six poses are **hold-outs**: captured identically, never used to fit `T2`, reserved for
measuring calibration accuracy honestly. One or two per distance, so the fit set keeps the
full 1/Z spread (1.000) and normal range (±75°).

#### 0.6 m

| Pose | Forward (m) | Lateral (m) | Bearing | Board yaw | Normal | Role |
|---|---|---|---|---|---|---|
| A01 | 0.60 | +0.00 | 0° | −45° | −45° | fit |
| A02 | 0.58 | +0.15 | +15° | −30° | −15° | fit |
| A03 | 0.58 | −0.15 | −15° | −15° | −30° | **hold-out** |
| A04 | 0.59 | +0.08 | +8° | 0° | +8° | fit |

#### 0.8 m

| Pose | Forward (m) | Lateral (m) | Bearing | Board yaw | Normal | Role |
|---|---|---|---|---|---|---|
| A05 | 0.75 | +0.27 | +20° | +15° | +35° | fit |
| A06 | 0.75 | −0.27 | −20° | +30° | +10° | fit |
| A07 | 0.79 | +0.14 | +10° | +45° | +55° | **hold-out** |
| A08 | 0.79 | −0.14 | −10° | −45° | −55° | fit |

#### 1.0 m

| Pose | Forward (m) | Lateral (m) | Bearing | Board yaw | Normal | Role |
|---|---|---|---|---|---|---|
| A09 | 1.00 | +0.00 | 0° | −30° | −30° | fit |
| A10 | 0.91 | +0.42 | +25° | −15° | +10° | fit |
| A11 | 0.91 | −0.42 | −25° | 0° | −25° | fit |
| A12 | 0.98 | +0.21 | +12° | +15° | +27° | **hold-out** |
| A13 | 0.98 | −0.21 | −12° | +30° | +18° | fit |

#### 1.2 m

| Pose | Forward (m) | Lateral (m) | Bearing | Board yaw | Normal | Role |
|---|---|---|---|---|---|---|
| A14 | 1.04 | +0.60 | +30° | +45° | +75° | fit |
| A15 | 1.04 | −0.60 | −30° | −45° | −75° | fit |
| A16 | 1.16 | +0.31 | +15° | −30° | −15° | **hold-out** |
| A17 | 1.16 | −0.31 | −15° | −15° | −30° | fit |
| A18 | 1.20 | +0.10 | +5° | 0° | +5° | fit |

#### 1.5 m

| Pose | Forward (m) | Lateral (m) | Bearing | Board yaw | Normal | Role |
|---|---|---|---|---|---|---|
| A19 | 1.50 | +0.00 | 0° | +15° | +15° | fit |
| A20 | 1.23 | +0.86 | +35° | +30° | +65° | **hold-out** |
| A21 | 1.23 | −0.86 | −35° | +45° | +10° | fit |
| A22 | 1.41 | +0.51 | +20° | −45° | −25° | fit |
| A23 | 1.41 | −0.51 | −20° | −30° | −50° | **hold-out** |
| A24 | 1.48 | +0.26 | +10° | −15° | −5° | fit |

Machine-readable copy: `poses_sceneA_v3.csv`.

### Coverage

| Property | v2 design | **v3** |
|---|---|---|
| Distance range | 0.8 – 3.0 m | **0.6 – 1.5 m** |
| 1/Z spread | 0.917 | **1.000** |
| Board normal range | ±55° | **±75°** |
| Board yaw (incidence) range | ±45° | ±45° |
| Board-yaw-zero poses | 3 | 3 |
| Total / hold-out poses | 28 / 6 | 24 / 6 |
| Minimum frame margin | 60 px | **160 px** |

### Two captures: plain, then ChArUco

Capture the full 24-pose sequence with the **plain board first**, then repeat all 24 poses
with the **ChArUco board**, without touching the rig in between. Swap the board at each floor
mark rather than running two separate laps, if that is faster — the order does not matter,
the rig stability does.

Use **one `--out` session directory for both boards**, so exposure lock, the session
manifest and the IMU reference attitude are shared. Distinguish them by pose id:

| Board | Pose ids | Bag names |
|---|---|---|
| Plain | `A01` – `A24` | `A01` – `A24` |
| ChArUco | `AC01` – `AC24` | `AC01` – `AC24` |

**Why twice:** two independent captures give two independent solutions for `T2`, and their
spread bounds the real calibration accuracy far better than any single fit's residual. This
is the repeatability evidence that was missing during the `data_2026-06-28` diagnostics.

**Keep the failed first capture separate.** Do not reuse its session directory or its pose
ids.

### Grazing check before committing to ±45°

Watch `ros2 topic echo /scan --once` while increasing board yaw. Stop where intensity drops
sharply or range scatter visibly grows. If ±45° proves marginal, cap at ±35° and record the
reason.

### Per-pose procedure

1. Place the board on its marked position; set yaw using the line-of-sight string.
2. Start the bag: `ros2 bag record -s mcap -o A01 /scan`
3. Capture, **passing the geometry flags** (they were omitted last session, leaving every
   distance as `?` in the diagnostics):

   ```
   python3 zed_capture.py --pose-id A01 --distance 0.6 --bearing 0 --yaw -45
   ```

4. **Do not touch the rig or the board until the script prints DONE.** The whole burst is
   used downstream (Fix 1), so any movement during recording blurs every averaged corner.
5. Stop the bag.
6. Check the QC snapshot: board fully in frame, unoccluded, pattern crisp.

### Gate: check the first three poses before continuing

After A01–A03, **stop and run `check_corners.py`** (with the updated `BOARD CONFIG`) on those
captures. Expect all 18 corners found, and roughly 43 px per checker at 0.6 m. If detection
is flaky, apply the quiet-zone fix from §2 before continuing. Five minutes here protects the
other 21 poses.

### Downstream requirement: burst averaging (Fix 1)

**No change to capture. This is a processing requirement recorded here so it is not lost.**

With only 18 corners, single-frame PnP gives a scan-line plane error of roughly 12 mm RMS
(25 mm p95) at a far frontal pose — comparable to the S3's ±30 mm spec. Averaging over the
burst cuts random corner noise by roughly √N:

| Pose | Single frame | Averaged over 100 frames |
|---|---|---|
| 1.5 m, frontal | 5.5 mm RMS | **0.5 mm** |
| 1.2 m, frontal | 2.8 mm RMS | **0.3 mm** |

**Implementation:** extract a **temporal-mean image** from each pose's SVO2 (rectified left,
≥100 frames) and feed that to the calibration pipeline in place of the single QC snapshot.
This leaves `cam_lidar_2d_icp.py` untouched and works because the scene is static and
exposure is locked.

**Caveat:** averaging removes random noise only. Systematic error — print scale, board
flatness, residual rectification — is untouched, which is why the §2 pitch and flatness checks
are mandatory.

The default `zed_capture.py` burst (10 s at 15 fps ≈ 150 frames) is sufficient.

---

## 5. Scene B — verification

**Contents: 3–4 thin flat vertical plates. No checkerboard** (except the optional B4).

### What "vertical plates" means, concretely

Not boxes. Not cylinders. Flat, thin, rigid panels standing upright on the floor:

- **Size:** roughly 400 mm wide × 700 mm tall each. Height must comfortably span the scan
  plane (≤ 175 mm).
- **Thickness: 3–6 mm.** Foam core, thin plywood, or stiff mounting card. Thin matters: the
  camera and LiDAR are offset laterally by ~8 cm, so a thick object lets them see different
  faces at a silhouette.
- **Finish: high contrast against the background.** Matte white plates against a dark
  backdrop, or matte black plates against a white wall, so the silhouette is locatable to
  ±2 px.
- **Each plate faces roughly toward the camera** (ψ ≈ 0), so its thin vertical edge is what
  gets silhouetted.
- **Support:** a foot or clamp that does **not** intrude at scan height and does not stick
  out sideways past the plate's edges.

**Why flat plates and not cylinders:** a cylinder's silhouette is a tangent line, and the
camera and LiDAR tangent it from different positions, so they see genuinely different
physical points. That produces a systematic column offset indistinguishable from a
calibration error. A flat plate's edge is the same physical edge for both sensors.

### Layout

All plates in the same capture, at different bearings **and** different distances. Leave
**≥ 0.5 m of clear space behind each plate** so its range discontinuity is unambiguous, and
stagger the distances so no plate hides behind another.

| Arrangement | Plate 1 | Plate 2 | Plate 3 |
|---|---|---|---|
| **B1** | β = −30°, d = 1.5 m | β = −10°, d = 2.5 m | β = +10°, d = 2.0 m |
| **B2** | β = −20°, d = 2.0 m | β = +20°, d = 1.2 m | β = +35°, d = 2.5 m |
| **B3** | β = −35°, d = 2.8 m | β = 0°, d = 1.0 m | β = +25°, d = 1.8 m |
| **B4** *(optional, recommended)* **[v3]** | plain checkerboard at β = 0°, **d = 1.2 m**, ψ = 0° | plate at β = −30°, d = 2.0 m | plate at β = +30°, d = 2.0 m |
| **B5** **[v3 — new]** | β = −15°, **d = 3.5 m** | β = +5°, **d = 5.0 m** | β = +20°, **d = 4.0 m** |

**[v3] B4:** the checkerboard moves from 2.0 m to 1.2 m to stay inside the A3 board's
comfortable detection range. B4 puts the on-board residual and the off-board column
measurement in the *same frame*, directly testing whether agreement on the board implies
agreement off it.

**[v3] B5 is mandatory.** Scene A now stops at 1.5 m, so B5 is the only evidence that the
projection holds at the ranges the depth models will be evaluated at. A 400 mm plate at 5 m
still gives ~40 LiDAR returns and ~42 px of image width, enough for both edges to be located.
If the room cannot accommodate 5 m, go as far as it allows and record the maximum achieved.

Measure each plate's forward and lateral position with a tape, and its distance with the
laser rangefinder. Record all of it with `--note`.

### Capture

```
python3 zed_capture.py --pose-id B1 --scene verification --note "plates at -30/-10/+10, 1.5/2.5/2.0 m"
```

**Capture B1 twice: once at the very start of the session and once at the very end.** If the
two disagree, the rig moved partway through. If they agree, you have positive evidence the
extrinsics held across everything in between.

---

## 6. Scene C — evaluation scenes

**Contents: realistic scenes. No checkerboard.**

### [v3] Declared operating range

**`Z_min` = 0.5 m, `Z_max` = 8 m.** `Z_min` is a declared system parameter, not a scene
measurement: it sets the maximum disparity every depth-model engine is built with (128 px at
FULL tier, 64 px at HALF). Therefore:

- **Design every evaluation scene so no surface falls nearer than 0.5 m** to the camera.
- Ground-truth points with `d_gt < 0.5 m` or `d_gt > 8 m` are flagged out-of-range and
  excluded from headline metrics.
- Changing `Z_min` later requires rebuilding every engine.

### What "multiple depth layers" means

Objects at several distinct distances arranged so that, sweeping across the image from left
to right, the depth changes repeatedly with gaps of at least 0.3 m. A single wall gives one
depth layer and is nearly useless as an evaluation scene.

### Recipe — 8 scenes

| Scene | Contents | What it tests |
|---|---|---|
| **C1** | The room as-is: furniture, clutter, mixed distances | General case |
| **C2** | Boxes staggered at 1, 2, 3, 5 m, offset laterally so none occludes another | Depth range, clean layers |
| **C3** | Blank textureless wall filling most of the frame | Stereo failure mode: no texture |
| **C4** | Repeating texture: blinds, bookshelf spines, tiled or grid surface | Stereo failure mode: ambiguous matching |
| **C5** | Thin structures: chair legs, broom handle, tripod legs, cable | Thin-structure and occlusion handling |
| **C6** | Dark and specular: black monitor, metal, glass, gloss paint | Low-return surfaces, for both sensors |
| **C7** | Strong occlusion boundaries: near object partly overlapping a far one | Edge bleeding, the parallax case |
| **C8** | **Metrology scene**: two or three flat panels at rangefinder-verified distances, nothing else | Absolute accuracy anchor |

### Rig height and tilt variants

For each scene, capture at **3–5 different whole-rig heights or tilts**. Raising, lowering or
tilting the ROSbot as a rigid unit leaves `T2` and `Δz` untouched while the scan plane sweeps
through a different slice of the scene. Each setting yields an independent, fully valid
(image, sparse-GT) pair.

**Never move one sensor relative to the other.** Check the `zed_capture.py` IMU warning after
each move.

### Record per scene

Lighting conditions (lux if you have a meter), surface materials present, and anything
unusual. Use `--note`.

---

## 7. S3 characterisation

**Contents: a flat matte target and a rangefinder. No checkerboard, ever.**

This workstream produces `bias(r)` and `sigma(r)`, which are properties of the LiDAR alone.
It is completely independent of `T2`, so **run it as a separate session**, ideally a different
day. It requires moving the robot constantly and swapping targets, which is incompatible with
the "do not disturb the rig" rule governing Scenes A, B and C.

### What "incidence angle" means here

**The angle between the LiDAR beam and the target surface's normal.** `0°` means the beam
strikes head-on; `60°` means a glancing strike. For a target on the centreline, incidence
angle is numerically the same as board yaw from §1.

### Targets

- **A movable flat matte panel**, roughly 600 × 600 mm, on a rotatable stand that spans the
  scan plane. Main target for incidence and material sweeps.
- **A large blank painted wall** for the long-distance end of the sweep.

### Measurements

| Test | Setup | Output |
|---|---|---|
| **Accuracy and precision vs distance** | Wall normal to the beam; move the robot from 0.5 m to 8 m in 0.25–0.5 m steps; 300 scans each; rangefinder as truth | `bias(r)`, `sigma(r)` |
| **Incidence angle** | Panel on the centreline at 2 and 4 m; rotate to 0°, 15°, 30°, 45°, 60° | bias vs incidence |
| **Surface material** | Panel at 2 m, 0° incidence; swap facing: white paper, matte grey, matte black, bare wood, painted wall | bias and dropout vs albedo |
| **Bearing dependence** | Panel at 2 m, moved to several bearings across the field of view | bias vs bearing; tests the scan-flatness spec |
| **Edge / discontinuity** | Panel at 1.5 m with the wall 1–3 m behind; sweep the gap | **sets `τ` and `k` in the occlusion filter** |

### Rangefinder reference point

The S3's laser emission origin is **not** at the housing's outer surface. Take the internal
offset from the mechanical drawing in the datasheet and apply it consistently, or measure to a
documented physical reference point on the housing and record the offset separately.

---

## 8. Pre-session checklist [v3 — revised]

**Rig**

- [ ] `config/rig_<id>.json` populated with real metrology, including the measured scan
      plane height; validator passes.
- [ ] Inclinometer roll and pitch recorded, with instrument spec.
- [ ] `python3 zed_capture.py --level` run; roll and pitch near zero.
- [ ] LaserScan handedness verified empirically and written into the README.

**Boards**

- [ ] **Checker pitch measured** across 6 squares on both boards; measured value recorded.
- [ ] **Flatness checked** across both diagonals of both panels; worst gap recorded.
- [ ] `board_standoff_m = 0` recorded (scan plane strikes the print).
- [ ] **Board config updated** to 6 × 3 inner corners and measured pitch in both
      `cam_lidar_2d_icp.py` and `check_corners.py`.
- [ ] Stands fitted behind the panels; nothing protrudes at scan height.
- [ ] Verification plates built: 3–6 mm thick, high contrast, feet clear of scan height.

**Capture system**

- [ ] **Disk write throughput tested.** If the Orin Nano cannot sustain lossless HD720,
      reduce fps, never compression quality.
- [ ] **Free space checked.** Roughly 48 Scene A captures (24 × 2 boards) × 10 s × ~20 MB/s
      ≈ 10 GB, plus B and C. Budget 30 GB.
- [ ] **Concurrent load test**: both sensors streaming for 60 s, counts verified.
- [ ] `SN#####.conf` confirmed present in `/usr/local/zed/settings/`.
- [ ] **New session directory** for this capture; the failed first capture kept separate.

**Floor**

- [ ] Centreline taped; all 24 Scene A positions marked at 0.6, 0.8, 1.0, 1.2 and 1.5 m, and
      labelled.
- [ ] Line-of-sight string and protractor ready for setting board yaw.
- [ ] Space confirmed for B5 plates at 3.5–5 m.

## 9. Time budget [v3 — revised]

| Block | Estimate |
|---|---|
| Rig metrology, levelling, board checks, marking the floor | 1.5 h |
| Scene A, plain board, 24 poses at 2–3 min | 1.0 h |
| Scene A, ChArUco board, 24 poses | 1.0 h |
| Scene B, 5 arrangements plus the repeat of B1 | 0.75 h |
| Scene C, 8 scenes × 3–5 rig settings | 2.5 h |
| **Session total** | **~7 h — budget a full day** |
| S3 characterisation | **Separate session, ~4 h** |

If the day runs short, **Scene A (both boards) and Scene B are the priority**: they establish
and validate `T2`. Scene C can be captured in a later session provided the mount has not been
touched and B1 is repeated at the start of that session to prove it.
