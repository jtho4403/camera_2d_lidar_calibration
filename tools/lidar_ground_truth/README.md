# tools/lidar_ground_truth/

Sparse LiDAR-referenced ground-truth depth pipeline. See `PLAN.md` at the
repository root for the full specification; this README covers frame
conventions, the projection derivation, and how to run what exists so far
(Phase 0/1: rig parameters + projection core + overlay diagnostic).

Headless, standalone consumer of the existing calibration artifacts
(`lidar_to_camera_2d.npy`, `calibration_result.json`), following the
precedent of `tools/lidar_characterisation/`. Not part of the interactive
GUI flow. Run every script from the **repository root** with this
directory's modules resolved via Python's automatic script-directory
`sys.path` entry (same pattern as `cam_lidar_2d_icp.py`'s `import icp_2d`
and `tools/lidar_characterisation`'s `import config`) -- plain
`import config` / `import rig` etc. inside this package only resolves when
the entry-point script also lives in `tools/lidar_ground_truth/`.

## Frames (PLAN.md Sec 1.1)

| Symbol | Frame | Convention |
|---|---|---|
| `L` | LiDAR scan frame | right-handed, x forward, y left, z up; origin at the scanner rotation centre; every return has z = 0 |
| `R` | Camera "robot" frame | right-handed, x forward, y left, z up; origin at the rectified left camera optical centre |
| `C` | Camera OpenCV frame | x right, y down, z forward; same origin as `R` |

`R -> C` is the fixed rotation already encoded by `rot_cam_to_robot` in
`camera_2d_lidar_calibration/gui.py::ImageVisInterface.done_callback`.

## The projection derivation (PLAN.md Sec 1.2)

The existing calibration gives `T2` (3x3, SE(2)), mapping a LiDAR point into
the camera's horizontal plane: `[x_R, y_R]^T = R2 [x_L, y_L]^T + t2`. Under
the zero relative roll/pitch assumption, every LiDAR return sits at a
constant height `z_R = delta_z` in frame R. Substituting into the pinhole
model on **rectified** images (distortion exactly zero):

```
d_gt = x_R                          ground-truth depth (camera-frame Z)
u    = c_x - f_x * y_R / x_R        image column
v    = c_y - f_y * delta_z / x_R    image row
```

`d_gt` and `u` depend only on `T2`. `delta_z`, pitch and roll enter in
exactly one place: the image row `v`. This is why the whole approach is
defensible: the parameters the simple wall-parallel capture protocol cannot
observe (vertical offset, relative pitch/roll) do not contaminate the
ground-truth depth value or column -- see `projection.py` for the
implementation and PLAN.md Sec 2/9 for the full error-budget and
observability discussion. **Never let a vertical parameter leak into the
depth or column computation.**

## Modules (Phase 0/1)

| Module | Responsibility |
|---|---|
| `config.py` | paths, thresholds, constants |
| `rig.py` | load/validate `config/rig_<id>.json`; row-geometry sanity check CLI (PLAN.md Sec 1.3) |
| `scan_io.py` | `.pcd` reading, invalid-return filtering, image/laser folder pairing |
| `calibration_io.py` | load `lidar_to_camera_2d.npy`, `calibration_result.json`, rectified `K` |
| `projection.py` | the `project_scan` core above |
| `filters.py` | model-independent occlusion/parallax rejection (incidence-angle filter deferred to Phase 5) |
| `overlay_diagnostic.py` | Phase 1 verification-gate CLI (PLAN.md Sec 5.5) |

## Rig parameters (PLAN.md Sec 3)

`config/rig_template.json` is a **placeholder** -- every metrology value in
it (delta_z, pitch, roll, the delta_z provenance chain) is illustrative, not
measured. `rig.load_rig(path, allow_placeholder=False)` (the default)
refuses to load it and raises `RuntimeError`; every diagnostic tool that
must run before real metrology exists (this package's Phase 0/1 tools) opts
in with `allow_placeholder=True` and must call `rig.warn_if_placeholder`
immediately after, which prints a loud banner. Do not trust `v`/row output
from any run against the template file.

## Running

```bash
# Phase 0: schema-validate the rig file, demonstrate the strict placeholder
# guard, and print the Sec 1.3 row-geometry table using a real metadata K.
python tools/lidar_ground_truth/rig.py \
    --metadata data/data_2026-06-28/additional_image_data/metadata_pose_01.json

# Phase 1: project an existing session's curated LiDAR scans into their
# paired calibration images and save overlay diagnostics.
python tools/lidar_ground_truth/overlay_diagnostic.py \
    --image-dir data/data_2026-06-28/images \
    --laser-dir data/data_2026-06-28/lasers \
    --metadata-dir data/data_2026-06-28/additional_image_data \
    --out-dir results/lidar_ground_truth/data_2026-06-28
```

## Open questions / not yet verified

- **RPLIDAR S3 invalid-return convention.** `scan_io.load_valid_xy` checks
  every convention a common ROS `LaserScan -> PointCloud` path could
  produce (exact zero, NaN/Inf, out of `[0.05, 40]` m datasheet range), but
  none of this repo's existing `.pcd` data was captured on an S3 -- it is
  from the STL-19P via the old rig. Reconfirm against a real captured S3
  PCD in Phase 3.
- **RPLIDAR S3 LaserScan handedness.** PLAN.md Sec 6.2 flags that the S3
  datasheet specifies a left-handed, clockwise-increasing-angle frame,
  while ROS REP-103 is right-handed with y left. The existing `.pcd` data
  (STL-19P) is consistent with frame `L`'s right-handed x-forward/y-left
  convention -- the existing SE(2) calibration solved a physically sensible
  transform from it, and this package's own overlay diagnostic confirms the
  same data projects onto the checkerboard at the predicted row. Neither
  fact says anything about the S3's own driver convention; that must be
  verified empirically (a known-bearing object test) once real S3 capture
  exists (Phase 3), per PLAN.md's explicit instruction not to guess this.
