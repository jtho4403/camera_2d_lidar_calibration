# tools/lidar_ground_truth/

Sparse LiDAR-referenced ground-truth depth pipeline. See `PLAN.md` at the
repository root for the full specification; this README covers frame
conventions, the projection derivation, and how to run what exists so far
(Phase 0/1: rig parameters + projection core + overlay diagnostic).

Headless, standalone consumer of a session's calibration artifacts
(`results/calibration/<session>/lidar_to_camera_2d.npy`,
`calibration_result.json`) and its `session_manifest.json` (rectified `K`),
all passed explicitly on the command line, following the
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
| `calibration_io.py` | load `lidar_to_camera_2d.npy`, `calibration_result.json`, rectified `K` from `session_manifest.json` (refuses a `K` that differs from the calibration's) |
| `projection.py` | the `project_scan` core above |
| `filters.py` | model-independent occlusion/parallax rejection (incidence-angle filter deferred to Phase 5) |
| `overlay_diagnostic.py` | Phase 1 verification-gate CLI (PLAN.md Sec 5.5) |

## Rig parameters (PLAN.md Sec 3)

`config/rig_template.json` is a **placeholder** for the ROSbot XL + ZED 2i +
RPLIDAR S3 rig: no metrology has been recorded yet (delta_z, pitch, roll,
the delta_z provenance chain and board_standoff are zeroed).
`rig.load_rig(path, allow_placeholder=False)` (the default) refuses to load
it and raises `RuntimeError`; tools that must run before real metrology
exists opt in with `allow_placeholder=True` and must call
`rig.warn_if_placeholder` immediately after. Schema validation always runs,
and requires `delta_z_tolerance_m > 0`. Do not trust `v`/row output from any
run against the template file.

## Running

```bash
# Phase 0: schema-validate the rig file, demonstrate the strict placeholder
# guard, and print the Sec 1.3 row-geometry table using the session's K.
python tools/lidar_ground_truth/rig.py \
    --camera-manifest data/<session>/captures/session_manifest.json

# Phase 1: project a session's staged LiDAR scans into their paired
# calibration images and save overlay diagnostics.
python tools/lidar_ground_truth/overlay_diagnostic.py \
    --image-dir data/<session>/images \
    --laser-dir data/<session>/lasers \
    --camera-manifest data/<session>/captures/session_manifest.json \
    --transform results/calibration/<session>/lidar_to_camera_2d.npy \
    --calibration-result results/calibration/<session>/calibration_result.json \
    --out-dir results/lidar_ground_truth/<session>
```

## Sensor conventions confirmed on real S3 data (data_2026-09-23)

- **Invalid returns.** The extracted S3 PCDs contain no exact `(0, 0)`
  points and no NaN/Inf; no-return bearings are omitted, so the point count
  per revolution varies. Returns lie on a fixed 0.1108 deg bearing grid.
- **LaserScan handedness and mounting.** Frame `L` is right-handed
  (x forward, y left, per REP-103) but its +x axis points ~180 deg away from
  the camera's optical axis on this rig: the solved SE(2) yaw is 178.87 deg.
  A mirrored (left-handed) frame could not be fitted by an SE(2) transform
  across the session's +-70 deg of board yaw; the fitted per-pose line
  angles are 0.05-2.7 deg.
