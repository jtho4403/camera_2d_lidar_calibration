# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A camera + 2D LiDAR extrinsic/intrinsic calibration toolkit (originally an ACFR-RPG ROS2 package, `ament_python` build type, but the calibration scripts themselves are plain standalone Python — the `rclpy`/ROS node imports in `cam_lidar_2d_icp.py` are commented out). It computes:

1. Camera intrinsics + distortion via `camera_2d_lidar_calibration/cam_intrinsic.py`.
2. The SE(2) transform between a 2D LiDAR frame and the camera frame via `camera_2d_lidar_calibration/cam_lidar_2d_icp.py`, using checkerboard-derived lines matched with 2D ICP.

See `README.md` for the full experimental procedure (rig setup, ROS bag recording, assumptions about wall/checkerboard geometry) — that detail is not repeated here.

## Running

Both entrypoints must be run from the **repository root** (not from inside `camera_2d_lidar_calibration/`), because `cam_lidar_2d_icp.py` imports sibling modules (`gui`, `icp_2d`) via plain `import gui` / `import icp_2d`, which only resolves via Python's automatic script-directory `sys.path` entry:

```bash
python camera_2d_lidar_calibration/cam_intrinsic.py <path/to/image/folder>
python camera_2d_lidar_calibration/cam_lidar_2d_icp.py <path/to/image/folder> <path/to/laser/folder>
```

The extrinsic script (`cam_lidar_2d_icp.py`) is interactive: for each image/cloud pair it opens a Tk/matplotlib window to confirm checkerboard detection (`ImageVisInterface`), then another to let you zoom in and select the LiDAR points on the wall (`SelectPointsInterface`). It cannot run headless.

`test.py` and `tools/plot_lidar_candidates.py` are standalone diagnostic scripts, not a test suite — there is no pytest suite despite `python3-pytest` being listed as a `test_depend` in `package.xml`. Run them directly with `python test.py` / `python tools/plot_lidar_candidates.py --input-root ...` when needed.

## Setup

```bash
pip install -e .
```

`setup.py`'s `install_requires` (`opencv-python`, `tk`, `matplotlib`, `scikit-learn`, `numpy<2.0`) is incomplete relative to actual runtime needs — `open3d` and `scipy` are required (per README and actual imports) but not declared. The checked-in `.venv` currently has `numpy 2.2.6`, which also violates the `numpy<2.0` pin in `setup.py` — don't assume the pin is authoritative.

## Architecture

**Pipeline shape of `cam_lidar_2d_icp.py::main()`** (the core of the repo):

1. Load images/point clouds from two folders via `load_images_from_folder` / `load_clouds_from_folder`. Pairing between an image and a cloud is purely by **sorted filename order** within each folder — there is no timestamp or ID matching, so folder contents must already be curated 1:1 (see `data/` layout below).
2. For each pair: detect the checkerboard (`cv2.findChessboardCorners` + `solvePnP`), then launch `ImageVisInterface` (`gui.py`) so a human confirms the detection and the interface derives a 3D line along the checkerboard's local X axis (representing the wall), converted from OpenCV camera convention into a robotics convention (`ImageVisInterface.done_callback` does the `rot_cam_to_robot` axis swap — this is the one place where the CV/robot frame convention mismatch is handled, and it's load-bearing for everything downstream).
3. For the same pair, launch `SelectPointsInterface` (`gui.py`) so a human manually selects the LiDAR points that lie on the wall.
4. Once every pose has a camera-derived line and a LiDAR point selection, run **staged coarse-to-fine 2D ICP** (`run_staged_icp`) starting from a rough manually-measured initial transform (hardcoded rig offsets), through three distance-threshold stages (0.30 m → 0.15 m → 0.10 m). ICP correspondence search is done **per-pose in isolation** (`icp_2d.icp_per_line`) so that one pose's points never match against another pose's line.
5. Validate the result: orthogonal point-to-infinite-line residuals per pose (`evaluate_alignment_residuals`), a sanity check against the rough manual rig measurement (`validate_against_manual_rig_geometry`), and an independent fixed-threshold sensitivity sweep (`run_threshold_sensitivity`) that reruns ICP from scratch at several thresholds to check the staged result isn't threshold-fragile.
6. Persist everything: `lidar_to_camera_2d.npy` (the 3x3 transform), `calibration_result.json` (full payload — transform, intrinsics, residuals, sanity checks, sensitivity sweep), `calibration_residual_summary.csv`, and two diagnostic plots (`checkerboard_lidar_pre_alignment.png`, `aligned_point_clouds.png`).

**`icp_2d.py`** is a vendored/adapted 2D point-to-point ICP (credited to github.com/richardos/icp in a comment) with two entry points: `icp()` for a single point set, and `icp_per_line()` which does the same but keeps N separate reference/query point sets correspondence-isolated per iteration while still solving one shared rigid transform per iteration (used exclusively by the staged pipeline above).

**`gui.py`** holds both Tk/matplotlib interactive interfaces described above. Both follow the same pattern: build a Tk window with a matplotlib canvas embedded via `FigureCanvasTkAgg`, block on `self.app.mainloop()` inside `run()`, and communicate results back through instance state read after `mainloop()` returns.

**Hardcoded, must-stay-in-sync calibration parameters**: checkerboard dimensions/square size are hardcoded independently in both `cam_intrinsic.py` and `cam_lidar_2d_icp.py::main()` (`checkerboard_width`/`_height`/`_size`) — there is no shared config, so changing checkerboard rigs means editing both files. `cam_lidar_2d_icp.py::main()` also hardcodes camera intrinsics/distortion (currently ZED 2i HD720 left-camera values) and asserts input images are exactly `1280x720`, and hardcodes the rough manual rig offset used both as the ICP initial transform and as the sanity-check reference (`measured_camera_origin_in_lidar`, `measured_yaw_deg`).

## Data layout

`data/` contains dated capture sessions (e.g. `data/data_2026-06-28/`, `data/data_2026-07-10/`), each following the same internal structure: `images/left_pose_NN.png`, `lasers/laser_pose_NN.pcd`, `additional_image_data/{depth_raw,metadata}_pose_NN.{npy,json}`, and `raw_lasers/extracted_pcd/` (raw pre-selection candidate point clouds, consumed by `tools/plot_lidar_candidates.py` for picking good candidate scans before curating the final `lasers/` folder). `examples/` holds a small fixed image/laser pair set matching the README's walkthrough video. `debug_checkerboard/` and `raw_lasers/` are gitignored scratch output.
