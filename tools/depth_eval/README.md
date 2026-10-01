# tools/depth_eval/

Depth-model evaluation harness against this project's sparse LiDAR-referenced ground truth
(`tools/lidar_ground_truth/export_scene.py` output). Metrics and their justification are in
`docs/DEPTH_METRICS.md`. Headless, standalone consumer of a session's ground-truth export and
raw `.svo2` captures, following the precedent of `tools/lidar_ground_truth/` and
`tools/lidar_characterisation/`: explicit CLI args, no hidden defaults, `config.py` holds every
threshold. Run every script from the **repository root**, modules resolved via Python's
automatic script-directory `sys.path` entry (same pattern as the other `tools/` packages).

## What this evaluates, and what it doesn't

Every ground-truth point is sampled from a predictor's **dense** depth map (bilinear, at the
ground truth's sub-pixel `(u, v)`, `sampling.py`) — the model still predicts a full frame, as it
always does; only the *scoring* is restricted to the sparse ground truth, which itself only
covers a thin horizontal strip (a single 2D LiDAR scan line, `REPORT.md` Sec 7). This is the
same sparse-evaluation protocol used by, e.g., KITTI depth, just with much sparser (line, not
scattered-point) coverage — not a novel or looser methodology.

## Architecture

| Module | Responsibility |
|---|---|
| `config.py` | Depth buckets, metric guards, ZED SDK runtime-parameter defaults |
| `ground_truth.py` | Load one scene's exported ground truth, grouped by capture, with each capture's `.svo2` resolved |
| `sampling.py` | Bilinear sampling of a dense depth map at sparse sub-pixel `(u, v)`; any-neighbour-invalid rejects the sample |
| `metrics.py` | The metric set (`docs/DEPTH_METRICS.md`), vectorised, over an arbitrary group of points |
| `zed_sdk_predictor.py` | Backend: opens a capture's `.svo2`, runs one ZED SDK depth mode, retrieves a dense depth map |
| `evaluate.py` | CLI: ties the above together, reports overall / per-scene / per-bucket / per-capture |

A future TensorRT-engine or native-PyTorch-model backend plugs in as a new `predict()`-shaped
module alongside `zed_sdk_predictor.py`; `evaluate.py`'s `--backend` flag and `evaluate_capture()`
are the extension point. `ground_truth.py`, `sampling.py` and `metrics.py` are already
backend-agnostic.

## Known environment requirement: ZED SDK NEURAL* modes need an explicit `LD_LIBRARY_PATH`

`NEURAL`/`NEURAL_PLUS`/`NEURAL_LIGHT` need TensorRT's `libnvinfer_builder_resource` to build
their engine (a one-time, per-mode, per-GPU optimisation step cached afterward; `QUALITY`,
`ULTRA`, `PERFORMANCE` are classical algorithms and don't need this). On this machine the file
is physically present (`/usr/local/cuda-13.3/targets/x86_64-linux/lib/`) but `ldconfig` doesn't
resolve it automatically (NVIDIA's CUDA packaging appears to deliberately exclude this specific
library from automatic linking — confirmed read-only, not changed). Until/unless that's fixed at
the system level, run NEURAL* modes with:

```bash
LD_LIBRARY_PATH=/usr/local/cuda-13.3/targets/x86_64-linux/lib python tools/depth_eval/evaluate.py ...
```

This is a per-invocation environment variable, not a system change.

## Running

```bash
python tools/depth_eval/evaluate.py \
    --session data/<session> \
    --scenes scene_B scene_C \
    --backend zed_sdk --depth-mode NEURAL \
    --baseline-m <session_manifest.json's calibration.baseline_m> \
    --out-dir results/depth_eval/<session>/zed_sdk_NEURAL
```

`--capture-ids` restricts to specific captures (e.g. for a smoke test); omit for every capture in
the chosen scenes. `--export-dir` overrides the default
`results/lidar_ground_truth/<session>/export`.
