# Reusable Claude Code prompt — ONNX export for a depth model candidate (v2)

Drop `DEPTH_MODEL_BUILD_SPEC.md` into the repo root alongside this, run `/init`, then paste
everything below the line.

### Changelog v1 → v2

| # | Change | Why |
|---|---|---|
| 1 | Stereo pair must span a wide depth range (Assumptions) | A pair with only far surfaces never exercises the large-disparity end of the cost volume, so gate G3 could miss fp16 degradation that only appears near the camera |
| 2 | `baseline_pytorch.py` is device-portable, with a benchmark mode (Step 4) | The same script becomes the native-PyTorch test on the Jetson, unchanged |
| 3 | Reference outputs computed with **TF32 disabled** (Step 4) | PyTorch enables TF32 for convolutions by default on Ampere-or-newer GPUs, which can fail G1 spuriously against ONNX Runtime's CPU fp32 |
| 4 | Exact preprocessed input tensors and ONNX Runtime outputs saved as artefacts (Step 6) | These become the gate G2 references on the Jetson: the engine is fed identical tensors and compared directly, so preprocessing drops out of the parity chain and ONNX Runtime is not needed on the Jetson |
| 5 | Iteration count in every artefact name for iterative models | Prevents engines at different iteration counts being confused |

**Assumed already done by the operator before this prompt is given:**

- The model's conda/venv environment is created and activated, and the repo's own
  dependencies install and import cleanly.
- The pretrained checkpoint is downloaded and its path is known.
- A real rectified stereo pair from the ZED 2i is present as PNGs (left and right,
  1280 × 720, rectified), and its path is known.
- **[v2] The pair spans a wide depth range**: at least one textured surface around
  0.5–1 m and at least one beyond about 4 m, so that disparities cover roughly 15–125 px at
  FULL tier. **Nothing in the pair nearer than 0.5 m** — that exceeds the locked max
  disparity of 128 px and will look like model failure. This pair is for parity checking
  only, not evaluation; the gates will be re-run on an evaluation-scene frame once one
  exists.

---

Read `DEPTH_MODEL_BUILD_SPEC.md` at the repo root in full before doing anything else. It is
the authoritative specification and its decisions are locked; do not renegotiate them. Then
explore this repository to understand the model's architecture, its inference entry point,
and how it loads a checkpoint.

Your job is **ONNX export only**. You are on the build machine, not the Jetson. Do not
attempt TensorRT, do not install `tensorrt`, do not benchmark for reporting purposes. The
engine build happens separately on the target device.

## Deliverables

Create a self-contained `export/` directory in this repo containing:

```
export/
  manifest.json                  per SPEC §10, filled as far as this stage allows
  preprocess.py                  the model's preprocessing, per SPEC §4
  baseline_pytorch.py            PyTorch reference + benchmark modes; portable (Step 4)
  export_onnx.py                 the export itself
  check_g1.py                    GATE G1: PyTorch vs ONNX Runtime
  run_all.sh                     one command that runs the above in order
  artifacts/
    <model>_full.onnx            1280x704
    <model>_half.onnx            640x352
    inputs_full.npz              [v2] exact preprocessed left/right tensors, FULL
    inputs_half.npz              [v2] same, HALF
    baseline_full.npy            PyTorch fp32 reference disparity, FULL
    baseline_half.npy
    ort_full.npy                 [v2] ONNX Runtime fp32 output, FULL -- the G2 reference
    ort_half.npy                 [v2]
    g1_report.json
    g1_diff_full.png             visualised difference map
    g1_diff_half.png
    timing_smoketest_<host>.json [v2] benchmark-mode smoke test; not a reported number
  README.md                      what was exported, what was decided, what is left for the Jetson
```

**[v2] Iterative models** carry the iteration count in every per-configuration artefact:
`<model>_<tier>_it<N>.onnx`, `baseline_<tier>_it<N>.npy`, `ort_<tier>_it<N>.npy`, and so on.
Input tensors do not depend on iteration count and are shared.

## Step 1 — investigate and report before writing code

Read the model's inference path and report to me:

1. **Output type**: disparity, metric depth, inverse depth, or scale-ambiguous relative
   depth. If scale-ambiguous, say so loudly — it needs a different evaluation protocol.
2. **Input arity**: separate `left`/`right` tensors, or one concatenated tensor.
3. **Exact preprocessing**: colour order, value range, normalisation constants, resize
   method, any padding. Quote the source lines. Do not infer from convention.
4. **Downsampling factor**: does the architecture require /8, /16, /32 or /64? Find this in
   the feature pyramid or encoder stride, not in the README.
5. **Max disparity parameter**: where it is set, its default, whether it is a build-time
   constant or runtime argument.
6. **Iteration count**: if the model refines iteratively, where the count is set and its
   default.
7. **Control flow**: any data-dependent loops or conditionals in the forward pass that will
   not trace cleanly.
8. **Checkpoint**: confirm it loads, report its SHA-256 and the number of parameters.
9. **[v2] Custom compiled ops**: any CUDA/C++ extensions the model needs at inference
   (deformable convolution, correlation samplers, custom samplers). These must be rebuilt
   for the Jetson for the native run, and are also export risks. List them.

**Stop here and report. Wait for my confirmation before writing any export code.** If
anything in the spec cannot be honoured for this model, say so now rather than working
around it silently.

## Step 2 — locked parameters (do not deviate without asking)

| Parameter | Value |
|---|---|
| Tiers | FULL 1280×704, HALF 640×352 |
| Batch | 1, **static shape**, no dynamic axes |
| Opset | 17 |
| Export precision | fp32 |
| Max disparity | **128** at FULL, **64** at HALF |
| Iterations (iterative models only) | export separately at 4, 8, 16 |
| Input names | `left`, `right` (or `input` for monocular) |
| Output name | `disparity` (or `depth`) |

If the architecture requires /64 and cannot take 640 × 352, pad to 640 × 384 by reflection,
crop the output back, and record the exception in the manifest. **Never change the aspect
ratio.** The evaluation frame is a pure crop plus uniform scale, and the whole
ground-truth mapping depends on that.

## Step 3 — preprocessing module

Write `preprocess.py` implementing exactly what you found in Step 1, and emit the
preprocessing manifest block from SPEC §4. This module is the single source of truth and
will be reused verbatim by the Jetson runtime harness, so it must not depend on the repo's
internals — pure numpy plus PIL or cv2, no model imports.

Include the 1280 × 720 → 1280 × 704 evaluation-frame crop (8 rows off the top, 8 off the
bottom) as the first operation, before any model-specific preprocessing.

## Step 4 — PyTorch baseline [v2 — revised]

`baseline_pytorch.py` has two modes. **It must run unchanged on this build machine and later
on the Jetson**, in a different environment, as the native-PyTorch test. So:

- **Device-portable.** `--device {auto,cuda,cpu}`, default `auto`. No hardcoded `.cuda()`, no
  build-machine paths, and **no imports of `onnx`, `onnxruntime`, `onnxsim` or anything
  export-only** — those will not exist in the Jetson container.
- **Configured by CLI**, matching Step 2: `--tier {full,half}`, `--max-disp`, `--iters`
  (iterative models only), `--checkpoint`, `--left`, `--right`, `--out`.
- All input handling goes through `preprocess.py`.

### Mode 1 — reference (default)

- fp32, with **TF32 disabled** and deterministic kernels:
  ```python
  torch.backends.cuda.matmul.allow_tf32 = False
  torch.backends.cudnn.allow_tf32 = False
  torch.backends.cudnn.benchmark = False
  ```
  PyTorch enables TF32 for convolutions by default on Ampere-or-newer GPUs. That lowers
  precision enough to fail G1 spuriously against ONNX Runtime's CPU fp32, and you would
  misdiagnose a correct export as broken.
- Save raw disparity as `.npy` plus a colour-mapped PNG.
- Print min, max, mean, and percentage of non-finite values. **Sanity-check the disparity
  range against the expected physical range**: at FULL tier, roughly 6 px (10 m) to 125 px
  (0.5 m); at HALF, half that. If the output is wildly outside that, preprocessing or the
  output convention is wrong. Report it and stop.

### Mode 2 — benchmark (`--benchmark`)

- **Never overwrites reference artefacts.** Writes only `timing_*.json`.
- Measures boundary **B2**: preprocessed tensor already on the device → disparity on the
  device. Exclude file I/O, preprocessing, and host transfers.
- Warm-up runs (default 20), then timed runs (default 100), synchronising the device before
  and after each timed run.
- Reports median, p95 and p99 latency in ms; on CUDA, peak **allocated and reserved**
  memory (`torch.cuda.max_memory_allocated`, `torch.cuda.max_memory_reserved`), since the
  gap between them is PyTorch's allocator overhead; and the environment: hostname, device
  name, torch and CUDA versions, precision, TF32 state.
- `--fp16` runs under `torch.autocast(dtype=torch.float16)`. **Benchmark mode only** —
  never used for reference outputs.
- Output file: `timing_<tier>[_it<N>]_<fp32|fp16>_<hostname>.json`.

On this machine, run benchmark mode **once per tier as a smoke test** to prove the mode works
before it ever reaches the Jetson. Save it as `timing_smoketest_<host>.json` and say plainly
in the README that these are build-machine numbers and are not to be reported.

## Step 5 — export

`export_onnx.py`: `torch.onnx.export` with static shapes, then simplify with
`onnx-simplifier`, then `polygraphy surgeon sanitize` if available.

After export, verify and report:

- Graph contains **no** `If`, `Loop`, `NonZero`, or `NonMaxSuppression` nodes. These either
  fail in TRT or force fallback that destroys performance. If present, stop and tell me.
- Input and output names and shapes match Step 2.
- File size and SHA-256.
- Node count before and after simplification.

## Step 6 — GATE G1 [v2 — revised]

`check_g1.py`:

1. Preprocess the stereo pair once and **save the exact tensors** fed to both runtimes as
   `inputs_<tier>.npz` (keys `left`, `right`, dtype and shape exactly as fed).
2. Run those saved tensors through ONNX Runtime (CPU execution provider, fp32) and **save
   the output** as `ort_<tier>.npy`.
3. Compare against the PyTorch reference from Step 4.

**Pass criterion: max absolute disparity difference < 1e-3 px.**

Report max, mean, median and p99 absolute difference; the location of the worst pixel; and
save a difference visualisation. Write `g1_report.json`, including the SHA-256 of each
`inputs_*.npz` and `ort_*.npy`.

**Why the saved tensors matter:** on the Jetson, gate G2 feeds `inputs_<tier>.npz` straight
into the TensorRT engine and compares against `ort_<tier>.npy`. Preprocessing is removed
from the comparison entirely, so a G2 failure can only be an engine problem, and ONNX
Runtime never needs installing on the Jetson.

**If G1 fails, the export is wrong. Stop and diagnose — do not proceed.** The usual causes,
in order: TF32 left enabled in the reference run, a preprocessing mismatch between the two
paths, an unsupported op silently replaced during export, or the checkpoint not actually
being loaded.

## Step 7 — README and manifest

`export/README.md` states plainly: what was exported, every decision made and why, anything
in the spec that could not be honoured, the G1 result, any custom compiled ops, and the exact
next steps on the Jetson (SPEC §6), including:

- the transfer list: both `.onnx` files, `inputs_*.npz`, `ort_*.npy`, `baseline_*.npy`,
  `manifest.json`, `preprocess.py`, `baseline_pytorch.py`;
- the exact `baseline_pytorch.py --benchmark` commands for the native run, per tier and
  precision.

Fill `manifest.json` with every field from SPEC §10 that this stage can populate, plus
[v2] `torch_version`, `build_device`, `tf32_disabled`, `custom_ops`, and the SHA-256 of
every artefact. Leave the Jetson fields as `null`, not absent.

## Constraints

- **Do not modify the model's source code** except where unavoidable for tracing. If you
  must, keep the diff minimal, keep it in a patch file under `export/`, and explain each
  change in the README. Silent edits to the model are the fastest way to produce an export
  that is not the published model.
- Do not install `tensorrt` or `pycuda`.
- Do not change the locked parameters in Step 2.
- No hardcoded absolute paths; everything takes CLI arguments with sensible defaults.
- `run_all.sh` must work end to end from a clean checkout with only the checkpoint path and
  stereo pair path supplied. It runs: reference mode, export, G1, then the benchmark-mode
  smoke test.

## What I want back

1. The Step 1 investigation report, **before any code**.
2. After my go-ahead: the export, with the G1 numbers, the node-type check, and confirmation
   that `baseline_pytorch.py` runs in benchmark mode with no export-only imports.
3. A plain statement of anything that surprised you, anything you are unsure about, and
   anything in the spec that does not fit this model.

If this model cannot be exported to a static-shape ONNX graph without data-dependent control
flow, say so early. That is a legitimate finding about the model's deployability on
constrained hardware and belongs in the results, not something to be engineered around.
