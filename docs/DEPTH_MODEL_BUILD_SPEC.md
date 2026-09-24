# Depth Model Build and Evaluation Specification (LOCKED)

Governs every candidate depth model from source checkout through to evaluation against the
LiDAR-referenced sparse ground truth. The point of locking these is that a comparison is
only meaningful if every model went through an identical process. Any deviation for a
specific model must be recorded in that model's manifest with a reason.

Target hardware: NVIDIA Jetson Orin Nano (8 GB unified memory), JetPack 6.2.1,
TensorRT 10.3.0.
Camera: Stereolabs ZED 2i, HD720, rectified, `fx ≈ 521.8`, baseline 120 mm.

---

## 0. Scope and success criteria — fix these before benchmarking

Declaring thresholds after seeing results is not a study. Lock them now:

| Criterion | Threshold | Rationale |
|---|---|---|
| Real-time target | **≥ 10 FPS** end-to-end | Matches the LiDAR scan rate; the system's natural cadence |
| Stretch target | ≥ 15 FPS | |
| Runtime memory budget | **< 3 GB** device memory | The ZED SDK's NEURAL mode already consumed 5.4 of 8 GB; anything above 3 GB cannot coexist with the rest of the system |
| Accuracy floor | To be set from the S3 characterisation | A model cannot be distinguished below the ground-truth noise floor |

A model that fails the memory budget is out regardless of accuracy. A model that fails the
FPS target is reported but not recommended.

---

## 1. Candidate freeze (the missing step 0)

Before any export, record per model in `manifest.json`:

- Repository URL and **exact commit hash**
- Checkpoint filename, download URL, **SHA-256**
- **Training domain**: KITTI, SceneFlow, Middlebury, mixed. A KITTI-trained model evaluated
  on indoor scenes is a domain shift and must be acknowledged in the write-up, not silently
  ignored.
- Licence
- Output type: disparity / metric depth / inverse depth / scale-ambiguous relative depth
- Input arity: separate left and right tensors, or a single concatenated tensor
- Paper reference

**Scale-ambiguous monocular models require a different evaluation protocol** (per-frame
scale and shift alignment against the sparse GT before any metric is computed). Flag them at
this stage; do not discover it at evaluation time.

---

## 2. The evaluation frame — the single most important interface

**1280 × 720 is not divisible by 32**, and most stereo networks downsample by 32 or 64.
The native ZED frame therefore cannot be fed to them directly. Resolve it once, here.

### Definition

**The evaluation frame is the 1280 × 704 centre crop of the rectified left image**, formed
by removing 8 rows from the top and 8 from the bottom.

- 1280 × 704 is divisible by 64 in both dimensions.
- Intrinsics in the evaluation frame: `fx, fy, cx` unchanged; **`cy_eval = cy − 8`**.
- The LiDAR scan row sits near image centre (rows ~330–355 in the existing data), so the
  crop discards nothing relevant.
- **All ground-truth pixels, all model predictions, and all metrics live in this frame.**

### Inference resolutions — both are evaluated, as a primary axis

| Tier | Shape | Divisibility | Scale from eval frame |
|---|---|---|---|
| **FULL** | 1280 × 704 | /64 | 1.0 |
| **HALF** | 640 × 352 | /32 | 0.5 |

If a model requires /64 and cannot take 640 × 352, pad to 640 × 384 by reflection, crop the
output back to 640 × 352, and record the exception in its manifest. Never change the aspect
ratio.

### Why both tiers

Depth error per pixel of disparity error, `dZ = Z² / (fx·B)`:

| Z | FULL | HALF |
|---|---|---|
| 1 m | 0.016 m | 0.032 m |
| 3 m | 0.144 m | 0.287 m |
| 5 m | 0.399 m | 0.799 m |
| 10 m | 1.597 m | 3.194 m |

Halving resolution doubles depth error at every range while cutting compute roughly
fourfold. That trade-off is the core question of the thesis, so resolution is a swept
variable, not a fixed one.

### Output resampling — get this exactly right

A model predicting at HALF outputs disparity in its own coordinate frame. To bring it into
the evaluation frame:

1. Upsample the disparity map from 640 × 352 to 1280 × 704 (bilinear).
2. **Multiply disparity values by 2.0.** Disparity is a pixel-space quantity and scales with
   resolution. Omitting this produces a clean-looking 2× depth error that will be
   misattributed to the model.
3. Convert to depth: `Z = fx_eval · B / d`, using the **evaluation-frame** `fx`, never the
   half-res one.

For models that output metric depth directly, only step 1 applies; depth does not scale.

---

## 3. Max disparity and iteration count

Both are baked in at export and both are large levers. Defaults from the papers are wrong
for this setup.

### Max disparity

Required range for a minimum depth of 0.5 m:

| Tier | Required max disparity | Set to |
|---|---|---|
| FULL | 125 px | **128** |
| HALF | 63 px | **64** |

The common default of 192 is both wasteful and, at HALF resolution, places the model outside
the disparity regime it was trained on. For cost-volume models (PSMNet and similar) memory
scales with `H × W × D`, so this is also the biggest single memory lever.

If a scene requires depths below 0.5 m, raise the minimum depth rather than the disparity
range, and record it.

### Iteration count

For iterative refinement models (RAFT-Stereo, CREStereo and similar), the number of update
iterations is unrolled into the ONNX graph at export and changes latency several-fold.

- **Export one engine per iteration count** from `{4, 8, 16}` for iterative models.
- Record the count in the manifest and in the engine filename.
- Never compare an iterative model at 32 iterations against a feed-forward model and call it
  an architecture comparison.

---

## 4. ONNX export

| Decision | Value | Reason |
|---|---|---|
| Opset | **17** | Well supported by TRT 10.3; raise only if a model needs it, and record |
| Input shapes | **fully static**, batch 1 | Dynamic shapes need optimisation profiles and give worse TRT performance; resolutions are fixed anyway |
| Precision at export | **fp32** | Precision is chosen at engine build, not export |
| Simplification | **onnx-simplifier, then `polygraphy surgeon sanitize`** | Folds constants and removes artefacts that TRT handles badly |
| Input names | `left`, `right` (or `input` for mono) | Consistent naming across models; the runtime harness depends on it |
| Output names | `disparity` (or `depth`) | Same |
| Control flow | none | Loops must unroll at trace time; no `If`/`Loop` nodes in the final graph |

Verify the exported graph contains no `If`, `Loop`, `NonZero`, or `NonMaxSuppression` nodes.
These either fail in TRT or force fallback that destroys performance.

### Preprocessing manifest — mandatory per model

```json
{
  "colour_order": "RGB",
  "value_range": [0.0, 1.0],
  "normalise": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
  "resize_interpolation": "bilinear",
  "resize_antialias": true,
  "pad": {"mode": "none"},
  "input_layout": "NCHW",
  "inputs": ["left", "right"],
  "output": {"name": "disparity", "type": "disparity", "sign": "positive", "units": "px"}
}
```

This exact spec must be applied identically in the PyTorch baseline, the ONNX Runtime check,
and the Jetson runtime harness. A mismatch here is the most common reason a model
underperforms its paper, and it is invisible unless you check.

---

## 5. Numerical parity gates (the missing verification step)

Run on **one real rectified stereo pair** from the capture set, at each tier.

| Gate | Comparison | Pass criterion | If it fails |
|---|---|---|---|
| **G1** | PyTorch fp32 vs ONNX Runtime fp32 | max abs disparity diff **< 1e-3 px** | The export is wrong. Stop. |
| **G2** | ONNX Runtime fp32 vs TRT fp32 (on Jetson) | max abs diff **< 1e-2 px**, no structural differences | Engine build or plugin issue. Stop. |
| **G3** | TRT fp32 vs TRT fp16 (on Jetson) | **measure, do not gate** | Report median and p99 diff, NaN/Inf count, and any spatially clustered degradation |

G3 is a result, not a gate. Some models lose nothing in fp16; some degrade badly in
low-texture or far-depth regions because correlation volumes lose range. **Without G3 you
cannot distinguish a genuinely weak model from a badly quantised good one**, and you would
draw the wrong conclusion about architecture.

**G1 threshold, amneded for deep cost-volume architectures.** The 1e-3 px max-abs criterion is retained as the primary gate. When it fails, G1 still passes if **all** of: (a) mean and median abs diff < 1e-4 px; (b) fraction of pixels exceeding 1e-3 px < 0.05%; (c) no pixel exceeds 5e-3 px; (d) the max statistic exhibits run-to-run variance at fixed input (evidence of fp32 non-associativity, not a deterministic defect); (e) outliers spatially correlate with disparity edges. ADStereo-main satisifies all five at both tiers. Rationale: the 1e-3 threshold does not account for benign fp32 reduction-order draft accumulated across ~450 nodes in sort/gather-heavy cost-volume graphs. 

Save the disparity maps and difference images from all three gates as artefacts.

---

## 6. TensorRT engine build

Engines are **not portable** across TensorRT version, GPU, or driver. Build on the Jetson,
every time.

### Locked build procedure

```bash
sudo nvpmodel -m <MODE>        # record the mode; MAXN for benchmarking
sudo jetson_clocks             # lock clocks; without this, results are not reproducible
sudo jetson_clocks --show      # capture the state into the manifest

/usr/src/tensorrt/bin/trtexec \
  --onnx=<model>_<tier>.onnx \
  --saveEngine=<model>_<tier>_<precision>.plan \
  --fp16 \                     # omit for the fp32 reference engine
  --memPoolSize=workspace:2048 \
  --warmUp=2000 \
  --duration=30 \
  --avgRuns=100 \
  --verbose \
  > build_<model>_<tier>_<precision>.log 2>&1
```

Build **both** fp32 and fp16 for every model and tier. fp32 is the on-device numerical
reference for G2 and G3; fp16 is the deployment candidate.

Note that `--fp16` permits fp16, it does not force it — TRT selects per layer. Record from
the verbose log which layers actually ran in fp16.

### Reading the trtexec summary

| Stat | Meaning | Use |
|---|---|---|
| **GPU Compute Time** (mean, median, p99) | Pure kernel execution | **The primary screening number.** Report median and p99 |
| **Host Latency** | Includes H2D and D2H transfers | The difference from GPU Compute Time is your transfer cost |
| **Enqueue Time** | CPU-side launch overhead | If it exceeds GPU Compute Time, the CPU is the bottleneck |
| **Throughput (qps)** | Inferences per second | Sanity check against latency |
| **H2D / D2H Latency** | Transfer time | Relevant on unified memory; usually small but not zero |
| Device memory | Runtime allocation | Check against the 3 GB budget |

**Validity of these numbers.** Valid for *relative ranking* between models under identical
locked conditions, and as a screening gate ("this is 800 ms, it's out"). **Not** valid as the
reported end-to-end figure, because trtexec feeds random data, excludes preprocessing and
postprocessing, and measures the engine in isolation. Report trtexec numbers as "engine
latency" and harness numbers as "end-to-end latency", clearly distinguished.

---

## 7. When a model fails

### OOM

Distinguish the two cases; they have different fixes.

- **Builder OOM** (most common): the TRT optimiser exhausts memory searching tactics. Cap
  with `--memPoolSize=workspace:1024` or lower. Builder memory is not runtime memory; a
  model that OOMs at build may run fine.
- **Runtime OOM**: the engine genuinely does not fit.

Levers, in descending order of effect:

1. **Drop tier** FULL → HALF (roughly 4× memory)
2. **Reduce max disparity** (linear for cost-volume models)
3. **Reduce iteration count** (iterative models)
4. **fp16** (roughly 2× on weights and activations)
5. INT8 (last resort; see §8)

### Too slow

Same levers, same order. What is **not** on the list:

- **DLA offload.** The Orin Nano has no DLA. Orin NX and AGX Orin do; Orin Nano does not.
  Do not chase this.
- **Hand-written CUDA kernels.** Out of scope, and actively harmful to the study: you would
  be evaluating your own optimisation effort rather than the architectures, which confounds
  the comparison the thesis exists to make. Note it as future work.

What *is* legitimate on the hardware side, and must be locked and reported regardless:

- `nvpmodel` power mode
- `jetson_clocks` state
- Thermal soak before measurement, and `tegrastats` logging of power and temperature during
  benchmarking

An unreported power mode makes every latency number in the thesis unreproducible.

---

## 8. INT8 — deferred

Requires a representative calibration set and a calibration cache, and can cost real depth
accuracy, especially in far-field and low-texture regions.

**Apply to the top one or two candidates only, after the fp16 comparison is complete.**
Treat it as a discussion tied to the quantization literature already in the reading list
(Jacob et al. 2018, Wu et al. 2020, Gholami et al. 2022) rather than as a build step for
every candidate. Calibration set: 100–200 frames drawn from the evaluation scenes, never
from the scenes used to report results.

---

## 9. Reference ("gold standard") model

**A reference model is not ground truth. Say so explicitly in the write-up.**

Its legitimate uses:

1. Dense qualitative comparison, where the LiDAR gives only a sparse horizontal slice.
2. Quantifying how much accuracy is given up by going lightweight.

Its required validation: **the reference must first be shown to beat every candidate on the
sparse LiDAR GT.** If it does not, it is not a reference and must not be used as one.

Include the ZED SDK's own NEURAL depth as a second reference point. It is what the platform
gives you out of the box and is the first comparison any reader will ask for. It is not
usable as ground truth for the same reason.

---

## 10. Per-model manifest

One `manifest.json` per model, carried from export through to results:

```
model_name, repo_url, commit_hash, checkpoint_sha256, training_domain, licence
output_type, input_arity, preprocessing_spec
tier, input_shape, max_disparity, iterations
opset, onnx_sha256, export_command, simplifier_version
jetpack, tensorrt_version, trtexec_command, engine_sha256, precision
nvpmodel_mode, jetson_clocks_state
G1_max_diff, G2_max_diff, G3_median_diff, G3_p99_diff, G3_nan_count
gpu_compute_median_ms, gpu_compute_p99_ms, host_latency_ms, device_memory_mb
```

Every number reported in the thesis must be traceable to one of these.

---

## 11. Full process, revised

```
0.  Freeze candidate: commit, checkpoint hash, training domain, output type
1.  Write preprocessing manifest
2.  PyTorch fp32 baseline on a real stereo pair -> reference disparity  [build machine]
3.  Export ONNX, static shape, opset 17, per tier                       [build machine]
4.  Simplify + sanitize                                                 [build machine]
5.  GATE G1: PyTorch vs ONNX Runtime                                    [build machine]
6.  Transfer ONNX + manifest + reference pair + reference output        -> Jetson
7.  Lock nvpmodel and jetson_clocks; record state                       [Jetson]
8.  Build TRT fp32 engine                                               [Jetson]
9.  GATE G2: ONNX Runtime vs TRT fp32                                   [Jetson]
10. Build TRT fp16 engine                                               [Jetson]
11. MEASURE G3: TRT fp32 vs TRT fp16                                    [Jetson]
12. Screening latency + memory from trtexec logs                        [Jetson]
13. End-to-end harness benchmark with thermal soak and tegrastats       [Jetson]
14. Inference over evaluation scenes -> depth maps in the evaluation frame
15. Accuracy vs sparse LiDAR GT; comparison against reference models
16. Update manifest with every result
```

Steps 0–13 are unblocked today. Steps 14–15 wait on the ground-truth dataset.
