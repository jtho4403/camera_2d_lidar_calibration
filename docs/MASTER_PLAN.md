# Master Plan and Status Register — Embedded Stereo Depth Evaluation

**Revision 2.** Changes since revision 1: device audit findings (§3, WP0), capture confirmed unblocked by the ZED SDK mismatch (1.24), export prompt v2 (2.2, 3.1), benchmark-environment strategy (0.12–0.14).

Single source of truth for **what the study is, what is locked, and what is left**. The three
specification documents hold the detail; this document indexes them and tracks status.

| Document | Governs |
|---|---|
| `EXPERIMENT_DESIGN_v3.md` | Capture of the sparse LiDAR ground truth (Scenes A, B, C, S3) |
| `PLAN.md` | Projection of LiDAR returns into pixels; validation of the ground truth |
| `DEPTH_MODEL_BUILD_SPEC.md` | Export, engine build, parity gates, benchmarking rules |
| `ONNX_EXPORT_PROMPT_v2.md` | Per-model export procedure for Claude Code |

---

## 1. Objective

**Research question.** Which stereo depth estimation approach gives the best trade-off between
depth accuracy and computational cost on an NVIDIA Jetson Orin Nano, within a real-time budget
compatible with running as one component of a Dynamic SLAM perception pipeline?

**Deliverables.**

1. A hardware characterisation of every candidate configuration: latency, throughput, memory,
   power, energy per frame.
2. An accuracy characterisation of every deployable configuration against LiDAR-referenced
   sparse ground truth, with quantified ground-truth uncertainty.
3. A recommendation, justified by the joint trade-off, plus documented failure modes and
   future-work directions.

**Scope.** Stereo depth from a ZED 2i at HD720; indoor scenes; operating range 0.5–8 m;
Orin Nano 8 GB at a single locked power mode. Monocular models are out of scope unless added
deliberately (they need a different evaluation protocol, SPEC §1).

---

## 2. Study structure

Your six parts (a)–(f) are right, with three corrections and three additions.

### The shape of the study

```
          INSTRUMENTS                    ROWS (configurations)            SYNTHESIS
 ┌──────────────────────────┐    ┌───────────────────────────────┐
 │ I1 Ground-truth           │    │ P1 TensorRT engines           │    ┌──────────────────┐
 │    (calibration, proj.,   │───▶│ P2 Native PyTorch             │───▶│ Feasibility      │
 │    validation, S3 char.)  │    │ P3 ZED SDK depth modes        │    │ filters, then     │
 │ I2 Benchmark harness      │───▶│                               │    │ Pareto trade-off  │
 │ I3 Accuracy harness       │───▶│ each row measured on:         │    │ -> recommendation │
 └──────────────────────────┘    │   A1 hardware  A2 accuracy    │    └──────────────────┘
                                  └───────────────────────────────┘
```

### Corrections to the (a)–(f) framing

**1. Rows are configurations, not models.** Each model appears as several rows:
model × tier (FULL/HALF) × precision (fp32/fp16) × iteration count (iterative models only).
"How accurate is RAFT-Stereo" has no single answer; "how accurate is RAFT-Stereo, HALF, fp16,
8 iterations" does. Both axes are measured per row. See §4 for the count.

**2. Native PyTorch does not need its own accuracy evaluation.** Parity gates G1 and G2 establish
that PyTorch fp32, ONNX fp32 and TRT fp32 produce the same output to within 0.01 px. Running
the full accuracy evaluation natively would reproduce the TRT fp32 result. For P2, axis A2 is
satisfied by the parity gates. **Exception:** a model that cannot export has native as its only
path, and gets a full native accuracy evaluation.

**3. The two axes are measured independently but concluded jointly.** A model's accuracy
number is only meaningful paired with the latency of the *same row*. The recommendation comes
from the joint analysis, not from either axis alone.

### Additions

**4. The instruments are work packages in their own right.** The ground-truth instrument has
its own validation (Scene B, S3 characterisation) and its own uncertainty, which sets the floor
below which models cannot be distinguished. The benchmark harness and accuracy harness do not
yet exist.

**5. The deployment context sets the budget.** The real-time target and memory budget depend
on what else runs on the Jetson alongside depth estimation in the deployed DynoSAM pipeline.
This has been open since the build spec was written.

**6. A reference model.** SPEC §9 defines its role; no candidate has been chosen.

### Path × axis matrix

| Path | A1 Hardware | A2 Accuracy |
|---|---|---|
| **P1 TensorRT** | B1 (`trtexec`), B2, B3 via harness | Full evaluation, every row |
| **P2 Native PyTorch** | B2 via harness, like-for-like precision | **Parity gates only** (full evaluation only for export failures) |
| **P3 ZED SDK modes** | **B3 only** (depth is fused into `grab()`) | Full evaluation via SVO playback of Scene C |

Boundaries: **B1** engine (tensor in → tensor out on GPU), **B2** model (preprocessed tensor →
disparity), **B3** pipeline (`grab()` → depth map in the evaluation frame). Cross-path
comparisons happen at B3. The real-time target is a B3 criterion.

---

## 3. Locked parameters register

| Parameter | Value | Defined in | State |
|---|---|---|---|
| Capture device | ROSbot XL (Orin Nano + ZED 2i + RPLIDAR S3) | this doc | Locked |
| **Reference configuration** | **The ROSbot's stack** — deployment platform, and matches SPEC's JetPack 6.2.1 | this doc | **Proposed (0.12)** |
| Benchmark device | Benchtop, **only if** proven equivalent to the reference (0.13); otherwise ROSbot | this doc | **Open (0.13)** |
| L4T | ROSbot R36.4.4 (= JetPack 6.2.1); benchtop R36.5.2 | audit | **Mismatch** |
| Kernel | ROSbot Canonical `5.15.0-1022-nvidia-tegra-igx`; benchtop NVIDIA `5.15.199-tegra` | audit | **Mismatch** |
| TensorRT / CUDA | SPEC says 10.3.0; actual on either device not yet captured | audit | **Unknown — `hw_audit.sh`** |
| ZED SDK | ROSbot 5.3.0; benchtop 5.2.3 | audit | **Mismatch — resolve by upgrading benchtop to 5.3.0** |
| Swap | ROSbot 0 B; benchtop 3.7 GiB | audit | **Mismatch — disable for benchmarks (0.14)** |
| Power mode | ROSbot's mode, applied identically on both | SPEC §6 | **Open (0.10); benchtop is 25W; SPEC still says MAXN** |
| Camera | HD720, rectified, `fx ≈ 521.8`, `B = 120 mm` | SPEC header | Locked |
| Capture format | SVO2 lossless, 15 fps, 10 s bursts | EXP §4 | Locked |
| Live benchmark frame rate | 30 fps | this doc | Proposed |
| Evaluation frame | 1280 × 704 crop, `cy_eval = cy − 8` | SPEC §2 | Locked |
| Inference tiers | FULL 1280 × 704, HALF 640 × 352 | SPEC §2 | Locked |
| Operating range | `Z_min` 0.5 m, `Z_max` 8 m | EXP §6 | Locked |
| Max disparity | 128 FULL, 64 HALF | SPEC §3 | Locked |
| Iteration counts | {4, 8, 16} | SPEC §3 | Locked |
| ONNX | opset 17, static, batch 1, fp32 | SPEC §4 | Locked |
| Precisions | fp32 (reference), fp16 (deployment); INT8 deferred | SPEC §6, §8 | Locked |
| Parity gates | G1 < 1e-3 px, G2 < 1e-2 px, G3 measured | SPEC §5 | Locked |
| Measurement boundaries | B1 / B2 / B3 | this doc | **Proposed — not yet in SPEC** |
| Real-time target | ≥ 10 FPS at B3 | SPEC §0 | **Provisional (0.5)** |
| Memory budget | Derived from state ladder | SPEC §0 says 3 GB | **Provisional (0.6)** |
| Calibration board and poses | A3, 50 mm, 6 × 3; 24 poses, 0.6–1.5 m | EXP §2, §4 | Locked |
| Accuracy metrics | — | — | **Missing (6.1)** |
| ZED SDK depth settings | — | — | **Missing (4.2)** |
| Reference model | — | SPEC §9 (role only) | **Not chosen (6.3)** |

---

## 4. Configuration grid

| Path | Rows | Count |
|---|---|---|
| P1, 8 non-iterative models | 8 × 2 tiers × 2 precisions | 32 |
| P1, 2 iterative models | 2 × 2 tiers × 2 precisions × 3 iteration counts | 24 |
| **P1 total** | | **56 engines** |
| P2, full grid | same as P1 | 56 |
| P3 | available SDK depth modes | ≤ 6 |

The full grid is roughly 118 hardware rows and 62 accuracy rows. That is a real workload.
**Proposed pruning rule for P2** (decision 0.8): native fp32 at both tiers for non-iterative
models, and 8 iterations only for iterative models. That gives **20 native rows**, enough to
report the TensorRT speedup for every model without duplicating the grid.

---

## 5. Dependencies and critical path

```
CRITICAL PATH (accuracy axis)
  board checks + rig metrology -> Scene A recapture -> T2 solve -> Phase 2 validation
      -> Scene B analysis -> Scene C -> GT export -> accuracy harness over all rows

PARALLEL TRACK (hardware axis) — not blocked by anything above
  exports (x10) -> engines on benchmark device -> G2/G3
  harness build -> state ladder -> derived budget -> all-row benchmarks
  native container -> native runs

JOIN
  both axes complete -> joint trade-off analysis -> recommendation
```

The hardware axis can progress to completion without any ground-truth data. **Nothing on the
parallel track should wait for the capture session.**

---

## 6. Status register

**Legend.** `COMPLETED` · `IN PROGRESS` · `PENDING`, qualified as **NOT-BLOCKED**,
**BLOCKED (by …)**, **NEW (since …)**, **NOT YET ADDRESSED (why)**, **MISSING** (no plan or
code exists yet) · `SUPERSEDED` · `UNCONFIRMED` (status not reported to me).

### WP0 — Scope and criteria

| ID | Task | Status |
|---|---|---|
| 0.1 | Research objective and scope (§1) | PENDING · NOT-BLOCKED — drafted here; confirm with supervisor |
| 0.2 | Candidate list triaged into families | COMPLETED |
| 0.3 | Candidate freeze manifests (commit, checkpoint SHA, training domain) | PENDING · NOT-BLOCKED — produced per model by the export prompt |
| 0.4 | Operating range 0.5–8 m | COMPLETED |
| 0.5 | Real-time target | PENDING · NOT YET ADDRESSED — provisional 10 FPS; depends on 0.7 |
| 0.6 | Memory budget | PENDING · BLOCKED by 5.4 — 3 GB provisional; **drop no model on it** |
| 0.7 | Co-resident workload in deployment (ROS 2, LiDAR driver, other DynoSAM front-end models) | PENDING · NOT YET ADDRESSED — open since the build spec; decides 0.5 and 0.6 |
| 0.8 | Configuration grid pruning rule | MISSING — proposal in §4 |
| 0.9 | Device audit: full software and hardware state of both devices | IN PROGRESS · BLOCKED by ROSbot access (tomorrow) — partial audit found L4T, kernel, ZED SDK and swap mismatches; run `hw_audit.sh` on both |
| 0.10 | Deployment power mode, matched on both devices | PENDING · BLOCKED by 0.9 — benchtop is 25W; ROSbot unknown |
| 0.11 | Accuracy floor | PENDING · BLOCKED by 1.13 |
| 0.12 | **Pinned benchmark environment**: one container image (TensorRT, CUDA, cuDNN, ZED SDK, Python stack) built on the reference L4T, run identically on both devices | NEW (device audit) · BLOCKED by 0.9 |
| 0.13 | **Cross-device equivalence test**: same container, same power mode, same engines on both devices; pass threshold declared in advance | NEW (device audit) · BLOCKED by 0.12 and one built engine |
| 0.14 | Swap policy: disabled on both devices for all benchmarks; permitted for engine builds, and recorded | NEW (device audit) · NOT-BLOCKED — decision proposed |

### WP1 — Ground-truth instrument

| ID | Task | Status |
|---|---|---|
| 1.1 | Projection design (SE(2) + `Δz` reduction) | COMPLETED |
| 1.2 | Projection core, filters, overlay (Phases 0–1) | COMPLETED |
| 1.3 | Horizontal-offset diagnostics (Phases 1.5–1.7) | SUPERSEDED by Scene B — old STL-19P data; Phase 1.7 outcome UNCONFIRMED |
| 1.4 | Correspondence dump in `cam_lidar_2d_icp.py` | COMPLETED (code) · never executed through the GUI — verify on first real solve |
| 1.5 | `zed_capture.py` | COMPLETED |
| 1.6 | Scene design v3 and pose table | COMPLETED |
| 1.7 | Boards printed and mounted (plain + ChArUco) | COMPLETED |
| 1.8 | Board checks: measured pitch, flatness, quiet zone | PENDING · NOT-BLOCKED · UNCONFIRMED |
| 1.9 | Board config updated to 6 × 3 and measured pitch in `cam_lidar_2d_icp.py` and `check_corners.py` | PENDING · NOT-BLOCKED |
| 1.10 | Rig metrology into `rig_<id>.json` (`Δz` chain, inclinometer, scan height) | PENDING · NOT-BLOCKED — scan height bounded at 174.8 mm |
| 1.11 | LaserScan handedness verified on the S3 | PENDING · NOT-BLOCKED |
| 1.12 | **MCAP `/scan` ingestion with per-bearing burst aggregation** | **MISSING** · NOT-BLOCKED — the pipeline reads STL-19P `.pcd`; the S3 data is MCAP LaserScan |
| 1.13 | S3 characterisation, including the edge test that sets `τ` and `k` | PENDING · NOT-BLOCKED — separate session |
| 1.14 | Scene A recapture, both boards | PENDING · BLOCKED by 1.8, 1.9, 1.10 |
| 1.15 | **Burst-mean image extraction from SVO2 (Fix 1)** | **MISSING** · NEW (board redesign) · NOT-BLOCKED |
| 1.16 | `T2` solve, plain board | PENDING · BLOCKED by 1.12, 1.14, 1.15 |
| 1.17 | **`T2` solve, ChArUco board** | **MISSING** · NEW (supervisor meeting) — the pipeline detects plain chessboards only; ChArUco needs its own board-pose path feeding the same ICP |
| 1.18 | Phase 2: board-plane validation, bootstrap, hold-out residual | PENDING — code NOT-BLOCKED; results BLOCKED by 1.16 |
| 1.19 | Scene B plates built | PENDING · UNCONFIRMED |
| 1.20 | Scene B capture (B1–B5, B1 twice) | PENDING · BLOCKED by 1.19 |
| 1.21 | **Scene B `Δu` analysis tool** | **MISSING** · NOT-BLOCKED |
| 1.22 | Scene C capture | PENDING · NOT-BLOCKED for capture (same session, mount untouched) |
| 1.23 | GT export (Phase 4) | PENDING — code NOT-BLOCKED; results BLOCKED by 1.18, 1.21 |
| 1.24 | **Frame extraction under one fixed ZED SDK version**, with `K` taken from that same run | **MISSING** · NEW — **does not block capture**: SVO2 stores raw frames plus factory calibration, and rectification happens at extraction. Rule: extract every frame with a single SDK version ≥ the capture version (5.3.0) |

### WP2 — TensorRT path (P1)

| ID | Task | Status |
|---|---|---|
| 2.1 | `DEPTH_MODEL_BUILD_SPEC.md` | COMPLETED — amendments pending, §7 |
| 2.2 | `ONNX_EXPORT_PROMPT.md` | COMPLETED — **v2**: portable baseline, TF32 disabled for references, saved input tensors and ONNX Runtime outputs, wide-depth-range pair |
| 2.3 | Build-partner chat | COMPLETED |
| 2.4 | Early ad-hoc exports | SUPERSEDED — redo under the spec |
| 2.5 | ONNX export, 10 models | IN PROGRESS — first model underway |
| 2.6 | Engines, fp32 and fp16 at both tiers | PENDING · BLOCKED per model by 2.5, and by 0.10, 0.12 — **builds paused on the benchtop** |
| 2.7 | **G2/G3 runner** (TRT inference on the real pair, compare against transferred references) | **MISSING** · NOT-BLOCKED |
| 2.8 | MobileStereoNet build OOM | PENDING · NOT-BLOCKED — check variant, run natively at HALF/64, then builder settings |

### WP3 — Native PyTorch path (P2)

| ID | Task | Status |
|---|---|---|
| 3.1 | `baseline_pytorch.py` device-portable, with timing and peak memory | COMPLETED as a requirement (export prompt v2) — produced per model by each export |
| 3.2 | Jetson PyTorch container base (JetPack 6) | MISSING · NEW (supervisor meeting) |
| 3.3 | Native runs at B2, pruned grid | PENDING · BLOCKED by 3.1, 3.2, 0.8 |
| 3.4 | Native accuracy | NOT REQUIRED — parity gates suffice, except for export failures |

### WP4 — ZED SDK baseline (P3)

| ID | Task | Status |
|---|---|---|
| 4.1 | Enumerate depth modes available in the installed SDK | PENDING · NOT-BLOCKED |
| 4.2 | **Lock SDK depth settings** (confidence thresholds, fill mode, range 0.5–8 m) | **MISSING** |
| 4.3 | Pre-run neural-mode first-use optimisation | PENDING · NOT-BLOCKED |
| 4.4 | Hardware measurement at B3 (state ladder S4–S9) | PENDING · BLOCKED by 5.2 |
| 4.5 | Accuracy via SVO playback of Scene C | PENDING · BLOCKED by 1.23 |

### WP5 — Benchmark harness (I2)

| ID | Task | Status |
|---|---|---|
| 5.1 | Measurement protocol in SPEC (boundaries, state ladder, system protocol) | PENDING · NEW (supervisor meeting) |
| 5.2 | **Harness**: zed / trt / torch backends, live or SVO source, per-stage timing, `tegrastats` | **MISSING** · NOT-BLOCKED |
| 5.3 | State ladder S0–S3: capture-only overhead | PENDING · BLOCKED by 5.2 |
| 5.4 | Budget derivation, including the ROS + LiDAR term measured on the ROSbot | PENDING · BLOCKED by 5.3, 0.7 |
| 5.5 | Benchmarks for all rows | PENDING · BLOCKED by 5.2, 2.6, 3.3 |
| 5.6 | Deployment-gap spot check: 1–2 models on the ROSbot | PENDING · NEW (this session) · BLOCKED by 5.2 |

### WP6 — Accuracy evaluation (I3)

| ID | Task | Status |
|---|---|---|
| 6.1 | **Metric set and reporting protocol locked** | **MISSING** — `Depth_Evaluation_Metrics` notes exist in the project files but are not in the spec |
| 6.2 | **Accuracy harness**: predictions → evaluation frame → sample at GT pixels → filters → metrics | **MISSING** · NOT-BLOCKED — buildable against synthetic GT now |
| 6.3 | Reference model chosen | PENDING · NOT YET ADDRESSED — not needed until 6.4 |
| 6.4 | Inference over Scene C for all rows | PENDING · BLOCKED by 1.23, 2.6 |
| 6.5 | Training-domain annotation per model | PENDING — recorded in freeze manifests |

### WP7 — Synthesis

| ID | Task | Status |
|---|---|---|
| 7.1 | **Joint analysis method**: feasibility filters (memory, FPS), then Pareto front of accuracy against latency, memory and energy | **MISSING** |
| 7.2 | Recommendation for the DynoSAM context | PENDING · BLOCKED by 0.7, 5.5, 6.4 |
| 7.3 | Future-work log (INT8, custom kernels, failed exports) | PENDING |

---

## 7. Document change register

| Document | Change | Urgency |
|---|---|---|
| SPEC header | Replace the single version line with the reference configuration and the pinned container (0.12) | **Before the next engine build** — waits on 0.9 |
| SPEC §6 | Replace "MAXN for benchmarking" with the locked reference power mode; add swap-off; build inside the pinned container | **Before the next engine build** — waits on 0.10 |
| SPEC, new | Cross-device equivalence test and its pass threshold (0.13) | Before any benchmark on the benchtop is reported |
| ONNX prompt | v2 issued | **DONE** |
| SPEC §5 | G2 feeds the saved `inputs_*.npz` to the engine and compares against the saved `ort_*.npy` — no ONNX Runtime on the Jetson | Before the first G2 — the artefacts now exist from export v2 |
| SPEC §0 | Mark 3 GB provisional; add the budget equation; no model dropped on it | Before any pass/fail decision |
| SPEC §3 | Replace the confused `Z_min` sentence | Low |
| SPEC §9, new §12–14 | SDK modes as B3 baseline rows; measurement protocol; native screen | Before the harness build |
| SPEC, new section | Accuracy metrics and reporting | Before the accuracy harness |
| EXP v3 | ChArUco solve path; single-SDK frame extraction; MCAP ingestion | **Not urgent — capture-ready as is** |

---

## 8. Decisions required from Jack

1. **Run `hw_audit.sh` on both devices** (tomorrow). This closes 0.9 and supplies power mode, TensorRT, CUDA, cuDNN and container runtime facts.
2. **Reference configuration** — accept the ROSbot's stack as the reference (0.12).
3. **Co-resident workload** — what runs alongside depth estimation in the deployed system
   (ROS 2 and LiDAR driver certainly; which DynoSAM front-end components?).
4. **Native grid pruning** — accept the 20-row rule in §4, or run the full grid.
5. **Reference model** — can wait, but should be settled before Scene C inference.

---

## 9. Immediate next actions

**Before the next engine build:** settle decisions 1 and 2, then apply the two SPEC §6 changes.
**Before the current export finishes:** amend the export prompt's baseline script.

**Capture track:** board checks (1.8), board config (1.9), rig metrology (1.10), handedness
(1.11) — then Scene A, B and C in one session.

**Build track, in parallel:** keep exporting; write the G2/G3 runner (2.7); start the harness
(5.2) with the ZED backend first, since the state ladder unblocks the budget.

**Code that can be written now, against no new data:** MCAP ingestion (1.12), burst-mean
extraction (1.15), Scene B analysis (1.21), accuracy harness on synthetic GT (6.2).
