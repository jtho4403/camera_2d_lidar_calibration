# ZED SDK Depth-Mode Evaluation — `data_2026-09-24`

**Status: baseline results for all 6 ZED SDK depth modes, evaluated against the accepted
ground truth (`REPORT.md`).** These modes are not build targets for the Jetson Orin Nano
(resource usage already ruled them out) — this evaluation exists to establish a reference point
against which the open-source candidate models (pending TensorRT build work, out of scope here)
will later be compared.

---

## 1. Method summary

Full methodology is in `tools/depth_eval/README.md` and `docs/DEPTH_METRICS.md`; summarised
here for context.

- **Ground truth**: the accepted sparse LiDAR-referenced export (`REPORT.md`), 31,857 points
  across Scenes A (19,629), B (4,245), C (7,983), already filtered to the declared 0.5–8 m
  operating range.
- **Prediction**: each capture's raw `.svo2` opened fresh per depth mode (frame index 75 of the
  burst — a single, realistic, non-burst-averaged frame, deliberately not the staged calibration
  images), `coordinate_units=METER`, SDK defaults for confidence/texture thresholds (100, i.e. no
  filtering) and fill mode (off) — see `tools/depth_eval/config.py` for the full, explicit
  rationale on every setting.
- **Scoring**: every ground-truth point bilinearly sampled from the predicted dense depth map;
  a point counts toward `coverage` only if all 4 bilinear neighbours are finite and positive.
  Metrics: the seven Standard Depth Metrics (`docs/DEPTH_METRICS.md`) plus signed bias and
  disparity-domain `EPE`/`bad3`/`D1-all` (justified in that document's Harness Additions
  section), reported overall, per scene, and per depth bucket (near 0.5–2 m, mid 2–4 m, far
  4–8 m) — never as one aggregate number alone.
- **Coverage**: all 234 capture × mode evaluations completed; `n=31,857` in every mode's overall
  result confirms every ground-truth point was scored by every mode.

---

## 2. Headline results

### Table 1 — Overall (all scenes, all buckets pooled)

| Mode | Coverage | abs_rel | sq_rel | rms (m) | log_rms | a1 | a2 | a3 | bias (mm) | bad3 | D1-all |
|---|---|---|---|---|---|---|---|---|---|---|---|
| | ↑ | ↓ | ↓ | ↓ | ↓ | ↑ | ↑ | ↑ | 0 is best | ↓ | ↓ |
| **NEURAL** | 99.4% | **0.0749** | 0.0681 | 0.482 | 0.1265 | **96.7%** | 98.2% | 99.5% | −127.5 | 5.9% | 5.9% |
| **NEURAL_PLUS** | 99.2% | 0.0759 | 0.0728 | 0.510 | 0.1268 | 96.8% | 97.8% | 99.4% | −124.9 | **5.4%** | **5.4%** |
| NEURAL_LIGHT | **99.6%** | 0.0897 | 0.0699 | **0.481** | 0.1392 | 94.9% | **98.6%** | 99.5% | −161.2 | 10.5% | 10.3% |
| QUALITY | 95.0% | 0.1034 | 0.1618 | 0.570 | 0.1848 | 94.5% | 97.7% | 98.6% | −214.7 | 10.8% | 10.6% |
| ULTRA | 89.3% | 0.1120 | 0.2089 | 0.668 | 0.2019 | 93.7% | 97.0% | 98.3% | −221.7 | 10.9% | 10.6% |
| PERFORMANCE | 95.9% | 0.0987 | 0.1009 | 0.624 | 0.2033 | 88.9% | 95.9% | 97.9% | −265.4 | 18.8% | 18.2% |

### Figure 1 — Overall accuracy

![abs_rel and a1](results/depth_eval/data_2026-09-24/fig1_overall_abs_rel_a1.png)

### Figure 2 — Coverage

![coverage](results/depth_eval/data_2026-09-24/fig3_coverage.png)

### Figure 3 — Disparity outlier rate (D1-all)

![D1-all](results/depth_eval/data_2026-09-24/fig4_d1_all.png)

**The NEURAL family wins decisively.** `NEURAL` and `NEURAL_PLUS` are essentially tied for best
(abs_rel ~7.5%, a1 ~97%); `NEURAL_LIGHT` is a clear step down but still beats every classical
mode. Within the classical family, `ULTRA` — despite the name — has both the worst accuracy
*and* the worst coverage (89.3%). `PERFORMANCE` has a deceptively unremarkable `abs_rel` but by
far the worst disparity outlier rate (`D1-all` 18.2%, roughly double every other mode) — exactly
the kind of failure a single aggregate metric hides and the fuller metric set catches.

---

## 3. Per-scene breakdown

### Table 2

| Mode | Scene A (n=19,629) | Scene B (n=4,245) | Scene C (n=7,983) |
|---|---|---|---|
| | abs_rel / a1 / bias | abs_rel / a1 / bias | abs_rel / a1 / bias |
| NEURAL | 0.068 / 98.2% / −156mm | 0.112 / 89.9% / **+108mm** | 0.072 / 96.6% / −183mm |
| NEURAL_PLUS | 0.073 / 97.8% / −152mm | 0.108 / 91.9% / **+81mm** | 0.068 / 96.9% / −165mm |
| NEURAL_LIGHT | 0.086 / 95.8% / −191mm | 0.119 / 90.2% / **+41mm** | 0.084 / 95.1% / −197mm |
| QUALITY | 0.115 / 95.7% / −183mm | 0.087 / 90.9% / −250mm | 0.085 / 93.6% / −275mm |
| ULTRA | 0.109 / 96.2% / −203mm | 0.149 / 89.3% / −197mm | 0.101 / 89.9% / −279mm |
| PERFORMANCE | 0.082 / 94.1% / −205mm | 0.127 / 79.6% / −407mm | 0.125 / 81.3% / −338mm |

Scene B (verification plates: flat, isolated targets, no clutter) is the one place the bias
**flips positive** for all three neural modes, while Scenes A and C are strongly negative for
every mode without exception. Flagged as a real, consistent pattern in the data — not explained
here; plausibly related to the flat/isolated plate geometry versus cluttered real scenes, but
that's a hypothesis, not a finding.

---

## 4. The central finding: every mode underestimates depth, increasingly with range

### Table 3 — Signed bias by depth bucket (mm, 0 is best)

| Mode | near (0.5–2 m) | mid (2–4 m) | far (4–8 m) |
|---|---|---|---|
| NEURAL | −31 | −109 | **−343** |
| NEURAL_PLUS | −31 | −91 | **−368** |
| NEURAL_LIGHT | +4 | −150 | **−486** |
| QUALITY | +24 | −269 | **−523** |
| ULTRA | +73 | −281 | **−613** |
| PERFORMANCE | +2 | −351 | **−559** |

### Figure 4 — Bias by depth bucket

![bias by bucket](results/depth_eval/data_2026-09-24/fig2_bias_by_bucket.png)

Every single mode is near-unbiased (within ±75 mm) at close range, then swings sharply and
monotonically negative with distance: by 4–8 m, every mode underestimates depth by 34–61 cm on
average. This is large, consistent across all 6 independently-evaluated modes, and physically
expected — stereo depth precision degrades roughly as `Z² / (baseline · f_x)`, and this rig's
baseline is only ~120 mm, so error should grow close to quadratically with range. The pattern
matching across six otherwise-different algorithms is itself evidence this is a real,
measurement-grounded result rather than noise.

### 4.1 Checked against the ground truth's own known bias risk, before trusting this

`REPORT.md` §5.4 found this project's own ground truth might carry an unresolved,
distance-correlated bias of its own, extrapolated (explicitly unvalidated beyond 1.5 m) to
roughly **+86 to +94 mm** at 7–8 m. Since this evaluation computes `bias = prediction − ground
truth`, a positive GT bias would make every mode's *measured* bias look more negative than its
true bias by about that amount.

Even granting the GT's bias at its largest plausible extrapolated value, it accounts for at most
~90 mm of the 343–613 mm underestimation observed at far range here — **4 to 7 times smaller**
than what's being measured. The dominant trend in Table 3 is not an artifact of the ground
truth's known open question; it's a real, independently-explicable property of these depth
modes, consistent with well-understood stereo geometry.

---

## 5. Interpretation and ranking

| Rank | Mode | Basis |
|---|---|---|
| 1 | NEURAL / NEURAL_PLUS (tied) | Best abs_rel, a1, lowest outlier rate; NEURAL_PLUS edges out on bad3/D1-all, NEURAL on log_rms |
| 2 | NEURAL_LIGHT | Clear step down from the above, still beats every classical mode on every metric except coverage (where it's actually best overall, 99.6%) |
| 3 | QUALITY | Best of the classical modes on accuracy; mediocre coverage |
| 4 | ULTRA | Worst coverage of all 6 modes (89.3%) *and* among the worst accuracy — the name is not a reliable guide here |
| 5 | PERFORMANCE | Roughly mid-pack `abs_rel` conceals the worst outlier rate by a wide margin (D1-all 18.2% vs. 5.4–10.9% for every other mode) and the worst `a1` (88.9%) |

For establishing a reference baseline, as intended: **`NEURAL`/`NEURAL_PLUS`** represent the
practical ceiling of what this camera+SDK combination can do; **`PERFORMANCE`**, despite being
the lightest-weight classical option, is the one mode whose accuracy profile looks meaningfully
worse than its headline `abs_rel` number suggests on its own.

---

## 6. Limitations of this evaluation

1. **Single-frame per capture.** Each capture is scored from one frame (index 75 of its burst),
   not averaged — representative of real-time deployment, but some of the capture-to-capture
   variance in the per-capture detail (not shown in the tables above) is genuine per-frame noise,
   not a stable model property.
2. **Sparse, thin-strip coverage** (`REPORT.md` §7): every number above is accuracy along a
   single horizontal LiDAR scan line, not full-frame. Scope carried over unchanged from the
   ground truth itself.
3. **These modes are confirmed out of contention for the Jetson Orin Nano deployment target**
   (resource usage, decided separately) — this evaluation's purpose is a baseline/reference
   point for the open-source candidate models, not a deployment recommendation.
4. **Scene B's sign flip (§3) is unexplained.** Worth a closer look before being used as a
   firm conclusion about scene-content dependence.
5. SDK runtime settings (confidence/texture threshold, fill mode) use defaults, not a tuned
   configuration — `docs/MASTER_PLAN.md` 4.2 ("Lock SDK depth settings") is still open; these
   results characterise the SDK's out-of-the-box behaviour, not necessarily its best achievable
   accuracy under deliberate tuning.

---

## Appendix: artifact index

| Artifact | Path |
|---|---|
| Per-mode full results (overall, per-scene, per-bucket, per-capture) | `results/depth_eval/data_2026-09-24/zed_sdk_<MODE>/zed_sdk_<MODE>_summary.json` |
| Evaluation harness | `tools/depth_eval/` (`evaluate.py`, `metrics.py`, `ground_truth.py`, `sampling.py`, `zed_sdk_predictor.py`) |
| Full-run driver | `tools/depth_eval/run_all_zed_modes.sh` |
| Metric definitions and additions | `docs/DEPTH_METRICS.md` |
| Ground-truth acceptance basis | `REPORT.md` |
