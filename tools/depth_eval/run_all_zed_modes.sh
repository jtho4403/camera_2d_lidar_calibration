#!/bin/bash
# Run the full ZED SDK depth-mode evaluation: all 6 modes x Scenes A/B/C, data_2026-09-24.
# Sequential, not parallel -- every mode needs exclusive GPU/sl.Camera() access.
set -e
cd "$(dirname "$0")/../.."
source .venv/bin/activate

SESSION=data/data_2026-09-24
BASELINE_M=0.1198631078004837
OUT_ROOT=results/depth_eval/data_2026-09-24
LOG_DIR=/tmp/claude-1000/-home-andre-camera-2d-lidar-calibration/58a28a40-32f4-4364-b137-2effb1c79dec/scratchpad/depth_eval_full_run
mkdir -p "$LOG_DIR"

CLASSICAL_MODES="QUALITY ULTRA PERFORMANCE"
NEURAL_MODES="NEURAL NEURAL_PLUS NEURAL_LIGHT"
NEURAL_LD_PATH=/usr/local/cuda-13.3/targets/x86_64-linux/lib

for MODE in $CLASSICAL_MODES; do
  echo "=== $MODE (classical) ==="
  python tools/depth_eval/evaluate.py \
    --session "$SESSION" --scenes scene_A scene_B scene_C \
    --backend zed_sdk --depth-mode "$MODE" --baseline-m "$BASELINE_M" \
    --out-dir "$OUT_ROOT/zed_sdk_$MODE" \
    > "$LOG_DIR/$MODE.log" 2>&1
  tail -5 "$LOG_DIR/$MODE.log"
done

for MODE in $NEURAL_MODES; do
  echo "=== $MODE (neural, TensorRT) ==="
  LD_LIBRARY_PATH="$NEURAL_LD_PATH" python tools/depth_eval/evaluate.py \
    --session "$SESSION" --scenes scene_A scene_B scene_C \
    --backend zed_sdk --depth-mode "$MODE" --baseline-m "$BASELINE_M" \
    --out-dir "$OUT_ROOT/zed_sdk_$MODE" \
    > "$LOG_DIR/$MODE.log" 2>&1
  tr '\r' '\n' < "$LOG_DIR/$MODE.log" | tail -5
done

echo "ALL MODES COMPLETE"
