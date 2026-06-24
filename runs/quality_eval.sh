#!/usr/bin/env bash
set -euo pipefail

DEVICE="${DEVICE:-cuda}"
MODEL_TAG="${MODEL_TAG:?Set MODEL_TAG}"
STEP="${STEP:?Set STEP}"
OUT="${OUT:-repro_results/quality}"
EVAL_TOKENS="${EVAL_TOKENS:-2097152}"
CALIBRATION_FILE="${CALIBRATION_FILE:-paper/calibration.jsonl}"

mkdir -p "$OUT"

# Downloads only the pinned validation shard if it is not already present.
python -m nanochat.dataset -n 0

run_quality() {
  local precision="$1"
  local dtype="$2"
  local calibration_samples="$3"
  local suffix="$precision"
  local calibration_args=()
  if [[ "$precision" == "static-int8" ]]; then
    suffix="${precision}_cal${calibration_samples}"
    calibration_args+=(--calibration-file "$CALIBRATION_FILE" --calibration-samples "$calibration_samples")
  fi
  NANOCHAT_DTYPE="$dtype" python -m scripts.precision_eval \
    --precision "$precision" --model-tag "$MODEL_TAG" --step "$STEP" \
    --device-type "$DEVICE" --eval-tokens "$EVAL_TOKENS" \
    --output "$OUT/quality_${suffix}.json" "${calibration_args[@]}"
}

run_quality fp32 float32 32
run_quality fp16 float16 32
run_quality bf16 bfloat16 32
run_quality dynamic-int8 float32 32
run_quality static-int8 float32 1
run_quality static-int8 float32 8
run_quality static-int8 float32 32

python -m scripts.quality_report "$OUT"
