#!/usr/bin/env bash
set -euo pipefail

# Conference experiment: 20 prompts * 10 replicas * 5 fresh processes =
# exactly 1,000 generations for every precision/batch condition.
DEVICE="${DEVICE:-cuda}"
MODEL_TAG="${MODEL_TAG:?Set MODEL_TAG to the exact checkpoint tag used in the paper}"
STEP="${STEP:?Set STEP to the exact checkpoint step used in the paper}"
TRIALS="${TRIALS:-5}"
REPLICAS_PER_PROMPT="${REPLICAS_PER_PROMPT:-10}"
MAX_TOKENS="${MAX_TOKENS:-32}"
DECODE="${DECODE:-greedy}"
PROMPTS="${PROMPTS:-paper/prompts.jsonl}"
CALIBRATION_FILE="${CALIBRATION_FILE:-paper/calibration.jsonl}"
CALIBRATION_SAMPLES="${CALIBRATION_SAMPLES:-32}"
OUT="${OUT:-repro_results/conference_${DECODE}}"
REFERENCE="$OUT/reference.pt"
BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
STEP_PADDED="$(printf '%06d' "$STEP")"
MODEL_FILE="$BASE_DIR/base_checkpoints/$MODEL_TAG/model_${STEP_PADDED}.pt"
META_FILE="$BASE_DIR/base_checkpoints/$MODEL_TAG/meta_${STEP_PADDED}.json"

if [[ "$TRIALS" -ne 5 || "$REPLICAS_PER_PROMPT" -ne 10 ]]; then
  echo "Conference protocol requires TRIALS=5 and REPLICAS_PER_PROMPT=10 (1,000 generations/condition)." >&2
  exit 2
fi

mkdir -p "$OUT"
if [[ ! -f "$MODEL_FILE" || ! -f "$META_FILE" ]]; then
  echo "Missing pinned checkpoint: $MODEL_FILE or $META_FILE" >&2
  exit 2
fi
cp "$PROMPTS" "$OUT/prompts.jsonl"
cp "$CALIBRATION_FILE" "$OUT/calibration.jsonl"
git rev-parse HEAD > "$OUT/git_commit.txt"
git status --porcelain > "$OUT/git_status.txt"
nvidia-smi -q > "$OUT/nvidia_smi.txt"
uv pip freeze > "$OUT/pip_freeze.txt"
sha256sum "$MODEL_FILE" "$META_FILE" "$BASE_DIR/tokenizer/tokenizer.pkl" > "$OUT/artifact_sha256.txt"

# method:precision:import-time dtype:deterministic flag
conditions=(
  "fp32:fp32:float32:0"
  "fp32-det:fp32:float32:1"
  "fp16:fp16:float16:0"
  "fp16-det:fp16:float16:1"
  "bf16:bf16:bfloat16:0"
  "static-int8:static-int8:float32:0"
  "dynamic-int8:dynamic-int8:float32:0"
)

for condition in "${conditions[@]}"; do
  IFS=: read -r method precision dtype deterministic <<< "$condition"
  for batch_size in 1 8 64; do
    for trial in $(seq 0 $((TRIALS - 1))); do
      reference_args=()
      deterministic_args=()
      calibration_args=()
      env_args=("NANOCHAT_DTYPE=$dtype")
      if [[ "$deterministic" == 1 ]]; then
        deterministic_args+=(--deterministic)
        env_args+=("CUBLAS_WORKSPACE_CONFIG=:4096:8")
      fi
      if [[ "$precision" == static-int8 ]]; then
        calibration_args+=(--calibration-file "$CALIBRATION_FILE" --calibration-samples "$CALIBRATION_SAMPLES")
      fi
      if [[ "$method" == fp32 && "$batch_size" == 1 && "$trial" == 0 ]]; then
        reference_args+=(--write-reference)
      fi
      env "${env_args[@]}" python -m scripts.repro_experiment \
        --precision "$precision" \
        --batch-size "$batch_size" \
        --trial "$trial" \
        --replicas-per-prompt "$REPLICAS_PER_PROMPT" \
        --max-tokens "$MAX_TOKENS" \
        --decode "$DECODE" \
        --device-type "$DEVICE" \
        --prompts "$PROMPTS" \
        --model-tag "$MODEL_TAG" \
        --step "$STEP" \
        --reference-artifact "$REFERENCE" \
        --output "$OUT/${method}_b${batch_size}_trial${trial}.json" \
        "${deterministic_args[@]}" \
        "${calibration_args[@]}" \
        "${reference_args[@]}"
    done
  done
done

python -m scripts.repro_report "$OUT"
