#!/usr/bin/env bash
set -euo pipefail

# Cross-platform (CPU) replication of the frozen reproducibility matrix.
#
# This is the second-platform run for the paper's cross-platform claim. It is
# scoped to the four Feynman-protocol methods and a reduced trial count because
# CPU throughput is ~5x lower for the INT8 path. Within-platform output was
# already shown deterministic (unique_outputs = 1.0 on GPU), so 3 fresh
# processes are sufficient to establish the GPU<->CPU comparison.
#
# Loops are ordered batch-ascending so the fast, most-informative b1 conditions
# for every method complete first; partial results are usable at any time.

DEVICE=cpu
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-/Users/jjburrell/MITN/IntChat/artifacts/nanochat-paper}"
MODEL_TAG="${MODEL_TAG:-paper-d12}"
STEP="${STEP:-2520}"
TRIALS="${TRIALS:-3}"
REPLICAS_PER_PROMPT="${REPLICAS_PER_PROMPT:-10}"
MAX_TOKENS="${MAX_TOKENS:-32}"
DECODE="${DECODE:-greedy}"
PROMPTS="${PROMPTS:-paper/prompts.jsonl}"
CALIBRATION_FILE="${CALIBRATION_FILE:-paper/calibration.jsonl}"
CALIBRATION_SAMPLES="${CALIBRATION_SAMPLES:-32}"
OUT="${OUT:-repro_results/cpu_${DECODE}}"
REFERENCE="$OUT/reference.pt"
BASE_DIR="$NANOCHAT_BASE_DIR"
STEP_PADDED="$(printf '%06d' "$STEP")"
MODEL_FILE="$BASE_DIR/base_checkpoints/$MODEL_TAG/model_${STEP_PADDED}.pt"
META_FILE="$BASE_DIR/base_checkpoints/$MODEL_TAG/meta_${STEP_PADDED}.json"

mkdir -p "$OUT"
if [[ ! -f "$MODEL_FILE" || ! -f "$META_FILE" ]]; then
  echo "Missing pinned checkpoint: $MODEL_FILE or $META_FILE" >&2
  exit 2
fi
cp "$PROMPTS" "$OUT/prompts.jsonl"
cp "$CALIBRATION_FILE" "$OUT/calibration.jsonl"
git rev-parse HEAD > "$OUT/git_commit.txt"
git status --porcelain > "$OUT/git_status.txt"
# macOS platform capture (no nvidia-smi); record CPU + OS for the artifact.
{ uname -a; echo; sysctl -n machdep.cpu.brand_string 2>/dev/null || true; sysctl -n hw.ncpu 2>/dev/null || true; } > "$OUT/platform.txt"
uv pip freeze > "$OUT/pip_freeze.txt"
shasum -a 256 "$MODEL_FILE" "$META_FILE" "$BASE_DIR/tokenizer/tokenizer.pkl" > "$OUT/artifact_sha256.txt"

# method:precision:import-time dtype  (Feynman-protocol core set; no det/fp16)
conditions=(
  "fp32:fp32:float32"
  "bf16:bf16:bfloat16"
  "static-int8:static-int8:float32"
  "dynamic-int8:dynamic-int8:float32"
)

for batch_size in 1 8 64; do
  for condition in "${conditions[@]}"; do
    IFS=: read -r method precision dtype <<< "$condition"
    for trial in $(seq 0 $((TRIALS - 1))); do
      reference_args=()
      calibration_args=()
      if [[ "$precision" == static-int8 ]]; then
        calibration_args+=(--calibration-file "$CALIBRATION_FILE" --calibration-samples "$CALIBRATION_SAMPLES")
      fi
      if [[ "$method" == fp32 && "$batch_size" == 1 && "$trial" == 0 ]]; then
        reference_args+=(--write-reference)
      fi
      out_json="$OUT/${method}_b${batch_size}_trial${trial}.json"
      if [[ -f "$out_json" ]]; then
        echo "skip existing $out_json"
        continue
      fi
      echo "=== $method b$batch_size trial$trial ==="
      env "NANOCHAT_DTYPE=$dtype" python -m scripts.repro_experiment \
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
        --output "$out_json" \
        ${calibration_args[@]+"${calibration_args[@]}"} \
        ${reference_args[@]+"${reference_args[@]}"}
    done
  done
done

python -m scripts.repro_report "$OUT"
