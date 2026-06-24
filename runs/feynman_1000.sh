#!/usr/bin/env bash
set -euo pipefail

DEVICE="${DEVICE:-cuda}"
MODEL_TAG="${MODEL_TAG:?Set MODEL_TAG}"
STEP="${STEP:?Set STEP}"
OUT="${OUT:-repro_results/feynman_1000}"
CALIBRATION_FILE="${CALIBRATION_FILE:-paper/calibration.jsonl}"
CALIBRATION_SAMPLES="${CALIBRATION_SAMPLES:-32}"
REFERENCE="$OUT/reference.pt"
BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
STEP_PADDED="$(printf '%06d' "$STEP")"
MODEL_FILE="$BASE_DIR/base_checkpoints/$MODEL_TAG/model_${STEP_PADDED}.pt"
META_FILE="$BASE_DIR/base_checkpoints/$MODEL_TAG/meta_${STEP_PADDED}.json"

# Space-separated subset is allowed for staged execution. The complete paper
# run should use: fp32 bf16 static-int8 dynamic-int8 fp32-det fp16 fp16-det
METHODS="${METHODS:-fp32 bf16 static-int8 dynamic-int8}"

mkdir -p "$OUT"
if [[ ! -f "$MODEL_FILE" || ! -f "$META_FILE" ]]; then
  echo "Missing pinned checkpoint: $MODEL_FILE or $META_FILE" >&2
  exit 2
fi
cp "$CALIBRATION_FILE" "$OUT/calibration.jsonl"
git rev-parse HEAD > "$OUT/git_commit.txt"
git status --porcelain > "$OUT/git_status.txt"
nvidia-smi -q > "$OUT/nvidia_smi.txt"
uv pip freeze > "$OUT/pip_freeze.txt"
sha256sum "$MODEL_FILE" "$META_FILE" "$BASE_DIR/tokenizer/tokenizer.pkl" > "$OUT/artifact_sha256.txt"

for method in $METHODS; do
  precision="$method"
  dtype=float32
  deterministic=0
  case "$method" in
    fp16) dtype=float16 ;;
    bf16) dtype=bfloat16 ;;
    static-int8|dynamic-int8|fp32) dtype=float32 ;;
    fp32-det) precision=fp32; dtype=float32; deterministic=1 ;;
    fp16-det) precision=fp16; dtype=float16; deterministic=1 ;;
    *) echo "Unknown Feynman method: $method" >&2; exit 2 ;;
  esac
  for batch_size in 1 8 64; do
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
    if [[ "$method" == fp32 && "$batch_size" == 1 ]]; then
      reference_args+=(--write-reference)
    fi
    if [[ ! -f "$REFERENCE" && ${#reference_args[@]} -eq 0 ]]; then
      echo "Run fp32 batch 1 first so $REFERENCE exists" >&2
      exit 2
    fi
    env "${env_args[@]}" python -m scripts.feynman_1000 \
      --precision "$precision" --batch-size "$batch_size" \
      --model-tag "$MODEL_TAG" --step "$STEP" --device-type "$DEVICE" \
      --calibration-file "$CALIBRATION_FILE" --calibration-samples "$CALIBRATION_SAMPLES" \
      --reference-artifact "$REFERENCE" \
      --output "$OUT/feynman_${method}_b${batch_size}.json" \
      "${deterministic_args[@]}" "${calibration_args[@]}" "${reference_args[@]}"
  done
done

python -m scripts.feynman_report "$OUT"
