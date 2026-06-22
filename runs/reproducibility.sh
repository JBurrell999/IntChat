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
OUT="${OUT:-repro_results/conference_${DECODE}}"
REFERENCE="$OUT/reference.pt"

if [[ "$TRIALS" -ne 5 || "$REPLICAS_PER_PROMPT" -ne 10 ]]; then
  echo "Conference protocol requires TRIALS=5 and REPLICAS_PER_PROMPT=10 (1,000 generations/condition)." >&2
  exit 2
fi

mkdir -p "$OUT"
cp "$PROMPTS" "$OUT/prompts.jsonl"
git rev-parse HEAD > "$OUT/git_commit.txt"
git status --porcelain > "$OUT/git_status.txt"
nvidia-smi -q > "$OUT/nvidia_smi.txt"
python -m pip freeze > "$OUT/pip_freeze.txt"

for precision in fp32 fp16 static-int8; do
  dtype=float32
  if [[ "$precision" == fp16 ]]; then dtype=float16; fi
  for batch_size in 1 8 64; do
    for trial in $(seq 0 $((TRIALS - 1))); do
      reference_args=()
      if [[ "$precision" == fp32 && "$batch_size" == 1 && "$trial" == 0 ]]; then
        reference_args+=(--write-reference)
      fi
      NANOCHAT_DTYPE="$dtype" python -m scripts.repro_experiment \
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
        --output "$OUT/${precision}_b${batch_size}_trial${trial}.json" \
        "${reference_args[@]}"
    done
  done
done

python -m scripts.repro_report "$OUT"
