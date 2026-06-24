"""Exact Feynman-1000 end-to-end determinism condition.

The benchmark follows Thinking Machines Lab's behavioral protocol: 1,000
greedy completions of "Tell me about Richard Feynman", each 1,000 tokens long.
Unlike a continuous-batching server, this runner controls physical batch shape
explicitly so batch invariance can be measured at batch sizes 1, 8, and 64.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import time

import torch

from nanochat.checkpoint_manager import load_model
from nanochat.common import compute_init
from nanochat.static_quant import convert_dynamic_linears
from scripts.repro_experiment import (
    calibrate_static_int8,
    environment_metadata,
    generate_batch,
    generate_replicas,
    load_calibration_texts,
    validate_process_dtype,
)


PROMPT = "Tell me about Richard Feynman"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--precision", required=True, choices=("fp32", "fp16", "bf16", "static-int8", "dynamic-int8"))
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--batch-size", type=int, required=True, choices=(1, 8, 64))
    parser.add_argument("--completions", type=int, default=1000)
    parser.add_argument("--max-tokens", type=int, default=1000)
    parser.add_argument("--model-tag", required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--device-type", choices=("cuda", "cpu", "mps"), default="cuda")
    parser.add_argument("--calibration-file", type=Path, default=Path("paper/calibration.jsonl"))
    parser.add_argument("--calibration-samples", type=int, default=32)
    parser.add_argument("--reference-artifact", type=Path, required=True)
    parser.add_argument("--write-reference", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def token_hash(tokens: tuple[int, ...]) -> str:
    payload = b"".join(int(token).to_bytes(4, "little", signed=False) for token in tokens)
    return hashlib.sha256(payload).hexdigest()


def first_divergence(tokens: tuple[int, ...], reference: tuple[int, ...]) -> int | None:
    for index, (actual, expected) in enumerate(zip(tokens, reference)):
        if actual != expected:
            return index
    return None if len(tokens) == len(reference) else min(len(tokens), len(reference))


def main() -> None:
    args = parse_args()
    if args.completions != 1000 or args.max_tokens != 1000:
        raise ValueError("the exact Feynman-1000 protocol requires 1,000 completions of 1,000 tokens")
    validate_process_dtype(args.precision)
    _, _, _, _, device = compute_init(args.device_type)
    if args.deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        if device.type == "cuda" and os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {":4096:8", ":16:8"}:
            raise RuntimeError("deterministic CUDA runs require CUBLAS_WORKSPACE_CONFIG=:4096:8")
    if args.precision in {"fp32", "static-int8", "dynamic-int8"}:
        torch.set_float32_matmul_precision("highest")
        if device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = False

    model, tokenizer, meta = load_model("base", device, phase="eval", model_tag=args.model_tag, step=args.step)
    if args.precision == "fp16":
        model.half()
    elif args.precision == "bf16":
        model.bfloat16()
    else:
        model.float()
    if args.precision == "static-int8":
        texts = load_calibration_texts(args.calibration_file, args.calibration_samples)
        calibrate_static_int8(model, tokenizer, device, texts)
    elif args.precision == "dynamic-int8":
        convert_dynamic_linears(model)

    prompt_ids = tokenizer.encode(PROMPT, prepend=tokenizer.get_bos_token_id())
    if len(prompt_ids) + args.max_tokens > model.config.sequence_len:
        raise ValueError("prompt plus completion exceeds the model context length")
    uniforms = [0.0] * args.max_tokens  # ignored by greedy selection

    if args.write_reference:
        if args.precision != "fp32" or args.deterministic or args.batch_size != 1:
            raise ValueError("reference creation requires non-deterministic-control FP32 batch 1")
        reference_tokens = generate_batch(
            model, prompt_ids, 1, args.max_tokens, uniforms, "greedy", 0.0, device
        )[0].tolist()
        args.reference_artifact.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"prompt": PROMPT, "prompt_ids": prompt_ids, "tokens": reference_tokens}, args.reference_artifact)

    reference_record = torch.load(args.reference_artifact, map_location="cpu", weights_only=True)
    if reference_record["prompt"] != PROMPT or reference_record["prompt_ids"] != prompt_ids:
        raise ValueError("Feynman reference prompt/tokenization mismatch")
    reference = tuple(reference_record["tokens"])

    started = time.perf_counter()
    outputs = generate_replicas(
        model, prompt_ids, args.completions, args.batch_size, uniforms,
        "greedy", 0.0, device,
    )
    elapsed = time.perf_counter() - started
    counts = Counter(outputs)
    divergence_counts = Counter(first_divergence(tokens, reference) for tokens in outputs)
    divergent_positions = [position for position in divergence_counts if position is not None]
    sequences = [
        {
            "sha256": token_hash(tokens),
            "count": count,
            "first_divergence": first_divergence(tokens, reference),
            "tokens": list(tokens),
        }
        for tokens, count in counts.most_common()
    ]
    method = args.precision + ("-det" if args.deterministic else "")
    result = {
        "schema_version": 1,
        "protocol": "thinking-machines-feynman-1000",
        "condition": {
            "method": method, "precision": args.precision,
            "deterministic": args.deterministic, "batch_size": args.batch_size,
            "completions": args.completions, "max_tokens": args.max_tokens,
            "temperature": 0.0, "prompt": PROMPT,
            "model_tag": args.model_tag, "step": args.step,
            "calibration_samples": args.calibration_samples if args.precision == "static-int8" else None,
        },
        "results": {
            "unique_outputs": len(counts),
            "modal_count": counts.most_common(1)[0][1],
            "modal_fraction": counts.most_common(1)[0][1] / args.completions,
            "reference_match_count": counts.get(reference, 0),
            "reference_match_rate": counts.get(reference, 0) / args.completions,
            "earliest_divergence": min(divergent_positions) if divergent_positions else None,
            "divergence_position_counts": {
                "match" if position is None else str(position): count
                for position, count in sorted(divergence_counts.items(), key=lambda item: -1 if item[0] is None else item[0])
            },
            "elapsed_seconds": elapsed,
            "tokens_per_second": args.completions * args.max_tokens / elapsed,
        },
        "sequences": sequences,
        "environment": environment_metadata(device),
        "model_config": meta["model_config"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"condition": result["condition"], "results": result["results"]}, indent=2))


if __name__ == "__main__":
    main()
