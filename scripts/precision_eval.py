"""Held-out NanoChat BPB evaluation for each inference arithmetic condition."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import torch

from nanochat.checkpoint_manager import load_model
from nanochat.common import compute_init
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from nanochat.loss_eval import evaluate_bpb
from nanochat.static_quant import convert_dynamic_linears
from nanochat.tokenizer import get_token_bytes
from scripts.repro_experiment import (
    calibrate_static_int8,
    environment_metadata,
    load_calibration_texts,
    validate_process_dtype,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--precision", required=True, choices=("fp32", "fp16", "bf16", "static-int8", "dynamic-int8"))
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--model-tag", required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--device-type", choices=("cuda", "cpu", "mps"), default="cuda")
    parser.add_argument("--device-batch-size", type=int, default=4)
    parser.add_argument("--eval-tokens", type=int, default=2_097_152)
    parser.add_argument("--calibration-file", type=Path, default=Path("paper/calibration.jsonl"))
    parser.add_argument("--calibration-samples", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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

    sequence_len = meta["model_config"]["sequence_len"]
    steps = args.eval_tokens // (args.device_batch_size * sequence_len)
    if steps < 1:
        raise ValueError("eval-tokens must cover at least one complete batch")
    loader = tokenizing_distributed_data_loader_bos_bestfit(
        tokenizer, args.device_batch_size, sequence_len, split="val", device=device
    )
    token_bytes = get_token_bytes(device=device)
    started = time.perf_counter()
    bpb = evaluate_bpb(model, loader, steps, token_bytes)
    elapsed = time.perf_counter() - started
    result = {
        "schema_version": 1,
        "condition": {
            "precision": args.precision,
            "method": args.precision + ("-det" if args.deterministic else ""),
            "deterministic": args.deterministic,
            "calibration_samples": args.calibration_samples if args.precision == "static-int8" else None,
            "model_tag": args.model_tag,
            "step": args.step,
        },
        "quality": {
            "validation_bpb": bpb,
            "eval_tokens": steps * args.device_batch_size * sequence_len,
            "elapsed_seconds": elapsed,
        },
        "environment": environment_metadata(device),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
