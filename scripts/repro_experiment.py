"""One trial of the NanoChat precision/reproducibility experiment.

The conference protocol launches this module in fresh processes. Each trial
evaluates every preregistered prompt with the exact requested physical batch
shape and writes prompt-level observations for cluster bootstrap analysis.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import platform
import subprocess
import time

import torch
import torch.nn.functional as F

from nanochat.checkpoint_manager import load_model
from nanochat.common import compute_init
from nanochat.engine import KVCache
from nanochat.static_quant import attach_linear_calibrators, convert_calibrated_linears


CALIBRATION_TEXTS = (
    "The capital of France is Paris.",
    "Water freezes at zero degrees Celsius.",
    "A short story begins in a quiet village.",
    "If five plus seven equals twelve, then",
    "Machine learning systems transform inputs into outputs.",
    "The planets orbit the Sun because of gravity.",
    "Write a polite response to a simple question.",
    "Reproducibility requires controlling randomness and numerical behavior.",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--precision", required=True, choices=("fp32", "fp16", "static-int8"))
    parser.add_argument("--batch-size", type=int, required=True, choices=(1, 8, 64))
    parser.add_argument("--replicas-per-prompt", type=int, default=10)
    parser.add_argument("--trial", type=int, required=True)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--decode", choices=("greedy", "fixed-sample"), default="greedy")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--source", choices=("base", "sft", "rl"), default="base")
    parser.add_argument("--model-tag", default=None)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--device-type", choices=("cuda", "cpu", "mps"), default="cuda")
    parser.add_argument("--reference-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--write-reference", action="store_true")
    return parser.parse_args()


def load_prompts(path: Path) -> list[dict[str, str]]:
    prompts = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(prompts) < 20:
        raise ValueError("conference protocol requires at least 20 prompts")
    ids = [item["id"] for item in prompts]
    if len(ids) != len(set(ids)):
        raise ValueError("prompt IDs must be unique")
    if any(not item.get("text", "").strip() for item in prompts):
        raise ValueError("every prompt must contain non-empty text")
    return prompts


def validate_process_dtype(precision: str) -> None:
    expected = "float16" if precision == "fp16" else "float32"
    actual = os.environ.get("NANOCHAT_DTYPE")
    if actual != expected:
        raise RuntimeError(f"precision={precision} requires NANOCHAT_DTYPE={expected}; got {actual!r}")


@torch.inference_mode()
def calibrate_static_int8(model, tokenizer, device) -> None:
    calibrators = attach_linear_calibrators(model)
    bos = tokenizer.get_bos_token_id()
    for text in CALIBRATION_TEXTS:
        ids = torch.tensor([tokenizer.encode(text, prepend=bos)], dtype=torch.long, device=device)
        model(ids)
    convert_calibrated_linears(model, calibrators)


def make_cache(model, batch_size: int, sequence_length: int, device: torch.device) -> KVCache:
    config = model.config
    return KVCache(
        batch_size=batch_size,
        num_heads=config.n_kv_head,
        seq_len=sequence_length,
        head_dim=config.n_embd // config.n_head,
        num_layers=config.n_layer,
        device=device,
        dtype=model.transformer.wte.weight.dtype,
    )


def fixed_uniforms(count: int, seed: int) -> list[float]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.rand(count, generator=generator, dtype=torch.float64).tolist()


def select_tokens(logits: torch.Tensor, method: str, uniform: float, temperature: float) -> torch.Tensor:
    if method == "greedy":
        return logits.argmax(dim=-1)
    probabilities = F.softmax(logits.float() / temperature, dim=-1)
    cdf = probabilities.cumsum(dim=-1)
    threshold = torch.tensor(uniform, device=logits.device, dtype=cdf.dtype)
    return (cdf < threshold).sum(dim=-1).clamp_max(logits.size(-1) - 1)


@torch.inference_mode()
def generate_batch(model, prompt_ids, batch_size, max_tokens, uniforms, method, temperature, device):
    cache = make_cache(model, batch_size, len(prompt_ids) + max_tokens, device)
    ids = torch.tensor(prompt_ids, dtype=torch.long, device=device).repeat(batch_size, 1)
    logits = model(ids, kv_cache=cache)[:, -1, :]
    generated = []
    for step in range(max_tokens):
        next_ids = select_tokens(logits, method, uniforms[step], temperature)
        generated.append(next_ids.cpu())
        if step + 1 < max_tokens:
            logits = model(next_ids[:, None], kv_cache=cache)[:, -1, :]
    return torch.stack(generated, dim=1)


@torch.inference_mode()
def generate_replicas(model, prompt_ids, replicas, batch_size, uniforms, method, temperature, device):
    # Always execute the declared physical batch shape. Padding rows are ignored
    # in statistics, avoiding a smaller final batch as a hidden condition.
    outputs = []
    for start in range(0, replicas, batch_size):
        valid = min(batch_size, replicas - start)
        matrix = generate_batch(
            model, prompt_ids, batch_size, len(uniforms), uniforms, method, temperature, device
        )
        outputs.extend(tuple(row) for row in matrix[:valid].tolist())
    return outputs


@torch.inference_mode()
def teacher_forced_logits(model, prompt_ids, continuation, batch_size, device):
    cache = make_cache(model, batch_size, len(prompt_ids) + len(continuation), device)
    ids = torch.tensor(prompt_ids, dtype=torch.long, device=device).repeat(batch_size, 1)
    logits = model(ids, kv_cache=cache)[:, -1, :]
    traces = []
    for step, token in enumerate(continuation):
        traces.append(logits.float().cpu())
        if step + 1 < len(continuation):
            next_ids = torch.full((batch_size, 1), token, dtype=torch.long, device=device)
            logits = model(next_ids, kv_cache=cache)[:, -1, :]
    return torch.stack(traces, dim=1)


def logit_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    reference = reference.unsqueeze(0).expand(candidate.size(0), -1, -1).float()
    difference = candidate.float() - reference
    ref_prob = F.softmax(reference, dim=-1)
    kl = F.kl_div(F.log_softmax(candidate.float(), dim=-1), ref_prob, reduction="none").sum(-1)
    return {
        "logit_abs_mean": difference.abs().mean().item(),
        "logit_abs_max": difference.abs().max().item(),
        "logit_rmse": difference.square().mean().sqrt().item(),
        "kl_mean": kl.mean().item(),
        "kl_max": kl.max().item(),
    }


def environment_metadata(device: torch.device) -> dict:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    data = {
        "platform": platform.platform(), "python": platform.python_version(),
        "torch": torch.__version__, "device": str(device), "git_commit": commit,
        "git_dirty": dirty, "pid": os.getpid(),
    }
    if device.type == "cuda":
        data.update({
            "gpu": torch.cuda.get_device_name(device), "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "capability": list(torch.cuda.get_device_capability(device)),
            "driver": torch.cuda.driver_version() if hasattr(torch.cuda, "driver_version") else None,
        })
    return data


def mean_metrics(records: list[dict], section: str) -> dict[str, float]:
    keys = records[0][section].keys()
    return {key: sum(record[section][key] for record in records) / len(records) for key in keys}


def main() -> None:
    args = parse_args()
    if args.decode == "fixed-sample" and args.temperature <= 0:
        raise ValueError("fixed-sample requires temperature > 0")
    prompts = load_prompts(args.prompts)
    validate_process_dtype(args.precision)
    _, _, _, _, device = compute_init(args.device_type)
    if args.precision in {"fp32", "static-int8"}:
        torch.set_float32_matmul_precision("highest")
        if device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = False
    model, tokenizer, meta = load_model(args.source, device, phase="eval", model_tag=args.model_tag, step=args.step)
    model.eval()
    model.half() if args.precision == "fp16" else model.float()
    if args.precision == "static-int8":
        calibrate_static_int8(model, tokenizer, device)

    bos = tokenizer.get_bos_token_id()
    tokenized = {item["id"]: tokenizer.encode(item["text"], prepend=bos) for item in prompts}
    uniform_map = {
        item["id"]: fixed_uniforms(args.max_tokens, args.seed + index * 104729)
        for index, item in enumerate(prompts)
    }

    if args.write_reference:
        if args.precision != "fp32" or args.batch_size != 1 or args.trial != 0:
            raise ValueError("reference creation requires FP32, batch 1, trial 0")
        references = {}
        for item in prompts:
            prompt_id = item["id"]
            output = generate_batch(
                model, tokenized[prompt_id], 1, args.max_tokens, uniform_map[prompt_id],
                args.decode, args.temperature, device,
            )[0].tolist()
            logits = teacher_forced_logits(model, tokenized[prompt_id], output, 1, device)[0]
            references[prompt_id] = {"prompt_ids": tokenized[prompt_id], "continuation": output, "logits": logits}
        args.reference_artifact.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "prompts": references, "decode": args.decode, "temperature": args.temperature,
            "seed": args.seed, "max_tokens": args.max_tokens,
        }, args.reference_artifact)

    reference = torch.load(args.reference_artifact, map_location="cpu", weights_only=True)
    expected_config = (args.decode, args.temperature, args.seed, args.max_tokens)
    actual_config = (reference["decode"], reference["temperature"], reference["seed"], reference["max_tokens"])
    if actual_config != expected_config:
        raise ValueError(f"reference configuration mismatch: {actual_config} != {expected_config}")

    prompt_records = []
    started = time.perf_counter()
    for item in prompts:
        prompt_id = item["id"]
        prompt_reference = reference["prompts"][prompt_id]
        if prompt_reference["prompt_ids"] != tokenized[prompt_id]:
            raise ValueError(f"tokenization differs from reference for {prompt_id}")
        outputs = generate_replicas(
            model, tokenized[prompt_id], args.replicas_per_prompt, args.batch_size,
            uniform_map[prompt_id], args.decode, args.temperature, device,
        )
        candidate_logits = teacher_forced_logits(
            model, tokenized[prompt_id], prompt_reference["continuation"], args.batch_size, device
        )
        counts = Counter(outputs)
        reference_output = tuple(prompt_reference["continuation"])
        generation = {
            "unique_outputs": len(counts),
            "reference_match_rate": counts.get(reference_output, 0) / args.replicas_per_prompt,
            "modal_fraction": counts.most_common(1)[0][1] / args.replicas_per_prompt,
        }
        prompt_records.append({
            "prompt_id": prompt_id, "category": item.get("category", "unspecified"),
            "generation": generation,
            "logits": logit_metrics(prompt_reference["logits"], candidate_logits),
            "output_counts": [
                {"tokens": list(tokens), "count": count}
                for tokens, count in sorted(counts.items())
            ],
        })
    elapsed = time.perf_counter() - started
    generations = len(prompts) * args.replicas_per_prompt
    result = {
        "schema_version": 2,
        "condition": {
            "precision": args.precision, "batch_size": args.batch_size,
            "trial": args.trial, "prompts": len(prompts),
            "replicas_per_prompt": args.replicas_per_prompt,
            "generations": generations, "decode": args.decode,
            "temperature": args.temperature, "seed": args.seed,
            "max_tokens": args.max_tokens, "source": args.source,
            "model_tag": args.model_tag, "step": args.step,
        },
        "aggregate": {
            "generation": mean_metrics(prompt_records, "generation"),
            "logits": mean_metrics(prompt_records, "logits"),
            "elapsed_seconds": elapsed,
            "generations_per_second": generations / elapsed,
        },
        "prompt_records": prompt_records,
        "environment": environment_metadata(device),
        "model_config": meta["model_config"],
        "prototype_note": "static-int8 uses INT8 linear matmuls; nonlinear, residual, and dequantization operations remain floating point",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["aggregate"], indent=2))


if __name__ == "__main__":
    main()
