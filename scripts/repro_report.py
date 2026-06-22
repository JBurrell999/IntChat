"""Aggregate trial JSON and compute prompt-cluster bootstrap intervals."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
import random


METRICS = (
    "unique_outputs", "reference_match_rate", "modal_fraction",
    "logit_abs_mean", "logit_abs_max", "logit_rmse", "kl_mean", "kl_max",
)


def summarize_prompts(trials: list[dict]) -> dict[str, dict[str, float]]:
    records = defaultdict(list)
    for trial in trials:
        for record in trial["prompt_records"]:
            records[record["prompt_id"]].append(record)
    summaries = {}
    for prompt_id, prompt_records in records.items():
        output_counts = Counter()
        for record in prompt_records:
            for item in record["output_counts"]:
                output_counts[tuple(item["tokens"])] += item["count"]
        total = sum(output_counts.values())
        summary = {
            "unique_outputs": float(len(output_counts)),
            "reference_match_rate": sum(r["generation"]["reference_match_rate"] for r in prompt_records) / len(prompt_records),
            "modal_fraction": max(output_counts.values()) / total,
        }
        for metric in ("logit_abs_mean", "logit_abs_max", "logit_rmse", "kl_mean", "kl_max"):
            summary[metric] = sum(r["logits"][metric] for r in prompt_records) / len(prompt_records)
        summaries[prompt_id] = summary
    return summaries


def cluster_interval(summaries: dict[str, dict[str, float]], metric: str, iterations: int = 10000) -> tuple[float, float, float]:
    prompt_ids = sorted(summaries)
    estimate = sum(summaries[prompt_id][metric] for prompt_id in prompt_ids) / len(prompt_ids)
    rng = random.Random(20260621)
    samples = []
    for _ in range(iterations):
        selected = [rng.choice(prompt_ids) for _ in prompt_ids]
        values = [summaries[prompt_id][metric] for prompt_id in selected]
        samples.append(sum(values) / len(values))
    samples.sort()
    return estimate, samples[int(0.025 * iterations)], samples[int(0.975 * iterations)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=Path)
    args = parser.parse_args()
    paths = sorted(args.results_dir.glob("*_b*_trial*.json"))
    if len(paths) != 45:
        raise RuntimeError(f"expected 45 completed trial files, found {len(paths)}")
    trials = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    groups = defaultdict(list)
    for trial in trials:
        condition = trial["condition"]
        groups[(condition["precision"], condition["batch_size"])].append(trial)

    rows = []
    order = {"fp32": 0, "fp16": 1, "static-int8": 2}
    for (precision, batch_size), condition_trials in sorted(groups.items(), key=lambda item: (order[item[0][0]], item[0][1])):
        if len(condition_trials) != 5:
            raise RuntimeError(f"{precision}/batch{batch_size} has {len(condition_trials)} trials, expected 5")
        summaries = summarize_prompts(condition_trials)
        row = {"precision": precision, "batch_size": batch_size, "generations": 1000}
        for metric in METRICS:
            estimate, low, high = cluster_interval(summaries, metric)
            row[metric] = estimate
            row[f"{metric}_ci_low"] = low
            row[f"{metric}_ci_high"] = high
        row["generations_per_second"] = sum(t["aggregate"]["generations_per_second"] for t in condition_trials) / 5
        rows.append(row)

    fieldnames = list(rows[0])
    csv_path = args.results_dir / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    table_fields = ("precision", "batch_size", "generations", "unique_outputs", "reference_match_rate", "logit_abs_max", "kl_mean")
    header = "| " + " | ".join(table_fields) + " |"
    separator = "|" + "|".join("---" for _ in table_fields) + "|"
    body = []
    for row in rows:
        values = []
        for field in table_fields:
            value = row[field]
            if field in METRICS:
                value = f"{value:.6g} [{row[field + '_ci_low']:.6g}, {row[field + '_ci_high']:.6g}]"
            values.append(str(value))
        body.append("| " + " | ".join(values) + " |")
    markdown_path = args.results_dir / "summary.md"
    markdown_path.write_text("\n".join((header, separator, *body)) + "\n", encoding="utf-8")
    print(f"Wrote {csv_path} and {markdown_path}")


if __name__ == "__main__":
    main()
