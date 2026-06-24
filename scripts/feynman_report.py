"""Aggregate Feynman-1000 conditions, including outputs pooled over batch size."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=Path)
    args = parser.parse_args()
    paths = sorted(args.results_dir.glob("feynman_*_b*.json"))
    if not paths:
        raise FileNotFoundError("no Feynman condition files found")
    records = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    rows = []
    pooled = defaultdict(Counter)
    for record in records:
        condition, result = record["condition"], record["results"]
        rows.append({
            "method": condition["method"], "batch_size": condition["batch_size"],
            "unique_outputs": result["unique_outputs"], "modal_fraction": result["modal_fraction"],
            "reference_match_rate": result["reference_match_rate"],
            "earliest_divergence": result["earliest_divergence"],
            "tokens_per_second": result["tokens_per_second"],
        })
        for sequence in record["sequences"]:
            pooled[condition["method"]][sequence["sha256"]] += sequence["count"]
    rows.sort(key=lambda row: (row["method"], row["batch_size"]))
    fields = tuple(rows[0])
    with (args.results_dir / "feynman_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    pooled_rows = [
        {
            "method": method, "completions_across_batches": sum(counts.values()),
            "unique_outputs_across_batches": len(counts),
            "pooled_modal_fraction": max(counts.values()) / sum(counts.values()),
        }
        for method, counts in sorted(pooled.items())
    ]
    pooled_fields = tuple(pooled_rows[0])
    with (args.results_dir / "feynman_pooled.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=pooled_fields)
        writer.writeheader()
        writer.writerows(pooled_rows)
    lines = ["| " + " | ".join(fields) + " |", "|" + "|".join("---" for _ in fields) + "|"]
    lines.extend("| " + " | ".join(str(row[field]) for field in fields) + " |" for row in rows)
    (args.results_dir / "feynman_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
