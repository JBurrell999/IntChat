"""Create paper-ready CSV/Markdown tables from held-out BPB results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=Path)
    args = parser.parse_args()
    paths = sorted(args.results_dir.glob("quality_*.json"))
    if not paths:
        raise FileNotFoundError("no quality result files found")
    rows = []
    for path in paths:
        record = json.loads(path.read_text(encoding="utf-8"))
        rows.append({
            "method": record["condition"]["method"],
            "calibration_samples": record["condition"]["calibration_samples"],
            "validation_bpb": record["quality"]["validation_bpb"],
            "eval_tokens": record["quality"]["eval_tokens"],
            "elapsed_seconds": record["quality"]["elapsed_seconds"],
        })
    rows.sort(key=lambda row: (row["method"], row["calibration_samples"] or 0))
    fields = tuple(rows[0])
    with (args.results_dir / "quality_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    lines = ["| " + " | ".join(fields) + " |", "|" + "|".join("---" for _ in fields) + "|"]
    lines.extend("| " + " | ".join(str(row[field]) for field in fields) + " |" for row in rows)
    (args.results_dir / "quality_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
