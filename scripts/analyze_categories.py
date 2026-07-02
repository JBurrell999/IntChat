"""Per-prompt-category reference-match rate, showing where precision flips tokens.

Writes paper/tables/categories.tex.
"""
from __future__ import annotations

import collections
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GPU = ROOT / "repro_results" / "conference_greedy"
METHODS = [("fp16", "FP16"), ("bf16", "BF16"),
           ("static-int8", "Static W8A8"), ("dynamic-int8", "Dynamic W8A8")]
CATS = ["factual", "reasoning", "numerical", "technical",
        "instruction", "completion", "ambiguous"]


def per_category(method: str, batch=1, trial=0) -> dict[str, float]:
    d = json.loads((GPU / f"{method}_b{batch}_trial{trial}.json").read_text())
    by = collections.defaultdict(list)
    for r in d["prompt_records"]:
        by[r["category"]].append(r["generation"]["reference_match_rate"])
    return {c: sum(v) / len(v) for c, v in by.items()}


def main():
    data = {m: per_category(m) for m, _ in METHODS}
    header = " & ".join(["Category"] + [p for _, p in METHODS])
    lines = [r"\begin{tabular}{l" + "r" * len(METHODS) + "}", r"\toprule",
             header + r" \\", r"\midrule"]
    for c in CATS:
        row = [c.capitalize()]
        for m, _ in METHODS:
            row.append(f"{data[m].get(c, float('nan')):.2f}")
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    out = ROOT / "paper" / "tables" / "categories.tex"
    out.write_text("\n".join(lines) + "\n")
    print("Wrote", out)
    for m, p in METHODS:
        print(f"{p:14s}", {c: round(data[m].get(c, float('nan')), 2) for c in CATS})


if __name__ == "__main__":
    main()
