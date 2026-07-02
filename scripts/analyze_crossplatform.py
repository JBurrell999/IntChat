"""Compare generated token sequences across GPU and CPU platforms.

For each method we take the modal (most frequent) output sequence per prompt on
each platform and test whether the two platforms produced the *same text*. This
is the cleanest cross-platform reproducibility metric: it needs no logit
alignment and directly answers whether integer matmul makes generation
platform-invariant.

Writes:
  paper/tables/crossplatform.tex
  paper/tables/crossplatform.csv
  paper/figures/fig5_crossplatform.pdf
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
METHODS = ["fp32", "bf16", "static-int8", "dynamic-int8"]
PRETTY = {"fp32": "FP32", "bf16": "BF16",
          "static-int8": "Static W8A8", "dynamic-int8": "Dynamic W8A8"}


def modal_sequences(path: Path) -> dict[str, tuple]:
    """prompt_id -> modal token tuple for one condition file."""
    d = json.loads(path.read_text())
    out = {}
    for rec in d["prompt_records"]:
        counts = rec["output_counts"]
        best = max(counts, key=lambda c: c["count"])
        out[rec["prompt_id"]] = tuple(best["tokens"])
    return out


def platform_match(gpu_dir: Path, cpu_dir: Path, method: str, batch: int, trial: int):
    gpu_f = gpu_dir / f"{method}_b{batch}_trial{trial}.json"
    cpu_f = cpu_dir / f"{method}_b{batch}_trial{trial}.json"
    if not gpu_f.exists() or not cpu_f.exists():
        return None
    g = modal_sequences(gpu_f)
    c = modal_sequences(cpu_f)
    shared = sorted(set(g) & set(c))
    if not shared:
        return None
    matches = sum(1 for p in shared if g[p] == c[p])
    return matches / len(shared), len(shared)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="repro_results/conference_greedy")
    ap.add_argument("--cpu", default="repro_results/cpu_greedy")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--trial", type=int, default=0)
    args = ap.parse_args()

    gpu_dir = ROOT / args.gpu
    cpu_dir = ROOT / args.cpu
    tabs = ROOT / "paper" / "tables"
    figs = ROOT / "paper" / "figures"
    tabs.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)

    rows = []
    for m in METHODS:
        res = platform_match(gpu_dir, cpu_dir, m, args.batch, args.trial)
        if res is None:
            continue
        rate, n = res
        rows.append((m, rate, n))
        print(f"{PRETTY[m]:14s}  GPU<->CPU text match: {rate*100:5.1f}%  (n={n} prompts)")

    if not rows:
        print("No overlapping GPU/CPU conditions yet; leaving placeholder table.")
        return

    # CSV
    with (tabs / "crossplatform.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "gpu_cpu_text_match", "n_prompts", "batch", "trial"])
        for m, rate, n in rows:
            w.writerow([m, f"{rate:.4f}", n, args.batch, args.trial])

    # LaTeX
    lines = [r"\begin{tabular}{lr}", r"\toprule",
             r"Method & GPU$\leftrightarrow$CPU text match \\", r"\midrule"]
    for m, rate, n in rows:
        lines.append(f"{PRETTY[m]} & {rate*100:.1f}\\% \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (tabs / "crossplatform.tex").write_text("\n".join(lines) + "\n")

    # Figure
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    labels = [PRETTY[m] for m, _, _ in rows]
    vals = [rate * 100 for _, rate, _ in rows]
    colors = ["#4c72b0", "#55a868", "#c44e52", "#8172b3"][: len(rows)]
    ax.bar(labels, vals, color=colors)
    ax.set_ylabel("GPU$\\leftrightarrow$CPU text match (%)")
    ax.set_ylim(0, 105)
    ax.set_title(f"Cross-platform generation agreement (batch {args.batch})")
    for i, v in enumerate(vals):
        ax.text(i, v + 1, f"{v:.0f}%", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(figs / "fig5_crossplatform.pdf", bbox_inches="tight")
    fig.savefig(figs / "fig5_crossplatform.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Wrote crossplatform table + figure.")


if __name__ == "__main__":
    main()
