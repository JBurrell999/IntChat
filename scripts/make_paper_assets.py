"""Generate paper figures, a consolidated results CSV, and LaTeX tables.

Reads the reproducibility summary(ies) produced by scripts.repro_report and the
quality summary, then writes:
  paper/figures/fig1_match_rate.pdf
  paper/figures/fig2_kl.pdf
  paper/figures/fig3_logit_error.pdf
  paper/figures/fig4_fidelity.pdf
  paper/tables/results_main.csv
  paper/tables/results_main.tex
  paper/tables/fidelity.tex

Re-run any time; it uses whatever conditions are present, so it works on the
partial CPU sweep as well as the complete GPU matrix.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]

# Methods shown in the main figures (the det variants are numerically identical
# to their non-det counterparts here, so they are omitted from the figures and
# reported only in the throughput/determinism discussion).
FIG_METHODS = ["fp32", "fp16", "bf16", "static-int8", "dynamic-int8"]
PRETTY = {
    "fp32": "FP32",
    "fp32-det": "FP32 (det)",
    "fp16": "FP16",
    "fp16-det": "FP16 (det)",
    "bf16": "BF16",
    "static-int8": "Static W8A8",
    "dynamic-int8": "Dynamic W8A8",
}
COLORS = {
    "fp32": "#4c72b0",
    "fp16": "#dd8452",
    "bf16": "#55a868",
    "static-int8": "#c44e52",
    "dynamic-int8": "#8172b3",
}
BATCHES = [1, 8, 64]


def load_summary(path: Path) -> dict:
    rows = {}
    with path.open() as fh:
        for r in csv.DictReader(fh):
            rows[(r["method"], int(r["batch_size"]))] = r
    return rows


def _grouped_bar(ax, summary, metric, methods, log=False):
    n = len(methods)
    width = 0.8 / n
    x = range(len(BATCHES))
    for i, m in enumerate(methods):
        vals = []
        for b in BATCHES:
            r = summary.get((m, b))
            vals.append(float(r[metric]) if r else 0.0)
        offs = [xi - 0.4 + width * (i + 0.5) for xi in x]
        ax.bar(offs, vals, width=width, label=PRETTY[m], color=COLORS.get(m, "#777"))
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"batch {b}" for b in BATCHES])
    if log:
        ax.set_yscale("log")


def fig_match_rate(summary, methods, out):
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    _grouped_bar(ax, summary, "reference_match_rate", methods)
    ax.set_ylabel("Reference-match rate")
    ax.set_ylim(0, 1.05)
    ax.set_title("Greedy match to FP32 batch-1 reference")
    ax.legend(fontsize=7, ncol=2, loc="lower left")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_kl(summary, methods, out):
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    _grouped_bar(ax, summary, "kl_mean", methods, log=True)
    ax.set_ylabel("Mean KL from FP32 (nats, log)")
    ax.set_title("Distributional divergence from FP32 batch-1")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_logit_error(summary, methods, out):
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    _grouped_bar(ax, summary, "logit_abs_max", methods, log=True)
    ax.set_ylabel("Max |logit - FP32| (log)")
    ax.set_title("Worst-case logit error vs FP32 batch-1")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_fidelity(quality_path, out):
    rows = []
    with quality_path.open() as fh:
        for r in csv.DictReader(fh):
            label = r["method"]
            if r.get("calibration_samples"):
                label += f" (cal {r['calibration_samples']})"
            rows.append((label, float(r["validation_bpb"])))
    rows.sort(key=lambda t: t[1])
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    labels = [t[0] for t in rows]
    vals = [t[1] for t in rows]
    ax.barh(labels, vals, color="#4c72b0")
    ax.set_xlabel("Validation bits-per-byte (lower is better)")
    ax.set_xlim(min(vals) - 0.002, max(vals) + 0.002)
    ax.set_title("Fidelity: held-out BPB (2.1M tokens)")
    for i, v in enumerate(vals):
        ax.text(v, i, f" {v:.4f}", va="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_main_table(summary, quality, csv_out, tex_out):
    bpb = {}
    with quality.open() as fh:
        for r in csv.DictReader(fh):
            # main table uses the paper's default static calibration (32)
            if r["method"] == "static-int8" and r.get("calibration_samples") not in ("", "32"):
                continue
            bpb[r["method"]] = float(r["validation_bpb"])
    cols = ["method", "batch_size", "reference_match_rate", "logit_abs_mean",
            "logit_abs_max", "kl_mean", "validation_bpb", "generations_per_second"]
    with csv_out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for m in FIG_METHODS:
            for b in BATCHES:
                r = summary.get((m, b))
                if not r:
                    continue
                w.writerow([m, b, r["reference_match_rate"], r["logit_abs_mean"],
                            r["logit_abs_max"], r["kl_mean"],
                            f"{bpb.get(m, float('nan')):.4f}", r["generations_per_second"]])
    # LaTeX booktabs table
    lines = [
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Method & Batch & Match & Mean$|\Delta|$ & Max$|\Delta|$ & KL & BPB \\",
        r"\midrule",
    ]
    for m in FIG_METHODS:
        for b in BATCHES:
            r = summary.get((m, b))
            if not r:
                continue
            lines.append(
                f"{PRETTY[m]} & {b} & {float(r['reference_match_rate']):.2f} & "
                f"{float(r['logit_abs_mean']):.2e} & {float(r['logit_abs_max']):.2e} & "
                f"{float(r['kl_mean']):.2e} & {bpb.get(m, float('nan')):.4f} \\\\"
            )
        lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}"]
    tex_out.write_text("\n".join(lines) + "\n")


def write_fidelity_table(quality, tex_out):
    rows = []
    with quality.open() as fh:
        for r in csv.DictReader(fh):
            label = PRETTY.get(r["method"], r["method"])
            cal = r.get("calibration_samples") or "--"
            rows.append((label, cal, float(r["validation_bpb"])))
    lines = [r"\begin{tabular}{llr}", r"\toprule",
             r"Method & Calib.\ samples & Validation BPB \\", r"\midrule"]
    for label, cal, v in rows:
        lines.append(f"{label} & {cal} & {v:.5f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    tex_out.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="repro_results/conference_greedy/summary.csv")
    ap.add_argument("--quality", default="repro_results/quality/quality_summary.csv")
    args = ap.parse_args()

    summary = load_summary(ROOT / args.summary)
    quality = ROOT / args.quality
    figs = ROOT / "paper" / "figures"
    tabs = ROOT / "paper" / "tables"
    figs.mkdir(parents=True, exist_ok=True)
    tabs.mkdir(parents=True, exist_ok=True)

    present = [m for m in FIG_METHODS if any((m, b) in summary for b in BATCHES)]
    fig_match_rate(summary, present, figs / "fig1_match_rate.pdf")
    fig_kl(summary, present, figs / "fig2_kl.pdf")
    fig_logit_error(summary, present, figs / "fig3_logit_error.pdf")
    if quality.exists():
        fig_fidelity(quality, figs / "fig4_fidelity.pdf")
    write_main_table(summary, quality, tabs / "results_main.csv", tabs / "results_main.tex")
    if quality.exists():
        write_fidelity_table(quality, tabs / "fidelity.tex")
    print("Wrote figures to", figs)
    print("Wrote tables to", tabs)


if __name__ == "__main__":
    main()
