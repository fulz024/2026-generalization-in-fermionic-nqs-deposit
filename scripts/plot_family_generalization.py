"""Plot existing pure/neuralized MPS fits for the family-organized revision.

No fits or raw inputs are changed. The full four-arm ordering figure remains
in the Supplement; this view isolates the three main-text comparisons.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import plot_fig2 as source
import prl_palette as palette


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper/figures/main/fig5_mps_generalization"
AUDIT = ROOT / "tmp/pdfs/family_generalization"
STYLES = (
    ("pure_mps_d64", r"pure MPS, $D=64$", palette.PHYSICAL_BASELINE, "s", "white"),
    ("neural_product_d1", r"neural, $D=1$", palette.DIRECT_MID, "D", "white"),
    ("nn_fmps_d64", r"NN-fMPS, $D=64$", palette.MPS, "o", palette.MPS),
)
NAMES = {
    "mps_d64_nested_control": "pure_mps_d64",
    "neuralized_mps_scan_d1_w282": "neural_product_d1",
    "neuralized_mps_d64_w18": "nn_fmps_d64",
}


def plot_orders(axis, rows, orders, labels, title):
    report = []
    for offset, (architecture, label, color, marker, face) in zip((-0.09, 0, 0.09), STYLES):
        means = []
        for position, ordering in enumerate(orders):
            fits = rows[rows["architecture"].eq(architecture) & rows["ordering"].eq(ordering)]
            fits = fits.sort_values("seed")
            if len(fits) != 5 or fits["seed"].nunique() != 5:
                raise ValueError(f"Expected five distinct seeds: {title}, {architecture}, {ordering}")
            values = fits["C_hidden"].to_numpy(dtype=float)
            mean = float(values.mean())
            means.append(mean)
            axis.scatter(position + offset + np.linspace(-0.027, 0.027, 5), values,
                         marker=marker, s=16, facecolors="none", edgecolors=color,
                         linewidths=0.7, alpha=0.3, zorder=2)
            report.append({"system": title, "architecture": architecture, "ordering": ordering,
                           "n_seed": len(fits), "C_U_mean": mean,
                           "C_U_std": float(values.std(ddof=1))})
        axis.plot(np.arange(len(orders)) + offset, means, color=color, marker=marker,
                  markerfacecolor=face, markersize=5.3, linewidth=1.0, label=label, zorder=3)
    axis.axhline(0, color="0.72", linewidth=0.65, zorder=1)
    axis.set(xticks=np.arange(len(orders)), xticklabels=labels,
             xlim=(-0.33, len(orders) - 0.67), ylim=(-0.035, 0.63),
             yticks=[0, 0.2, 0.4, 0.6])
    axis.set_title(title, fontsize=9, pad=8)
    axis.tick_params(axis="x", labelsize=8)
    source.style_axes(axis)
    return report


def main():
    inputs = [source.MPS / "per_seed_metrics.csv", source.MPS / "aggregate_metrics.csv",
              source.HUB34_PHYSICAL_SECTORS, source.HUB34_NNFMPS_SECTORS,
              source.HUB34_MIDORDER_SECTORS, source.HUB34_MPS_BURDEN]
    chain = pd.read_csv(inputs[0])
    aggregate = pd.read_csv(inputs[1])
    orders = ("physical", "random_04", "random_05", "even_odd")
    tails = []
    for ordering in orders:
        row = aggregate[aggregate["architecture"].eq("pure_mps_d64")
                        & aggregate["ordering"].eq(ordering)].iloc[0]
        tails.append(float(row["d64_schmidt_tail_sum"]))
    physical = pd.read_csv(source.HUB34_PHYSICAL_SECTORS)[["ansatz", "seed", "C_hidden"]]
    nnfmps = pd.read_csv(source.HUB34_NNFMPS_SECTORS)
    nnfmps = nnfmps[nnfmps["K"].eq(4096)][["ansatz", "seed", "C_hidden"]]
    physical = pd.concat([physical, nnfmps], ignore_index=True)
    physical["ordering"] = "physical"
    reordered = pd.read_csv(source.HUB34_MIDORDER_SECTORS)[["ansatz", "seed", "C_hidden"]]
    reordered["ordering"] = "random_05"
    cluster = pd.concat([physical, reordered], ignore_index=True)
    cluster["architecture"] = cluster["ansatz"].map(NAMES)
    burden = pd.read_csv(source.HUB34_MPS_BURDEN)
    physical_tail = float(burden.loc[burden["ordering"].eq("physical"), "schmidt_tail_sum"].iloc[0])

    plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8,
                         "ytick.labelsize": 8, "legend.fontsize": 8,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "axes.linewidth": 0.7})
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.9), sharey=True,
                             gridspec_kw={"width_ratios": [1.55, 1], "wspace": 0.14})
    labels = ["physical\n" + r"$3.96\!\times\!10^{-7}$", "random A\n" + f"{tails[1]:.3f}",
              "random B\n" + f"{tails[2]:.3f}", "even-odd\n" + f"{tails[3]:.2f}"]
    rows = plot_orders(axes[0], chain, orders, labels, r"(a) Open $L=12$ chain")
    rows += plot_orders(axes[1], cluster, ("physical", "random_05"),
                        ["original\n" + f"{physical_tail:.3f}", "random\n" + f"{source.HUB34_MIDORDER_BURDEN:.3f}"],
                        r"(b) Periodic $3\times4$ cluster")
    axes[0].set_ylabel(r"unseen agreement $C_U$")
    for axis in axes:
        axis.set_xlabel(r"site order and discarded weight $\mathcal{T}_{\rm Sch}^{(64)}$", labelpad=7)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", bbox_to_anchor=(0.51, 1.02),
               ncol=3, frameon=False, columnspacing=1.8)
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.25, top=0.82)
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(fig)
    AUDIT.mkdir(parents=True, exist_ok=True)
    report = {"purpose": "Re-layout of existing three-arm MPS comparisons; no new training",
              "input_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in inputs},
              "chain_D64_discarded_weights": dict(zip(orders, tails)),
              "cluster_D64_discarded_weights": {"physical": physical_tail,
                                                "random_05": source.HUB34_MIDORDER_BURDEN},
              "scores": rows}
    (AUDIT / "mps_figure_audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
