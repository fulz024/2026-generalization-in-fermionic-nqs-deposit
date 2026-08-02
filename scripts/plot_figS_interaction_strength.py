#!/usr/bin/env python3
"""Plot the final multi-U generalization control from its public table."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "analysis/figS_interaction_strength/sector_metrics.csv"
OUT = ROOT / "figures/figS_interaction_strength"

ORDER = ["mlp_direct", "nnfmps", "nnbf", "transformer_nnbf"]
LABELS = {
    "mlp_direct": "direct MLP",
    "nnfmps": "NN-fMPS",
    "nnbf": "NNBF",
    "transformer_nnbf": "Transformer-NNBF",
}
COLORS = {
    "mlp_direct": "#4477AA",
    "nnfmps": "#228833",
    "nnbf": "#EE7733",
    "transformer_nnbf": "#F3A36B",
}
MARKERS = {
    "mlp_direct": "o",
    "nnfmps": "D",
    "nnbf": "s",
    "transformer_nnbf": "p",
}


def load() -> pd.DataFrame:
    combined = pd.read_csv(INPUT)
    required = {"U", "ansatz", "seed", "K", "C_hidden", "D_hidden"}
    missing = required - set(combined.columns)
    if missing:
        raise RuntimeError(f"missing columns in {INPUT}: {sorted(missing)}")
    if set(combined["ansatz"].unique()) != set(ORDER):
        raise RuntimeError("unexpected ansatz set in multi-U table")
    expected_k = {512, 2048, 4096, 8192, 32768}
    for interaction in (2, 8):
        current = combined[
            combined["U"].eq(interaction) & combined["ansatz"].eq("nnfmps")
        ]
        if set(current["K"].astype(int)) != expected_k or current.shape[0] != 25:
            raise RuntimeError(f"incomplete NN-fMPS U={interaction} grid")
    current_u4 = combined[
        combined["U"].eq(4) & combined["ansatz"].eq("nnfmps")
    ]
    if current_u4["seed"].nunique() != 5:
        raise RuntimeError("incomplete production NN-fMPS U=4 grid")
    return combined


def style(axis: plt.Axes) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(True, color="#DDDDDD", linewidth=0.5, alpha=0.7)
    axis.tick_params(direction="out", length=3.0, width=0.7)


def main() -> None:
    frame = load()
    plt.rcParams.update(
        {
            "font.size": 8.0,
            "axes.labelsize": 8.0,
            "legend.fontsize": 7.7,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(3, 2, figsize=(7.15, 7.25), sharex="row", constrained_layout=True)
    for row, interaction in enumerate((2, 4, 8)):
        selected = frame[frame["U"].eq(interaction)]
        for column, (metric, label, ylim) in enumerate(
            (
                ("C_hidden", r"unseen directional agreement $C_U$", (-0.12, 1.04)),
                ("D_hidden", r"unseen mass agreement $D_U$", (-0.02, 1.04)),
            )
        ):
            axis = axes[row, column]
            for ansatz in ORDER:
                current = selected[selected["ansatz"].eq(ansatz)]
                axis.scatter(
                    current["K"],
                    current[metric],
                    color=COLORS[ansatz],
                    marker=MARKERS[ansatz],
                    s=10,
                    linewidths=0,
                    alpha=0.17,
                )
                mean = current.groupby("K", as_index=False)[metric].mean().sort_values("K")
                axis.plot(
                    mean["K"],
                    mean[metric],
                    color=COLORS[ansatz],
                    marker=MARKERS[ansatz],
                    markersize=3.8,
                    linewidth=1.5,
                    label=LABELS[ansatz],
                )
            axis.set_xscale("log", base=2)
            axis.set_ylim(ylim)
            axis.set_ylabel(label)
            axis.set_xlabel(r"training-set size $K$")
            axis.set_title(rf"$3\times4$ Hubbard, $U/t={interaction}$")
            style(axis)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.025))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
