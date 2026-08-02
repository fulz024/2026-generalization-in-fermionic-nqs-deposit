#!/usr/bin/env python3
"""Plot the equilibrium-H10 2x2 completion/VMC comparison."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TRUNCATION = ROOT / "analysis/figS_h10/h10_truncation_curve.csv"
DEMAND = ROOT / "analysis/figS_h10/h10_completion_demand.csv"
SUPERVISED = ROOT / "analysis/figS_h10/h10_supervised_completion.csv"
ENDPOINTS = ROOT / "analysis/figS_h10/h10_vmc_endpoints.csv"
OUT = ROOT / "figures/figS_h10_bridge"
OUT_TRUNCATION = ROOT / "figures/figS_h10_truncation_curve"

COLORS = {
    "canonical": "#2C6EAA",
    "lowdin": "#E76F31",
    "mlp_direct": "#4477AA",
    "nnbf": "#EE7733",
}
LABELS = {
    "canonical": "canonical",
    "lowdin": "Löwdin",
    "mlp_direct": "MLP",
    "nnbf": "NNBF",
}
MARKERS = {"mlp_direct": "o", "nnbf": "s"}
GROUPS = [
    ("canonical", "mlp_direct"),
    ("canonical", "nnbf"),
    ("lowdin", "mlp_direct"),
    ("lowdin", "nnbf"),
]
POSITIONS = np.array([0.0, 0.9, 2.6, 3.5])


def panel_label(ax: plt.Axes, label: str, title: str) -> None:
    ax.set_title(rf"$\mathbf{{{label}}}$  {title}", loc="left", pad=4)


def style_axes(ax: plt.Axes, *, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis=grid_axis, linewidth=0.35, color="0.90")


def validate_inputs() -> None:
    supervised = pd.read_csv(SUPERVISED)
    endpoints = pd.read_csv(ENDPOINTS)
    for basis, ansatz in GROUPS:
        fit_rows = supervised[
            supervised["basis"].eq(basis) & supervised["ansatz"].eq(ansatz)
        ]
        vmc_rows = endpoints[
            endpoints["basis"].eq(basis) & endpoints["ansatz"].eq(ansatz)
        ]
        if fit_rows.shape[0] != 5 or set(fit_rows["K"]) != {512}:
            raise ValueError(
                f"expected five K=512 fits for {basis}/{ansatz}, "
                f"found {fit_rows.shape[0]}"
            )
        if vmc_rows.shape[0] != 10 or set(vmc_rows["epoch"]) != {2000}:
            raise ValueError(
                f"expected ten epoch-2000 VMC runs for {basis}/{ansatz}, "
                f"found {vmc_rows.shape[0]}"
            )


def add_basis_context(ax: plt.Axes, *, include_weight: bool) -> None:
    demand = pd.read_csv(DEMAND).set_index("basis")
    for center, basis in ((0.45, "canonical"), (3.05, "lowdin")):
        label = LABELS[basis]
        if include_weight:
            outside = 100.0 * float(demand.loc[basis, "target_hidden_mass"])
            label += f"\noutside weight {outside:.1f}%"
        ax.text(
            center,
            0.98,
            label,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=6.1,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 0.3},
            zorder=6,
        )
    ax.axvline(1.75, color="0.82", linewidth=0.65)


def plot_supervised(ax: plt.Axes) -> None:
    rows = pd.read_csv(SUPERVISED)
    for position, (basis, ansatz) in zip(POSITIONS, GROUPS):
        current = rows[
            rows["basis"].eq(basis) & rows["ansatz"].eq(ansatz)
        ].sort_values("seed")
        jitter = np.linspace(-0.09, 0.09, current.shape[0])
        ax.scatter(
            position + jitter,
            current["C_hidden"],
            color=COLORS[ansatz],
            marker=MARKERS[ansatz],
            s=15,
            alpha=0.34,
            linewidths=0,
            zorder=2,
        )
        ax.scatter(
            position,
            current["C_hidden"].mean(),
            color=COLORS[ansatz],
            marker=MARKERS[ansatz],
            s=43,
            edgecolor="white",
            linewidth=0.7,
            zorder=4,
        )
    ax.axhline(0.0, color="0.65", linewidth=0.7)
    ax.set(
        xticks=POSITIONS,
        xticklabels=["MLP", "NNBF", "MLP", "NNBF"],
        ylabel="signed completion outside top 512",
        ylim=(-0.09, 0.91),
        xlim=(-0.42, 3.92),
    )
    style_axes(ax)
    add_basis_context(ax, include_weight=True)
    panel_label(ax, "a", r"Matched top-512 coefficient fits")


def plot_vmc_error(ax: plt.Axes) -> None:
    rows = pd.read_csv(ENDPOINTS)
    for position, (basis, ansatz) in zip(POSITIONS, GROUPS):
        current = rows[
            rows["basis"].eq(basis) & rows["ansatz"].eq(ansatz)
        ].sort_values("seed")
        jitter = np.linspace(-0.12, 0.12, current.shape[0])
        ax.scatter(
            position + jitter,
            current["energy_error"],
            color=COLORS[ansatz],
            marker=MARKERS[ansatz],
            s=15,
            alpha=0.34,
            linewidths=0,
            zorder=2,
        )
        mean = float(current["energy_error"].mean())
        ax.scatter(
            position,
            mean,
            color=COLORS[ansatz],
            marker=MARKERS[ansatz],
            s=47,
            edgecolor="white",
            linewidth=0.7,
            zorder=4,
        )
        ax.annotate(
            f"{mean:.3g}",
            (position, mean),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=5.8,
        )
    ax.set_yscale("log")
    ax.set(
        xticks=POSITIONS,
        xticklabels=["MLP", "NNBF", "MLP", "NNBF"],
        ylabel=r"exact energy error after 2000 epochs  [Ha]",
        xlim=(-0.42, 3.92),
        ylim=(1.5e-3, 1.5),
    )
    style_axes(ax)
    ax.axvline(1.75, color="0.82", linewidth=0.65)
    panel_label(ax, "b", "VMC after the same training budget")


def plot_supplement_truncation() -> None:
    rows = pd.read_csv(TRUNCATION)
    fig, ax = plt.subplots(figsize=(5.2, 3.35), constrained_layout=True)
    for basis, marker in (("canonical", "o"), ("lowdin", "s")):
        group = rows[rows["basis"].eq(basis)].sort_values("K")
        group = group[group["energy_error"] > 1.0e-10]
        ax.plot(
            group["log2_K"],
            group["energy_error"],
            color=COLORS[basis],
            marker=marker,
            markersize=3.2,
            linewidth=1.25,
            label=LABELS[basis],
        )
    ax.axvline(9.0, color="0.70", linestyle=":", linewidth=0.75)
    ax.set_yscale("log")
    ax.set(
        xlabel=r"retained target coefficients $\log_2K$",
        ylabel=r"exact truncation error $\Delta E$ [Ha]",
        xlim=(4.8, 15.2),
    )
    style_axes(ax)
    ax.legend(frameon=False)
    fig.savefig(OUT_TRUNCATION.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(OUT_TRUNCATION.with_suffix(".png"), dpi=320, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    validate_inputs()
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.5,
            "axes.titlesize": 8.0,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(2, 1, figsize=(3.45, 4.25), constrained_layout=True)
    plot_supervised(axes[0])
    plot_vmc_error(axes[1])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT.with_suffix(".png"), dpi=320, bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    plot_supplement_truncation()


if __name__ == "__main__":
    main()
