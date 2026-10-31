#!/usr/bin/env python3
"""Plot the stochastic Born-sampled support robustness control for the SI."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "analysis/gate_f_nnfmps_20260802/seed_metrics.csv"
REFERENCE = ROOT / "reference/hub3x4_U4_t1_Nup6_Ndn6_pbc/psi0.npy"
OUT = ROOT / "paper/figures/main/figS_stochastic_born_support.pdf"
REPORT = ROOT / "analysis/gate_f_nnfmps_20260802/si_figure_report.json"

ORDER = [
    "mlp_direct",
    "neuralized_mps_d64_w18",
    "nnbf",
    "transformer_nnbf",
]
LABELS = {
    "mlp_direct": "direct MLP",
    "neuralized_mps_d64_w18": "NN-fMPS",
    "nnbf": "NNBF",
    "transformer_nnbf": "Transformer-NNBF",
}
COLORS = {
    "mlp_direct": "#4477AA",
    "neuralized_mps_d64_w18": "#228833",
    "nnbf": "#EE7733",
    "transformer_nnbf": "#F3A36B",
}
MARKERS = {
    "mlp_direct": "o",
    "neuralized_mps_d64_w18": "D",
    "nnbf": "s",
    "transformer_nnbf": "D",
}


def style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, color="#D8D8D8", linewidth=0.45, alpha=0.65)
    ax.tick_params(direction="out", length=2.8, width=0.7)


def main() -> None:
    frame = pd.read_csv(INPUT)
    frame = frame[frame["ansatz"].isin(ORDER)].copy()
    supports = sorted(int(value) for value in frame["K"].unique())
    if supports != [1024, 4096, 16384]:
        raise RuntimeError(f"unexpected stochastic-support grid: {supports}")
    expected = (
        frame.groupby(["ansatz", "K"])
        .size()
        .reindex(
            pd.MultiIndex.from_product(
                [ORDER, supports], names=["ansatz", "K"]
            )
        )
    )
    if not np.all(expected.to_numpy() == 6):
        raise RuntimeError(f"expected six fits per ansatz and K:\n{expected}")

    target = np.asarray(np.load(REFERENCE), dtype=np.float64)
    target /= np.linalg.norm(target)
    ranked_weights = np.sort(np.square(target))[::-1]
    ranked_mass = {
        support: float(np.sum(ranked_weights[:support])) for support in supports
    }

    plt.rcParams.update(
        {
            "font.size": 7.2,
            "axes.labelsize": 7.2,
            "xtick.labelsize": 6.6,
            "ytick.labelsize": 6.6,
            "legend.fontsize": 6.4,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, (ax_mass, ax_agreement) = plt.subplots(
        1,
        2,
        figsize=(7.05, 2.45),
        gridspec_kw={"width_ratios": [0.83, 1.55]},
    )
    x = np.arange(len(supports), dtype=float)

    mass_rows = frame[
        ["K", "support_seed", "support_probability_mass"]
    ].drop_duplicates()
    for index, support in enumerate(supports):
        current = mass_rows[mass_rows["K"].eq(support)].sort_values(
            "support_seed"
        )
        if current.shape[0] != 3:
            raise RuntimeError(
                f"expected three support seeds at K={support}, got {current.shape[0]}"
            )
        jitter = np.linspace(-0.045, 0.045, current.shape[0])
        ax_mass.scatter(
            index + jitter,
            current["support_probability_mass"],
            color="#777777",
            s=12,
            alpha=0.34,
            linewidths=0,
            zorder=2,
        )
    stochastic_mean = (
        mass_rows.groupby("K")["support_probability_mass"]
        .mean()
        .reindex(supports)
        .to_numpy()
    )
    ax_mass.plot(
        x,
        stochastic_mean,
        color="#555555",
        marker="o",
        markersize=3.5,
        linewidth=1.25,
        label="stochastic support",
        zorder=3,
    )
    ax_mass.plot(
        x,
        [ranked_mass[support] for support in supports],
        color="#111111",
        linestyle="--",
        marker="o",
        markersize=3.2,
        linewidth=1.1,
        label="target-ranked prefix",
        zorder=3,
    )
    ax_mass.set_ylabel("training-set target weight")
    ax_mass.set_ylim(0.0, 0.52)
    ax_mass.legend(frameon=False, loc="upper left", handlelength=1.7)
    style_axes(ax_mass)

    for ansatz_index, ansatz in enumerate(ORDER):
        current_ansatz = frame[frame["ansatz"].eq(ansatz)]
        mean_y: list[float] = []
        for support_index, support in enumerate(supports):
            current = current_ansatz[current_ansatz["K"].eq(support)].sort_values(
                ["support_seed", "init_seed"]
            )
            jitter = np.linspace(-0.055, 0.055, current.shape[0])
            ax_agreement.scatter(
                support_index + jitter,
                current["full_signed_cosine"],
                color=COLORS[ansatz],
                marker=MARKERS[ansatz],
                s=12,
                alpha=0.20,
                linewidths=0,
                zorder=1,
            )
            mean_y.append(float(current["full_signed_cosine"].mean()))
        ax_agreement.plot(
            x,
            mean_y,
            color=COLORS[ansatz],
            marker=MARKERS[ansatz],
            markersize=3.8,
            linewidth=1.35,
            label=LABELS[ansatz],
            zorder=3,
        )
    ax_agreement.axhline(0.0, color="#999999", linewidth=0.55, zorder=0)
    ax_agreement.set_ylabel(r"full directional agreement $C_{\rm full}$")
    ax_agreement.set_ylim(-0.055, 1.035)
    ax_agreement.legend(
        frameon=False,
        loc="upper left",
        ncol=2,
        columnspacing=0.8,
        handlelength=1.55,
    )
    style_axes(ax_agreement)

    ticklabels = [r"$2^{10}$", r"$2^{12}$", r"$2^{14}$"]
    for ax in (ax_mass, ax_agreement):
        ax.set_xticks(x)
        ax.set_xticklabels(ticklabels)
        ax.set_xlim(-0.23, len(supports) - 0.77)
        ax.set_xlabel(r"training-set size $K$")
        ax.xaxis.grid(False)
    ax_mass.text(
        -0.17,
        1.04,
        "(a)",
        transform=ax_mass.transAxes,
        fontweight="bold",
        fontsize=8.0,
    )
    ax_agreement.text(
        -0.11,
        1.04,
        "(b)",
        transform=ax_agreement.transAxes,
        fontweight="bold",
        fontsize=8.0,
    )

    fig.tight_layout(w_pad=1.65)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    plt.close(fig)

    aggregate = (
        frame.groupby(["ansatz", "K"], as_index=False)
        .agg(
            n_fit=("full_signed_cosine", "size"),
            visible_overlap_norm_mean=("visible_overlap_norm", "mean"),
            support_probability_mass_mean=("support_probability_mass", "mean"),
            C_full_mean=("full_signed_cosine", "mean"),
            C_full_std=("full_signed_cosine", "std"),
        )
        .sort_values(["ansatz", "K"])
    )
    payload = {
        "figure": str(OUT.relative_to(ROOT)).replace("\\", "/"),
        "input": str(INPUT.relative_to(ROOT)).replace("\\", "/"),
        "design": "three nested Born-sampled supports by two model initializations",
        "support_sizes": supports,
        "ranked_prefix_mass": {
            str(key): value for key, value in ranked_mass.items()
        },
        "aggregate": aggregate.to_dict(orient="records"),
    }
    REPORT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
