#!/usr/bin/env python3
"""Plot the candidate PRL Fig. 1 for the ansatz-structure narrative."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import prl_palette as palette

ROOT = Path(__file__).resolve().parents[1]
ZOO = ROOT / "analysis/fig1a_zoo_k4096"
SCALING = ROOT / "analysis/half_octave_completion_hub34_u4_5seeds"
NNFMPS = ROOT / "analysis/half_octave_completion_fig1_nnfmps_d64"
ANALYSIS_OUT = ROOT / "analysis/fig1_hubbard_generalization"
FIGURE_OUT = ROOT / "paper/figures/main/fig1_hubbard_generalization"

FIXED_K = 4096
R_MIN = 9.0
R_MAX = 19.5
BAND_WIDTH = 0.5
AREA_MIN = 12.0
AREA_MAX = 19.5
AREA_NORMALIZATION = AREA_MAX - AREA_MIN + BAND_WIDTH

ORDER = [
    "mlp_direct",
    "cnn_direct",
    "transformer_direct_phase",
    "nnfmps",
    "nnbf",
    "transformer_nnbf",
]
LABELS = {
    "mlp_direct": "direct MLP",
    "cnn_direct": "direct CNN",
    "transformer_direct_phase": "direct Transformer",
    "nnfmps": "NN-fMPS",
    "nnbf": "NNBF",
    "transformer_nnbf": "Transformer-NNBF",
}
SHORT_LABELS = {
    "mlp_direct": "direct MLP",
    "cnn_direct": "direct CNN",
    "transformer_direct_phase": "direct Transformer",
    "nnfmps": "NN-fMPS",
    "nnbf": "NNBF",
    "transformer_nnbf": "Transformer-NNBF",
}
COLORS = {
    "mlp_direct": palette.DIRECT,
    "cnn_direct": palette.DIRECT_LIGHT,
    "transformer_direct_phase": palette.DIRECT_MID,
    "nnfmps": palette.MPS,
    "visible_corrected_slater": palette.SCAFFOLD,
    "neural_jastrow": palette.NEURAL_CORRECTION,
    "nnbf": palette.NNBF,
    "transformer_nnbf": palette.NNBF_LIGHT,
}
LINESTYLES = {
    "mlp_direct": "-",
    "cnn_direct": "--",
    "transformer_direct_phase": ":",
    "nnfmps": "-",
    "visible_corrected_slater": "--",
    "neural_jastrow": "-",
    "nnbf": "-",
    "transformer_nnbf": "-",
}
MARKERS = {
    "mlp_direct": "o",
    "cnn_direct": "v",
    "transformer_direct_phase": "P",
    "nnfmps": "D",
    "visible_corrected_slater": "v",
    "neural_jastrow": "^",
    "nnbf": "s",
    "transformer_nnbf": "p",
}
FRONT_ORDER = [
    ("mlp_direct", "direct MLP"),
    ("nnfmps", "NN-fMPS"),
    ("nnbf", "NNBF"),
    ("transformer_nnbf", "Transformer-\nNNBF"),
]
BOUNDARY_COLOR = palette.BOUNDARY


def style_axes(ax: plt.Axes, *, grid: bool = True) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(direction="out", length=2.7, width=0.7)
    if grid:
        ax.grid(True, color="#D8D8D8", linewidth=0.45, alpha=0.62)


def panel_label(
    ax: plt.Axes, label: str, *, x: float = -0.12, y: float = 1.05
) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=9.0,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def profile_panel(ax: plt.Axes, bands: pd.DataFrame) -> None:
    selected = bands[
        bands["ansatz"].isin(ORDER)
        & bands["K"].eq(FIXED_K)
        & bands["r"].between(R_MIN, R_MAX)
    ].copy()
    for ansatz in ORDER:
        current = selected[selected["ansatz"].eq(ansatz)]
        summary = (
            current.groupby("r", as_index=False)["C_band"]
            .agg(["mean", "std", "count"])
            .reset_index()
            .sort_values("r")
        )
        if summary.empty:
            raise RuntimeError(f"missing K={FIXED_K} profile for {ansatz}")
        x = summary["r"].to_numpy(dtype=float)
        mean = summary["mean"].to_numpy(dtype=float)
        std = summary["std"].fillna(0.0).to_numpy(dtype=float)
        if np.any(std > 0.0):
            ax.fill_between(
                x,
                mean - std,
                mean + std,
                color=COLORS[ansatz],
                alpha=0.10,
                linewidth=0,
                zorder=1,
            )
        linewidth = 1.75 if ansatz in {"nnbf", "transformer_nnbf"} else 1.25
        line, = ax.plot(
            x,
            mean,
            color=COLORS[ansatz],
            linestyle=LINESTYLES[ansatz],
            marker=MARKERS[ansatz],
            markersize=3.1,
            markevery=2,
            linewidth=linewidth,
            label=LABELS[ansatz],
            zorder=3,
        )
        if ansatz == "transformer_direct_phase":
            line.set_markerfacecolor("none")
            line.set_markeredgewidth(0.8)

    # Mark the final fully visible band B11b (r=11.5); the next plotted
    # band, B12a at r=12, is the first fully unseen band.
    boundary = np.log2(FIXED_K) - BAND_WIDTH
    ax.axvspan(R_MIN, boundary, color="#D0D0D0", alpha=0.22, linewidth=0)
    ax.axvline(
        boundary,
        color=BOUNDARY_COLOR,
        linestyle=(0, (2.5, 2.0)),
        linewidth=0.9,
        zorder=2,
    )
    ax.axhline(0.0, color="#777777", linewidth=0.55, zorder=1)
    ax.set(
        xlim=(R_MIN - 0.1, R_MAX + 0.1),
        ylim=(-0.14, 1.045),
        xticks=[10, 12, 14, 16, 18],
        xlabel=r"target-ranked band $r$",
        ylabel=r"band agreement $C_B$",
    )
    ax.text(
        0.02,
        0.06,
        r"$K=4096$",
        transform=ax.transAxes,
        fontsize=8.8,
        color="#4A4A4A",
    )
    style_axes(ax)


def unseen_panel(ax: plt.Axes, sectors: pd.DataFrame) -> None:
    values = sectors.groupby("ansatz")["C_hidden"].mean()
    positions = {
        "mlp_direct": 7.0,
        "cnn_direct": 6.0,
        "transformer_direct_phase": 5.0,
        "nnfmps": 3.0,
        "nnbf": 1.0,
        "transformer_nnbf": 0.0,
    }
    ax.axhspan(
        4.5,
        7.5,
        color=COLORS["mlp_direct"],
        alpha=0.055,
        linewidth=0,
    )
    ax.axhspan(
        2.5, 3.5, color=COLORS["nnfmps"], alpha=0.055, linewidth=0
    )
    ax.axhspan(
        -0.5, 1.5, color=COLORS["nnbf"], alpha=0.055, linewidth=0
    )
    ax.axvline(0.0, color="#666666", linewidth=0.65)

    for ansatz in ORDER:
        value = float(values.loc[ansatz])
        y = positions[ansatz]
        ax.hlines(
            y,
            0.0,
            value,
            color=COLORS[ansatz],
            linewidth=1.25,
            alpha=0.72,
            zorder=2,
        )
        ax.scatter(
            [value],
            [y],
            color=COLORS[ansatz],
            marker=MARKERS[ansatz],
            s=24,
            edgecolors="white",
            linewidths=0.45,
            zorder=3,
        )
        place_right = ansatz == "transformer_nnbf" or value <= 0.92
        text_x = value + 0.018 if place_right else value - 0.014
        ax.text(
            text_x,
            y,
            f"{value:.3f}",
            fontsize=8.8,
            color=COLORS[ansatz],
            ha="left" if place_right else "right",
            va="center",
        )

    ax.set_yticks(
        [positions[ansatz] for ansatz in ORDER],
        [SHORT_LABELS[ansatz] for ansatz in ORDER],
    )
    ax.set(
        xlim=(-0.025, 1.075),
        ylim=(-0.65, 7.65),
        xticks=[0.0, 0.25, 0.5, 0.75, 1.0],
        xlabel=r"unseen agreement $C_U$",
    )
    # Aligned labels at the center of each shaded strip read as group headers,
    # not as annotations attached to individual points.
    for y, label, color in (
        (6.0, "Direct", COLORS["mlp_direct"]),
        (3.0, "MPS", COLORS["nnfmps"]),
        (0.5, "Determinant", COLORS["nnbf"]),
    ):
        ax.text(
            0.62,
            y,
            label,
            color=color,
            fontsize=8.0,
            fontweight="medium",
            ha="center",
            va="center",
        )
    style_axes(ax)
    ax.yaxis.grid(False)


def global_panel(ax: plt.Axes, sectors: pd.DataFrame) -> None:
    learned_order = ["mlp_direct", "nnfmps", "nnbf", "transformer_nnbf"]
    learned = sectors[sectors["ansatz"].isin(learned_order)].copy()
    for ansatz in learned_order:
        current = learned[learned["ansatz"].eq(ansatz)]
        for top_k, condition in current.groupby("K"):
            condition = condition.sort_values("seed")
            log_jitter = np.linspace(-0.032, 0.032, len(condition))
            ax.scatter(
                float(top_k) * np.power(2.0, log_jitter),
                condition["C_full"],
                color=COLORS[ansatz],
                marker=MARKERS[ansatz],
                s=10,
                linewidths=0,
                alpha=0.14,
                zorder=1,
            )
        mean = (
            current.groupby("K", as_index=False)["C_full"]
            .mean()
            .sort_values("K")
        )
        ax.plot(
            mean["K"],
            mean["C_full"],
            color=COLORS[ansatz],
            marker=MARKERS[ansatz],
            markersize=3.1,
            linewidth=1.5,
            label=LABELS[ansatz],
            zorder=3,
        )

    truncation = (
        learned.groupby("K", as_index=False)["W0"].first().sort_values("K")
    )
    ax.plot(
        truncation["K"],
        np.sqrt(truncation["W0"]),
        color="#222222",
        linestyle=(0, (2.5, 2.0)),
        linewidth=1.0,
        label="training-set truncation",
        zorder=2,
    )
    ax.set_xscale("log", base=2)
    ax.set(
        xlim=(430, 39000),
        ylim=(-0.035, 1.035),
        xticks=[512, 2048, 8192, 32768],
        xticklabels=[r"$2^9$", r"$2^{11}$", r"$2^{13}$", r"$2^{15}$"],
        xlabel=r"training-set size $K$",
        ylabel=r"full-state agreement $C_{\rm full}$",
    )
    style_axes(ax)
    ax.xaxis.grid(False)


def cell_edges(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 1:
        return np.asarray([values[0] - 0.25, values[0] + 0.25])
    mids = 0.5 * (values[:-1] + values[1:])
    return np.concatenate(
        (
            [values[0] - (mids[0] - values[0])],
            mids,
            [values[-1] + (values[-1] - mids[-1])],
        )
    )


def completion_map(
    ax: plt.Axes,
    bands: pd.DataFrame,
    ansatz: str,
    title: str,
    *,
    show_y: bool,
) -> object:
    current = bands[
        bands["ansatz"].eq(ansatz) & bands["r"].between(R_MIN, R_MAX)
    ].copy()
    mean = (
        current.groupby(["K", "r"], as_index=False)["C_band"]
        .mean()
        .assign(log2K=lambda frame: np.log2(frame["K"]))
    )
    x = np.sort(mean["r"].unique().astype(float))
    y = np.sort(mean["log2K"].unique().astype(float))
    matrix = (
        mean.pivot(index="log2K", columns="r", values="C_band")
        .reindex(index=y, columns=x)
        .to_numpy()
    )
    mesh = ax.pcolormesh(
        cell_edges(x),
        cell_edges(y),
        np.ma.masked_invalid(matrix),
        cmap="viridis",
        vmin=-0.1,
        vmax=1.0,
        shading="flat",
        rasterized=True,
    )
    ax.plot(
        [R_MIN, 15.5],
        [R_MIN, 15.5],
        color=BOUNDARY_COLOR,
        linewidth=0.85,
        zorder=4,
    )
    ax.set(
        xlim=(R_MIN, R_MAX + BAND_WIDTH),
        ylim=(8.75, 15.35),
        xticks=[10, 14, 18],
        yticks=[9, 11, 13, 15],
        title=title,
    )
    ax.title.set_fontsize(8.6)
    ax.title.set_linespacing(0.9)
    ax.tick_params(direction="out", length=2.2, width=0.6)
    for spine in ax.spines.values():
        spine.set_linewidth(0.65)
    if not show_y:
        ax.tick_params(axis="y", labelleft=False)
    ax.grid(False)
    return mesh


def fixed_k_decision(
    bands: pd.DataFrame,
    ansatz: str,
    parameter_count: int,
) -> dict[str, object]:
    unseen = bands[
        bands["ansatz"].eq(ansatz)
        & bands["K"].eq(FIXED_K)
        & bands["unseen"].astype(bool)
    ]
    summary = (
        unseen.groupby("r", as_index=False)
        .agg(C_mean=("C_band", "mean"), n_seed=("seed", "nunique"))
        .sort_values("r")
    )
    if summary.empty:
        raise RuntimeError(f"missing K={FIXED_K} unseen bands for {ansatz}")
    passing = summary[summary["C_mean"] >= 0.5]
    area = float(
        0.5
        * np.clip(
            summary.loc[
                summary["r"].between(AREA_MIN, AREA_MAX), "C_mean"
            ].to_numpy(),
            0.0,
            None,
        ).sum()
    )
    return {
        "ansatz": ansatz,
        "K": FIXED_K,
        "parameter_count": int(parameter_count),
        "n_seed": int(summary["n_seed"].max()),
        "first_fully_unseen_r": float(summary["r"].min()),
        "C_first_fully_unseen": float(summary.iloc[0]["C_mean"]),
        "ell_0p5": (
            float(passing["r"].max() - np.log2(FIXED_K))
            if not passing.empty
            else float("nan")
        ),
        "A_unseen_r12_to_r19p5": area,
    }


def main() -> None:
    ANALYSIS_OUT.mkdir(parents=True, exist_ok=True)
    FIGURE_OUT.parent.mkdir(parents=True, exist_ok=True)

    scaling_bands = pd.read_csv(SCALING / "half_octave_metrics.csv")
    scaling_sectors = pd.read_csv(SCALING / "sector_metrics.csv")
    nnfmps_bands = pd.read_csv(NNFMPS / "half_octave_metrics.csv").assign(
        ansatz="nnfmps"
    )
    nnfmps_sectors = pd.read_csv(NNFMPS / "sector_metrics.csv").assign(
        ansatz="nnfmps"
    )
    scaling_bands = pd.concat(
        [
            scaling_bands[~scaling_bands["ansatz"].eq("mps")],
            nnfmps_bands,
        ],
        ignore_index=True,
    )
    scaling_sectors = pd.concat(
        [
            scaling_sectors[~scaling_sectors["ansatz"].eq("mps")],
            nnfmps_sectors,
        ],
        ignore_index=True,
    )
    zoo_bands = pd.read_csv(ZOO / "band_metrics.csv")
    zoo_bands = zoo_bands[~zoo_bands["ansatz"].eq("mps")]
    fixed_nnfmps_bands = nnfmps_bands[nnfmps_bands["K"].eq(FIXED_K)]
    tnnbf_bands = scaling_bands[
        scaling_bands["ansatz"].eq("transformer_nnbf")
        & scaling_bands["K"].eq(FIXED_K)
    ].copy()
    zoo_bands = pd.concat(
        [zoo_bands, fixed_nnfmps_bands, tnnbf_bands],
        ignore_index=True,
    )
    decisions = pd.read_csv(ZOO / "decision_metrics.csv")
    decisions = decisions[~decisions["ansatz"].eq("mps")]
    decisions = pd.concat(
        [
            decisions,
            pd.DataFrame(
                [
                    fixed_k_decision(nnfmps_bands, "nnfmps", 658352),
                    fixed_k_decision(
                        scaling_bands, "transformer_nnbf", 652080
                    ),
                ]
            ),
        ],
        ignore_index=True,
    )
    front_bands = scaling_bands[
        scaling_bands["ansatz"].isin(
            ["mlp_direct", "nnfmps", "nnbf", "transformer_nnbf"]
        )
    ].copy()

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.8,
            "axes.labelsize": 8.8,
            "legend.fontsize": 8.6,
            "xtick.labelsize": 8.6,
            "ytick.labelsize": 8.6,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )

    fig = plt.figure(figsize=(7.2, 5.25))
    outer = fig.add_gridspec(
        2,
        12,
        left=0.075,
        right=0.982,
        bottom=0.11,
        top=0.85,
        wspace=0.78,
        hspace=0.48,
        height_ratios=[1.05, 1.0],
    )
    ax_global = fig.add_subplot(outer[0, :4])
    # Reserve one grid column as a true horizontal gutter.  Increasing
    # ``wspace`` alone does not protect the long y labels of the right-hand
    # panels because both axes span several GridSpec columns.
    ax_profile = fig.add_subplot(outer[0, 5:])
    ax_area = fig.add_subplot(outer[1, :3])
    front_grid = outer[1, 4:].subgridspec(1, 4, wspace=0.12)
    front_axes = [
        fig.add_subplot(front_grid[0, index]) for index in range(4)
    ]

    profile_panel(ax_profile, zoo_bands)
    fixed_k_sectors = pd.concat(
        [
            pd.read_csv(ZOO / "sector_metrics.csv")[
                lambda frame: frame["ansatz"].isin(
                    [
                        "mlp_direct",
                        "cnn_direct",
                        "transformer_direct_phase",
                        "nnbf",
                    ]
                )
            ],
            nnfmps_sectors[nnfmps_sectors["K"].eq(FIXED_K)],
            scaling_sectors[
                scaling_sectors["ansatz"].eq("transformer_nnbf")
                & scaling_sectors["K"].eq(FIXED_K)
            ],
        ],
        ignore_index=True,
    )
    unseen_panel(ax_area, fixed_k_sectors)
    global_panel(ax_global, scaling_sectors)
    meshes = []
    for index, (ansatz, title) in enumerate(FRONT_ORDER):
        meshes.append(
            completion_map(
                front_axes[index],
                front_bands,
                ansatz,
                title,
                show_y=index == 0,
            )
        )
    front_axes[0].set_ylabel(r"$\log_2 K$")
    for ax in front_axes:
        ax.set_xlabel(r"$r$")

    panel_label(ax_global, "(a)", x=-0.20)
    panel_label(ax_profile, "(b)", x=-0.10)
    panel_label(ax_area, "(c)", x=-0.30)
    panel_label(front_axes[0], "(d)", x=-0.38)

    handles, labels = ax_profile.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.51, 0.99),
        ncol=4,
        frameon=False,
        handlelength=2.3,
        columnspacing=1.1,
        handletextpad=0.45,
    )
    # Keep the long label out of the data and use a light leader to identify
    # the dashed prefix-truncation baseline unambiguously.
    ax_global.annotate(
        "prefix truncation",
        xy=(3072, 0.444),
        xytext=(18000, 0.315),
        fontsize=8.0,
        color="#222222",
        ha="center",
        va="bottom",
        arrowprops={
            "arrowstyle": "-",
            "color": "#555555",
            "linewidth": 0.65,
            "shrinkA": 2.0,
            "shrinkB": 2.0,
        },
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.88,
            "pad": 0.15,
        },
    )
    colorbar = fig.colorbar(
        meshes[-1],
        ax=front_axes,
        location="right",
        fraction=0.026,
        pad=0.015,
        aspect=22,
    )
    colorbar.set_label(r"$C_B$", labelpad=1.5)
    colorbar.set_ticks([0.0, 0.5, 1.0])
    fig.savefig(FIGURE_OUT.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(
        FIGURE_OUT.with_suffix(".png"),
        dpi=350,
        bbox_inches="tight",
    )
    plt.close(fig)

    unseen_summary = (
        fixed_k_sectors.groupby("ansatz", as_index=False)
        .agg(C_U_mean=("C_hidden", "mean"), C_U_std=("C_hidden", "std"))
        .sort_values(
            "ansatz",
            key=lambda column: column.map(
                {ansatz: index for index, ansatz in enumerate(ORDER)}
            ),
        )
    )
    unseen_summary.to_csv(
        ANALYSIS_OUT / "unseen_agreement.csv", index=False
    )
    report = {
        "figure": str(FIGURE_OUT.with_suffix(".pdf").relative_to(ROOT)),
        "fixed_K_panel_b_c": FIXED_K,
        "summary_metric": (
            "C_U: signed directional agreement on the full unseen sector"
        ),
        "front_ansatzes": [ansatz for ansatz, _ in FRONT_ORDER],
        "direct_transformer_status": (
            "optimization-aided favorable direct control; used only in the "
            "fixed-K architecture screen, not in the K-scaling panels"
        ),
        "fixed_k_transformer_nnbf_source": (
            "K=4096 rows from the same five-seed matched scaling data used "
            "for panels c-d"
        ),
    }
    (ANALYSIS_OUT / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
