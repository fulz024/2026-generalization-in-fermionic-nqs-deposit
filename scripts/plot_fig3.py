#!/usr/bin/env python3
"""Plot the chemical-scale positive-negative structure gate in PRL Fig. 3."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

import prl_palette as palette


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "analysis/fig3_selected_spaces"
OUT = ROOT / "figures/fig3_selected_spaces"
COLORS = {
    "transformer_direct_phase": palette.DIRECT,
    "static_determinant": palette.PHYSICAL_BASELINE,
    "affine_backflow": palette.PHYSICAL_BASELINE_DARK,
    "nnbf": palette.NNBF,
    "teacher": palette.REFERENCE,
}
MARKERS = {
    "transformer_direct_phase": "o",
    "static_determinant": "v",
    "affine_backflow": "^",
    "nnbf": "s",
    "teacher": "D",
}
LINESTYLES = {
    "transformer_direct_phase": "-",
    "static_determinant": (0, (3.0, 1.6)),
    "affine_backflow": "-",
    "nnbf": "-",
}
MODEL_SERIES = (
    ("transformer_direct_phase", "direct Transformer"),
    ("nnbf", "NNBF"),
)
ENERGY_SERIES = (
    ("transformer_direct_phase", "direct Transformer"),
    ("static_determinant", "static 1-det"),
    ("affine_backflow", "affine BF 1-det"),
    ("nnbf", "NNBF"),
)
PROFILE_SERIES = (
    ("transformer_direct_phase", "direct Transformer"),
    ("static_determinant", "static 1-det"),
    ("affine_backflow", "affine BF 1-det"),
    ("nnbf", "NNBF"),
)
BOUNDARY_COLOR = palette.BOUNDARY


def read_csv(filename: str) -> pd.DataFrame:
    path = ANALYSIS / filename
    if not path.exists() or not path.stat().st_size:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def style_axes(ax: plt.Axes, *, grid_axis: str = "both") -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(direction="out", length=2.7, width=0.7)
    ax.grid(
        axis=grid_axis,
        color="#D8D8D8",
        linewidth=0.45,
        alpha=0.62,
        zorder=0,
    )


def panel_label(ax: plt.Axes, label: str, title: str) -> None:
    ax.set_title(rf"$\mathbf{{({label})}}$", loc="left", pad=4)


def profile_panel(
    ax: plt.Axes,
    bands: pd.DataFrame,
    reference_bands: pd.DataFrame,
    *,
    top_k: int,
    trusted_stop: int,
    label: str,
    title: str,
    xlim: tuple[float, float],
    supplied_fraction: float,
    scaffold_holdout: tuple[int, int, float],
    show_ylabel: bool,
) -> None:
    boundary = math.log2(top_k) - 0.5
    selected = bands[
        bands["rank_stop"].le(trusted_stop)
        & bands["r"].ge(boundary - 1.5)
        & (bands["rank_stop"] - bands["rank_start"] + 1).ge(32)
    ].copy()
    if selected.empty:
        raise RuntimeError(f"no profile bands survive for {title}")
    x_min, _ = xlim
    ax.axvspan(x_min, boundary, color="#A8A8A8", alpha=0.10, linewidth=0)
    ax.axvline(
        boundary,
        color=BOUNDARY_COLOR,
        linewidth=0.9,
        linestyle=(0, (2.5, 2.0)),
        zorder=2,
    )

    teacher = reference_bands[
        reference_bands["rank_stop"].le(trusted_stop)
        & reference_bands["r"].ge(xlim[0])
    ].sort_values("r")
    if teacher.empty:
        raise RuntimeError(f"missing teacher agreement rows for {title}")
    ax.plot(
        teacher["r"],
        teacher["teacher_teacher_directional_agreement"],
        color="0.45",
        linewidth=0.85,
        linestyle=(0, (1.8, 1.5)),
        alpha=0.48,
        zorder=2,
    )

    for ansatz, ansatz_label in PROFILE_SERIES:
        rows = selected[selected["ansatz"].eq(ansatz)]
        if rows.empty:
            raise RuntimeError(f"missing {ansatz} profile rows for {title}")
        if rows["seed"].nunique() > 1:
            for _, seed_rows in rows.groupby("seed"):
                seed_rows = seed_rows.sort_values("r")
                ax.plot(
                    seed_rows["r"],
                    seed_rows["band_directional_agreement"],
                    color=COLORS[ansatz],
                    linewidth=0.55,
                    alpha=0.18,
                    zorder=1,
                )
        summary = (
            rows.groupby("r")["band_directional_agreement"]
            .agg(["mean", "std"])
            .reset_index()
            .sort_values("r")
        )
        x = summary["r"].to_numpy(dtype=float)
        mean = summary["mean"].to_numpy(dtype=float)
        std = summary["std"].fillna(0.0).to_numpy(dtype=float)
        ax.fill_between(
            x,
            mean - std,
            mean + std,
            color=COLORS[ansatz],
            alpha=0.10,
            linewidth=0,
            zorder=1,
        )
        ax.plot(
            x,
            mean,
            color=COLORS[ansatz],
            marker=MARKERS[ansatz],
            markersize=3.0,
            linewidth=1.45,
            linestyle=LINESTYLES[ansatz],
            label=ansatz_label,
            zorder=4,
        )

    ax.axhline(0.0, color="#777777", linewidth=0.55, zorder=1)
    ax.set(
        xlim=xlim,
        ylim=(-0.16, 1.045),
        xticks=list(
            range(
                int(math.ceil(xlim[0])),
                int(math.floor(xlim[1])) + 1,
            )
        ),
        xlabel=r"DMRG-ranked band $r$",
        ylabel=r"band agreement $C_B$" if show_ylabel else "",
    )
    if not show_ylabel:
        ax.tick_params(axis="y", labelleft=False)
    start, stop, scaffold_c = scaffold_holdout
    ax.text(
        0.97,
        0.42,
        f"prefix weight: {supplied_fraction:.1%}",
        transform=ax.transAxes,
        fontsize=8.0,
        color="#444444",
        ha="right",
        va="top",
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.80,
            "pad": 0.15,
        },
    )
    style_axes(ax)
    panel_label(ax, label, title)


def k_scan_facet(
    ax: plt.Axes,
    scan: pd.DataFrame,
    *,
    system_label: str,
    holdout_label: str,
    show_ylabel: bool,
) -> None:
    models = scan[
        scan["top_k"].gt(0)
        & scan["ansatz"].isin([ansatz for ansatz, _ in MODEL_SERIES])
    ].copy()
    top_ks = sorted(int(value) for value in models["top_k"].unique())
    if len(top_ks) < 3:
        raise RuntimeError(f"incomplete K scan for {system_label}")
    offsets = {
        "transformer_direct_phase": 2.0**-0.035,
        "nnbf": 2.0**0.035,
    }
    for ansatz, ansatz_label in MODEL_SERIES:
        rows = models[models["ansatz"].eq(ansatz)]
        ax.scatter(
            rows["top_k"] * offsets[ansatz],
            rows["common_directional_agreement"],
            color=COLORS[ansatz],
            marker=MARKERS[ansatz],
            s=10,
            alpha=0.25,
            linewidths=0,
            zorder=2,
        )
        summary = (
            rows.groupby("top_k")["common_directional_agreement"]
            .agg(["mean", "std"])
            .reset_index()
            .sort_values("top_k")
        )
        ax.plot(
            summary["top_k"],
            summary["mean"],
            color=COLORS[ansatz],
            marker=MARKERS[ansatz],
            markersize=3.2,
            linewidth=1.35,
            label=ansatz_label,
            zorder=4,
        )
    ax.axhline(0.0, color="#777777", linewidth=0.55, zorder=1)
    ax.set_xscale("log", base=2)
    ax.set(
        xlim=(top_ks[0] / 1.18, top_ks[-1] * 1.18),
        ylim=(-0.16, 1.045),
        xticks=top_ks,
        xticklabels=[
            str(value) if index in (0, 2, len(top_ks) - 1) else ""
            for index, value in enumerate(top_ks)
        ],
        xlabel=r"training-set size $K$",
        ylabel="agreement on fixed held-out ranks" if show_ylabel else "",
    )
    ax.text(
        0.05,
        0.95,
        holdout_label,
        transform=ax.transAxes,
        fontsize=8.3,
        color="#444444",
        ha="left",
        va="top",
    )
    ax.set_title(system_label, pad=3)
    style_axes(ax)
    tick_labels = ax.get_xticklabels()
    if tick_labels:
        tick_labels[0].set_ha("left")
        tick_labels[-1].set_ha("right")


def holdout_label(start: int, stop: int) -> str:
    if stop >= 1_000_000:
        return f"held-out ranks\n{start} to 1.05M"
    return f"held-out ranks\n{start} to {stop}"


def energy_panel(
    ax: plt.Axes,
    energy: pd.DataFrame,
    *,
    top_k: int,
    subspace_count: int,
) -> None:
    if energy.empty:
        raise RuntimeError("missing Fe cumulative energy table")
    energy_column = "energy"
    finite_energy = energy[energy_column].to_numpy(dtype=float)
    finite_energy = finite_energy[np.isfinite(finite_energy)]
    if finite_energy.size == 0:
        raise RuntimeError("Fe cumulative energy table has no finite values")
    lower_limit = 0.05 * math.floor(
        (float(np.min(finite_energy)) - 0.01) / 0.05
    )
    upper_limit = 0.05 * math.ceil(
        (float(np.max(finite_energy)) + 0.02) / 0.05
    )
    teacher = (
        energy[energy["ansatz"].eq("teacher")]
        .groupby("rank_stop", as_index=False)[energy_column]
        .mean()
        .sort_values("rank_stop")
    )
    rank_stops = sorted(int(value) for value in energy["rank_stop"].unique())
    if top_k not in rank_stops or subspace_count not in rank_stops:
        raise RuntimeError("energy panel is missing its K or P endpoint")
    visible_fraction = 0.27
    minimum_rank = rank_stops[0]

    def support_coordinate(values: object) -> np.ndarray:
        ranks = np.asarray(values, dtype=float)
        coordinates = np.empty_like(ranks)
        visible = ranks <= top_k
        visible_denominator = math.log2(top_k / minimum_rank)
        coordinates[visible] = visible_fraction * (
            np.log2(ranks[visible] / minimum_rank)
            / visible_denominator
        )
        coordinates[~visible] = visible_fraction + (
            1.0 - visible_fraction
        ) * (
            (ranks[~visible] - top_k)
            / (subspace_count - top_k)
        )
        return coordinates

    ax.axvspan(
        0.0,
        visible_fraction,
        color="#A8A8A8",
        alpha=0.10,
        linewidth=0,
        zorder=0,
    )
    ax.axvline(
        visible_fraction,
        color=BOUNDARY_COLOR,
        linewidth=0.9,
        linestyle=(0, (2.5, 2.0)),
        zorder=2,
    )
    for ansatz, ansatz_label in ENERGY_SERIES:
        rows = energy[energy["ansatz"].eq(ansatz)]
        if rows["seed"].nunique() > 1:
            for _, seed_rows in rows.groupby("seed"):
                seed_rows = seed_rows.sort_values("rank_stop")
                ax.plot(
                    support_coordinate(seed_rows["rank_stop"]),
                    seed_rows[energy_column],
                    color=COLORS[ansatz],
                    linewidth=0.55,
                    alpha=0.18,
                    zorder=1,
                )
                ax.scatter(
                    support_coordinate(seed_rows["rank_stop"]),
                    seed_rows[energy_column],
                    color=COLORS[ansatz],
                    marker=MARKERS[ansatz],
                    s=9,
                    alpha=0.18,
                    linewidths=0,
                    zorder=2,
                )
        summary = (
            rows.groupby("rank_stop")[energy_column]
            .agg(["mean", "std"])
            .reset_index()
            .sort_values("rank_stop")
        )
        ax.plot(
            support_coordinate(summary["rank_stop"]),
            summary["mean"],
            color=COLORS[ansatz],
            marker=MARKERS[ansatz],
            markersize=3.4,
            linewidth=1.45,
            linestyle=LINESTYLES[ansatz],
            label=ansatz_label,
            zorder=4,
        )
    ax.plot(
        support_coordinate(teacher["rank_stop"]),
        teacher[energy_column],
        color=COLORS["teacher"],
        marker=MARKERS["teacher"],
        markersize=3.0,
        linewidth=1.0,
        linestyle=(0, (2.0, 1.6)),
        zorder=3,
    )
    visible_ticks = [minimum_rank, top_k]
    unseen_ticks = [
        value
        for value in rank_stops
        if value > top_k
        and value in (top_k + (subspace_count - top_k) // 2, subspace_count)
    ]
    labeled_ticks = visible_ticks + unseen_ticks
    ax.set(
        xlim=(-0.02, 1.02),
        xticks=support_coordinate(labeled_ticks),
        xticklabels=[str(value) for value in labeled_ticks],
        xlabel=r"support $R$",
        ylabel=r"$E_R[m^{(K)}]$  (Ha)",
        ylim=(lower_limit, upper_limit),
    )
    ax.get_xticklabels()[-1].set_ha("right")
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    ax.text(
        0.04,
        0.92,
        rf"$K={top_k}$",
        transform=ax.transAxes,
        fontsize=8.3,
        color="#444444",
        ha="left",
        va="top",
    )
    ax.text(
        0.97,
        0.035,
        "DMRG prefix",
        transform=ax.transAxes,
        fontsize=8.3,
        color=COLORS["teacher"],
        ha="right",
        va="bottom",
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.78,
            "pad": 0.2,
        },
    )
    style_axes(ax)
    panel_label(ax, "d", "")


def main() -> None:
    global ANALYSIS, OUT
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=ANALYSIS,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUT,
        help="Output path without extension.",
    )
    args = parser.parse_args()
    ANALYSIS = args.analysis_dir
    OUT = args.output
    report = json.loads((ANALYSIS / "report.json").read_text(encoding="utf-8"))
    fe_bands = read_csv("fe_band_metrics.csv")
    n2_bands = read_csv("n2_band_metrics.csv")
    fe_reference_bands = read_csv("fe_reference_bands.csv")
    n2_reference_bands = read_csv("n2_reference_bands.csv")
    fe_scan = read_csv("fe_k_scan.csv")
    n2_scan = read_csv("n2_k_scan.csv")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.3,
            "axes.labelsize": 8.3,
            "axes.titlesize": 8.4,
            "legend.fontsize": 8.3,
            "xtick.labelsize": 8.3,
            "ytick.labelsize": 8.3,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig = plt.figure(figsize=(3.40, 6.10))
    outer = fig.add_gridspec(
        3,
        1,
        left=0.20,
        right=0.985,
        bottom=0.075,
        top=0.875,
        hspace=0.60,
        height_ratios=(1.0, 1.0, 1.10),
    )
    ax_fe = fig.add_subplot(outer[0, 0])
    ax_n2 = fig.add_subplot(outer[1, 0], sharey=ax_fe)
    scan_grid = outer[2, 0].subgridspec(1, 2, wspace=0.28)
    ax_scan_fe = fig.add_subplot(scan_grid[0, 0])
    ax_scan_n2 = fig.add_subplot(scan_grid[0, 1], sharey=ax_scan_fe)

    scaffold = report["static_scaffold"]
    profile_top_k = int(report["fe_top_k"])
    n2_profile_top_k = int(report["n2_top_k"])
    fe_display_rank_stop = int(
        report.get(
            "fe_display_rank_stop",
            report["fe_reference"]["trusted_rank_stop"],
        )
    )
    fe_xlim = (
        math.log2(profile_top_k) - 1.65,
        math.log2(fe_display_rank_stop) - 0.30,
    )
    n2_xlim = (
        math.log2(n2_profile_top_k) - 1.65,
        math.log2(int(report["n2_reference"]["trusted_rank_stop"])) - 0.30,
    )
    profile_panel(
        ax_fe,
        fe_bands,
        fe_reference_bands,
        top_k=profile_top_k,
        trusted_stop=int(
            fe_display_rank_stop
        ),
        label="a",
        title=r"Fe$_2$S$_2$: determinant alignment beyond $K$",
        xlim=fe_xlim,
        supplied_fraction=float(
            report["fe_supplied_fraction_of_selected_pool"]
        ),
        scaffold_holdout=(
            int(scaffold["fe_profile_holdout"][0]),
            int(scaffold["fe_profile_holdout"][1]),
            float(scaffold["fe_profile_holdout_agreement"]),
        ),
        show_ylabel=True,
    )
    ax_fe.set_xlabel("")
    n2_reference = report["n2_reference"]
    profile_panel(
        ax_n2,
        n2_bands,
        n2_reference_bands,
        top_k=n2_profile_top_k,
        trusted_stop=int(n2_reference["trusted_rank_stop"]),
        label="b",
        title=r"N$_2$: little determinant alignment beyond $K$",
        xlim=n2_xlim,
        supplied_fraction=float(
            n2_reference["supplied_fraction_of_selected_pool"]
        ),
        scaffold_holdout=(
            int(scaffold["n2_profile_holdout"][0]),
            int(scaffold["n2_profile_holdout"][1]),
            float(scaffold["n2_profile_holdout_agreement"]),
        ),
        show_ylabel=True,
    )
    k_scan_facet(
        ax_scan_fe,
        fe_scan,
        system_label=r"Fe$_2$S$_2$",
        holdout_label=holdout_label(*report["fe_common_holdout"]),
        show_ylabel=True,
    )
    k_scan_facet(
        ax_scan_n2,
        n2_scan,
        system_label=r"N$_2$",
        holdout_label=holdout_label(*report["n2_common_holdout"]),
        show_ylabel=False,
    )
    ax_scan_n2.tick_params(axis="y", labelleft=False)
    ax_scan_fe.text(
        -0.24,
        1.14,
        r"$\mathbf{(c)}$",
        transform=ax_scan_fe.transAxes,
        fontsize=8.4,
        ha="left",
        va="bottom",
    )
    handles = [
        Line2D(
            [0],
            [0],
            color=COLORS[ansatz],
            marker=MARKERS[ansatz],
            linestyle=LINESTYLES[ansatz],
            linewidth=1.4,
            markersize=3.5,
            label=label,
        )
        for ansatz, label in PROFILE_SERIES
    ]
    handles.append(
        Line2D(
            [0],
            [0],
            color="0.45",
            linestyle=(0, (1.8, 1.5)),
            linewidth=0.85,
            alpha=0.55,
            label="DMRG reference",
        )
    )
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.005),
        ncol=2,
        frameon=False,
        fontsize=7.4,
        handlelength=1.8,
        columnspacing=0.95,
        labelspacing=0.30,
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT.with_suffix(".pdf"))
    plt.close(fig)


if __name__ == "__main__":
    main()
