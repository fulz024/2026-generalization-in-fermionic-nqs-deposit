#!/usr/bin/env python3
"""Build the prospective PRL ansatz-structure mechanism figure."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import prl_palette as palette


ROOT = Path(__file__).resolve().parents[1]
COMPONENTS = ROOT / "analysis/fig2_visible_corrected_components"
HEADS = ROOT / "analysis/fig2_exact_nnbf_heads_summary"
ATTRIBUTION = (
    ROOT / "analysis/s3_main_checkpoint_attribution/band_attribution.csv"
)
MPS = ROOT / "analysis/fig2c_mps_neuralization_ordering_20260730"
MPBS = ROOT / "analysis/fig2c_mpbs_all_neighbors_20260731"
MPS_ACTIVE_REPLICATION = (
    ROOT
    / "analysis/fig2c_active_mps_carrier_pilot_20260731"
    / "matched_arm_replication"
)
HUB34_MPS_BURDEN = (
    ROOT
    / "analysis/fig2c_active_mps_carrier_pilot_20260731"
    / "hub34_d64_compressibility"
    / "order_compressibility.csv"
)
HUB34_MIDORDER_BANDS = (
    ROOT / "analysis/fig2d_hub34_midorder_20260731/half_octave_metrics.csv"
)
HUB34_MIDORDER_SECTORS = (
    ROOT / "analysis/fig2d_hub34_midorder_20260731/sector_metrics.csv"
)
HUB34_PHYSICAL_SECTORS = (
    ROOT / "analysis/fig2c_hub34_component_replication_20260731/sector_metrics.csv"
)
HUB34_NNFMPS_SECTORS = (
    ROOT / "analysis/half_octave_completion_fig1_nnfmps_d64/sector_metrics.csv"
)
HUB34_MPBS = (
    ROOT
    / "analysis"
    / "fig2d_hub34_mpbs_pair_resolved_20260731"
    / "per_seed_metrics.csv"
)
HUB34_MIDORDER_BURDEN = 0.5833356945306273
# Main-text representatives are chosen from the target-only D=64 Schmidt-tail
# burden: the least disrupted random order, an intermediate random order, and
# the deterministic even--odd extreme.  The complete ordering scan remains in
# the source tables and Supplemental analysis.
MAIN_MPS_ORDERINGS = ("physical", "random_04", "random_05", "even_odd")
H10_COMPLETION = ROOT / "analysis/vmc_fig4/h10_supervised_completion.csv"
H10_DEMAND = ROOT / "analysis/vmc_fig4/h10_completion_demand.csv"
OUT = ROOT / "paper/figures/main/fig2_structure_mechanism"


COLORS = {
    "direct": palette.DIRECT,
    "nnbf": palette.NNBF,
    "determinant_component": palette.PHYSICAL_BASELINE,
    "symmetric": palette.SYMMETRIC,
    "jastrow": palette.JASTROW,
    "aux": palette.PHYSICAL_BASELINE,
    "aux_dark": palette.PHYSICAL_BASELINE_DARK,
    "green": palette.MPS,
    "black": palette.REFERENCE,
    "light": palette.NULL,
    "lattice_outline": "#882255",
}


def panel_label(ax: plt.Axes, label: str, title: str) -> None:
    ax.set_title(rf"$\mathbf{{({label})}}$", loc="left", pad=4)


def style_axes(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(direction="out")


def plot_readout_component_map(ax: plt.Axes) -> None:
    per_seed = pd.read_csv(COMPONENTS / "per_seed_metrics.csv")
    readout_seed = pd.read_csv(HEADS / "per_seed_metrics.csv")
    categories = [
        "additive\n/ MLP",
        "product",
        "det.\n/ NNBF",
        "neural\nJastrow",
        "Fermi\nsea",
        "simple\nBF",
    ]
    component_specs = (
        (0.00, "visible_corrected_direct_mlp", COLORS["direct"]),
        (2.00, "visible_corrected_nnbf", COLORS["nnbf"]),
        (3.00, "visible_corrected_neural_jastrow", COLORS["jastrow"]),
        (4.00, "visible_corrected_slater", COLORS["aux"]),
        (5.00, "visible_corrected_simple_backflow", COLORS["aux_dark"]),
    )
    readout_specs = (
        (-0.15, "global additive", COLORS["direct"]),
        (0.15, "occupied additive", palette.DIRECT_LIGHT),
        (1.00, "symmetric product", COLORS["symmetric"]),
        (2.12, "determinant", COLORS["nnbf"]),
    )

    def add_series(
        x_value: float,
        values: np.ndarray,
        color: str,
        marker: str,
        filled: bool,
    ) -> None:
        jitter = (
            np.linspace(-0.045, 0.045, values.size)
            if values.size > 1
            else np.zeros(1)
        )
        ax.scatter(
            x_value + jitter,
            values,
            s=12,
            marker=marker,
            facecolor=color if filled else "white",
            edgecolor=color,
            linewidth=0.55,
            alpha=0.24,
            zorder=2,
        )
        ax.scatter(
            [x_value],
            [float(np.mean(values))],
            s=29 if filled else 25,
            marker=marker,
            facecolor=color if filled else "white",
            edgecolor=color,
            linewidth=1.05,
            zorder=4,
        )

    for x_value, name, color in component_specs:
        values = per_seed.loc[
            per_seed["component"].eq(name), "C_hidden"
        ].to_numpy(dtype=float)
        add_series(x_value, values, color, "o", False)
    for x_value, name, color in readout_specs:
        values = readout_seed.loc[
            readout_seed["label"].eq(name), "C_hidden"
        ].to_numpy(dtype=float)
        add_series(x_value, values, color, "D", True)

    ax.legend(
        handles=[
            Line2D(
                [],
                [],
                marker="D",
                linestyle="None",
                markersize=4.8,
                markerfacecolor="0.45",
                markeredgecolor="0.45",
                label="readout swap",
            ),
            Line2D(
                [],
                [],
                marker="o",
                linestyle="None",
                markersize=4.8,
                markerfacecolor="white",
                markeredgecolor="0.45",
                label="separate fit / probe",
            ),
        ],
        loc="center",
        bbox_to_anchor=(0.64, 0.48),
        frameon=True,
        facecolor="white",
        edgecolor="0.82",
        framealpha=0.94,
        fontsize=7.0,
        handletextpad=0.45,
        labelspacing=0.35,
        borderpad=0.35,
    )
    ax.axhline(0.0, color="0.72", linewidth=0.7)
    ax.set(
        ylabel=r"unseen agreement $C_U$",
        xticks=np.arange(len(categories), dtype=float),
        xticklabels=categories,
        xlim=(-0.45, 5.45),
        ylim=(-0.04, 1.02),
        yticks=[0.0, 0.5, 1.0],
    )
    ax.tick_params(axis="x", labelsize=7.4, pad=1.2)
    panel_label(ax, "a", "Unseen agreement across amplitude maps")
    style_axes(ax)


def plot_determinant_component_attribution(ax: plt.Axes) -> None:
    bands = pd.read_csv(ATTRIBUTION)
    rows = bands[
        bands["K"].eq(4096)
        & bands["strict_unseen"].astype(bool)
        & bands["r"].between(13.0, 19.0)
    ].copy()
    for column, label, color in (
        ("C_total", r"NNBF total", COLORS["nnbf"]),
        (
            "C_scaffold_carried",
            r"Slater-aligned part",
            COLORS["determinant_component"],
        ),
    ):
        for _, seed_rows in rows.groupby("seed"):
            seed_rows = seed_rows.sort_values("r")
            ax.plot(
                seed_rows["r"],
                seed_rows[column],
                color=color,
                alpha=0.11,
                linewidth=0.55,
            )
        mean = rows.groupby("r", as_index=False)[column].mean().sort_values("r")
        ax.plot(
            mean["r"],
            mean[column],
            color=color,
            marker="o" if column == "C_total" else "s",
            markersize=3.1,
            linewidth=1.25,
            label=label,
        )
    broad = rows[rows["r"].between(13.0, 16.5)]
    broad_share = float(
        broad["C_scaffold_carried"].sum() / broad["C_total"].sum()
    )
    if not np.isclose(broad_share, 0.994, atol=0.001):
        raise ValueError(f"unexpected broad-region contribution: {broad_share}")
    ax.axvspan(13.0, 16.5, color=COLORS["light"], alpha=0.06)
    ax.axhline(0.0, color="0.72", linewidth=0.7)
    ax.set(
        xlabel="target-ranked band $r$",
        ylabel=r"band agreement $C_B$",
        xlim=(12.8, 19.2),
        ylim=(-0.04, 1.04),
        yticks=[0, 0.5, 1],
    )
    ax.legend(
        frameon=True,
        facecolor="white",
        edgecolor="0.82",
        framealpha=0.94,
        fontsize=7.6,
        loc="lower left",
        handlelength=1.6,
    )
    panel_label(ax, "b", "NNBF agreement follows its Slater-aligned component")
    style_axes(ax)


def plot_mps_ordering(ax: plt.Axes) -> None:
    per_seed = pd.read_csv(MPS / "per_seed_metrics.csv")
    aggregate = pd.read_csv(MPS / "aggregate_metrics.csv")
    mpbs_per_seed = pd.read_csv(MPBS / "per_seed_metrics.csv").drop(
        columns=["d64_schmidt_tail_sum", "order"],
        errors="ignore",
    )
    per_seed = pd.concat([per_seed, mpbs_per_seed], ignore_index=True)
    aggregate = pd.concat(
        [aggregate, pd.read_csv(MPBS / "aggregate_metrics.csv")],
        ignore_index=True,
    )
    hub34_rows = pd.read_csv(HUB34_PHYSICAL_SECTORS)
    hub34_rows = hub34_rows[["ansatz", "seed", "C_hidden"]].copy()
    hub34_nnf = pd.read_csv(HUB34_NNFMPS_SECTORS)
    hub34_nnf = hub34_nnf[hub34_nnf["K"].eq(4096)][
        ["ansatz", "seed", "C_hidden"]
    ].copy()
    hub34_rows = pd.concat([hub34_rows, hub34_nnf], ignore_index=True)
    hub34_rows["architecture"] = hub34_rows["ansatz"].map(
        {
            "mps_d64_nested_control": "pure_mps_d64",
            "neuralized_mps_scan_d1_w282": "neural_product_d1",
            "neuralized_mps_d64_w18": "nn_fmps_d64",
        }
    )
    hub34_mpbs_rows = pd.DataFrame()
    if HUB34_MPBS.exists():
        hub34_mpbs_rows = pd.read_csv(HUB34_MPBS)
        hub34_rows = pd.concat(
            [
                hub34_rows,
                hub34_mpbs_rows[
                    hub34_mpbs_rows["ordering"].eq("physical")
                ][["architecture", "seed", "C_hidden"]],
            ],
            ignore_index=True,
        )
    hub34_burden_rows = pd.read_csv(HUB34_MPS_BURDEN)
    hub34_burden = float(
        hub34_burden_rows.loc[
            hub34_burden_rows["ordering"].eq("physical"),
            "schmidt_tail_sum",
        ].iloc[0]
    )
    hub34_midorder_rows = pd.DataFrame()
    if HUB34_MIDORDER_SECTORS.exists():
        hub34_midorder_rows = pd.read_csv(HUB34_MIDORDER_SECTORS)[
            ["ansatz", "seed", "C_hidden"]
        ].copy()
        hub34_midorder_rows["architecture"] = hub34_midorder_rows["ansatz"].map(
            {
                "mps_d64_nested_control": "pure_mps_d64",
                "neuralized_mps_scan_d1_w282": "neural_product_d1",
                "neuralized_mps_d64_w18": "nn_fmps_d64",
            }
        )
    if not hub34_mpbs_rows.empty:
        hub34_midorder_rows = pd.concat(
            [
                hub34_midorder_rows,
                hub34_mpbs_rows[
                    hub34_mpbs_rows["ordering"].eq("random_05")
                ][["architecture", "seed", "C_hidden"]],
            ],
            ignore_index=True,
        )
    disrupted_min = 0.09
    disrupted_max = 2.4

    def broken_log_coordinate(values: pd.Series | np.ndarray) -> np.ndarray:
        numeric = np.asarray(values, dtype=float)
        return 1.0 + 3.0 * (
            np.log10(numeric) - np.log10(disrupted_min)
        ) / (np.log10(disrupted_max) - np.log10(disrupted_min))

    styles = {
        "pure_mps_d64": (
            palette.PHYSICAL_BASELINE,
            "s",
            r"pure MPS",
            ":",
            "white",
        ),
        "nn_fmps_d64": (
            palette.MPS,
            "o",
            r"NN-fMPS",
            "-",
            palette.MPS,
        ),
        "mpbs_d64": (
            palette.PHYSICAL_BASELINE_DARK,
            "v",
            r"MPBS",
            "-.",
            palette.PHYSICAL_BASELINE_DARK,
        ),
        "neural_product_d1": (
            palette.DIRECT_MID,
            "D",
            r"neural, $D=1$",
            "--",
            "white",
        ),
    }
    for architecture, (color, marker, label, linestyle, facecolor) in styles.items():
        means = aggregate[aggregate["architecture"].eq(architecture)].copy()
        means = means[means["ordering"].isin(MAIN_MPS_ORDERINGS)].copy()
        physical_mean = means[means["ordering"].eq("physical")]
        disrupted_means = means[~means["ordering"].eq("physical")].sort_values(
            "d64_schmidt_tail_sum",
            kind="stable",
        )
        seed_rows = per_seed[
            per_seed["architecture"].eq(architecture)
            & per_seed["ordering"].isin(MAIN_MPS_ORDERINGS)
        ].merge(
            means[["ordering", "d64_schmidt_tail_sum"]],
            on="ordering",
            how="left",
        )
        physical_seed = seed_rows[seed_rows["ordering"].eq("physical")]
        disrupted_seed = seed_rows[~seed_rows["ordering"].eq("physical")]
        ax.scatter(
            np.zeros(len(physical_seed)),
            physical_seed["C_hidden"],
            s=9,
            marker=marker,
            facecolors="none",
            edgecolors=color,
            linewidths=0.45,
            alpha=0.17,
            zorder=1,
        )
        ax.scatter(
            [0.0],
            physical_mean["C_hidden_mean"],
            s=18,
            marker=marker,
            facecolor=facecolor,
            edgecolor=color,
            linewidth=0.8,
            zorder=3,
        )
        ax.scatter(
            broken_log_coordinate(
                disrupted_seed["d64_schmidt_tail_sum"]
            ),
            disrupted_seed["C_hidden"],
            s=9,
            marker=marker,
            facecolors="none",
            edgecolors=color,
            linewidths=0.45,
            alpha=0.17,
            zorder=1,
        )
        ax.scatter(
            broken_log_coordinate(
                disrupted_means["d64_schmidt_tail_sum"]
            ),
            disrupted_means["C_hidden_mean"],
            color=color,
            marker=marker,
            s=18,
            facecolor=facecolor,
            edgecolor=color,
            linewidth=0.8,
            label=label,
            zorder=3,
        )
        hub34_architecture = hub34_rows[
            hub34_rows["architecture"].eq(architecture)
        ].sort_values("seed")
        if not hub34_architecture.empty:
            hub34_x = float(broken_log_coordinate(np.asarray([hub34_burden]))[0])
            hub34_values = hub34_architecture["C_hidden"].to_numpy(dtype=float)
            hub34_jitter = np.linspace(-0.025, 0.025, hub34_values.size)
            ax.scatter(
                hub34_x + hub34_jitter,
                hub34_values,
                s=9,
                marker=marker,
                facecolors="none",
                edgecolors=color,
                linewidths=0.45,
                alpha=0.20,
                zorder=2,
            )
            ax.scatter(
                [hub34_x],
                [float(np.mean(hub34_values))],
                s=34,
                marker=marker,
                facecolor="white",
                edgecolor=COLORS["lattice_outline"],
                linewidth=1.05,
                zorder=3,
            )
            hub34_mean = float(np.mean(hub34_values))
            ax.scatter(
                [hub34_x],
                [hub34_mean],
                s=22,
                marker=marker,
                facecolor=facecolor,
                edgecolor=color,
                linewidth=0.9,
                zorder=4,
            )
            midorder_architecture = (
                hub34_midorder_rows[
                    hub34_midorder_rows["architecture"].eq(architecture)
                ].sort_values("seed")
                if not hub34_midorder_rows.empty
                else pd.DataFrame()
            )
            if not midorder_architecture.empty:
                midorder_x = float(
                    broken_log_coordinate(
                        np.asarray([HUB34_MIDORDER_BURDEN])
                    )[0]
                )
                midorder_values = midorder_architecture[
                    "C_hidden"
                ].to_numpy(dtype=float)
                midorder_mean = float(np.mean(midorder_values))
                midorder_jitter = np.linspace(
                    -0.025, 0.025, midorder_values.size
                )
                ax.scatter(
                    midorder_x + midorder_jitter,
                    midorder_values,
                    s=9,
                    marker=marker,
                    facecolors="none",
                    edgecolors=color,
                    linewidths=0.45,
                    alpha=0.20,
                    zorder=2,
                )
                ax.scatter(
                    [midorder_x],
                    [midorder_mean],
                    s=34,
                    marker=marker,
                    facecolor="white",
                    edgecolor=COLORS["lattice_outline"],
                    linewidth=1.05,
                    zorder=3,
                )
                ax.scatter(
                    [midorder_x],
                    [midorder_mean],
                    s=22,
                    marker=marker,
                    facecolor=facecolor,
                    edgecolor=color,
                    linewidth=0.9,
                    zorder=4,
                )
    disrupted_ticks = np.asarray([0.1, 1.0, 2.0])
    reordered_burdens = aggregate[
        aggregate["ordering"].isin(MAIN_MPS_ORDERINGS)
        & ~aggregate["ordering"].eq("physical")
    ]["d64_schmidt_tail_sum"].dropna().to_numpy(dtype=float)
    if not hub34_midorder_rows.empty:
        reordered_burdens = np.append(
            reordered_burdens,
            HUB34_MIDORDER_BURDEN,
        )
    first_reordered_x = float(
        np.min(broken_log_coordinate(reordered_burdens))
    )
    ax.set(
        xlabel=r"cumulative discarded Schmidt weight at $D=64$",
        ylabel=r"unseen agreement $C_U$",
        xlim=(-0.28, 4.08),
        ylim=(-0.025, 0.715),
        yticks=[0.0, 0.2, 0.4, 0.6],
        xticks=np.concatenate(
            ([0.0], broken_log_coordinate(disrupted_ticks))
        ),
        xticklabels=[
            r"$4.0{\times}10^{-7}$",
            r"$10^{-1}$",
            r"$10^{0}$",
            r"$2{\times}10^{0}$",
        ],
    )
    hub34_x = float(broken_log_coordinate(np.asarray([hub34_burden]))[0])
    hub34_label_x = hub34_x
    if not hub34_midorder_rows.empty:
        hub34_midorder_x = float(
            broken_log_coordinate(np.asarray([HUB34_MIDORDER_BURDEN]))[0]
        )
        hub34_label_x = 0.5 * (hub34_x + hub34_midorder_x)
    # Explicit brackets show that only the two leftmost burden values use the
    # physical ordering; all displayed values to their right are reordered.
    bracket_y = 0.675
    bracket_cap = 0.012
    for left, right, text in (
        (0.0, hub34_x, "physical"),
        (first_reordered_x, 4.0, "reordered"),
    ):
        ax.plot(
            [left, right],
            [bracket_y, bracket_y],
            color="0.45",
            linewidth=0.65,
            clip_on=False,
            zorder=6,
        )
        ax.vlines(
            [left, right],
            bracket_y - bracket_cap,
            bracket_y + bracket_cap,
            color="0.45",
            linewidth=0.65,
            clip_on=False,
            zorder=6,
        )
        ax.text(
            0.5 * (left + right),
            bracket_y + 0.014,
            text,
            ha="center",
            va="bottom",
            fontsize=7.3,
            color="0.35",
        )
    ansatz_legend = ax.legend(
        frameon=False,
        fontsize=7.1,
        loc="upper center",
        bbox_to_anchor=(0.62, 0.87),
        ncol=2,
        columnspacing=0.8,
        handlelength=1.8,
        handletextpad=0.35,
        labelspacing=0.22,
    )
    ax.add_artist(ansatz_legend)
    ax.legend(
        handles=[
            Line2D(
                [],
                [],
                marker="o",
                linestyle="None",
                markersize=4.2,
                markerfacecolor="white",
                markeredgecolor="0.45",
                label="1D",
            ),
            Line2D(
                [],
                [],
                marker="o",
                linestyle="None",
                markersize=4.2,
                markerfacecolor="white",
                markeredgecolor=COLORS["lattice_outline"],
                label=r"$3{\times}4$",
            ),
        ],
        frameon=True,
        facecolor="white",
        edgecolor="0.82",
        framealpha=0.94,
        fontsize=6.9,
        loc="center right",
        bbox_to_anchor=(0.98, 0.53),
        ncol=1,
        handletextpad=0.35,
        labelspacing=0.20,
        borderpad=0.25,
    )
    break_center = 0.56
    break_half_width = 0.035
    ax.plot(
        [break_center - break_half_width, break_center],
        [-0.025, 0.025],
        transform=ax.get_xaxis_transform(),
        color="0.25",
        clip_on=False,
        linewidth=1.0,
        zorder=5,
    )
    ax.plot(
        [break_center + 0.015, break_center + break_half_width + 0.015],
        [-0.025, 0.025],
        transform=ax.get_xaxis_transform(),
        color="0.25",
        clip_on=False,
        linewidth=1.0,
        zorder=5,
    )
    panel_label(ax, "d", "Agreement versus discarded Schmidt weight")
    style_axes(ax)


def plot_h10_basis_control(ax: plt.Axes) -> None:
    completion = pd.read_csv(H10_COMPLETION)
    demand = pd.read_csv(H10_DEMAND).set_index("basis")
    basis_order = ["canonical", "lowdin"]
    series = [
        (
            "mlp_direct",
            "C_hidden",
            "MLP",
            COLORS["direct"],
            "s",
            -0.20,
            False,
        ),
        (
            "nnbf",
            "C_hidden",
            "NNBF",
            COLORS["nnbf"],
            "o",
            0.00,
            False,
        ),
        (
            "nnbf",
            "C_hidden_scaffold_carried",
            "RHF component",
            COLORS["determinant_component"],
            "D",
            0.20,
            True,
        ),
    ]
    basis_x = np.asarray([0.0, 1.0])
    for ansatz, column, label, color, marker, offset, filled in series:
        rows = completion[completion["ansatz"].eq(ansatz)]
        means: list[float] = []
        for basis_index, basis in enumerate(basis_order):
            values = (
                rows[rows["basis"].eq(basis)]
                .sort_values("seed")[column]
                .to_numpy(dtype=float)
            )
            jitter = np.linspace(-0.025, 0.025, values.size)
            ax.scatter(
                basis_x[basis_index] + offset + jitter,
                values,
                s=11,
                marker=marker,
                facecolor=color if filled else "white",
                edgecolor=color,
                linewidth=0.58,
                alpha=0.28,
                zorder=2,
            )
            means.append(float(np.mean(values)))
        ax.scatter(
            basis_x + offset,
            means,
            s=27,
            marker=marker,
            facecolor=color if filled else "white",
            edgecolor=color,
            linewidth=1.0,
            label=label,
            zorder=4,
        )
    tick_labels = [
        "canonical\nRHF unseen weight\n"
        + f"{demand.loc['canonical', 'scaffold_hidden_mass']:.3f}",
        "Löwdin\nRHF unseen weight\n"
        + f"{demand.loc['lowdin', 'scaffold_hidden_mass']:.3f}",
    ]
    ax.axhline(0.0, color="0.72", linewidth=0.7)
    ax.set(
        ylabel=r"unseen agreement $C_U$",
        xticks=basis_x,
        xticklabels=tick_labels,
        xlim=(-0.45, 1.45),
        ylim=(-0.10, 0.83),
        yticks=[0.0, 0.2, 0.4, 0.6, 0.8],
    )
    ax.legend(
        frameon=True,
        facecolor="white",
        edgecolor="0.82",
        framealpha=0.94,
        fontsize=6.7,
        loc="upper center",
        ncol=3,
        handletextpad=0.4,
        columnspacing=0.65,
        scatterpoints=1,
    )
    ax.tick_params(axis="x", labelsize=7.2, pad=2.0)
    panel_label(ax, "c", r"RHF weight and agreement across orbital bases")
    style_axes(ax)


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.8,
            "axes.titlesize": 8.9,
            "axes.labelsize": 8.8,
            "legend.fontsize": 8.6,
            "xtick.labelsize": 8.6,
            "ytick.labelsize": 8.6,
            "axes.linewidth": 0.75,
            "lines.linewidth": 1.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig = plt.figure(figsize=(7.15, 4.65), constrained_layout=True)
    grid = fig.add_gridspec(2, 1, height_ratios=(1.0, 1.0))
    top_grid = grid[0, 0].subgridspec(1, 2, width_ratios=(1.0, 1.0))
    bottom_grid = grid[1, 0].subgridspec(
        1,
        2,
        width_ratios=(0.76, 1.24),
    )
    component_ax = fig.add_subplot(top_grid[0, 0])
    attribution_ax = fig.add_subplot(top_grid[0, 1])
    h10_ax = fig.add_subplot(bottom_grid[0, 0])
    mps_ax = fig.add_subplot(bottom_grid[0, 1])

    plot_readout_component_map(component_ax)
    plot_determinant_component_attribution(attribution_ax)
    plot_mps_ordering(mps_ax)
    plot_h10_basis_control(h10_ax)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".png"), dpi=350, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
