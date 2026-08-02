#!/usr/bin/env python3
"""Build the final four-panel VMC completion figure from frozen exact outputs."""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
TRAJECTORY = (
    ROOT
    / "analysis/fig4_vmc/positive_bridge"
    / "certified_completion_trajectory.csv"
)
JVP_RUNS = (
    ROOT
    / "analysis/fig4_vmc/direct_jvp"
)
MATCHED_RUNS = (
    ROOT
    / "analysis/fig4_vmc/matched_control"
)
AUDIT = (
    ROOT
    / "analysis/fig4_vmc"
    / "fig4_final"
)
OUT = ROOT / "figures/figS_band_control_source"
CONTROL_OUT = (
    ROOT
    / "figures"
    / "figS_band_control"
)

ARCHITECTURES = ("mlp_direct", "nnbf")
COLORS = {"mlp_direct": "#4477AA", "nnbf": "#EE7733"}
LABELS = {"mlp_direct": "MLP", "nnbf": "NNBF"}
MARKERS = {"mlp_direct": "o", "nnbf": "s"}
ROOT_OUTSIDE_ASINH_WIDTH = 1.0e-3
MAIN_LOG_EPOCH_COUNT = 16

# Frozen after inspecting the five instrumented pairs.  Each epoch is the
# first NNBF update whose pre-update exact energy is below the corresponding
# MLP terminal energy.
CROSSING_EPOCH = {
    20260791: 18,
    20260792: 19,
    20260793: 30,
    20260794: 18,
    20260795: 13,
}

PHASES = (
    ("MLP\nfast", "mlp_direct", tuple(range(1, 21))),
    ("NNBF\nfast", "nnbf", tuple(range(1, 21))),
    ("NNBF crosses\nMLP plateau", "nnbf", CROSSING_EPOCH),
    ("MLP\nplateau", "mlp_direct", (1500, 1750, 2000)),
    ("NNBF\nplateau", "nnbf", (1500, 1750, 2000)),
)


def panel_label(ax: plt.Axes, label: str, title: str) -> None:
    ax.set_title(rf"$\mathbf{{({label})}}$ {title}", loc="left", pad=3.0)


def style_axes(ax: plt.Axes, *, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis=grid_axis, linewidth=0.38, color="0.90", zorder=0)


def parse_run(path: Path) -> tuple[str, int]:
    match = re.fullmatch(r"(mlp_direct|nnbf)_seed(\d+)", path.parent.name)
    if match is None:
        raise ValueError(f"cannot parse run directory: {path.parent}")
    return match.group(1), int(match.group(2))


def log_uniform_epochs(ledger: pd.DataFrame) -> tuple[int, ...]:
    available = np.sort(ledger["epoch"].unique())
    targets = np.linspace(
        np.log1p(available.min()),
        np.log1p(available.max()),
        MAIN_LOG_EPOCH_COUNT,
    )
    chosen = []
    for target in targets:
        index = np.abs(np.log1p(available) - target).argmin()
        epoch = int(available[index])
        if epoch not in chosen:
            chosen.append(epoch)
    if len(chosen) != MAIN_LOG_EPOCH_COUNT:
        raise ValueError(
            f"log-time selection collapsed to {len(chosen)} epochs: {chosen}"
        )
    return tuple(chosen)


def load_band_jvp() -> pd.DataFrame:
    rows = []
    for path in sorted(JVP_RUNS.glob("*/band_tangent_rows.csv")):
        architecture, seed = parse_run(path)
        current = pd.read_csv(path)
        current = current[current["epoch"].between(1, 20)].copy()
        current["architecture"] = architecture
        current["seed"] = seed
        rows.append(current)
    if not rows:
        raise FileNotFoundError(f"no band JVP rows below {JVP_RUNS}")
    data = pd.concat(rows, ignore_index=True)

    pooled = []
    for (architecture, seed, rank_center), group in data.groupby(
        ["architecture", "seed", "rank_center"], sort=True
    ):
        inner = group["common_never_tangent_inner"].sum()
        ideal_norm = np.sqrt(
            np.square(group["common_never_ideal_norm"]).sum()
        )
        tangent_norm = np.sqrt(
            np.square(group["common_never_tangent_norm"]).sum()
        )
        pooled.append(
            {
                "architecture": architecture,
                "seed": seed,
                "rank_center": rank_center,
                "pooled_cosine": inner / (ideal_norm * tangent_norm),
                "cumulative_DA": group[
                    "common_never_tangent_energy_descent"
                ].sum(),
            }
        )
    return pd.DataFrame(pooled)


def load_ledgers() -> pd.DataFrame:
    rows = []
    for path in sorted(JVP_RUNS.glob("*/ledger_rows.csv")):
        architecture, seed = parse_run(path)
        current = pd.read_csv(path)
        current = current[np.isclose(current["scale"], 1.0)].copy()
        current["architecture"] = architecture
        current["seed"] = seed
        rows.append(current)
    if not rows:
        raise FileNotFoundError(f"no ledger rows below {JVP_RUNS}")
    return pd.concat(rows, ignore_index=True)


def select_phase_rows(
    ledger: pd.DataFrame,
    architecture: str,
    epochs: tuple[int, ...] | dict[int, int],
) -> pd.DataFrame:
    current = ledger[ledger["architecture"].eq(architecture)]
    if isinstance(epochs, dict):
        parts = [
            current[
                current["seed"].eq(seed) & current["epoch"].eq(epoch)
            ]
            for seed, epoch in epochs.items()
        ]
        selected = pd.concat(parts, ignore_index=True)
        if len(selected) != len(epochs):
            raise ValueError(f"missing crossing rows for {architecture}")
        return selected
    return current[current["epoch"].isin(epochs)].copy()


def summarize_phases(ledger: pd.DataFrame) -> pd.DataFrame:
    summaries = []
    for phase, architecture, epochs in PHASES:
        current = select_phase_rows(ledger, architecture, epochs)
        if current.empty:
            raise ValueError(f"empty ledger phase: {phase}")
        raw = {
            "D_R_tan": current["root_tangent_energy_descent"].mean(),
            "D_A_tan": current["off_root_tangent_energy_descent"].mean(),
            "minus_C_amb": -current["ambient_rayleigh_correction"].mean(),
            "minus_C_ret": -current["network_retraction_correction"].mean(),
            "actual": current["exact_finite_energy_descent"].mean(),
        }
        tangent_scale = raw["D_R_tan"] + raw["D_A_tan"]
        summaries.append(
            {
                "phase": phase,
                "architecture": architecture,
                "row_count": len(current),
                "tangent_scale": tangent_scale,
                **raw,
                **{
                    f"normalized_{key}": value / tangent_scale
                    for key, value in raw.items()
                },
            }
        )
    return pd.DataFrame(summaries)


def load_matched_ablation() -> tuple[pd.DataFrame, pd.DataFrame]:
    outcomes = []
    matches = []
    for summary_path in sorted(MATCHED_RUNS.glob("batch*/summary.json")):
        summary = json.loads(summary_path.read_text())
        match = summary["selected_match"]
        batch_match = re.search(r"batch(\d+)_", summary_path.parent.name)
        if batch_match is None:
            raise ValueError(f"cannot parse batch: {summary_path.parent}")
        batch = int(batch_match.group(1))
        matches.append(
            {
                "batch": batch,
                "match_pass": bool(summary["match_pass"]),
                "root_residual_mismatch": match["root_residual_mismatch"],
                "batch_descent_mismatch": match["batch_descent_mismatch"],
                "fs_norm_mismatch": match["fs_norm_mismatch"],
            }
        )
        if not summary["match_pass"]:
            continue

        reference_energy = {
            "mlp_direct": summary["teacher"]["gate"]["energy"],
            "nnbf": summary["student"]["gate"]["energy"],
        }
        ablation = pd.read_csv(summary_path.parent / "hidden_ablation.csv")
        ablation = ablation[np.isclose(ablation["target_fs_chord"], 0.01)]
        if len(ablation) != 2:
            raise ValueError(f"unexpected ablation rows: {summary_path.parent}")
        for row in ablation.itertuples(index=False):
            architecture = (
                "mlp_direct" if row.architecture == "LogMLPDirect" else "nnbf"
            )
            outcomes.append(
                {
                    "batch": batch,
                    "architecture": architecture,
                    "full_descent": reference_energy[architecture]
                    - row.full_tangent_energy,
                    "root_only_descent": reference_energy[architecture]
                    - row.root_ablated_energy,
                    "ablation_gain": row.hidden_tangent_ablation_gain,
                    "target_fs_chord": row.target_fs_chord,
                }
            )
    return pd.DataFrame(outcomes), pd.DataFrame(matches)


def validate(
    trajectory: pd.DataFrame,
    band_jvp: pd.DataFrame,
    ledger: pd.DataFrame,
    phases: pd.DataFrame,
    ablation: pd.DataFrame,
    matches: pd.DataFrame,
) -> dict[str, object]:
    lowdin = trajectory[trajectory["basis"].eq("lowdin")]
    for architecture in ARCHITECTURES:
        current = lowdin[lowdin["ansatz"].eq(architecture)]
        if current["seed"].nunique() != 10 or current["epoch"].nunique() != 12:
            raise ValueError(f"unexpected exact trajectories for {architecture}")
        current = band_jvp[band_jvp["architecture"].eq(architecture)]
        if current["seed"].nunique() != 5:
            raise ValueError(f"unexpected band-JVP seeds for {architecture}")

    identity = (
        ledger["root_tangent_energy_descent"]
        + ledger["off_root_tangent_energy_descent"]
        - ledger["ambient_rayleigh_correction"]
        - ledger["network_retraction_correction"]
        - ledger["exact_finite_energy_descent"]
    )
    max_identity_error = float(identity.abs().max())
    if max_identity_error > 1e-10:
        raise ValueError(f"ledger identity error: {max_identity_error}")

    phase_identity = (
        phases["D_R_tan"]
        + phases["D_A_tan"]
        + phases["minus_C_amb"]
        + phases["minus_C_ret"]
        - phases["actual"]
    )
    if float(phase_identity.abs().max()) > 1e-10:
        raise ValueError("phase-averaged ledger no longer closes")

    passed = matches[matches["match_pass"]]
    if len(passed) != 14 or len(ablation) != 28:
        raise ValueError("expected 14 accepted batches and 28 ablation rows")
    if passed["root_residual_mismatch"].max() > 0.02:
        raise ValueError("root residual mismatch exceeds frozen gate")
    if passed["fs_norm_mismatch"].max() > 0.02:
        raise ValueError("FS mismatch exceeds frozen gate")
    if passed["batch_descent_mismatch"].max() > 0.02:
        raise ValueError("batch-descent mismatch exceeds frozen gate")

    paired = ablation.pivot(
        index="batch", columns="architecture", values="ablation_gain"
    )
    nnbf_wins = int((paired["nnbf"] > paired["mlp_direct"]).sum())
    if nnbf_wins != 14:
        raise ValueError("expected NNBF ablation gain to win all matched batches")

    removal = (
        ablation.groupby("architecture")[["ablation_gain", "full_descent"]]
        .mean()
    )
    removal_fraction = (
        removal["ablation_gain"] / removal["full_descent"]
    ).to_dict()

    band_pairs = band_jvp.pivot_table(
        index=["seed", "rank_center"],
        columns="architecture",
        values="pooled_cosine",
    )
    band_wins = int((band_pairs["nnbf"] > band_pairs["mlp_direct"]).sum())
    if band_wins != len(band_pairs):
        raise ValueError("NNBF did not win every paired seed-band JVP cosine")

    scatter_summary: dict[str, dict[str, dict[str, float]]] = {}
    for phase, lower, upper in (
        ("fast", 1, 20),
        ("mlp_approach", 500, 1000),
    ):
        scatter_summary[phase] = {}
        current = ledger[ledger["epoch"].between(lower, upper)]
        for architecture in ARCHITECTURES:
            rows = current[current["architecture"].eq(architecture)]
            scatter_summary[phase][architecture] = {
                "mean_D_R_tan": float(
                    rows["root_tangent_energy_descent"].mean()
                ),
                "mean_D_A_tan": float(
                    rows["off_root_tangent_energy_descent"].mean()
                ),
                "root_positive_outside_negative_count": int(
                    (
                        rows["root_tangent_energy_descent"].gt(0)
                        & rows["off_root_tangent_energy_descent"].lt(0)
                    ).sum()
                ),
                "row_count": int(len(rows)),
            }

    gate_means = (
        ledger[ledger["epoch"].isin(log_uniform_epochs(ledger))]
        .groupby(["architecture", "epoch"], as_index=False)
        .agg(
            D_R=("root_tangent_energy_descent", "mean"),
            D_A=("off_root_tangent_energy_descent", "mean"),
            actual=("exact_finite_energy_descent", "mean"),
        )
    )
    gate_means["first_quadrant"] = gate_means[["D_R", "D_A"]].min(
        axis=1
    ).gt(0)
    gate_means["actual_positive"] = gate_means["actual"].gt(0)
    gate_counts = {
        "stable_descent_first_quadrant": int(
            (gate_means["first_quadrant"] & gate_means["actual_positive"]).sum()
        ),
        "stable_descent_outside_first_quadrant": int(
            (~gate_means["first_quadrant"] & gate_means["actual_positive"]).sum()
        ),
        "nonlinear_reversal_inside_first_quadrant": int(
            (gate_means["first_quadrant"] & ~gate_means["actual_positive"]).sum()
        ),
        "conflict_and_actual_rise": int(
            (~gate_means["first_quadrant"] & ~gate_means["actual_positive"]).sum()
        ),
    }
    if gate_counts != {
        "stable_descent_first_quadrant": 22,
        "stable_descent_outside_first_quadrant": 0,
        "nonlinear_reversal_inside_first_quadrant": 3,
        "conflict_and_actual_rise": 7,
    }:
        raise ValueError(f"unexpected first-quadrant gate counts: {gate_counts}")

    return {
        "max_ledger_identity_error": max_identity_error,
        "matched_batches": int(len(passed)),
        "failed_match_batches": int((~matches["match_pass"]).sum()),
        "max_accepted_root_residual_mismatch": float(
            passed["root_residual_mismatch"].max()
        ),
        "max_accepted_fs_norm_mismatch": float(
            passed["fs_norm_mismatch"].max()
        ),
        "band_cosine_nnbf_wins": band_wins,
        "band_cosine_comparisons": int(len(band_pairs)),
        "ablation_gain_nnbf_wins": nnbf_wins,
        "ablation_gain_comparisons": int(len(paired)),
        "mean_descent_removed_fraction": removal_fraction,
        "main_panel_b_log_uniform_epochs": list(log_uniform_epochs(ledger)),
        "main_panel_b_asinh_width_Ha": ROOT_OUTSIDE_ASINH_WIDTH,
        "main_panel_c_epoch_mean_gate_counts": gate_counts,
        "root_outside_scatter": scatter_summary,
    }


def plot_energy(ax: plt.Axes, trajectory: pd.DataFrame) -> None:
    rows = trajectory[trajectory["basis"].eq("lowdin")]
    for architecture in ARCHITECTURES:
        current = rows[rows["ansatz"].eq(architecture)]
        for index, (_, seed_rows) in enumerate(current.groupby("seed")):
            seed_rows = seed_rows.sort_values("epoch")
            ax.plot(
                seed_rows["epoch"],
                seed_rows["energy_error"],
                color=COLORS[architecture],
                marker=MARKERS[architecture],
                markersize=1.8,
                markeredgewidth=0,
                linewidth=0.75,
                alpha=0.66,
                label=LABELS[architecture] if index == 0 else None,
                zorder=2,
            )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set(
        xlabel="VMC epoch",
        ylabel=r"exact energy error [Ha]",
        xlim=(0.8, 2600),
        ylim=(1.2e-3, 7.0),
    )
    ax.legend(frameon=False, loc="lower left", handlelength=1.6)
    style_axes(ax)
    panel_label(ax, "a", "Individual VMC trajectories")


def plot_band_jvp_control(ax: plt.Axes, data: pd.DataFrame) -> None:
    for architecture in ARCHITECTURES:
        current = data[data["architecture"].eq(architecture)]
        for _, seed_rows in current.groupby("seed"):
            seed_rows = seed_rows.sort_values("rank_center")
            ax.plot(
                seed_rows["rank_center"],
                seed_rows["pooled_cosine"],
                color=COLORS[architecture],
                linewidth=0.55,
                alpha=0.22,
                zorder=1,
            )
        mean = (
            current.groupby("rank_center", as_index=False)["pooled_cosine"]
            .mean()
            .sort_values("rank_center")
        )
        ax.plot(
            mean["rank_center"],
            mean["pooled_cosine"],
            color=COLORS[architecture],
            marker=MARKERS[architecture],
            markersize=2.5,
            linewidth=1.55,
            label=LABELS[architecture],
            zorder=3,
        )

    totals = (
        data.groupby(["architecture", "seed"])["cumulative_DA"]
        .sum()
        .groupby("architecture")
        .mean()
    )
    ax.text(
        0.97,
        0.96,
        "\n".join(
            [
                r"$\sum_{t\leq20}D_{A,\rm probe}^{\rm tan}$ [Ha]",
                rf"MLP {totals['mlp_direct']:.2f}",
                rf"NNBF {totals['nnbf']:.2f}",
                r"NNBF $>$ MLP: 70/70",
            ]
        ),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.0,
        linespacing=1.18,
    )
    ax.set(
        xlabel=r"target-rank band center $\log_2 r$",
        ylabel=r"direct-JVP cosine on $A_t$",
        xlim=(8.8, 15.7),
        ylim=(0.07, 0.52),
        xticks=[9, 11, 13, 15],
    )
    ax.legend(frameon=False, loc="lower left", handlelength=1.6)
    style_axes(ax)
    ax.set_title("Band-resolved off-prefix probe", loc="left", pad=3.0)


def plot_root_outside_scatter(ax: plt.Axes, ledger: pd.DataFrame) -> None:
    x_column = "root_tangent_energy_descent"
    y_column = "off_root_tangent_energy_descent"
    selected_epochs = log_uniform_epochs(ledger)
    selected = ledger[ledger["epoch"].isin(selected_epochs)].copy()
    transform = lambda values: np.arcsinh(
        np.asarray(values) / ROOT_OUTSIDE_ASINH_WIDTH
    )
    epoch_norm = LogNorm(vmin=min(selected_epochs), vmax=max(selected_epochs))
    epoch_cmap = plt.get_cmap("viridis")

    for architecture in ARCHITECTURES:
        current = selected[selected["architecture"].eq(architecture)]
        for epoch, epoch_rows in current.groupby("epoch", sort=True):
            color = epoch_cmap(epoch_norm(epoch))
            ax.scatter(
                transform(epoch_rows[x_column]),
                transform(epoch_rows[y_column]),
                s=8,
                marker=MARKERS[architecture],
                facecolor="none",
                edgecolor=color,
                linewidth=0.48,
                alpha=0.42,
                zorder=1,
            )
            ax.scatter(
                transform([epoch_rows[x_column].mean()]),
                transform([epoch_rows[y_column].mean()]),
                s=25,
                marker=MARKERS[architecture],
                facecolor=color,
                edgecolor="white",
                linewidth=0.55,
                zorder=3,
            )

    diagonal = np.geomspace(1e-5, 1.0, 200)
    ax.plot(
        transform(diagonal),
        transform(diagonal),
        color="0.65",
        linewidth=0.55,
        linestyle=":",
        zorder=0,
    )
    ax.axhline(0, color="0.35", linewidth=0.65, zorder=0)
    ax.axvline(0, color="0.35", linewidth=0.65, zorder=0)
    handles = [
        Line2D(
            [0],
            [0],
            marker=MARKERS[architecture],
            linestyle="none",
            markerfacecolor="0.35",
            markeredgecolor="white",
            markersize=5.0,
            label=LABELS[architecture],
        )
        for architecture in ARCHITECTURES
    ]
    ax.legend(
        handles=handles,
        frameon=False,
        loc="upper left",
        handletextpad=0.3,
        borderpad=0.1,
    )

    raw_x_ticks = np.array([-1e-2, -1e-3, 0, 1e-3, 1e-2, 1e-1])
    raw_y_ticks = np.array(
        [-1e-1, -1e-2, -1e-3, 0, 1e-3, 1e-2, 1e-1, 1.0]
    )
    tick_labels_x = [
        r"$-10^{-2}$",
        r"$-10^{-3}$",
        "$0$",
        r"$10^{-3}$",
        r"$10^{-2}$",
        r"$10^{-1}$",
    ]
    tick_labels_y = [
        r"$-10^{-1}$",
        r"$-10^{-2}$",
        r"$-10^{-3}$",
        "$0$",
        r"$10^{-3}$",
        r"$10^{-2}$",
        r"$10^{-1}$",
        r"$10^{0}$",
    ]
    ax.set(
        xlabel=r"sample descent $D_R^{\rm tan}$ [Ha]",
        ylabel=r"batch-external descent $D_A^{\rm tan}$ [Ha]",
        xlim=transform([-0.018, 0.55]),
        ylim=transform([-0.18, 1.05]),
        xticks=transform(raw_x_ticks),
        xticklabels=tick_labels_x,
        yticks=transform(raw_y_ticks),
        yticklabels=tick_labels_y,
    )

    colorbar_ax = ax.inset_axes([0.62, 0.055, 0.32, 0.030])
    scalar_mappable = matplotlib.cm.ScalarMappable(
        norm=epoch_norm, cmap=epoch_cmap
    )
    colorbar = ax.figure.colorbar(
        scalar_mappable,
        cax=colorbar_ax,
        orientation="horizontal",
    )
    colorbar.set_ticks([1, 10, 100, 2000])
    colorbar.set_ticklabels(["1", "10", "100", "2000"])
    colorbar.ax.tick_params(labelsize=4.5, length=1.5, pad=1.0)
    colorbar.ax.xaxis.set_ticks_position("top")
    colorbar.ax.set_title("epoch", fontsize=4.8, pad=1.0)

    style_axes(ax, grid_axis="both")
    panel_label(ax, "b", "Joint tangent propagation")


def plot_ledger(ax: plt.Axes, phases: pd.DataFrame) -> None:
    components = (
        ("normalized_D_R_tan", r"$D_R^{\rm tan}$", "#4477AA"),
        ("normalized_D_A_tan", r"$D_A^{\rm tan}$", "#EEAA33"),
        ("normalized_minus_C_amb", r"$-C_{\rm amb}$", "#CC6677"),
        ("normalized_minus_C_ret", r"$-C_{\rm ret}$", "#AA4499"),
    )
    x = np.arange(len(phases))
    positive = np.zeros(len(phases))
    negative = np.zeros(len(phases))
    for column, label, color in components:
        values = phases[column].to_numpy()
        bottoms = np.where(values >= 0, positive, negative)
        ax.bar(
            x,
            values,
            bottom=bottoms,
            width=0.66,
            color=color,
            edgecolor="white",
            linewidth=0.35,
            label=label,
            zorder=2,
        )
        positive += np.where(values >= 0, values, 0)
        negative += np.where(values < 0, values, 0)

    ax.scatter(
        x,
        phases["normalized_actual"],
        s=24,
        marker="D",
        color="black",
        edgecolor="white",
        linewidth=0.45,
        label=r"$\Delta E_{\rm actual}$",
        zorder=4,
    )
    ax.axhline(0, color="0.3", linewidth=0.65, zorder=1)
    ax.text(
        0.02,
        0.97,
        r"$D_R^{\rm tan}+D_A^{\rm tan}-C_{\rm amb}-C_{\rm ret}"
        r"=\Delta E_{\rm actual}$",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.2,
    )
    ax.set(
        xticks=x,
        xticklabels=phases["phase"],
        ylabel=r"mean contribution / $D^{\rm tan}$",
        ylim=(-1.78, 1.22),
    )
    ax.tick_params(axis="x", labelsize=5.2, pad=2)
    ax.legend(
        frameon=False,
        loc="lower left",
        ncol=3,
        fontsize=5.25,
        handlelength=1.1,
        columnspacing=0.8,
        bbox_to_anchor=(0.0, 0.005),
    )
    style_axes(ax)
    panel_label(ax, "c", "Where finite VMC descent goes")


def plot_first_quadrant_gate(ax: plt.Axes, ledger: pd.DataFrame) -> None:
    selected_epochs = log_uniform_epochs(ledger)
    rows = ledger[ledger["epoch"].isin(selected_epochs)]
    means = (
        rows.groupby(["architecture", "epoch"], as_index=False)
        .agg(
            D_R_tan=("root_tangent_energy_descent", "mean"),
            D_A_tan=("off_root_tangent_energy_descent", "mean"),
            D_tan=("tangent_energy_descent", "mean"),
            actual=("exact_finite_energy_descent", "mean"),
        )
        .sort_values(["architecture", "epoch"])
    )
    if not means["D_R_tan"].gt(0).all():
        raise ValueError("conversion plot requires positive mean root descent")
    means["tangent_conversion"] = means["D_tan"] / means["D_R_tan"]
    means["actual_conversion"] = means["actual"] / means["D_R_tan"]
    means["outside_ratio"] = means["D_A_tan"] / means["D_R_tan"]
    means["nonlinear_loss_ratio"] = (
        means["tangent_conversion"] - means["actual_conversion"]
    )
    means["in_first_quadrant"] = means["tangent_conversion"].gt(1)
    means["actual_descent_positive"] = means["actual"].gt(0)

    counterexamples = means[
        means["actual_descent_positive"]
        & ~means["in_first_quadrant"]
    ]
    nonlinear_reversals = means[
        ~means["actual_descent_positive"]
        & means["in_first_quadrant"]
    ]
    if not counterexamples.empty or len(nonlinear_reversals) != 3:
        raise ValueError(
            "first-quadrant gate changed: "
            f"{len(counterexamples)} counterexamples, "
            f"{len(nonlinear_reversals)} nonlinear reversals"
        )

    epoch_norm = LogNorm(vmin=min(selected_epochs), vmax=max(selected_epochs))
    epoch_cmap = plt.get_cmap("viridis")
    xlim = (-12.7, 27.0)
    ylim = (-25.0, 20.0)

    ax.fill_between(
        [xlim[0], 1],
        0,
        ylim[1],
        color="#CC6677",
        alpha=0.055,
        linewidth=0,
        zorder=0,
    )
    ax.fill_between(
        [1, xlim[1]],
        ylim[0],
        0,
        color="#EEAA33",
        alpha=0.075,
        linewidth=0,
        zorder=0,
    )
    diagonal_min = max(xlim[0], ylim[0])
    diagonal_max = min(xlim[1], ylim[1])
    ax.plot(
        [diagonal_min, diagonal_max],
        [diagonal_min, diagonal_max],
        color="0.45",
        linestyle=(0, (2.0, 1.8)),
        linewidth=0.7,
        zorder=1,
    )

    for row in means.itertuples(index=False):
        color = epoch_cmap(epoch_norm(row.epoch))
        marker = MARKERS[row.architecture]
        x_value = row.tangent_conversion
        y_value = row.actual_conversion
        ax.plot(
            [x_value, x_value],
            [x_value, y_value],
            color=color,
            linewidth=0.55,
            alpha=0.42,
            zorder=2,
        )
        ax.scatter(
            [x_value],
            [y_value],
            s=27,
            marker=marker,
            facecolor=color,
            edgecolor="white",
            linewidth=0.55,
            zorder=3,
        )

    ax.axhline(0, color="0.3", linewidth=0.7, zorder=1)
    ax.axvline(0, color="0.55", linewidth=0.6, zorder=1)
    ax.axvline(1, color="0.3", linewidth=0.8, zorder=1)
    ax.text(
        0.035,
        0.94,
        "stable descent without\npositive outside propagation:\nnot observed",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.15,
        color="#9A3F45",
    )
    ax.text(
        0.96,
        0.94,
        "stable descent",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=5.5,
        color="#2F6B3A",
    )
    ax.text(
        0.04,
        0.06,
        "global tangent reversal",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=5.4,
        color="#9A3F45",
    )
    ax.text(
        0.96,
        0.06,
        "finite-step reversal",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.4,
        color="#9A6A20",
    )
    ax.text(
        1.25,
        ylim[1] - 1.0,
        r"$D_A^{\rm tan}>0$",
        ha="left",
        va="top",
        fontsize=5.2,
        color="0.25",
    )
    ax.text(
        diagonal_max - 0.5,
        diagonal_max - 1.4,
        r"tangent prediction $T_{\rm VMC}=T_{\rm tan}$",
        rotation=45,
        rotation_mode="anchor",
        ha="right",
        va="top",
        fontsize=4.9,
        color="0.38",
    )
    ax.set(
        xlabel=(
            r"sample-to-global tangent conversion "
            r"$T_{\rm tan}=(D_R^{\rm tan}+D_A^{\rm tan})/D_R^{\rm tan}$"
        ),
        ylabel=(
            r"sample-to-VMC conversion "
            r"$T_{\rm VMC}=\Delta E_{\rm actual}/D_R^{\rm tan}$"
        ),
        xlim=xlim,
        ylim=ylim,
        xticks=(-10, 0, 1, 5, 10, 20),
        yticks=(-20, -10, 0, 5, 10, 20),
    )
    style_axes(ax, grid_axis="both")
    panel_label(ax, "c", "From batch descent to VMC descent")


def plot_ablation(ax: plt.Axes, data: pd.DataFrame) -> None:
    positions = {
        ("mlp_direct", "full_descent"): 0,
        ("mlp_direct", "root_only_descent"): 1,
        ("nnbf", "full_descent"): 2.5,
        ("nnbf", "root_only_descent"): 3.5,
    }
    for architecture in ARCHITECTURES:
        current = data[data["architecture"].eq(architecture)]
        x_full = positions[(architecture, "full_descent")]
        x_root = positions[(architecture, "root_only_descent")]
        for row in current.itertuples(index=False):
            ax.plot(
                [x_full, x_root],
                [row.full_descent, row.root_only_descent],
                color=COLORS[architecture],
                linewidth=0.6,
                alpha=0.24,
                zorder=1,
            )
            ax.scatter(
                [x_full, x_root],
                [row.full_descent, row.root_only_descent],
                s=7,
                facecolor="white",
                edgecolor=COLORS[architecture],
                linewidth=0.45,
                alpha=0.65,
                zorder=2,
            )
        means = current[["full_descent", "root_only_descent"]].mean()
        ax.plot(
            [x_full, x_root],
            means.to_numpy(),
            color=COLORS[architecture],
            linewidth=2.0,
            zorder=3,
        )
        ax.scatter(
            [x_full, x_root],
            means.to_numpy(),
            s=29,
            marker=MARKERS[architecture],
            facecolor=COLORS[architecture],
            edgecolor="white",
            linewidth=0.6,
            zorder=4,
        )
        removed = current["ablation_gain"].mean() / means["full_descent"]
        ax.text(
            (x_full + x_root) / 2,
            0.0252,
            f"{removed:.0%} removed\n(14/14)",
            color=COLORS[architecture],
            ha="center",
            va="top",
            fontsize=5.8,
        )

    ax.axhline(0, color="0.35", linewidth=0.65)
    ax.set(
        xticks=[0, 1, 2.5, 3.5],
        xticklabels=[
            "MLP\nfull",
            "MLP\nroot only",
            "NNBF\nfull",
            "NNBF\nroot only",
        ],
        ylabel=r"exact descent at FS chord $0.01$ [Ha]",
        xlim=(-0.45, 3.95),
        ylim=(-0.001, 0.027),
    )
    ax.tick_params(axis="x", labelsize=5.7)
    style_axes(ax)
    panel_label(ax, "d", "Ablating completion erases descent")


def main() -> None:
    trajectory = pd.read_csv(TRAJECTORY)
    band_jvp = load_band_jvp()
    ledger = load_ledgers()
    phases = summarize_phases(ledger)
    ablation, matches = load_matched_ablation()
    validation = validate(
        trajectory, band_jvp, ledger, phases, ablation, matches
    )

    AUDIT.mkdir(parents=True, exist_ok=True)
    band_jvp.to_csv(AUDIT / "panel_b_direct_jvp.csv", index=False)
    scatter_audit = ledger[
        [
            "architecture",
            "seed",
            "epoch",
            "root_tangent_energy_descent",
            "off_root_tangent_energy_descent",
            "tangent_energy_descent",
        ]
    ].copy()
    scatter_audit["main_log_uniform"] = scatter_audit["epoch"].isin(
        log_uniform_epochs(ledger)
    )
    scatter_audit.to_csv(
        AUDIT / "panel_b_root_outside_scatter.csv", index=False
    )
    gate_audit = (
        ledger[ledger["epoch"].isin(log_uniform_epochs(ledger))]
        .groupby(["architecture", "epoch"], as_index=False)
        .agg(
            D_R_tan=("root_tangent_energy_descent", "mean"),
            D_A_tan=("off_root_tangent_energy_descent", "mean"),
            D_tan=("tangent_energy_descent", "mean"),
            C_amb=("ambient_rayleigh_correction", "mean"),
            C_ret=("network_retraction_correction", "mean"),
            actual=("exact_finite_energy_descent", "mean"),
        )
    )
    gate_audit["quadrant_margin"] = gate_audit[
        ["D_R_tan", "D_A_tan"]
    ].min(axis=1)
    gate_audit["outside_ratio"] = (
        gate_audit["D_A_tan"] / gate_audit["D_R_tan"]
    )
    gate_audit["tangent_conversion"] = (
        gate_audit["D_tan"] / gate_audit["D_R_tan"]
    )
    gate_audit["actual_conversion"] = (
        gate_audit["actual"] / gate_audit["D_R_tan"]
    )
    gate_audit["nonlinear_loss_ratio"] = (
        gate_audit["tangent_conversion"]
        - gate_audit["actual_conversion"]
    )
    gate_audit["in_first_quadrant"] = gate_audit[
        "quadrant_margin"
    ].gt(0)
    gate_audit["actual_descent_positive"] = gate_audit["actual"].gt(0)
    gate_audit.to_csv(
        AUDIT / "panel_c_first_quadrant_gate.csv", index=False
    )
    phases.to_csv(AUDIT / "panel_c_energy_ledger.csv", index=False)
    ablation.to_csv(AUDIT / "panel_d_hidden_ablation.csv", index=False)
    matches.to_csv(AUDIT / "panel_d_matching_audit.csv", index=False)
    (AUDIT / "validation.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n"
    )

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 6.6,
            "axes.titlesize": 7.0,
            "axes.labelsize": 6.5,
            "xtick.labelsize": 5.9,
            "ytick.labelsize": 5.9,
            "legend.fontsize": 5.9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    control_fig, control_ax = plt.subplots(figsize=(3.45, 2.45))
    plot_band_jvp_control(control_ax, band_jvp)
    control_fig.subplots_adjust(
        left=0.17, right=0.98, bottom=0.20, top=0.91
    )
    CONTROL_OUT.parent.mkdir(parents=True, exist_ok=True)
    control_fig.savefig(CONTROL_OUT.with_suffix(".pdf"), bbox_inches="tight")
    control_fig.savefig(
        CONTROL_OUT.with_suffix(".png"), dpi=360, bbox_inches="tight"
    )
    plt.close(control_fig)

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(7.05, 5.15),
        gridspec_kw={"hspace": 0.39, "wspace": 0.27},
    )
    plot_energy(axes[0, 0], trajectory)
    plot_root_outside_scatter(axes[0, 1], ledger)
    plot_first_quadrant_gate(axes[1, 0], ledger)
    plot_ablation(axes[1, 1], ablation)
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.105, top=0.96)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".png"), dpi=360, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
