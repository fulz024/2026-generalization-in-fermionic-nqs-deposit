#!/usr/bin/env python3
"""Build the necessity -> conflict -> plateau -> scaffold Fig. 4."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D

import plot_fig4_finite as base


OUT = (
    base.ROOT
    / "figures/fig4_vmc"
)
AUDIT = (
    base.ROOT
    / "analysis/fig4_vmc"
    / "fig4_completion_conflict"
)
CLOSEST_SLATER = (
    base.ROOT
    / "analysis/fig4_vmc"
    / "closest_slater_visible_tangent_loggrid"
)
RANDOM_SUBSPACE_NULL = (
    base.ROOT
    / "analysis/fig4_vmc"
    / "closest_slater_random_subspace_null"
    / "order8/nnbf"
)
WINDOW_POINTS = 5


def diagnostic_windows(finite: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for (_, _), current in finite.groupby(
        ["architecture", "seed"],
        sort=True,
    ):
        current = current.sort_values("epoch").copy()
        current["window_root_gain"] = current["gain_root"].rolling(
            WINDOW_POINTS,
            center=True,
            min_periods=WINDOW_POINTS,
        ).sum()
        current["window_external_gain"] = current["gain_external"].rolling(
            WINDOW_POINTS,
            center=True,
            min_periods=WINDOW_POINTS,
        ).sum()
        current["valid_window"] = current["window_root_gain"].notna()
        current["window_activity"] = (
            current["window_root_gain"].abs()
            + current["window_external_gain"].abs()
        )
        current["window_conversion_efficiency"] = (
            current["window_root_gain"]
            + current["window_external_gain"]
        ) / current["window_activity"].where(
            current["window_activity"].gt(0.0)
        )
        denominator = (
            current["window_root_gain"].pow(2)
            + current["window_external_gain"].pow(2)
        )
        current["window_coordination"] = (
            2.0
            * current["window_root_gain"]
            * current["window_external_gain"]
            / denominator.where(denominator.gt(0.0))
        )
        current["persistent_conflict"] = current[
            "window_coordination"
        ].lt(0.0)
        rows.append(current)
    return pd.concat(rows, ignore_index=True)


def load_closest_slater() -> pd.DataFrame:
    architecture_map = {
        "LogMLPDirect": "mlp_direct",
        "LogNNBF": "nnbf",
    }
    by_order: dict[int, pd.DataFrame] = {}
    for order in (4, 8):
        rows: list[pd.DataFrame] = []
        for path in sorted(
            (CLOSEST_SLATER / f"order{order}").glob(
                "*/*/closest_slater_tangent_summary.csv"
            )
        ):
            current = pd.read_csv(path)
            current["architecture"] = current["architecture"].map(
                architecture_map
            )
            if current["architecture"].isna().any():
                raise ValueError(
                    f"unknown closest-Slater architecture in {path}"
                )
            current = current.rename(columns={"run_seed": "seed"})
            rows.append(current)
        if not rows:
            raise FileNotFoundError(
                f"no order-{order} closest-Slater summaries below "
                f"{CLOSEST_SLATER}"
            )
        by_order[order] = pd.concat(rows, ignore_index=True)

    keys = ["architecture", "seed", "epoch"]
    order4_columns = keys + [
        "gain_external",
        "gain_moving_state_orbital",
        "gain_moving_orbital",
    ]
    order4 = by_order[4][order4_columns].rename(
        columns={
            "gain_external": "gain_external_order4",
            "gain_moving_state_orbital": (
                "gain_moving_state_orbital_order4"
            ),
            "gain_moving_orbital": "gain_moving_orbital_order4",
        }
    )
    result = by_order[8].merge(
        order4,
        on=keys,
        how="left",
        validate="one_to_one",
    )
    expected_epochs = {
        1,
        2,
        3,
        4,
        5,
        7,
        8,
        9,
        11,
        13,
        15,
        17,
        20,
        23,
        27,
        32,
        37,
        43,
        50,
        58,
        67,
        78,
        90,
        105,
        122,
        141,
        163,
        190,
        220,
        255,
        295,
        342,
        396,
        459,
        532,
        616,
        714,
        827,
        959,
        1110,
        1286,
        1490,
        1726,
        2000,
    }
    for architecture in base.ARCHITECTURES:
        current = result[result["architecture"].eq(architecture)]
        if current["seed"].nunique() != 5:
            raise ValueError(
                f"expected five closest-Slater seeds for {architecture}"
            )
        if not current.groupby("seed")["epoch"].apply(
            lambda values: set(values) == expected_epochs
        ).all():
            raise ValueError(
                f"unexpected closest-Slater epochs for {architecture}"
            )
    return result


def load_random_subspace_null() -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[pd.DataFrame] = []
    draw_rows: list[pd.DataFrame] = []
    for seed_dir in sorted(RANDOM_SUBSPACE_NULL.glob("seed*")):
        summary_path = seed_dir / "random_subspace_null_summary.csv"
        draws_path = seed_dir / "random_subspace_null_draws.csv"
        if not summary_path.exists() or not draws_path.exists():
            raise FileNotFoundError(
                f"incomplete random-subspace null below {seed_dir}"
            )
        summary_rows.append(pd.read_csv(summary_path))
        draw_rows.append(pd.read_csv(draws_path))
    if not summary_rows:
        raise FileNotFoundError(
            f"no random-subspace null tables below {RANDOM_SUBSPACE_NULL}"
        )
    summary = pd.concat(summary_rows, ignore_index=True).rename(
        columns={"run_seed": "seed"}
    )
    draws = pd.concat(draw_rows, ignore_index=True).rename(
        columns={"run_seed": "seed"}
    )
    expected_epochs = {
        11,
        13,
        15,
        17,
        20,
        23,
        27,
        32,
        37,
        43,
        50,
        58,
        67,
        78,
        90,
    }
    if summary["seed"].nunique() != 5:
        raise ValueError("expected five random-null NNBF seeds")
    if not summary.groupby("seed")["epoch"].apply(
        lambda values: set(values) == expected_epochs
    ).all():
        raise ValueError("unexpected random-null epochs")
    draw_counts = draws.groupby(["seed", "epoch"]).size()
    if not draw_counts.eq(256).all():
        raise ValueError("expected 256 random subspaces per update")
    return summary, draws


def selected_random_subspace_null(
    draws: pd.DataFrame,
    updates: pd.DataFrame,
) -> pd.DataFrame:
    selected = updates[updates["architecture"].eq("nnbf")][
        ["seed", "epoch", "energy_error", "gain_external"]
    ].copy()
    result = draws.merge(
        selected,
        on=["seed", "epoch"],
        how="inner",
        suffixes=("_null", ""),
        validate="many_to_one",
    )
    expected_rows = 256 * len(selected)
    if len(result) != expected_rows:
        raise ValueError(
            f"expected {expected_rows} selected null draws, found {len(result)}"
        )
    if not np.allclose(
        result["energy_error_null"],
        result["energy_error"],
        atol=1.0e-10,
        rtol=0.0,
    ):
        raise ValueError("random-null and closest-Slater energies differ")
    if not np.allclose(
        result["gain_external_null"],
        result["gain_external"],
        atol=1.0e-10,
        rtol=0.0,
    ):
        raise ValueError("random-null and closest-Slater gains differ")
    return result


def resolved_moving_scaffold_updates(
    closest: pd.DataFrame,
    exact_energy: float,
) -> pd.DataFrame:
    current = closest[closest["epoch"].between(10, 100)].copy()
    current["external_quadrature_relative_change"] = (
        current["gain_external"]
        - current["gain_external_order4"]
    ).abs() / current["gain_external"].abs().clip(lower=1.0e-30)
    current["scaffold_quadrature_relative_change"] = (
        current["gain_moving_orbital"]
        - current["gain_moving_orbital_order4"]
    ).abs() / current["gain_moving_orbital"].abs().clip(
        lower=1.0e-30
    )
    current = current[
        current["gain_external"].gt(0.0)
        & current["external_quadrature_relative_change"].lt(0.02)
        & current["scaffold_quadrature_relative_change"].lt(0.02)
    ].copy()
    current["energy_error"] = current["energy_pre"] - exact_energy
    current["scaffold_fraction"] = (
        current["gain_moving_orbital"]
        / current["gain_external"]
    )
    current["state_plus_orbital_fraction"] = (
        current["gain_moving_state_orbital"]
        / current["gain_external"]
    )
    return current


def closest_slater_seed_fractions(
    closest: pd.DataFrame,
) -> pd.DataFrame:
    selected = closest[
        closest["epoch"].between(10, 100)
        & closest["gain_external"].gt(0.0)
    ].copy()
    rows: list[dict[str, float | int | str]] = []
    for (architecture, seed), current in selected.groupby(
        ["architecture", "seed"],
        sort=True,
    ):
        external_motion = float(current["motion_external_l1"].sum())
        orbital_motion = float(
            current["motion_moving_orbital_l1"].sum()
        )
        external_gain = float(current["gain_external"].sum())
        orbital_gain = float(current["gain_moving_orbital"].sum())
        rows.append(
            {
                "architecture": architecture,
                "seed": int(seed),
                "positive_update_count": int(len(current)),
                "external_motion": external_motion,
                "orbital_motion": orbital_motion,
                "external_gain": external_gain,
                "orbital_gain": orbital_gain,
                "orbital_motion_fraction": (
                    orbital_motion / external_motion
                ),
                "orbital_gain_fraction": orbital_gain / external_gain,
                "median_closest_slater_overlap": float(
                    current["closest_slater_overlap"].median()
                ),
            }
        )
    result = pd.DataFrame(rows)
    if len(result) != 10:
        raise ValueError("expected ten closest-Slater seed summaries")
    return result


def activity_marker_size(values: pd.Series | np.ndarray) -> np.ndarray:
    activity = np.asarray(values, dtype=np.float64)
    lower = -4.0
    upper = np.log10(3.0)
    scaled = (
        np.log10(np.clip(activity, 10.0**lower, 10.0**upper)) - lower
    ) / (upper - lower)
    return 3.0 + 14.0 * np.clip(scaled, 0.0, 1.0)


def dense_shape_seed_fractions(dense: pd.DataFrame) -> pd.DataFrame:
    required = {
        "architecture",
        "seed",
        "epoch",
        "cumulative_external_fraction",
        "cumulative_external_shape_fraction",
    }
    missing = required.difference(dense.columns)
    if missing:
        raise ValueError(
            "dense shape ledger is incomplete; missing "
            + ", ".join(sorted(missing))
        )
    final = (
        dense.sort_values(["architecture", "seed", "epoch"])
        .groupby(["architecture", "seed"], as_index=False)
        .tail(1)
        .copy()
    )
    if len(final) != 10 or not final["epoch"].eq(100).all():
        raise ValueError("expected five dense epoch-100 rows per architecture")
    return final


def plot_offroot_share(axis: plt.Axes, fractions: pd.DataFrame) -> None:
    offsets = np.linspace(-0.10, 0.10, 5)
    for architecture in base.ARCHITECTURES:
        current = fractions[
            fractions["architecture"].eq(architecture)
        ].sort_values("seed")
        raw = current["cumulative_external_fraction"].to_numpy()
        shape = current["cumulative_external_shape_fraction"].to_numpy()
        for raw_value, shape_value, offset in zip(
            raw,
            shape,
            offsets,
            strict=True,
        ):
            axis.plot(
                [raw_value, shape_value],
                [offset, offset],
                color=base.COLORS[architecture],
                linewidth=0.85,
                alpha=0.75,
                zorder=2,
            )
        axis.scatter(
            shape,
            offsets,
            s=24,
            marker=base.MARKERS[architecture],
            facecolor=base.COLORS[architecture],
            edgecolor=base.COLORS[architecture],
            linewidth=0.75,
            zorder=3,
        )
        axis.scatter(
            raw,
            offsets,
            s=24,
            marker=base.MARKERS[architecture],
            facecolor="white",
            edgecolor=base.COLORS[architecture],
            linewidth=0.95,
            zorder=4,
            label=base.LABELS[architecture],
        )
    axis.text(
        1.002,
        -0.205,
        "open: total   filled: configuration-specific",
        fontsize=7.8,
        color="0.28",
        ha="right",
        va="bottom",
    )
    axis.set(
        xlabel="off-sample share of exact descent",
        xlim=(0.68, 1.015),
        ylim=(-0.23, 0.23),
        yticks=[],
    )
    axis.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.54, 0.99),
        ncol=2,
        handletextpad=0.35,
        columnspacing=0.85,
    )
    base.style_axes(axis, grid_axis="x")
    base.panel_label(
        axis,
        "a",
        "",
    )


def plot_trajectory_locator(
    axis: plt.Axes,
    finite: pd.DataFrame,
    groups: dict[str, pd.DataFrame],
    terminal_ranges: dict[str, tuple[float, float]],
    exact_energy: float,
) -> None:
    for architecture in base.ARCHITECTURES:
        current = finite[finite["architecture"].eq(architecture)]
        for _, seed_rows in current.groupby("seed"):
            seed_rows = seed_rows.sort_values("epoch")
            axis.plot(
                seed_rows["epoch"],
                seed_rows["energy_pre"] - exact_energy,
                color=base.COLORS[architecture],
                linewidth=0.62,
                alpha=0.52,
                zorder=1,
            )

    styles = {
        "mlp_terminal": ("mlp_direct", "o", 24),
        "nnbf_at_mlp": ("nnbf", "^", 27),
        "nnbf_terminal": ("nnbf", "s", 29),
    }
    for name, (architecture, marker, size) in styles.items():
        group = groups[name]
        axis.scatter(
            group["epoch"],
            group["energy_pre"] - exact_energy,
            s=size,
            marker=marker,
            facecolor="white",
            edgecolor=base.COLORS[architecture],
            linewidth=0.90,
            zorder=4,
        )

    mlp_lower, mlp_upper = terminal_ranges["mlp_direct"]
    axis.axhspan(
        mlp_lower,
        mlp_upper,
        color=base.COLORS["mlp_direct"],
        alpha=0.07,
        linewidth=0.0,
        zorder=0,
    )
    nnbf_lower, nnbf_upper = terminal_ranges["nnbf"]
    axis.axhspan(
        nnbf_lower,
        nnbf_upper,
        color=base.COLORS["nnbf"],
        alpha=0.07,
        linewidth=0.0,
        zorder=0,
    )
    handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor=base.COLORS["mlp_direct"],
            markersize=4.2,
            label="MLP terminal",
        ),
        Line2D(
            [],
            [],
            marker="^",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor=base.COLORS["nnbf"],
            markersize=4.5,
            label="NNBF at MLP",
        ),
        Line2D(
            [],
            [],
            marker="s",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor=base.COLORS["nnbf"],
            markersize=4.2,
            label="NNBF terminal",
        ),
    ]
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set(
        xlabel="VMC epoch",
        ylabel="exact error\n" + r"$E-E_0$ (Ha)",
        xlim=(0.8, 2600),
        ylim=(1.2e-3, 7.0),
    )
    axis.legend(
        handles=handles,
        frameon=False,
        loc="lower left",
        ncol=3,
        handletextpad=0.25,
        columnspacing=0.65,
        borderaxespad=0.15,
        fontsize=7.3,
    )
    base.style_axes(axis)
    base.panel_label(axis, "b", "VMC energy trajectories")


def selected_window_groups(
    windows: pd.DataFrame,
    exact_energy: float,
) -> tuple[dict[str, pd.DataFrame], dict[str, tuple[float, float]]]:
    current = windows[windows["valid_window"]].copy()
    current["energy_error"] = current["energy_pre"] - exact_energy
    endpoints = (
        current.sort_values(["architecture", "seed", "epoch"])
        .groupby(["architecture", "seed"], as_index=False)
        .tail(1)
    )
    terminal_ranges = {
        architecture: (
            float(
                endpoints.loc[
                    endpoints["architecture"].eq(architecture),
                    "energy_error",
                ].min()
            ),
            float(
                endpoints.loc[
                    endpoints["architecture"].eq(architecture),
                    "energy_error",
                ].max()
            ),
        )
        for architecture in base.ARCHITECTURES
    }
    mlp_lower, mlp_upper = terminal_ranges["mlp_direct"]
    groups = {
        "mlp_terminal": current[
            current["architecture"].eq("mlp_direct")
            & current["energy_error"].between(mlp_lower, mlp_upper)
        ].copy(),
        "nnbf_at_mlp": current[
            current["architecture"].eq("nnbf")
            & current["energy_error"].between(mlp_lower, mlp_upper)
        ].copy(),
        "nnbf_terminal": endpoints[
            endpoints["architecture"].eq("nnbf")
        ].copy(),
    }
    expected = {
        "mlp_terminal": 18,
        "nnbf_at_mlp": 10,
        "nnbf_terminal": 5,
    }
    observed = {name: len(group) for name, group in groups.items()}
    if observed != expected:
        raise ValueError(
            f"unexpected highlighted-window counts: {observed}"
        )
    return groups, terminal_ranges


def plot_phase_with_windows(
    phase_axis: plt.Axes,
    finite: pd.DataFrame,
    windows: pd.DataFrame,
    exact_energy: float,
) -> None:
    current = finite.copy()
    current["energy_error"] = current["energy_pre"] - exact_energy
    groups, terminal_ranges = selected_window_groups(
        windows,
        exact_energy,
    )
    transform = lambda values: np.arcsinh(
        np.asarray(values, dtype=np.float64) / base.ASINH_WIDTH
    )

    # Retain individual finite updates as a deliberately quiet background.
    for architecture in base.ARCHITECTURES:
        selected = current[current["architecture"].eq(architecture)]
        for _, seed_rows in selected.groupby("seed"):
            seed_rows = seed_rows.sort_values("epoch")
            phase_axis.plot(
                transform(seed_rows["gain_root"]),
                transform(seed_rows["gain_external"]),
                color="0.68",
                linewidth=0.50,
                alpha=0.045,
                zorder=1,
            )
            phase_axis.scatter(
                transform(seed_rows["gain_root"]),
                transform(seed_rows["gain_external"]),
                s=9.0,
                marker=base.MARKERS[architecture],
                facecolor=base.COLORS[architecture],
                edgecolor="none",
                alpha=0.20,
                zorder=2,
            )

    phase_styles = {
        "mlp_terminal": ("mlp_direct", "o", 27),
        "nnbf_at_mlp": ("nnbf", "^", 31),
        "nnbf_terminal": ("nnbf", "s", 35),
    }
    for name, (architecture, marker, size) in phase_styles.items():
        group = groups[name]
        phase_axis.scatter(
            transform(group["window_root_gain"]),
            transform(group["window_external_gain"]),
            s=size,
            marker=marker,
            facecolor="white",
            edgecolor=base.COLORS[architecture],
            linewidth=0.95,
            zorder=5,
        )

    phase_axis.axhline(0.0, color="0.28", linewidth=0.62, zorder=0)
    phase_axis.axvline(0.0, color="0.28", linewidth=0.62, zorder=0)
    diagonal_raw = np.linspace(-0.06, 0.20, 200)
    phase_axis.plot(
        transform(diagonal_raw),
        transform(-diagonal_raw),
        color="0.42",
        linewidth=0.55,
        linestyle=(0, (2.4, 2.0)),
        zorder=0,
    )
    phase_axis.text(
        transform(0.108),
        transform(-0.118),
        "zero net gain",
        fontsize=7.2,
        color="0.35",
        rotation=-36,
        ha="left",
        va="center",
    )

    # Direct labels identify clouds rather than singling out an arbitrary
    # median window with a leader line.
    phase_axis.text(
        transform(-0.048),
        transform(0.135),
        "MLP terminal",
        color=base.COLORS["mlp_direct"],
        fontsize=7.8,
        ha="left",
        va="center",
        zorder=7,
    )
    phase_axis.text(
        transform(0.024),
        transform(0.560),
        "NNBF in MLP\nerror band",
        color=base.COLORS["nnbf"],
        fontsize=7.8,
        ha="left",
        va="top",
        zorder=7,
    )
    phase_axis.annotate(
        "NNBF terminal",
        xy=(
            transform(
                groups["nnbf_terminal"]["window_root_gain"].median()
            ),
            transform(
                groups["nnbf_terminal"]["window_external_gain"].median()
            ),
        ),
        xytext=(transform(0.015), transform(0.021)),
        arrowprops={
            "arrowstyle": "-",
            "color": base.COLORS["nnbf"],
            "linewidth": 0.55,
        },
        color=base.COLORS["nnbf"],
        fontsize=7.8,
        ha="left",
        va="bottom",
        zorder=7,
    )

    x_ticks = np.array([-5e-2, -1e-2, 0.0, 1e-2, 1e-1])
    y_ticks = np.array([-1e-1, -1e-2, 0.0, 1e-2, 1e-1, 5e-1])
    phase_axis.set(
        xlabel=r"sample gain: step $G_R$, window $D_R$ (Ha)",
        ylabel=r"off-sample gain: step $G_A$, window $D_A$ (Ha)",
        xticks=transform(x_ticks),
        xticklabels=[
            r"$-5{\times}10^{-2}$",
            r"$-10^{-2}$",
            "$0$",
            r"$10^{-2}$",
            r"$10^{-1}$",
        ],
        yticks=transform(y_ticks),
        yticklabels=[
            r"$-10^{-1}$",
            r"$-10^{-2}$",
            "$0$",
            r"$10^{-2}$",
            r"$10^{-1}$",
            r"$5{\times}10^{-1}$",
        ],
        xlim=transform([-0.06, 0.20]),
        ylim=transform([-0.26, 0.65]),
    )
    base.style_axes(phase_axis, grid_axis="both")


def plot_scaffold(
    axis: plt.Axes,
    updates: pd.DataFrame,
    _random_null: pd.DataFrame,
    windows: pd.DataFrame,
    exact_energy: float,
) -> None:
    base.plot_scaffold_fraction(
        axis,
        updates,
        annotation="",
    )
    scaffold_legend = axis.get_legend()
    if scaffold_legend is not None:
        scaffold_legend.remove()
    mlp = windows[
        windows["architecture"].eq("mlp_direct")
        & windows["epoch"].eq(windows["epoch"].max())
    ]
    lower = float((mlp["energy_pre"] - exact_energy).min())
    upper = float((mlp["energy_pre"] - exact_energy).max())
    axis.axvspan(
        lower,
        upper,
        color=base.COLORS["mlp_direct"],
        alpha=0.09,
        linewidth=0,
        zorder=0,
    )
    axis.text(
        float(np.sqrt(lower * upper)),
        0.34,
        "MLP terminal band",
        color=base.COLORS["mlp_direct"],
        fontsize=7.5,
        rotation=90,
        ha="left",
        va="center",
        alpha=0.78,
        zorder=1,
    )
    axis.set_ylabel(
        r"orbital-tangent share $G_A^{\rm orb}/G_A$"
    )
    axis.set(
        ylim=(-0.22, 1.05),
        yticks=[-0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    )
    lower, upper = axis.get_ylim()
    for architecture in base.ARCHITECTURES:
        current = updates[updates["architecture"].eq(architecture)]
        for side, marker, boundary in (
            ("low", "v", lower + 0.012),
            ("high", "^", upper - 0.012),
        ):
            if side == "low":
                overflow = current[current["scaffold_fraction"].lt(lower)]
            else:
                overflow = current[current["scaffold_fraction"].gt(upper)]
            axis.scatter(
                overflow["energy_error"],
                np.full(len(overflow), boundary),
                s=19,
                marker=marker,
                facecolor="white",
                edgecolor=base.COLORS[architecture],
                linewidth=0.75,
                zorder=5,
            )
    base.panel_label(
        axis,
        "c",
        "Slater-orbital share of off-sample change",
    )


def main() -> None:
    finite = base.load_finite()
    dense = base.load_dense_ledger()
    dense_fractions = dense_shape_seed_fractions(dense)
    exact_energy = float(base.json.loads(base.REFERENCE.read_text())["E0"])
    windows = diagnostic_windows(finite)
    highlighted_groups, terminal_ranges = selected_window_groups(
        windows,
        exact_energy,
    )
    closest = load_closest_slater()
    random_null_summary, random_null_draws = load_random_subspace_null()
    closest_fractions = closest_slater_seed_fractions(closest)
    moving_scaffold_updates = resolved_moving_scaffold_updates(
        closest,
        exact_energy,
    )
    selected_null = selected_random_subspace_null(
        random_null_draws,
        moving_scaffold_updates,
    )

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.2,
            "axes.titlesize": 8.4,
            "axes.labelsize": 8.2,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 8.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig = plt.figure(figsize=(7.1, 5.05))
    grid = fig.add_gridspec(
        3,
        2,
        width_ratios=(0.95, 1.28),
        height_ratios=(0.68, 1.0, 1.0),
    )
    share_axis = fig.add_subplot(grid[0, 0])
    scaffold_axis = fig.add_subplot(grid[1:, 0])
    locator_axis = fig.add_subplot(grid[0, 1])
    phase_axis = fig.add_subplot(grid[1:, 1])
    plot_offroot_share(share_axis, dense_fractions)
    plot_trajectory_locator(
        locator_axis,
        finite,
        highlighted_groups,
        terminal_ranges,
        exact_energy,
    )
    plot_phase_with_windows(
        phase_axis,
        finite,
        windows,
        exact_energy,
    )
    plot_scaffold(
        scaffold_axis,
        moving_scaffold_updates,
        selected_null,
        windows,
        exact_energy,
    )
    fig.subplots_adjust(
        left=0.085,
        right=0.955,
        bottom=0.085,
        top=0.96,
        wspace=0.32,
        hspace=0.46,
    )
    # Stagger the left-column split so the three-panel structure reads at a
    # glance: (b) is one compound panel, not two panels in a 2x2 grid.
    left_split_offset = 0.045
    share_box = share_axis.get_position()
    scaffold_box = scaffold_axis.get_position()
    share_axis.set_position(
        [
            share_box.x0,
            share_box.y0 - left_split_offset,
            share_box.width,
            share_box.height + left_split_offset,
        ]
    )
    scaffold_axis.set_position(
        [
            scaffold_box.x0,
            scaffold_box.y0,
            scaffold_box.width,
            scaffold_box.height - left_split_offset,
        ]
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT.with_suffix(".png"), dpi=360, bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    AUDIT.mkdir(parents=True, exist_ok=True)
    dense_fractions.to_csv(
        AUDIT / "panel_a_dense_shape_fractions.csv",
        index=False,
    )
    windows.to_csv(AUDIT / "diagnostic_windows.csv", index=False)
    pd.concat(
        [
            group.assign(highlight_group=name)
            for name, group in highlighted_groups.items()
        ],
        ignore_index=True,
    ).to_csv(
        AUDIT / "panel_b_highlighted_windows.csv",
        index=False,
    )
    pd.DataFrame(
        [
            {
                "architecture": architecture,
                "terminal_error_lower": limits[0],
                "terminal_error_upper": limits[1],
            }
            for architecture, limits in terminal_ranges.items()
        ]
    ).to_csv(
        AUDIT / "panel_b_terminal_error_ranges.csv",
        index=False,
    )
    conflict_counts = (
        windows.groupby(["architecture", "seed"], as_index=False)
        .agg(
            window_centers=("valid_window", "sum"),
            persistent_conflict_centers=("persistent_conflict", "sum"),
        )
    )
    conflict_counts.to_csv(
        AUDIT / "persistent_conflict_counts.csv",
        index=False,
    )
    closest.to_csv(
        AUDIT / "panel_c_closest_slater_updates.csv",
        index=False,
    )
    closest.to_csv(
        AUDIT / "panel_d_closest_slater_updates.csv",
        index=False,
    )
    closest_fractions.to_csv(
        AUDIT / "panel_c_closest_slater_seed_fractions.csv",
        index=False,
    )
    closest_fractions.to_csv(
        AUDIT / "panel_d_closest_slater_seed_fractions.csv",
        index=False,
    )
    moving_scaffold_updates.to_csv(
        AUDIT / "panel_c_resolved_moving_scaffold_updates.csv",
        index=False,
    )
    moving_scaffold_updates.to_csv(
        AUDIT / "panel_d_resolved_moving_scaffold_updates.csv",
        index=False,
    )
    random_null_summary.to_csv(
        AUDIT / "panel_c_random_subspace_null_summary.csv",
        index=False,
    )
    random_null_summary.to_csv(
        AUDIT / "panel_d_random_subspace_null_summary.csv",
        index=False,
    )
    selected_null.to_csv(
        AUDIT / "panel_c_random_subspace_null_draws.csv",
        index=False,
    )
    selected_null.to_csv(
        AUDIT / "panel_d_random_subspace_null_draws.csv",
        index=False,
    )
    null_quantiles = selected_null["random_fraction"].quantile(
        [0.01, 0.50, 0.99]
    )
    print(
        "random 50D null selected draws="
        f"{len(selected_null)} "
        f"q01={null_quantiles.loc[0.01]:+.6e} "
        f"median={null_quantiles.loc[0.50]:+.6e} "
        f"q99={null_quantiles.loc[0.99]:+.6e}"
    )
    print(conflict_counts.to_string(index=False))
    print(closest_fractions.to_string(index=False))


if __name__ == "__main__":
    main()
