#!/usr/bin/env python3
"""Build the finite-completion four-panel VMC figure from frozen outputs."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D

import prl_palette as palette


ROOT = Path(__file__).resolve().parents[1]
FINITE_RUNS = (
    ROOT
    / "analysis/fig4_vmc/finite_step"
)
DENSE_LEDGER = (
    ROOT
    / "analysis/fig4_vmc/dense_completion_ledger"
    / "dense_cumulative_ledger.csv"
)
REFERENCE = (
    ROOT / "reference/hchain_H10_R1p8_sto3g_lowdin_interleaved/reference.json"
)
AUDIT = (
    ROOT
    / "analysis/fig4_vmc"
    / "fig4_finite_completion"
)
OUT = ROOT / "figures/fig4_vmc_finite"

ARCHITECTURES = ("mlp_direct", "nnbf")
COLORS = {"mlp_direct": palette.DIRECT, "nnbf": palette.NNBF}
SCAFFOLD_COLOR = palette.SCAFFOLD
LABELS = {"mlp_direct": "Direct MLP", "nnbf": "NNBF"}
MARKERS = {"mlp_direct": "o", "nnbf": "^"}
ASINH_WIDTH = 2.0e-2
ACTIVE_EPOCH_WINDOW = (10, 100)


def panel_label(axis: plt.Axes, label: str, title: str) -> None:
    axis.set_title(
        rf"$\mathbf{{({label})}}$",
        loc="left",
        pad=3.0,
    )


def style_axes(axis: plt.Axes, *, grid_axis: str = "y") -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis=grid_axis, linewidth=0.35, color="0.90", zorder=0)


def parse_arch_seed(path: Path) -> tuple[str, int]:
    match = re.fullmatch(r"(mlp_direct|nnbf)_seed(\d+)", path.parent.name)
    if match is None:
        raise ValueError(f"cannot parse architecture/seed from {path}")
    return match.group(1), int(match.group(2))


def load_finite() -> pd.DataFrame:
    rows = []
    for path in sorted(FINITE_RUNS.glob("*/finite_completion_summary.csv")):
        architecture, seed = parse_arch_seed(path)
        current = pd.read_csv(path)
        convergence_path = path.parent / "quadrature_convergence.csv"
        convergence = pd.read_csv(convergence_path)
        order4 = convergence[convergence["quadrature_order"].eq(4)][
            [
                "epoch",
                "gain_external",
                "gain_external_rhf_subspace",
            ]
        ].rename(
            columns={
                "gain_external": "gain_external_order4",
                "gain_external_rhf_subspace": (
                    "gain_external_rhf_subspace_order4"
                ),
            }
        )
        current = current.merge(
            order4,
            on="epoch",
            how="left",
            validate="one_to_one",
        )
        current["architecture"] = architecture
        current["seed"] = seed
        rows.append(current)
    if not rows:
        raise FileNotFoundError(f"no finite completion rows under {FINITE_RUNS}")
    return pd.concat(rows, ignore_index=True)


def load_dense_ledger() -> pd.DataFrame:
    if not DENSE_LEDGER.exists():
        raise FileNotFoundError(DENSE_LEDGER)
    return pd.read_csv(DENSE_LEDGER)


def scaffold_fractions(finite: pd.DataFrame) -> pd.DataFrame:
    active = finite[
        finite["epoch"].between(*ACTIVE_EPOCH_WINDOW)
        & finite["gain_external"].gt(1.0e-5)
    ]
    rows = []
    for (architecture, seed), current in active.groupby(
        ["architecture", "seed"],
        sort=True,
    ):
        external = float(current["gain_external"].sum())
        scaffold = float(current["gain_external_rhf_subspace"].sum())
        rows.append(
            {
                "architecture": architecture,
                "seed": int(seed),
                "row_count": int(len(current)),
                "external_gain": external,
                "scaffold_gain": scaffold,
                "scaffold_fraction": scaffold / external,
                "residual_fraction": 1.0 - scaffold / external,
            }
        )
    return pd.DataFrame(rows)


def resolved_scaffold_updates(
    finite: pd.DataFrame,
    exact_energy: float,
) -> pd.DataFrame:
    current = finite[finite["epoch"].between(*ACTIVE_EPOCH_WINDOW)].copy()
    current["external_quadrature_relative_change"] = (
        current["gain_external"] - current["gain_external_order4"]
    ).abs() / current["gain_external"].abs().clip(lower=1.0e-30)
    current["scaffold_quadrature_relative_change"] = (
        current["gain_external_rhf_subspace"]
        - current["gain_external_rhf_subspace_order4"]
    ).abs() / current["gain_external_rhf_subspace"].abs().clip(
        lower=1.0e-30
    )
    current = current[
        current["gain_external"].gt(0.0)
        & current["external_quadrature_relative_change"].lt(0.02)
        & current["scaffold_quadrature_relative_change"].lt(0.02)
    ].copy()
    current["energy_error"] = current["energy_pre"] - exact_energy
    current["scaffold_fraction"] = (
        current["gain_external_rhf_subspace"]
        / current["gain_external"]
    )
    return current


def validate(
    finite: pd.DataFrame,
    dense: pd.DataFrame,
    fractions: pd.DataFrame,
) -> dict[str, Any]:
    for architecture in ARCHITECTURES:
        current = finite[finite["architecture"].eq(architecture)]
        if current["seed"].nunique() != 5 or len(current) != 220:
            raise ValueError(f"expected 220 finite rows for {architecture}")
        if not current.groupby("seed")["epoch"].nunique().eq(44).all():
            raise ValueError(f"expected 44 epochs per seed for {architecture}")
        if current[
            [
                "gain_external_order4",
                "gain_external_rhf_subspace_order4",
            ]
        ].isna().any().any():
            raise ValueError(f"missing order-4 audit for {architecture}")
        current = fractions[fractions["architecture"].eq(architecture)]
        if current["seed"].nunique() != 5:
            raise ValueError(f"expected five scaffold fractions for {architecture}")
        current = dense[dense["architecture"].eq(architecture)]
        if current["seed"].nunique() != 5 or len(current) != 500:
            raise ValueError(f"expected 500 dense-ledger rows for {architecture}")
        if not current.groupby("seed")["epoch"].nunique().eq(100).all():
            raise ValueError(
                f"expected consecutive epochs 1--100 for {architecture}"
            )

    max_closure = float(finite["closure_absolute_error"].max())
    if max_closure > 5.0e-4:
        raise ValueError(f"finite path no longer closes: {max_closure}")
    max_cumulative_closure = float(
        dense["cumulative_closure_error"].abs().max()
    )
    if max_cumulative_closure > 5.0e-4:
        raise ValueError(
            "dense cumulative ledger no longer closes: "
            f"{max_cumulative_closure}"
        )

    quadrant: dict[str, dict[str, int]] = {}
    for architecture in ARCHITECTURES:
        current = finite[finite["architecture"].eq(architecture)]
        q1 = current["gain_root"].gt(0) & current["gain_external"].gt(0)
        q4 = current["gain_root"].gt(0) & current["gain_external"].lt(0)
        early = current["epoch"].le(100)
        quadrant[architecture] = {
            "row_count": int(len(current)),
            "first_quadrant": int(q1.sum()),
            "root_external_conflict": int(q4.sum()),
            "exact_descent": int(current["exact_finite_gain"].gt(0).sum()),
            "through_epoch_100_row_count": int(early.sum()),
            "through_epoch_100_first_quadrant": int((q1 & early).sum()),
            "through_epoch_100_root_external_conflict": int(
                (q4 & early).sum()
            ),
        }

    dense_endpoints = (
        dense.sort_values("epoch")
        .groupby(["architecture", "seed"], as_index=False)
        .tail(1)
    )
    return {
        "max_finite_path_closure_error_Ha": max_closure,
        "max_dense_cumulative_closure_error_Ha": max_cumulative_closure,
        "quadrant_counts": quadrant,
        "scaffold_fractions": {
            architecture: [
                float(value)
                for value in fractions[
                    fractions["architecture"].eq(architecture)
                ]
                .sort_values("seed")["scaffold_fraction"]
                .to_numpy()
            ]
            for architecture in ARCHITECTURES
        },
        "dense_epoch100": {
            architecture: {
                "exact_gain_Ha": [
                    float(value)
                    for value in dense_endpoints[
                        dense_endpoints["architecture"].eq(architecture)
                    ]["cumulative_exact_gain"].to_numpy()
                ],
                "external_fraction": [
                    float(value)
                    for value in dense_endpoints[
                        dense_endpoints["architecture"].eq(architecture)
                    ]["cumulative_external_fraction"].to_numpy()
                ],
            }
            for architecture in ARCHITECTURES
        },
    }


def plot_energy(
    axis: plt.Axes,
    finite: pd.DataFrame,
    exact_energy: float,
) -> None:
    for architecture in ARCHITECTURES:
        current = finite[finite["architecture"].eq(architecture)]
        for index, (_, seed_rows) in enumerate(current.groupby("seed")):
            seed_rows = seed_rows.sort_values("epoch")
            axis.plot(
                seed_rows["epoch"],
                seed_rows["energy_pre"] - exact_energy,
                color=COLORS[architecture],
                marker=MARKERS[architecture],
                markersize=1.6,
                markeredgewidth=0,
                linewidth=0.70,
                alpha=0.62,
                label=LABELS[architecture] if index == 0 else None,
                zorder=2,
            )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set(
        xlabel="VMC epoch",
        ylabel=r"exact energy error [Ha]",
        xlim=(0.8, 2600),
        ylim=(1.2e-3, 7.0),
    )
    axis.legend(frameon=False, loc="lower left", handlelength=1.5)
    style_axes(axis)
    panel_label(axis, "a", "Individual exact-energy trajectories")


def plot_finite_phase(
    axis: plt.Axes,
    finite: pd.DataFrame,
    exact_energy: float,
) -> None:
    current = finite.copy()
    current["energy_error"] = current["energy_pre"] - exact_energy
    norm = LogNorm(
        vmin=float(max(current["energy_error"].min(), 1.0e-5)),
        vmax=float(current["energy_error"].max()),
    )
    cmap = plt.get_cmap("viridis")
    transform = lambda values: np.arcsinh(
        np.asarray(values, dtype=np.float64) / ASINH_WIDTH
    )

    for architecture in ARCHITECTURES:
        selected = current[current["architecture"].eq(architecture)]
        for _, seed_rows in selected.groupby("seed"):
            seed_rows = seed_rows.sort_values("epoch")
            axis.plot(
                transform(seed_rows["gain_root"]),
                transform(seed_rows["gain_external"]),
                color="0.62",
                linewidth=0.25,
                alpha=0.08,
                zorder=1,
            )
            axis.scatter(
                transform(seed_rows["gain_root"]),
                transform(seed_rows["gain_external"]),
                c=seed_rows["energy_error"],
                cmap=cmap,
                norm=norm,
                s=9,
                marker=MARKERS[architecture],
                edgecolor=COLORS[architecture],
                linewidth=0.35,
                zorder=2,
            )

    axis.axhline(0.0, color="0.30", linewidth=0.65)
    axis.axvline(0.0, color="0.30", linewidth=0.65)
    x_ticks = np.array([-5e-2, -1e-2, 0.0, 1e-2, 1e-1])
    x_tick_labels = [
        r"$-5{\times}10^{-2}$",
        r"$-10^{-2}$",
        "$0$",
        r"$10^{-2}$",
        r"$10^{-1}$",
    ]
    y_ticks = np.array([-1e-1, -1e-2, 0.0, 1e-2, 1e-1, 5e-1])
    y_tick_labels = [
        r"$-10^{-1}$",
        r"$-10^{-2}$",
        "$0$",
        r"$10^{-2}$",
        r"$10^{-1}$",
        r"$5{\times}10^{-1}$",
    ]
    axis.set(
        xlabel=r"finite sample gain $G_R$ [Ha]",
        ylabel=r"finite off-sample gain $G_A$ [Ha]",
        xticks=transform(x_ticks),
        xticklabels=x_tick_labels,
        yticks=transform(y_ticks),
        yticklabels=y_tick_labels,
        xlim=transform([-0.06, 0.20]),
        ylim=transform([-0.26, 0.65]),
    )
    handles = [
        Line2D(
            [0],
            [0],
            marker=MARKERS[architecture],
            linestyle="none",
            markerfacecolor="0.6",
            markeredgecolor=COLORS[architecture],
            markersize=4.5,
            label=LABELS[architecture],
        )
        for architecture in ARCHITECTURES
    ]
    axis.legend(
        handles=handles,
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.82,
        loc="upper left",
    )
    colorbar_axis = axis.inset_axes([1.02, 0.13, 0.027, 0.30])
    colorbar = axis.figure.colorbar(
        matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=colorbar_axis,
        orientation="vertical",
    )
    colorbar.ax.tick_params(labelsize=5.2, length=1.5, pad=1.1)
    colorbar.ax.set_title(r"$E-E_0$" "\n[Ha]", fontsize=5.3, pad=1.2)
    style_axes(axis, grid_axis="both")
    panel_label(axis, "b", "Sample and off-sample finite gains")


def plot_scaffold_fraction(
    axis: plt.Axes,
    updates: pd.DataFrame,
    *,
    annotation: str | None = None,
) -> None:
    for architecture in ARCHITECTURES:
        current = updates[updates["architecture"].eq(architecture)]
        role_color = COLORS[architecture]
        axis.scatter(
            current["energy_error"],
            current["scaffold_fraction"],
            s=17,
            marker=MARKERS[architecture],
            facecolor="none",
            edgecolor=role_color,
            linewidth=0.70,
            alpha=0.82,
            label=LABELS[architecture],
            zorder=2,
        )
    axis.axhline(0.0, color="0.35", linewidth=0.6)
    axis.set_xscale("log")
    axis.set(
        xlabel=r"pre-step exact error $E-E_0$ (Ha)",
        ylabel=r"scaffold fraction $G_A^{\rm scaf}/G_A$",
        xlim=(0.05, 5.2),
        ylim=(-0.06, 0.90),
        yticks=np.linspace(0, 0.8, 5),
    )
    axis.legend(
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.82,
        loc="upper right",
        fontsize=8.0,
    )
    if annotation != "":
        axis.text(
            0.02,
            0.075,
            annotation
            if annotation is not None
            else (
                r"epochs $10$-$100$; $G_A>0$"
                "\n"
                r"4$\to$8 quadrature change $<2\%$"
            ),
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=8.0,
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.82,
                "pad": 0.3,
            },
        )
    style_axes(axis)
    panel_label(axis, "d", "Scaffold share of positive off-sample gain")


def plot_finite_lesion(axis: plt.Axes, lesion: pd.DataFrame) -> None:
    current = lesion[lesion["architecture_key"].eq("nnbf")].copy()
    denominator = current["external_finite_benefit"]
    if not denominator.gt(0).all():
        raise ValueError("NNBF lesion panel requires positive external benefit")
    current["scaffold_only"] = current["root_plus_scaffold_recovery"]
    current["residual_only"] = current["root_plus_residual_recovery"]
    current["wrong_replacement"] = (
        current["random_replacement_descent_median"]
        - current["root_only_descent"]
    ) / denominator
    columns = ("scaffold_only", "residual_only", "wrong_replacement")
    labels = (
        "sample +\nscaffold",
        "sample +\nresidual",
        "sample + residual\n+ wrong mode",
    )
    positions = np.arange(len(columns))
    epoch_colors = {
        10: "#3B528B",
        20: "#21918C",
        50: "#5EC962",
        100: "#FDE725",
    }
    signed_transform = lambda values: np.arcsinh(
        np.asarray(values, dtype=np.float64) / 0.1
    )
    rng = np.random.default_rng(20260726)
    for index, column in enumerate(columns):
        jitter = rng.uniform(-0.11, 0.11, len(current))
        for x, row in zip(positions[index] + jitter, current.itertuples(index=False)):
            axis.scatter(
                [x],
                signed_transform([getattr(row, column)]),
                s=17,
                marker="o",
                facecolor=epoch_colors[int(row.epoch)],
                edgecolor="white",
                linewidth=0.40,
                zorder=2,
            )
    axis.axhline(signed_transform(0.0), color="0.35", linewidth=0.65)
    axis.axhline(
        signed_transform(1.0),
        color="0.35",
        linewidth=0.65,
        linestyle=":",
    )
    axis.text(
        2.48,
        signed_transform(1.0) + 0.035,
        "full actual",
        ha="right",
        va="bottom",
        fontsize=5.4,
    )
    axis.text(
        2.48,
        signed_transform(0.0) + 0.035,
        "sample-only",
        ha="right",
        va="bottom",
        fontsize=5.4,
    )
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=color,
            markeredgecolor="white",
            markersize=4.5,
            label=f"epoch {epoch}",
        )
        for epoch, color in epoch_colors.items()
    ]
    axis.legend(
        handles=handles,
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.82,
        loc="upper left",
        ncol=2,
        fontsize=5.3,
        handletextpad=0.25,
        columnspacing=0.7,
    )
    axis.set(
        xticks=positions,
        xticklabels=labels,
        ylabel="recovered external finite benefit",
        xlim=(-0.55, 2.55),
    )
    values = current[list(columns)].to_numpy().reshape(-1)
    raw_ticks = np.array([-5.0, -1.0, -0.1, 0.0, 0.1, 1.0])
    axis.set(
        yticks=signed_transform(raw_ticks),
        yticklabels=["-5", "-1", "-0.1", "0", "0.1", "1"],
        ylim=(
            signed_transform(min(-6.2, float(np.nanmin(values)) - 0.1)),
            signed_transform(max(1.35, float(np.nanmax(values)) + 0.1)),
        ),
    )
    style_axes(axis)
    panel_label(axis, "d", "Finite scaffold intervention in NNBF")


def plot_cumulative_ledger(axis: plt.Axes, dense: pd.DataFrame) -> None:
    marker_epochs = (10, 30, 100)
    marker_sizes = {10: 9, 30: 16, 100: 26}
    for architecture in ARCHITECTURES:
        current = dense[dense["architecture"].eq(architecture)]
        for index, (_, seed_rows) in enumerate(current.groupby("seed")):
            seed_rows = seed_rows.sort_values("epoch")
            axis.plot(
                seed_rows["cumulative_root_gain"],
                seed_rows["cumulative_external_gain"],
                color=COLORS[architecture],
                linewidth=0.75,
                linestyle=(
                    (0, (3.0, 1.8))
                    if architecture == "mlp_direct"
                    else "-"
                ),
                alpha=0.63,
                label=LABELS[architecture] if index == 0 else None,
                zorder=2,
            )
            selected = seed_rows[seed_rows["epoch"].isin(marker_epochs)]
            axis.scatter(
                selected["cumulative_root_gain"],
                selected["cumulative_external_gain"],
                s=selected["epoch"].map(marker_sizes),
                marker=MARKERS[architecture],
                facecolor="white",
                edgecolor=COLORS[architecture],
                linewidth=0.65,
                alpha=0.90,
                zorder=3,
            )
    equal_limit = 1.58
    axis.plot(
        [0.0, equal_limit],
        [0.0, equal_limit],
        color="0.55",
        linewidth=0.55,
        linestyle=":",
        zorder=1,
    )
    axis.text(
        1.54,
        1.63,
        r"$\sum G_A=\sum G_R$",
        ha="right",
        va="bottom",
        fontsize=8.0,
        color="0.35",
    )
    axis.text(
        0.03,
        0.97,
        (
            r"$\sum(G_R+G_A)=E_{\rm init}-E_t$"
            "\n"
            r"marks: $t=10,30,100$"
        ),
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=8.0,
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.82,
            "pad": 0.3,
        },
    )
    axis.set(
        xlabel=r"cumulative sample gain $\sum_{\tau\leq t}G_R$ (Ha)",
        ylabel=r"cumulative off-sample gain $\sum_{\tau\leq t}G_A$ (Ha)",
        xlim=(-0.03, equal_limit),
        ylim=(-0.08, 4.18),
    )
    axis.legend(
        frameon=False,
        loc="lower right",
        handlelength=1.5,
    )
    style_axes(axis, grid_axis="both")
    panel_label(axis, "c", "Cumulative exact energy ledger")


def main() -> None:
    finite = load_finite()
    dense = load_dense_ledger()
    fractions = scaffold_fractions(finite)
    exact_energy = float(json.loads(REFERENCE.read_text())["E0"])
    scaffold_updates = resolved_scaffold_updates(finite, exact_energy)
    audit = validate(finite, dense, fractions)
    audit["panel_d_resolved_update_counts"] = {
        architecture: int(
            scaffold_updates["architecture"].eq(architecture).sum()
        )
        for architecture in ARCHITECTURES
    }

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.3,
            "axes.titlesize": 7.8,
            "axes.labelsize": 7.3,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.45))
    plot_energy(axes[0, 0], finite, exact_energy)
    plot_finite_phase(axes[0, 1], finite, exact_energy)
    plot_cumulative_ledger(axes[1, 0], dense)
    plot_scaffold_fraction(axes[1, 1], scaffold_updates)
    fig.subplots_adjust(
        left=0.085,
        right=0.985,
        bottom=0.095,
        top=0.96,
        wspace=0.30,
        hspace=0.34,
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT.with_suffix(".png"), dpi=360, bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    AUDIT.mkdir(parents=True, exist_ok=True)
    dense.to_csv(AUDIT / "panel_c_dense_cumulative_ledger.csv", index=False)
    fractions.to_csv(AUDIT / "panel_d_scaffold_fractions.csv", index=False)
    scaffold_updates.to_csv(
        AUDIT / "panel_d_resolved_scaffold_updates.csv",
        index=False,
    )
    (AUDIT / "summary.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
