"""Reference-centered figures from existing numerical outputs; no new fits.

Connect the Hubbard baselines to the learned-output projection, then test
the same reference interpretation in H10. Decoder and MPS-ordering controls
remain in the Supplement. Original figures and numerical data are unchanged.
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

import plot_fig2 as structure
import plot_fig4 as vmc
import prl_palette as palette


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper/figures/main"
QA = ROOT / "tmp/pdfs/merged_reference"
SLATER_FITS = ROOT / "analysis/s3_static_scaffold_n1_transition_highk/summary.csv"


def panel_label(axis: plt.Axes, label: str) -> None:
    axis.set_title(rf"$\mathbf{{({label})}}$", loc="left", pad=4)


def save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def score_marks(axis, position, values, color, marker="o"):
    values = np.asarray(values, dtype=float)
    if len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("Empty or nonfinite benchmark values")
    if len(values) > 1:
        axis.scatter(
            position + np.linspace(-0.045, 0.045, len(values)), values,
            s=13, marker=marker, facecolor="none", edgecolor=color,
            alpha=0.35, linewidth=0.65, zorder=2,
        )
    axis.scatter(
        [position], [values.mean()], s=34, marker=marker,
        facecolor=color, edgecolor=color, linewidth=0.85, zorder=3,
    )


def benchmark_axes(axis, labels):
    axis.axhline(0, color="0.72", linewidth=0.7)
    axis.set(
        ylabel=r"unseen agreement $C_U$",
        xticks=range(len(labels)), xticklabels=labels,
        xlim=(-0.45, len(labels) - 0.55), ylim=(-0.04, 1.04),
        yticks=[0, 0.25, 0.5, 0.75, 1],
    )
    axis.tick_params(axis="x", labelsize=7.7)
    structure.style_axes(axis)


def plot_determinant_baselines(axis):
    baselines = pd.read_csv(structure.COMPONENTS / "per_seed_metrics.csv")
    summaries = []
    baseline_specs = (
        ("visible_corrected_direct_mlp", "direct\nMLP", palette.DIRECT, 5),
        ("visible_corrected_slater", "Slater", palette.PHYSICAL_BASELINE, 1),
        ("visible_corrected_simple_backflow", "lattice BF\n(fixed orbitals)", palette.PHYSICAL_BASELINE_DARK, 1),
        ("visible_corrected_neural_jastrow", "neural\nJastrow", palette.JASTROW, 5),
        ("visible_corrected_nnbf", "NNBF", palette.NNBF, 5),
    )
    for position, (key, label, color, count) in enumerate(baseline_specs):
        if key == "visible_corrected_slater":
            fits = pd.read_csv(SLATER_FITS)
            fits = fits[fits["K"].eq(4096)]
            # Select by the training objective only, never by unseen agreement.
            best = fits.sort_values("fit_objective", ascending=False).iloc[0]
            values = np.asarray([best["C_hidden"]], dtype=float)
        else:
            values = baselines.loc[baselines["component"].eq(key), "C_hidden"].to_numpy()
        if len(values) != count:
            raise ValueError(f"Unexpected baseline count for {key}")
        score_marks(axis, position, values, color)
        summaries.append({"panel": "a", "model": key, "n": count, "C_U": float(values.mean())})
    benchmark_axes(axis, [item[1] for item in baseline_specs])
    axis.set_title("Simple references and neural models", loc="center", fontsize=8.4)
    panel_label(axis, "a")
    return summaries


def plot_decoder_control(axis):
    heads = pd.read_csv(structure.HEADS / "per_seed_metrics.csv")
    summaries = []
    head_specs = (
        ("global additive", "all-entry\nsum", palette.DIRECT),
        ("occupied additive", "occupied-row\nsum", palette.DIRECT_LIGHT),
        ("symmetric product", "row-sum\nproduct", palette.SYMMETRIC),
        ("determinant", "determinant", palette.NNBF),
    )
    for position, (key, label, color) in enumerate(head_specs):
        selected = heads[heads["label"].eq(key)]
        if len(selected) != 5 or not selected["common_shell_eligible"].all():
            raise ValueError(f"Decoder fits not all error-matched: {key}")
        values = selected["C_hidden"].to_numpy()
        score_marks(axis, position, values, color, marker="D")
        summaries.append({"model": key, "n": 5, "C_U": float(values.mean())})
    benchmark_axes(axis, [item[1] for item in head_specs])
    axis.set_title("Same encoder architecture; separately retrained", loc="center", fontsize=8.4)
    return summaries


def plot_basis_benchmark(axis):
    completion = pd.read_csv(structure.H10_COMPLETION)
    demand = pd.read_csv(structure.H10_DEMAND).set_index("basis")
    for architecture, label, color, marker, offset in (
        ("mlp_direct", "MLP", palette.DIRECT, "s", -0.08),
        ("nnbf", "NNBF", palette.NNBF, "o", 0.08),
    ):
        means = []
        for position, basis in enumerate(("canonical", "lowdin")):
            values = completion.loc[
                completion["ansatz"].eq(architecture) & completion["basis"].eq(basis),
                "C_hidden",
            ].to_numpy()
            if len(values) != 5:
                raise ValueError("H10 comparison requires five seeds per case")
            means.append(values.mean())
            score_marks(axis, position + offset, values, color, marker)
        axis.plot([], [], marker=marker, color=color, linestyle="none", label=label)
    labels = [
        name + "\nunseen target weight\n" + f"{100 * demand.loc[basis, 'target_hidden_mass']:.1f}%"
        for basis, name in (("canonical", "canonical"), ("lowdin", "Löwdin"))
    ]
    axis.axhline(0, color="0.72", linewidth=0.7)
    axis.set(
        ylabel=r"unseen agreement $C_U$", xticks=[0, 1], xticklabels=labels,
        xlim=(-0.35, 1.35), ylim=(-0.09, 0.83), yticks=[0, 0.2, 0.4, 0.6, 0.8],
    )
    axis.legend(frameon=False, loc="upper left", ncol=2, fontsize=8)
    axis.tick_params(axis="x", labelsize=8)
    axis.set_title(r"H$_{10}$: orbital basis", loc="center", fontsize=8.4)
    structure.style_axes(axis)
    panel_label(axis, "a")


def plot_minimal_ordering(axis: plt.Axes) -> list[dict]:
    rows = pd.read_csv(structure.MPS / "per_seed_metrics.csv")
    aggregate = pd.read_csv(structure.MPS / "aggregate_metrics.csv")
    orderings = ("physical", "even_odd")
    summary = []
    for architecture, label, color, marker, offset in (
        ("pure_mps_d64", "pure MPS", palette.PHYSICAL_BASELINE, "s", -0.07),
        ("nn_fmps_d64", "NN-fMPS", palette.MPS, "o", 0.07),
    ):
        means = []
        for position, ordering in enumerate(orderings):
            selected = rows[
                rows["architecture"].eq(architecture)
                & rows["ordering"].eq(ordering)
            ].sort_values("seed")
            if len(selected) != 5:
                raise ValueError(f"Expected five fits: {architecture}, {ordering}")
            values = selected["C_hidden"].to_numpy(dtype=float)
            mean = float(values.mean())
            recorded = aggregate[
                aggregate["architecture"].eq(architecture)
                & aggregate["ordering"].eq(ordering)
            ].iloc[0]
            if not np.isclose(mean, recorded["C_hidden_mean"], atol=1e-12):
                raise ValueError("Per-seed and aggregate MPS values disagree")
            means.append(mean)
            axis.scatter(
                position + offset + np.linspace(-0.025, 0.025, len(values)),
                values,
                s=14,
                marker=marker,
                facecolor="none",
                edgecolor=color,
                alpha=0.35,
                linewidth=0.65,
            )
            summary.append(
                {
                    "architecture": architecture,
                    "ordering": ordering,
                    "n_seed": len(values),
                    "C_U_mean": mean,
                    "schmidt_tail_sum": float(recorded["d64_schmidt_tail_sum"]),
                }
            )
        axis.plot(
            np.arange(2) + offset,
            means,
            marker=marker,
            markersize=5,
            color=color,
            markerfacecolor="white" if architecture == "pure_mps_d64" else color,
            linewidth=0.9,
            label=label,
        )
    axis.axhline(0, color="0.72", linewidth=0.7)
    axis.set(
        ylabel=r"unseen agreement $C_U$",
        xticks=[0, 1],
        xticklabels=["physical order", "even-odd order"],
        xlim=(-0.35, 1.35),
        ylim=(-0.055, 0.66),
        yticks=[0, 0.2, 0.4, 0.6],
    )
    axis.legend(frameon=False, loc="upper right", fontsize=8)
    axis.tick_params(axis="x", labelsize=8)
    structure.style_axes(axis)
    axis.set_title("Hubbard chain: site order", loc="center", fontsize=8.4)
    panel_label(axis, "b")
    return summary


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    QA.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.4,
            "axes.titlesize": 8.8,
            "axes.labelsize": 8.4,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.75,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.6), layout="constrained")
    determinant_summary = plot_determinant_baselines(axes[0])
    structure.plot_determinant_component_attribution(axes[1])
    panel_label(axes[1], "b")
    axes[1].set_title("Reference alignment of learned predictions", loc="center", fontsize=8.4)
    save(fig, "fig2_reference_predictions")

    fig, axis = plt.subplots(figsize=(3.4, 2.7), layout="constrained")
    structure.plot_h10_basis_control(axis)
    axis.set_title("", loc="left")
    axis.set_title(r"H$_{10}$: the same RHF state in two bases", loc="center", fontsize=8.4)
    handles, labels = axis.get_legend_handles_labels()
    axis.legend(
        handles, ["RHF-aligned part" if label == "RHF component" else label for label in labels],
        frameon=True, facecolor="white", edgecolor="0.82", framealpha=0.94,
        fontsize=6.7, loc="upper center", ncol=3, handletextpad=0.4,
        columnspacing=0.65, scatterpoints=1,
    )
    save(fig, "fig3_h10_reference")

    fig, axis = plt.subplots(figsize=(4.1, 2.7), layout="constrained")
    decoder_summary = plot_decoder_control(axis)
    save(fig, "figS_decoder_comparison")

    fig, axis = plt.subplots(figsize=(7.15, 2.8), layout="constrained")
    structure.plot_mps_ordering(axis)
    axis.set_title("", loc="left")
    save(fig, "figS_extended_mps_ordering")

    finite = vmc.base.load_finite()
    dense = vmc.base.load_dense_ledger()
    fractions = vmc.dense_shape_seed_fractions(dense)
    exact_energy = float(json.loads(vmc.base.REFERENCE.read_text())["E0"])
    windows = vmc.diagnostic_windows(finite)
    groups, terminal_ranges = vmc.selected_window_groups(windows, exact_energy)
    closest = vmc.load_closest_slater()
    updates = vmc.resolved_moving_scaffold_updates(closest, exact_energy)
    fig = plt.figure(figsize=(7.15, 5.0), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=(0.85, 1.35), width_ratios=(1.1, 1))
    energy_axis = fig.add_subplot(grid[0, 0])
    share_axis = fig.add_subplot(grid[0, 1])
    phase_axis = fig.add_subplot(grid[1, 0])
    orbital_axis = fig.add_subplot(grid[1, 1])
    vmc.plot_trajectory_locator(energy_axis, finite, groups, terminal_ranges, exact_energy)
    vmc.plot_offroot_share(share_axis, fractions)
    for annotation in share_axis.texts:
        if annotation.get_text().startswith("open: total"):
            annotation.set_text("open: total\nfilled: configuration-specific")
            annotation.set_transform(share_axis.transAxes)
            annotation.set_position((0.98, 0.025))
            annotation.set_fontsize(7.2)
    vmc.plot_phase_with_windows(phase_axis, finite, windows, exact_energy)
    for annotation in phase_axis.texts:
        if annotation.get_text() == "zero net gain":
            annotation.set_transform(phase_axis.transAxes)
            annotation.set_position((0.82, 0.06))
            annotation.set_ha("center")
    # The plotting function does not use its random-null argument. The null
    # remains in the Supplement; no new draws or numerical estimates are made.
    vmc.plot_scaffold(orbital_axis, updates, pd.DataFrame(), windows, exact_energy)
    for axis, label in zip(
        (energy_axis, share_axis, phase_axis, orbital_axis), "abcd", strict=True
    ):
        panel_label(axis, label)
    save(fig, "fig4_vmc_reorganized")

    input_paths = [
        structure.COMPONENTS / "per_seed_metrics.csv",
        structure.HEADS / "per_seed_metrics.csv",
        structure.ATTRIBUTION,
        structure.H10_COMPLETION,
        structure.H10_DEMAND,
        structure.MPS / "per_seed_metrics.csv",
        structure.MPS / "aggregate_metrics.csv",
        vmc.base.DENSE_LEDGER,
    ]
    bands = pd.read_csv(structure.ATTRIBUTION)
    selected = bands[
        bands["K"].eq(4096) & bands["strict_unseen"].astype(bool)
        & bands["r"].between(13, 16.5)
    ]
    if selected["seed"].nunique() != 5 or len(selected) != 40:
        raise ValueError("Reference attribution requires eight bands for each of five seeds")
    completion = pd.read_csv(structure.H10_COMPLETION)
    report = {
        "new_training_runs": 0,
        "determinant_benchmark_summary": determinant_summary,
        "supplement_decoder_summary": decoder_summary,
        "hubbard_reference_signed_share": float(
            selected["C_scaffold_carried"].sum() / selected["C_total"].sum()
        ),
        "h10_mean_agreements": completion.groupby(["basis", "ansatz"])[
            ["C_hidden", "C_hidden_scaffold_carried"]
        ].mean().reset_index().to_dict("records"),
        "vmc_retained_updates": updates.groupby("architecture").size().to_dict(),
        "source_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in input_paths
        },
        "outputs": [
            str(OUT / f"{stem}.pdf")
            for stem in (
                "fig2_reference_predictions",
                "fig3_h10_reference",
                "figS_decoder_comparison",
                "figS_extended_mps_ordering",
                "fig4_vmc_reorganized",
            )
        ],
    }
    (QA / "figure_audit.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
