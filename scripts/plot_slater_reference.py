"""Update the reference terminology without overwriting archived figure art."""

from __future__ import annotations

import hashlib
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import plot_evidence_reorganization as layout
import plot_fig2 as source


def main():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.4,
        "axes.titlesize": 8.8, "axes.labelsize": 8.4, "legend.fontsize": 8,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.linewidth": 0.75,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.6), layout="constrained")
    scores = layout.plot_determinant_baselines(axes[0])
    source.plot_determinant_component_attribution(axes[1])
    layout.panel_label(axes[1], "b")
    axes[1].set_title("Reference alignment of learned predictions", loc="center", fontsize=8.4)
    layout.save(fig, "fig2_slater_reference_predictions")

    bands = pd.read_csv(source.ATTRIBUTION)
    bands = bands[bands["K"].eq(4096) & bands["strict_unseen"].astype(bool)
                  & bands["r"].between(13, 16.5)]
    inputs = [source.COMPONENTS / "per_seed_metrics.csv", source.ATTRIBUTION,
              layout.SLATER_FITS]
    report = {
        "purpose": "Training-fitted Slater baseline; analytic U=0 direction for projection",
        "input_sha256": {
            str(path.relative_to(layout.ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in inputs
        },
        "scores": scores,
        "signed_reference_share": float(bands["C_scaffold_carried"].sum() / bands["C_total"].sum()),
    }
    target = layout.ROOT / "tmp/pdfs/family_generalization/slater_figure_audit.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
