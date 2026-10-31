"""Vector ansatz schematic using the typography and palette of the result plots.

This depicts the paper's chosen module partition: complete scalar-output
networks have no separate decoder; determinant and matrix-product decoders
also use the occupation labels. No numerical input or model implementation
is changed, and preceding artwork is retained under its original names.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.patches import FancyArrowPatch, Rectangle

import prl_palette as palette


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper/figures/main/fig0_ansatz_schematic_decoder"
AUDIT = ROOT / "tmp/pdfs/decoder_definitions"
OCCUPATIONS = (1, 0, 1, 1, 0, 0, 1, 0)


def text(axis, x, y, label, *, size=8.3, color="#222222", **kwargs):
    return axis.text(x, y, label, fontsize=size, color=color,
                     ha=kwargs.pop("ha", "center"), va="center", **kwargs)


def arrow(axis, start, end, *, color="#555555", width=0.75, head=5.1):
    axis.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>",
                                  mutation_scale=head, linewidth=width,
                                  color=color, shrinkA=0, shrinkB=0))


def occupations(axis, center, *, sites=False):
    width, height = 0.103, 0.105
    left = center - 4 * width
    bottom = 2.115 - height / 2
    for index, occupied in enumerate(OCCUPATIONS):
        axis.add_patch(Rectangle((left + index * width, bottom), width, height,
                                 facecolor="#555555" if occupied else "white",
                                 edgecolor="#888888", linewidth=0.45))
    text(axis, left - 0.10, 2.115, r"$x$", size=8.8)
    if sites:
        for index in (2, 4, 6):
            axis.plot([left + index * width] * 2,
                      [bottom - 0.018, bottom + height + 0.018],
                      color="#333333", linewidth=0.65)
        text(axis, left + 8 * width + 0.09, 2.115,
             r"$s_1,\ldots,s_4$", size=7.0, ha="left")


def encoder(axis, center, color):
    axis.add_patch(Rectangle((center - 0.59, 1.55), 1.18, 0.28,
                             facecolor="white", edgecolor=color, linewidth=0.85))
    text(axis, center, 1.69, r"encoder $\phi_\theta$")
    arrow(axis, (center, 2.035), (center, 1.85))


def matrix(axis, center, rows, *, selected):
    cell = 0.086
    left, top = center[0] - 2 * cell, center[1] + rows * cell / 2
    for row in range(rows):
        for column in range(4):
            occupied = selected or OCCUPATIONS[row]
            axis.add_patch(Rectangle((left + column * cell, top - (row + 1) * cell),
                                     cell, cell,
                                     facecolor=to_rgba(palette.NNBF, 0.36) if occupied else "white",
                                     edgecolor="#999999", linewidth=0.38))


def headings(axis):
    for left, tag, title in (
        (0.03, "a", "Direct network"),
        (1.78, "b", "Neural-network backflow"),
        (4.49, "c", "Neuralized fermionic MPS"),
    ):
        text(axis, left, 2.51, f"({tag})", size=9.0, ha="left", fontweight="bold")
        text(axis, left + 0.25, 2.51, title, size=8.8, ha="left")


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.3,
                         "mathtext.fontset": "dejavusans", "pdf.fonttype": 42,
                         "ps.fonttype": 42, "axes.linewidth": 0.7})
    fig = plt.figure(figsize=(7.15, 2.65))
    axis = fig.add_axes([0.015, 0.015, 0.97, 0.97])
    axis.set(xlim=(0, 7.15), ylim=(0, 2.65))
    axis.set_axis_off()
    headings(axis)

    # Direct coefficient map.
    occupations(axis, 0.80)
    encoder(axis, 0.80, palette.DIRECT)
    arrow(axis, (0.80, 1.53), (0.80, 1.045))
    text(axis, 0.89, 1.285, "one number", size=7.2, ha="left", color="#555555")
    text(axis, 0.80, 0.94, r"$\psi_\theta(x)$", size=9.7)
    text(axis, 0.80, 0.24, "decoder: none", size=7.5)
    text(axis, 0.80, 0.075, "complete scalar-output network", size=7.0, color="#555555")

    # NNBF: all orbitals, occupied rows, determinant.
    occupations(axis, 2.92)
    encoder(axis, 2.92, palette.NNBF)
    matrix(axis, (2.12, 0.94), 8, selected=False)
    matrix(axis, (3.13, 0.94), 4, selected=True)
    arrow(axis, (2.64, 1.53), (2.12, 1.307))
    arrow(axis, (2.33, 0.94), (2.92, 0.94))
    text(axis, 2.62, 1.135, "occupied\nrows", size=7.0, linespacing=1.1)
    arrow(axis, (3.35, 0.94), (3.70, 0.94))
    text(axis, 3.52, 1.105, "det", size=7.3)
    text(axis, 4.02, 0.94, r"$\psi_\theta(x)$", size=9.7)
    text(axis, 2.12, 0.48, r"$O_{\theta,d}(x)$", size=7.5)
    text(axis, 2.12, 0.325, r"$M\times N$", size=7.0, color="#555555")
    text(axis, 3.13, 0.65, r"$N\times N$", size=7.0, color="#555555")
    text(axis, 3.00, 0.24, "decoder: occupied-row determinants", size=7.2)
    text(axis, 3.00, 0.075, "reference: single Slater", size=7.0, color="#555555")

    # Neuralized MPS: corrected cores coupled by D-dimensional bonds.
    occupations(axis, 5.80, sites=True)
    encoder(axis, 5.80, palette.MPS)
    centers = (4.72, 5.24, 5.76, 6.28)
    for index, center in enumerate(centers, start=1):
        axis.add_patch(Rectangle((center - 0.18, 0.80), 0.36, 0.28,
                                 facecolor="white", edgecolor=palette.MPS, linewidth=0.8))
        text(axis, center, 0.94, rf"$\tilde A_{{{index}}}^{{s_{index}}}$", size=8.0)
        arrow(axis, (5.80, 1.53), (center, 1.105), color=palette.MPS, width=0.65, head=4.6)
    text(axis, 5.80, 1.37, r"$\Delta A_{\theta,i}(x)$", size=6.8,
         color=palette.MPS, bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.5})
    for left, right in zip(centers, centers[1:]):
        axis.plot([left + 0.18, right - 0.18], [0.94, 0.94],
                  color=palette.MPS, linewidth=0.8)
        text(axis, (left + right) / 2, 0.815, r"$D$", size=7.0, color="#555555")
    arrow(axis, (6.50, 0.94), (6.76, 0.94))
    text(axis, 6.64, 1.13, "product", size=6.8)
    text(axis, 6.99, 0.94, r"$\psi_\theta(x)$", size=9.7)
    text(axis, 5.80, 0.49,
         r"$\tilde A_i^{s_i}(x)=A_i^{s_i(x)}+\Delta A_{\theta,i}(x)$", size=7.3)
    text(axis, 5.80, 0.24, "decoder: matrix product", size=7.5)
    text(axis, 5.80, 0.075, r"$\Delta A=0$: pure MPS", size=7.0, color="#555555")

    # Check bounds at the actual publication size before exporting vector art.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bounds = fig.bbox
    outside = [artist.get_text() for artist in axis.texts
               if not bounds.contains(*artist.get_window_extent(renderer).get_points()[0])
               or not bounds.contains(*artist.get_window_extent(renderer).get_points()[1])]
    if outside:
        raise ValueError(f"Off-canvas schematic labels: {outside}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT.with_suffix(".pdf"), facecolor="white")
    fig.savefig(OUT.with_suffix(".png"), dpi=300, facecolor="white")
    plt.close(fig)
    AUDIT.mkdir(parents=True, exist_ok=True)
    report = {"figure": "Main-text Fig. 1", "purpose": "Explicit encoder--decoder partition",
              "size_inches": [7.15, 2.65], "font": "DejaVu Sans",
              "family_colors": {"direct": palette.DIRECT, "NNBF": palette.NNBF, "MPS": palette.MPS},
              "occupations": OCCUPATIONS, "occupied_rows_zero_based": [i for i, value in enumerate(OCCUPATIONS) if value],
              "off_canvas_labels": outside, "raw_data_read_or_modified": False,
              "preceding_artwork_preserved": ["paper/figures/src/fig0_ansatz_schematic.tex",
                                               "paper/figures/main/fig0_ansatz_schematic_unified.pdf"],
              "direct_transformer_readout_inside_encoder": True,
              "occupation_input_explained_in_caption": True,
              "occupation_bypass_lines_drawn": False,
              "single_slater_is_reference_not_four_determinant_ablation": True}
    (AUDIT / "figure_audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
