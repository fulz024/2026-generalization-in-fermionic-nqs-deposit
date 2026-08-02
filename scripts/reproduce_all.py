#!/usr/bin/env python3
"""Regenerate every manuscript figure shipped by this public archive."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
JOBS = (
    ("plot_fig1.py", "fig1_hubbard_generalization.pdf"),
    ("plot_fig2.py", "fig2_structure_mechanism.pdf"),
    ("plot_fig3.py", "fig3_selected_spaces.pdf"),
    ("plot_fig4.py", "fig4_vmc.pdf"),
    ("plot_figS_interaction_strength.py", "figS_interaction_strength.pdf"),
    ("plot_figS_stochastic_born_support.py", "figS_stochastic_born_support.pdf"),
    ("plot_figS_h10_truncation.py", "figS_h10_truncation_curve.pdf"),
    ("plot_figS_band_control.py", "figS_band_control.pdf"),
)


def main() -> None:
    for script_name, output_name in JOBS:
        script = ROOT / "scripts" / script_name
        print(f"[reproduce] {script_name}", flush=True)
        subprocess.run([sys.executable, "-B", str(script)], check=True)
        output = FIGURES / output_name
        if not output.is_file() or output.stat().st_size == 0:
            raise RuntimeError(f"expected output was not created: {output}")
    print(f"[reproduce] complete: {len(JOBS)} figures in {FIGURES}")


if __name__ == "__main__":
    main()
