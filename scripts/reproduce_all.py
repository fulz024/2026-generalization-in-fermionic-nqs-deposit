#!/usr/bin/env python3
"""Regenerate the five main-text and eight supplemental figures from tables."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import runpy
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "paper/figures/main"
JOBS = (
    ("plot_ansatz_schematic.py", ("fig0_ansatz_schematic_decoder",)),
    ("plot_fig1.py", ("fig1_hubbard_generalization",)),
    ("plot_slater_reference.py", ("fig2_slater_reference_predictions",)),
    ("plot_evidence_reorganization.py", ("fig3_h10_reference", "figS_decoder_comparison",
                                        "figS_extended_mps_ordering", "fig4_vmc_reorganized")),
    ("plot_family_generalization.py", ("fig5_mps_generalization",)),
    ("plot_si_nnfmps_multi_u.py", ("figS_interaction_strength",)),
    ("plot_figS_stochastic_born_support.py", ("figS_stochastic_born_support",)),
    ("plot_fig3.py", ("fig3_selected_spaces",)),
    ("plot_figS_h10_truncation.py", ("figS_h10_truncation_curve",)),
    ("plot_figS_band_control.py", ("figS_band_control",)),
)
ITEMS = {
    "fig0_ansatz_schematic_decoder": "Main Fig. 1",
    "fig1_hubbard_generalization": "Main Fig. 2",
    "fig2_slater_reference_predictions": "Main Fig. 3",
    "fig3_h10_reference": "Main Fig. 4",
    "fig5_mps_generalization": "Main Fig. 5",
    "figS_interaction_strength": "Supplement Fig. S1",
    "figS_stochastic_born_support": "Supplement Fig. S2",
    "figS_decoder_comparison": "Supplement Fig. S3",
    "figS_extended_mps_ordering": "Supplement Fig. S4",
    "fig3_selected_spaces": "Supplement Fig. S5",
    "figS_h10_truncation_curve": "Supplement Fig. S6",
    "fig4_vmc_reorganized": "Supplement Fig. S7",
    "figS_band_control": "Supplement Fig. S8",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_recorded(script_name: str, record: Path) -> None:
    """Record actual table reads and imported local modules, not raw-run guesses."""
    inputs: set[str] = set()

    def audit(event: str, args: tuple) -> None:
        if event != "open" or not isinstance(args[0], (str, bytes)):
            return
        mode = args[1]
        if isinstance(mode, str) and any(flag in mode for flag in "wax+"):
            return
        path = Path(args[0]).resolve()
        try:
            relative = path.relative_to(ROOT)
        except ValueError:
            return
        if relative.parts and relative.parts[0] in ("analysis", "reference"):
            inputs.add(relative.as_posix())

    sys.dont_write_bytecode = True
    sys.path.insert(0, str(ROOT / "scripts"))
    sys.addaudithook(audit)
    script = ROOT / "scripts" / script_name
    sys.argv = [str(script)]
    runpy.run_path(str(script), run_name="__main__")
    modules = {f"scripts/{script_name}", "scripts/reproduce_all.py"}
    for module in tuple(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if not filename:
            continue
        path = Path(filename).resolve()
        try:
            relative = path.relative_to(ROOT / "scripts")
        except ValueError:
            continue
        if path.suffix == ".py":
            modules.add("scripts/" + relative.as_posix())
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(json.dumps({"script": script_name, "inputs": sorted(inputs),
                                  "modules": sorted(modules)}, indent=2) + "\n",
                      encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-inputs", action="store_true")
    parser.add_argument("--worker", nargs=2, metavar=("SCRIPT", "RECORD"))
    args = parser.parse_args()
    if args.worker:
        run_recorded(args.worker[0], Path(args.worker[1]))
        return
    FIGURES.mkdir(parents=True, exist_ok=True)
    records = ROOT / "tmp/reproduction_inputs"
    for script_name, stems in JOBS:
        print(f"[reproduce] {script_name}", flush=True)
        if args.record_inputs:
            command = [sys.executable, "-B", str(Path(__file__)), "--worker", script_name,
                       str(records / (Path(script_name).stem + ".json"))]
        else:
            command = [sys.executable, "-B", str(ROOT / "scripts" / script_name)]
        subprocess.run(command, cwd=ROOT, check=True)
        for stem in stems:
            output = FIGURES / f"{stem}.pdf"
            if not output.is_file() or output.stat().st_size == 0:
                raise RuntimeError(f"Expected figure was not created: {output}")
    print(f"[reproduce] complete: {len(ITEMS)} manuscript figures", flush=True)


if __name__ == "__main__":
    main()
