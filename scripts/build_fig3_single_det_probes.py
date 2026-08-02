#!/usr/bin/env python3
"""Build Fig. 3 tables with static and affine-backflow single determinants."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from summarize_fig3 import cosine, half_octave_bands


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "analysis/fig3_selected_spaces_affine_20260801"
OUT = ROOT / "analysis/fig3_selected_spaces"
FE_ROOT = ROOT / "analysis/fig3_fe_probes"
N2_ROOT = ROOT / "analysis/fig3_n2_probes/d1"
ENERGY = FE_ROOT / "single_det_projected_subspace_energy.json"

PROBES = (
    ("static_determinant", "static 1-det", "static_d1", "static"),
    ("affine_backflow", "affine BF 1-det", "affine_d1", "affine"),
)


def probe_band_rows(
    wave_path: Path,
    *,
    system: str,
    top_k: int,
    ansatz: str,
    ansatz_label: str,
) -> list[dict[str, object]]:
    with np.load(wave_path) as payload:
        model = payload["model_coefficients"].astype(np.float64)
        teacher = payload["teacher_coefficients"].astype(np.float64)
        supplied = payload["supplied_indices"].astype(np.int64)
    if float(np.dot(model[supplied], teacher[supplied])) < 0.0:
        model = -model
    visible = cosine(model[supplied], teacher[supplied])
    rows: list[dict[str, object]] = []
    for band in half_octave_bands(len(teacher)):
        left = int(band["slice_start"])
        right = int(band["slice_stop"])
        rows.append(
            {
                "system": system,
                "ansatz": ansatz,
                "ansatz_label": ansatz_label,
                "seed": -1,
                "top_k": top_k,
                "visible_directional_agreement": visible,
                "visible_projective_error": max(0.0, 1.0 - visible**2),
                "band_id": band["band_id"],
                "r": band["r"],
                "rank_start": band["rank_start"],
                "rank_stop": band["rank_stop"],
                "fully_visible": right <= top_k,
                "fully_unseen": left >= top_k,
                "band_directional_agreement": cosine(
                    model[left:right], teacher[left:right]
                ),
                "teacher_band_mass": float(np.sum(teacher[left:right] ** 2)),
            }
        )
    return rows


def holdout_agreement(wave_path: Path, *, top_k: int, stop: int) -> float:
    with np.load(wave_path) as payload:
        model = payload["model_coefficients"].astype(np.float64)
        teacher = payload["teacher_coefficients"].astype(np.float64)
        supplied = payload["supplied_indices"].astype(np.int64)
    if float(np.dot(model[supplied], teacher[supplied])) < 0.0:
        model = -model
    return float(cosine(model[top_k:stop], teacher[top_k:stop]))


def replace_probe_rows(
    source_csv: Path,
    *,
    system: str,
    top_k: int,
    root: Path,
    use_n2_directories: bool = False,
) -> pd.DataFrame:
    table = pd.read_csv(source_csv)
    table = table[
        ~table["ansatz"].isin(
            ["visible_static_scaffold", "static_determinant", "affine_backflow"]
        )
    ]
    additions = []
    for ansatz, label, fe_directory, n2_directory in PROBES:
        directory = n2_directory if use_n2_directories else fe_directory
        additions.extend(
            probe_band_rows(
                root / directory / "pool_wavefunction.npz",
                system=system,
                top_k=top_k,
                ansatz=ansatz,
                ansatz_label=label,
            )
        )
    return pd.concat([table, pd.DataFrame(additions)], ignore_index=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for filename in (
        "fe_reference_bands.csv",
        "n2_reference_bands.csv",
        "fe_k_scan.csv",
        "n2_k_scan.csv",
    ):
        shutil.copy2(BASE / filename, OUT / filename)

    replace_probe_rows(
        BASE / "fe_band_metrics.csv",
        system="fe2s2",
        top_k=8192,
        root=FE_ROOT,
    ).to_csv(OUT / "fe_band_metrics.csv", index=False)
    replace_probe_rows(
        BASE / "n2_band_metrics.csv",
        system="n2_cdfci_eq_ccpvdz",
        top_k=4096,
        root=N2_ROOT,
        use_n2_directories=True,
    ).to_csv(OUT / "n2_band_metrics.csv", index=False)

    energy = pd.read_csv(BASE / "fe_energy_support.csv")
    energy = energy[
        ~energy["ansatz"].isin(
            ["visible_static_scaffold", "static_determinant", "affine_backflow"]
        )
    ]
    payload = json.loads(ENERGY.read_text(encoding="utf-8"))
    energy_rows = []
    for ansatz, label, fe_directory, _ in PROBES:
        for item in payload["models"][fe_directory]["prefix_curve"]:
            energy_rows.append(
                {
                    "kind": "support_expansion",
                    "ansatz": ansatz,
                    "ansatz_label": label,
                    "seed": -1,
                    "fixed_subspace_count": int(payload["subspace_count"]),
                    "rank_stop": int(item["rank_stop"]),
                    "energy": float(item["energy"]),
                    "energy_excess": float(
                        item["energy_excess_over_projected_ground"]
                    ),
                }
            )
    pd.concat([energy, pd.DataFrame(energy_rows)], ignore_index=True).to_csv(
        OUT / "fe_energy_support.csv", index=False
    )

    report = json.loads((BASE / "report.json").read_text(encoding="utf-8"))
    report["static_scaffold"] = {
        "definition": (
            "independently optimized static and affine-backflow single "
            "determinants fitted on the same visible shell"
        ),
        "fe_profile_holdout": [8193, 1048576],
        "n2_profile_holdout": [4097, 46341],
        "probes": {},
    }
    for ansatz, label, fe_directory, n2_directory in PROBES:
        report["static_scaffold"]["probes"][ansatz] = {
            "label": label,
            "fe_profile_holdout_agreement": holdout_agreement(
                FE_ROOT / fe_directory / "pool_wavefunction.npz",
                top_k=8192,
                stop=1048576,
            ),
            "n2_profile_holdout_agreement": holdout_agreement(
                N2_ROOT / n2_directory / "pool_wavefunction.npz",
                top_k=4096,
                stop=46341,
            ),
        }
    # Retain the old scalar fields for the plotting interface; the affine
    # single determinant is the more expressive of the two probes.
    report["static_scaffold"]["fe_profile_holdout_agreement"] = report[
        "static_scaffold"
    ]["probes"]["affine_backflow"]["fe_profile_holdout_agreement"]
    report["static_scaffold"]["n2_profile_holdout_agreement"] = report[
        "static_scaffold"
    ]["probes"]["affine_backflow"]["n2_profile_holdout_agreement"]
    report["preview_control"] = "static_and_affine_single_determinants"
    report["row_counts"] = {
        filename: len(pd.read_csv(OUT / filename))
        for filename in (
            "fe_band_metrics.csv",
            "fe_reference_bands.csv",
            "fe_energy_support.csv",
            "fe_k_scan.csv",
            "n2_band_metrics.csv",
            "n2_k_scan.csv",
            "n2_reference_bands.csv",
        )
    }
    (OUT / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(OUT)


if __name__ == "__main__":
    main()
