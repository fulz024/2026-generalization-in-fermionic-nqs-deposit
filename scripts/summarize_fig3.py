#!/usr/bin/env python3
"""Assemble the four-panel PRL Fig. 3 scale-gate tables."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SHELL_TAG = "0p99999"
SEEDS = (20260790, 20260791, 20260792, 20260793, 20260794)
K_GRID = (128, 256, 512, 1024, 2048, 4096)
FE_K_SCAN_SEEDS = {
    256: (20260790, 20260791, 20260792, 20260793, 20260797),
    512: SEEDS,
    1024: SEEDS,
}
ANSATZES = {
    "transformer_direct_phase": ("direct", "direct Transformer"),
    "nnbf": ("nnbf", "NNBF"),
}
FE_K_SCAN_SEEDS_BY_ANSATZ = {
    ansatz: dict(FE_K_SCAN_SEEDS)
    for ansatz in ANSATZES
}
SEED_MANIFEST: dict[str, dict[int, dict[str, tuple[int, ...]]]] = {}
MAX_VISIBLE_PROJECTIVE_ERROR = 2.0e-5
FE_ANALYSIS = (
    ROOT / "analysis/fig3_scale_pilot" / f"shell_{SHELL_TAG}" / "fe2s2"
)
FE_PROFILE_ANALYSIS = FE_ANALYSIS
FE_ENERGY_ANALYSIS = FE_ANALYSIS
FE_AUDIT = (
    ROOT
    / "analysis/fig3_reference_pool_audit"
    / "fe2s2_high_vs_continued"
)
FE_SAME_D_AUDIT = FE_AUDIT
N2_NAME = "n2_cdfci_eq_ccpvdz"
N2_ANALYSIS = (
    ROOT
    / "analysis/fig3_scale_pilot"
    / f"shell_{SHELL_TAG}"
    / N2_NAME
)
N2_AUDIT = ROOT / "analysis/fig3_reference_pool_audit" / N2_NAME
N2_SAME_D_AUDIT = N2_AUDIT
STATIC_ANALYSIS = ROOT / "analysis/fig3_visible_static_scaffold"
FE_PROFILE_STATIC_ANALYSIS = STATIC_ANALYSIS
N2_STATIC_ANALYSIS = STATIC_ANALYSIS
STATIC_ENERGY = (
    ROOT
    / "analysis/fig3_visible_static_scaffold_energy"
    / "projected_subspace_energy.json"
)
FE_K512_LEGACY = True


def half_octave_bands(dimension: int) -> list[dict[str, Any]]:
    bands: list[dict[str, Any]] = []
    for octave in range(int(math.ceil(math.log2(dimension)))):
        start = 2**octave
        stop = min(2 ** (octave + 1), dimension)
        if start >= stop:
            continue
        split = min(
            max(int(round(start * math.sqrt(2.0))), start + 1),
            stop,
        )
        for left, right, suffix, offset in (
            (start, split, "a", 0.0),
            (split, stop, "b", 0.5),
        ):
            if left < right:
                bands.append(
                    {
                        "band_id": f"B{octave}{suffix}",
                        "r": float(octave + offset),
                        "rank_start": left + 1,
                        "rank_stop": right,
                        "slice_start": left,
                        "slice_stop": right,
                    }
                )
    return bands


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator else float("nan")


def seed_dir(
    analysis: Path,
    *,
    ansatz: str,
    seed: int,
    top_k: int,
    legacy_k512: bool = False,
) -> Path:
    if legacy_k512 and top_k == 512:
        return analysis / ansatz / f"seed{seed}"
    return analysis / f"k{top_k}" / ansatz / f"seed{seed}"


def seeds_for(system: str, top_k: int, ansatz: str) -> tuple[int, ...]:
    return SEED_MANIFEST.get(system, {}).get(top_k, {}).get(ansatz, SEEDS)


def checked_visible_agreement(
    metrics: dict[str, Any],
    *,
    system: str,
    top_k: int,
    ansatz: str,
    seed: int,
) -> tuple[float, float]:
    agreement = float(metrics["supplied"]["directional_agreement"])
    projective_error = max(0.0, 1.0 - agreement * agreement)
    if projective_error > MAX_VISIBLE_PROJECTIVE_ERROR:
        raise ValueError(
            f"{system} K={top_k} {ansatz} seed={seed} lies outside "
            f"the visible-error shell: 1-C_V^2={projective_error:.3e} "
            f"> {MAX_VISIBLE_PROJECTIVE_ERROR:.3e}"
        )
    return agreement, projective_error


def band_rows(
    analysis: Path,
    *,
    system: str,
    top_k: int,
    legacy_k512: bool = False,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for ansatz, (_, label) in ANSATZES.items():
        for seed in seeds_for(system, top_k, ansatz):
            directory = seed_dir(
                analysis,
                ansatz=ansatz,
                seed=seed,
                top_k=top_k,
                legacy_k512=legacy_k512,
            )
            wave_path = directory / "pool_wavefunction.npz"
            metrics_path = directory / "pool_metrics.json"
            if not wave_path.exists() or not metrics_path.exists():
                continue
            with np.load(wave_path) as payload:
                model = payload["model_coefficients"].astype(np.float64)
                teacher = payload["teacher_coefficients"].astype(np.float64)
                supplied = payload["supplied_indices"].astype(np.int64)
            if float(np.dot(model[supplied], teacher[supplied])) < 0.0:
                model = -model
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            visible, visible_error = checked_visible_agreement(
                metrics,
                system=system,
                top_k=top_k,
                ansatz=ansatz,
                seed=seed,
            )
            for band in half_octave_bands(len(teacher)):
                left = int(band["slice_start"])
                right = int(band["slice_stop"])
                rows.append(
                    {
                        "system": system,
                        "ansatz": ansatz,
                        "ansatz_label": label,
                        "seed": seed,
                        "top_k": top_k,
                        "visible_directional_agreement": visible,
                        "visible_projective_error": visible_error,
                        "band_id": band["band_id"],
                        "r": band["r"],
                        "rank_start": band["rank_start"],
                        "rank_stop": band["rank_stop"],
                        "fully_visible": right <= top_k,
                        "fully_unseen": left >= top_k,
                        "band_directional_agreement": cosine(
                            model[left:right], teacher[left:right]
                        ),
                        "teacher_band_mass": float(
                            np.sum(teacher[left:right] ** 2)
                        ),
                    }
                )
    return rows


def reference_band_rows(audit_dir: Path, system: str) -> list[dict[str, Any]]:
    path = audit_dir / "result.json"
    if not path.exists():
        return []
    audit = json.loads(path.read_text(encoding="utf-8"))
    return [
        {
            "system": system,
            "band_id": row["band_id"],
            "r": float(row["r"]),
            "rank_start": int(row["rank_start"]),
            "rank_stop": int(row["rank_stop"]),
            "teacher_teacher_directional_agreement": float(
                row["directional_agreement"]
            ),
            "high_weight": float(row["high_weight"]),
            "low_weight": float(row["low_weight"]),
        }
        for row in audit.get("rank_bands", [])
    ]


def reference_audit_summary(
    audit_dir: Path,
    *,
    agreement_threshold: float = 0.99,
) -> dict[str, Any]:
    path = audit_dir / "result.json"
    if not path.exists():
        return {}
    audit = json.loads(path.read_text(encoding="utf-8"))
    top_k = int(audit.get("top_k", 0))

    def is_complete_half_octave(row: dict[str, Any]) -> bool:
        band_id = str(row["band_id"])
        octave = int(band_id[1:-1])
        suffix = band_id[-1]
        expected_stop = (
            int(round((2**octave) * math.sqrt(2.0)))
            if suffix == "a"
            else 2 ** (octave + 1)
        )
        return int(row["rank_stop"]) == expected_stop

    unseen_bands = sorted(
        (
            row
            for row in audit.get("rank_bands", [])
            if int(row["rank_start"]) > top_k
            and is_complete_half_octave(row)
        ),
        key=lambda row: int(row["rank_start"]),
    )
    trusted_band_stop = None
    for row in unseen_bands:
        if float(row["directional_agreement"]) < agreement_threshold:
            break
        trusted_band_stop = int(row["rank_stop"])
    supplied_weight = float(audit.get("supplied", {}).get("high_weight", math.nan))
    pool_weight = float(audit.get("pool", {}).get("high_weight", math.nan))
    return {
        "pool_count": int(audit.get("pool", {}).get("count", 0)),
        "pool_full_mps_weight": pool_weight,
        "supplied_full_mps_weight": supplied_weight,
        "supplied_fraction_of_selected_pool": (
            supplied_weight / pool_weight if pool_weight > 0.0 else math.nan
        ),
        "trusted_rank_stop": (
            trusted_band_stop
        ),
        "teacher_repeat_threshold": agreement_threshold,
        "teacher_repeat_rule": (
            "largest contiguous fully unseen half-octave band for which "
            "every preceding unseen band passes the threshold"
        ),
    }


def combined_reference_audit_summary(
    cross_d_audit: Path,
    same_d_audit: Path,
) -> dict[str, Any]:
    cross_d = reference_audit_summary(
        cross_d_audit,
        agreement_threshold=0.99,
    )
    same_d = reference_audit_summary(
        same_d_audit,
        agreement_threshold=0.999,
    )
    if not cross_d or not same_d:
        return {}
    trusted_stops = [
        int(summary["trusted_rank_stop"])
        for summary in (cross_d, same_d)
        if summary.get("trusted_rank_stop") is not None
    ]
    if len(trusted_stops) != 2:
        trusted_stop = None
    else:
        trusted_stop = min(trusted_stops)
    return {
        **cross_d,
        "trusted_rank_stop": trusted_stop,
        "trusted_rank_rule": (
            "minimum of the contiguous same-D repeat endpoint at 0.999 "
            "per band and the cross-D endpoint at 0.99 per band"
        ),
        "same_d_repeat": same_d,
        "cross_d": cross_d,
    }


def fe_energy_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        path = (
            FE_ENERGY_ANALYSIS
            / f"projected_subspace_energy_seed{seed}.json"
        )
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        fixed_count = int(payload["subspace_count"])
        for item in payload.get("teacher_prefix_curve", []):
            rows.append(
                {
                    "kind": "teacher",
                    "ansatz": "teacher",
                    "ansatz_label": "DMRG teacher",
                    "seed": seed,
                    "fixed_subspace_count": fixed_count,
                    "rank_stop": int(item["rank_stop"]),
                    "energy": float(item["energy"]),
                    "energy_excess": float(
                        item["energy_excess_over_projected_ground"]
                    ),
                }
            )
        for ansatz, (model_key, label) in ANSATZES.items():
            for item in payload["models"][model_key].get("prefix_curve", []):
                rows.append(
                    {
                        "kind": "support_expansion",
                        "ansatz": ansatz,
                        "ansatz_label": label,
                        "seed": seed,
                        "fixed_subspace_count": fixed_count,
                        "rank_stop": int(item["rank_stop"]),
                        "energy": float(item["energy"]),
                        "energy_excess": float(
                            item["energy_excess_over_projected_ground"]
                        ),
                    }
                )
    if STATIC_ENERGY.exists():
        payload = json.loads(STATIC_ENERGY.read_text(encoding="utf-8"))
        for item in payload["models"]["four_slater"]["prefix_curve"]:
            rows.append(
                {
                    "kind": "support_expansion",
                    "ansatz": "visible_static_scaffold",
                    "ansatz_label": "supplied-only 1-Slater",
                    "seed": -1,
                    "fixed_subspace_count": int(payload["subspace_count"]),
                    "rank_stop": int(item["rank_stop"]),
                    "energy": float(item["energy"]),
                    "energy_excess": float(
                        item["energy_excess_over_projected_ground"]
                    ),
                }
            )
    return rows


def fe_k_scan_rows(
    *,
    common_start: int,
    common_stop: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    indices = np.arange(common_start - 1, common_stop, dtype=np.int64)
    for top_k in K_GRID:
        for ansatz, (_, label) in ANSATZES.items():
            for seed in FE_K_SCAN_SEEDS_BY_ANSATZ[ansatz][top_k]:
                directory = seed_dir(
                    FE_ANALYSIS,
                    ansatz=ansatz,
                    seed=seed,
                    top_k=top_k,
                    legacy_k512=FE_K512_LEGACY,
                )
                wave_path = directory / "pool_wavefunction.npz"
                metrics_path = directory / "pool_metrics.json"
                if not metrics_path.exists():
                    continue
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                visible, visible_error = checked_visible_agreement(
                    metrics,
                    system="fe2s2",
                    top_k=top_k,
                    ansatz=ansatz,
                    seed=seed,
                )
                fixed = metrics.get("fixed_holdout", {})
                if (
                    int(fixed.get("rank_start", -1)) == common_start
                    and int(fixed.get("rank_stop", -1)) == common_stop
                ):
                    common_agreement = float(
                        fixed["directional_agreement"]
                    )
                else:
                    if not wave_path.exists():
                        raise ValueError(
                            f"{metrics_path} has neither the requested fixed "
                            "holdout nor a wavefunction fallback"
                        )
                    with np.load(wave_path) as payload:
                        model = payload["model_coefficients"].astype(
                            np.float64
                        )
                        teacher = payload["teacher_coefficients"].astype(
                            np.float64
                        )
                        supplied = payload["supplied_indices"].astype(
                            np.int64
                        )
                    if common_stop > len(teacher):
                        raise ValueError(
                            "common Fe holdout exceeds the frozen pool"
                        )
                    if (
                        float(np.dot(model[supplied], teacher[supplied]))
                        < 0.0
                    ):
                        model = -model
                    common_agreement = cosine(
                        model[indices], teacher[indices]
                    )
                rows.append(
                    {
                        "ansatz": ansatz,
                        "ansatz_label": label,
                        "seed": seed,
                        "top_k": top_k,
                        "visible_directional_agreement": visible,
                        "visible_projective_error": visible_error,
                        "common_rank_start": common_start,
                        "common_rank_stop": common_stop,
                        "common_directional_agreement": common_agreement,
                    }
                )
    coefficient_path = FE_AUDIT / "coefficients.npz"
    if coefficient_path.exists():
        with np.load(coefficient_path) as payload:
            high = payload["high_coefficients"].astype(np.float64)
            low = payload["lower_coefficients_aligned"].astype(np.float64)
        rows.append(
            {
                "ansatz": "dmrg_repeat",
                "ansatz_label": "DMRG repeat",
                "seed": -1,
                "top_k": -1,
                "visible_directional_agreement": float("nan"),
                "common_rank_start": common_start,
                "common_rank_stop": common_stop,
                "common_directional_agreement": cosine(
                    high[indices], low[indices]
                ),
            }
        )
    return rows


def n2_k_scan_rows(
    analysis: Path,
    *,
    common_start: int,
    common_stop: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    indices = np.arange(common_start - 1, common_stop, dtype=np.int64)
    for top_k in K_GRID:
        for ansatz, (_, label) in ANSATZES.items():
            for seed in seeds_for(N2_NAME, top_k, ansatz):
                directory = seed_dir(
                    analysis,
                    ansatz=ansatz,
                    seed=seed,
                    top_k=top_k,
                )
                wave_path = directory / "pool_wavefunction.npz"
                metrics_path = directory / "pool_metrics.json"
                if not metrics_path.exists():
                    continue
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                visible, visible_error = checked_visible_agreement(
                    metrics,
                    system=N2_NAME,
                    top_k=top_k,
                    ansatz=ansatz,
                    seed=seed,
                )
                fixed = metrics.get("fixed_holdout", {})
                if (
                    int(fixed.get("rank_start", -1)) == common_start
                    and int(fixed.get("rank_stop", -1)) == common_stop
                ):
                    common_agreement = float(
                        fixed["directional_agreement"]
                    )
                else:
                    if not wave_path.exists():
                        raise ValueError(
                            f"{metrics_path} has neither the requested fixed "
                            "holdout nor a wavefunction fallback"
                        )
                    with np.load(wave_path) as payload:
                        model = payload["model_coefficients"].astype(
                            np.float64
                        )
                        teacher = payload["teacher_coefficients"].astype(
                            np.float64
                        )
                        supplied = payload["supplied_indices"].astype(
                            np.int64
                        )
                    if common_stop > len(teacher):
                        raise ValueError(
                            "common N2 holdout exceeds the frozen pool"
                        )
                    if (
                        float(np.dot(model[supplied], teacher[supplied]))
                        < 0.0
                    ):
                        model = -model
                    common_agreement = cosine(
                        model[indices], teacher[indices]
                    )
                rows.append(
                    {
                        "ansatz": ansatz,
                        "ansatz_label": label,
                        "seed": seed,
                        "top_k": top_k,
                        "visible_directional_agreement": visible,
                        "visible_projective_error": visible_error,
                        "common_rank_start": common_start,
                        "common_rank_stop": common_stop,
                        "common_directional_agreement": common_agreement,
                    }
                )
    return rows


def prefix_fraction(pool_wavefunction: Path, top_k: int) -> float:
    with np.load(pool_wavefunction) as payload:
        teacher = payload["teacher_coefficients"].astype(np.float64)
    denominator = float(np.dot(teacher, teacher))
    return float(np.dot(teacher[:top_k], teacher[:top_k]) / denominator)


def static_profile_holdout(
    system: str,
    *,
    top_k: int,
    rank_stop: int,
) -> float:
    analysis = (
        N2_STATIC_ANALYSIS
        if system == N2_NAME
        else FE_PROFILE_STATIC_ANALYSIS
    )
    path = analysis / f"{system}_k{top_k}_static_scaffold.npz"
    if not path.exists():
        scan = pd.read_csv(analysis / "k_scan.csv")
        row = scan[
            scan["system"].eq(system)
            & scan["top_k"].eq(top_k)
            & scan["common_rank_start"].eq(top_k + 1)
            & scan["common_rank_stop"].eq(rank_stop)
        ]
        if len(row) != 1:
            raise FileNotFoundError(
                f"missing static profile state and unique k-scan fallback: {path}"
            )
        return float(row.iloc[0]["common_directional_agreement"])
    with np.load(path) as payload:
        scaffold = payload["scaffold_coefficients"].astype(np.float64)
        teacher = payload["teacher_coefficients"].astype(np.float64)
    indices = slice(top_k, rank_stop)
    return cosine(scaffold[indices], teacher[indices])


def main() -> None:
    global FE_ANALYSIS, FE_PROFILE_ANALYSIS, FE_ENERGY_ANALYSIS
    global FE_AUDIT, FE_SAME_D_AUDIT
    global STATIC_ANALYSIS, FE_PROFILE_STATIC_ANALYSIS
    global N2_STATIC_ANALYSIS, STATIC_ENERGY
    global N2_ANALYSIS, N2_AUDIT, N2_SAME_D_AUDIT, K_GRID
    global FE_K512_LEGACY, FE_K_SCAN_SEEDS_BY_ANSATZ, SEED_MANIFEST
    global MAX_VISIBLE_PROJECTIVE_ERROR
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile-top-k",
        dest="profile_top_k",
        type=int,
        default=4096,
    )
    parser.add_argument("--n2-profile-top-k", type=int, default=4096)
    parser.add_argument(
        "--k-grid",
        type=int,
        nargs="+",
        default=list(K_GRID),
    )
    parser.add_argument("--fe-common-start", type=int, default=4097)
    parser.add_argument(
        "--fe-common-stop",
        type=int,
        required=True,
        help=(
            "Teacher-audited endpoint of the fixed Fe holdout.  This is "
            "required so the final figure cannot silently fall back to the "
            "obsolete rank-8192 pilot."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "analysis/fig3_selected_spaces",
    )
    parser.add_argument(
        "--fe-analysis",
        type=Path,
        default=FE_ANALYSIS,
    )
    parser.add_argument(
        "--fe-profile-analysis",
        type=Path,
        default=None,
        help="Optional separate Fe analysis directory for panel (a).",
    )
    parser.add_argument(
        "--fe-energy-analysis",
        type=Path,
        default=FE_ENERGY_ANALYSIS,
        help=(
            "Directory containing the independent K=512, P4096 cumulative "
            "energy JSON files."
        ),
    )
    parser.add_argument(
        "--fe-energy-top-k",
        type=int,
        default=512,
        help="Legacy projected-energy prefix size (unused if no energy tables are present).",
    )
    parser.add_argument(
        "--fe-audit",
        type=Path,
        default=FE_AUDIT,
    )
    parser.add_argument(
        "--fe-same-d-audit",
        type=Path,
        default=FE_SAME_D_AUDIT,
    )
    parser.add_argument(
        "--n2-analysis",
        type=Path,
        default=N2_ANALYSIS,
    )
    parser.add_argument(
        "--n2-audit",
        type=Path,
        default=N2_AUDIT,
    )
    parser.add_argument(
        "--n2-same-d-audit",
        type=Path,
        default=N2_SAME_D_AUDIT,
    )
    parser.add_argument("--n2-common-start", type=int, default=4097)
    parser.add_argument(
        "--n2-common-stop",
        type=int,
        help="Default: use the full frozen N2 pool.",
    )
    parser.add_argument(
        "--static-analysis",
        type=Path,
        default=STATIC_ANALYSIS,
    )
    parser.add_argument(
        "--fe-profile-static-analysis",
        type=Path,
        default=None,
        help="Optional separate Fe supplied-only profile for panel (a).",
    )
    parser.add_argument(
        "--n2-static-analysis",
        type=Path,
        default=N2_STATIC_ANALYSIS,
        help=(
            "Optional separate directory for the N2 supplied-only "
            "1-Slater profile."
        ),
    )
    parser.add_argument(
        "--static-energy",
        type=Path,
        default=STATIC_ENERGY,
    )
    parser.add_argument(
        "--fe-k512-nested",
        action="store_true",
        help="Read Fe K=512 results from k512/<ansatz>/seed<seed>.",
    )
    parser.add_argument(
        "--fe-k256-use-standard-seeds",
        action="store_true",
        help="Use seeds 20260790--20260794 at Fe K=256.",
    )
    parser.add_argument(
        "--fe-direct-k1024-seeds",
        default="",
        help=(
            "Optional comma-separated five-seed list for the Fe direct "
            "Transformer at K=1024."
        ),
    )
    parser.add_argument(
        "--seed-manifest",
        type=Path,
        help=(
            "Optional JSON mapping system -> K -> ansatz -> seed list. "
            "This records paired replacement seeds without silently mixing "
            "failed and replacement runs."
        ),
    )
    parser.add_argument(
        "--max-visible-projective-error",
        type=float,
        default=MAX_VISIBLE_PROJECTIVE_ERROR,
        help=(
            "Abort if any selected neural result has 1-C_V^2 above this "
            "limit.  The default allows only a narrow numerical shell around "
            "the declared 1e-5 checkpoint."
        ),
    )
    args = parser.parse_args()
    FE_ANALYSIS = args.fe_analysis
    FE_PROFILE_ANALYSIS = args.fe_profile_analysis or FE_ANALYSIS
    FE_ENERGY_ANALYSIS = args.fe_energy_analysis
    FE_AUDIT = args.fe_audit
    FE_SAME_D_AUDIT = args.fe_same_d_audit
    N2_ANALYSIS = args.n2_analysis
    N2_AUDIT = args.n2_audit
    N2_SAME_D_AUDIT = args.n2_same_d_audit
    STATIC_ANALYSIS = args.static_analysis
    FE_PROFILE_STATIC_ANALYSIS = (
        args.fe_profile_static_analysis or STATIC_ANALYSIS
    )
    N2_STATIC_ANALYSIS = args.n2_static_analysis
    STATIC_ENERGY = args.static_energy
    MAX_VISIBLE_PROJECTIVE_ERROR = float(
        args.max_visible_projective_error
    )
    if MAX_VISIBLE_PROJECTIVE_ERROR <= 0.0:
        raise ValueError("--max-visible-projective-error must be positive")
    K_GRID = tuple(args.k_grid)
    if not K_GRID or tuple(sorted(set(K_GRID))) != K_GRID:
        raise ValueError("--k-grid must be distinct and strictly increasing")
    energy_paths = sorted(
        FE_ENERGY_ANALYSIS.glob("projected_subspace_energy_seed*.json")
    )
    energy_subspace_count = None
    if energy_paths:
        energy_metadata = json.loads(
            energy_paths[0].read_text(encoding="utf-8")
        )
        energy_prefix_counts = tuple(
            int(value) for value in energy_metadata["prefix_counts"]
        )
        if args.fe_energy_top_k not in energy_prefix_counts:
            raise ValueError(
                "--fe-energy-top-k must be one of the evaluated projected "
                f"supports, got {args.fe_energy_top_k} versus "
                f"{energy_prefix_counts}"
            )
        energy_subspace_count = int(energy_metadata["subspace_count"])
    SEED_MANIFEST = {}
    if args.seed_manifest is not None:
        raw_manifest = json.loads(
            args.seed_manifest.read_text(encoding="utf-8")
        )
        for system, system_rows in raw_manifest.items():
            SEED_MANIFEST[system] = {}
            for top_k_text, ansatz_rows in system_rows.items():
                top_k = int(top_k_text)
                SEED_MANIFEST[system][top_k] = {}
                for ansatz, raw_seeds in ansatz_rows.items():
                    if ansatz not in ANSATZES:
                        raise ValueError(
                            f"unknown ansatz {ansatz!r} in seed manifest"
                        )
                    seeds = tuple(int(seed) for seed in raw_seeds)
                    if len(seeds) != 5 or len(set(seeds)) != 5:
                        raise ValueError(
                            "each seed-manifest entry must contain five "
                            "distinct seeds"
                        )
                    SEED_MANIFEST[system][top_k][ansatz] = seeds
    FE_K_SCAN_SEEDS_BY_ANSATZ = {
        ansatz: {
            top_k: seeds_for("fe2s2", top_k, ansatz)
            for top_k in K_GRID
        }
        for ansatz in ANSATZES
    }
    FE_K512_LEGACY = not args.fe_k512_nested
    if args.fe_k256_use_standard_seeds:
        for ansatz in ANSATZES:
            if 256 in FE_K_SCAN_SEEDS_BY_ANSATZ[ansatz]:
                FE_K_SCAN_SEEDS_BY_ANSATZ[ansatz][256] = SEEDS
    if args.fe_direct_k1024_seeds:
        direct_seeds = tuple(
            int(item.strip())
            for item in args.fe_direct_k1024_seeds.split(",")
            if item.strip()
        )
        if len(direct_seeds) != 5 or len(set(direct_seeds)) != 5:
            raise ValueError(
                "--fe-direct-k1024-seeds must contain five distinct seeds"
            )
        FE_K_SCAN_SEEDS_BY_ANSATZ[
            "transformer_direct_phase"
        ][1024] = direct_seeds

    n2_analysis = N2_ANALYSIS
    n2_audit = N2_AUDIT
    n2_reference_summary = combined_reference_audit_summary(
        n2_audit,
        N2_SAME_D_AUDIT,
    )
    fe_reference_summary = combined_reference_audit_summary(
        FE_AUDIT,
        FE_SAME_D_AUDIT,
    )
    if not fe_reference_summary or not n2_reference_summary:
        raise ValueError("both same-D and cross-D teacher audits are required")
    fe_trusted_stop = fe_reference_summary.get("trusted_rank_stop")
    if (
        fe_trusted_stop is None
        or int(args.fe_common_stop) > int(fe_trusted_stop)
    ):
        raise ValueError(
            "--fe-common-stop exceeds the endpoint jointly supported by "
            "the same-D and cross-D teacher audits"
        )
    fe_static_bands = pd.read_csv(
        FE_PROFILE_STATIC_ANALYSIS / "band_metrics.csv"
    )
    fe_static_scan = pd.read_csv(STATIC_ANALYSIS / "k_scan.csv")
    n2_static_bands = pd.read_csv(N2_STATIC_ANALYSIS / "band_metrics.csv")
    n2_static_scan = pd.read_csv(N2_STATIC_ANALYSIS / "k_scan.csv")
    static_bands = pd.concat(
        [
            fe_static_bands[
                fe_static_bands["system"].eq("fe2s2")
                & fe_static_bands["top_k"].eq(args.profile_top_k)
            ],
            n2_static_bands[n2_static_bands["system"].eq(N2_NAME)],
        ],
        ignore_index=True,
    )
    static_scan = pd.concat(
        [
            fe_static_scan[fe_static_scan["system"].eq("fe2s2")],
            n2_static_scan[n2_static_scan["system"].eq(N2_NAME)],
        ],
        ignore_index=True,
    )
    static_bands["ansatz_label"] = "supplied-only 1-Slater scaffold"
    static_scan["ansatz_label"] = "supplied-only 1-Slater scaffold"
    fe_bands = pd.DataFrame(
        band_rows(
            FE_PROFILE_ANALYSIS,
            system="fe2s2",
            top_k=args.profile_top_k,
            legacy_k512=FE_K512_LEGACY,
        )
    )
    fe_bands = pd.concat(
        [
            fe_bands,
            static_bands[static_bands["system"].eq("fe2s2")],
        ],
        ignore_index=True,
    )
    n2_bands = pd.DataFrame(
        band_rows(
            n2_analysis,
            system=N2_NAME,
            top_k=args.n2_profile_top_k,
        )
    )
    n2_bands = pd.concat(
        [
            n2_bands,
            static_bands[static_bands["system"].eq(N2_NAME)],
        ],
        ignore_index=True,
    )
    fe_scan = pd.DataFrame(
        fe_k_scan_rows(
            common_start=args.fe_common_start,
            common_stop=args.fe_common_stop,
        )
    )
    fe_scan = pd.concat(
        [
            fe_scan,
            static_scan[static_scan["system"].eq("fe2s2")],
        ],
        ignore_index=True,
    )
    n2_common_start = args.n2_common_start
    n2_common_stop = (
        int(args.n2_common_stop)
        if args.n2_common_stop is not None
        else int(n2_reference_summary["trusted_rank_stop"])
    )
    if n2_common_stop > int(n2_reference_summary["trusted_rank_stop"]):
        raise ValueError(
            "--n2-common-stop exceeds the endpoint jointly supported by "
            "the same-D and cross-D teacher audits"
        )
    n2_scan = pd.DataFrame(
        n2_k_scan_rows(
            n2_analysis,
            common_start=n2_common_start,
            common_stop=n2_common_stop,
        )
    )
    n2_scan = pd.concat(
        [
            n2_scan,
            static_scan[static_scan["system"].eq(N2_NAME)],
        ],
        ignore_index=True,
    )
    tables = {
        "fe_band_metrics.csv": fe_bands,
        "fe_reference_bands.csv": pd.DataFrame(
            reference_band_rows(FE_AUDIT, "fe2s2")
        ),
        "fe_k_scan.csv": fe_scan,
        "n2_band_metrics.csv": n2_bands,
        "n2_k_scan.csv": n2_scan,
        "n2_reference_bands.csv": pd.DataFrame(
            reference_band_rows(n2_audit, N2_NAME)
        ),
    }
    if energy_paths:
        tables["fe_energy_support.csv"] = pd.DataFrame(fe_energy_rows())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename, table in tables.items():
        table.to_csv(args.output_dir / filename, index=False)

    report = {
        "fe_top_k": args.profile_top_k,
        "k_grid": list(K_GRID),
        "fe_display_rank_stop": args.fe_common_stop,
        "fe_common_holdout": [
            args.fe_common_start,
            args.fe_common_stop,
        ],
        "fe_reference": fe_reference_summary,
        "fe_supplied_fraction_of_selected_pool": prefix_fraction(
            seed_dir(
                FE_PROFILE_ANALYSIS,
                ansatz="nnbf",
                seed=seeds_for(
                    "fe2s2", args.profile_top_k, "nnbf"
                )[0],
                top_k=args.profile_top_k,
                legacy_k512=FE_K512_LEGACY,
            )
            / "pool_wavefunction.npz",
            args.profile_top_k,
        ),
        "n2_top_k": args.n2_profile_top_k,
        "n2_common_holdout": [n2_common_start, n2_common_stop],
        "n2_reference": n2_reference_summary,
        "static_scaffold": {
            "definition": (
                "one static fixed-spin Slater determinant fitted and "
                "restart-selected using supplied coefficients only; profile "
                "agreements use the raw fitted determinant"
            ),
            "fe_profile_holdout": [
                args.profile_top_k + 1,
                args.fe_common_stop,
            ],
            "fe_profile_holdout_agreement": static_profile_holdout(
                "fe2s2",
                top_k=args.profile_top_k,
                rank_stop=args.fe_common_stop,
            ),
            "n2_profile_holdout": [
                args.n2_profile_top_k + 1,
                n2_common_stop,
            ],
            "n2_profile_holdout_agreement": static_profile_holdout(
                N2_NAME,
                top_k=args.n2_profile_top_k,
                rank_stop=n2_common_stop,
            ),
        },
        "row_counts": {
            filename: int(len(table)) for filename, table in tables.items()
        },
    }
    if energy_paths:
        report["fe_energy_top_k"] = args.fe_energy_top_k
        report["fe_energy_subspace_count"] = energy_subspace_count
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
