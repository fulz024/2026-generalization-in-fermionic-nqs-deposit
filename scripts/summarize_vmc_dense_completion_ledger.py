#!/usr/bin/env python3
"""Validate and summarize consecutive finite-completion ledgers."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS = (
    ROOT
    / "analysis/fig4_vmc/evidence"
    / "decision_021_dense_completion_ledger_31847_31857"
)
DEFAULT_OUT = (
    ROOT
    / "analysis/fig4_vmc"
    / "dense_completion_ledger"
)
ARCHITECTURES = ("mlp_direct", "nnbf")
EXPECTED_EPOCHS = np.arange(1, 101)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def parse_architecture_seed(path: Path) -> tuple[str, int]:
    match = re.fullmatch(r"(mlp_direct|nnbf)_seed(\d+)", path.parent.name)
    if match is None:
        raise ValueError(f"cannot parse architecture and seed from {path}")
    return match.group(1), int(match.group(2))


def load_runs(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries: list[pd.DataFrame] = []
    convergence: list[pd.DataFrame] = []
    for path in sorted(root.glob("*/finite_completion_summary.csv")):
        architecture, seed = parse_architecture_seed(path)
        current = pd.read_csv(path)
        current["architecture"] = architecture
        current["seed"] = seed
        summaries.append(current)

        audit = pd.read_csv(path.parent / "quadrature_convergence.csv")
        audit["architecture"] = architecture
        audit["seed"] = seed
        convergence.append(audit)
    if len(summaries) != 10:
        raise ValueError(f"expected ten dense ledgers under {root}")
    return (
        pd.concat(summaries, ignore_index=True),
        pd.concat(convergence, ignore_index=True),
    )


def add_cumulative_columns(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for (architecture, seed), current in frame.groupby(
        ["architecture", "seed"],
        sort=True,
    ):
        current = current.sort_values("epoch").copy()
        if not np.array_equal(
            current["epoch"].to_numpy(dtype=int),
            EXPECTED_EPOCHS,
        ):
            raise ValueError(
                f"nonconsecutive ledger for {architecture} seed {seed}"
            )
        initial_energy = float(current.iloc[0]["energy_pre"])
        current["cumulative_root_gain"] = current["gain_root"].cumsum()
        current["cumulative_external_gain"] = (
            current["gain_external"].cumsum()
        )
        if {
            "gain_external_mass",
            "gain_external_shape",
        }.issubset(current.columns):
            current["cumulative_external_mass_gain"] = (
                current["gain_external_mass"].cumsum()
            )
            current["cumulative_external_shape_gain"] = (
                current["gain_external_shape"].cumsum()
            )
        current["cumulative_integrated_gain"] = (
            current["cumulative_root_gain"]
            + current["cumulative_external_gain"]
        )
        current["cumulative_exact_gain"] = (
            initial_energy - current["energy_post"]
        )
        current["cumulative_closure_error"] = (
            current["cumulative_integrated_gain"]
            - current["cumulative_exact_gain"]
        )
        current["cumulative_external_fraction"] = (
            current["cumulative_external_gain"]
            / current["cumulative_exact_gain"]
        )
        if "cumulative_external_shape_gain" in current:
            current["cumulative_external_mass_fraction"] = (
                current["cumulative_external_mass_gain"]
                / current["cumulative_exact_gain"]
            )
            current["cumulative_external_shape_fraction"] = (
                current["cumulative_external_shape_gain"]
                / current["cumulative_exact_gain"]
            )
        current["cumulative_root_fraction"] = (
            current["cumulative_root_gain"]
            / current["cumulative_exact_gain"]
        )
        rows.append(current)
    return pd.concat(rows, ignore_index=True)


def validate_convergence(
    convergence: pd.DataFrame,
) -> dict[str, float]:
    counts = convergence.groupby(
        ["architecture", "seed", "quadrature_order"],
        sort=True,
    )["epoch"].nunique()
    if not counts.eq(100).all():
        raise ValueError("quadrature convergence table is incomplete")
    order8 = convergence[convergence["quadrature_order"].eq(8)]
    order4 = convergence[convergence["quadrature_order"].eq(4)]
    merged = order8.merge(
        order4,
        on=["architecture", "seed", "epoch"],
        suffixes=("_8", "_4"),
        validate="one_to_one",
    )
    return {
        "max_order8_step_closure_Ha": float(
            order8["closure_absolute_error"].max()
        ),
        "max_order4_to_8_root_change_Ha": float(
            (
                merged["gain_root_8"]
                - merged["gain_root_4"]
            ).abs().max()
        ),
        "max_order4_to_8_external_change_Ha": float(
            (
                merged["gain_external_8"]
                - merged["gain_external_4"]
            ).abs().max()
        ),
    }


def main() -> None:
    args = parse_args()
    runs = args.runs.resolve()
    out_dir = args.out_dir.resolve()
    summary, convergence = load_runs(runs)
    cumulative = add_cumulative_columns(summary)
    convergence_audit = validate_convergence(convergence)
    shape_audit: dict[str, float] = {}
    if {
        "gain_external_mass",
        "gain_external_shape",
    }.issubset(cumulative.columns):
        shape_audit["max_external_mass_shape_closure_Ha"] = float(
            (
                cumulative["gain_external_mass"]
                + cumulative["gain_external_shape"]
                - cumulative["gain_external"]
            ).abs().max()
        )

    seed_rows: list[dict[str, object]] = []
    for (architecture, seed), current in cumulative.groupby(
        ["architecture", "seed"],
        sort=True,
    ):
        final = current.sort_values("epoch").iloc[-1]
        seed_rows.append(
            {
                "architecture": architecture,
                "seed": int(seed),
                "cumulative_root_gain_Ha": float(
                    final["cumulative_root_gain"]
                ),
                "cumulative_external_gain_Ha": float(
                    final["cumulative_external_gain"]
                ),
                "cumulative_exact_gain_Ha": float(
                    final["cumulative_exact_gain"]
                ),
                "cumulative_root_fraction": float(
                    final["cumulative_root_fraction"]
                ),
                "cumulative_external_fraction": float(
                    final["cumulative_external_fraction"]
                ),
                **(
                    {
                        "cumulative_external_mass_fraction": float(
                            final["cumulative_external_mass_fraction"]
                        ),
                        "cumulative_external_shape_fraction": float(
                            final["cumulative_external_shape_fraction"]
                        ),
                    }
                    if "cumulative_external_shape_fraction" in final
                    else {}
                ),
                "final_closure_error_Ha": float(
                    final["cumulative_closure_error"]
                ),
                "max_absolute_cumulative_closure_error_Ha": float(
                    current["cumulative_closure_error"].abs().max()
                ),
            }
        )
    seed_summary = pd.DataFrame(seed_rows)

    architecture_summary: dict[str, dict[str, object]] = {}
    for architecture in ARCHITECTURES:
        current = seed_summary[
            seed_summary["architecture"].eq(architecture)
        ]
        architecture_summary[architecture] = {
            column: {
                "min": float(current[column].min()),
                "median": float(current[column].median()),
                "max": float(current[column].max()),
            }
            for column in (
                "cumulative_root_gain_Ha",
                "cumulative_external_gain_Ha",
                "cumulative_exact_gain_Ha",
                "cumulative_root_fraction",
                "cumulative_external_fraction",
                *(
                    (
                        "cumulative_external_mass_fraction",
                        "cumulative_external_shape_fraction",
                    )
                    if "cumulative_external_shape_fraction"
                    in seed_summary.columns
                    else ()
                ),
                "final_closure_error_Ha",
            )
        }

    audit = {
        "run_root": str(runs),
        "row_count": int(len(cumulative)),
        "architectures": architecture_summary,
        **convergence_audit,
        **shape_audit,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    cumulative.to_csv(
        out_dir / "dense_cumulative_ledger.csv",
        index=False,
    )
    seed_summary.to_csv(
        out_dir / "dense_cumulative_seed_summary.csv",
        index=False,
    )
    (out_dir / "summary.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
