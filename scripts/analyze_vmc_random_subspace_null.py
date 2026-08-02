#!/usr/bin/env python3
"""Haar-random equal-dimension null for off-root VMC gain attribution.

For each selected finite VMC update, this script draws independent Haar-random
rank-k subspaces of the batch complement A_t and computes the exact same
path-integrated projected energy gain used for the closest-Slater orbital
tangent.  The random subspaces depend only on A_t, their dimension, and a
recorded random seed; they do not use the update direction, Hamiltonian
residual, post-update state, or closest-Slater fit.

The implementation samples the restriction of a Haar projector to the
low-rank span needed by the energy bilinear.  This is distributionally exact
but avoids materializing an external_count by k matrix for every null draw.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT))

from analyze_vmc_direct_jvp_ledger import (  # noqa: E402
    direct_normalized_jvp,
    histories_and_future,
    interpolated_state,
    normalized_state_from_dict,
    rayleigh,
)
from analyze_vmc_finite_completion_path import (  # noqa: E402
    state_plus_full_direction,
)
from evaluate_checkpoints import build_model  # noqa: E402
from evaluate_vmc_completion_checkpoints import hamiltonian_action  # noqa: E402


DEFAULT_EPOCHS = (11, 13, 15, 17, 20, 23, 27, 32, 37, 43, 50, 58, 67, 78, 90)


def comma_ints(value: str) -> tuple[int, ...]:
    result = tuple(int(item) for item in value.split(",") if item.strip())
    if not result:
        raise ValueError("at least one epoch is required")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=comma_ints, default=DEFAULT_EPOCHS)
    parser.add_argument("--quadrature-order", type=int, default=8)
    parser.add_argument("--random-subspace-dimension", type=int, default=50)
    parser.add_argument("--random-draws", type=int, default=256)
    parser.add_argument("--null-seed-base", type=int, default=2026072800)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--chunk-size", type=int, default=4096)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def bartlett_wishart_identity(
    rng: np.random.Generator,
    *,
    dimension: int,
    degrees_of_freedom: int,
) -> np.ndarray:
    """Sample Wishart_p(df, I) by the Bartlett decomposition."""
    if degrees_of_freedom < dimension:
        raise ValueError("Wishart degrees of freedom must be >= dimension")
    lower = np.zeros((dimension, dimension), dtype=np.float64)
    lower[np.diag_indices(dimension)] = np.sqrt(
        rng.chisquare(
            degrees_of_freedom - np.arange(dimension, dtype=np.int64)
        )
    )
    row_indices, column_indices = np.tril_indices(dimension, k=-1)
    lower[row_indices, column_indices] = rng.standard_normal(
        row_indices.size
    )
    return lower @ lower.T


def energy_bilinear_in_span(
    tangents: np.ndarray,
    ideals: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, float, int]:
    """Return the symmetric energy bilinear in its numerical column span."""
    if tangents.shape != ideals.shape:
        raise ValueError("tangent and residual node matrices differ")
    if tangents.ndim != 2 or tangents.shape[1] != weights.size:
        raise ValueError("unexpected node-matrix shape")
    combined = np.concatenate((tangents, ideals), axis=1)
    left, singular_values, _ = np.linalg.svd(
        combined,
        full_matrices=False,
    )
    tolerance = (
        singular_values[0]
        * max(combined.shape)
        * np.finfo(np.float64).eps
    )
    rank = int(np.count_nonzero(singular_values > tolerance))
    span = left[:, :rank]
    tangent_coordinates = span.T @ tangents
    ideal_coordinates = span.T @ ideals
    bilinear = np.zeros((rank, rank), dtype=np.float64)
    for index, weight in enumerate(weights):
        tangent = tangent_coordinates[:, index]
        ideal = ideal_coordinates[:, index]
        bilinear += float(weight) * (
            np.outer(ideal, tangent) + np.outer(tangent, ideal)
        )
    gain = float(np.trace(bilinear))
    direct_gain = 2.0 * float(
        sum(
            float(weight) * np.dot(ideals[:, index], tangents[:, index])
            for index, weight in enumerate(weights)
        )
    )
    closure = gain - direct_gain
    if abs(closure) > 1.0e-10 * max(abs(direct_gain), 1.0):
        raise ValueError(f"low-rank energy-bilinear closure failed: {closure}")
    return bilinear, direct_gain, rank


def sample_haar_projected_gains(
    bilinear: np.ndarray,
    *,
    ambient_dimension: int,
    subspace_dimension: int,
    draws: int,
    seed: int,
) -> np.ndarray:
    """Sample tr(P B) for Haar-random rank-k projectors P in R^n.

    Only the restriction of P to the support of B is needed.  If A contains
    the corresponding rows of an n by k Gaussian matrix and W is an
    independent Wishart_k(n-rank, I), then

        P_support = A (A^T A + W)^(-1) A^T

    has exactly the required marginal distribution.
    """
    rank = int(bilinear.shape[0])
    if bilinear.shape != (rank, rank):
        raise ValueError("bilinear must be square")
    if not 0 < subspace_dimension <= ambient_dimension:
        raise ValueError("invalid random-subspace dimension")
    complement_degrees = ambient_dimension - rank
    if complement_degrees < subspace_dimension:
        raise ValueError("ambient complement is too small for exact sampler")
    rng = np.random.default_rng(seed)
    gains = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        top = rng.standard_normal((rank, subspace_dimension))
        complement_gram = bartlett_wishart_identity(
            rng,
            dimension=subspace_dimension,
            degrees_of_freedom=complement_degrees,
        )
        gram = top.T @ top + complement_gram
        compressed = top.T @ bilinear @ top
        gains[draw] = float(np.trace(np.linalg.solve(gram, compressed)))
    return gains


def main() -> None:
    args = parse_args()
    if args.quadrature_order <= 0:
        raise ValueError("--quadrature-order must be positive")
    if args.random_draws <= 0:
        raise ValueError("--random-draws must be positive")

    config_path = args.config.resolve()
    run_dir = args.run_dir.resolve()
    reference_dir = args.reference_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    architecture = str(cfg["ansatz"])
    run_seed = int(cfg["seed"])
    device = torch.device(args.device)
    model_dtype = (
        torch.float32 if str(cfg["model_fp"]) == "float32" else torch.float64
    )
    model = build_model(cfg, device, model_dtype)
    n_qubits = int(cfg["n_qubits"])
    with np.load(reference_dir / "basis.npz") as payload:
        basis_states = payload["states"].astype(np.uint64, copy=False)
    h_action = hamiltonian_action(reference_dir)
    current, _, _ = histories_and_future(
        run_dir,
        basis_states=basis_states,
        epochs=args.epochs,
        future_window=0,
    )
    reference_metadata = json.loads(
        (reference_dir / "reference.json").read_text(encoding="utf-8")
    )
    exact_energy = float(reference_metadata["E0"])

    raw_nodes, raw_weights = np.polynomial.legendre.leggauss(
        args.quadrature_order
    )
    nodes = 0.5 * (raw_nodes + 1.0)
    weights = 0.5 * raw_weights
    draw_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for epoch in args.epochs:
        payload_path = (
            run_dir / "step_payloads" / f"epoch_{epoch:06d}_step.pt"
        )
        payload = torch.load(
            payload_path,
            map_location="cpu",
            weights_only=False,
        )
        pre = payload["model_pre_state_dict"]
        post = payload["model_post_state_dict"]
        root = current[epoch]
        external = ~root
        external_count = int(np.count_nonzero(external))

        psi_pre = normalized_state_from_dict(
            model,
            pre,
            basis_states=basis_states,
            n_qubits=n_qubits,
            chunk_size=args.chunk_size,
            device=device,
            dtype=model_dtype,
        )
        energy_pre = rayleigh(psi_pre, h_action)
        tangent_nodes: list[np.ndarray] = []
        ideal_nodes: list[np.ndarray] = []
        for alpha in nodes:
            anchor = interpolated_state(pre, post, float(alpha))
            direction_endpoint = state_plus_full_direction(
                anchor,
                pre,
                post,
            )
            psi, tangent, _ = direct_normalized_jvp(
                model,
                anchor,
                direction_endpoint,
                basis_states=basis_states,
                n_qubits=n_qubits,
                chunk_size=args.chunk_size,
                device=device,
                dtype=model_dtype,
            )
            h_psi = np.asarray(h_action(psi), dtype=np.float64)
            energy = float(np.dot(psi, h_psi))
            ideal = -(h_psi - energy * psi)
            tangent_nodes.append(np.asarray(tangent[external], dtype=np.float64))
            ideal_nodes.append(np.asarray(ideal[external], dtype=np.float64))

        tangents = np.column_stack(tangent_nodes)
        ideals = np.column_stack(ideal_nodes)
        bilinear, gain_external, span_rank = energy_bilinear_in_span(
            tangents,
            ideals,
            weights,
        )
        null_seed = (
            int(args.null_seed_base)
            + 100_000 * (run_seed % 100_000)
            + int(epoch)
        )
        gains = sample_haar_projected_gains(
            bilinear,
            ambient_dimension=external_count,
            subspace_dimension=args.random_subspace_dimension,
            draws=args.random_draws,
            seed=null_seed,
        )
        fractions = gains / gain_external
        expected_fraction = (
            float(args.random_subspace_dimension) / external_count
        )
        for draw, (gain, fraction) in enumerate(zip(gains, fractions)):
            draw_rows.append(
                {
                    "architecture": architecture,
                    "run_seed": run_seed,
                    "epoch": int(epoch),
                    "draw": int(draw),
                    "null_seed": null_seed,
                    "external_count": external_count,
                    "random_subspace_dimension": int(
                        args.random_subspace_dimension
                    ),
                    "energy_pre": energy_pre,
                    "energy_error": energy_pre - exact_energy,
                    "gain_external": gain_external,
                    "gain_random": float(gain),
                    "random_fraction": float(fraction),
                }
            )
        quantiles = np.quantile(
            fractions,
            [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99],
        )
        summary_rows.append(
            {
                "architecture": architecture,
                "run_seed": run_seed,
                "epoch": int(epoch),
                "null_seed": null_seed,
                "external_count": external_count,
                "random_subspace_dimension": int(
                    args.random_subspace_dimension
                ),
                "random_draws": int(args.random_draws),
                "energy_pre": energy_pre,
                "energy_error": energy_pre - exact_energy,
                "gain_external": gain_external,
                "energy_bilinear_span_rank": span_rank,
                "expected_random_fraction": expected_fraction,
                "mean_random_fraction": float(np.mean(fractions)),
                "std_random_fraction": float(np.std(fractions, ddof=1)),
                "q01_random_fraction": float(quantiles[0]),
                "q05_random_fraction": float(quantiles[1]),
                "q25_random_fraction": float(quantiles[2]),
                "q50_random_fraction": float(quantiles[3]),
                "q75_random_fraction": float(quantiles[4]),
                "q95_random_fraction": float(quantiles[5]),
                "q99_random_fraction": float(quantiles[6]),
            }
        )
        print(
            f"{architecture} seed={run_seed} epoch={epoch} "
            f"Eerr={energy_pre - exact_energy:.6e} "
            f"G_A={gain_external:+.6e} "
            f"random median={quantiles[3]:+.3e} "
            f"q05:q95={quantiles[1]:+.3e}:{quantiles[5]:+.3e}",
            flush=True,
        )

    write_csv(out_dir / "random_subspace_null_draws.csv", draw_rows)
    write_csv(out_dir / "random_subspace_null_summary.csv", summary_rows)
    metadata = {
        "analysis": "equal-dimension Haar-random off-root subspace null",
        "architecture": architecture,
        "run_seed": run_seed,
        "config": str(config_path),
        "run_dir": str(run_dir),
        "reference_dir": str(reference_dir),
        "epochs": list(args.epochs),
        "quadrature_order": int(args.quadrature_order),
        "random_subspace_dimension": int(args.random_subspace_dimension),
        "random_draws": int(args.random_draws),
        "null_seed_base": int(args.null_seed_base),
        "selection_information": (
            "each random subspace is selected from the external support A_t, "
            "its dimension, and a recorded pseudorandom seed only"
        ),
        "excluded_information": [
            "update direction",
            "Hamiltonian residual",
            "post-update state",
            "closest-Slater fit",
            "ansatz internal representation",
        ],
        "sampler": (
            "distributionally exact restriction of a real Haar-Stiefel "
            "rank-k projector, sampled through Gaussian top rows and a "
            "Bartlett Wishart complement"
        ),
    }
    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
