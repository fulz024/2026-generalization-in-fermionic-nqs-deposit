#!/usr/bin/env python3
"""Path-integrated root/external completion along saved finite VMC steps."""

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
from evaluate_checkpoints import build_model  # noqa: E402
from evaluate_vmc_completion_checkpoints import hamiltonian_action  # noqa: E402


DEFAULT_EPOCHS = (1, 10, 20, 50, 100, 500, 2000)
DEFAULT_ORDERS = (2, 4, 8)


def comma_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(",") if item.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=comma_ints, default=DEFAULT_EPOCHS)
    parser.add_argument(
        "--quadrature-orders",
        type=comma_ints,
        default=DEFAULT_ORDERS,
    )
    parser.add_argument(
        "--scaffold",
        type=Path,
        default=None,
        help=(
            "Optional fixed normalized determinant scaffold. Its external "
            "restriction is used only for an exploratory additive attribution."
        ),
    )
    parser.add_argument(
        "--scaffold-subspace",
        type=Path,
        default=None,
        help=(
            "Optional NPZ with a target-independent coefficient-space "
            "`directions` matrix. The external tangent is projected onto its "
            "external restriction at each saved update."
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--chunk-size", type=int, default=4096)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def state_plus_full_direction(
    anchor: dict[str, torch.Tensor],
    pre: dict[str, torch.Tensor],
    post: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Return anchor + (post-pre), for evaluating a JVP at the anchor."""
    if anchor.keys() != pre.keys() or pre.keys() != post.keys():
        raise ValueError("anchor/pre/post model-state keys differ")
    result: dict[str, torch.Tensor] = {}
    for key in anchor:
        value = anchor[key]
        before = pre[key]
        after = post[key]
        if value.is_floating_point() or value.is_complex():
            result[key] = value + (after - before)
        else:
            if not torch.equal(before, after) or not torch.equal(value, before):
                raise ValueError(f"non-floating model state changed for {key}")
            result[key] = value
    return result


def normalized(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(value))
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError("cannot normalize an empty or non-finite vector")
    return value / norm


def projection_component(
    vector: np.ndarray,
    direction: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Project a masked vector onto one masked coefficient-space direction."""
    result = np.zeros_like(vector)
    restricted = direction[mask]
    norm2 = float(np.dot(restricted, restricted))
    if norm2 > 1.0e-30:
        result[mask] = restricted * (
            float(np.dot(restricted, vector[mask])) / norm2
        )
    return result


def restricted_subspace_basis(
    directions: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, int, float]:
    """Orthonormal basis for the restriction of fixed directions to a mask."""
    restricted = np.asarray(directions[mask, :], dtype=np.float64)
    singular_values = np.linalg.svd(restricted, compute_uv=False)
    if singular_values.size == 0 or singular_values[0] <= 0.0:
        return np.empty((restricted.shape[0], 0)), 0, float("nan")
    threshold = max(restricted.shape) * np.finfo(np.float64).eps * singular_values[0]
    rank = int(np.count_nonzero(singular_values > threshold))
    q, _ = np.linalg.qr(restricted, mode="reduced")
    q = q[:, :rank]
    condition = float(
        singular_values[0] / singular_values[rank - 1]
        if rank > 0
        else float("nan")
    )
    return q, rank, condition


def main() -> None:
    args = parse_args()
    if not args.epochs or not args.quadrature_orders:
        raise ValueError("epochs and quadrature orders must be nonempty")
    if min(args.quadrature_orders) <= 0:
        raise ValueError("quadrature orders must be positive")

    config_path = args.config.resolve()
    run_dir = args.run_dir.resolve()
    reference_dir = args.reference_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    device = torch.device(args.device)
    dtype = torch.float32 if str(cfg["model_fp"]) == "float32" else torch.float64
    model = build_model(cfg, device, dtype)
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

    scaffold: np.ndarray | None = None
    if args.scaffold is not None:
        scaffold = normalized(np.load(args.scaffold.resolve()))
        if scaffold.size != basis_states.size:
            raise ValueError("scaffold and basis dimensions differ")
    scaffold_subspace: np.ndarray | None = None
    if args.scaffold_subspace is not None:
        with np.load(args.scaffold_subspace.resolve(), allow_pickle=False) as payload:
            scaffold_subspace = np.asarray(
                payload["directions"],
                dtype=np.float64,
            )
        if (
            scaffold_subspace.ndim != 2
            or scaffold_subspace.shape[0] != basis_states.size
        ):
            raise ValueError("invalid scaffold-subspace directions matrix")

    summary_rows: list[dict[str, Any]] = []
    node_rows: list[dict[str, Any]] = []
    convergence_rows: list[dict[str, Any]] = []
    highest_order = max(args.quadrature_orders)

    for epoch in args.epochs:
        payload_path = run_dir / "step_payloads" / f"epoch_{epoch:06d}_step.pt"
        if not payload_path.exists():
            raise FileNotFoundError(payload_path)
        payload = torch.load(payload_path, map_location="cpu", weights_only=False)
        pre = payload["model_pre_state_dict"]
        post = payload["model_post_state_dict"]
        root = current[epoch]
        external = ~root
        external_scaffold_basis: np.ndarray | None = None
        external_scaffold_rank = 0
        external_scaffold_condition = float("nan")
        if scaffold_subspace is not None:
            (
                external_scaffold_basis,
                external_scaffold_rank,
                external_scaffold_condition,
            ) = restricted_subspace_basis(scaffold_subspace, external)

        psi_pre = normalized_state_from_dict(
            model,
            pre,
            basis_states=basis_states,
            n_qubits=n_qubits,
            chunk_size=args.chunk_size,
            device=device,
            dtype=dtype,
        )
        psi_post = normalized_state_from_dict(
            model,
            post,
            basis_states=basis_states,
            n_qubits=n_qubits,
            chunk_size=args.chunk_size,
            device=device,
            dtype=dtype,
        )
        if float(np.dot(psi_pre, psi_post)) < 0.0:
            psi_post = -psi_post
        energy_pre = rayleigh(psi_pre, h_action)
        energy_post = rayleigh(psi_post, h_action)
        exact_gain = energy_pre - energy_post
        overlap = float(np.clip(abs(np.dot(psi_pre, psi_post)), 0.0, 1.0))
        fs_angle = float(math.acos(overlap))

        best: dict[str, float] | None = None
        for order in args.quadrature_orders:
            raw_nodes, raw_weights = np.polynomial.legendre.leggauss(order)
            nodes = 0.5 * (raw_nodes + 1.0)
            weights = 0.5 * raw_weights
            integrals = {
                "gain_root": 0.0,
                "gain_external": 0.0,
                "gain_external_mass": 0.0,
                "gain_external_shape": 0.0,
                "gain_external_rhf_raw": 0.0,
                "gain_external_rhf_tangentized": 0.0,
                "gain_external_rhf_subspace": 0.0,
            }

            for node_index, (alpha, weight) in enumerate(zip(nodes, weights)):
                anchor = interpolated_state(pre, post, float(alpha))
                direction_endpoint = state_plus_full_direction(anchor, pre, post)
                psi, tangent, diagnostics = direct_normalized_jvp(
                    model,
                    anchor,
                    direction_endpoint,
                    basis_states=basis_states,
                    n_qubits=n_qubits,
                    chunk_size=args.chunk_size,
                    device=device,
                    dtype=dtype,
                )
                h_psi = np.asarray(h_action(psi), dtype=np.float64)
                energy = float(np.dot(psi, h_psi))
                ideal = -(h_psi - energy * psi)
                d_root = 2.0 * float(np.dot(ideal[root], tangent[root]))
                d_external = 2.0 * float(
                    np.dot(ideal[external], tangent[external])
                )
                psi_external = psi[external]
                tangent_external = tangent[external]
                ideal_external = ideal[external]
                external_mass = float(np.dot(psi_external, psi_external))
                mass_coefficient = (
                    float(np.dot(psi_external, tangent_external))
                    / external_mass
                    if external_mass > 1.0e-30
                    else 0.0
                )
                tangent_external_mass = mass_coefficient * psi_external
                tangent_external_shape = (
                    tangent_external - tangent_external_mass
                )
                d_external_mass = 2.0 * float(
                    np.dot(ideal_external, tangent_external_mass)
                )
                d_external_shape = 2.0 * float(
                    np.dot(ideal_external, tangent_external_shape)
                )
                d_rhf_raw = float("nan")
                d_rhf_tangentized = float("nan")
                d_rhf_subspace = float("nan")
                if scaffold is not None:
                    tangent_rhf_raw = projection_component(
                        tangent,
                        scaffold,
                        external,
                    )
                    scaffold_tangent = scaffold - float(
                        np.dot(psi, scaffold)
                    ) * psi
                    tangent_rhf_tangentized = projection_component(
                        tangent,
                        scaffold_tangent,
                        external,
                    )
                    d_rhf_raw = 2.0 * float(
                        np.dot(ideal[external], tangent_rhf_raw[external])
                    )
                    d_rhf_tangentized = 2.0 * float(
                        np.dot(
                            ideal[external],
                            tangent_rhf_tangentized[external],
                        )
                    )
                if external_scaffold_basis is not None:
                    tangent_external = tangent[external]
                    tangent_rhf_subspace = external_scaffold_basis @ (
                        external_scaffold_basis.T @ tangent_external
                    )
                    d_rhf_subspace = 2.0 * float(
                        np.dot(ideal[external], tangent_rhf_subspace)
                    )

                integrals["gain_root"] += float(weight) * d_root
                integrals["gain_external"] += float(weight) * d_external
                integrals["gain_external_mass"] += (
                    float(weight) * d_external_mass
                )
                integrals["gain_external_shape"] += (
                    float(weight) * d_external_shape
                )
                if scaffold is not None:
                    integrals["gain_external_rhf_raw"] += (
                        float(weight) * d_rhf_raw
                    )
                    integrals["gain_external_rhf_tangentized"] += (
                        float(weight) * d_rhf_tangentized
                    )
                if scaffold_subspace is not None:
                    integrals["gain_external_rhf_subspace"] += (
                        float(weight) * d_rhf_subspace
                    )
                if order == highest_order:
                    node_rows.append(
                        {
                            "epoch": int(epoch),
                            "quadrature_order": int(order),
                            "node_index": int(node_index),
                            "alpha": float(alpha),
                            "weight": float(weight),
                            "energy": energy,
                            "descent_density_root": d_root,
                            "descent_density_external": d_external,
                            "descent_density_external_mass": d_external_mass,
                            "descent_density_external_shape": d_external_shape,
                            "descent_density_total": d_root + d_external,
                            "descent_density_external_rhf_raw": d_rhf_raw,
                            "descent_density_external_rhf_tangentized": (
                                d_rhf_tangentized
                            ),
                            "descent_density_external_rhf_subspace": (
                                d_rhf_subspace
                            ),
                            "tangent_norm": diagnostics[
                                "normalized_tangent_norm"
                            ],
                            "tangent_orthogonality": diagnostics[
                                "normalized_tangent_orthogonality"
                            ],
                            "external_mass_coefficient": mass_coefficient,
                        }
                    )

            integrated_total = (
                integrals["gain_root"] + integrals["gain_external"]
            )
            closure_error = integrated_total - exact_gain
            convergence_rows.append(
                {
                    "epoch": int(epoch),
                    "quadrature_order": int(order),
                    "energy_pre": energy_pre,
                    "energy_post": energy_post,
                    "exact_finite_gain": exact_gain,
                    "fubini_study_angle": fs_angle,
                    **integrals,
                    "integrated_total_gain": integrated_total,
                    "closure_error": closure_error,
                    "closure_absolute_error": abs(closure_error),
                    "closure_relative_error": (
                        abs(closure_error) / max(abs(exact_gain), 1.0e-12)
                    ),
                }
            )
            if order == highest_order:
                best = {
                    **integrals,
                    "integrated_total_gain": integrated_total,
                    "closure_error": closure_error,
                    "closure_absolute_error": abs(closure_error),
                    "closure_relative_error": (
                        abs(closure_error) / max(abs(exact_gain), 1.0e-12)
                    ),
                }

        if best is None:
            raise RuntimeError("highest-order quadrature result was not produced")
        rhf_raw = best["gain_external_rhf_raw"]
        rhf_tangentized = best["gain_external_rhf_tangentized"]
        rhf_subspace = best["gain_external_rhf_subspace"]
        summary_rows.append(
            {
                "epoch": int(epoch),
                "root_count": int(np.count_nonzero(root)),
                "external_count": int(np.count_nonzero(external)),
                "energy_pre": energy_pre,
                "energy_post": energy_post,
                "exact_finite_gain": exact_gain,
                "fubini_study_angle": fs_angle,
                "quadrature_order": int(highest_order),
                "external_rhf_subspace_rank": external_scaffold_rank,
                "external_rhf_subspace_condition": (
                    external_scaffold_condition
                ),
                **best,
                "external_mass_shape_gain_closure": (
                    best["gain_external_mass"]
                    + best["gain_external_shape"]
                    - best["gain_external"]
                ),
                "gain_external_residual_rhf_raw": (
                    best["gain_external"] - rhf_raw
                    if scaffold is not None
                    else float("nan")
                ),
                "gain_external_residual_rhf_tangentized": (
                    best["gain_external"] - rhf_tangentized
                    if scaffold is not None
                    else float("nan")
                ),
                "gain_external_residual_rhf_subspace": (
                    best["gain_external"] - rhf_subspace
                    if scaffold_subspace is not None
                    else float("nan")
                ),
                "rhf_raw_fraction_of_external_gain": (
                    rhf_raw / best["gain_external"]
                    if scaffold is not None
                    and abs(best["gain_external"]) > 1.0e-12
                    else float("nan")
                ),
                "rhf_tangentized_fraction_of_external_gain": (
                    rhf_tangentized / best["gain_external"]
                    if scaffold is not None
                    and abs(best["gain_external"]) > 1.0e-12
                    else float("nan")
                ),
                "rhf_subspace_fraction_of_external_gain": (
                    rhf_subspace / best["gain_external"]
                    if scaffold_subspace is not None
                    and abs(best["gain_external"]) > 1.0e-12
                    else float("nan")
                ),
            }
        )
        print(
            f"epoch={epoch} exact={exact_gain:+.8e} "
            f"GR={best['gain_root']:+.8e} "
            f"GA={best['gain_external']:+.8e} "
            f"closure={best['closure_error']:+.3e}",
            flush=True,
        )

    write_csv(out_dir / "finite_completion_summary.csv", summary_rows)
    write_csv(out_dir / "finite_completion_nodes.csv", node_rows)
    write_csv(out_dir / "quadrature_convergence.csv", convergence_rows)
    metadata = {
        "analysis": "finite VMC completion by path-integrated exact energy derivative",
        "config": str(config_path),
        "run_dir": str(run_dir),
        "reference_dir": str(reference_dir),
        "epochs": list(args.epochs),
        "quadrature_orders": list(args.quadrature_orders),
        "state_count": int(basis_states.size),
        "root_definition": "unique configurations in the generating batch",
        "external_definition": "all basis configurations outside the sampled roots",
        "identity": "G_root + G_external = E_pre - E_post",
        "external_shape_definition": (
            "at each quadrature node, remove from the normalized external "
            "tangent its projection onto the external state restriction"
        ),
        "scaffold": str(args.scaffold.resolve()) if args.scaffold else None,
        "scaffold_subspace": (
            str(args.scaffold_subspace.resolve())
            if args.scaffold_subspace
            else None
        ),
        "scaffold_attribution": (
            "exploratory one-dimensional projection of the external tangent "
            "onto either the raw fixed RHF coefficient direction or its "
            "instantaneous globally tangentized direction"
            if scaffold is not None
            else None
        ),
    }
    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
