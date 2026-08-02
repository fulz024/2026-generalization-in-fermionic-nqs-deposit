#!/usr/bin/env python3
"""Post-hoc closest-Slater attribution of finite VMC updates.

For each selected pre-update full wavefunction, this script fits the closest
single fixed-spin Slater determinant.  It then constructs the determinant
state plus all occupied-virtual orbital tangents at that fitted point and
projects the batch-external part of the exact normalized JVP onto this moving
subspace.  The same procedure is applied to any architecture.

The output separates two questions:

1. how much external wavefunction motion lies in the fitted Slater subspace;
2. how much external exact-energy gain is carried by that motion.

Their ratio is an enrichment diagnostic that guards against the tautology
"the leading component carries a leading fraction of the update."
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
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
    normalized,
    restricted_subspace_basis,
    state_plus_full_direction,
)
from evaluate_checkpoints import build_model  # noqa: E402
from evaluate_vmc_completion_checkpoints import hamiltonian_action  # noqa: E402
from s3_fit_static_scaffold import (  # noqa: E402
    FixedSpinMultiSlater,
    ReferenceData,
    absolute_cosine,
    fixed_spin_wavefunction_from_orbitals,
    load_reference,
    spin_one_body_rdm,
)


DEFAULT_EPOCHS = (10, 20, 50, 100)


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
    parser.add_argument("--quadrature-order", type=int, default=4)
    parser.add_argument("--fit-restarts", type=int, default=3)
    parser.add_argument("--fit-steps", type=int, default=1000)
    parser.add_argument("--fit-learning-rate", type=float, default=0.03)
    parser.add_argument("--fit-patience", type=int, default=200)
    parser.add_argument("--fit-min-delta", type=float, default=1.0e-11)
    parser.add_argument("--fit-seed-base", type=int, default=2026072700)
    parser.add_argument(
        "--fixed-scaffold-subspace",
        type=Path,
        default=None,
        help="Optional fixed state-plus-orbital-tangent subspace for comparison.",
    )
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


def masks_from_occupations(occupations: np.ndarray) -> np.ndarray:
    masks = np.zeros(occupations.shape[0], dtype=np.uint32)
    for row, occupied in enumerate(occupations):
        mask = 0
        for orbital in occupied:
            mask |= 1 << int(orbital)
        masks[row] = mask
    return masks


def natural_orbital_initialization(
    reference: ReferenceData,
    target: np.ndarray,
    fit_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return natural-orbital subspaces using only the fitted coefficients."""
    masked_target = np.zeros_like(target, dtype=np.float64)
    masked_target[fit_mask] = np.asarray(target, dtype=np.float64)[fit_mask]
    masked_target = normalized(masked_target)
    target_ab = (
        masked_target.reshape(
            reference.n_up_configs,
            reference.n_down_configs,
        )
        * reference.ordering_signs
    )
    up_masks = masks_from_occupations(reference.up_occupations)
    down_masks = masks_from_occupations(reference.down_occupations)
    up_rdm = spin_one_body_rdm(target_ab, up_masks, reference.n_sites)
    down_rdm = spin_one_body_rdm(target_ab.T, down_masks, reference.n_sites)
    up_values, up_vectors = np.linalg.eigh(up_rdm)
    down_values, down_vectors = np.linalg.eigh(down_rdm)
    up = up_vectors[:, np.argsort(up_values)[::-1][: reference.n_up]]
    down = down_vectors[:, np.argsort(down_values)[::-1][: reference.n_down]]
    return up, down


def fit_one_slater(
    reference: ReferenceData,
    target: np.ndarray,
    fit_mask: np.ndarray,
    *,
    seed: int,
    steps: int,
    learning_rate: float,
    patience: int,
    min_delta: float,
    device: torch.device,
    natural_initialization: tuple[np.ndarray, np.ndarray] | None,
) -> dict[str, Any]:
    """Fit one determinant by its directional agreement on ``fit_mask``."""
    dtype = torch.float64
    model = FixedSpinMultiSlater(
        1,
        reference.n_sites,
        reference.n_up,
        reference.n_down,
        seed=seed,
        device=device,
        dtype=dtype,
    )
    if natural_initialization is not None:
        up_initial, down_initial = natural_initialization
        with torch.no_grad():
            model.up_orbitals[0].copy_(
                torch.as_tensor(up_initial, device=device, dtype=dtype)
            )
            model.down_orbitals[0].copy_(
                torch.as_tensor(down_initial, device=device, dtype=dtype)
            )
            model.coefficients.fill_(1.0)
            model.retract_orbitals()

    up_occ = torch.as_tensor(
        reference.up_occupations,
        device=device,
        dtype=torch.long,
    )
    down_occ = torch.as_tensor(
        reference.down_occupations,
        device=device,
        dtype=torch.long,
    )
    signs = torch.as_tensor(
        reference.ordering_signs,
        device=device,
        dtype=dtype,
    )
    target_tensor = torch.as_tensor(
        np.asarray(target, dtype=np.float64).reshape(
            reference.n_up_configs,
            reference.n_down_configs,
        ),
        device=device,
        dtype=dtype,
    )
    fit_mask_tensor = torch.as_tensor(
        np.asarray(fit_mask, dtype=bool).reshape(
            reference.n_up_configs,
            reference.n_down_configs,
        ),
        device=device,
        dtype=torch.bool,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_objective = -float("inf")
    best_step = -1
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    trace: list[dict[str, float | int]] = []
    started = time.perf_counter()

    for step in range(steps + 1):
        optimizer.zero_grad(set_to_none=True)
        prediction = model.full_wavefunction(up_occ, down_occ) * signs
        objective = absolute_cosine(
            prediction[fit_mask_tensor],
            target_tensor[fit_mask_tensor],
        )
        value = float(objective.detach())
        if best_state is None or value > best_objective + min_delta:
            best_objective = value
            best_step = step
            best_state = {
                name: tensor.detach().clone()
                for name, tensor in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if step == 0 or step % 50 == 0 or step == steps:
            trace.append(
                {
                    "step": int(step),
                    "objective": value,
                    "loss": 1.0 - value,
                }
            )
        if step == steps or stale >= patience:
            break
        (1.0 - objective).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        model.retract_orbitals()

    if best_state is None:
        raise RuntimeError("closest-Slater fit did not produce a state")
    model.load_state_dict(best_state)
    model.retract_orbitals()
    up = model.up_orbitals[0].detach().cpu().numpy().copy()
    down = model.down_orbitals[0].detach().cpu().numpy().copy()
    wavefunction = fixed_spin_wavefunction_from_orbitals(reference, up, down)
    visible_signed_overlap = float(
        np.dot(wavefunction[fit_mask], target[fit_mask])
    )
    if visible_signed_overlap < 0.0:
        wavefunction = -wavefunction
        visible_signed_overlap = -visible_signed_overlap
    visible_objective = float(
        np.dot(wavefunction[fit_mask], target[fit_mask])
        / (
            np.linalg.norm(wavefunction[fit_mask])
            * np.linalg.norm(target[fit_mask])
        )
    )
    full_overlap = abs(float(np.dot(wavefunction, target)))
    return {
        "seed": int(seed),
        "best_step": int(best_step),
        "objective": visible_objective,
        "full_overlap_diagnostic": full_overlap,
        "optimizer_objective": best_objective,
        "elapsed_seconds": time.perf_counter() - started,
        "up_orbitals": up,
        "down_orbitals": down,
        "wavefunction": wavefunction,
        "trace": trace,
    }


def closest_slater_multistart(
    reference: ReferenceData,
    target: np.ndarray,
    fit_mask: np.ndarray,
    *,
    epoch: int,
    restarts: int,
    steps: int,
    learning_rate: float,
    patience: int,
    min_delta: float,
    seed_base: int,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    target = normalized(target)
    fit_mask = np.asarray(fit_mask, dtype=bool)
    if fit_mask.shape != target.shape or not np.any(fit_mask):
        raise ValueError("fit_mask must select coefficients from target")
    natural = natural_orbital_initialization(reference, target, fit_mask)
    fits: list[dict[str, Any]] = []
    for restart in range(restarts):
        fit = fit_one_slater(
            reference,
            target,
            fit_mask,
            seed=seed_base + 1000 * int(epoch) + restart,
            steps=steps,
            learning_rate=learning_rate,
            patience=patience,
            min_delta=min_delta,
            device=device,
            natural_initialization=natural if restart == 0 else None,
        )
        fit["restart"] = restart
        fit["initialization"] = "natural_orbitals" if restart == 0 else "random"
        fits.append(fit)
    best = max(fits, key=lambda item: float(item["objective"]))
    records = [
        {
            "epoch": int(epoch),
            "restart": int(fit["restart"]),
            "initialization": str(fit["initialization"]),
            "seed": int(fit["seed"]),
            "best_step": int(fit["best_step"]),
            "objective": float(fit["objective"]),
            "optimizer_objective": float(fit["optimizer_objective"]),
            "full_overlap_diagnostic": float(
                fit["full_overlap_diagnostic"]
            ),
            "elapsed_seconds": float(fit["elapsed_seconds"]),
            "selected": bool(fit is best),
        }
        for fit in fits
    ]
    return best, records


def orbital_complement(occupied: np.ndarray) -> np.ndarray:
    occupied = np.asarray(occupied, dtype=np.float64)
    _, _, vh = np.linalg.svd(occupied.T, full_matrices=True)
    complement = vh[occupied.shape[1] :, :].T
    error = np.linalg.norm(occupied.T @ complement)
    if error > 1.0e-10:
        raise ValueError(f"invalid occupied/virtual complement: {error}")
    return complement


def determinant_tangent_subspace(
    reference: ReferenceData,
    up_orbitals: np.ndarray,
    down_orbitals: np.ndarray,
) -> tuple[np.ndarray, list[str], dict[str, float | int]]:
    """State and all spin-conserving occupied-virtual determinant tangents."""
    up_virtual = orbital_complement(up_orbitals)
    down_virtual = orbital_complement(down_orbitals)
    state = fixed_spin_wavefunction_from_orbitals(
        reference,
        up_orbitals,
        down_orbitals,
    )
    directions = [state]
    labels = ["slater_state"]

    for spin, occupied, virtual in (
        ("alpha", up_orbitals, up_virtual),
        ("beta", down_orbitals, down_virtual),
    ):
        for occupied_index in range(occupied.shape[1]):
            for virtual_index in range(virtual.shape[1]):
                changed = occupied.copy()
                changed[:, occupied_index] = virtual[:, virtual_index]
                if spin == "alpha":
                    direction = fixed_spin_wavefunction_from_orbitals(
                        reference,
                        changed,
                        down_orbitals,
                    )
                else:
                    direction = fixed_spin_wavefunction_from_orbitals(
                        reference,
                        up_orbitals,
                        changed,
                    )
                directions.append(direction)
                labels.append(
                    f"{spin}_occ{occupied_index}_virt{virtual_index}"
                )

    matrix = np.column_stack(directions)
    gram = matrix.T @ matrix
    orthogonality_error = float(
        np.linalg.norm(gram - np.eye(matrix.shape[1]))
    )
    if orthogonality_error > 1.0e-8:
        raise ValueError(
            "fitted Slater state/tangent directions are not orthonormal: "
            f"{orthogonality_error}"
        )
    report: dict[str, float | int] = {
        "dimension_state_plus_orbitals": int(matrix.shape[1]),
        "dimension_orbitals_only": int(matrix.shape[1] - 1),
        "orthogonality_error": orthogonality_error,
    }
    return matrix, labels, report


def project_metrics(
    tangent_external: np.ndarray,
    ideal_external: np.ndarray,
    basis: np.ndarray,
) -> dict[str, float]:
    projection = basis @ (basis.T @ tangent_external)
    gain = 2.0 * float(np.dot(ideal_external, projection))
    return {
        "gain": gain,
        "motion_l1": float(np.linalg.norm(projection)),
        "motion_sq": float(np.dot(projection, projection)),
    }


def safe_ratio(numerator: float, denominator: float) -> float:
    if not math.isfinite(numerator) or not math.isfinite(denominator):
        return float("nan")
    if abs(denominator) <= 1.0e-14:
        return float("nan")
    return numerator / denominator


def main() -> None:
    args = parse_args()
    if args.quadrature_order <= 0:
        raise ValueError("--quadrature-order must be positive")
    if args.fit_restarts <= 0:
        raise ValueError("--fit-restarts must be positive")

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
    reference = load_reference(reference_dir)
    with np.load(reference_dir / "basis.npz") as payload:
        basis_states = payload["states"].astype(np.uint64, copy=False)
    if basis_states.size != reference.dimension:
        raise ValueError("reference loaders disagree on Hilbert-space dimension")

    h_action = hamiltonian_action(reference_dir)
    current, _, _ = histories_and_future(
        run_dir,
        basis_states=basis_states,
        epochs=args.epochs,
        future_window=0,
    )

    fixed_directions: np.ndarray | None = None
    if args.fixed_scaffold_subspace is not None:
        with np.load(
            args.fixed_scaffold_subspace.resolve(),
            allow_pickle=False,
        ) as payload:
            fixed_directions = np.asarray(
                payload["directions"],
                dtype=np.float64,
            )
        if fixed_directions.shape[0] != reference.dimension:
            raise ValueError("fixed scaffold and reference dimensions differ")

    fit_rows: list[dict[str, Any]] = []
    node_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    raw_nodes, raw_weights = np.polynomial.legendre.leggauss(
        args.quadrature_order
    )
    nodes = 0.5 * (raw_nodes + 1.0)
    weights = 0.5 * raw_weights

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

        psi_pre = normalized_state_from_dict(
            model,
            pre,
            basis_states=basis_states,
            n_qubits=n_qubits,
            chunk_size=args.chunk_size,
            device=device,
            dtype=model_dtype,
        )
        psi_post = normalized_state_from_dict(
            model,
            post,
            basis_states=basis_states,
            n_qubits=n_qubits,
            chunk_size=args.chunk_size,
            device=device,
            dtype=model_dtype,
        )
        if float(np.dot(psi_pre, psi_post)) < 0.0:
            psi_post = -psi_post
        energy_pre = rayleigh(psi_pre, h_action)
        energy_post = rayleigh(psi_post, h_action)
        exact_gain = energy_pre - energy_post

        fit, restart_rows = closest_slater_multistart(
            reference,
            psi_pre,
            root,
            epoch=epoch,
            restarts=args.fit_restarts,
            steps=args.fit_steps,
            learning_rate=args.fit_learning_rate,
            patience=args.fit_patience,
            min_delta=args.fit_min_delta,
            seed_base=args.fit_seed_base + run_seed,
            device=device,
        )
        for row in restart_rows:
            fit_rows.append(
                {
                    "architecture": architecture,
                    "run_seed": run_seed,
                    **row,
                }
            )

        moving_directions, moving_labels, tangent_report = (
            determinant_tangent_subspace(
                reference,
                np.asarray(fit["up_orbitals"]),
                np.asarray(fit["down_orbitals"]),
            )
        )
        moving_state_orbital, moving_full_rank, moving_full_condition = (
            restricted_subspace_basis(moving_directions, external)
        )
        moving_orbital, moving_orbital_rank, moving_orbital_condition = (
            restricted_subspace_basis(moving_directions[:, 1:], external)
        )
        fixed_basis: np.ndarray | None = None
        fixed_rank = 0
        fixed_condition = float("nan")
        if fixed_directions is not None:
            fixed_basis, fixed_rank, fixed_condition = (
                restricted_subspace_basis(fixed_directions, external)
            )

        np.savez_compressed(
            out_dir / f"epoch_{epoch:06d}_closest_slater.npz",
            up_orbitals=np.asarray(fit["up_orbitals"]),
            down_orbitals=np.asarray(fit["down_orbitals"]),
            wavefunction=np.asarray(fit["wavefunction"]),
            tangent_labels=np.asarray(moving_labels),
            tangent_report_json=np.asarray(json.dumps(tangent_report)),
        )

        integrals = {
            "gain_root": 0.0,
            "gain_external": 0.0,
            "gain_external_mass": 0.0,
            "gain_external_shape": 0.0,
            "motion_external_l1": 0.0,
            "motion_external_sq": 0.0,
            "motion_external_mass_sq": 0.0,
            "motion_external_shape_sq": 0.0,
            "gain_moving_state_orbital": 0.0,
            "motion_moving_state_orbital_l1": 0.0,
            "motion_moving_state_orbital_sq": 0.0,
            "gain_moving_orbital": 0.0,
            "motion_moving_orbital_l1": 0.0,
            "motion_moving_orbital_sq": 0.0,
            "gain_fixed_state_orbital": 0.0,
            "motion_fixed_state_orbital_l1": 0.0,
            "motion_fixed_state_orbital_sq": 0.0,
        }

        for node_index, (alpha, weight) in enumerate(zip(nodes, weights)):
            anchor = interpolated_state(pre, post, float(alpha))
            direction_endpoint = state_plus_full_direction(
                anchor,
                pre,
                post,
            )
            psi, tangent, diagnostics = direct_normalized_jvp(
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
            tangent_external = tangent[external]
            ideal_external = ideal[external]
            psi_external = psi[external]
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
            gain_root = 2.0 * float(np.dot(ideal[root], tangent[root]))
            gain_external = 2.0 * float(
                np.dot(ideal_external, tangent_external)
            )
            gain_external_mass = 2.0 * float(
                np.dot(ideal_external, tangent_external_mass)
            )
            gain_external_shape = 2.0 * float(
                np.dot(ideal_external, tangent_external_shape)
            )
            motion_external_l1 = float(np.linalg.norm(tangent_external))
            motion_external_sq = float(
                np.dot(tangent_external, tangent_external)
            )
            moving_full_metrics = project_metrics(
                tangent_external,
                ideal_external,
                moving_state_orbital,
            )
            moving_orbital_metrics = project_metrics(
                tangent_external,
                ideal_external,
                moving_orbital,
            )
            fixed_metrics = (
                project_metrics(tangent_external, ideal_external, fixed_basis)
                if fixed_basis is not None
                else {
                    "gain": float("nan"),
                    "motion_l1": float("nan"),
                    "motion_sq": float("nan"),
                }
            )

            weighted = float(weight)
            integrals["gain_root"] += weighted * gain_root
            integrals["gain_external"] += weighted * gain_external
            integrals["gain_external_mass"] += (
                weighted * gain_external_mass
            )
            integrals["gain_external_shape"] += (
                weighted * gain_external_shape
            )
            integrals["motion_external_l1"] += (
                weighted * motion_external_l1
            )
            integrals["motion_external_sq"] += (
                weighted * motion_external_sq
            )
            integrals["motion_external_mass_sq"] += weighted * float(
                np.dot(tangent_external_mass, tangent_external_mass)
            )
            integrals["motion_external_shape_sq"] += weighted * float(
                np.dot(tangent_external_shape, tangent_external_shape)
            )
            for prefix, metrics in (
                ("moving_state_orbital", moving_full_metrics),
                ("moving_orbital", moving_orbital_metrics),
                ("fixed_state_orbital", fixed_metrics),
            ):
                if math.isfinite(float(metrics["gain"])):
                    integrals[f"gain_{prefix}"] += (
                        weighted * float(metrics["gain"])
                    )
                    integrals[f"motion_{prefix}_l1"] += (
                        weighted * float(metrics["motion_l1"])
                    )
                    integrals[f"motion_{prefix}_sq"] += (
                        weighted * float(metrics["motion_sq"])
                    )

            node_rows.append(
                {
                    "architecture": architecture,
                    "run_seed": run_seed,
                    "epoch": int(epoch),
                    "node_index": int(node_index),
                    "alpha": float(alpha),
                    "weight": weighted,
                    "energy": energy,
                    "gain_density_root": gain_root,
                    "gain_density_external": gain_external,
                    "gain_density_external_mass": gain_external_mass,
                    "gain_density_external_shape": gain_external_shape,
                    "external_mass_coefficient": mass_coefficient,
                    "motion_external_l1": motion_external_l1,
                    "motion_external_sq": motion_external_sq,
                    "gain_density_moving_state_orbital": (
                        moving_full_metrics["gain"]
                    ),
                    "motion_moving_state_orbital_l1": (
                        moving_full_metrics["motion_l1"]
                    ),
                    "motion_moving_state_orbital_sq": (
                        moving_full_metrics["motion_sq"]
                    ),
                    "gain_density_moving_orbital": (
                        moving_orbital_metrics["gain"]
                    ),
                    "motion_moving_orbital_l1": (
                        moving_orbital_metrics["motion_l1"]
                    ),
                    "motion_moving_orbital_sq": (
                        moving_orbital_metrics["motion_sq"]
                    ),
                    "gain_density_fixed_state_orbital": fixed_metrics["gain"],
                    "motion_fixed_state_orbital_l1": fixed_metrics[
                        "motion_l1"
                    ],
                    "motion_fixed_state_orbital_sq": fixed_metrics[
                        "motion_sq"
                    ],
                    "tangent_norm": diagnostics[
                        "normalized_tangent_norm"
                    ],
                    "tangent_orthogonality": diagnostics[
                        "normalized_tangent_orthogonality"
                    ],
                }
            )

        integrated_total = (
            integrals["gain_root"] + integrals["gain_external"]
        )
        summary: dict[str, Any] = {
            "architecture": architecture,
            "run_seed": run_seed,
            "epoch": int(epoch),
            "root_count": int(np.count_nonzero(root)),
            "external_count": int(np.count_nonzero(external)),
            "energy_pre": energy_pre,
            "energy_post": energy_post,
            "exact_finite_gain": exact_gain,
            "quadrature_order": int(args.quadrature_order),
            "integrated_total_gain": integrated_total,
            "closure_error": integrated_total - exact_gain,
            "closure_relative_error": abs(integrated_total - exact_gain)
            / max(abs(exact_gain), 1.0e-12),
            "closest_slater_overlap": float(fit["objective"]),
            "closest_slater_full_overlap_diagnostic": float(
                fit["full_overlap_diagnostic"]
            ),
            "closest_slater_best_restart": int(fit["restart"]),
            "closest_slater_best_initialization": str(
                fit["initialization"]
            ),
            "moving_state_orbital_rank_external": moving_full_rank,
            "moving_state_orbital_condition_external": moving_full_condition,
            "moving_orbital_rank_external": moving_orbital_rank,
            "moving_orbital_condition_external": moving_orbital_condition,
            "fixed_state_orbital_rank_external": fixed_rank,
            "fixed_state_orbital_condition_external": fixed_condition,
            **integrals,
        }
        summary["external_shape_fraction_gain"] = safe_ratio(
            float(integrals["gain_external_shape"]),
            float(integrals["gain_external"]),
        )
        summary["external_shape_fraction_motion_sq"] = safe_ratio(
            float(integrals["motion_external_shape_sq"]),
            float(integrals["motion_external_sq"]),
        )
        summary["external_mass_shape_gain_closure"] = (
            float(integrals["gain_external_mass"])
            + float(integrals["gain_external_shape"])
            - float(integrals["gain_external"])
        )
        summary["external_mass_shape_motion_sq_closure"] = (
            float(integrals["motion_external_mass_sq"])
            + float(integrals["motion_external_shape_sq"])
            - float(integrals["motion_external_sq"])
        )
        for prefix in (
            "moving_state_orbital",
            "moving_orbital",
            "fixed_state_orbital",
        ):
            gain_fraction = safe_ratio(
                float(integrals[f"gain_{prefix}"]),
                float(integrals["gain_external"]),
            )
            motion_fraction_l1 = safe_ratio(
                float(integrals[f"motion_{prefix}_l1"]),
                float(integrals["motion_external_l1"]),
            )
            motion_fraction_sq = safe_ratio(
                float(integrals[f"motion_{prefix}_sq"]),
                float(integrals["motion_external_sq"]),
            )
            summary[f"{prefix}_fraction_external_gain"] = gain_fraction
            summary[f"{prefix}_fraction_external_motion_l1"] = (
                motion_fraction_l1
            )
            summary[f"{prefix}_fraction_external_motion_sq"] = (
                motion_fraction_sq
            )
            summary[f"{prefix}_gain_enrichment_over_motion_l1"] = (
                safe_ratio(gain_fraction, motion_fraction_l1)
            )
            summary[f"{prefix}_gain_enrichment_over_motion_sq"] = (
                safe_ratio(gain_fraction, motion_fraction_sq)
            )
        summary_rows.append(summary)
        print(
            f"{architecture} seed={run_seed} epoch={epoch} "
            f"C_Slater={fit['objective']:.6f} "
            f"G_A={integrals['gain_external']:+.6e} "
            "f_gain="
            f"{summary['moving_state_orbital_fraction_external_gain']:+.3f} "
            "f_motion_sq="
            f"{summary['moving_state_orbital_fraction_external_motion_sq']:.3f} "
            f"closure={summary['closure_error']:+.3e}",
            flush=True,
        )

    write_csv(out_dir / "closest_slater_fit_restarts.csv", fit_rows)
    write_csv(out_dir / "closest_slater_tangent_nodes.csv", node_rows)
    write_csv(out_dir / "closest_slater_tangent_summary.csv", summary_rows)
    metadata = {
        "analysis": "post-hoc closest-Slater moving-tangent attribution",
        "architecture": architecture,
        "run_seed": run_seed,
        "config": str(config_path),
        "run_dir": str(run_dir),
        "reference_dir": str(reference_dir),
        "epochs": list(args.epochs),
        "quadrature_order": int(args.quadrature_order),
        "fit_restarts": int(args.fit_restarts),
        "fit_steps": int(args.fit_steps),
        "fit_learning_rate": float(args.fit_learning_rate),
        "fit_patience": int(args.fit_patience),
        "fit_definition": (
            "maximum absolute sampled-root overlap with one fixed-spin "
            "Slater determinant; restart 0 uses root-masked natural orbitals "
            "and remaining restarts are random"
        ),
        "moving_subspace_definition": (
            "closest-Slater state plus all spin-conserving occupied-virtual "
            "orbital tangents, fitted to sampled-root coefficients of the "
            "pre-update wavefunction"
        ),
        "projection_definition": (
            "orthogonal projection of the batch-external normalized JVP onto "
            "the external restriction of the fitted coefficient-space "
            "subspace"
        ),
        "motion_fraction_sq_definition": (
            "path integral of projected external tangent squared norm divided "
            "by the path integral of total external tangent squared norm"
        ),
        "gain_fraction_definition": (
            "path-integrated exact external energy derivative carried by the "
            "projected tangent divided by total external gain"
        ),
        "external_mass_shape_definition": (
            "at each quadrature node the external normalized tangent is "
            "split into its projection parallel to the external state "
            "restriction (mass/rescaling) and its orthogonal shape change"
        ),
        "interpretation_guard": (
            "gain fraction must be compared with motion fraction; gain "
            "enrichment above motion enrichment is evidence that the fitted "
            "Slater channel selectively carries useful external descent"
        ),
        "post_hoc": True,
        "target_symmetric_across_architectures": True,
        "fixed_scaffold_subspace": (
            str(args.fixed_scaffold_subspace.resolve())
            if args.fixed_scaffold_subspace is not None
            else None
        ),
    }
    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
