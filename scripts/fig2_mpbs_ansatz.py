"""Non-neural matrix-product backflow states for the Fig. 2(d) controls.

Following Lami, Carleo, and Collura, Phys. Rev. B 106, L081111 (2022), each
MPS tensor is made configuration dependent through explicit pair tensors.  In
the retained nearest-neighbor specialization,

    A_l(s_l) -> A_l(s_l)
        + sum_{j in N_phys(l)} F_l(s_l, s_j).

The physical neighbor graph is fixed before training, independently of the
MPS contraction order.  The one-dimensional control ties the pair correction
over the neighbors of a source site, matching the distance-tied chain
construction.  The two-dimensional control instead retains a separate
``F^[i,j]`` tensor for every directed physical edge, as in the generic MPBS
definition and its square-lattice specialization in the reference.  There is
no neural network.  The F tensors start at zero, so every topology is exactly
the paired ordinary D=64 MPS at initialization.

The legacy ``oriented_edges`` topology is retained only so that the rejected
one-sided pilot checkpoints remain loadable for audit.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from qknet.ansatz.standard.references.mps import (
    MPS_PHYS_DIM,
    LogMPS,
    _group_phys_indices,
)


class LogMatrixProductBackflow(nn.Module):
    """Open-boundary MPS with explicit non-neural physical-edge corrections."""

    def __init__(
        self,
        n_elecs: int,
        n_qubits: int,
        *,
        max_bond_dim: int = 64,
        site_order: Sequence[int] | None = None,
        backflow_init_scale: float = 0.0,
        backflow_topology: str = "oriented_edges",
        physical_edges: Sequence[Sequence[int]] | None = None,
        dtype: torch.dtype = torch.float64,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        self.n_elecs = int(n_elecs)
        self.n_qubits = int(n_qubits)
        self.n_sites = self.n_qubits // 2
        if self.n_sites < 2:
            raise ValueError("matrix-product backflow requires at least two sites")
        self.max_bond_dim = int(max_bond_dim)
        self.dtype = dtype
        self.device = torch.device(device)
        self.backflow_topology = str(backflow_topology)
        if self.backflow_topology not in {
            "oriented_edges",
            "all_physical_neighbors_shared",
            "pair_resolved_physical_edges",
        }:
            raise ValueError(
                "backflow_topology must be 'oriented_edges', "
                "'all_physical_neighbors_shared', or "
                "'pair_resolved_physical_edges'"
            )

        # Constructing the base first makes its cores bitwise identical to the
        # paired LogMPS arm under a common random seed.
        self.base_mps = LogMPS(
            n_elecs=self.n_elecs,
            n_qubits=self.n_qubits,
            max_bond_dim=self.max_bond_dim,
            site_order=site_order,
            dtype=dtype,
            device=self.device,
        )

        # Reordering the MPS moves these physical bonds along the contraction
        # path but never changes the physical graph.
        if physical_edges is None:
            normalized_edges = tuple(
                (site, site + 1) for site in range(self.n_sites - 1)
            )
        else:
            edge_set: set[tuple[int, int]] = set()
            for raw_edge in physical_edges:
                if len(raw_edge) != 2:
                    raise ValueError(
                        "every physical edge must contain exactly two sites"
                    )
                left, right = (int(raw_edge[0]), int(raw_edge[1]))
                if left == right:
                    raise ValueError("physical edges may not contain self-loops")
                if not (0 <= left < self.n_sites and 0 <= right < self.n_sites):
                    raise ValueError(
                        f"physical edge {(left, right)} is outside "
                        f"[0, {self.n_sites})"
                    )
                edge_set.add((min(left, right), max(left, right)))
            normalized_edges = tuple(sorted(edge_set))
            if not normalized_edges:
                raise ValueError("physical_edges may not be empty")
        self.physical_edges = normalized_edges
        position_of_site = {
            physical_site: position
            for position, physical_site in enumerate(self.base_mps.site_order)
        }
        neighbors: list[list[int]] = [[] for _ in range(self.n_sites)]
        for left, right in self.physical_edges:
            neighbors[left].append(right)
            neighbors[right].append(left)
        self.physical_neighbors = tuple(
            tuple(sorted(site_neighbors)) for site_neighbors in neighbors
        )
        self.directed_physical_edges = tuple(
            (source_site, partner_site)
            for source_site, site_neighbors in enumerate(self.physical_neighbors)
            for partner_site in site_neighbors
        )
        edges_by_position: list[list[int]] = [[] for _ in range(self.n_sites)]
        corrections: list[nn.Parameter] = []
        if self.backflow_topology == "oriented_edges":
            for edge_index, (source_site, _) in enumerate(self.physical_edges):
                source_position = position_of_site[source_site]
                chi_l, chi_r = self.base_mps.bond_pairs[source_position]
                tensor = torch.zeros(
                    MPS_PHYS_DIM,
                    MPS_PHYS_DIM,
                    chi_l,
                    chi_r,
                    dtype=dtype,
                    device=self.device,
                )
                if backflow_init_scale > 0.0:
                    nn.init.normal_(tensor, std=float(backflow_init_scale))
                corrections.append(nn.Parameter(tensor))
                edges_by_position[source_position].append(edge_index)
        elif self.backflow_topology == "all_physical_neighbors_shared":
            # One source-site tensor for every physical site, shared over all
            # of that site's physical nearest neighbors.  Iterating in physical
            # site order makes the checkpoint layout independent of ordering.
            for source_site in range(self.n_sites):
                source_position = position_of_site[source_site]
                chi_l, chi_r = self.base_mps.bond_pairs[source_position]
                tensor = torch.zeros(
                    MPS_PHYS_DIM,
                    MPS_PHYS_DIM,
                    chi_l,
                    chi_r,
                    dtype=dtype,
                    device=self.device,
                )
                if backflow_init_scale > 0.0:
                    nn.init.normal_(tensor, std=float(backflow_init_scale))
                corrections.append(nn.Parameter(tensor))
        else:
            # A distinct F^[i,j](s_i,s_j) for every directed physical edge.
            # Edges are enumerated in physical-source order, independently of
            # the MPS contraction order.  The tensor's virtual legs follow the
            # position occupied by its source site in that contraction.
            for edge_index, (source_site, _) in enumerate(
                self.directed_physical_edges
            ):
                source_position = position_of_site[source_site]
                chi_l, chi_r = self.base_mps.bond_pairs[source_position]
                tensor = torch.zeros(
                    MPS_PHYS_DIM,
                    MPS_PHYS_DIM,
                    chi_l,
                    chi_r,
                    dtype=dtype,
                    device=self.device,
                )
                if backflow_init_scale > 0.0:
                    nn.init.normal_(tensor, std=float(backflow_init_scale))
                corrections.append(nn.Parameter(tensor))
                edges_by_position[source_position].append(edge_index)
        self.backflow_tensors = nn.ParameterList(corrections)
        self._edges_by_position = tuple(
            tuple(indices) for indices in edges_by_position
        )

    @property
    def bond_pairs(self) -> list[tuple[int, int]]:
        return self.base_mps.bond_pairs

    def signed_log_amplitude(
        self,
        states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        states = states.to(device=self.device, dtype=self.dtype)
        canonical_phys = _group_phys_indices(states)
        phys, fermionic_sign = self.base_mps._ordered_phys_and_fermionic_sign(states)

        batch_size = states.shape[0]
        vector = torch.ones(
            batch_size,
            1,
            dtype=self.dtype,
            device=self.device,
        )
        log_scale = torch.zeros(
            batch_size,
            dtype=self.dtype,
            device=self.device,
        )
        for position, core in enumerate(self.base_mps.cores):
            local_token = phys[:, position]
            base_matrix = core[:, local_token, :].permute(1, 0, 2)
            local_matrix = base_matrix
            if self.backflow_topology == "oriented_edges":
                for edge_index in self._edges_by_position[position]:
                    source_site, partner_site = self.physical_edges[edge_index]
                    if source_site != self.base_mps.site_order[position]:
                        raise RuntimeError(
                            "backflow edge/source position mismatch"
                        )
                    partner_token = canonical_phys[:, partner_site]
                    correction = self.backflow_tensors[edge_index]
                    local_matrix = (
                        local_matrix + correction[local_token, partner_token]
                    )
            elif self.backflow_topology == "all_physical_neighbors_shared":
                source_site = self.base_mps.site_order[position]
                correction = self.backflow_tensors[source_site]
                for partner_site in self.physical_neighbors[source_site]:
                    partner_token = canonical_phys[:, partner_site]
                    local_matrix = (
                        local_matrix + correction[local_token, partner_token]
                    )
            else:
                source_site = self.base_mps.site_order[position]
                for edge_index in self._edges_by_position[position]:
                    edge_source, partner_site = self.directed_physical_edges[
                        edge_index
                    ]
                    if edge_source != source_site:
                        raise RuntimeError(
                            "backflow edge/source position mismatch"
                        )
                    partner_token = canonical_phys[:, partner_site]
                    correction = self.backflow_tensors[edge_index]
                    local_matrix = (
                        local_matrix + correction[local_token, partner_token]
                    )
            vector = torch.bmm(vector.unsqueeze(1), local_matrix).squeeze(1)
            norm = torch.linalg.vector_norm(vector, dim=-1).clamp(min=1.0e-30)
            vector = vector / norm[:, None]
            log_scale = log_scale + torch.log(norm)

        scalar = vector.squeeze(-1)
        amplitude_sign = fermionic_sign * torch.sign(scalar)
        amplitude_sign = torch.where(
            amplitude_sign == 0,
            torch.ones_like(amplitude_sign),
            amplitude_sign,
        )
        log_abs = log_scale + torch.log(scalar.abs().clamp(min=1.0e-30))
        return amplitude_sign.to(dtype=self.dtype), log_abs.to(dtype=self.dtype)

    def calc_log_psi(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.signed_log_amplitude(states)

    def _amplitude_batch(self, states: torch.Tensor) -> torch.Tensor:
        sign, log_abs = self.calc_log_psi(states)
        finite = torch.isfinite(log_abs)
        if not bool(torch.any(finite)):
            return torch.zeros_like(log_abs)
        shift = torch.max(log_abs[finite]).detach()
        amplitude = sign * torch.exp(log_abs - shift)
        return torch.where(finite, amplitude, torch.zeros_like(amplitude))

    def forward(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.calc_log_psi(states)
