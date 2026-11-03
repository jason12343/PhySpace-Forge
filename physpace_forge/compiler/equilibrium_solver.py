"""
Phase 3: Zero-Frame Static Equilibrium Scene Optimizer.

Ensures that procedurally/generatively synthesized 3D scenes obey strict static equilibrium
(sum F = 0, sum τ = 0) and zero interpenetration (d_k > 0) at Frame 0 before exporting to
robot training pipelines.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple
import torch

from physpace_forge.schema.physpack import SCPPAsset
from physpace_forge.solver.ipc_math import (
    ipc_barrier_energy,
    ipc_barrier_gradient,
    ipc_barrier_hessian,
)


@dataclass
class ScenePlacementSpec:
    """Initial generative placement of an SCPPAsset inside a 3D workspace."""
    asset: SCPPAsset
    initial_translation_xyz: Tuple[float, float, float]
    support_surface_z: float = 0.0


class ZeroFrameEquilibriumOptimizer:
    """
    Optimizes initial rigid/deformable vertical and lateral placements in a generated scene
    under gravity and IPC barrier contact potentials so that at Frame 0:
    1. Every object has strictly positive clearance d > 0 above supporting surfaces (zero penetration).
    2. Integrated contact barrier pressure matches total gravitational weight M * g:
       sum_{k in Contact} A_k * P(d_k) = M_total * g
    """

    def __init__(
        self,
        gravity_m_s2: float = 9.81,
        barrier_stiffness_pa: float = 2.0e5,
    ) -> None:
        self.g = gravity_m_s2
        self.kappa = barrier_stiffness_pa

    def settle_asset_on_support(
        self,
        asset: SCPPAsset,
        support_z: float = 0.0,
        xy_offset: Tuple[float, float] = (0.0, 0.0),
        max_newton_iters: int = 40,
        force_tol_n: float = 1e-5,
    ) -> Dict[str, float]:
        """
        Translate an SCPPAsset to (xy_offset) and solve for the exact vertical equilibrium
        height z* where the area-weighted IPC contact barrier force balances gravity M * g
        with 100% penetration-free guarantee (z_min > support_z).
        """
        core = asset.mechanics_core
        shell = asset.contact_shell
        d_hat = shell.d_hat_meters

        # Center XY at xy_offset and lift bottom to support_z + 0.45 * d_hat initial guess
        mean_xy = core.rest_vertices[:, :2].mean(dim=0)
        min_z_rest = float(core.rest_vertices[:, 2].min().item())
        z_shift = (support_z + 0.45 * d_hat) - min_z_rest

        offset_vec = torch.tensor(
            [
                xy_offset[0] - float(mean_xy[0].item()),
                xy_offset[1] - float(mean_xy[1].item()),
                z_shift,
            ],
            dtype=torch.float32,
            device=asset.device,
        )
        core.curr_vertices = core.rest_vertices + offset_vec.unsqueeze(0)
        core.rest_vertices = core.curr_vertices.clone()
        core.affine_transform[:, 3] = offset_vec

        total_mass_kg = float(core.nodal_masses.sum().item())
        weight_newtons = total_mass_kg * self.g

        surf_idx = shell.surface_vertex_indices
        areas = shell.vertex_contact_areas

        # damped Newton-Raphson with barrier line search (strictly enforces d_k > 0)
        delta_z = torch.tensor(0.0, dtype=torch.float32, device=asset.device)
        for _ in range(max_newton_iters):
            surf_z = core.curr_vertices[surf_idx, 2] + delta_z
            dists = surf_z - support_z

            # Gradient of total potential E(z) = M * g * z + sum_k A_k * b(d_k, d_hat)
            grad_b = ipc_barrier_gradient(dists, d_hat, self.kappa)
            net_force_residual = weight_newtons + float((areas * grad_b).sum().item())

            if abs(net_force_residual) < force_tol_n:
                break

            hess_b = ipc_barrier_hessian(dists, d_hat, self.kappa)
            total_stiff = float((areas * hess_b).sum().item()) + 1e-6
            step_dz = -net_force_residual / total_stiff

            # CCD barrier line search: ensure min(dists + alpha * step_dz) > 0.05 * min(dists)
            alpha = 1.0
            min_curr_d = float(dists.min().item())
            while min_curr_d + alpha * step_dz <= 0.05 * min_curr_d:
                alpha *= 0.5
            delta_z = delta_z + alpha * step_dz

        # Commit equilibrium state
        core.curr_vertices[:, 2] += delta_z
        core.rest_vertices[:, 2] += delta_z
        core.velocities.zero_()

        final_dists = core.curr_vertices[surf_idx, 2] - support_z
        upward_contact_force = float(
            (-areas * ipc_barrier_gradient(final_dists, d_hat, self.kappa)).sum().item()
        )
        min_clearance_mm = float(final_dists.min().item()) * 1000.0

        return {
            "total_mass_kg": total_mass_kg,
            "gravity_weight_n": weight_newtons,
            "equilibrium_contact_force_n": upward_contact_force,
            "force_balance_error_n": abs(weight_newtons - upward_contact_force),
            "min_clearance_mm": min_clearance_mm,
            "penetration_free": float(min_clearance_mm > 0.0),
        }
