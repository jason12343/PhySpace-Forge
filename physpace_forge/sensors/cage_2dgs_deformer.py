"""
Real-Time Cage-Based 2D/3D Gaussian Splatting Deformer.

Synchronizes the outer VisualSplatLayer with the deformed MechanicsCoreLayer in <0.05 ms:
- Splat centers μ_s update via barycentric interpolation on deformed surface triangles.
- Splat 3x3 covariance matrices Σ_s update via local triangle Jacobian J_f * Σ_0 * J_f^T.
"""

from typing import Tuple
import torch

from physpace_forge.schema.physpack import SCPPAsset
from physpace_forge.solver.ipc_math import _det_and_inv_T_3x3


class Cage2DGSDeformer:
    """
    Deforms surface-bound Gaussian Splats directly from tetrahedral boundary motion.
    """

    @staticmethod
    def deform_splats(asset: SCPPAsset) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute deformed 3D splat centers (N_splats, 3) and deformed 3x3 covariance
        matrices (N_splats, 3, 3) from current vs. rest surface triangle geometry.
        """
        vis = asset.visual_layer
        shell = asset.contact_shell
        core = asset.mechanics_core

        faces = shell.surface_faces[vis.bound_face_indices]  # (N_splats, 3)
        bary = vis.barycentric_weights  # (N_splats, 3)

        # Current deformed triangle vertices
        x0 = core.curr_vertices[faces[:, 0]]
        x1 = core.curr_vertices[faces[:, 1]]
        x2 = core.curr_vertices[faces[:, 2]]

        # Current triangle unit normal
        e1_curr = x1 - x0
        e2_curr = x2 - x0
        n_curr_raw = torch.linalg.cross(e1_curr, e2_curr, dim=-1)
        n_curr = n_curr_raw / torch.clamp(torch.linalg.norm(n_curr_raw, dim=-1, keepdim=True), min=1e-12)

        # Barycentric interpolated center + normal offset
        deformed_means = (
            bary[:, 0:1] * x0
            + bary[:, 1:2] * x1
            + bary[:, 2:3] * x2
            + vis.normal_offsets * n_curr
        )

        # Rest triangle vertices for local Jacobian J_f = D_s_curr * D_s_rest^{-1}
        X0 = core.rest_vertices[faces[:, 0]]
        X1 = core.rest_vertices[faces[:, 1]]
        X2 = core.rest_vertices[faces[:, 2]]
        e1_rest = X1 - X0
        e2_rest = X2 - X0
        n_rest_raw = torch.linalg.cross(e1_rest, e2_rest, dim=-1)
        n_rest = n_rest_raw / torch.clamp(torch.linalg.norm(n_rest_raw, dim=-1, keepdim=True), min=1e-12)

        frame_curr = torch.stack([e1_curr, e2_curr, n_curr], dim=-1)  # (N_splats, 3, 3)
        frame_rest = torch.stack([e1_rest, e2_rest, n_rest], dim=-1)  # (N_splats, 3, 3)

        # Analytical register-level 3x3 inverse of rest surface frame (zero cuSOLVER sync)
        eye_eps = 1e-7 * torch.eye(3, device=asset.device).unsqueeze(0)
        _, frame_rest_inv_T = _det_and_inv_T_3x3(frame_rest + eye_eps)
        J_f = torch.bmm(frame_curr, frame_rest_inv_T.transpose(1, 2))

        # Covariance update: Σ_curr = J_f * Σ_0 * J_f^T
        deformed_covariances = torch.bmm(torch.bmm(J_f, vis.rest_covariances), J_f.transpose(1, 2))
        return deformed_means, deformed_covariances
