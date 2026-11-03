"""
Phase 4: High-Resolution Tactile Pressure Map & Incipient Slip Array Sensor.

Projects 3D surface contact tractions (normal pressure P_k in Pascals, 2D shear stress τ_k in Pascals,
and Incipient Slip Index s_k in [0, 1]) onto a standardized 2D fingertip taxel grid (H x W)
for training force-aware Visuotactile-VLA and Reactive Diffusion Policies (RDP).
"""

from dataclasses import dataclass
from typing import Dict
import torch

from physpace_forge.schema.physpack import SCPPAsset
from physpace_forge.solver.rt_patch_ipc import ContactPatchTelemetry, FingerIndenterSpec


@dataclass
class TactileFrameObservation:
    """Standardized 2D tactile sensor frame emitted at 200 Hz per fingertip."""
    finger_id: str
    # 2D Normal Pressure Map P(u, v) in Pascals (Pa): shape (H, W)
    pressure_map_pa: torch.Tensor
    # 2D Tangential Shear Stress Map (τ_u, τ_v) in Pascals (Pa): shape (H, W, 2)
    shear_stress_map_pa: torch.Tensor
    # 2D Incipient Slip Warning Mask S(u, v) in [0, 1]: shape (H, W)
    slip_mask: torch.Tensor
    # Net 3D contact wrench vector [F_normal, F_tangent_u, F_tangent_v] in Newtons: shape (3,)
    net_contact_force_n: torch.Tensor


class TactilePressureGridSensor:
    """
    Renders a dense 2D taxel array (H x W) centered on a robotic fingertip pad
    using Gaussian spatial kernel splatting from the SCPP ContactShell.
    """

    def __init__(
        self,
        grid_height: int = 16,
        grid_width: int = 16,
        sensor_fov_meters: float = 0.024,  # 24mm x 24mm tactile sensing pad
        kernel_sigma_meters: float = 0.0022,
    ) -> None:
        self.H = grid_height
        self.W = grid_width
        self.fov_m = sensor_fov_meters
        self.sigma_m = kernel_sigma_meters

    def sense_finger_pad(
        self,
        asset: SCPPAsset,
        finger: FingerIndenterSpec,
        telemetry: ContactPatchTelemetry,
        approach_axis: torch.Tensor,
        up_axis: torch.Tensor,
    ) -> TactileFrameObservation:
        """
        Project active 3D surface contact tractions onto the fingertip's local 2D tangent plane (u, v).
        """
        device = asset.device
        shell = asset.contact_shell
        surf_x = asset.mechanics_core.curr_vertices[shell.surface_vertex_indices]

        # Construct orthonormal local tangent frame (u_axis, v_axis, n_axis) on fingertip pad
        n_axis = approach_axis / torch.clamp(torch.linalg.norm(approach_axis), min=1e-9)
        v_raw = up_axis - (up_axis @ n_axis) * n_axis
        v_axis = v_raw / torch.clamp(torch.linalg.norm(v_raw), min=1e-9)
        u_axis = torch.linalg.cross(v_axis, n_axis, dim=-1)

        # Filter surface points near this fingertip
        rel_pos = surf_x - finger.center_xyz.unsqueeze(0)
        dist_to_finger = torch.linalg.norm(rel_pos, dim=-1) - finger.radius_m
        active = (dist_to_finger < 2.0 * shell.d_hat_meters) & (telemetry.surface_pressure_pa > 0.1)

        pressure_grid = torch.zeros((self.H, self.W), dtype=torch.float32, device=device)
        shear_grid = torch.zeros((self.H, self.W, 2), dtype=torch.float32, device=device)
        slip_grid = torch.zeros((self.H, self.W), dtype=torch.float32, device=device)

        if not torch.any(active):
            return TactileFrameObservation(
                finger_id=finger.finger_id,
                pressure_map_pa=pressure_grid,
                shear_stress_map_pa=shear_grid,
                slip_mask=slip_grid,
                net_contact_force_n=torch.zeros(3, dtype=torch.float32, device=device),
            )

        pts_rel = rel_pos[active]
        p_vals = telemetry.surface_pressure_pa[active]
        tau_3d = telemetry.surface_shear_stress_pa[active]
        s_vals = telemetry.surface_slip_index[active]
        areas = shell.vertex_contact_areas[active]

        # Local 2D coordinates (u, v) in meters on sensor pad
        u_coords = pts_rel @ u_axis  # (N_act,)
        v_coords = pts_rel @ v_axis  # (N_act,)
        tau_u = tau_3d @ u_axis
        tau_v = tau_3d @ v_axis

        # 2D sensor grid coordinates
        half = 0.5 * self.fov_m
        us = torch.linspace(-half, half, self.W, device=device)
        vs = torch.linspace(-half, half, self.H, device=device)
        grid_v, grid_u = torch.meshgrid(vs, us, indexing="ij")  # (H, W)

        du = grid_u.unsqueeze(-1) - u_coords.view(1, 1, -1)  # (H, W, N_act)
        dv = grid_v.unsqueeze(-1) - v_coords.view(1, 1, -1)
        dist_sq = du * du + dv * dv
        weights = torch.exp(-0.5 * dist_sq / (self.sigma_m * self.sigma_m))  # (H, W, N_act)
        w_sum = torch.clamp(weights.sum(dim=-1), min=1e-6)

        # Kernel-interpolated physical pressure (Pa), shear stress (Pa), and slip mask
        pressure_grid = (weights * p_vals.view(1, 1, -1)).sum(dim=-1) / torch.maximum(
            w_sum, torch.ones_like(w_sum)
        )
        shear_u_grid = (weights * tau_u.view(1, 1, -1)).sum(dim=-1) / torch.maximum(
            w_sum, torch.ones_like(w_sum)
        )
        shear_v_grid = (weights * tau_v.view(1, 1, -1)).sum(dim=-1) / torch.maximum(
            w_sum, torch.ones_like(w_sum)
        )
        shear_grid = torch.stack([shear_u_grid, shear_v_grid], dim=-1)
        slip_grid = (weights * s_vals.view(1, 1, -1)).sum(dim=-1) / torch.maximum(
            w_sum, torch.ones_like(w_sum)
        )

        f_norm = (areas * p_vals).sum()
        f_tu = (areas * tau_u).sum()
        f_tv = (areas * tau_v).sum()
        net_force = torch.stack([f_norm, f_tu, f_tv])

        return TactileFrameObservation(
            finger_id=finger.finger_id,
            pressure_map_pa=pressure_grid,
            shear_stress_map_pa=shear_grid,
            slip_mask=torch.clamp(slip_grid, 0.0, 1.0),
            net_contact_force_n=net_force,
        )

    @staticmethod
    def to_lerobot_observation_dict(
        obs_list: list[TactileFrameObservation],
    ) -> Dict[str, torch.Tensor]:
        """
        Pack multi-finger tactile observations into a LeRobot / ForceVLA compatible tensor dict.
        """
        out: Dict[str, torch.Tensor] = {}
        for obs in obs_list:
            prefix = f"observation.tactile.{obs.finger_id}"
            out[f"{prefix}.pressure_pa"] = obs.pressure_map_pa
            out[f"{prefix}.shear_pa"] = obs.shear_stress_map_pa
            out[f"{prefix}.slip_mask"] = obs.slip_mask
            out[f"{prefix}.net_force_n"] = obs.net_contact_force_n
        return out
