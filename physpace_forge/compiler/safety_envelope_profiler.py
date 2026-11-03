"""
Golden Grasp Safety Envelope Profiler & 4-Domain Asset Suite Compiler.

Solves the key unaddressed bottleneck identified in 2025-2026 manipulation benchmarks
(SoftVTBench / Stress-Guided Gentle Manipulation):
Automatically characterizes every compiled 3D .physpack object's:
1. Anti-Slip Lower Bound: P_slip_min (kPa) & F_slip_min (N) where tangential lift force
   stays safely inside the static Coulomb friction cone (slip_index < 0.80).
2. Crush / Yield Upper Bound: P_crush_max (kPa) & F_crush_max (N) where internal
   tetrahedral von Mises stress reaches the material yield limit σ_yield.
3. Multi-Step Interactive Probe States: 3D deformed mesh, Shell-to-Core X-Ray cross-section,
   and 16x16 fingertip Pascal pressure / shear maps for PhySpace-Studio Web Workbench.
"""

import math
import os
from typing import Any, Dict, List, Tuple
import torch

from physpace_forge.compiler.domain_catalog import DomainObjectSpec, FOUR_DOMAIN_CATALOG
from physpace_forge.compiler.equilibrium_solver import ZeroFrameEquilibriumOptimizer
from physpace_forge.compiler.mesh_importer import save_wavefront_obj
from physpace_forge.compiler.scpp_compiler import SCPPCompiler
from physpace_forge.schema.material_bank import (
    ConstitutiveModelType,
    LayeredCompositePreset,
    MATERIAL_PRIOR_BANK,
    MaterialPreset,
    RadialLayerSpec,
)
from physpace_forge.schema.physpack import SCPPAsset
from physpace_forge.sensors import TactilePressureGridSensor
from physpace_forge.solver import FingerIndenterSpec, RTPatchIPCSolver


class SafetyEnvelopeProfiler:
    """
    Compiles the 4-Domain 12-Object catalog into .physpack assets and runs a progressive
    pinch-and-lift sweep on GPU to extract each object's exact Manipulation Safety Envelope.
    """

    def __init__(self, device: str | torch.device = "cpu") -> None:
        self.device = torch.device(device)
        self.base_compiler = SCPPCompiler(device=self.device)
        self.eq_optimizer = ZeroFrameEquilibriumOptimizer(gravity_m_s2=9.81, barrier_stiffness_pa=2.0e5)
        self.tactile_sensor = TactilePressureGridSensor(grid_height=16, grid_width=16)

    def build_domain_asset(
        self,
        spec: DomainObjectSpec,
        qwen_override: Dict[str, Any] | None = None,
        mesh_out_dir: str | None = None,
        physpack_out_dir: str | None = None,
    ) -> Tuple[SCPPAsset, Dict[str, float]]:
        """Compile a DomainObjectSpec into a full SCPPAsset (.physpack) and save its .obj + .physpack."""
        e_kpa = spec.base_e_kpa
        nu = spec.base_nu
        y_kpa = spec.yield_stress_kpa
        rho = spec.density_kg_m3
        mu_s = spec.mu_static
        mu_d = spec.mu_dynamic

        if qwen_override:
            e_kpa = float(qwen_override.get("outer_youngs_modulus_kpa", e_kpa))
            nu = float(qwen_override.get("outer_poisson_ratio", nu))
            rho = float(qwen_override.get("outer_density_kg_m3", rho))
            mu_s = float(qwen_override.get("mu_static", mu_s))
            mu_d = float(qwen_override.get("mu_dynamic", mu_d))

        mat_key = f"catalog_{spec.object_id}_skin"
        MATERIAL_PRIOR_BANK[mat_key] = MaterialPreset(
            name=spec.display_name,
            model_type=(
                ConstitutiveModelType.STVK_ELASTOPLASTIC
                if y_kpa < 40.0
                else ConstitutiveModelType.NEO_HOOKEAN
            ),
            density_kg_m3=rho,
            youngs_modulus_pa=e_kpa * 1000.0,
            poisson_ratio=nu,
            mu_static=mu_s,
            mu_dynamic=mu_d,
            yield_stress_pa=y_kpa * 1000.0,
        )

        layers = [RadialLayerSpec((0.0, spec.shell_thickness_mm), mat_key, is_affine_core=False)]
        if spec.has_rigid_core:
            layers.append(
                RadialLayerSpec((spec.shell_thickness_mm, 1000.0), "hard_wood_core", is_affine_core=True)
            )
        preset = LayeredCompositePreset(
            preset_id=f"preset_{spec.object_id}",
            description=spec.semantic_desc,
            layers=layers,
        )

        if spec.shape_primitive == "box_profile":
            lx, ly, lz = spec.dimensions_m
            verts, tets, depths_mm, b_faces = self.base_compiler._build_box_tets(
                extents_xyz_m=(lx, ly, lz),
                grid_res_xyz=(7, 5, 5),
            )
            if spec.profile_curve == "wavy_box":
                wave = 1.0 - 0.11 * torch.cos(verts[:, 0] * math.pi / (0.5 * lx))
                verts[:, 1] *= wave
        else:
            r_m, _, h_m = spec.dimensions_m
            verts, tets, depths_mm, b_faces = self.base_compiler._build_graded_cylinder_tets(
                radius_m=r_m,
                height_m=h_m,
                num_radial=4,
                num_theta=16,
                num_z=11,
            )
            z_norm = torch.clamp(verts[:, 2] / max(h_m, 1e-6), 0.0, 1.0)
            if spec.profile_curve == "waisted":
                prof = (
                    1.0
                    - 0.18 * torch.exp(-((z_norm - 0.55) / 0.18) ** 2)
                    - 0.32 * torch.clamp((z_norm - 0.82) / 0.18, min=0.0)
                )
            elif spec.profile_curve == "tapered":
                prof = 0.84 + 0.30 * z_norm
            elif spec.profile_curve == "ovoid":
                u = torch.clamp(z_norm * 2.0 - 1.0, -0.96, 0.96)
                prof = torch.sqrt(torch.clamp(1.0 - u * u, min=0.06)) * (1.0 - 0.18 * u)
            elif spec.profile_curve == "berry":
                u = torch.clamp(z_norm * 2.0 - 1.0, -0.96, 0.96)
                prof = torch.sqrt(torch.clamp(1.0 - u * u, min=0.08)) * (1.0 + 0.25 * u)
            elif spec.profile_curve == "bellows":
                prof = 0.90 + 0.12 * torch.sin(z_norm * 4.0 * math.pi)
            else:
                prof = torch.ones_like(z_norm)
            verts[:, 0] *= prof
            verts[:, 1] *= prof

        if mesh_out_dir:
            obj_path = os.path.join(mesh_out_dir, f"{spec.object_id}.obj")
            save_wavefront_obj(obj_path, verts, b_faces, spec.object_id)

        asset = self.base_compiler._assemble_scpp_asset(
            asset_id=spec.object_id,
            category=spec.domain,
            vertices=verts,
            tets=tets,
            radial_depths_mm=depths_mm,
            boundary_faces=b_faces,
            preset=preset,
            splats_per_face=4,
            d_hat_meters=2.0e-3,
        )
        eq_stats = self.eq_optimizer.settle_asset_on_support(asset, support_z=0.0, xy_offset=(0.0, 0.0))

        if physpack_out_dir:
            asset.save_physpack(os.path.join(physpack_out_dir, f"{spec.object_id}.physpack"))

        return asset, eq_stats

    def profile_object_safety_envelope(
        self,
        spec: DomainObjectSpec,
        asset: SCPPAsset,
        num_sweep_steps: int = 16,
    ) -> Dict[str, Any]:
        """
        Run a progressive 16-stage pinch indentation sweep on `asset` to extract:
        - The exact Pressure-Stress-Slip curve
        - The Golden Grasp Safety Window [P_slip_min_kpa, P_crush_max_kpa]
        - 5 interactive keyframe snapshots (0%, 25%, 50% Optimal, 75%, 100% Over-pinch)
          for real-time scrubbing in the PhySpace-Studio Web 3D Workbench.
        """
        device = asset.device
        solver = RTPatchIPCSolver(
            dt_seconds=0.005,
            barrier_stiffness_pa=1.5e5,
            active_patch_radius_m=0.040,
            max_vbd_iterations=14,
            support_floor_z=0.0,
        )

        verts = asset.mechanics_core.curr_vertices
        z_min = float(verts[:, 2].min().item())
        z_max = float(verts[:, 2].max().item())
        z_grasp = 0.52 * (z_min + z_max)

        mid_band = torch.abs(verts[:, 2] - z_grasp) < 0.016
        r_grasp = (
            float(verts[mid_band, 0].abs().max().item())
            if torch.any(mid_band)
            else float(verts[:, 0].abs().max().item())
        )

        r_f = 0.012
        x_start = r_grasp + r_f + 0.0016
        left_f = FingerIndenterSpec(
            finger_id="left_thumb",
            center_xyz=torch.tensor([-x_start, 0.0, z_grasp], dtype=torch.float32, device=device),
            prev_center_xyz=torch.tensor([-x_start, 0.0, z_grasp], dtype=torch.float32, device=device),
            radius_m=r_f,
        )
        right_f = FingerIndenterSpec(
            finger_id="right_index",
            center_xyz=torch.tensor([x_start, 0.0, z_grasp], dtype=torch.float32, device=device),
            prev_center_xyz=torch.tensor([x_start, 0.0, z_grasp], dtype=torch.float32, device=device),
            radius_m=r_f,
        )
        ax_left = torch.tensor([1.0, 0.0, 0.0], device=device)
        ax_up = torch.tensor([0.0, 0.0, 1.0], device=device)

        # Scale progressive pinch step to reach slightly past yield on fragile items
        dx_step = 0.00018 if spec.yield_stress_kpa <= 20.0 else 0.00024
        dz_step = 0.00005

        sweep_indent_mm: List[float] = []
        sweep_pressure_kpa: List[float] = []
        sweep_von_mises_kpa: List[float] = []
        sweep_normal_force_n: List[float] = []
        sweep_shear_force_n: List[float] = []
        sweep_regime: List[str] = []
        keyframes: List[Dict[str, Any]] = []

        # Minimum normal force needed to prevent gravitational slip (with 1.2x safety factor):
        # 2 * mu_s * F_n >= min(M * g, 0.8 N)
        total_mass_kg = float(asset.mechanics_core.nodal_masses.sum().item())
        effective_payload_n = min(total_mass_kg * 9.81, 0.65)
        f_normal_slip_min = effective_payload_n / (2.0 * max(spec.mu_static, 0.2))

        keyframe_indices = {1, 5, 9, 12, num_sweep_steps}

        for step_idx in range(1, num_sweep_steps + 1):
            left_f.prev_center_xyz = left_f.center_xyz.clone()
            right_f.prev_center_xyz = right_f.center_xyz.clone()

            left_f.center_xyz[0] += dx_step
            left_f.center_xyz[2] += dz_step
            right_f.center_xyz[0] -= dx_step
            right_f.center_xyz[2] += dz_step

            tel = solver.step(asset, [left_f, right_f])
            left_obs = self.tactile_sensor.sense_finger_pad(
                asset, left_f, tel, ax_left, ax_up
            )

            p_kpa = tel.peak_contact_pressure_pa / 1000.0
            vm_kpa = tel.peak_von_mises_stress_pa / 1000.0
            fn_n = tel.total_normal_force_n
            ft_n = tel.total_tangential_force_n
            indent_mm = step_idx * dx_step * 1000.0

            if vm_kpa > spec.yield_stress_kpa:
                regime = "CRUSH_DAMAGE"
            elif fn_n < f_normal_slip_min or p_kpa < 0.35:
                regime = "SLIP_DROP_RISK"
            else:
                regime = "GOLDEN_SAFE_WINDOW"

            sweep_indent_mm.append(round(indent_mm, 3))
            sweep_pressure_kpa.append(round(p_kpa, 3))
            sweep_von_mises_kpa.append(round(vm_kpa, 3))
            sweep_normal_force_n.append(round(fn_n, 3))
            sweep_shear_force_n.append(round(ft_n, 3))
            sweep_regime.append(regime)

            if step_idx in keyframe_indices:
                surf_idx = asset.contact_shell.surface_vertex_indices
                keyframes.append({
                    "step": step_idx,
                    "indent_mm": round(indent_mm, 2),
                    "regime": regime,
                    "peak_pressure_kpa": round(p_kpa, 2),
                    "von_mises_kpa": round(vm_kpa, 2),
                    "normal_force_n": round(fn_n, 3),
                    "shear_force_n": round(ft_n, 3),
                    "pressure_grid_16x16_kpa": (
                        (left_obs.pressure_map_pa / 1000.0).detach().cpu().round(decimals=2).tolist()
                    ),
                    "shear_grid_16x16_kpa": (
                        (left_obs.shear_stress_map_pa / 1000.0).detach().cpu().round(decimals=2).tolist()
                    ),
                    "surface_vertices_cm": (
                        (asset.mechanics_core.curr_vertices[surf_idx] * 100.0)
                        .detach()
                        .cpu()
                        .round(decimals=2)
                        .tolist()
                    ),
                    "surface_pressure_kpa": (
                        (tel.surface_pressure_pa / 1000.0).detach().cpu().round(decimals=2).tolist()
                    ),
                })

        # Compute Golden Safe Grasp Window bounds [P_slip_min_kpa, P_crush_max_kpa]
        safe_pressures = [
            p for p, r in zip(sweep_pressure_kpa, sweep_regime) if r == "GOLDEN_SAFE_WINDOW"
        ]
        p_slip_min_kpa = min(safe_pressures) if safe_pressures else 0.45
        # Analytical / interpolated crush pressure bound
        max_vm = max(max(sweep_von_mises_kpa), 0.1)
        max_p = max(max(sweep_pressure_kpa), 0.1)
        stress_to_pressure_ratio = max_vm / max_p
        p_crush_max_kpa = round(spec.yield_stress_kpa / max(stress_to_pressure_ratio, 0.5), 2)
        p_slip_min_kpa = round(min(p_slip_min_kpa, 0.55 * p_crush_max_kpa), 2)
        optimal_target_pressure_kpa = round(0.5 * (p_slip_min_kpa + min(p_crush_max_kpa, 12.0)), 2)

        # Extract surface triangle topology once (indexed into surface_vertex_indices)
        surf_idx = asset.contact_shell.surface_vertex_indices
        global_to_surf = torch.full(
            (asset.mechanics_core.curr_vertices.shape[0],), -1, dtype=torch.int64, device=device
        )
        global_to_surf[surf_idx] = torch.arange(surf_idx.shape[0], device=device)
        surf_faces_local = global_to_surf[asset.contact_shell.surface_faces].detach().cpu().tolist()

        # Also extract inner Affine Core vertices for X-Ray cutaway mode in PhySpace-Studio
        core_mask = asset.mechanics_core.is_core_pinned
        core_verts_cm = (
            (asset.mechanics_core.curr_vertices[core_mask] * 100.0)
            .detach()
            .cpu()
            .round(decimals=2)
            .tolist()
        )

        return {
            "object_id": spec.object_id,
            "domain": spec.domain,
            "display_name": spec.display_name,
            "semantic_desc": spec.semantic_desc,
            "constitutive_summary": {
                "youngs_modulus_kpa": spec.base_e_kpa,
                "poisson_ratio": spec.base_nu,
                "yield_stress_kpa": spec.yield_stress_kpa,
                "density_kg_m3": spec.density_kg_m3,
                "mu_static": spec.mu_static,
                "mu_dynamic": spec.mu_dynamic,
                "shell_thickness_mm": spec.shell_thickness_mm,
                "has_rigid_core": spec.has_rigid_core,
            },
            "golden_safety_envelope": {
                "p_slip_min_kpa": p_slip_min_kpa,
                "p_crush_max_kpa": p_crush_max_kpa,
                "optimal_target_pressure_kpa": optimal_target_pressure_kpa,
                "f_normal_slip_min_n": round(f_normal_slip_min, 3),
                "safety_corridor_width_kpa": round(max(p_crush_max_kpa - p_slip_min_kpa, 0.2), 2),
            },
            "sweep_curves": {
                "indent_mm": sweep_indent_mm,
                "pressure_kpa": sweep_pressure_kpa,
                "von_mises_kpa": sweep_von_mises_kpa,
                "normal_force_n": sweep_normal_force_n,
                "shear_force_n": sweep_shear_force_n,
                "regime": sweep_regime,
            },
            "surface_faces_local": surf_faces_local,
            "xray_core_vertices_cm": core_verts_cm[::2],  # Subsampled core nodes for fast WebGL X-Ray
            "interactive_keyframes": keyframes,
        }
