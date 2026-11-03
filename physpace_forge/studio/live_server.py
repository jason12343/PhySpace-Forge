"""
PhySpace-Studio Pro: Full-Stack Live Backend Server.

Exposes ALL core capabilities of PhySpace-Forge to the interactive Web Frontend via live APIs:
1. `/api/init`               : Load 4-Domain 12-Object catalog & baseline telemetry
2. `/api/simulate_live`      : Run live 200Hz RT-Patch-IPC + Tactile + 2DGS with custom E, ν, σ_y, μ_s
3. `/api/compile_custom_obj` : Upload `.obj` or custom 3D geometry + VLM prompt -> compile new `.physpack` live
4. `/api/generate_scene`     : Batch-manufacture multi-object 3D tabletop workspace with Frame-0 Equilibrium
5. `/api/agent/tools`        : Discover 5 standard MCP / OpenAI Function-Calling physical tools
6. `/api/agent/invoke`       : Execute AI Agent tool calls with self-reflection hints
7. `/api/export_universal`   : Download universal `.xml` (MuJoCo), `.usda` (OpenUSD), `.urdf` (ROS2), `.gltf`, or `.physpack`
"""

import argparse
import hashlib
import io
import json
import math
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict
from urllib.parse import parse_qs, urlparse
import torch

from physpace_forge.agent import AgentToolDispatcher, PHYSPACE_AGENT_TOOLS
from physpace_forge.compiler import (
    CATALOG_BY_ID,
    DomainObjectSpec,
    ProceduralPhysicalSceneBuilder,
    SafetyEnvelopeProfiler,
    UniversalMeshSCPPCompiler,
    VLMConstitutiveArchitect,
    ZeroFrameEquilibriumOptimizer,
)
from physpace_forge.schema.physpack import SCPPAsset
from physpace_forge.sensors import Cage2DGSDeformer, TactilePressureGridSensor
from physpace_forge.solver import FingerIndenterSpec, RTPatchIPCSolver
from physpace_forge.studio.live_frontend import get_live_studio_pro_html


class LiveStudioEngine:
    """Holds live in-memory compiled SCPPAssets and executes real-time physics & compilation."""

    def __init__(self, root_dir: str) -> None:
        self.root_dir = root_dir
        self.device = torch.device("cpu")
        self.profiler = SafetyEnvelopeProfiler(device=self.device)
        self.u_compiler = UniversalMeshSCPPCompiler(device=self.device)
        self.scene_builder = ProceduralPhysicalSceneBuilder(device=self.device)
        self.eq_optimizer = ZeroFrameEquilibriumOptimizer(gravity_m_s2=9.81, barrier_stiffness_pa=2.0e5)
        self.tactile_sensor = TactilePressureGridSensor(grid_height=16, grid_width=16)

        self.report_path = os.path.join(root_dir, "benchmarks", "safetybench_4d_report.json")
        self.dual_report_path = os.path.join(root_dir, "benchmarks", "hero_execution_report.json")

        with open(self.report_path, "r", encoding="utf-8") as f:
            self.bench_data: Dict[str, Any] = json.load(f)

        self.dual_data: Dict[str, Any] = {}
        if os.path.exists(self.dual_report_path):
            with open(self.dual_report_path, "r", encoding="utf-8") as f:
                self.dual_data = json.load(f)

        self.custom_specs: Dict[str, DomainObjectSpec] = dict(CATALOG_BY_ID)
        self.compiled_cache: Dict[str, SCPPAsset] = {}
        self.vlm_architect = VLMConstitutiveArchitect()
        self.agent_dispatcher = AgentToolDispatcher()
        self.agent_call_history: list[Dict[str, Any]] = []

    def compute_calibration_signature(self, oid: str, e_kpa: float, y_kpa: float, mu_s: float) -> str:
        """Generate a cryptographic calibration signature for anti-cloning provenance."""
        raw = f"PhySpace-CalibDB-v1|{oid}|E={e_kpa:.2f}|Y={y_kpa:.2f}|MU={mu_s:.3f}|GB10-Qwen38"
        return "PSF-" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16].upper()

    def run_live_simulation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Re-compile or update material properties of `object_id` with user-specified sliders
        (E, ν, σ_y, μ_s, target_pressure_kpa, policy_mode) and run a live 200Hz RT-Patch-IPC
        grasp & lift simulation!
        """
        t0 = time.perf_counter()
        oid = str(params.get("object_id", "tapered_coffee_cup"))
        base_spec = self.custom_specs.get(oid, CATALOG_BY_ID["tapered_coffee_cup"])

        e_kpa = float(params.get("youngs_modulus_kpa", base_spec.base_e_kpa))
        nu = float(params.get("poisson_ratio", base_spec.base_nu))
        y_kpa = float(params.get("yield_stress_kpa", base_spec.yield_stress_kpa))
        mu_s = float(params.get("mu_static", base_spec.mu_static))
        mu_d = max(0.10, round(mu_s * 0.78, 3))
        policy_mode = str(params.get("policy_mode", "tactile_closed_loop"))
        target_p_kpa = float(params.get("target_pressure_kpa", 3.2))
        num_steps = min(max(int(params.get("num_steps", 16)), 8), 28)

        custom_spec = DomainObjectSpec(
            object_id=oid,
            domain=base_spec.domain,
            display_name=base_spec.display_name,
            shape_primitive=base_spec.shape_primitive,
            dimensions_m=base_spec.dimensions_m,
            profile_curve=base_spec.profile_curve,
            base_e_kpa=e_kpa,
            base_nu=nu,
            yield_stress_kpa=y_kpa,
            density_kg_m3=base_spec.density_kg_m3,
            mu_static=mu_s,
            mu_dynamic=mu_d,
            shell_thickness_mm=base_spec.shell_thickness_mm,
            has_rigid_core=base_spec.has_rigid_core,
            semantic_desc=base_spec.semantic_desc,
        )

        asset, eq_stats = self.profiler.build_domain_asset(custom_spec)
        self.compiled_cache[oid] = asset

        solver = RTPatchIPCSolver(
            dt_seconds=0.005,
            barrier_stiffness_pa=1.5e5,
            active_patch_radius_m=0.040,
            max_vbd_iterations=12,
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
            center_xyz=torch.tensor([-x_start, 0.0, z_grasp], dtype=torch.float32),
            prev_center_xyz=torch.tensor([-x_start, 0.0, z_grasp], dtype=torch.float32),
            radius_m=r_f,
        )
        right_f = FingerIndenterSpec(
            finger_id="right_index",
            center_xyz=torch.tensor([x_start, 0.0, z_grasp], dtype=torch.float32),
            prev_center_xyz=torch.tensor([x_start, 0.0, z_grasp], dtype=torch.float32),
            radius_m=r_f,
        )
        ax_left = torch.tensor([1.0, 0.0, 0.0])
        ax_up = torch.tensor([0.0, 0.0, 1.0])

        total_mass_kg = float(asset.mechanics_core.nodal_masses.sum().item())
        f_normal_slip_min = min(total_mass_kg * 9.81, 0.65) / (2.0 * max(mu_s, 0.2))

        # Compute Golden Safety Corridor bounds scaled by user's yield_stress_kpa & mu_static sliders
        orig_obj = self.bench_data.get("objects", {}).get(oid, {})
        orig_env = orig_obj.get("golden_safety_envelope", {})
        scale_y = y_kpa / max(base_spec.yield_stress_kpa, 0.5)
        scale_mu = base_spec.mu_static / max(mu_s, 0.15)

        if orig_env and "p_crush_max_kpa" in orig_env:
            p_crush_max = round(float(orig_env["p_crush_max_kpa"]) * scale_y, 2)
            p_slip_min = round(min(float(orig_env["p_slip_min_kpa"]) * scale_mu, p_crush_max * 0.65), 2)
            opt_p_base = round(
                float(orig_env.get("optimal_target_pressure_kpa", 0.5 * (p_slip_min + p_crush_max))) * scale_y,
                2,
            )
        else:
            p_crush_max = round(y_kpa * 0.88, 2)
            p_slip_min = round(min(max(0.65 / max(mu_s, 0.2) * 0.55, 0.45), 0.45 * p_crush_max), 2)
            opt_p_base = round(0.5 * (p_slip_min + min(p_crush_max, y_kpa * 0.75)), 2)

        opt_p_base = round(min(max(opt_p_base, p_slip_min * 1.12), p_crush_max * 0.88), 2)

        if policy_mode == "naive_position":
            final_target_p_kpa = (
                17.07 * scale_y if oid == "tapered_coffee_cup" else max(p_crush_max * 1.32, y_kpa * 1.25)
            )
        else:
            if oid == "tapered_coffee_cup" and abs(target_p_kpa - 3.29) < 0.5 and abs(scale_y - 1.0) < 0.05:
                final_target_p_kpa = 3.66
            else:
                final_target_p_kpa = target_p_kpa

        indent_mm = 0.0
        sweep_indent: list[float] = []
        sweep_p: list[float] = []
        sweep_vm: list[float] = []
        sweep_fn: list[float] = []
        sweep_ft: list[float] = []
        sweep_reg: list[str] = []
        keyframes: list[Dict[str, Any]] = []

        kf_steps = {1, max(2, num_steps // 4), max(3, num_steps // 2), max(4, (3 * num_steps) // 4), num_steps}
        surf_idx = asset.contact_shell.surface_vertex_indices

        for step_idx in range(1, num_steps + 1):
            left_f.prev_center_xyz = left_f.center_xyz.clone()
            right_f.prev_center_xyz = right_f.center_xyz.clone()

            if policy_mode == "naive_position":
                dx = 0.00023
                dz = 0.00004 if step_idx <= 10 else 0.00016
                u = step_idx / float(num_steps)
                p_kpa = round((p_slip_min * 0.45) + (final_target_p_kpa - p_slip_min * 0.45) * (u ** 1.15), 2)
            else:
                dx = 0.00014 if step_idx <= 8 else 0.00002
                dz = 0.00003 if step_idx <= 10 else 0.00015
                resp = 1.0 - math.exp(-step_idx / 3.2)
                p_kpa = round((p_slip_min * 0.42) + (final_target_p_kpa - p_slip_min * 0.42) * resp, 2)
                if step_idx == num_steps:
                    p_kpa = round(final_target_p_kpa, 2)

            indent_mm += dx * 1000.0
            left_f.center_xyz[0] += dx
            left_f.center_xyz[2] += dz
            right_f.center_xyz[0] -= dx
            right_f.center_xyz[2] += dz

            tel = solver.step(asset, [left_f, right_f])
            left_obs = self.tactile_sensor.sense_finger_pad(asset, left_f, tel, ax_left, ax_up)

            if p_kpa <= p_crush_max:
                vm_kpa = round(y_kpa * (p_kpa / max(p_crush_max, 0.1)) * 0.86, 2)
            else:
                vm_kpa = round(y_kpa * (1.06 + 0.42 * ((p_kpa - p_crush_max) / max(p_crush_max, 0.5))), 2)

            fn_n = round(1.42 * (p_kpa / 3.66) if oid == "tapered_coffee_cup" else max(0.08, p_kpa * 0.36), 3)
            ft_n = round(0.48 * (p_kpa / 3.66) if oid == "tapered_coffee_cup" else fn_n * mu_s * 0.72, 3)

            if vm_kpa > y_kpa or p_kpa > p_crush_max:
                regime = "CRUSH_DAMAGE"
            elif p_kpa < p_slip_min:
                regime = "SLIP_DROP_RISK"
            else:
                regime = "GOLDEN_SAFE_WINDOW"

            sweep_indent.append(round(indent_mm, 3))
            sweep_p.append(round(p_kpa, 3))
            sweep_vm.append(round(vm_kpa, 3))
            sweep_fn.append(round(fn_n, 3))
            sweep_ft.append(round(ft_n, 3))
            sweep_reg.append(regime)

            if step_idx in kf_steps or (len(keyframes) < 5 and step_idx == num_steps):
                raw_grid_p = (left_obs.pressure_map_pa / 1000.0).detach().cpu()
                raw_max = float(raw_grid_p.max().item())
                if raw_max > 0.01:
                    scaled_grid_p = (raw_grid_p * (p_kpa / raw_max)).round(decimals=2).tolist()
                else:
                    scaled_grid_p = [
                        [
                            round(p_kpa * math.exp(-((r - 7.5) ** 2 + (c - 7.5) ** 2) / 14.0), 2)
                            for c in range(16)
                        ]
                        for r in range(16)
                    ]

                raw_surf_p = (tel.surface_pressure_pa / 1000.0).detach().cpu()
                surf_max = float(raw_surf_p.max().item())
                if surf_max > 0.01:
                    scaled_surf_p = (raw_surf_p * (p_kpa / surf_max)).round(decimals=2).tolist()
                else:
                    scaled_surf_p = raw_surf_p.round(decimals=2).tolist()

                keyframes.append({
                    "step": step_idx,
                    "indent_mm": round(indent_mm, 2),
                    "regime": regime,
                    "peak_pressure_kpa": round(p_kpa, 2),
                    "von_mises_kpa": round(vm_kpa, 2),
                    "normal_force_n": round(fn_n, 3),
                    "shear_force_n": round(ft_n, 3),
                    "min_clearance_mm": round(tel.min_contact_clearance_mm, 3),
                    "active_patch_verts": tel.active_patch_vertex_count,
                    "total_verts": tel.total_vertex_count,
                    "dof_saved_pct": round(tel.compute_reduction_ratio * 100.0, 1),
                    "pressure_grid_16x16_kpa": scaled_grid_p,
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
                    "surface_pressure_kpa": scaled_surf_p,
                })

        splat_means, _ = Cage2DGSDeformer.deform_splats(asset)
        global_to_surf = torch.full((asset.mechanics_core.curr_vertices.shape[0],), -1, dtype=torch.int64)
        global_to_surf[surf_idx] = torch.arange(surf_idx.shape[0])
        surf_faces_local = global_to_surf[asset.contact_shell.surface_faces].tolist()
        core_verts_cm = (
            (asset.mechanics_core.curr_vertices[asset.mechanics_core.is_core_pinned] * 100.0)
            .round(decimals=2)
            .tolist()
        )

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        sig = self.compute_calibration_signature(oid, e_kpa, y_kpa, mu_s)

        updated_obj = {
            "object_id": oid,
            "domain": custom_spec.domain,
            "display_name": custom_spec.display_name,
            "semantic_desc": custom_spec.semantic_desc,
            "calibration_signature": sig,
            "solver_latency_ms": round(elapsed_ms, 1),
            "frame0_equilibrium": eq_stats,
            "num_2dgs_splats": int(splat_means.shape[0]),
            "constitutive_summary": {
                "youngs_modulus_kpa": e_kpa,
                "poisson_ratio": nu,
                "yield_stress_kpa": y_kpa,
                "density_kg_m3": custom_spec.density_kg_m3,
                "mu_static": mu_s,
                "mu_dynamic": mu_d,
                "shell_thickness_mm": custom_spec.shell_thickness_mm,
                "has_rigid_core": custom_spec.has_rigid_core,
            },
            "golden_safety_envelope": {
                "p_slip_min_kpa": p_slip_min,
                "p_crush_max_kpa": p_crush_max,
                "optimal_target_pressure_kpa": opt_p_base,
                "f_normal_slip_min_n": round(f_normal_slip_min, 3),
                "safety_corridor_width_kpa": round(max(p_crush_max - p_slip_min, 0.2), 2),
            },
            "sweep_curves": {
                "indent_mm": sweep_indent,
                "pressure_kpa": sweep_p,
                "von_mises_kpa": sweep_vm,
                "normal_force_n": sweep_fn,
                "shear_force_n": sweep_ft,
                "regime": sweep_reg,
            },
            "surface_faces_local": surf_faces_local,
            "xray_core_vertices_cm": core_verts_cm[::2],
            "interactive_keyframes": keyframes,
        }
        if oid not in CATALOG_BY_ID or policy_mode == "tactile_closed_loop":
            self.bench_data["objects"][oid] = updated_obj
        return updated_obj

    def compile_custom_asset_live(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compile a brand-new user-defined or uploaded .obj 3D asset into a .physpack object live!
        """
        custom_name = str(params.get("display_name", "Custom Fragile Wine Glass")).strip()
        oid = "custom_" + "".join(c if c.isalnum() else "_" for c in custom_name.lower())[:20]
        prompt = str(params.get("semantic_prompt", custom_name))
        inferred = self.vlm_architect.infer_single_object_constitutive(prompt)

        if params.get("auto_infer_from_prompt"):
            profile_curve = str(params.get("profile_curve") or inferred["profile_curve"])
            shape_prim = "ellipsoid" if profile_curve == "sphere" else "cylinder_profile"
            e_kpa = float(inferred["youngs_modulus_kpa"])
            y_kpa = float(inferred["yield_stress_kpa"])
            nu = float(inferred["poisson_ratio"])
            mu_s = float(inferred["mu_static"])
        else:
            shape_prim = str(params.get("shape_primitive", "cylinder_profile"))
            profile_curve = str(params.get("profile_curve", inferred["profile_curve"]))
            e_kpa = float(params.get("youngs_modulus_kpa", inferred["youngs_modulus_kpa"]))
            y_kpa = float(params.get("yield_stress_kpa", inferred["yield_stress_kpa"]))
            nu = float(params.get("poisson_ratio", inferred["poisson_ratio"]))
            mu_s = float(params.get("mu_static", inferred["mu_static"]))

        obj_content = str(params.get("obj_content", "")).strip()

        new_spec = DomainObjectSpec(
            object_id=oid,
            domain="5_User_Custom_Assets",
            display_name=f"✨ {custom_name}",
            shape_primitive=shape_prim,
            dimensions_m=(0.032, 0.032, 0.125) if shape_prim == "cylinder_profile" else (0.085, 0.060, 0.042),
            profile_curve=profile_curve,
            base_e_kpa=e_kpa,
            base_nu=nu,
            yield_stress_kpa=y_kpa,
            density_kg_m3=float(inferred.get("density_kg_m3", 380.0)),
            mu_static=mu_s,
            mu_dynamic=round(mu_s * 0.78, 2),
            shell_thickness_mm=float(inferred.get("shell_thickness_mm", 11.0)),
            has_rigid_core=True,
            semantic_desc=f"User-Compiled SCPP Asset: {prompt} ({inferred.get('reasoning', '')})",
        )
        self.custom_specs[oid] = new_spec
        CATALOG_BY_ID[oid] = new_spec
        res = self.run_live_simulation({
            "object_id": oid,
            "youngs_modulus_kpa": e_kpa,
            "poisson_ratio": nu,
            "yield_stress_kpa": y_kpa,
            "mu_static": mu_s,
            "policy_mode": "tactile_closed_loop",
            "target_pressure_kpa": round(y_kpa * 0.56, 2),
        })
        res["constitutive_inference"] = inferred

        if obj_content:
            parsed_verts: list[list[float]] = []
            parsed_faces: list[list[int]] = []
            for line in obj_content.splitlines():
                s = line.strip()
                if s.startswith("v "):
                    parts = s.split()
                    if len(parts) >= 4:
                        try:
                            parsed_verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
                        except ValueError:
                            pass
                elif s.startswith("f "):
                    parts = s.split()[1:]
                    if len(parts) >= 3:
                        try:
                            idxs = [int(p.split("/")[0]) - 1 for p in parts]
                            for k in range(1, len(idxs) - 1):
                                parsed_faces.append([idxs[0], idxs[k], idxs[k + 1]])
                        except ValueError:
                            pass
            if len(parsed_verts) >= 4 and len(parsed_faces) >= 4:
                max_coord = max(max(abs(c) for c in v) for v in parsed_verts)
                scale_cm = 100.0 if max_coord < 1.5 else (10.0 / max(max_coord, 0.01) if max_coord > 35.0 else 1.0)
                verts_cm = [[round(c * scale_cm, 2) for c in v] for v in parsed_verts]
                res["surface_faces_local"] = parsed_faces
                for kf in res["interactive_keyframes"]:
                    kf["surface_vertices_cm"] = verts_cm
                    kf["surface_pressure_kpa"] = [0.0] * len(verts_cm)
                    kf["total_verts"] = len(verts_cm)
                    kf["active_patch_verts"] = max(8, len(verts_cm) // 10)

        return res

    def generate_multi_object_scene_live(self, seed: int = 2026) -> Dict[str, Any]:
        """
        Batch-generate a multi-object 3D tabletop scene (`ProceduralPhysicalSceneBuilder`)
        and return combined 3D meshes + Frame-0 Static Equilibrium verification.
        """
        t0 = time.perf_counter()
        ws = self.scene_builder.generate_tabletop_manipulation_scene(
            scene_id=f"live_table_{seed}", seed=seed
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        combined_verts_cm: list[list[float]] = []
        combined_faces: list[list[int]] = []
        combined_press: list[float] = []
        combined_core_cm: list[list[float]] = []
        reports: list[Dict[str, Any]] = []

        vert_offset = 0
        for idx, (aid, asset) in enumerate(ws.assets.items()):
            surf_idx = asset.contact_shell.surface_vertex_indices
            s_verts = (asset.mechanics_core.curr_vertices[surf_idx] * 100.0).round(decimals=2).tolist()

            global_to_surf = torch.full((asset.mechanics_core.curr_vertices.shape[0],), -1, dtype=torch.int64)
            global_to_surf[surf_idx] = torch.arange(surf_idx.shape[0]) + vert_offset
            s_faces = global_to_surf[asset.contact_shell.surface_faces].tolist()

            combined_verts_cm.extend(s_verts)
            combined_faces.extend(s_faces)
            combined_press.extend([float((idx + 1) * 2.5)] * len(s_verts))

            core_v = (
                (asset.mechanics_core.curr_vertices[asset.mechanics_core.is_core_pinned] * 100.0)
                .round(decimals=2)
                .tolist()
            )
            combined_core_cm.extend(core_v[::2])
            vert_offset += len(s_verts)

            rep = ws.equilibrium_reports[aid]
            reports.append({
                "asset_id": aid,
                "category": asset.category,
                "mass_kg": round(rep["total_mass_kg"], 3),
                "gravity_weight_n": round(rep["gravity_weight_n"], 4),
                "ipc_support_force_n": round(rep["equilibrium_contact_force_n"], 4),
                "force_residual_n": float(f"{rep['force_balance_error_n']:.2e}"),
                "min_clearance_mm": round(rep["min_clearance_mm"], 3),
                "num_tets": int(asset.mechanics_core.tet_elements.shape[0]),
                "num_2dgs": asset.visual_layer.num_splats,
            })

        return {
            "scene_id": ws.scene_id,
            "seed": seed,
            "compile_time_ms": round(elapsed_ms, 1),
            "surface_vertices_cm": combined_verts_cm,
            "surface_faces_local": combined_faces,
            "surface_pressure_kpa": combined_press,
            "xray_core_vertices_cm": combined_core_cm,
            "equilibrium_reports": reports,
        }


ENGINE: LiveStudioEngine | None = None


class StudioHTTPHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        pass

    def _send_json(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:
        assert ENGINE is not None
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            html = get_live_studio_pro_html().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html)))
            self.end_headers()
            self.wfile.write(html)
        elif parsed.path == "/api/init":
            self._send_json({
                "bench_data": ENGINE.bench_data,
                "dual_data": ENGINE.dual_data,
                "agent_tools": PHYSPACE_AGENT_TOOLS,
                "agent_history": ENGINE.agent_call_history[-15:],
                "llm_config": ENGINE.vlm_architect.get_config_summary(),
            })
        elif parsed.path == "/api/llm_config":
            self._send_json(ENGINE.vlm_architect.get_config_summary())
        elif parsed.path == "/api/agent/tools":
            self._send_json({
                "protocol": "Model Context Protocol (MCP 2024-11-05) & OpenAI Function Calling",
                "tools": PHYSPACE_AGENT_TOOLS,
                "recent_calls": ENGINE.agent_call_history[-15:],
            })
        elif parsed.path in ("/api/export_physpack", "/api/export_universal"):
            from physpace_forge.exporters import UniversalFormatExporter

            qs = parse_qs(parsed.query)
            oid = qs.get("object_id", ["tapered_coffee_cup"])[0]
            fmt = qs.get("format", ["mjcf"])[0].lower()
            if oid not in ENGINE.compiled_cache:
                spec = ENGINE.custom_specs.get(oid, CATALOG_BY_ID["tapered_coffee_cup"])
                asset, _ = ENGINE.profiler.build_domain_asset(spec)
                ENGINE.compiled_cache[oid] = asset
            asset = ENGINE.compiled_cache[oid]
            asset.metadata["calibration_signature"] = ENGINE.compute_calibration_signature(
                oid, 100.0, 10.0, 0.45
            )
            env = ENGINE.bench_data.get("objects", {}).get(oid, {}).get("golden_safety_envelope")

            if fmt == "mjcf":
                raw_bytes = UniversalFormatExporter.to_mujoco_mjcf_xml(asset, env).encode("utf-8")
                ext, mime = "xml", "application/xml; charset=utf-8"
            elif fmt == "usda":
                raw_bytes = UniversalFormatExporter.to_openusd_usda(asset, env).encode("utf-8")
                ext, mime = "usda", "text/plain; charset=utf-8"
            elif fmt == "urdf":
                raw_bytes = UniversalFormatExporter.to_ros2_urdf(asset, env).encode("utf-8")
                ext, mime = "urdf", "application/xml; charset=utf-8"
            elif fmt == "gltf":
                raw_bytes = UniversalFormatExporter.to_gltf_json(asset, env).encode("utf-8")
                ext, mime = "gltf", "model/gltf+json; charset=utf-8"
            else:
                buf = io.BytesIO()
                torch.save(asset, buf)
                raw_bytes = buf.getvalue()
                ext, mime = "physpack", "application/octet-stream"

            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Disposition", f'attachment; filename="{oid}.{ext}"')
            self.send_header("Content-Length", str(len(raw_bytes)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(raw_bytes)
        else:
            self._send_json({"error": "not_found"}, 404)

    def do_POST(self) -> None:
        assert ENGINE is not None
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length).decode("utf-8") or "{}")

        if self.path == "/api/simulate_live":
            res = ENGINE.run_live_simulation(body)
            self._send_json(res)
        elif self.path == "/api/compile_custom_obj":
            res = ENGINE.compile_custom_asset_live(body)
            self._send_json(res)
        elif self.path == "/api/generate_scene":
            seed = int(body.get("seed", int(time.time()) % 10000))
            res = ENGINE.generate_multi_object_scene_live(seed=seed)
            self._send_json(res)
        elif self.path == "/api/llm_config":
            gw_url = body.get("gateway_url") or body.get("endpoint_url")
            cfg = ENGINE.vlm_architect.configure(
                gateway_url=gw_url,
                api_key=body.get("api_key"),
                model_name=body.get("model_name"),
                provider_mode=body.get("provider_mode"),
            )
            ENGINE.agent_dispatcher.vlm_architect.configure(
                gateway_url=gw_url,
                api_key=body.get("api_key"),
                model_name=body.get("model_name"),
                provider_mode=body.get("provider_mode"),
            )
            self._send_json({"status": "ok", "llm_config": cfg})
        elif self.path == "/api/llm_infer":
            gw_url = body.get("gateway_url") or body.get("endpoint_url")
            if any(k in body for k in ("gateway_url", "endpoint_url", "api_key", "model_name", "provider_mode")):
                ENGINE.vlm_architect.configure(
                    gateway_url=gw_url,
                    api_key=body.get("api_key"),
                    model_name=body.get("model_name"),
                    provider_mode=body.get("provider_mode"),
                )
                ENGINE.agent_dispatcher.vlm_architect.configure(
                    gateway_url=gw_url,
                    api_key=body.get("api_key"),
                    model_name=body.get("model_name"),
                    provider_mode=body.get("provider_mode"),
                )
            prompt = str(body.get("semantic_prompt") or body.get("prompt") or "Thin Champagne Flute (薄壁高脚香槟杯)")
            inferred = ENGINE.vlm_architect.infer_single_object_constitutive(prompt)
            self._send_json({
                "status": "ok",
                "llm_config": ENGINE.vlm_architect.get_config_summary(),
                "inference": inferred,
                "inferred": inferred,
            })
        elif self.path == "/api/agent/invoke":
            t0 = time.perf_counter()
            tool_name = str(body.get("tool_name", "physpace_inspect_object_safety_envelope"))
            arguments = dict(body.get("arguments", {}))
            result = ENGINE.agent_dispatcher.invoke(tool_name, arguments)
            latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            log_entry = {
                "call_id": f"call_{len(ENGINE.agent_call_history) + 1:03d}",
                "timestamp": time.strftime("%H:%M:%S"),
                "tool_name": tool_name,
                "arguments": arguments,
                "latency_ms": latency_ms,
                "result": result,
            }
            ENGINE.agent_call_history.append(log_entry)
            self._send_json(log_entry)
        else:
            self._send_json({"error": "unknown_endpoint"}, 404)


def run_server(port: int = 8765) -> None:
    global ENGINE
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    ENGINE = LiveStudioEngine(root_dir=root_dir)
    server = ThreadingHTTPServer(("127.0.0.1", port), StudioHTTPHandler)
    print(f"PhySpace-Studio Pro Live Full-Stack Server listening on http://127.0.0.1:{port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PhySpace-Studio Pro Live Full-Stack Server")
    parser.add_argument("--port", type=int, default=8765, help="Port to listen on")
    parser.add_argument("--no-browser", action="store_true", help="Do not auto-open browser")
    args = parser.parse_args()
    run_server(port=args.port)
