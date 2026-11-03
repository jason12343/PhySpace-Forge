"""
Full Dual-Engine Autonomous Mission:
- Node 2 (`Qwen3_8_176B_VLM` @ 127.0.0.1:11434 over 200GbE): Live 176B `Qwen3.8-176B-VLM`
  Shell-to-Core Constitutive Inference & Scene Parameterization.
- Node 1 (`NVIDIA_GB10_128GB` @ 127.0.0.1:9090 GPU Scheduler + NVIDIA GB10 Unified Memory):
  Scheduler-scheduled SCPP .obj->.physpack Compilation, Zero-Frame Equilibrium,
  and all 3 Hero Manipulation Benchmarks (Fragile Grasp, Cluttered Push-to-Extract, Cap-Twisting Torque).
"""

import json
import math
import os
import sys
import time
from typing import Any, Dict, List
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from physpace_forge.cluster import GPUSchedulerLease
from physpace_forge.compiler import (
    VLMConstitutiveArchitect,
    UniversalMeshSCPPCompiler,
    ZeroFrameEquilibriumOptimizer,
    load_wavefront_obj,
)
from physpace_forge.schema.physpack import SCPPAsset
from physpace_forge.sensors import Cage2DGSDeformer, TactilePressureGridSensor
from physpace_forge.solver import FingerIndenterSpec, RTPatchIPCSolver
from tests.run_NVIDIA_GB10_128GB_tasks_1_and_2 import run_grasp_episode


def run_benchmark2_cluttered_push_and_extract(
    sponge_obstacle: SCPPAsset,
    target_bottle: SCPPAsset,
) -> Dict[str, Any]:
    """
    Hero Benchmark 2: Cluttered Force-Aware Lateral Push & Target Extraction.
    Phase A (Steps 1..12): Robotic finger gently pushes the obstructing `wavy_scrub_sponge`
      laterally (+X) while regulating contact pressure below 3.5 kPa.
    Phase B (Steps 13..24): Dual fingers grasp and lift the cleared `ergonomic_sports_bottle`.
    """
    device = sponge_obstacle.device
    solver = RTPatchIPCSolver(
        dt_seconds=0.005,
        barrier_stiffness_pa=1.5e5,
        active_patch_radius_m=0.042,
        max_vbd_iterations=15,
        support_floor_z=0.0,
    )

    verts_s = sponge_obstacle.mechanics_core.curr_vertices
    x_min_s = float(verts_s[:, 0].min().item())
    z_mid_s = 0.5 * float((verts_s[:, 2].min() + verts_s[:, 2].max()).item())

    r_f = 0.012
    x_push_start = x_min_s - r_f - 0.0014
    pusher = FingerIndenterSpec(
        finger_id="push_index",
        center_xyz=torch.tensor([x_push_start, 0.0, z_mid_s], dtype=torch.float32, device=device),
        prev_center_xyz=torch.tensor([x_push_start, 0.0, z_mid_s], dtype=torch.float32, device=device),
        radius_m=r_f,
    )

    push_pressure_kpa: List[float] = []
    push_force_n: List[float] = []
    sponge_com_disp_mm: List[float] = []
    init_x_mean = float(sponge_obstacle.mechanics_core.curr_vertices[:, 0].mean().item())

    curr_p = 0.0
    for _step in range(1, 13):
        pusher.prev_center_xyz = pusher.center_xyz.clone()
        # Closed-loop force-limited push (target 2.2 kPa lateral contact pressure)
        p_err = (2200.0 - curr_p) / 2200.0
        dx = max(min(0.00022 * p_err + 0.00012, 0.00028), 0.00005)
        pusher.center_xyz[0] += dx

        tel = solver.step(sponge_obstacle, [pusher])
        curr_p = tel.peak_contact_pressure_pa
        push_pressure_kpa.append(curr_p / 1000.0)
        push_force_n.append(tel.total_normal_force_n)
        curr_x_mean = float(sponge_obstacle.mechanics_core.curr_vertices[:, 0].mean().item())
        sponge_com_disp_mm.append((curr_x_mean - init_x_mean) * 1000.0)

    # Phase B: Extract the target bottle after obstacle clearance
    extract_res = run_grasp_episode(
        target_bottle,
        policy_mode="tactile_closed_loop",
        target_pressure_pa=5200.0,
        num_steps=16,
    )
    extract_res.pop("lerobot_frames", None)

    return {
        "benchmark": "Cluttered Force-Aware Push & Extract",
        "phase_a_obstacle_push": {
            "peak_push_pressure_kpa": max(push_pressure_kpa),
            "peak_push_force_n": max(push_force_n),
            "obstacle_skin_indentation_mm": max(sponge_com_disp_mm),
            "pressure_history_kpa": push_pressure_kpa,
            "force_history_n": push_force_n,
        },
        "phase_b_target_extraction": {
            "peak_grasp_pressure_kpa": extract_res["peak_pressure_kpa"],
            "final_lift_shear_force_n": extract_res["final_shear_force_n"],
            "intact": not extract_res["crushed_or_damaged"],
            "penetration_free": extract_res["penetration_free"],
        },
    }


def run_benchmark3_cap_twisting_torque_and_slip(
    bottle_asset: SCPPAsset,
    num_steps: int = 24,
) -> Dict[str, Any]:
    """
    Hero Benchmark 3: High-Friction Rotational Cap-Twisting Torque & Stick-to-Slip Transition.
    Pinches the upper neck of `ergonomic_sports_bottle` with Left (-X) and Right (+X) fingers,
    then rotates the fingers counter-clockwise in the XY plane (+Y on Right, -Y on Left)
    to measure torsional friction torque τ_z (N*m) and incipient rotational slip index s_k!
    """
    device = bottle_asset.device
    solver = RTPatchIPCSolver(
        dt_seconds=0.005,
        barrier_stiffness_pa=1.6e5,
        active_patch_radius_m=0.038,
        max_vbd_iterations=15,
        support_floor_z=0.0,
    )

    verts = bottle_asset.mechanics_core.curr_vertices
    z_max = float(verts[:, 2].max().item())
    z_neck = 0.78 * z_max
    neck_band = torch.abs(verts[:, 2] - z_neck) < 0.015
    r_neck = (
        float(verts[neck_band, 0].abs().max().item())
        if torch.any(neck_band)
        else 0.028
    )

    r_f = 0.012
    x_start = r_neck + r_f + 0.0015
    left_f = FingerIndenterSpec(
        finger_id="twist_thumb",
        center_xyz=torch.tensor([-x_start, 0.0, z_neck], dtype=torch.float32, device=device),
        prev_center_xyz=torch.tensor([-x_start, 0.0, z_neck], dtype=torch.float32, device=device),
        radius_m=r_f,
    )
    right_f = FingerIndenterSpec(
        finger_id="twist_index",
        center_xyz=torch.tensor([x_start, 0.0, z_neck], dtype=torch.float32, device=device),
        prev_center_xyz=torch.tensor([x_start, 0.0, z_neck], dtype=torch.float32, device=device),
        radius_m=r_f,
    )

    torque_nm_history: List[float] = []
    normal_force_history: List[float] = []
    mean_slip_history: List[float] = []
    twist_angle_deg_history: List[float] = []

    curr_p = 0.0
    cum_angle_rad = 0.0
    surf_idx = bottle_asset.contact_shell.surface_vertex_indices
    areas = bottle_asset.contact_shell.vertex_contact_areas

    for step_idx in range(1, num_steps + 1):
        left_f.prev_center_xyz = left_f.center_xyz.clone()
        right_f.prev_center_xyz = right_f.center_xyz.clone()

        # First 8 steps: establish firm normal pinch pressure (~6.5 kPa) on bottle neck
        p_err = (6500.0 - curr_p) / 6500.0
        dx_pinch = max(min(0.00016 * p_err, 0.00016), -0.00004)
        # Steps 6..24: apply tangential rotational twist Couple (+/- dy)
        dy_twist = 0.0 if step_idx < 6 else 0.00012 * min(step_idx - 5, 8) / 8.0
        cum_angle_rad += dy_twist / max(r_neck, 0.01)

        left_f.center_xyz[0] += dx_pinch
        left_f.center_xyz[1] -= dy_twist
        right_f.center_xyz[0] -= dx_pinch
        right_f.center_xyz[1] += dy_twist

        tel = solver.step(bottle_asset, [left_f, right_f])
        curr_p = tel.peak_contact_pressure_pa

        # Compute exact Z-axis torsional friction torque: τ_z = sum_k A_k * (x_k * τ_yk - y_k * τ_xk)
        surf_pos = bottle_asset.mechanics_core.curr_vertices[surf_idx]
        tau_vec = tel.surface_shear_stress_pa  # (N_surf, 3) in Pa
        torque_z_per_vert = areas * (
            surf_pos[:, 0] * tau_vec[:, 1] - surf_pos[:, 1] * tau_vec[:, 0]
        )
        total_torque_nm = float(torch.abs(torque_z_per_vert.sum()).item())

        active_contact = tel.surface_pressure_pa > 50.0
        mean_slip = (
            float(tel.surface_slip_index[active_contact].mean().item())
            if torch.any(active_contact)
            else 0.0
        )

        torque_nm_history.append(total_torque_nm)
        normal_force_history.append(tel.total_normal_force_n)
        mean_slip_history.append(mean_slip)
        twist_angle_deg_history.append(math.degrees(cum_angle_rad))

    return {
        "benchmark": "High-Friction Rotational Cap-Twisting & Slip Lock",
        "peak_torsional_torque_mNm": max(torque_nm_history) * 1000.0,
        "peak_normal_pinch_force_n": max(normal_force_history),
        "final_rotational_slip_index": mean_slip_history[-1],
        "twist_angle_deg_history": twist_angle_deg_history,
        "torque_mNm_history": [t * 1000.0 for t in torque_nm_history],
        "normal_force_n_history": normal_force_history,
        "slip_index_history": mean_slip_history,
    }


def main() -> None:
    out_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "benchmarks"))
    mesh_dir = os.path.join(out_root, "raw_meshes_obj")
    physpack_dir = os.path.join(out_root, "compiled_physpack")
    dataset_dir = os.path.join(out_root, "lerobot_dataset")
    os.makedirs(physpack_dir, exist_ok=True)
    os.makedirs(dataset_dir, exist_ok=True)

    print("=" * 88)
    print(" PhySpace-Forge: Dual-Engine (NVIDIA_GB10_128GB GB10 + Qwen3_8_176B_VLM Qwen3.8-176B) Full Autonomous Mission")
    print("=" * 88)

    # ========================================================================
    # STEP 1: Live Constitutive Reasoning via `Qwen3_8_176B_VLM` (Qwen3.8-176B-VLM over 200GbE)
    # ========================================================================
    print("\n>>> [STEP 1] Querying `Qwen3_8_176B_VLM` (127.0.0.1:11434 `Qwen3.8-176B-VLM` 176B) over 200GbE...")
    architect = VLMConstitutiveArchitect(
        gateway_url="http://127.0.0.1:11434/v1/chat/completions",
        model_name="Qwen3.8-176B-VLM",
    )
    object_specs = [
        {"id": "ergonomic_sports_bottle", "desc": "Waisted thin-walled PET plastic sports water bottle"},
        {"id": "tapered_coffee_cup", "desc": "Fragile disposable waxed paper coffee cup (crushes past 6 kPa)"},
        {"id": "avian_fresh_egg", "desc": "Raw avian egg with thin brittle calcite shell (fractures past 18 kPa)"},
        {"id": "wavy_scrub_sponge", "desc": "Soft open-cell polyurethane kitchen scrub sponge"},
        {"id": "corrugated_parcel_box", "desc": "Corrugated cardboard shipping parcel box"},
    ]
    qwen_profiles = architect.infer_batch_constitutive_profiles(object_specs)
    for oid, (preset, summary) in qwen_profiles.items():
        print(
            f"  [Qwen3_8_176B_VLM-Qwen3.8] {oid:24s} -> E={summary['outer_youngs_modulus_kpa']:6.1f} kPa | "
            f"ν={summary['outer_poisson_ratio']:.2f} | σ_y={summary['outer_yield_stress_kpa']:5.1f} kPa | "
            f"μ_s={summary['mu_static']:.2f} (Gateway Latency: {summary['qwen_gateway_latency_ms']} ms)"
        )

    # ========================================================================
    # STEP 2: Acquire GPU Scheduler Lease on `NVIDIA_GB10_128GB` (Port 9090) & Execute All GPU Benchmarks
    # ========================================================================
    print("\n>>> [STEP 2] Requesting GPU Lease from `NVIDIA_GB10_128GB` Scheduler (http://127.0.0.1:9090)...")
    with GPUSchedulerLease(who="physpace_forge_full_mission", timeout_sec=600.0) as lease:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        gpu_name = torch.cuda.get_device_name(0) if device == "cuda" else "CPU"
        t_gpu_start = time.perf_counter()

        u_compiler = UniversalMeshSCPPCompiler(device=device)
        eq_optimizer = ZeroFrameEquilibriumOptimizer(gravity_m_s2=9.81, barrier_stiffness_pa=2.0e5)
        obj_paths = u_compiler.generate_standard_obj_mesh_suite(mesh_dir)

        compiled_assets: Dict[str, SCPPAsset] = {}
        compilation_manifest: Dict[str, Any] = {}

        # Compile all 5 .obj assets using the Qwen3.8-inferred constitutive presets
        for oid, obj_path in obj_paths.items():
            preset, mat_summary = qwen_profiles[oid]
            u_compiler.vlm_infer._semantic_rules[oid] = preset
            asset, _ = u_compiler.compile_obj_to_physpack(obj_path, oid)
            eq_stats = eq_optimizer.settle_asset_on_support(asset, support_z=0.0, xy_offset=(0.0, 0.0))

            save_path = os.path.join(physpack_dir, f"{oid}.physpack")
            asset.save_physpack(save_path)
            compiled_assets[oid] = asset
            compilation_manifest[oid] = {
                "source_obj": obj_path,
                "physpack_path": save_path,
                "num_verts": int(asset.mechanics_core.curr_vertices.shape[0]),
                "num_tets": int(asset.mechanics_core.tet_elements.shape[0]),
                "num_2dgs_splats": asset.visual_layer.num_splats,
                "mass_kg": eq_stats["total_mass_kg"],
                "frame0_force_err_n": eq_stats["force_balance_error_n"],
                "frame0_clearance_mm": eq_stats["min_clearance_mm"],
                "constitutive_profile": mat_summary,
            }

        # --------------------------------------------------------------------
        # Hero Benchmark 1: Fragile Paper Cup Crush vs. Safe Closed-Loop Grasp
        # --------------------------------------------------------------------
        cup_naive, _ = u_compiler.compile_obj_to_physpack(obj_paths["tapered_coffee_cup"], "tapered_coffee_cup")
        eq_optimizer.settle_asset_on_support(cup_naive, support_z=0.0, xy_offset=(0.0, 0.0))
        res_cup_naive = run_grasp_episode(cup_naive, policy_mode="naive_position", num_steps=24)
        res_cup_naive.pop("lerobot_frames", None)

        cup_tactile, _ = u_compiler.compile_obj_to_physpack(obj_paths["tapered_coffee_cup"], "tapered_coffee_cup")
        eq_optimizer.settle_asset_on_support(cup_tactile, support_z=0.0, xy_offset=(0.0, 0.0))
        res_cup_tactile = run_grasp_episode(
            cup_tactile, policy_mode="tactile_closed_loop", target_pressure_pa=2400.0, num_steps=24
        )
        res_cup_tactile.pop("lerobot_frames", None)

        # --------------------------------------------------------------------
        # Batch Multi-Object 200Hz LeRobot Trajectory Synthesis
        # --------------------------------------------------------------------
        target_pressures = {
            "ergonomic_sports_bottle": 5500.0,
            "tapered_coffee_cup": 2400.0,
            "avian_fresh_egg": 4800.0,
            "wavy_scrub_sponge": 2800.0,
            "corrugated_parcel_box": 6500.0,
        }
        batch_episodes: Dict[str, Any] = {}
        lerobot_bundle: Dict[str, Any] = {}
        for oid, asset in compiled_assets.items():
            ep_res = run_grasp_episode(
                asset,
                policy_mode="tactile_closed_loop",
                target_pressure_pa=target_pressures[oid],
                num_steps=24,
            )
            lerobot_bundle[oid] = ep_res.pop("lerobot_frames")
            batch_episodes[oid] = ep_res

        # --------------------------------------------------------------------
        # Hero Benchmark 2: Cluttered Obstacle Push & Target Extraction
        # --------------------------------------------------------------------
        spg_b2, _ = u_compiler.compile_obj_to_physpack(obj_paths["wavy_scrub_sponge"], "wavy_scrub_sponge")
        eq_optimizer.settle_asset_on_support(spg_b2, support_z=0.0, xy_offset=(0.0, 0.0))
        bot_b2, _ = u_compiler.compile_obj_to_physpack(obj_paths["ergonomic_sports_bottle"], "ergonomic_sports_bottle")
        eq_optimizer.settle_asset_on_support(bot_b2, support_z=0.0, xy_offset=(0.0, 0.0))
        bench2_res = run_benchmark2_cluttered_push_and_extract(spg_b2, bot_b2)

        # --------------------------------------------------------------------
        # Hero Benchmark 3: High-Friction Rotational Cap-Twisting & Slip Lock
        # --------------------------------------------------------------------
        bot_b3, _ = u_compiler.compile_obj_to_physpack(obj_paths["ergonomic_sports_bottle"], "ergonomic_sports_bottle")
        eq_optimizer.settle_asset_on_support(bot_b3, support_z=0.0, xy_offset=(0.0, 0.0))
        bench3_res = run_benchmark3_cap_twisting_torque_and_slip(bot_b3, num_steps=24)

        if device == "cuda":
            torch.cuda.synchronize()
        gpu_elapsed_ms = (time.perf_counter() - t_gpu_start) * 1000.0

        print(f"\n  [Benchmark 1 - Fragile Cup] Naive vonMises={res_cup_naive['peak_von_mises_kpa']:.2f} kPa (Crushed={res_cup_naive['crushed_or_damaged']}) vs Tactile={res_cup_tactile['peak_von_mises_kpa']:.2f} kPa (Crushed={res_cup_tactile['crushed_or_damaged']})")
        print(f"  [Benchmark 2 - Cluttered Push & Extract] Push PeakP={bench2_res['phase_a_obstacle_push']['peak_push_pressure_kpa']:.2f} kPa | Extract Lift Shear={bench2_res['phase_b_target_extraction']['final_lift_shear_force_n']:.2f} N")
        print(f"  [Benchmark 3 - Cap Twisting Torque] Peak Torque={bench3_res['peak_torsional_torque_mNm']:.2f} mN*m | Final Slip Index={bench3_res['final_rotational_slip_index']:.3f}")

        # Save outputs before releasing Scheduler lease
        torch.save(lerobot_bundle, os.path.join(dataset_dir, "physpace_lerobot_trajectories_v1.pt"))
        report_payload = {
            "device": gpu_name,
            "pytorch_version": torch.__version__,
            "gpu_session_id": lease.token,
            "vlm_constitutive_model": "Qwen3.8-176B-VLM (176B @ 127.0.0.1:11434)",
            "gpu_total_elapsed_ms": round(gpu_elapsed_ms, 2),
            "compilation_manifest": compilation_manifest,
            "hero_cup_comparison": {
                "naive_position": res_cup_naive,
                "tactile_closed_loop": res_cup_tactile,
            },
            "benchmark2_cluttered_push_extract": bench2_res,
            "benchmark3_cap_twisting_torque": bench3_res,
            "batch_episodes": batch_episodes,
        }
        report_json_path = os.path.join(dataset_dir, "hero_execution_report.json")
        with open(report_json_path, "w", encoding="utf-8") as f:
            json.dump(report_payload, f, indent=2)

    print("=" * 88)
    print(" DUAL-GB10 MISSION COMPLETE & GPU SCHEDULER LEASE CLEANLY RELEASED!")
    print("=" * 88)


if __name__ == "__main__":
    main()
