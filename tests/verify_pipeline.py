"""
End-to-End Verification & Hero Benchmark Suite for PhySpace-Forge.

Runs cleanly on CPU (zero interference with running GPU tasks) or CUDA, verifying:
1. Phase 1: SCPP Shell-to-Core Constitutive Compilation (.physpack)
2. Phase 3: Zero-Frame Static Equilibrium Optimization (M*g == IPC Barrier Force, d_min > 0)
3. Phase 2: RT-Patch-IPC Local Micro-Contact Solver (Active Patch wake-up + VBD + zero penetration)
4. Phase 4: 200Hz 16x16 Pascal Tactile Pressure Map, Shear Stress, Incipient Slip & 2DGS Deformation
"""

import sys
import os
import torch

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from physpace_forge.compiler import (
    SCPPCompiler,
    ZeroFrameEquilibriumOptimizer,
    ProceduralPhysicalSceneBuilder,
)
from physpace_forge.solver import RTPatchIPCSolver, FingerIndenterSpec
from physpace_forge.sensors import TactilePressureGridSensor, Cage2DGSDeformer


def format_ascii_pressure_heatmap(pressure_grid_pa: torch.Tensor) -> str:
    """Render a compact ASCII heatmap of the 2D fingertip pressure distribution (in kPa)."""
    chars = " .:-=+*#%@"
    max_p = float(pressure_grid_pa.max().item())
    if max_p <= 1.0:
        return "[No active contact pressure]"
    lines = []
    H, W = pressure_grid_pa.shape
    for r in range(0, H, 2):
        row_str = "    |"
        for c in range(W):
            val = float(pressure_grid_pa[r, c].item()) / max_p
            idx = min(int(val * (len(chars) - 1)), len(chars) - 1)
            row_str += chars[idx]
        row_str += "|"
        lines.append(row_str)
    return "\n".join(lines)


def main() -> None:
    # Use CPU by default so we do not disturb any full-load GPU jobs currently running
    device = "cpu"
    print("=" * 78)
    print(" PhySpace-Forge: End-to-End Mathematical & Pipeline Verification")
    print(f" Execution Backend: {device.upper()} (Zero GPU Interference Mode)")
    print("=" * 78)

    # ------------------------------------------------------------------------
    # 1. Phase 1: Compile Deformable PET Bottle into SCPP (.physpack)
    # ------------------------------------------------------------------------
    compiler = SCPPCompiler(device=device)
    bottle_asset = compiler.compile_cylindrical_vessel(
        asset_id="hero_deformable_pet_bottle_01",
        radius_m=0.035,
        height_m=0.16,
        num_radial=4,
        num_circumferential=16,
        num_vertical=12,
        composite_preset_key="deformable_pet_bottle",
        splats_per_surface_face=4,
        d_hat_meters=2.0e-3,
    )

    core = bottle_asset.mechanics_core
    shell = bottle_asset.contact_shell
    vis = bottle_asset.visual_layer
    core_pinned_count = int(core.is_core_pinned.sum().item())

    print("\n[Phase 1] SCPP Shell-to-Core Constitutive Compilation Complete:")
    print(f"  - Asset ID                 : {bottle_asset.asset_id}")
    print(f"  - Total Tetrahedral Nodes  : {core.curr_vertices.shape[0]} (Affine Core Pinned: {core_pinned_count})")
    print(f"  - Total Tetrahedra         : {core.tet_elements.shape[0]} (min V_e = {float(core.rest_volumes.min()):.3e} m^3 > 0)")
    print(f"  - Outer Contact Vertices   : {shell.surface_vertex_indices.shape[0]} (Total Area = {bottle_asset.metadata['surface_area_m2']} m^2)")
    print(f"  - Bound 2DGS Visual Splats : {vis.num_splats}")
    print(f"  - Total Calibrated Mass    : {bottle_asset.metadata['total_mass_kg']} kg")

    # ------------------------------------------------------------------------
    # 2. Phase 3: Zero-Frame Static Equilibrium Optimization on Table (z = 0)
    # ------------------------------------------------------------------------
    eq_optimizer = ZeroFrameEquilibriumOptimizer(gravity_m_s2=9.81, barrier_stiffness_pa=2.0e5)
    eq_stats = eq_optimizer.settle_asset_on_support(
        asset=bottle_asset,
        support_z=0.0,
        xy_offset=(0.0, 0.0),
    )

    print("\n[Phase 3] Zero-Frame Static Equilibrium Verification (Table z = 0.0m):")
    print(f"  - Gravitational Weight M*g : {eq_stats['gravity_weight_n']:.6f} N")
    print(f"  - Upward IPC Barrier Force : {eq_stats['equilibrium_contact_force_n']:.6f} N")
    print(f"  - Static Force Residual    : {eq_stats['force_balance_error_n']:.2e} N")
    print(f"  - Min Bottom Clearance d_0 : {eq_stats['min_clearance_mm']:.4f} mm (Penetration-Free: {bool(eq_stats['penetration_free'])})")
    assert eq_stats["penetration_free"] == 1.0, "Frame 0 must be strictly penetration-free!"
    assert eq_stats["force_balance_error_n"] < 1e-3, "Frame 0 static equilibrium residual too large!"

    # ------------------------------------------------------------------------
    # 3. Phase 2 & 4: Dual-Finger Pinch & Micro-Slip Simulation (RT-Patch-IPC)
    # ------------------------------------------------------------------------
    solver = RTPatchIPCSolver(
        dt_seconds=0.005,              # 200 Hz
        barrier_stiffness_pa=1.5e5,
        active_patch_radius_m=0.038,
        max_vbd_iterations=18,
        support_floor_z=0.0,
    )
    tactile_sensor = TactilePressureGridSensor(grid_height=16, grid_width=16)

    # Place Left (-X) and Right (+X) robotic fingertips at mid-height z = 0.08m
    finger_radius = 0.012
    z_grasp = 0.08
    # Start just inside the d_hat=2mm barrier zone and pinch inward + lift slightly (+dz) to induce shear
    x_start = 0.035 + finger_radius + 0.0016

    left_finger = FingerIndenterSpec(
        finger_id="left_thumb",
        center_xyz=torch.tensor([-x_start, 0.0, z_grasp], dtype=torch.float32),
        prev_center_xyz=torch.tensor([-x_start, 0.0, z_grasp], dtype=torch.float32),
        radius_m=finger_radius,
    )
    right_finger = FingerIndenterSpec(
        finger_id="right_index",
        center_xyz=torch.tensor([x_start, 0.0, z_grasp], dtype=torch.float32),
        prev_center_xyz=torch.tensor([x_start, 0.0, z_grasp], dtype=torch.float32),
        radius_m=finger_radius,
    )

    print("\n[Phase 2] Running RT-Patch-IPC 200Hz Pinch & Lift Trajectory (12 Steps)...")
    print("  Step | ActivePatch | Clearance(mm) | PeakPress(kPa) | NormalForce(N) | ShearForce(N) | SlipIdx")
    print("  " + "-" * 90)

    telemetry = None
    for step_idx in range(1, 13):
        left_finger.prev_center_xyz = left_finger.center_xyz.clone()
        right_finger.prev_center_xyz = right_finger.center_xyz.clone()

        # Pinch inward by 0.22 mm/step for first 8 steps, then apply upward lifting shear (0.15 mm/step)
        dx_pinch = 0.00022 if step_idx <= 8 else 0.00005
        dz_lift = 0.00002 if step_idx <= 8 else 0.00018

        left_finger.center_xyz[0] += dx_pinch
        left_finger.center_xyz[2] += dz_lift
        right_finger.center_xyz[0] -= dx_pinch
        right_finger.center_xyz[2] += dz_lift

        telemetry = solver.step(bottle_asset, [left_finger, right_finger])
        assert telemetry.penetration_free, f"Penetration occurred at step {step_idx}!"

        if step_idx in (1, 4, 8, 12):
            active_str = (
                f"{telemetry.active_patch_vertex_count:3d}/{telemetry.total_vertex_count:3d} "
                f"(-{telemetry.compute_reduction_ratio * 100:4.1f}%)"
            )
            max_slip = float(telemetry.surface_slip_index.max().item())
            print(
                f"   {step_idx:2d}  | {active_str} |    {telemetry.min_contact_clearance_mm:6.4f}     | "
                f"    {telemetry.peak_contact_pressure_pa / 1000.0:6.2f}     | "
                f"    {telemetry.total_normal_force_n:6.3f}     | "
                f"   {telemetry.total_tangential_force_n:6.3f}     |  {max_slip:5.3f}"
            )

    # ------------------------------------------------------------------------
    # 4. Phase 4: Sense 16x16 Fingertip Tactile Grid & Update 2DGS Splats
    # ------------------------------------------------------------------------
    assert telemetry is not None
    left_obs = tactile_sensor.sense_finger_pad(
        asset=bottle_asset,
        finger=left_finger,
        telemetry=telemetry,
        approach_axis=torch.tensor([1.0, 0.0, 0.0]),
        up_axis=torch.tensor([0.0, 0.0, 1.0]),
    )
    right_obs = tactile_sensor.sense_finger_pad(
        asset=bottle_asset,
        finger=right_finger,
        telemetry=telemetry,
        approach_axis=torch.tensor([-1.0, 0.0, 0.0]),
        up_axis=torch.tensor([0.0, 0.0, 1.0]),
    )
    lerobot_dict = tactile_sensor.to_lerobot_observation_dict([left_obs, right_obs])
    deformed_means, deformed_covs = Cage2DGSDeformer.deform_splats(bottle_asset)

    max_splat_disp_mm = float(
        torch.linalg.norm(
            core.curr_vertices - core.rest_vertices, dim=-1
        ).max().item()
    ) * 1000.0

    print("\n[Phase 4] 200Hz Visuotactile & 2DGS Synchronization Verification:")
    print(f"  - Left Thumb 16x16 Peak Grid Pressure : {float(left_obs.pressure_map_pa.max()) / 1000.0:.2f} kPa")
    print(f"  - Left Thumb Net 3D Contact Wrench    : [{left_obs.net_contact_force_n[0]:.3f}, {left_obs.net_contact_force_n[1]:.3f}, {left_obs.net_contact_force_n[2]:.3f}] N")
    print(f"  - Peak Internal von Mises Stress      : {telemetry.peak_von_mises_stress_pa / 1000.0:.2f} kPa (Yield Exceeded: {telemetry.material_yield_exceeded})")
    print(f"  - Max Surface Elastic Indentation     : {max_splat_disp_mm:.3f} mm")
    print(f"  - Deformed 2DGS Splat Tensor Shapes   : centers={tuple(deformed_means.shape)}, covs={tuple(deformed_covs.shape)}")
    print(f"  - LeRobot / ForceVLA Export Keys      : {list(lerobot_dict.keys())}")
    print("\n  Left Thumb 16x16 Tactile Pressure Distribution Map P(u,v) [ASCII Visualization]:")
    print(format_ascii_pressure_heatmap(left_obs.pressure_map_pa))

    # ------------------------------------------------------------------------
    # 5. Multi-Object Tabletop Scene Synthesis (Bottle + Fragile Cup + Sponge)
    # ------------------------------------------------------------------------
    scene_builder = ProceduralPhysicalSceneBuilder(device=device)
    scene = scene_builder.generate_tabletop_manipulation_scene(
        scene_id="kitchen_table_001", seed=2026
    )
    print("\n[Phase 3+] Batch Multi-Object 3D Scene Generation ('kitchen_table_001'):")
    for aid, rep in scene.equilibrium_reports.items():
        print(
            f"  - {aid:34s} | mass={rep['total_mass_kg']:.3f}kg | "
            f"F_err={rep['force_balance_error_n']:.2e}N | d_min={rep['min_clearance_mm']:.3f}mm"
        )

    print("\n" + "=" * 78)
    print(" ALL MATHEMATICAL & PIPELINE ASSERTIONS PASSED (100% PENETRATION-FREE)!")
    print("=" * 78)


if __name__ == "__main__":
    main()
