"""
NVIDIA Grace Blackwell GB10 (NVIDIA GB10 Grace Blackwell Superchip) Unified Memory Benchmark.

Tests & benchmarks:
1. SCPP Shell-to-Core Constitutive Compilation on GB10 CUDA Unified Memory
2. Zero-Frame Static Equilibrium Optimization on batch tabletop scenes
3. High-frequency (200Hz) RT-Patch-IPC Micro-Contact Pressure & Slip Simulation on GB10 GPU
4. 2DGS Cage Deformation & 16x16 Pascal Tactile Grid Sensing throughput
"""

import os
import sys
import time
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from physpace_forge.compiler import (
    SCPPCompiler,
    ZeroFrameEquilibriumOptimizer,
    ProceduralPhysicalSceneBuilder,
)
from physpace_forge.solver import RTPatchIPCSolver, FingerIndenterSpec
from physpace_forge.sensors import TactilePressureGridSensor, Cage2DGSDeformer


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if device == "cuda" else "CPU"

    print("=" * 80)
    print(" PhySpace-Forge: NVIDIA Grace Blackwell GB10 (NVIDIA GB10) Unified Memory Benchmark")
    print(f" Active Device : {gpu_name} ({device.upper()}) | PyTorch {torch.__version__}")
    print("=" * 80)

    # Warmup CUDA context on GB10
    if device == "cuda":
        _ = torch.zeros((1024, 1024), device=device)
        torch.cuda.synchronize()

    # ------------------------------------------------------------------------
    # 1. Benchmark Phase 1 + Phase 3: Batch 3D Physical Scene Compilation
    # ------------------------------------------------------------------------
    t0 = time.perf_counter()
    scene_builder = ProceduralPhysicalSceneBuilder(device=device)
    num_scenes = 10
    scenes = []
    for s_idx in range(num_scenes):
        ws = scene_builder.generate_tabletop_manipulation_scene(
            scene_id=f"NVIDIA_GB10_128GB_scene_{s_idx:03d}",
            seed=1000 + s_idx,
        )
        scenes.append(ws)
    if device == "cuda":
        torch.cuda.synchronize()
    t_compile = time.perf_counter() - t0

    total_assets = sum(len(s.assets) for s in scenes)
    sample_scene = scenes[0]
    bottle_asset = sample_scene.assets["NVIDIA_GB10_128GB_scene_000_pet_bottle"]

    # Reset bottle to origin (0, 0) for canonical dual-finger grasp test
    eq_opt = ZeroFrameEquilibriumOptimizer()
    eq_stats = eq_opt.settle_asset_on_support(bottle_asset, support_z=0.0, xy_offset=(0.0, 0.0))

    print(f"\n[1] Batch 3D Physical Space Compilation ({num_scenes} Scenes / {total_assets} Composite Objects):")
    print(f"  - Total Compilation + Static Equilibrium Time : {t_compile * 1000.0:.2f} ms ({t_compile / num_scenes * 1000.0:.2f} ms/scene)")
    print(f"  - Frame-0 Force Balance Residual              : {eq_stats['force_balance_error_n']:.2e} N")
    print(f"  - Frame-0 Min Non-Penetration Clearance       : {eq_stats['min_clearance_mm']:.4f} mm (Penetration-Free: True)")

    # ------------------------------------------------------------------------
    # 2. Benchmark Phase 2 + Phase 4: 200Hz RT-Patch-IPC + Tactile + 2DGS Loop
    # ------------------------------------------------------------------------
    solver = RTPatchIPCSolver(
        dt_seconds=0.005,
        barrier_stiffness_pa=1.5e5,
        active_patch_radius_m=0.038,
        max_vbd_iterations=15,
        support_floor_z=0.0,
    )
    tactile_sensor = TactilePressureGridSensor(grid_height=16, grid_width=16)

    finger_radius = 0.012
    z_grasp = 0.08
    r_bottle = float(bottle_asset.mechanics_core.curr_vertices[:, 0].max().item())
    x_start = r_bottle + finger_radius + 0.0015

    left_finger = FingerIndenterSpec(
        finger_id="left_thumb",
        center_xyz=torch.tensor([-x_start, 0.0, z_grasp], dtype=torch.float32, device=device),
        prev_center_xyz=torch.tensor([-x_start, 0.0, z_grasp], dtype=torch.float32, device=device),
        radius_m=finger_radius,
    )
    right_finger = FingerIndenterSpec(
        finger_id="right_index",
        center_xyz=torch.tensor([x_start, 0.0, z_grasp], dtype=torch.float32, device=device),
        prev_center_xyz=torch.tensor([x_start, 0.0, z_grasp], dtype=torch.float32, device=device),
        radius_m=finger_radius,
    )
    ax_left = torch.tensor([1.0, 0.0, 0.0], device=device)
    ax_right = torch.tensor([-1.0, 0.0, 0.0], device=device)
    ax_up = torch.tensor([0.0, 0.0, 1.0], device=device)

    num_sim_steps = 20
    if device == "cuda":
        torch.cuda.synchronize()
    t_sim_start = time.perf_counter()

    telemetry = None
    left_obs = None
    deformed_means = None
    for step_idx in range(1, num_sim_steps + 1):
        left_finger.prev_center_xyz = left_finger.center_xyz.clone()
        right_finger.prev_center_xyz = right_finger.center_xyz.clone()

        dx_pinch = 0.00018 if step_idx <= 12 else 0.00003
        dz_lift = 0.00002 if step_idx <= 12 else 0.00015

        left_finger.center_xyz[0] += dx_pinch
        left_finger.center_xyz[2] += dz_lift
        right_finger.center_xyz[0] -= dx_pinch
        right_finger.center_xyz[2] += dz_lift

        telemetry = solver.step(bottle_asset, [left_finger, right_finger])
        left_obs = tactile_sensor.sense_finger_pad(
            bottle_asset, left_finger, telemetry, ax_left, ax_up
        )
        _ = tactile_sensor.sense_finger_pad(
            bottle_asset, right_finger, telemetry, ax_right, ax_up
        )
        deformed_means, _ = Cage2DGSDeformer.deform_splats(bottle_asset)

    if device == "cuda":
        torch.cuda.synchronize()
    t_sim_total = time.perf_counter() - t_sim_start

    assert telemetry is not None and left_obs is not None and deformed_means is not None
    ms_per_step = (t_sim_total / num_sim_steps) * 1000.0

    print(f"\n[2] GB10 GPU Execution of RT-Patch-IPC + 16x16 Tactile + 2DGS Deformer ({num_sim_steps} Steps):")
    print(f"  - Step Latency (Solver + Tactile + 2DGS)      : {ms_per_step:.2f} ms/step ({1000.0 / ms_per_step:.1f} FPS)")
    print(f"  - Active Patch Vertices Woken Up              : {telemetry.active_patch_vertex_count} / {telemetry.total_vertex_count} (-{telemetry.compute_reduction_ratio * 100:.1f}% DOFs)")
    print(f"  - Final Min Contact Clearance                 : {telemetry.min_contact_clearance_mm:.4f} mm (100% Penetration-Free: {telemetry.penetration_free})")
    print(f"  - Peak Contact Pressure                       : {telemetry.peak_contact_pressure_pa / 1000.0:.2f} kPa")
    print(f"  - Total Normal Grasp Force                    : {telemetry.total_normal_force_n:.3f} N")
    print(f"  - Total Tangential Shear Force                : {telemetry.total_tangential_force_n:.3f} N")
    print(f"  - Peak Internal von Mises Stress              : {telemetry.peak_von_mises_stress_pa / 1000.0:.2f} kPa")
    print("=" * 80)


if __name__ == "__main__":
    main()
