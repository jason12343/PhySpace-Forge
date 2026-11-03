"""
Publication-Grade 6-Panel Hero Figure Renderer for PhySpace-Forge Dual-Engine Mission.

Visualizes the complete end-to-end execution across:
- `Qwen3_8_176B_VLM` (176B `Qwen3.8-176B-VLM` over 200GbE `127.0.0.1:11434`)
- `NVIDIA_GB10_128GB` (Scheduler `127.0.0.1:9090` + `NVIDIA GB10` Unified Memory GPU)
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def main() -> None:
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    report_path = os.path.join(root_dir, "benchmarks", "hero_execution_report.json")
    out_img_workspace = os.path.join(root_dir, "benchmarks", "hero_benchmarks_showcase.png")

    with open(report_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    plt.style.use("dark_background")
    fig = plt.figure(figsize=(18, 12.5), dpi=160)
    fig.patch.set_facecolor("#0d1117")

    fig.suptitle(
        "PhySpace-Forge Benchmarks: VLM Constitutive Reasoning (Qwen3.8-176B) + 200Hz RT-Patch-IPC Solver (NVIDIA GB10 128GB)",
        fontsize=15.5,
        fontweight="bold",
        color="#f0f6fc",
        y=0.975,
    )

    # ------------------------------------------------------------------------
    # Panel A (Row 1, Col 1): Benchmark 1 - Fragile Paper Cup Crush vs. Safe Grasp
    # ------------------------------------------------------------------------
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.set_facecolor("#161b22")
    naive_hist = data["hero_cup_comparison"]["naive_position"]["history"]
    tact_hist = data["hero_cup_comparison"]["tactile_closed_loop"]["history"]
    steps_ms = np.arange(1, len(naive_hist["von_mises_kpa"]) + 1) * 5.0

    ax1.plot(
        steps_ms,
        naive_hist["von_mises_kpa"],
        color="#ff5555",
        linewidth=2.5,
        marker="o",
        markersize=3.5,
        label=f"Naive Position: von Mises ({max(naive_hist['von_mises_kpa']):.1f} kPa CRUSHED)",
    )
    ax1.plot(
        steps_ms,
        tact_hist["von_mises_kpa"],
        color="#3fb950",
        linewidth=2.6,
        marker="s",
        markersize=3.5,
        label=f"PhySpace Tactile: von Mises ({max(tact_hist['von_mises_kpa']):.2f} kPa SAFE)",
    )
    ax1.plot(
        steps_ms,
        tact_hist["peak_pressure_kpa"],
        color="#58a6ff",
        linestyle="-.",
        linewidth=1.9,
        label=f"PhySpace Tactile: Pressure ({max(tact_hist['peak_pressure_kpa']):.2f} kPa)",
    )
    ax1.axhline(6.0, color="#f0883e", linestyle=":", linewidth=2.2, label="Yield Limit (σ_y = 6.0 kPa)")
    ax1.fill_between(steps_ms, 6.0, 26.0, color="#ff5555", alpha=0.10)
    ax1.set_title("[A] Benchmark 1: Fragile Paper Cup Force Control", fontsize=11, fontweight="bold", color="#58a6ff")
    ax1.set_xlabel("Time (ms) @ 200 Hz", fontsize=9.5)
    ax1.set_ylabel("Stress / Pressure (kPa)", fontsize=9.5)
    ax1.set_ylim(0, max(max(naive_hist["von_mises_kpa"]) * 1.1, 15.0))
    ax1.grid(True, linestyle="--", alpha=0.25)
    ax1.legend(loc="upper left", fontsize=7.8, framealpha=0.85)

    # ------------------------------------------------------------------------
    # Panel B (Row 1, Col 2): 16x16 Fingertip Pascal Pressure Map + Shear Quiver
    # ------------------------------------------------------------------------
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.set_facecolor("#161b22")
    bottle_ep = data["batch_episodes"]["ergonomic_sports_bottle"]
    p_grid = np.array(bottle_ep["final_left_pressure_grid_kpa"])
    tau_grid = np.array(bottle_ep["final_left_shear_grid_kpa"])

    extent_mm = [-12.0, 12.0, -12.0, 12.0]
    im = ax2.imshow(p_grid, cmap="inferno", origin="lower", extent=extent_mm, interpolation="bicubic")
    cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label("Normal Pressure P(u,v) [kPa]", fontsize=9)

    H, W = p_grid.shape
    us = np.linspace(-11.0, 11.0, W)
    vs = np.linspace(-11.0, 11.0, H)
    U, V = np.meshgrid(us, vs)
    tau_u = tau_grid[:, :, 0]
    tau_v = tau_grid[:, :, 1]
    mask = p_grid > 0.15
    if np.any(mask):
        ax2.quiver(
            U[mask],
            V[mask],
            tau_u[mask],
            tau_v[mask],
            color="#58a6ff",
            scale=4.2,
            width=0.0055,
            headwidth=4.0,
            label="Shear Stress τ(u,v)",
        )
        ax2.legend(loc="upper right", fontsize=8.0, framealpha=0.85)

    ax2.set_title("[B] 16x16 Fingertip Pressure P(u,v) & Shear τ(u,v)", fontsize=11, fontweight="bold", color="#58a6ff")
    ax2.set_xlabel("Fingertip Pad u (mm)", fontsize=9.5)
    ax2.set_ylabel("Fingertip Pad v (mm)", fontsize=9.5)

    # ------------------------------------------------------------------------
    # Panel C (Row 1, Col 3): 3D Deformed SCPP Surface Mesh & Contact Hotspots
    # ------------------------------------------------------------------------
    ax3 = fig.add_subplot(2, 3, 3, projection="3d")
    ax3.set_facecolor("#161b22")
    verts = np.array(bottle_ep["deformed_vertices"]) * 100.0
    faces = np.array(bottle_ep["surface_faces"])
    surf_idx = np.array(bottle_ep["surface_vertex_indices"])
    surf_p = np.array(bottle_ep["surface_pressure_kpa"])

    vert_p_full = np.zeros(len(verts))
    vert_p_full[surf_idx] = surf_p
    face_p = vert_p_full[faces].mean(axis=1)
    norm_p = np.clip(face_p / max(face_p.max(), 0.5), 0.0, 1.0)
    face_colors = plt.cm.plasma(0.15 + 0.85 * norm_p)
    face_colors[:, 3] = 0.80

    poly3d = Poly3DCollection(verts[faces], facecolors=face_colors, edgecolors="#30363d", linewidths=0.22)
    ax3.add_collection3d(poly3d)
    ax3.set_xlim(-5.0, 5.0)
    ax3.set_ylim(-5.0, 5.0)
    ax3.set_zlim(0.0, 17.0)
    ax3.view_init(elev=20, azim=-48)
    ax3.set_title("[C] 3D SCPP Waisted Bottle Active Contact Patch", fontsize=11, fontweight="bold", color="#58a6ff")
    ax3.set_xlabel("X (cm)", fontsize=8.5)
    ax3.set_ylabel("Y (cm)", fontsize=8.5)
    ax3.set_zlabel("Z (cm)", fontsize=8.5)

    # ------------------------------------------------------------------------
    # Panel D (Row 2, Col 1): Benchmark 2 - Cluttered Obstacle Push & Extract
    # ------------------------------------------------------------------------
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.set_facecolor("#161b22")
    if "benchmark2_cluttered_push_extract" in data:
        b2 = data["benchmark2_cluttered_push_extract"]["phase_a_obstacle_push"]
        p_push = b2["pressure_history_kpa"]
        f_push = b2["force_history_n"]
        t_push = np.arange(1, len(p_push) + 1) * 5.0
        ax4.plot(t_push, p_push, color="#d2a8ff", linewidth=2.5, marker="o", label="Obstacle Sponge Push Pressure (kPa)")
        ax4.plot(t_push, f_push, color="#79c0ff", linewidth=2.2, linestyle="--", marker="^", label="Lateral Clearing Force (N)")
        ax4.axhline(3.5, color="#f0883e", linestyle=":", linewidth=1.8, label="Gentle Push Safety Cap (3.5 kPa)")
    ax4.set_title("[D] Benchmark 2: Cluttered Obstacle Force-Limited Push", fontsize=11, fontweight="bold", color="#58a6ff")
    ax4.set_xlabel("Push Phase Time (ms) @ 200 Hz", fontsize=9.5)
    ax4.set_ylabel("Pressure (kPa) / Force (N)", fontsize=9.5)
    ax4.grid(True, linestyle="--", alpha=0.25)
    ax4.legend(loc="upper left", fontsize=8.0, framealpha=0.85)

    # ------------------------------------------------------------------------
    # Panel E (Row 2, Col 2): Benchmark 3 - Cap-Twisting Torsional Torque & Slip
    # ------------------------------------------------------------------------
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.set_facecolor("#161b22")
    if "benchmark3_cap_twisting_torque" in data:
        b3 = data["benchmark3_cap_twisting_torque"]
        t_tw = np.arange(1, len(b3["torque_mNm_history"]) + 1) * 5.0
        ax5.plot(
            t_tw,
            b3["torque_mNm_history"],
            color="#ffa657",
            linewidth=2.6,
            marker="d",
            markersize=4,
            label=f"Torsional Friction Torque τ_z (Peak {b3['peak_torsional_torque_mNm']:.1f} mN·m)",
        )
        ax5_twin = ax5.twinx()
        ax5_twin.plot(
            t_tw,
            b3["slip_index_history"],
            color="#ff7b72",
            linewidth=2.0,
            linestyle="--",
            label="Rotational Slip Index s_k in [0, 1]",
        )
        ax5_twin.set_ylabel("Incipient Slip Index s_k", color="#ff7b72", fontsize=9.5)
        ax5_twin.set_ylim(0.0, 1.15)
        ax5_twin.legend(loc="lower right", fontsize=8.0, framealpha=0.85)
    ax5.set_title("[E] Benchmark 3: Cap-Twisting Torque & Stick-to-Slip", fontsize=11, fontweight="bold", color="#58a6ff")
    ax5.set_xlabel("Twist Trajectory Time (ms) @ 200 Hz", fontsize=9.5)
    ax5.set_ylabel("Torsional Torque τ_z (mN·m)", color="#ffa657", fontsize=9.5)
    ax5.grid(True, linestyle="--", alpha=0.25)
    ax5.legend(loc="upper left", fontsize=8.0, framealpha=0.85)

    # ------------------------------------------------------------------------
    # Panel F (Row 2, Col 3): Qwen3.8-176B VLM Multi-Material Spectrum
    # ------------------------------------------------------------------------
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.set_facecolor("#161b22")
    obj_keys = ["wavy_scrub_sponge", "tapered_coffee_cup", "ergonomic_sports_bottle", "avian_fresh_egg"]
    labels = ["Sponge\n(Qwen3.8)", "Paper Cup\n(Qwen3.8)", "PET Bottle\n(Qwen3.8)", "Fresh Egg\n(Qwen3.8)"]
    peak_p_vals = [data["batch_episodes"][k]["peak_pressure_kpa"] for k in obj_keys]
    vm_vals = [data["batch_episodes"][k]["peak_von_mises_kpa"] for k in obj_keys]
    yield_vals = [data["batch_episodes"][k]["yield_threshold_kpa"] for k in obj_keys]

    x_pos = np.arange(len(obj_keys))
    width = 0.25
    ax6.bar(x_pos - width, peak_p_vals, width, label="Peak Contact Pressure (kPa)", color="#58a6ff")
    ax6.bar(x_pos, vm_vals, width, label="Peak von Mises Stress (kPa)", color="#3fb950")
    ax6.bar(x_pos + width, yield_vals, width, label="Material Yield Limit σ_y (kPa)", color="#f0883e", alpha=0.85)
    ax6.set_yscale("log")
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels(labels, fontsize=9)
    ax6.set_ylabel("Stress / Pressure [kPa, Log Scale]", fontsize=9.5)
    ax6.set_title("[F] Qwen3.8-176B Constitutive Spectrum (100% Intact)", fontsize=11, fontweight="bold", color="#58a6ff")
    ax6.grid(True, which="both", linestyle="--", alpha=0.25)
    ax6.legend(loc="upper left", fontsize=7.8, framealpha=0.85)

    plt.tight_layout(rect=[0.0, 0.01, 1.0, 0.945])
    os.makedirs(os.path.dirname(out_img_workspace), exist_ok=True)
    fig.savefig(out_img_workspace, dpi=160)
    plt.close(fig)
    print(f"Saved 6-Panel Dual-Engine Hero Showcase Figure to:\n  1) {out_img_workspace}")


if __name__ == "__main__":
    main()
