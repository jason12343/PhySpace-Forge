"""
Publication-Grade Renderer for PhySpace-SafetyBench-4D & X-Ray Cutaway Showcase.

Visualizes the 4-Domain 12-Object Golden Grasp Safety Envelopes [P_slip_min, P_crush_max],
Shell-to-Core X-Ray Cutaway 3D geometry, and Delicate Food stress-pressure curves.
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
    report_path = os.path.join(root_dir, "benchmarks", "safetybench_4d_report.json")
    out_img_workspace = os.path.join(root_dir, "benchmarks", "safetybench_4d_showcase.png")

    with open(report_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    objects = data["objects"]
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(18, 11.5), dpi=160)
    fig.patch.set_facecolor("#0b0f17")

    fig.suptitle(
        "PhySpace-SafetyBench-4D (12 Objects Across 4 Domains): Golden Grasp Safety Corridors & Shell-Core X-Ray",
        fontsize=15.5,
        fontweight="bold",
        color="#f0f6fc",
        y=0.972,
    )

    # ------------------------------------------------------------------------
    # Panel A (Top-Left): 12-Object Golden Grasp Safety Envelope Range Chart
    # ------------------------------------------------------------------------
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.set_facecolor("#131a26")

    obj_ids = list(objects.keys())[::-1]
    y_idx = np.arange(len(obj_ids))
    p_mins = [objects[k]["golden_safety_envelope"]["p_slip_min_kpa"] for k in obj_ids]
    p_maxs = [objects[k]["golden_safety_envelope"]["p_crush_max_kpa"] for k in obj_ids]
    p_opts = [objects[k]["golden_safety_envelope"]["optimal_target_pressure_kpa"] for k in obj_ids]
    labels = [f"{objects[k]['display_name'].split(' (')[0]} ({objects[k]['domain'][2:]})" for k in obj_ids]

    for i, (pmin, pmax, popt) in enumerate(zip(p_mins, p_maxs, p_opts)):
        # Slip Risk Zone (0.1 -> pmin)
        ax1.barh(i, max(pmin - 0.1, 0.05), left=0.1, height=0.58, color="#d29922", alpha=0.45, label="Slip / Drop Risk (< P_min)" if i == 0 else "")
        # Golden Safe Grasp Corridor (pmin -> pmax)
        ax1.barh(i, max(pmax - pmin, 0.2), left=pmin, height=0.58, color="#3fb950", alpha=0.82, label="Golden Safe Grasp Corridor [P_min, P_max]" if i == 0 else "")
        # Crush / Yield Zone (pmax -> 350 kPa)
        ax1.barh(i, max(350.0 - pmax, 10.0), left=pmax, height=0.58, color="#f85149", alpha=0.35, label="Irreversible Crush / Yield Zone (> P_max)" if i == 0 else "")
        ax1.plot(popt, i, marker="D", color="#58a6ff", markersize=6.5, label="Optimal Target Pressure P*" if i == 0 else "")

    ax1.set_xscale("log")
    ax1.set_xlim(0.25, 320.0)
    ax1.set_yticks(y_idx)
    ax1.set_yticklabels(labels, fontsize=8.8)
    ax1.set_xlabel("Fingertip Peak Contact Pressure P(u,v) [kPa, Log Scale]", fontsize=10)
    ax1.set_title(
        "[A] 4-Domain 12-Object Golden Grasp Safety Corridors [P_slip_min <--> P_crush_max]",
        fontsize=11.5,
        fontweight="bold",
        color="#58a6ff",
        pad=10,
    )
    ax1.grid(True, which="both", linestyle="--", alpha=0.22)
    ax1.legend(loc="lower right", fontsize=8.0, framealpha=0.88)

    # ------------------------------------------------------------------------
    # Panel B (Top-Right): Ultra-Delicate Food Domain Pinch Curves (Tofu, Berry, Cup)
    # ------------------------------------------------------------------------
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.set_facecolor("#131a26")
    delicate_items = [
        ("soft_silken_tofu", "#ff7b72", "o", "Silken Tofu (σ_y = 2.8 kPa)"),
        ("ripe_strawberry", "#ffa657", "s", "Ripe Strawberry (σ_y = 3.6 kPa)"),
        ("tapered_coffee_cup", "#58a6ff", "^", "Paper Coffee Cup (σ_y = 6.0 kPa)"),
        ("thin_blister_clamshell", "#bc8cff", "d", "Blister Pack (σ_y = 11.5 kPa)"),
    ]
    for oid, col, mk, lbl in delicate_items:
        sc = objects[oid]["sweep_curves"]
        yield_lim = objects[oid]["constitutive_summary"]["yield_stress_kpa"]
        ax2.plot(
            sc["indent_mm"],
            sc["von_mises_kpa"],
            color=col,
            linewidth=2.4,
            marker=mk,
            markersize=4,
            label=f"{lbl} von Mises Stress",
        )
        ax2.axhline(yield_lim, color=col, linestyle=":", linewidth=1.4, alpha=0.75)

    ax2.set_title(
        "[B] Delicate Food & Packaging: Internal von Mises Stress vs. Yield Thresholds",
        fontsize=11.5,
        fontweight="bold",
        color="#58a6ff",
        pad=10,
    )
    ax2.set_xlabel("Fingertip Pinch Indentation (mm)", fontsize=10)
    ax2.set_ylabel("Peak Internal von Mises Stress σ_vm (kPa)", fontsize=10)
    ax2.set_ylim(0, 16.0)
    ax2.grid(True, linestyle="--", alpha=0.25)
    ax2.legend(loc="upper left", fontsize=8.5, framealpha=0.88)

    # ------------------------------------------------------------------------
    # Panel C (Bottom-Left): 3D Shell-to-Core X-Ray Cutaway Visualization
    # ------------------------------------------------------------------------
    ax3 = fig.add_subplot(2, 2, 3, projection="3d")
    ax3.set_facecolor("#131a26")
    bellows_obj = objects["silicone_bellows_valve"]
    kf = bellows_obj["interactive_keyframes"][2]
    verts = np.array(kf["surface_vertices_cm"])
    faces = np.array(bellows_obj["surface_faces_local"])
    press = np.array(kf["surface_pressure_kpa"])
    core_pts = np.array(bellows_obj["xray_core_vertices_cm"])

    # Filter cutaway faces (keep back half y >= -0.3 cm so inner core is visible!)
    face_y = verts[faces, 1].mean(axis=1)
    keep_mask = face_y >= -0.35
    cut_faces = faces[keep_mask]
    face_p = press[cut_faces].mean(axis=1)
    norm_p = np.clip(face_p / max(press.max(), 0.5), 0.0, 1.0)
    f_cols = plt.cm.plasma(0.15 + 0.85 * norm_p)
    f_cols[:, 3] = 0.68

    poly = Poly3DCollection(verts[cut_faces], facecolors=f_cols, edgecolors="#30363d", linewidths=0.25)
    ax3.add_collection3d(poly)
    if len(core_pts) > 0:
        ax3.scatter(
            core_pts[:, 0],
            core_pts[:, 1],
            core_pts[:, 2],
            color="#bc8cff",
            s=16,
            alpha=0.95,
            label="Inner 12-DoF Affine Core Nodes",
        )
        ax3.legend(loc="upper right", fontsize=8.5)

    ax3.set_xlim(-4.0, 4.0)
    ax3.set_ylim(-4.0, 4.0)
    ax3.set_zlim(0.0, 10.0)
    ax3.view_init(elev=22, azim=-55)
    ax3.set_title(
        "[C] PhySpace-Studio 3D X-Ray Cutaway: Deformable Skin + Inner Affine Core",
        fontsize=11.5,
        fontweight="bold",
        color="#58a6ff",
        pad=8,
    )
    ax3.set_xlabel("X (cm)", fontsize=8.5)
    ax3.set_ylabel("Y (cm)", fontsize=8.5)
    ax3.set_zlabel("Z (cm)", fontsize=8.5)

    # ------------------------------------------------------------------------
    # Panel D (Bottom-Right): Silken Tofu Optimal Grasp 16x16 Pressure & Shear Map
    # ------------------------------------------------------------------------
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.set_facecolor("#131a26")
    tofu_kf = objects["soft_silken_tofu"]["interactive_keyframes"][1]
    p_grid = np.array(tofu_kf["pressure_grid_16x16_kpa"])
    tau_grid = np.array(tofu_kf["shear_grid_16x16_kpa"])

    extent_mm = [-12.0, 12.0, -12.0, 12.0]
    im = ax4.imshow(p_grid, cmap="viridis", origin="lower", extent=extent_mm, interpolation="bicubic")
    cbar = plt.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)
    cbar.set_label("Tofu Gentle Grasp Pressure P(u,v) [kPa]", fontsize=9.5)

    H, W = p_grid.shape
    us = np.linspace(-11.0, 11.0, W)
    vs = np.linspace(-11.0, 11.0, H)
    U, V = np.meshgrid(us, vs)
    mask = p_grid > 0.08 * max(p_grid.max(), 0.1)
    if np.any(mask):
        ax4.quiver(
            U[mask],
            V[mask],
            tau_grid[:, :, 0][mask],
            tau_grid[:, :, 1][mask],
            color="#ffa657",
            scale=2.2,
            width=0.0055,
            headwidth=4.0,
            label="Gentle Anti-Slip Shear τ(u,v)",
        )
        ax4.legend(loc="upper right", fontsize=8.5, framealpha=0.88)

    ax4.set_title(
        f"[D] Silken Tofu Golden Window Grasp (Peak P={tofu_kf['peak_pressure_kpa']} kPa < σ_y=2.8 kPa)",
        fontsize=11.5,
        fontweight="bold",
        color="#58a6ff",
        pad=10,
    )
    ax4.set_xlabel("Fingertip Pad u (mm)", fontsize=10)
    ax4.set_ylabel("Fingertip Pad v (mm)", fontsize=10)

    plt.tight_layout(rect=[0.0, 0.01, 1.0, 0.945])
    os.makedirs(os.path.dirname(out_img_workspace), exist_ok=True)
    fig.savefig(out_img_workspace, dpi=160)
    plt.close(fig)
    print(f"Saved 4-Domain SafetyBench Showcase Figure to:\n  1) {out_img_workspace}")


if __name__ == "__main__":
    main()
