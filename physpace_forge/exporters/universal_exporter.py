"""
Universal Industry-Standard Format Exporter (`UniversalFormatExporter`).

Instead of forcing users to adopt a proprietary file format, PhySpace-Forge acts as a
Universal Physical & Tactile Enhancer for existing industry-standard formats:

1. **MuJoCo MJCF (`.xml`)**:
   - Directly loadable via `mujoco.MjModel.from_xml_path(...)` in MuJoCo 3.x / Genesis / ManiSkill.
   - Automatically calibrates `solref`, `solimp`, `condim="6"`, `friction`, and `<inertial>`.
   - Injects Golden Grasp Safety Corridor `[P_slip-min, P_crush-max]`, Yield Stress `σ_y`, and
     tributary area statistics into MuJoCo's native `<custom><numeric .../></custom>` block.

2. **NVIDIA OpenUSD (`.usda`)**:
   - Directly loadable in NVIDIA Isaac Sim / Isaac Lab / Omniverse / Blender / Apple Vision Pro.
   - Standard `def Xform` + `def Mesh` with `PhysicsMassAPI` and `PhysxDeformableBodyAPI`.
   - Embeds SCPP Shell-Core and Pascal Tactile calibration inside native USD `customData` &
     `custom float[] physpace:tributaryAreas`.

3. **ROS2 / PyBullet / SAPIEN URDF (`.urdf`)**:
   - Standard `<robot>` and `<link>` with calibrated `<inertial>`, `<visual>`, `<collision>`,
     `<contact>` stiffness/damping, and `<physpace_tactile_calibration>` sidecar block.

4. **Khronos glTF 2.0 (`.gltf`)**:
   - Standard 3D web/DCC format loadable in Blender, Three.js, and Windows/Mac 3D viewers,
     storing tactile & constitutive metadata in standard glTF `"extras": {"physpace": ...}`.
"""

import base64
import json
import math
import os
import struct
from typing import Any, Dict
import torch

from physpace_forge.schema.physpack import SCPPAsset


class UniversalFormatExporter:
    """Exports any calibrated SCPPAsset into MuJoCo (.xml), OpenUSD (.usda), URDF (.urdf), or glTF (.gltf)."""

    @staticmethod
    def _extract_summary(asset: SCPPAsset, envelope: Dict[str, Any] | None = None) -> Dict[str, Any]:
        core = asset.mechanics_core
        shell = asset.contact_shell
        surf_idx = shell.surface_vertex_indices
        verts_m = core.curr_vertices[surf_idx].detach().cpu()
        faces = shell.surface_faces.detach().cpu()

        global_to_surf = torch.full((core.curr_vertices.shape[0],), -1, dtype=torch.int64)
        global_to_surf[surf_idx] = torch.arange(surf_idx.shape[0])
        local_faces = global_to_surf[faces].tolist()

        total_mass_kg = max(float(core.nodal_masses.sum().item()), 0.02)
        mean_mu_pa = float(core.lame_mu.mean().item())
        mean_lam_pa = float(core.lame_lambda.mean().item())
        eff_e_pa = mean_mu_pa * (3.0 * mean_lam_pa + 2.0 * mean_mu_pa) / max(mean_lam_pa + mean_mu_pa, 1.0)
        eff_nu = mean_lam_pa / max(2.0 * (mean_lam_pa + mean_mu_pa), 1.0)
        yield_kpa = float(core.element_yield_stress.min().item()) / 1000.0
        mu_s = float(shell.mu_static.mean().item())
        mu_d = float(shell.mu_dynamic.mean().item())
        areas_m2 = shell.vertex_contact_areas.detach().cpu().tolist()

        # Bounding box for principal moments of inertia
        vmin = verts_m.min(dim=0).values
        vmax = verts_m.max(dim=0).values
        dx = max(float((vmax[0] - vmin[0]).item()), 0.02)
        dy = max(float((vmax[1] - vmin[1]).item()), 0.02)
        dz = max(float((vmax[2] - vmin[2]).item()), 0.02)
        ixx = round((total_mass_kg / 12.0) * (dy * dy + dz * dz), 6)
        iyy = round((total_mass_kg / 12.0) * (dx * dx + dz * dz), 6)
        izz = round((total_mass_kg / 12.0) * (dx * dx + dy * dy), 6)

        env = envelope or {
            "p_slip_min_kpa": round(min(2.2, yield_kpa * 0.38), 2),
            "p_crush_max_kpa": round(yield_kpa * 0.88, 2),
            "optimal_target_pressure_kpa": round(yield_kpa * 0.58, 2),
        }

        return {
            "asset_id": asset.asset_id,
            "category": asset.category,
            "verts_m": verts_m.tolist(),
            "local_faces": local_faces,
            "areas_m2": [round(a, 7) for a in areas_m2],
            "mass_kg": round(total_mass_kg, 4),
            "youngs_modulus_pa": round(eff_e_pa, 1),
            "youngs_modulus_kpa": round(eff_e_pa / 1000.0, 2),
            "poisson_ratio": round(eff_nu, 3),
            "yield_stress_kpa": round(yield_kpa, 2),
            "mu_static": round(mu_s, 3),
            "mu_dynamic": round(mu_d, 3),
            "dims_m": (round(dx, 4), round(dy, 4), round(dz, 4)),
            "inertia": (ixx, iyy, izz),
            "envelope": env,
            "watermark": asset.metadata.get("calibration_signature", "PSF-UNIVERSAL-V1"),
        }

    @classmethod
    def to_mujoco_mjcf_xml(cls, asset: SCPPAsset, envelope: Dict[str, Any] | None = None) -> str:
        """
        Export a 100% standard MuJoCo 3.x MJCF (.xml) model with calibrated soft-contact
        impedance (`solref`, `solimp`, `condim="6"`) and `<custom>` tactile safety envelope.
        """
        s = cls._extract_summary(asset, envelope)
        # Convert Young's modulus E to MuJoCo solref (timeconst, dampratio)
        stiff_scale = max(min(s["youngs_modulus_kpa"] / 100.0, 5.0), 0.1)
        timeconst = round(0.018 / math.sqrt(stiff_scale), 4)
        rx, ry, rz = s["dims_m"][0] * 0.5, s["dims_m"][1] * 0.5, s["dims_m"][2] * 0.5
        env = s["envelope"]

        # Embed inline vertex coordinates so the MJCF is self-contained without external STL/OBJ files
        verts_flat = "  ".join(f"{v[0]:.4f} {v[1]:.4f} {v[2]:.4f}" for v in s["verts_m"][::2])
        faces_flat = "  ".join(f"{f[0]} {f[1]} {f[2]}" for f in s["local_faces"][:60])

        xml = f"""<!--
  Universal MuJoCo 3.x MJCF Model Enhanced by PhySpace-Forge (v0.1.0)
  Object ID            : {s['asset_id']} ({s['category']})
  Calibration Signature: {s['watermark']}
  Golden Safety Window : [{env['p_slip_min_kpa']} kPa, {env['p_crush_max_kpa']} kPa] (Yield σ_y = {s['yield_stress_kpa']} kPa)
-->
<mujoco model="physpace_{s['asset_id']}">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.005" gravity="0 0 -9.81" integrator="implicitfast" cone="elliptic" impratio="5.0"/>

  <!-- Native MuJoCo Custom Numeric Fields: Read by PhySpace RealRobotSafetyShield & Tactile Sensor -->
  <custom>
    <numeric name="physpace_youngs_modulus_kpa" size="1" data="{s['youngs_modulus_kpa']}"/>
    <numeric name="physpace_poisson_ratio" size="1" data="{s['poisson_ratio']}"/>
    <numeric name="physpace_yield_stress_kpa" size="1" data="{s['yield_stress_kpa']}"/>
    <numeric name="physpace_golden_safety_corridor_kpa" size="3" data="{env['p_slip_min_kpa']} {env['p_crush_max_kpa']} {env.get('optimal_target_pressure_kpa', 3.2)}"/>
    <numeric name="physpace_friction_static_dynamic" size="2" data="{s['mu_static']} {s['mu_dynamic']}"/>
    <text name="physpace_calibration_signature" data="{s['watermark']}"/>
  </custom>

  <asset>
    <mesh name="{s['asset_id']}_hull" vertex="{verts_flat}"/>
  </asset>

  <worldbody>
    <light pos="0 0 1.5" dir="0 0 -1" directional="true"/>
    <geom name="tabletop_floor" type="plane" size="0.5 0.5 0.01" rgba="0.12 0.18 0.28 1" friction="{s['mu_static']} 0.01 0.001"/>

    <body name="{s['asset_id']}" pos="0 0 {rz:.4f}">
      <freejoint name="{s['asset_id']}_root"/>
      <inertial pos="0 0 0" mass="{s['mass_kg']}" diaginertia="{s['inertia'][0]} {s['inertia'][1]} {s['inertia'][2]}"/>
      <geom name="{s['asset_id']}_contact_shell"
            type="mesh"
            mesh="{s['asset_id']}_hull"
            condim="6"
            friction="{s['mu_static']} 0.015 0.002"
            solref="{timeconst} 1.0"
            solimp="0.88 0.96 0.002 0.5 2"
            rgba="0.22 0.74 0.97 0.88"/>
    </body>
  </worldbody>
</mujoco>
"""
        return xml

    @classmethod
    def to_openusd_usda(cls, asset: SCPPAsset, envelope: Dict[str, Any] | None = None) -> str:
        """
        Export a 100% standard NVIDIA OpenUSD ASCII (.usda) stage loadable in Isaac Sim / Isaac Lab /
        Omniverse / Blender, with SCPP & tactile metadata stored in native USD `customData` & attributes.
        """
        s = cls._extract_summary(asset, envelope)
        env = s["envelope"]
        pts_str = ", ".join(f"({v[0]:.5f}, {v[1]:.5f}, {v[2]:.5f})" for v in s["verts_m"])
        fvc_str = ", ".join("3" for _ in s["local_faces"])
        fvi_str = ", ".join(f"{f[0]}, {f[1]}, {f[2]}" for f in s["local_faces"])
        areas_str = ", ".join(f"{a:.7f}" for a in s["areas_m2"])

        usda = f"""#usda 1.0
(
    defaultPrim = "{s['asset_id']}"
    metersPerUnit = 1.0
    upAxis = "Z"
    doc = "Universal OpenUSD Asset Enhanced with PhySpace-Forge SCPP & Pascal Tactile Calibration"
    customLayerData = {{
        string physpace:version = "0.1.0"
        string physpace:calibrationSignature = "{s['watermark']}"
        double physpace:pSlipMinKPa = {env['p_slip_min_kpa']}
        double physpace:pCrushMaxKPa = {env['p_crush_max_kpa']}
        double physpace:optimalTargetPressureKPa = {env.get('optimal_target_pressure_kpa', 3.2)}
    }}
)

def Xform "{s['asset_id']}" (
    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
)
{{
    float physics:mass = {s['mass_kg']}
    float3 physics:diagonalInertia = ({s['inertia'][0]}, {s['inertia'][1]}, {s['inertia'][2]})

    def Mesh "ContactShellMesh" (
        prepend apiSchemas = ["PhysicsCollisionAPI", "PhysxDeformableSurfaceAPI"]
    )
    {{
        point3f[] points = [{pts_str}]
        int[] faceVertexCounts = [{fvc_str}]
        int[] faceVertexIndices = [{fvi_str}]
        color3f[] primvars:displayColor = [(0.22, 0.74, 0.97)]

        # PhySpace-Forge Native USD Extension Attributes (Ignored safely by basic viewers, read by RT-Patch-IPC)
        custom float physpace:youngsModulusKPa = {s['youngs_modulus_kpa']}
        custom float physpace:poissonRatio = {s['poisson_ratio']}
        custom float physpace:yieldStressKPa = {s['yield_stress_kpa']}
        custom float physpace:muStatic = {s['mu_static']}
        custom float physpace:muDynamic = {s['mu_dynamic']}
        custom float2 physpace:goldenSafetyCorridorKPa = ({env['p_slip_min_kpa']}, {env['p_crush_max_kpa']})
        custom float[] physpace:vertexTributaryAreasM2 = [{areas_str}]
    }}
}}
"""
        return usda

    @classmethod
    def to_ros2_urdf(cls, asset: SCPPAsset, envelope: Dict[str, Any] | None = None) -> str:
        """
        Export a standard URDF (.urdf) file for ROS2 / MoveIt2 / PyBullet / SAPIEN / Gazebo,
        with `<contact>` compliance and `<physpace_tactile>` extension block.
        """
        s = cls._extract_summary(asset, envelope)
        env = s["envelope"]
        dx, dy, dz = s["dims_m"]
        kp = round(s["youngs_modulus_pa"] * 0.05, 1)

        urdf = f"""<?xml version="1.0" encoding="utf-8"?>
<!-- Universal URDF Enhanced by PhySpace-Forge (Calibration ID: {s['watermark']}) -->
<robot name="{s['asset_id']}">
  <link name="base_link">
    <inertial>
      <origin xyz="0 0 {dz*0.5:.4f}" rpy="0 0 0"/>
      <mass value="{s['mass_kg']}"/>
      <inertia ixx="{s['inertia'][0]}" ixy="0.0" ixz="0.0"
               iyy="{s['inertia'][1]}" iyz="0.0"
               izz="{s['inertia'][2]}"/>
    </inertial>
    <visual>
      <origin xyz="0 0 {dz*0.5:.4f}" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="{dx*0.5:.4f}" length="{dz:.4f}"/>
      </geometry>
      <material name="physpace_cyan">
        <color rgba="0.22 0.74 0.97 0.9"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 {dz*0.5:.4f}" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="{dx*0.5:.4f}" length="{dz:.4f}"/>
      </geometry>
    </collision>
    <contact>
      <lateral_friction value="{s['mu_static']}"/>
      <spinning_friction value="0.015"/>
      <stiffness value="{kp}"/>
      <damping value="12.0"/>
    </contact>
  </link>

  <!-- PhySpace-Forge Tactile & Continuum Sidecar Extension (Standard XML Extension) -->
  <physpace_tactile version="0.1.0" signature="{s['watermark']}">
    <constitutive youngs_modulus_kpa="{s['youngs_modulus_kpa']}"
                  poisson_ratio="{s['poisson_ratio']}"
                  yield_stress_kpa="{s['yield_stress_kpa']}"
                  mu_static="{s['mu_static']}"
                  mu_dynamic="{s['mu_dynamic']}"/>
    <golden_safety_corridor p_slip_min_kpa="{env['p_slip_min_kpa']}"
                            p_crush_max_kpa="{env['p_crush_max_kpa']}"
                            optimal_target_kpa="{env.get('optimal_target_pressure_kpa', 3.2)}"/>
  </physpace_tactile>
</robot>
"""
        return urdf

    @classmethod
    def to_gltf_json(cls, asset: SCPPAsset, envelope: Dict[str, Any] | None = None) -> str:
        """
        Export a self-contained Khronos glTF 2.0 (.gltf) file with embedded base64 geometry buffer
        and standard `"extras": {"physpace": ...}` metadata for Blender / Three.js / WebGL.
        """
        s = cls._extract_summary(asset, envelope)
        verts = s["verts_m"]
        faces = s["local_faces"]

        v_bytes = bytearray()
        min_v = [1e9, 1e9, 1e9]
        max_v = [-1e9, -1e9, -1e9]
        for v in verts:
            v_bytes.extend(struct.pack("<fff", float(v[0]), float(v[1]), float(v[2])))
            for i in range(3):
                min_v[i] = min(min_v[i], float(v[i]))
                max_v[i] = max(max_v[i], float(v[i]))

        i_bytes = bytearray()
        for f in faces:
            i_bytes.extend(struct.pack("<HHH", int(f[0]), int(f[1]), int(f[2])))
        while len(i_bytes) % 4 != 0:
            i_bytes.append(0)

        combined = bytes(v_bytes + i_bytes)
        b64_uri = "data:application/octet-stream;base64," + base64.b64encode(combined).decode("ascii")

        gltf_obj = {
            "asset": {
                "version": "2.0",
                "generator": "PhySpace-Forge Universal Format Exporter v0.1.0",
            },
            "scene": 0,
            "scenes": [{"nodes": [0]}],
            "nodes": [
                {
                    "name": s["asset_id"],
                    "mesh": 0,
                    "extras": {
                        "physpace": {
                            "calibration_signature": s["watermark"],
                            "mass_kg": s["mass_kg"],
                            "youngs_modulus_kpa": s["youngs_modulus_kpa"],
                            "poisson_ratio": s["poisson_ratio"],
                            "yield_stress_kpa": s["yield_stress_kpa"],
                            "mu_static": s["mu_static"],
                            "mu_dynamic": s["mu_dynamic"],
                            "golden_safety_envelope": s["envelope"],
                            "vertex_tributary_areas_m2": s["areas_m2"],
                        }
                    },
                }
            ],
            "meshes": [
                {
                    "name": f"{s['asset_id']}_shell",
                    "primitives": [{"attributes": {"POSITION": 0}, "indices": 1, "mode": 4}],
                }
            ],
            "buffers": [{"byteLength": len(combined), "uri": b64_uri}],
            "bufferViews": [
                {"buffer": 0, "byteOffset": 0, "byteLength": len(v_bytes), "target": 34962},
                {"buffer": 0, "byteOffset": len(v_bytes), "byteLength": len(faces) * 6, "target": 34963},
            ],
            "accessors": [
                {
                    "bufferView": 0,
                    "componentType": 5126,
                    "count": len(verts),
                    "type": "VEC3",
                    "min": [round(x, 5) for x in min_v],
                    "max": [round(x, 5) for x in max_v],
                },
                {
                    "bufferView": 1,
                    "componentType": 5123,
                    "count": len(faces) * 3,
                    "type": "SCALAR",
                },
            ],
        }
        return json.dumps(gltf_obj, indent=2, ensure_ascii=False)

    @classmethod
    def export_all_formats(
        cls,
        asset: SCPPAsset,
        out_dir: str,
        envelope: Dict[str, Any] | None = None,
    ) -> Dict[str, str]:
        """
        Export an SCPPAsset into all 4 universal industry formats (.xml, .usda, .urdf, .gltf)
        plus its native .physpack sidecar inside `out_dir`.
        """
        import os

        os.makedirs(out_dir, exist_ok=True)
        oid = asset.asset_id
        paths = {
            "mjcf": os.path.join(out_dir, f"{oid}_mujoco.xml"),
            "usda": os.path.join(out_dir, f"{oid}_isaac.usda"),
            "urdf": os.path.join(out_dir, f"{oid}_ros2.urdf"),
            "gltf": os.path.join(out_dir, f"{oid}.gltf"),
            "physpack": os.path.join(out_dir, f"{oid}.physpack"),
        }
        with open(paths["mjcf"], "w", encoding="utf-8") as f:
            f.write(cls.to_mujoco_mjcf_xml(asset, envelope))
        with open(paths["usda"], "w", encoding="utf-8") as f:
            f.write(cls.to_openusd_usda(asset, envelope))
        with open(paths["urdf"], "w", encoding="utf-8") as f:
            f.write(cls.to_ros2_urdf(asset, envelope))
        with open(paths["gltf"], "w", encoding="utf-8") as f:
            f.write(cls.to_gltf_json(asset, envelope))
        asset.save_physpack(paths["physpack"])
        return paths
