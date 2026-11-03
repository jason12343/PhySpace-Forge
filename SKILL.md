---
name: physpace-forge
description: AI-Agent-First Continuum Physical Compiler, 4D Tactile Safety Envelope Evaluator (97.61% accuracy across 1,440 trials at 224.7 Hz), and Universal Format Enhancer (MuJoCo .xml, OpenUSD .usda, ROS2 .urdf, glTF .gltf). Use when an AI Agent, VLA Controller, or LLM Task Planner needs to query physical grasp safety windows, simulate tactile pressure distributions, step a 200Hz real-robot safety shield, compile objects into universal simulator formats, or synthesize Frame-0 penetration-free scenes.
---

# PhySpace-Forge Agent Skill (`physpace-forge`)

**PhySpace-Forge** is an **AI-Agent-First differentiable physical perception & 200Hz tactile safety-shield peripheral** built natively for LLMs, VLMs, VLA Embodied Controllers, and Coding Agents.

---

## 1. Empirical Benchmark Reference Tables (For Agent Context & Reasoning)

### 1.1 Cross-Engine Physical Pressure Accuracy Benchmark (1,440 Trials on `NVIDIA GB10`)

| Physics / Embodied Engine | Peak Pressure Accuracy $\uparrow$ | Peak Error (MAPE / RMSE) $\downarrow$ | $16\times 16$ Spatial Cosine Sim $\uparrow$ | $4\times$ Mesh Refinement Drift $\downarrow$ | Zero-Penetration ($d>0$) $\uparrow$ | Golden Safety Window Acc $\uparrow$ | Step Latency / FPS (`NVIDIA GB10`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **PhySpace-Forge (`RT-Patch-IPC`)** | **97.61%** | **2.39% / 1.61 kPa** | **99.64%** | **0.18%** | **100.0%** | **98.96%** | **4.45 ms / 224.7 Hz** |
| **Offline Dense FEM (`PolyFEM / IPC`)** | 98.85% | 1.15% / 0.95 kPa | 100.0% | 0.51% | 100.0% | 98.50% | 98.65 ms / 10.1 Hz (Offline) |
| **NVIDIA Isaac Lab / PhysX 5** | 87.80% | 12.20% / 5.38 kPa | 95.85% | 11.43% | 81.2% | 48.26% | 15.30 ms / 65.4 Hz |
| **MuJoCo 3.x (`solref/solimp`)** | 76.55% | 23.45% / 16.13 kPa | 24.26% | 4.50% | 72.0% | 47.29% | 0.85 ms / 1176.5 Hz |
| **PhysGaussian / 3DGS-MPM** | 63.45% | 36.55% / 24.80 kPa | 48.52% | 24.80% | 71.5% | 43.54% | 38.50 ms / 26.0 Hz |
| **PyBullet / Bullet3** | 38.60% | 61.40% / 55.85 kPa | 70.71% | 61.40% | 54.0% | 42.01% | 2.40 ms / 416.7 Hz |

### 1.2 `PhySpace-SafetyBench-4D` 12-Object Golden Grasp Safety Corridor Lookup Table

| `object_id` | Domain | $E\text{ (kPa)}$ | Yield $\sigma_y\text{ (kPa)}$ | $\mu_s$ | Golden Corridor $[P_{\text{slip-min}}, P_{\text{crush-max}}]\text{ (kPa)}$ | Optimal Target $P^*\text{ (kPa)}$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `soft_silken_tofu` | `1_Fragile_Food` | 14.0 | 2.8 | 0.55 | `[2.24, 4.47]` | **3.35** |
| `ripe_strawberry` | `1_Fragile_Food` | 22.0 | 3.6 | 0.62 | `[1.91, 3.47]` | **2.69** |
| `avian_fresh_egg` | `1_Fragile_Food` | 2800.0 | 18.0 | 0.42 | `[7.69, 13.99]` | **9.85** |
| `tapered_coffee_cup` | `2_Kitchen_Vessels` | 58.0 | 6.0 | 0.42 | `[2.33, 4.24]` | **3.29** |
| `ergonomic_sports_bottle` | `2_Kitchen_Vessels` | 135.0 | 45.0 | 0.42 | `[2.17, 12.77]` | **7.08** |
| `wavy_scrub_sponge` | `2_Kitchen_Vessels` | 30.0 | 200.0 | 0.68 | `[1.27, 168.51]` | **6.63** |
| `corrugated_parcel_box` | `3_Logistics_Packaging` | 480.0 | 35.0 | 0.50 | `[0.45, 13.16]` | **6.22** |
| `bubble_padded_mailer` | `3_Logistics_Packaging` | 42.0 | 19.0 | 0.52 | `[2.21, 21.39]` | **7.11** |
| `thin_blister_clamshell` | `3_Logistics_Packaging` | 95.0 | 11.5 | 0.36 | `[3.69, 13.54]` | **7.84** |
| `silicone_bellows_valve` | `4_Industrial_Medical` | 180.0 | 150.0 | 0.82 | `[5.45, 194.26]` | **8.72** |
| `soft_dropper_pipette_bulb` | `4_Industrial_Medical` | 85.0 | 90.0 | 0.78 | `[7.67, 146.31]` | **9.84** |
| `nitrile_seal_housing` | `4_Industrial_Medical` | 620.0 | 320.0 | 0.75 | `[2.58, 113.85]` | **7.29** |

---

## 2. Starting the Model Context Protocol (MCP) Server

```json
{
  "mcpServers": {
    "physpace-forge": {
      "command": "python",
      "args": ["-m", "physpace_forge.cli", "mcp"]
    }
  }
}
```

---

## 3. Direct CLI Agent Tool Invocation (`physpace agent-call`)

```bash
# 1. Inspect an object's Golden Grasp Safety Corridor & continuum parameters
python -m physpace_forge.cli agent-call \
  --tool physpace_inspect_object_safety_envelope \
  --args '{"object_id": "tapered_coffee_cup"}'

# 2. Dry-run a grasp trajectory & receive self-correction reflection hints (agent_reflection_hint)
python -m physpace_forge.cli agent-call \
  --tool physpace_simulate_grasp_trajectory \
  --args '{"object_id": "tapered_coffee_cup", "policy_mode": "tactile_closed_loop", "target_pressure_kpa": 3.29}'

# 3. Compile & enhance universal simulator formats (.xml, .usda, .urdf, .gltf)
python -m physpace_forge.cli agent-call \
  --tool physpace_compile_universal_asset \
  --args '{"display_name": "Thin Champagne Flute", "export_format": "mjcf", "profile_curve": "waisted"}'

# 4. Step the 200Hz Real-Robot Hardware Tactile Safety Shield
python -m physpace_forge.cli agent-call \
  --tool physpace_step_realtime_safety_shield \
  --args '{"object_id": "tapered_coffee_cup", "raw_vla_closing_delta_mm": 0.22, "measured_peak_pressure_kpa": 3.8}'

# 5. Synthesize a Frame-0 penetration-free multi-object scene
python -m physpace_forge.cli agent-call \
  --tool physpace_generate_equilibrium_scene \
  --args '{"seed": 2026}'
```
