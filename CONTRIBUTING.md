# Contributing to PhySpace-Forge & PhySpace-CalibDB

We welcome contributions from the global Embodied AI, Dexterous Manipulation, and Computational Mechanics communities!

## 1. Contributing New Calibrated `.physpack` Assets (1-Touch Probe SOP)

To contribute a new real-world object to `PhySpace-CalibDB`:

1. **Compile Initial Geometry (`.obj` $\to$ `.physpack`)**:
   ```bash
   python -m physpace_forge.cli compile --name "Your Object Name" --profile waisted --e-kpa 55.0 --yield-kpa 6.0 --output releases/custom_object.physpack
   ```
2. **Real-to-Sim 1-Touch Physical Probe (Optional for Verified Badge)**:
   - Indent the real physical object by $\delta = 1.5\text{ mm}$ using a calibrated force gauge or tactile fingertip (`Paxini PX-6AX`, `GelSight Mini`, `Xela uSkin`, or `Instron`).
   - Record the measured peak pressure $P_{\text{real}}$ ($\text{kPa}$) and update `--e-kpa` so the simulated pressure matches within $\pm 3\%$.
3. **Submit a Pull Request**:
   - Include your `.obj` mesh (or parametric spec in `physpace_forge/compiler/domain_catalog.py`) and the generated `.physpack` summary.

## 2. Adding New Commercial Tactile Sensor Profiles

To add a new tactile sensor specification, add a `CommercialTactileSpec` entry to `COMMERCIAL_TACTILE_SENSORS` in [`physpace_forge/bridge/hardware_taxel_mapper.py`](physpace_forge/bridge/hardware_taxel_mapper.py) specifying:
- `grid_rows`, `grid_cols`
- `pad_width_mm`, `pad_height_mm`
- `min_detectable_pa`, `saturation_pa`, and `quantization_bits`
