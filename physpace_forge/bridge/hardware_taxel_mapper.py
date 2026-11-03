"""
Real-World Tactile Sensor Hardware Alignment Bridge (`HardwareTaxelMapper`).

Aligns PhySpace-Forge's SI continuum Pascal pressure P(u,v) and shear stress τ(u,v)
directly to commercial tactile sensor hardware specifications used on real robots:
1. `paxini_px6ax_16x16` : Paxini (帕西尼) multi-dimensional fingertip tactile array (16x16, 0~150 kPa)
2. `gelsight_mini_32x24`: GelSight Mini optical elastomer visuotactile sensor (32x24 downsampled taxels, 0~80 kPa)
3. `xela_uskin_4x6`     : Xela Robotics uSkin 3-axis magnetic hall-effect tactile pad (6x4 taxels, 0~200 kPa)
4. `inspire_hand_tip`   : Inspire Robots (因时灵巧手) RH56BFX fingertip capacitive pad (8x8 taxels, 0~120 kPa)
"""

from dataclasses import dataclass
from typing import Dict
import torch
import torch.nn.functional as F

from physpace_forge.sensors.tactile_pressure_sensor import TactileFrameObservation


@dataclass(frozen=True)
class CommercialTactileSpec:
    sensor_id: str
    vendor_name: str
    grid_rows: int
    grid_cols: int
    pad_width_mm: float
    pad_height_mm: float
    min_detectable_pa: float
    saturation_pa: float
    quantization_bits: int


COMMERCIAL_TACTILE_SENSORS: Dict[str, CommercialTactileSpec] = {
    "paxini_px6ax_16x16": CommercialTactileSpec(
        sensor_id="paxini_px6ax_16x16",
        vendor_name="Paxini Tech (帕西尼感知科技 PX-6AX 指尖多维阵列)",
        grid_rows=16,
        grid_cols=16,
        pad_width_mm=20.0,
        pad_height_mm=24.0,
        min_detectable_pa=50.0,       # 0.05 kPa minimum sensitivity
        saturation_pa=150_000.0,      # 150 kPa full scale
        quantization_bits=12,
    ),
    "gelsight_mini_32x24": CommercialTactileSpec(
        sensor_id="gelsight_mini_32x24",
        vendor_name="GelSight Inc. (GelSight Mini 视触觉硅胶阵列)",
        grid_rows=24,
        grid_cols=32,
        pad_width_mm=18.6,
        pad_height_mm=14.0,
        min_detectable_pa=30.0,       # 0.03 kPa optical elastomer threshold
        saturation_pa=80_000.0,       # 80 kPa gel saturation
        quantization_bits=16,
    ),
    "xela_uskin_6x4": CommercialTactileSpec(
        sensor_id="xela_uskin_6x4",
        vendor_name="Xela Robotics (uSkin 3轴霍尔磁触觉指尖模块)",
        grid_rows=6,
        grid_cols=4,
        pad_width_mm=18.0,
        pad_height_mm=26.0,
        min_detectable_pa=100.0,      # 0.10 kPa threshold
        saturation_pa=200_000.0,      # 200 kPa saturation
        quantization_bits=12,
    ),
    "inspire_hand_8x8": CommercialTactileSpec(
        sensor_id="inspire_hand_8x8",
        vendor_name="Inspire Robots (因时机器人 RH56DFX 灵巧手指尖阵列)",
        grid_rows=8,
        grid_cols=8,
        pad_width_mm=16.0,
        pad_height_mm=18.0,
        min_detectable_pa=120.0,
        saturation_pa=120_000.0,
        quantization_bits=10,
    ),
}


class HardwareTaxelMapper:
    """
    Maps PhySpace-Forge canonical 16x16 SI tactile frames into exact commercial hardware
    taxel resolutions, deadzones, saturation curves, and ADC quantized counts.
    """

    @staticmethod
    def map_to_hardware(
        obs: TactileFrameObservation,
        sensor_id: str = "paxini_px6ax_16x16",
    ) -> Dict[str, torch.Tensor | str | float]:
        spec = COMMERCIAL_TACTILE_SENSORS.get(
            sensor_id, COMMERCIAL_TACTILE_SENSORS["paxini_px6ax_16x16"]
        )

        # Resample 16x16 SI pressure & shear maps to hardware taxel resolution (grid_rows, grid_cols)
        p_in = obs.pressure_map_pa.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        p_hw = F.interpolate(
            p_in, size=(spec.grid_rows, spec.grid_cols), mode="bilinear", align_corners=False
        ).squeeze(0).squeeze(0)

        s_in = obs.shear_stress_map_pa.permute(2, 0, 1).unsqueeze(0)  # (1, 2, H, W)
        s_hw = (
            F.interpolate(
                s_in, size=(spec.grid_rows, spec.grid_cols), mode="bilinear", align_corners=False
            )
            .squeeze(0)
            .permute(1, 2, 0)
        )

        # Apply hardware deadzone & saturation
        p_clamped = torch.where(
            p_hw >= spec.min_detectable_pa,
            torch.clamp(p_hw, max=spec.saturation_pa),
            torch.zeros_like(p_hw),
        )

        # Compute realistic ADC digital readout (0 .. 2^bits - 1)
        max_adc = float((1 << spec.quantization_bits) - 1)
        adc_counts = torch.round((p_clamped / spec.saturation_pa) * max_adc).to(torch.int32)

        return {
            "sensor_id": spec.sensor_id,
            "vendor_name": spec.vendor_name,
            "pressure_kpa": p_clamped / 1000.0,
            "shear_kpa": s_hw / 1000.0,
            "adc_digital_counts": adc_counts,
            "peak_kpa": float(p_clamped.max().item() / 1000.0),
        }
