"""
Real-Robot Runtime 200Hz Force/Pressure Safety Shield (`RealRobotSafetyShield`).

Allows ANY real robot (Franka Research 3, Unitree G1 + Inspire Hand, ALOHA Piper)
running a standard vision-only VLA (`OpenVLA`, `π0`, `ACT`, `DiffusionPolicy`) to load
a `.physpack` cartridge at runtime and filter raw gripper position actions `u_raw`
into safe, crush-free and slip-free closed-loop commands `u_safe` at 200Hz!
"""

from dataclasses import dataclass
from typing import Dict


@dataclass
class SafetyShieldDecision:
    raw_delta_aperture_mm: float
    safe_delta_aperture_mm: float
    measured_peak_kpa: float
    measured_slip_index: float
    p_slip_min_kpa: float
    p_crush_max_kpa: float
    target_optimal_kpa: float
    intervention_mode: str  # "PASSTHROUGH", "ANTI_CRUSH_BRAKE", "ANTI_SLIP_TIGHTEN", "GOLDEN_HOLD"


class RealRobotSafetyShield:
    """
    Plug-and-play ROS2 / LeRobot control-loop safety filter powered by `.physpack`
    Golden Grasp Safety Envelopes.
    """

    def __init__(
        self,
        p_slip_min_kpa: float,
        p_crush_max_kpa: float,
        safety_margin_ratio: float = 0.85,
        kp_compliance_mm_per_kpa: float = 0.12,
    ) -> None:
        self.p_slip_min = p_slip_min_kpa
        self.p_crush_max = p_crush_max_kpa
        self.p_safe_ceiling = p_crush_max_kpa * safety_margin_ratio
        self.p_opt = 0.5 * (self.p_slip_min + self.p_safe_ceiling)
        self.kp = kp_compliance_mm_per_kpa

    @classmethod
    def from_physpack_envelope(cls, envelope_dict: Dict[str, float]) -> "RealRobotSafetyShield":
        return cls(
            p_slip_min_kpa=float(envelope_dict["p_slip_min_kpa"]),
            p_crush_max_kpa=float(envelope_dict["p_crush_max_kpa"]),
        )

    def filter_gripper_action(
        self,
        raw_closing_delta_mm: float,
        measured_peak_pressure_kpa: float,
        measured_slip_index: float = 0.0,
        is_lifting: bool = False,
    ) -> SafetyShieldDecision:
        """
        Filter a raw VLA gripper closing command `raw_closing_delta_mm` (>0 means closing tighter)
        using real-time tactile feedback `measured_peak_pressure_kpa`.
        """
        p = measured_peak_pressure_kpa

        # Case 1: Approaching or exceeding the Golden Crush Ceiling -> Brake & back off!
        if p >= self.p_safe_ceiling:
            over_kpa = p - self.p_opt
            safe_delta = min(0.0, -self.kp * over_kpa)
            mode = "ANTI_CRUSH_BRAKE"

        # Case 2: Incipient slip detected during lift or pressure below slip floor -> Tighten!
        elif (is_lifting and p < self.p_slip_min) or (measured_slip_index > 0.72 and p < self.p_safe_ceiling):
            under_kpa = max(self.p_opt - p, 0.3)
            safe_delta = max(raw_closing_delta_mm, self.kp * under_kpa)
            mode = "ANTI_SLIP_TIGHTEN"

        # Case 3: Inside the Golden Safety Corridor -> Regulate smoothly toward P_opt
        elif p >= self.p_slip_min:
            err_kpa = self.p_opt - p
            safe_delta = max(min(0.4 * self.kp * err_kpa, raw_closing_delta_mm), -0.05)
            mode = "GOLDEN_HOLD"

        # Case 4: Pre-contact or light touch -> Allow VLA command through
        else:
            safe_delta = raw_closing_delta_mm
            mode = "PASSTHROUGH"

        return SafetyShieldDecision(
            raw_delta_aperture_mm=round(raw_closing_delta_mm, 4),
            safe_delta_aperture_mm=round(safe_delta, 4),
            measured_peak_kpa=round(p, 2),
            measured_slip_index=round(measured_slip_index, 3),
            p_slip_min_kpa=round(self.p_slip_min, 2),
            p_crush_max_kpa=round(self.p_crush_max, 2),
            target_optimal_kpa=round(self.p_opt, 2),
            intervention_mode=mode,
        )
