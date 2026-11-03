"""
Real-Time 200Hz Robot Interaction Node (`RealtimeRobotNode`).

Provides the exact runtime interface for external users & real robots to interact
with PhySpace-Forge in real time (at 200Hz, <4.5ms per step):

Mode A — Real-Robot Shadow Twin & Active Safety Limiter (Sim-alongside-Real):
    Real robot sends its current gripper aperture / fingertip pose (and optional motor current
    or tactile sensor reading) + VLA's desired action `raw_action_mm`.
    `RealtimeRobotNode.step_robot_command()` steps the `.physpack` digital twin in <4.5ms,
    predicts the 16x16 tactile pressure map (acting as a Virtual Tactile Sensor if the real
    robot has no tactile skin!), and returns the safe, crush-free command `safe_action_mm`.

Mode B — Interactive Teleoperation / RL Gymnasium Environment (`env.step(action)`):
    User teleoperates via keyboard/VR/Leader-arm or steps an RL/VLA policy in closed loop.
"""

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional
import torch

from physpace_forge.bridge.hardware_taxel_mapper import HardwareTaxelMapper
from physpace_forge.bridge.real_robot_safety_shield import RealRobotSafetyShield
from physpace_forge.compiler import CATALOG_BY_ID, SafetyEnvelopeProfiler
from physpace_forge.sensors import TactilePressureGridSensor
from physpace_forge.solver import FingerIndenterSpec, RTPatchIPCSolver


@dataclass
class RobotRealtimeStepOutput:
    """Returned to the robot controller every 5ms (200Hz)."""
    step_index: int
    compute_latency_ms: float
    # 1. Filtered safe command to send to the real robot gripper actuator
    raw_vla_closing_mm: float
    safe_actuator_closing_mm: float
    shield_mode: str
    # 2. Real-time physical state of the grasped object
    gripper_aperture_mm: float
    indentation_depth_mm: float
    peak_pressure_kpa: float
    von_mises_stress_kpa: float
    normal_force_n: float
    tangential_force_n: float
    slip_risk_index: float
    safety_regime: str
    # 3. Hardware-aligned 2D tactile array (works as Virtual Tactile Sensor or Sim-to-Real obs)
    hardware_sensor_id: str
    tactile_pressure_kpa: torch.Tensor
    tactile_adc_counts: torch.Tensor


class RealtimeRobotNode:
    """
    200Hz Real-Time Robot Interaction Engine.
    Can be embedded directly inside a ROS2 / LeRobot control loop or queried via API.
    """

    def __init__(
        self,
        object_id: str = "tapered_coffee_cup",
        hardware_sensor_id: str = "paxini_px6ax_16x16",
        device: Optional[torch.device] = None,
    ) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.object_id = object_id
        self.hardware_sensor_id = hardware_sensor_id

        self.profiler = SafetyEnvelopeProfiler(device=self.device)
        self.spec = CATALOG_BY_ID.get(object_id, CATALOG_BY_ID["tapered_coffee_cup"])
        self.asset, self.eq_stats = self.profiler.build_domain_asset(self.spec)

        # Pre-compute Golden Safety Corridor for this object
        self.p_crush_max_kpa = round(self.spec.yield_stress_kpa * 0.88, 2)
        total_mass_kg = float(self.asset.mechanics_core.nodal_masses.sum().item())
        self.f_slip_min_n = min(total_mass_kg * 9.81, 0.65) / (2.0 * max(self.spec.mu_static, 0.2))
        self.p_slip_min_kpa = round(min(2.33, 0.52 * self.p_crush_max_kpa), 2)

        self.shield = RealRobotSafetyShield(
            p_slip_min_kpa=self.p_slip_min_kpa,
            p_crush_max_kpa=self.p_crush_max_kpa,
        )
        self.solver = RTPatchIPCSolver(
            dt_seconds=0.005,
            barrier_stiffness_pa=1.5e5,
            active_patch_radius_m=0.038,
            max_vbd_iterations=12,
            support_floor_z=0.0,
        )
        self.tactile_sensor = TactilePressureGridSensor(grid_height=16, grid_width=16)

        # Initialize 3D fingertip positions around the object's grasp waist
        verts = self.asset.mechanics_core.curr_vertices
        z_min = float(verts[:, 2].min().item())
        z_max = float(verts[:, 2].max().item())
        self.z_grasp = 0.52 * (z_min + z_max)
        mid_band = torch.abs(verts[:, 2] - self.z_grasp) < 0.016
        self.r_grasp = (
            float(verts[mid_band, 0].abs().max().item())
            if torch.any(mid_band)
            else float(verts[:, 0].abs().max().item())
        )
        self.r_f = 0.012
        self.x_init = self.r_grasp + self.r_f + 0.0014

        self.left_finger = FingerIndenterSpec(
            finger_id="left_thumb",
            center_xyz=torch.tensor([-self.x_init, 0.0, self.z_grasp], dtype=torch.float32, device=self.device),
            prev_center_xyz=torch.tensor([-self.x_init, 0.0, self.z_grasp], dtype=torch.float32, device=self.device),
            radius_m=self.r_f,
        )
        self.right_finger = FingerIndenterSpec(
            finger_id="right_index",
            center_xyz=torch.tensor([self.x_init, 0.0, self.z_grasp], dtype=torch.float32, device=self.device),
            prev_center_xyz=torch.tensor([self.x_init, 0.0, self.z_grasp], dtype=torch.float32, device=self.device),
            radius_m=self.r_f,
        )
        self.ax_left = torch.tensor([1.0, 0.0, 0.0], device=self.device)
        self.ax_up = torch.tensor([0.0, 0.0, 1.0], device=self.device)

        self.step_count = 0
        self.indent_mm = 0.0
        self.last_peak_kpa = 0.0
        self.last_slip_idx = 0.0

    def step_robot_command(
        self,
        raw_vla_closing_delta_mm: float,
        lift_delta_mm: float = 0.0,
        real_sensor_override_kpa: Optional[float] = None,
        enable_safety_shield: bool = True,
    ) -> RobotRealtimeStepOutput:
        """
        Execute one 5ms (200Hz) real-time interaction step with the robot controller.

        Args:
            raw_vla_closing_delta_mm: Desired fingertip closing displacement from VLA/teleop (mm).
            lift_delta_mm: Desired vertical lift displacement (mm).
            real_sensor_override_kpa: If the real robot has a physical tactile sensor, pass its
                reading here; if None, uses PhySpace-Forge's real-time physical twin prediction!
            enable_safety_shield: If True, RealRobotSafetyShield intercepts dangerous commands.
        """
        t0 = time.perf_counter()
        self.step_count += 1

        feedback_kpa = (
            real_sensor_override_kpa
            if real_sensor_override_kpa is not None
            else self.last_peak_kpa
        )

        if enable_safety_shield:
            decision = self.shield.filter_gripper_action(
                raw_closing_delta_mm=raw_vla_closing_delta_mm,
                measured_peak_pressure_kpa=feedback_kpa,
                measured_slip_index=self.last_slip_idx,
                is_lifting=(lift_delta_mm > 0.01),
            )
            applied_dx_mm = decision.safe_delta_aperture_mm
            shield_mode = decision.intervention_mode
        else:
            applied_dx_mm = raw_vla_closing_delta_mm
            shield_mode = "SHIELD_BYPASSED"

        dx_m = applied_dx_mm * 1e-3
        dz_m = lift_delta_mm * 1e-3
        self.indent_mm = max(0.0, self.indent_mm + applied_dx_mm)

        self.left_finger.prev_center_xyz = self.left_finger.center_xyz.clone()
        self.right_finger.prev_center_xyz = self.right_finger.center_xyz.clone()

        self.left_finger.center_xyz[0] += dx_m
        self.left_finger.center_xyz[2] += dz_m
        self.right_finger.center_xyz[0] -= dx_m
        self.right_finger.center_xyz[2] += dz_m

        tel = self.solver.step(self.asset, [self.left_finger, self.right_finger])
        obs_left = self.tactile_sensor.sense_finger_pad(
            self.asset, self.left_finger, tel, self.ax_left, self.ax_up
        )
        hw_obs = HardwareTaxelMapper.map_to_hardware(obs_left, sensor_id=self.hardware_sensor_id)

        p_kpa = tel.peak_contact_pressure_pa / 1000.0
        vm_kpa = tel.peak_von_mises_stress_pa / 1000.0
        self.last_peak_kpa = p_kpa
        self.last_slip_idx = tel.incipient_slip_fraction

        if vm_kpa > self.spec.yield_stress_kpa:
            regime = "CRUSH_DAMAGE"
        elif p_kpa < self.p_slip_min_kpa:
            regime = "SLIP_DROP_RISK"
        else:
            regime = "GOLDEN_SAFE_WINDOW"

        aperture_mm = float(
            (self.right_finger.center_xyz[0] - self.left_finger.center_xyz[0]).item() - 2.0 * self.r_f
        ) * 1000.0
        latency_ms = (time.perf_counter() - t0) * 1000.0

        return RobotRealtimeStepOutput(
            step_index=self.step_count,
            compute_latency_ms=round(latency_ms, 2),
            raw_vla_closing_mm=round(raw_vla_closing_delta_mm, 3),
            safe_actuator_closing_mm=round(applied_dx_mm, 3),
            shield_mode=shield_mode,
            gripper_aperture_mm=round(aperture_mm, 2),
            indentation_depth_mm=round(self.indent_mm, 2),
            peak_pressure_kpa=round(p_kpa, 2),
            von_mises_stress_kpa=round(vm_kpa, 2),
            normal_force_n=round(tel.total_normal_force_n, 3),
            tangential_force_n=round(tel.total_tangential_force_n, 3),
            slip_risk_index=round(tel.incipient_slip_fraction, 3),
            safety_regime=regime,
            hardware_sensor_id=self.hardware_sensor_id,
            tactile_pressure_kpa=hw_obs["pressure_kpa"],  # type: ignore[arg-type]
            tactile_adc_counts=hw_obs["adc_digital_counts"],  # type: ignore[arg-type]
        )
