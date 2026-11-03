"""
Executes the 4-Domain 12-Object Safety-Envelope Compilation & Builds PhySpace-Studio Web Workbench.

1. Acquires the `NVIDIA_GB10_128GB` GPU Scheduler Lease (`127.0.0.1:9090`).
2. Compiles all 12 objects across 4 high-value manipulation domains into `.obj` and `.physpack`.
3. Runs the 16-stage progressive pinch & lift sweep on `NVIDIA GB10` to characterize each object's
   Golden Grasp Safety Corridor [P_slip_min_kpa, P_crush_max_kpa].
4. Generates the interactive Web 3D Constitutive X-Ray & Tactile Studio (`physpace_studio.html`).
"""

import json
import os
import sys
import time
from typing import Any, Dict
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from physpace_forge.cluster import GPUSchedulerLease
from physpace_forge.compiler import FOUR_DOMAIN_CATALOG, SafetyEnvelopeProfiler
from physpace_forge.studio import build_interactive_studio_html


def main() -> None:
    out_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "benchmarks"))
    mesh_dir = os.path.join(out_root, "safetybench_4d", "meshes_obj")
    physpack_dir = os.path.join(out_root, "safetybench_4d", "physpack")
    os.makedirs(mesh_dir, exist_ok=True)
    os.makedirs(physpack_dir, exist_ok=True)

    print("=" * 92)
    print(" PhySpace-SafetyBench-4D & PhySpace-Studio Interactive Workbench Builder")
    print("=" * 92)

    with GPUSchedulerLease(who="physpace_safetybench_4d", timeout_sec=600.0) as lease:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        gpu_name = torch.cuda.get_device_name(0) if device == "cuda" else "CPU"
        t0 = time.perf_counter()

        profiler = SafetyEnvelopeProfiler(device=device)
        objects_report: Dict[str, Any] = {}

        print(
            f"\n  {'Object ID':26s} | {'Domain':21s} | {'E(kPa)':>7s} | {'σ_y(kPa)':>8s} | "
            f"{'Golden Safe Window [P_min .. P_max]':35s} | {'Optimal P*'}"
        )
        print("  " + "-" * 116)

        for spec in FOUR_DOMAIN_CATALOG:
            asset, _eq = profiler.build_domain_asset(
                spec=spec,
                mesh_out_dir=mesh_dir,
                physpack_out_dir=physpack_dir,
            )
            env_report = profiler.profile_object_safety_envelope(
                spec=spec,
                asset=asset,
                num_sweep_steps=16,
            )
            objects_report[spec.object_id] = env_report
            g = env_report["golden_safety_envelope"]
            print(
                f"  {spec.object_id:26s} | {spec.domain:21s} | {spec.base_e_kpa:7.1f} | "
                f"{spec.yield_stress_kpa:8.1f} | "
                f"[{g['p_slip_min_kpa']:6.2f} kPa  <-->  {g['p_crush_max_kpa']:6.2f} kPa] "
                f"(Δ={g['safety_corridor_width_kpa']:6.2f}) | {g['optimal_target_pressure_kpa']:6.2f} kPa"
            )

        if device == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        bench_payload = {
            "benchmark_name": "PhySpace-SafetyBench-4D",
            "compute_node": gpu_name,
            "pytorch_version": torch.__version__,
            "gpu_session_id": lease.token,
            "num_domains": 4,
            "num_objects": len(objects_report),
            "total_elapsed_ms": round(elapsed_ms, 2),
            "objects": objects_report,
        }

        report_json_path = os.path.join(out_root, "safetybench_4d", "safetybench_4d_report.json")
        with open(report_json_path, "w", encoding="utf-8") as f:
            json.dump(bench_payload, f, indent=2)

        studio_html_path = os.path.join(out_root, "physpace_studio.html")
        build_interactive_studio_html(bench_payload, studio_html_path)

        print("\n" + "=" * 92)
        print(f" 4-DOMAIN 12-OBJECT SAFETY ENVELOPE SUITE COMPLETE in {elapsed_ms:.1f} ms!")
        print(f"  - Compiled .physpack Dir : {physpack_dir}")
        print(f"  - SafetyBench JSON       : {report_json_path}")
        print(f"  - PhySpace-Studio HTML5  : {studio_html_path}")
        print("=" * 92)


if __name__ == "__main__":
    main()
