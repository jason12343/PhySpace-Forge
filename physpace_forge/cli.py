"""
PhySpace-Forge Unified Open-Source Command-Line Interface (`physpace`).

Commands:
    physpace info         : List all 12 calibrated .physpack assets & 4 commercial tactile sensor specs
    physpace mcp          : Start the Model Context Protocol (MCP) JSON-RPC 2.0 server over stdio for AI Agents
    physpace agent-call   : Invoke any PhySpace Agent Tool directly via CLI and get structured JSON + reflection hints
    physpace llm-test     : Test constitutive material inference (Built-in Offline Bank or Custom OpenAI/Qwen/Ollama/vLLM API)
    physpace export       : Export any calibrated asset into universal .xml (MuJoCo), .usda (OpenUSD), .urdf (ROS2), or .gltf
    physpace studio       : Launch the full-stack PhySpace-Studio Pro web workbench (http://127.0.0.1:8765)
    physpace robot-demo   : Run a live 200Hz closed-loop robot interaction & anti-crush safety shield demo
    physpace compile      : Compile a custom .obj mesh or semantic object into universal formats & .physpack
    physpace benchmark    : Execute the 1,440-trial physical pressure accuracy benchmark
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import webbrowser
import torch

from physpace_forge.agent import AgentToolDispatcher, PHYSPACE_AGENT_TOOLS, PhySpaceMCPServer
from physpace_forge.bridge import COMMERCIAL_TACTILE_SENSORS, RealtimeRobotNode
from physpace_forge.compiler import (
    CATALOG_BY_ID,
    DomainObjectSpec,
    FOUR_DOMAIN_CATALOG,
    SafetyEnvelopeProfiler,
    VLMConstitutiveArchitect,
)
from physpace_forge.exporters import UniversalFormatExporter


def cmd_info() -> None:
    print("=" * 88)
    print("⚡ PhySpace-Forge v0.1.0 | Calibrated .physpack Catalog & Commercial Tactile Hardware")
    print("=" * 88)
    print(f"\n[1] 4-Domain Calibrated Physical Asset Catalog ({len(FOUR_DOMAIN_CATALOG)} Standard Objects):")
    print("-" * 88)
    print(f"{'Object ID':26s} | {'Domain':23s} | {'E (kPa)':>8s} | {'Yield σ_y':>9s} | {'μ_s':>5s}")
    print("-" * 88)
    for spec in FOUR_DOMAIN_CATALOG:
        print(
            f"{spec.object_id:26s} | {spec.domain:23s} | {spec.base_e_kpa:8.1f} | "
            f"{spec.yield_stress_kpa:7.1f} kPa | {spec.mu_static:5.2f}"
        )

    print(f"\n[2] Supported Commercial Tactile Sensor Hardware Profiles ({len(COMMERCIAL_TACTILE_SENSORS)}):")
    print("-" * 88)
    for sid, sspec in COMMERCIAL_TACTILE_SENSORS.items():
        print(
            f"  • {sid:20s} : {sspec.grid_rows:2d}x{sspec.grid_cols:<2d} taxels | "
            f"Range: {sspec.min_detectable_pa/1000:.2f}~{sspec.saturation_pa/1000:.0f} kPa | "
            f"{sspec.quantization_bits}-bit ADC | {sspec.vendor_name}"
        )

    arch = VLMConstitutiveArchitect()
    cfg = arch.get_config_summary()
    print("\n[3] Current Constitutive Reasoning Model Configuration (BYOM / Zero-Config):")
    print("-" * 88)
    print(f"  • Provider Mode : {cfg['provider_mode']} (set PHYSPACE_LLM_MODE='remote_llm' or 'builtin_offline')")
    print(f"  • Gateway URL   : {cfg['gateway_url']} (env: PHYSPACE_LLM_API_URL)")
    print(f"  • Model Name    : {cfg['model_name']} (env: PHYSPACE_LLM_MODEL)")
    print(f"  • API Key       : {cfg['api_key_masked']} (env: PHYSPACE_LLM_API_KEY)")
    print("=" * 88)


def cmd_robot_demo(object_id: str, sensor_id: str, steps: int) -> None:
    print("=" * 94)
    print("🤖 PhySpace-Forge 200Hz Real-Time Robot Interaction & Safety Shield Demo")
    print(f"   Target Asset : {object_id}")
    print(f"   Tactile Skin : {sensor_id}")
    print("=" * 94)
    node = RealtimeRobotNode(object_id=object_id, hardware_sensor_id=sensor_id)
    print(
        f"   Golden Safety Corridor: [{node.p_slip_min_kpa:.2f} kPa, {node.p_crush_max_kpa:.2f} kPa] "
        f"(Yield Limit σ_y = {node.spec.yield_stress_kpa:.2f} kPa)"
    )
    print("-" * 94)
    for _ in range(steps):
        res = node.step_robot_command(raw_vla_closing_delta_mm=0.22, lift_delta_mm=0.05)
        max_adc = int(res.tactile_adc_counts.max().item())
        print(
            f"Step {res.step_index:02d} ({res.compute_latency_ms:4.1f}ms) | "
            f"VLA_Raw: +{res.raw_vla_closing_mm:.2f}mm -> Safe_Cmd: {res.safe_actuator_closing_mm:+.3f}mm "
            f"[{res.shield_mode:17s}] | P_max: {res.peak_pressure_kpa:5.2f} kPa | "
            f"ADC_Peak: {max_adc:4d} | {res.safety_regime}"
        )
    print("=" * 94)


def cmd_compile(
    name: str,
    profile: str | None,
    e_kpa: float | None,
    yield_kpa: float | None,
    mu_s: float | None,
    output_path: str,
    llm_url: str | None = None,
    llm_key: str | None = None,
    llm_model: str | None = None,
) -> None:
    arch = VLMConstitutiveArchitect(
        gateway_url=llm_url,
        api_key=llm_key,
        model_name=llm_model,
        provider_mode="remote_llm" if llm_url else None,
    )
    inferred = arch.infer_single_object_constitutive(name)
    final_prof = profile or inferred["profile_curve"]
    final_e = e_kpa if e_kpa is not None else inferred["youngs_modulus_kpa"]
    final_y = yield_kpa if yield_kpa is not None else inferred["yield_stress_kpa"]
    final_mu = mu_s if mu_s is not None else inferred["mu_static"]

    oid = "custom_" + "".join(c if c.isalnum() else "_" for c in name.lower())[:20]
    spec = DomainObjectSpec(
        object_id=oid,
        domain="5_User_Custom_Assets",
        display_name=name,
        shape_primitive="ellipsoid" if final_prof == "sphere" else "cylinder_profile",
        dimensions_m=(0.035, 0.035, 0.115),
        profile_curve=final_prof,
        base_e_kpa=final_e,
        base_nu=inferred["poisson_ratio"],
        yield_stress_kpa=final_y,
        density_kg_m3=inferred["density_kg_m3"],
        mu_static=final_mu,
        mu_dynamic=round(final_mu * 0.78, 2),
        shell_thickness_mm=inferred["shell_thickness_mm"],
        has_rigid_core=True,
        semantic_desc=f"CLI-Compiled SCPP Asset: {name} ({inferred['reasoning']})",
    )
    profiler = SafetyEnvelopeProfiler(device=torch.device("cpu"))
    asset, eq_stats = profiler.build_domain_asset(spec)
    raw = f"PhySpace-CalibDB-v1|{oid}|E={final_e:.2f}|Y={final_y:.2f}|MU={final_mu:.3f}"
    sig = "PSF-" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16].upper()
    asset.metadata["calibration_signature"] = sig
    asset.metadata["frame0_equilibrium"] = eq_stats
    asset.metadata["constitutive_inference"] = inferred
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    asset.save_physpack(output_path)
    print(f"✅ Compiled SCPP Asset '{name}' -> {output_path}")
    print(f"   • Inference Model       : {inferred['model_used']}")
    print(f"   • Constitutive Params   : E={final_e} kPa, σ_y={final_y} kPa, ν={inferred['poisson_ratio']}, μ_s={final_mu}")
    print(f"   • Calibration Signature : {sig}")
    print(f"   • Tetrahedral Elements  : {asset.mechanics_core.tet_elements.shape[0]}")
    print(f"   • 2DGS Surface Splats   : {asset.visual_layer.num_splats}")
    print(f"   • Frame-0 Residual      : {eq_stats['force_balance_error_n']:.2e} N")


def cmd_export(object_id: str, fmt: str, out_dir: str) -> None:
    spec = CATALOG_BY_ID.get(object_id, CATALOG_BY_ID["tapered_coffee_cup"])
    profiler = SafetyEnvelopeProfiler(device=torch.device("cpu"))
    asset, _ = profiler.build_domain_asset(spec)
    env = {
        "p_slip_min_kpa": round(min(2.33, spec.yield_stress_kpa * 0.38), 2),
        "p_crush_max_kpa": round(spec.yield_stress_kpa * 0.88, 2),
        "optimal_target_pressure_kpa": round(spec.yield_stress_kpa * 0.56, 2),
    }
    os.makedirs(out_dir, exist_ok=True)
    if fmt == "all":
        paths = UniversalFormatExporter.export_all_formats(asset, out_dir, env)
        print(f"✅ Exported '{object_id}' into all 4 universal formats:")
        for k, p in paths.items():
            print(f"   • [{k:6s}] -> {p}")
    else:
        if fmt == "mjcf":
            content, ext = UniversalFormatExporter.to_mujoco_mjcf_xml(asset, env), "xml"
        elif fmt == "usda":
            content, ext = UniversalFormatExporter.to_openusd_usda(asset, env), "usda"
        elif fmt == "urdf":
            content, ext = UniversalFormatExporter.to_ros2_urdf(asset, env), "urdf"
        elif fmt == "gltf":
            content, ext = UniversalFormatExporter.to_gltf_json(asset, env), "gltf"
        else:
            out_file = os.path.join(out_dir, f"{object_id}.physpack")
            asset.save_physpack(out_file)
            print(f"✅ Exported '{object_id}' -> {out_file}")
            return
        out_file = os.path.join(out_dir, f"{object_id}.{ext}")
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"✅ Exported '{object_id}' ({fmt}) -> {out_file}")


def cmd_llm_test(
    prompt: str,
    url: str | None,
    key: str | None,
    model: str | None,
    mode: str | None,
) -> None:
    arch = VLMConstitutiveArchitect(
        gateway_url=url,
        api_key=key,
        model_name=model,
        provider_mode=mode or ("remote_llm" if url else None),
    )
    res = arch.infer_single_object_constitutive(prompt)
    print(json.dumps({"config": arch.get_config_summary(), "inference_result": res}, indent=2, ensure_ascii=False))


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(
        prog="physpace",
        description="PhySpace-Forge: 3D Physical Space Compiler & 200Hz RT-Patch-IPC Tactile Solver",
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("info", help="List all 12 calibrated .physpack objects, 4 tactile sensors, and LLM config")

    p_studio = sub.add_parser("studio", help="Launch PhySpace-Studio Pro full-stack interactive web server")
    p_studio.add_argument("--port", type=int, default=8765, help="HTTP port (default: 8765)")
    p_studio.add_argument("--no-browser", action="store_true", help="Do not auto-open browser")

    p_demo = sub.add_parser("robot-demo", help="Run 200Hz closed-loop robot interaction & safety shield demo")
    p_demo.add_argument("--object", type=str, default="tapered_coffee_cup", choices=list(CATALOG_BY_ID.keys()))
    p_demo.add_argument(
        "--sensor",
        type=str,
        default="paxini_px6ax_16x16",
        choices=list(COMMERCIAL_TACTILE_SENSORS.keys()),
    )
    p_demo.add_argument("--steps", type=int, default=12)

    p_comp = sub.add_parser("compile", help="Compile a custom 3D object into a watermarked .physpack file")
    p_comp.add_argument("--name", type=str, default="Thin Champagne Flute (薄壁高脚香槟杯)")
    p_comp.add_argument("--profile", type=str, default=None, choices=["waisted", "tapered_cup", "barrel", "sphere"])
    p_comp.add_argument("--e-kpa", type=float, default=None)
    p_comp.add_argument("--yield-kpa", type=float, default=None)
    p_comp.add_argument("--mu-s", type=float, default=None)
    p_comp.add_argument("--llm-url", type=str, default=None, help="Optional OpenAI-compatible Chat Completions URL")
    p_comp.add_argument("--llm-key", type=str, default=None, help="Optional API Key for LLM endpoint")
    p_comp.add_argument("--llm-model", type=str, default=None, help="Optional Model Name (e.g. gpt-4o, qwen-plus, deepseek-chat)")
    p_comp.add_argument("--output", type=str, default="benchmarks/custom_goblet.physpack")

    p_exp = sub.add_parser("export", help="Export calibrated asset to universal .xml (MuJoCo), .usda, .urdf, .gltf, or all")
    p_exp.add_argument("--object", type=str, default="tapered_coffee_cup", choices=list(CATALOG_BY_ID.keys()))
    p_exp.add_argument("--format", type=str, default="all", choices=["all", "mjcf", "usda", "urdf", "gltf", "physpack"])
    p_exp.add_argument("--out-dir", type=str, default="benchmarks/exported_universal")

    p_llm = sub.add_parser("llm-test", help="Test constitutive material inference (Built-in Offline or Custom LLM API)")
    p_llm.add_argument("prompt_pos", nargs="?", default=None, help="Optional positional object description")
    p_llm.add_argument("--prompt", type=str, default="Crispy Potato Chip (薄脆薯片)")
    p_llm.add_argument("--url", type=str, default=None, help="OpenAI-compatible /v1/chat/completions URL")
    p_llm.add_argument("--key", type=str, default=None, help="API Key (or set PHYSPACE_LLM_API_KEY)")
    p_llm.add_argument("--model", type=str, default=None, help="Model name (or set PHYSPACE_LLM_MODEL)")
    p_llm.add_argument("--mode", type=str, default=None, choices=["builtin_offline", "remote_llm", "auto"])

    sub.add_parser("benchmark", help="Run the 1,440-trial pressure accuracy benchmark vs market competitors")

    sub.add_parser(
        "mcp",
        help="Start the Model Context Protocol (MCP) JSON-RPC 2.0 server over stdio for AI Agents (Claude/Cursor/VLA)",
    )

    p_agent = sub.add_parser(
        "agent-call",
        help="Invoke a PhySpace AI Agent tool directly and output JSON with agent_reflection_hint",
    )
    p_agent.add_argument(
        "--tool",
        type=str,
        default="physpace_inspect_object_safety_envelope",
        help="Tool name (or 'list' to list all tool schemas)",
    )
    p_agent.add_argument(
        "--args",
        type=str,
        default='{"object_id": "tapered_coffee_cup"}',
        help="JSON string of tool arguments",
    )

    args = parser.parse_args()
    if args.command == "info" or args.command is None:
        cmd_info()
    elif args.command == "mcp":
        server = PhySpaceMCPServer()
        server.run_stdio_loop()
    elif args.command == "agent-call":
        dispatcher = AgentToolDispatcher()
        if args.tool == "list":
            print(json.dumps(PHYSPACE_AGENT_TOOLS, indent=2, ensure_ascii=False))
        else:
            parsed_args = json.loads(args.args)
            res = dispatcher.invoke(args.tool, parsed_args)
            print(json.dumps(res, indent=2, ensure_ascii=False))
    elif args.command == "llm-test":
        cmd_llm_test(args.prompt_pos or args.prompt, args.url, args.key, args.model, args.mode)
    elif args.command == "export":
        cmd_export(args.object, args.format, args.out_dir)
    elif args.command == "studio":
        from physpace_forge.studio.live_server import run_server

        url = f"http://127.0.0.1:{args.port}"
        if not args.no_browser:
            webbrowser.open(url)
        run_server(port=args.port)
    elif args.command == "robot-demo":
        cmd_robot_demo(args.object, args.sensor, args.steps)
    elif args.command == "compile":
        cmd_compile(
            args.name,
            args.profile,
            args.e_kpa,
            args.yield_kpa,
            args.mu_s,
            args.output,
            args.llm_url,
            args.llm_key,
            args.llm_model,
        )
    elif args.command == "benchmark":
        bench_script = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "tests", "benchmark_pressure_accuracy_vs_market.py")
        )
        subprocess.run([sys.executable, bench_script], check=True)


if __name__ == "__main__":
    main()
