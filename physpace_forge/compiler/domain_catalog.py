"""
4-Domain 12-Object Benchmark Catalog for PhySpace-SafetyBench-4D.

Focuses specifically on the industry blind spot where NVIDIA SimReady (rigid-only)
and PhysDreamer (visual-only) fail: objects with narrow "Golden Grasp Safety Windows"
bounded below by incipient slip (dropping) and above by material yield/crushing.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class DomainObjectSpec:
    object_id: str
    domain: str
    display_name: str
    shape_primitive: str  # "cylinder_profile" or "box_profile"
    dimensions_m: Tuple[float, float, float]  # (r, r, h) or (lx, ly, lz)
    profile_curve: str    # "waisted", "tapered", "ovoid", "wavy_box", "standard_box", "bellows", "berry"
    # Calibrated baseline SI properties (refined by Qwen3.8-176B VLM)
    base_e_kpa: float
    base_nu: float
    yield_stress_kpa: float
    density_kg_m3: float
    mu_static: float
    mu_dynamic: float
    shell_thickness_mm: float
    has_rigid_core: bool
    semantic_desc: str


FOUR_DOMAIN_CATALOG: List[DomainObjectSpec] = [
    # ========================================================================
    # Domain 1: Fragile Fresh & Food Handling (极易碎生鲜与食品 — 窄力窗刚需)
    # ========================================================================
    DomainObjectSpec(
        object_id="soft_silken_tofu",
        domain="1_Fragile_Food",
        display_name="Silken Tofu Block (嫩豆腐块)",
        shape_primitive="box_profile",
        dimensions_m=(0.065, 0.055, 0.042),
        profile_curve="standard_box",
        base_e_kpa=14.0,
        base_nu=0.42,
        yield_stress_kpa=2.8,   # Extremely fragile! Crushes past 2.8 kPa!
        density_kg_m3=1050.0,
        mu_static=0.55,
        mu_dynamic=0.42,
        shell_thickness_mm=100.0,
        has_rigid_core=False,
        semantic_desc="Ultra-delicate silken tofu block submerged in water film, fractures past 2.8 kPa",
    ),
    DomainObjectSpec(
        object_id="ripe_strawberry",
        domain="1_Fragile_Food",
        display_name="Ripe Red Strawberry (熟透草莓)",
        shape_primitive="cylinder_profile",
        dimensions_m=(0.019, 0.019, 0.038),
        profile_curve="berry",
        base_e_kpa=22.0,
        base_nu=0.38,
        yield_stress_kpa=3.6,   # Bruises if local stress exceeds 3.6 kPa
        density_kg_m3=920.0,
        mu_static=0.62,
        mu_dynamic=0.48,
        shell_thickness_mm=7.5,
        has_rigid_core=True,
        semantic_desc="Ripe strawberry with soft outer flesh (bruises past 3.6 kPa) and firmer pith core",
    ),
    DomainObjectSpec(
        object_id="avian_fresh_egg",
        domain="1_Fragile_Food",
        display_name="Raw Avian Egg (薄壳生鸡蛋)",
        shape_primitive="cylinder_profile",
        dimensions_m=(0.023, 0.023, 0.058),
        profile_curve="ovoid",
        base_e_kpa=2800.0,
        base_nu=0.29,
        yield_stress_kpa=18.0,
        density_kg_m3=1080.0,
        mu_static=0.42,
        mu_dynamic=0.34,
        shell_thickness_mm=9.0,
        has_rigid_core=True,
        semantic_desc="Brittle calcite eggshell over liquid albumen interior",
    ),

    # ========================================================================
    # Domain 2: Deformable Beverage & Kitchen Vessels (可变形餐饮容器)
    # ========================================================================
    DomainObjectSpec(
        object_id="tapered_coffee_cup",
        domain="2_Kitchen_Vessels",
        display_name="Waxed Paper Coffee Cup (薄壁纸咖啡杯)",
        shape_primitive="cylinder_profile",
        dimensions_m=(0.033, 0.033, 0.110),
        profile_curve="tapered",
        base_e_kpa=58.0,
        base_nu=0.26,
        yield_stress_kpa=6.0,
        density_kg_m3=320.0,
        mu_static=0.42,
        mu_dynamic=0.33,
        shell_thickness_mm=13.0,
        has_rigid_core=True,
        semantic_desc="Disposable waxed paper cup that buckles plastically past 6.0 kPa",
    ),
    DomainObjectSpec(
        object_id="ergonomic_sports_bottle",
        domain="2_Kitchen_Vessels",
        display_name="Waisted PET Water Bottle (收腰薄壁水瓶)",
        shape_primitive="cylinder_profile",
        dimensions_m=(0.036, 0.036, 0.165),
        profile_curve="waisted",
        base_e_kpa=135.0,
        base_nu=0.35,
        yield_stress_kpa=45.0,
        density_kg_m3=450.0,
        mu_static=0.42,
        mu_dynamic=0.32,
        shell_thickness_mm=11.5,
        has_rigid_core=True,
        semantic_desc="Thin-walled PET plastic bottle with waisted grip zone",
    ),
    DomainObjectSpec(
        object_id="wavy_scrub_sponge",
        domain="2_Kitchen_Vessels",
        display_name="Polyurethane Scrub Sponge (波浪洗碗海绵)",
        shape_primitive="box_profile",
        dimensions_m=(0.096, 0.064, 0.036),
        profile_curve="wavy_box",
        base_e_kpa=30.0,
        base_nu=0.12,
        yield_stress_kpa=200.0,
        density_kg_m3=85.0,
        mu_static=0.68,
        mu_dynamic=0.54,
        shell_thickness_mm=100.0,
        has_rigid_core=False,
        semantic_desc="Highly compressible open-cell polyurethane kitchen sponge",
    ),

    # ========================================================================
    # Domain 3: E-Commerce & Logistics Packaging (电商物流柔/脆包装)
    # ========================================================================
    DomainObjectSpec(
        object_id="corrugated_parcel_box",
        domain="3_Logistics_Packaging",
        display_name="Corrugated Parcel Box (瓦楞纸板快递箱)",
        shape_primitive="box_profile",
        dimensions_m=(0.105, 0.082, 0.088),
        profile_curve="standard_box",
        base_e_kpa=480.0,
        base_nu=0.20,
        yield_stress_kpa=35.0,
        density_kg_m3=210.0,
        mu_static=0.50,
        mu_dynamic=0.40,
        shell_thickness_mm=13.0,
        has_rigid_core=True,
        semantic_desc="Single-wall fluted corrugated cardboard shipping carton",
    ),
    DomainObjectSpec(
        object_id="bubble_padded_mailer",
        domain="3_Logistics_Packaging",
        display_name="Bubble-Padded Poly Mailer (气泡缓冲快递袋)",
        shape_primitive="box_profile",
        dimensions_m=(0.115, 0.075, 0.032),
        profile_curve="wavy_box",
        base_e_kpa=42.0,
        base_nu=0.28,
        yield_stress_kpa=19.0,  # Air bubbles pop if local stress > 19 kPa
        density_kg_m3=140.0,
        mu_static=0.52,
        mu_dynamic=0.40,
        shell_thickness_mm=10.5,
        has_rigid_core=True,
        semantic_desc="LDPE bubble-wrap lined courier pouch containing rigid electronics box inside",
    ),
    DomainObjectSpec(
        object_id="thin_blister_clamshell",
        domain="3_Logistics_Packaging",
        display_name="Thermoformed Blister Pack (薄壁吸塑包装盒)",
        shape_primitive="box_profile",
        dimensions_m=(0.088, 0.068, 0.048),
        profile_curve="standard_box",
        base_e_kpa=95.0,
        base_nu=0.33,
        yield_stress_kpa=11.5,  # Creases permanently past 11.5 kPa
        density_kg_m3=260.0,
        mu_static=0.36,
        mu_dynamic=0.27,
        shell_thickness_mm=11.0,
        has_rigid_core=True,
        semantic_desc="Thin clear thermoformed PET blister clamshell packaging",
    ),

    # ========================================================================
    # Domain 4: Precision Industrial & Medical Elastic Parts (工业与医疗弹性件)
    # ========================================================================
    DomainObjectSpec(
        object_id="silicone_bellows_valve",
        domain="4_Industrial_Medical",
        display_name="Medical Silicone Bellows (医疗硅胶波纹囊)",
        shape_primitive="cylinder_profile",
        dimensions_m=(0.030, 0.030, 0.095),
        profile_curve="bellows",
        base_e_kpa=180.0,
        base_nu=0.46,
        yield_stress_kpa=150.0,
        density_kg_m3=1120.0,
        mu_static=0.82,
        mu_dynamic=0.66,
        shell_thickness_mm=11.0,
        has_rigid_core=True,
        semantic_desc="Medical-grade platinum-cured silicone bellows pump body",
    ),
    DomainObjectSpec(
        object_id="soft_dropper_pipette_bulb",
        domain="4_Industrial_Medical",
        display_name="Latex Pipette Dropper Bulb (实验室滴管胶头)",
        shape_primitive="cylinder_profile",
        dimensions_m=(0.022, 0.022, 0.065),
        profile_curve="ovoid",
        base_e_kpa=85.0,
        base_nu=0.45,
        yield_stress_kpa=90.0,
        density_kg_m3=960.0,
        mu_static=0.78,
        mu_dynamic=0.62,
        shell_thickness_mm=9.5,
        has_rigid_core=True,
        semantic_desc="Compliant natural rubber suction bulb for精确 volume dispensing",
    ),
    DomainObjectSpec(
        object_id="nitrile_seal_housing",
        domain="4_Industrial_Medical",
        display_name="Nitrile-Coated Valve Sleeve (丁腈包胶阀套)",
        shape_primitive="cylinder_profile",
        dimensions_m=(0.032, 0.032, 0.085),
        profile_curve="waisted",
        base_e_kpa=620.0,
        base_nu=0.44,
        yield_stress_kpa=320.0,
        density_kg_m3=1250.0,
        mu_static=0.75,
        mu_dynamic=0.58,
        shell_thickness_mm=10.0,
        has_rigid_core=True,
        semantic_desc="Industrial rubber-overmolded metallic valve housing",
    ),
]


CATALOG_BY_ID: Dict[str, DomainObjectSpec] = {spec.object_id: spec for spec in FOUR_DOMAIN_CATALOG}
