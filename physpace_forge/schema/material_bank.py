"""
Engineering-grade Material Constitutive Prior Bank.

Provides SI-calibrated physical material priors (Density, Young's Modulus, Poisson's Ratio,
Yield Stress, Static/Dynamic Coulomb Friction) and radial shell-to-core composite presets
used by the NVIDIA Grace Blackwell GB10 VLM Material Compiler.
"""

from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List, Tuple


class ConstitutiveModelType(IntEnum):
    """Supported continuum mechanics models in the unified ABD-VBD-IPC solver."""
    AFFINE_RIGID = 0        # 12-DoF Affine Body Dynamics (ultra-fast core)
    NEO_HOOKEAN = 1         # Hyperelastic large deformation (sponge, rubber, soft tissue)
    STVK_ELASTOPLASTIC = 2  # St. Venant-Kirchhoff with von Mises yield (cardboard, thin plastic)
    COROTATED_LINEAR = 3    # Stiff elastic shells (hard plastic, wood, ceramic)


@dataclass(frozen=True)
class MaterialPreset:
    """Calibrated SI physical properties for a homogeneous material phase."""
    name: str
    model_type: ConstitutiveModelType
    density_kg_m3: float
    youngs_modulus_pa: float
    poisson_ratio: float
    mu_static: float
    mu_dynamic: float
    yield_stress_pa: float = 1e9
    rayleigh_damping_stiffness: float = 0.01
    surface_roughness_um: float = 5.0
    prior_std_ratio: float = 0.15


@dataclass(frozen=True)
class RadialLayerSpec:
    """Defines one radial layer from outer surface (depth=0) toward core."""
    depth_range_mm: Tuple[float, float]
    material_key: str
    is_affine_core: bool = False


@dataclass(frozen=True)
class LayeredCompositePreset:
    """Shell-to-Core multi-layer constitutive profile for realistic everyday objects."""
    preset_id: str
    description: str
    layers: List[RadialLayerSpec]


def compute_lame_parameters(youngs_modulus_pa: float, poisson_ratio: float) -> Tuple[float, float]:
    """
    Convert Young's modulus E (Pa) and Poisson's ratio ν into first and second
    Lamé parameters (λ, μ_shear) in Pascals.

    μ_shear = E / (2 * (1 + ν))
    λ = (E * ν) / ((1 + ν) * (1 - 2 * ν))
    """
    mu_shear = youngs_modulus_pa / (2.0 * (1.0 + poisson_ratio))
    lmbda = (youngs_modulus_pa * poisson_ratio) / (
        (1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio)
    )
    return lmbda, mu_shear


# ============================================================================
# SI-Calibrated Engineering Material Prior Bank
# ============================================================================
MATERIAL_PRIOR_BANK: Dict[str, MaterialPreset] = {
    "silicone_elastomer_finger": MaterialPreset(
        name="Tactile Sensor Silicone Elastomer (GelSight/Soft Skin)",
        model_type=ConstitutiveModelType.NEO_HOOKEAN,
        density_kg_m3=1070.0,
        youngs_modulus_pa=2.5e5,      # 0.25 MPa soft silicone pad
        poisson_ratio=0.47,           # Nearly incompressible
        yield_stress_pa=5.0e6,
        mu_static=0.95,
        mu_dynamic=0.80,
        rayleigh_damping_stiffness=0.02,
        surface_roughness_um=2.0,
    ),
    "polyurethane_sponge_wet": MaterialPreset(
        name="Household Polyurethane Dish Sponge",
        model_type=ConstitutiveModelType.NEO_HOOKEAN,
        density_kg_m3=85.0,
        youngs_modulus_pa=2.8e4,      # 28 kPa highly compressible foam
        poisson_ratio=0.12,           # Open-cell foam compresses without bulging laterally
        yield_stress_pa=2.0e5,
        mu_static=0.65,
        mu_dynamic=0.52,
        rayleigh_damping_stiffness=0.04,
        surface_roughness_um=120.0,
    ),
    "thin_pet_bottle_shell": MaterialPreset(
        name="Thin-Walled PET Plastic Bottle Shell (Effective Structural Bulk)",
        model_type=ConstitutiveModelType.NEO_HOOKEAN,
        density_kg_m3=450.0,
        youngs_modulus_pa=1.2e5,      # 120 kPa effective shell flexural stiffness
        poisson_ratio=0.35,
        yield_stress_pa=4.5e4,
        mu_static=0.42,
        mu_dynamic=0.32,
        rayleigh_damping_stiffness=0.015,
        surface_roughness_um=1.5,
    ),
    "paper_cup_wall": MaterialPreset(
        name="Disposable Waxed Paper Cup Wall",
        model_type=ConstitutiveModelType.STVK_ELASTOPLASTIC,
        density_kg_m3=320.0,
        youngs_modulus_pa=6.5e4,      # 65 kPa easily crushed if pinched > 5 kPa
        poisson_ratio=0.25,
        yield_stress_pa=6.0e3,        # Crushes permanently past 6 kPa local stress!
        mu_static=0.38,
        mu_dynamic=0.30,
        rayleigh_damping_stiffness=0.025,
        surface_roughness_um=15.0,
    ),
    "corrugated_cardboard": MaterialPreset(
        name="Corrugated Cardboard Parcel Box",
        model_type=ConstitutiveModelType.STVK_ELASTOPLASTIC,
        density_kg_m3=210.0,
        youngs_modulus_pa=4.5e5,      # 450 kPa
        poisson_ratio=0.18,
        yield_stress_pa=3.5e4,        # Buckles under heavy pinch/stacking
        mu_static=0.50,
        mu_dynamic=0.40,
        rayleigh_damping_stiffness=0.03,
        surface_roughness_um=45.0,
    ),
    "avian_eggshell_composite": MaterialPreset(
        name="Fragile Raw Egg Calcite Shell",
        model_type=ConstitutiveModelType.COROTATED_LINEAR,
        density_kg_m3=1080.0,
        youngs_modulus_pa=3.5e6,
        poisson_ratio=0.30,
        yield_stress_pa=1.8e4,        # Fractures if peak local pressure exceeds ~18 kPa
        mu_static=0.45,
        mu_dynamic=0.36,
        rayleigh_damping_stiffness=0.005,
        surface_roughness_um=8.0,
    ),
    "hard_wood_core": MaterialPreset(
        name="Hardwood / Rigid Structural Core",
        model_type=ConstitutiveModelType.AFFINE_RIGID,
        density_kg_m3=720.0,
        youngs_modulus_pa=1.0e8,      # 100 MPa affine rigid penalty stiffness
        poisson_ratio=0.30,
        yield_stress_pa=5.0e7,
        mu_static=0.48,
        mu_dynamic=0.38,
        rayleigh_damping_stiffness=0.005,
        surface_roughness_um=12.0,
    ),
    "glazed_ceramic": MaterialPreset(
        name="Glazed Ceramic Bowl / Mug",
        model_type=ConstitutiveModelType.AFFINE_RIGID,
        density_kg_m3=2400.0,
        youngs_modulus_pa=2.0e8,
        poisson_ratio=0.22,
        yield_stress_pa=8.0e7,
        mu_static=0.32,
        mu_dynamic=0.24,
        rayleigh_damping_stiffness=0.002,
        surface_roughness_um=0.5,
    ),
}


# ============================================================================
# Shell-to-Core Composite Presets (Used by SCPP Compiler)
# ============================================================================
COMPOSITE_OBJECT_PRESETS: Dict[str, LayeredCompositePreset] = {
    "deformable_pet_bottle": LayeredCompositePreset(
        preset_id="deformable_pet_bottle",
        description="Soft thin-walled PET bottle with deformable outer body and stiffer base/neck core",
        layers=[
            RadialLayerSpec(
                depth_range_mm=(0.0, 12.0),
                material_key="thin_pet_bottle_shell",
                is_affine_core=False,
            ),
            RadialLayerSpec(
                depth_range_mm=(12.0, 1000.0),
                material_key="hard_wood_core",
                is_affine_core=True,
            ),
        ],
    ),
    "fragile_paper_cup": LayeredCompositePreset(
        preset_id="fragile_paper_cup",
        description="Disposable paper cup that yields if grasp pressure exceeds 6 kPa",
        layers=[
            RadialLayerSpec(
                depth_range_mm=(0.0, 15.0),
                material_key="paper_cup_wall",
                is_affine_core=False,
            ),
            RadialLayerSpec(
                depth_range_mm=(15.0, 1000.0),
                material_key="hard_wood_core",
                is_affine_core=True,
            ),
        ],
    ),
    "dish_sponge": LayeredCompositePreset(
        preset_id="dish_sponge",
        description="Full-body compressible polyurethane dish sponge",
        layers=[
            RadialLayerSpec(
                depth_range_mm=(0.0, 1000.0),
                material_key="polyurethane_sponge_wet",
                is_affine_core=False,
            ),
        ],
    ),
}
