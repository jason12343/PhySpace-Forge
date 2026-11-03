"""
SCPP (Shell-Core Physical Primitive) Unified Asset Data Structure (.physpack).

Implements the Tri-Layer Neuro-Physical Representation:
1. VisualSplatLayer: 2D/3D Gaussian Splats bound to surface triangles via barycentric coords.
2. ContactShellLayer: Smooth narrow-band Analytical/Grid SDF + surface contact patches with area weights.
3. MechanicsCoreLayer: Graded tetrahedral mesh coupled to a 12-DoF Affine Body Dynamics (ABD) core.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import torch


@dataclass
class VisualSplatLayer:
    """
    Outer visual appearance layer (2D/3D Gaussian Splats) bound to the deformable surface mesh.
    When the tetrahedral mechanics core deforms, splat positions and covariance matrices
    update in <0.05 ms via local deformation gradients F * Σ * F^T.
    """
    num_splats: int
    # Local barycentric coordinates (u, v, w) on bound surface face: shape (N_splats, 3)
    barycentric_weights: torch.Tensor
    # Bound surface triangle index for each splat: shape (N_splats,), int64
    bound_face_indices: torch.Tensor
    # Normal offset along triangle normal in rest pose: shape (N_splats, 1)
    normal_offsets: torch.Tensor
    # Rest-pose 3x3 covariance matrices Σ_0: shape (N_splats, 3, 3)
    rest_covariances: torch.Tensor
    # RGB appearance colors or SH coefficients: shape (N_splats, 3)
    colors_rgb: torch.Tensor
    # Opacities α in [0, 1]: shape (N_splats, 1)
    opacities: torch.Tensor


@dataclass
class ContactShellLayer:
    """
    Middle contact mechanics layer: analytical/grid Signed Distance Field (SDF)
    and area-weighted surface contact elements for exact Pascal pressure P = F / A.
    """
    # Surface triangle vertex indices into MechanicsCoreLayer.vertices: shape (N_surf_faces, 3)
    surface_faces: torch.Tensor
    # Boundary vertex indices (subset of tet vertices lying on the outer surface): shape (N_surf_verts,)
    surface_vertex_indices: torch.Tensor
    # Tributary contact area A_k (m^2) assigned to each surface vertex: shape (N_surf_verts,)
    vertex_contact_areas: torch.Tensor
    # Outward unit surface normals at each surface vertex (updated per step): shape (N_surf_verts, 3)
    vertex_normals: torch.Tensor
    # Static Coulomb friction coefficient μ_s per surface vertex: shape (N_surf_verts,)
    mu_static: torch.Tensor
    # Dynamic Coulomb friction coefficient μ_d per surface vertex: shape (N_surf_verts,)
    mu_dynamic: torch.Tensor
    # Yield stress threshold (Pa) per surface vertex (for crush/damage detection): shape (N_surf_verts,)
    yield_stress_pa: torch.Tensor
    # IPC barrier activation hat distance d_hat (meters, default 2.0e-3 = 2mm)
    d_hat_meters: float = 2.0e-3


@dataclass
class MechanicsCoreLayer:
    """
    Inner continuum mechanics layer combining a 12-DoF Affine Body Dynamics (ABD) core
    with a graded deformable tetrahedral skin.
    """
    # Rest-pose vertex coordinates X_0 (meters): shape (N_verts, 3)
    rest_vertices: torch.Tensor
    # Current deformed vertex coordinates x (meters): shape (N_verts, 3)
    curr_vertices: torch.Tensor
    # Vertex velocities v (m/s): shape (N_verts, 3)
    velocities: torch.Tensor
    # Lumped nodal masses m_i (kg): shape (N_verts,)
    nodal_masses: torch.Tensor
    # Boolean mask: True if vertex is pinned to the 12-DoF Affine Body core: shape (N_verts,)
    is_core_pinned: torch.Tensor

    # Tetrahedral element connectivity (indices into vertices): shape (N_tets, 4), int64
    tet_elements: torch.Tensor
    # Rest-pose element volumes V_e (m^3): shape (N_tets,)
    rest_volumes: torch.Tensor
    # Inverse rest-pose edge matrices D_m^{-1} for computing F = D_s * D_m^{-1}: shape (N_tets, 3, 3)
    inv_rest_matrices: torch.Tensor
    # First Lamé parameter λ_e (Pa) per tetrahedron: shape (N_tets,)
    lame_lambda: torch.Tensor
    # Second Lamé parameter (shear modulus) μ_e (Pa) per tetrahedron: shape (N_tets,)
    lame_mu: torch.Tensor
    # Constitutive model enum per tetrahedron: shape (N_tets,), int32
    constitutive_type: torch.Tensor
    # Element yield stress threshold σ_y (Pa): shape (N_tets,)
    element_yield_stress: torch.Tensor

    # 12-DoF Affine Body core state: q_affine = [A_col0, A_col1, A_col2, translation] in R^{3x4}
    affine_transform: torch.Tensor = field(
        default_factory=lambda: torch.eye(3, 4, dtype=torch.float32)
    )
    affine_velocity: torch.Tensor = field(
        default_factory=lambda: torch.zeros((3, 4), dtype=torch.float32)
    )


@dataclass
class ArticulationJoint:
    """Kinematic joint connecting two SCPP sub-bodies (compatible with URDF / OpenUSD)."""
    joint_name: str
    joint_type: str  # "revolute", "prismatic", "fixed"
    parent_part_id: str
    child_part_id: str
    axis: Tuple[float, float, float] = (0.0, 0.0, 1.0)
    origin_xyz: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    limits_lower_upper: Tuple[float, float] = (-3.14159, 3.14159)
    damping: float = 0.1
    friction_torque: float = 0.05


@dataclass
class SCPPAsset:
    """Complete .physpack asset encapsulating Visual, Contact Shell, and Mechanics Core."""
    asset_id: str
    category: str
    visual_layer: VisualSplatLayer
    contact_shell: ContactShellLayer
    mechanics_core: MechanicsCoreLayer
    joints: List[ArticulationJoint] = field(default_factory=list)
    metadata: Dict[str, str] = field(default_factory=dict)

    @property
    def device(self) -> torch.device:
        return self.mechanics_core.curr_vertices.device

    def to(self, device: torch.device | str) -> "SCPPAsset":
        """Move all tensor fields across CPU / GPU / Unified Memory seamlessly."""
        dev = torch.device(device)
        for layer in (self.visual_layer, self.contact_shell, self.mechanics_core):
            for k, v in vars(layer).items():
                if isinstance(v, torch.Tensor):
                    setattr(layer, k, v.to(dev))
        return self

    def save_physpack(self, path: str) -> None:
        """Serialize the compiled SCPPAsset into a portable .physpack bundle."""
        torch.save(self, path)

    @staticmethod
    def load_physpack(path: str, map_location: Optional[str] = "cpu") -> "SCPPAsset":
        """Load a compiled .physpack asset."""
        return torch.load(path, map_location=map_location, weights_only=False)
