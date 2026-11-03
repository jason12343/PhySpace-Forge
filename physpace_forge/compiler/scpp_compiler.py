"""
Phase 1: SCPP (Shell-Core Physical Primitive) Constitutive Compiler.

Runs on the 2x NVIDIA Grace Blackwell GB10 (256GB Unified Memory) cluster to convert 3D geometries
and VLM semantic material profiles into ready-to-simulate .physpack assets with:
- Graded Shell-to-Core Tetrahedral Mesh (high resolution at contact skin, 12-DoF Affine Core inside)
- Exact Tributary Contact Surface Areas A_k (m^2) for Pascal-accurate pressure P = F / A
- Spatially-varying Lamé parameters (λ(x), μ(x)), yield stress σ_y(x), and friction fields
- Barycentric-bound 2D/3D Gaussian Splats for real-time deformation rendering
"""

from typing import Optional, Tuple
import math
import torch

from physpace_forge.schema.material_bank import (
    COMPOSITE_OBJECT_PRESETS,
    MATERIAL_PRIOR_BANK,
    LayeredCompositePreset,
    compute_lame_parameters,
)
from physpace_forge.schema.physpack import (
    ContactShellLayer,
    MechanicsCoreLayer,
    SCPPAsset,
    VisualSplatLayer,
)


class SCPPCompiler:
    """
    Compiles 3D objects into Shell-Core Physical Primitives (.physpack).
    """

    def __init__(self, device: str | torch.device = "cpu") -> None:
        self.device = torch.device(device)

    def compile_cylindrical_vessel(
        self,
        asset_id: str,
        radius_m: float = 0.035,
        height_m: float = 0.16,
        num_radial: int = 4,
        num_circumferential: int = 16,
        num_vertical: int = 12,
        composite_preset_key: str = "deformable_pet_bottle",
        splats_per_surface_face: int = 4,
        d_hat_meters: float = 2.0e-3,
    ) -> SCPPAsset:
        """
        Compile a graded cylindrical vessel (e.g., deformable PET bottle or fragile paper cup)
        into a full SCPPAsset with exact tetrahedral mechanics and radial material layers.
        """
        preset = COMPOSITE_OBJECT_PRESETS[composite_preset_key]
        vertices, tets, radial_depths_mm, boundary_faces = self._build_graded_cylinder_tets(
            radius_m=radius_m,
            height_m=height_m,
            num_radial=num_radial,
            num_theta=num_circumferential,
            num_z=num_vertical,
        )
        return self._assemble_scpp_asset(
            asset_id=asset_id,
            category=composite_preset_key,
            vertices=vertices,
            tets=tets,
            radial_depths_mm=radial_depths_mm,
            boundary_faces=boundary_faces,
            preset=preset,
            splats_per_face=splats_per_surface_face,
            d_hat_meters=d_hat_meters,
        )

    def compile_box_primitive(
        self,
        asset_id: str,
        extents_xyz_m: Tuple[float, float, float] = (0.09, 0.06, 0.035),
        grid_res_xyz: Tuple[int, int, int] = (7, 5, 4),
        composite_preset_key: str = "dish_sponge",
        splats_per_surface_face: int = 4,
        d_hat_meters: float = 2.0e-3,
    ) -> SCPPAsset:
        """
        Compile a graded box/sponge/parcel object into a full SCPPAsset.
        Subdivides each hexahedral voxel into 5 positively-oriented tetrahedra.
        """
        preset = COMPOSITE_OBJECT_PRESETS[composite_preset_key]
        vertices, tets, radial_depths_mm, boundary_faces = self._build_box_tets(
            extents_xyz_m=extents_xyz_m,
            grid_res_xyz=grid_res_xyz,
        )
        return self._assemble_scpp_asset(
            asset_id=asset_id,
            category=composite_preset_key,
            vertices=vertices,
            tets=tets,
            radial_depths_mm=radial_depths_mm,
            boundary_faces=boundary_faces,
            preset=preset,
            splats_per_face=splats_per_surface_face,
            d_hat_meters=d_hat_meters,
        )

    def _assemble_scpp_asset(
        self,
        asset_id: str,
        category: str,
        vertices: torch.Tensor,
        tets: torch.Tensor,
        radial_depths_mm: torch.Tensor,
        boundary_faces: torch.Tensor,
        preset: LayeredCompositePreset,
        splats_per_face: int,
        d_hat_meters: float,
    ) -> SCPPAsset:
        """
        Core compilation routine:
        1. Orient all tetrahedra so rest volume V_e > 0 and compute D_m^{-1}.
        2. Map radial depth φ(x) to material constitutive tensors (ρ, λ, μ, σ_y, μ_s, μ_d).
        3. Compute exact tributary surface areas A_k (m^2) and outward normals n_k.
        4. Generate barycentric-bound 2D/3D Gaussian Splats on the outer surface.
        """
        num_verts = vertices.shape[0]
        num_tets = tets.shape[0]

        # 1. Compute rest-pose edge matrices D_m = [X_1 - X_0, X_2 - X_0, X_3 - X_0]
        v0 = vertices[tets[:, 0]]
        v1 = vertices[tets[:, 1]]
        v2 = vertices[tets[:, 2]]
        v3 = vertices[tets[:, 3]]
        dm = torch.stack([v1 - v0, v2 - v0, v3 - v0], dim=-1)  # (N_tets, 3, 3)
        det_dm = torch.linalg.det(dm)

        # Fix any inverted tetrahedra by swapping nodes 2 and 3
        neg_mask = det_dm < 0.0
        if torch.any(neg_mask):
            tets_fixed = tets.clone()
            tets_fixed[neg_mask, 2] = tets[neg_mask, 3]
            tets_fixed[neg_mask, 3] = tets[neg_mask, 2]
            tets = tets_fixed
            v2 = vertices[tets[:, 2]]
            v3 = vertices[tets[:, 3]]
            dm = torch.stack([v1 - v0, v2 - v0, v3 - v0], dim=-1)
            det_dm = torch.linalg.det(dm)

        rest_volumes = torch.clamp(det_dm / 6.0, min=1e-12)
        inv_rest_matrices = torch.linalg.inv(dm)

        # 2. Evaluate radial depth at element centroids & bake constitutive fields
        tet_depths_mm = radial_depths_mm[tets].mean(dim=1)  # (N_tets,)
        lame_lambda = torch.zeros(num_tets, dtype=torch.float32, device=self.device)
        lame_mu = torch.zeros(num_tets, dtype=torch.float32, device=self.device)
        tet_density = torch.zeros(num_tets, dtype=torch.float32, device=self.device)
        constitutive_type = torch.zeros(num_tets, dtype=torch.int32, device=self.device)
        element_yield_stress = torch.zeros(num_tets, dtype=torch.float32, device=self.device)
        is_core_pinned = torch.zeros(num_verts, dtype=torch.bool, device=self.device)

        # Default outer surface material properties
        outer_mat = MATERIAL_PRIOR_BANK[preset.layers[0].material_key]
        for layer_spec in preset.layers:
            d_min, d_max = layer_spec.depth_range_mm
            in_layer = (tet_depths_mm >= d_min) & (tet_depths_mm < d_max)
            mat = MATERIAL_PRIOR_BANK[layer_spec.material_key]
            lmbda, mu_shear = compute_lame_parameters(mat.youngs_modulus_pa, mat.poisson_ratio)

            lame_lambda[in_layer] = lmbda
            lame_mu[in_layer] = mu_shear
            tet_density[in_layer] = mat.density_kg_m3
            constitutive_type[in_layer] = int(mat.model_type)
            element_yield_stress[in_layer] = mat.yield_stress_pa

            if layer_spec.is_affine_core:
                vert_in_core = (radial_depths_mm >= d_min) & (radial_depths_mm < d_max)
                is_core_pinned |= vert_in_core

        # Lumped nodal masses m_i = sum_{e ni i} (ρ_e * V_e / 4)
        nodal_masses = torch.zeros(num_verts, dtype=torch.float32, device=self.device)
        quarter_tet_mass = (tet_density * rest_volumes) * 0.25
        for c in range(4):
            nodal_masses.scatter_add_(0, tets[:, c], quarter_tet_mass)
        nodal_masses = torch.clamp(nodal_masses, min=1e-6)

        # 3. Build ContactShellLayer with exact tributary areas A_k (m^2) and outward normals
        surf_vert_indices, inverse_surf_idx = torch.unique(boundary_faces, return_inverse=True)
        num_surf_verts = surf_vert_indices.shape[0]

        f_v0 = vertices[boundary_faces[:, 0]]
        f_v1 = vertices[boundary_faces[:, 1]]
        f_v2 = vertices[boundary_faces[:, 2]]
        face_cross = torch.linalg.cross(f_v1 - f_v0, f_v2 - f_v0, dim=-1)
        face_areas = 0.5 * torch.linalg.norm(face_cross, dim=-1)  # (N_surf_faces,)
        face_normals = face_cross / torch.clamp(2.0 * face_areas.unsqueeze(-1), min=1e-12)

        # Ensure face normals point outward from object centroid
        obj_center = vertices.mean(dim=0, keepdim=True)
        face_centers = (f_v0 + f_v1 + f_v2) / 3.0
        inward_faces = ((face_centers - obj_center) * face_normals).sum(dim=-1) < 0.0
        if torch.any(inward_faces):
            boundary_faces = boundary_faces.clone()
            boundary_faces[inward_faces, 1], boundary_faces[inward_faces, 2] = (
                boundary_faces[inward_faces, 2].clone(),
                boundary_faces[inward_faces, 1].clone(),
            )
            face_normals[inward_faces] = -face_normals[inward_faces]
            _, inverse_surf_idx = torch.unique(boundary_faces, return_inverse=True)

        # Distribute 1/3 of each triangle's area and normal to its 3 boundary vertices
        vertex_contact_areas = torch.zeros(num_surf_verts, dtype=torch.float32, device=self.device)
        accum_normals = torch.zeros((num_surf_verts, 3), dtype=torch.float32, device=self.device)
        third_area = face_areas / 3.0
        for c in range(3):
            local_v_idx = inverse_surf_idx[:, c]
            vertex_contact_areas.scatter_add_(0, local_v_idx, third_area)
            accum_normals.index_add_(0, local_v_idx, face_normals * third_area.unsqueeze(-1))

        vertex_normals = accum_normals / torch.clamp(
            torch.linalg.norm(accum_normals, dim=-1, keepdim=True), min=1e-12
        )

        contact_shell = ContactShellLayer(
            surface_faces=boundary_faces,
            surface_vertex_indices=surf_vert_indices,
            vertex_contact_areas=vertex_contact_areas,
            vertex_normals=vertex_normals,
            mu_static=torch.full(
                (num_surf_verts,), outer_mat.mu_static, dtype=torch.float32, device=self.device
            ),
            mu_dynamic=torch.full(
                (num_surf_verts,), outer_mat.mu_dynamic, dtype=torch.float32, device=self.device
            ),
            yield_stress_pa=torch.full(
                (num_surf_verts,), outer_mat.yield_stress_pa, dtype=torch.float32, device=self.device
            ),
            d_hat_meters=d_hat_meters,
        )

        # 4. Build Barycentric-bound VisualSplatLayer (2DGS/3DGS on surface triangles)
        num_faces = boundary_faces.shape[0]
        num_splats = num_faces * splats_per_face
        bound_face_indices = torch.arange(num_faces, device=self.device).repeat_interleave(
            splats_per_face
        )
        # Uniform symmetric barycentric samples inside each triangle
        bary_u = torch.full((num_splats,), 1.0 / 3.0, dtype=torch.float32, device=self.device)
        bary_v = torch.full((num_splats,), 1.0 / 3.0, dtype=torch.float32, device=self.device)
        if splats_per_face > 1:
            offsets = torch.linspace(-0.15, 0.15, splats_per_face, device=self.device).repeat(num_faces)
            bary_u = torch.clamp(bary_u + offsets, 0.05, 0.90)
            bary_v = torch.clamp(1.0 - bary_u - 0.33, 0.05, 0.90)
        bary_w = 1.0 - bary_u - bary_v
        barycentric_weights = torch.stack([bary_u, bary_v, bary_w], dim=-1)

        # Surfel-like anisotropic covariance disk tangent to each surface face
        avg_radius = torch.sqrt(face_areas[bound_face_indices] / (splats_per_face * math.pi))
        rest_cov = torch.eye(3, dtype=torch.float32, device=self.device).unsqueeze(0).repeat(
            num_splats, 1, 1
        )
        rest_cov = rest_cov * (avg_radius.view(-1, 1, 1) ** 2)

        visual_layer = VisualSplatLayer(
            num_splats=num_splats,
            barycentric_weights=barycentric_weights,
            bound_face_indices=bound_face_indices,
            normal_offsets=torch.zeros((num_splats, 1), dtype=torch.float32, device=self.device),
            rest_covariances=rest_cov,
            colors_rgb=torch.full((num_splats, 3), 0.75, dtype=torch.float32, device=self.device),
            opacities=torch.full((num_splats, 1), 0.95, dtype=torch.float32, device=self.device),
        )

        mechanics_core = MechanicsCoreLayer(
            rest_vertices=vertices.clone(),
            curr_vertices=vertices.clone(),
            velocities=torch.zeros_like(vertices),
            nodal_masses=nodal_masses,
            is_core_pinned=is_core_pinned,
            tet_elements=tets,
            rest_volumes=rest_volumes,
            inv_rest_matrices=inv_rest_matrices,
            lame_lambda=lame_lambda,
            lame_mu=lame_mu,
            constitutive_type=constitutive_type,
            element_yield_stress=element_yield_stress,
        )

        return SCPPAsset(
            asset_id=asset_id,
            category=category,
            visual_layer=visual_layer,
            contact_shell=contact_shell,
            mechanics_core=mechanics_core,
            metadata={
                "compiler": "PhySpace-Forge SCPPCompiler v0.1",
                "preset": preset.preset_id,
                "total_mass_kg": f"{ float(nodal_masses.sum().item()):.4f}",
                "surface_area_m2": f"{float(vertex_contact_areas.sum().item()):.5f}",
            },
        )

    def _build_graded_cylinder_tets(
        self,
        radius_m: float,
        height_m: float,
        num_radial: int,
        num_theta: int,
        num_z: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate a graded cylindrical tetrahedral mesh with finer radial spacing near the outer skin.
        """
        # Graded radial rings: denser near r = radius_m
        s = torch.linspace(0.25, 1.0, num_radial, device=self.device)
        radii = radius_m * (1.0 - (1.0 - s) ** 1.4)
        thetas = torch.linspace(0.0, 2.0 * math.pi, num_theta + 1, device=self.device)[:-1]
        zs = torch.linspace(0.0, height_m, num_z, device=self.device)

        verts_list = []
        depths_mm_list = []

        # Central axis spine vertices (num_z)
        for z_val in zs:
            verts_list.append([0.0, 0.0, float(z_val.item())])
            depths_mm_list.append(radius_m * 1000.0)

        # Concentric ring vertices: idx = num_z + iz * (num_radial * num_theta) + ir * num_theta + it
        for z_val in zs:
            for r_val in radii:
                depth_mm = float((radius_m - r_val).item()) * 1000.0
                for th_val in thetas:
                    x = float((r_val * torch.cos(th_val)).item())
                    y = float((r_val * torch.sin(th_val)).item())
                    verts_list.append([x, y, float(z_val.item())])
                    depths_mm_list.append(depth_mm)

        vertices = torch.tensor(verts_list, dtype=torch.float32, device=self.device)
        radial_depths_mm = torch.tensor(depths_mm_list, dtype=torch.float32, device=self.device)

        def ring_idx(iz: int, ir: int, it: int) -> int:
            return num_z + iz * (num_radial * num_theta) + ir * num_theta + (it % num_theta)

        tets_list = []
        boundary_faces_list = []

        for iz in range(num_z - 1):
            c0 = iz
            c1 = iz + 1
            # Inner core wedge between axis spine and first radial ring (ir = 0)
            for it in range(num_theta):
                a0 = ring_idx(iz, 0, it)
                b0 = ring_idx(iz, 0, it + 1)
                a1 = ring_idx(iz + 1, 0, it)
                b1 = ring_idx(iz + 1, 0, it + 1)
                tets_list.append([c0, a0, b0, c1])
                tets_list.append([a0, b0, b1, c1])
                tets_list.append([a0, b1, a1, c1])

                if iz == 0:
                    boundary_faces_list.append([c0, b0, a0])
                if iz == num_z - 2:
                    boundary_faces_list.append([c1, a1, b1])

            # Concentric hexahedral shells subdivided into 5 tetrahedra each
            for ir in range(num_radial - 1):
                for it in range(num_theta):
                    v000 = ring_idx(iz, ir, it)
                    v100 = ring_idx(iz, ir + 1, it)
                    v110 = ring_idx(iz, ir + 1, it + 1)
                    v010 = ring_idx(iz, ir, it + 1)

                    v001 = ring_idx(iz + 1, ir, it)
                    v101 = ring_idx(iz + 1, ir + 1, it)
                    v111 = ring_idx(iz + 1, ir + 1, it + 1)
                    v011 = ring_idx(iz + 1, ir, it + 1)

                    tets_list.extend([
                        [v000, v100, v110, v101],
                        [v000, v110, v010, v011],
                        [v000, v001, v101, v011],
                        [v111, v101, v011, v110],
                        [v000, v110, v101, v011],
                    ])

                    # Outer lateral cylindrical surface
                    if ir == num_radial - 2:
                        boundary_faces_list.append([v100, v110, v111])
                        boundary_faces_list.append([v100, v111, v101])
                    if iz == 0:
                        boundary_faces_list.append([v000, v010, v110])
                        boundary_faces_list.append([v000, v110, v100])
                    if iz == num_z - 2:
                        boundary_faces_list.append([v001, v101, v111])
                        boundary_faces_list.append([v001, v111, v011])

        tets = torch.tensor(tets_list, dtype=torch.int64, device=self.device)
        boundary_faces = torch.tensor(boundary_faces_list, dtype=torch.int64, device=self.device)
        return vertices, tets, radial_depths_mm, boundary_faces

    def _build_box_tets(
        self,
        extents_xyz_m: Tuple[float, float, float],
        grid_res_xyz: Tuple[int, int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build a tetrahedralized box mesh with exact surface boundary faces and depth field."""
        lx, ly, lz = extents_xyz_m
        nx, ny, nz = grid_res_xyz
        xs = torch.linspace(-0.5 * lx, 0.5 * lx, nx, device=self.device)
        ys = torch.linspace(-0.5 * ly, 0.5 * ly, ny, device=self.device)
        zs = torch.linspace(0.0, lz, nz, device=self.device)

        grid_z, grid_y, grid_x = torch.meshgrid(zs, ys, xs, indexing="ij")
        vertices = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1), grid_z.reshape(-1)], dim=-1)

        # Exact distance to nearest box boundary face in mm
        dx = 0.5 * lx - torch.abs(vertices[:, 0])
        dy = 0.5 * ly - torch.abs(vertices[:, 1])
        dz = torch.minimum(vertices[:, 2], lz - vertices[:, 2])
        radial_depths_mm = torch.minimum(torch.minimum(dx, dy), dz) * 1000.0

        def v_idx(iz: int, iy: int, ix: int) -> int:
            return iz * (ny * nx) + iy * nx + ix

        tets_list = []
        boundary_faces_list = []

        for iz in range(nz - 1):
            for iy in range(ny - 1):
                for ix in range(nx - 1):
                    v000 = v_idx(iz, iy, ix)
                    v100 = v_idx(iz, iy, ix + 1)
                    v110 = v_idx(iz, iy + 1, ix + 1)
                    v010 = v_idx(iz, iy + 1, ix)

                    v001 = v_idx(iz + 1, iy, ix)
                    v101 = v_idx(iz + 1, iy, ix + 1)
                    v111 = v_idx(iz + 1, iy + 1, ix + 1)
                    v011 = v_idx(iz + 1, iy + 1, ix)

                    tets_list.extend([
                        [v000, v100, v110, v101],
                        [v000, v110, v010, v011],
                        [v000, v001, v101, v011],
                        [v111, v101, v011, v110],
                        [v000, v110, v101, v011],
                    ])

                    if ix == 0:
                        boundary_faces_list.extend([[v000, v001, v011], [v000, v011, v010]])
                    if ix == nx - 2:
                        boundary_faces_list.extend([[v100, v110, v111], [v100, v111, v101]])
                    if iy == 0:
                        boundary_faces_list.extend([[v000, v100, v101], [v000, v101, v001]])
                    if iy == ny - 2:
                        boundary_faces_list.extend([[v010, v011, v111], [v010, v111, v110]])
                    if iz == 0:
                        boundary_faces_list.extend([[v000, v010, v110], [v000, v110, v100]])
                    if iz == nz - 2:
                        boundary_faces_list.extend([[v001, v101, v111], [v001, v111, v011]])

        tets = torch.tensor(tets_list, dtype=torch.int64, device=self.device)
        boundary_faces = torch.tensor(boundary_faces_list, dtype=torch.int64, device=self.device)
        return vertices, tets, radial_depths_mm, boundary_faces
