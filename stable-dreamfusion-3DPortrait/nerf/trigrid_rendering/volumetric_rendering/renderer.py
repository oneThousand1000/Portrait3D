# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""
The renderer is a module that takes in rays, decides where to sample along each
ray, and computes pixel colors using the volume rendering equation.
"""

import math
import torch
from nerf.torch_utils import misc
from nerf.trigrid_rendering.volumetric_rendering.ray_marcher import MipRayMarcher2
from nerf.trigrid_rendering.volumetric_rendering import math_utils
# from training.aligned_smplx import AlignedSMPLX

#from training.aligned_smpl import AlignedSMPL
import smplx
from kaolin.ops.mesh import index_vertices_by_faces
from kaolin.metrics.trianglemesh import point_to_mesh_distance

from nerf.trigrid_rendering.aligned_smpl import AlignedSMPL
import trimesh




# def generate_planes():
#     """
#     Defines planes by the three vectors that form the "axes" of the
#     plane. Should work with arbitrary number of planes and planes of
#     arbitrary orientation.
#     """
#     return torch.tensor([[[1, 0, 0],
#                             [0, 1, 0],
#                             [0, 0, 1]],
#                             [[1, 0, 0],
#                             [0, 0, 1],
#                             [0, 1, 0]],
#                             [[0, 0, 1],
#                             [1, 0, 0],
#                             [0, 1, 0]]], dtype=torch.float32)

# correct tri-planes, see https://github.com/NVlabs/eg3d/issues/67
def generate_planes():
    """
    Defines planes by the three vectors that form the "axes" of the
    plane. Should work with arbitrary number of planes and planes of
    arbitrary orientation.
    """
    return torch.tensor([[[1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 1]],
                            [[1, 0, 0],
                            [0, 0, 1],
                            [0, 1, 0]],
                            [[0, 1, 0],
                            [0, 0, 1],
                            [1, 0, 0]]], dtype=torch.float32)

def project_onto_planes(planes, coordinates):
    """
    Does a projection of a 3D point onto a batch of 2D planes,
    returning 2D plane coordinates.

    Takes plane axes of shape n_planes, 3, 3
    # Takes coordinates of shape N, M, 3
    # returns projections of shape N*n_planes, M, 2
    """
    N, M, C = coordinates.shape
    n_planes, _, _ = planes.shape
    coordinates = coordinates.unsqueeze(1).expand(-1, n_planes, -1, -1).reshape(N*n_planes, M, 3)
    inv_planes = torch.linalg.inv(planes).unsqueeze(0).expand(N, -1, -1, -1).reshape(N*n_planes, 3, 3)
    projections = torch.bmm(coordinates, inv_planes)
    return projections

def sample_from_planes(plane_axes, plane_features, coordinates, mode='bilinear', padding_mode='zeros', box_warp=None, triplane_depth=1,render_high_freq = True):
    assert padding_mode == 'zeros'
    output_features = None


    _, M, _ = coordinates.shape
    coordinates = (2 / box_warp) * coordinates  # TODO: add specific box bounds
    projected_coordinates = project_onto_planes(plane_axes, coordinates).unsqueeze(1).unsqueeze(2)  # (N x n_planes) x 1 x 1 x M x 3
    for res_k in plane_features:
        plane_feature = plane_features[res_k]
        N, n_planes, CD, H, W = plane_feature.shape
        # _, M, _ = coordinates.shape
        C, D = CD // triplane_depth, triplane_depth
        plane_feature = plane_feature.view(N * n_planes, C, D, H, W)

        # coordinates = (2/box_warp) * coordinates # TODO: add specific box bounds

        # projected_coordinates = project_onto_planes(plane_axes, coordinates).unsqueeze(1).unsqueeze(2) # (N x n_planes) x 1 x 1 x M x 3
        output_feature = torch.nn.functional.grid_sample(plane_feature, projected_coordinates.float(), mode=mode,
                                                             padding_mode=padding_mode, align_corners=False).permute(0,
                                                                                                                     4,
                                                                                                                     3,
                                                                                                                     2,
                                                                                                                     1).reshape(N, n_planes, M, C)
        if output_features is None:
            output_features = output_feature
        else:
            output_features += output_feature

    output_features /= len(plane_features)

    return output_features

def sample_from_3dgrid(grid, coordinates):
    """
    Expects coordinates in shape (batch_size, num_points_per_batch, 3)
    Expects grid in shape (1, channels, H, W, D)
    (Also works if grid has batch size)
    Returns sampled features of shape (batch_size, num_points_per_batch, feature_channels)
    """
    batch_size, n_coords, n_dims = coordinates.shape
    sampled_features = torch.nn.functional.grid_sample(grid.expand(batch_size, -1, -1, -1, -1),
                                                       coordinates.reshape(batch_size, 1, 1, -1, n_dims),
                                                       mode='bilinear', padding_mode='zeros', align_corners=False)
    N, C, H, W, D = sampled_features.shape
    sampled_features = sampled_features.permute(0, 4, 3, 2, 1).reshape(N, H*W*D, C)
    return sampled_features

def triplane_crop_mask(xyz_unformatted, thresh, boxwarp, allow_bottom=True):
    # bw,tc = boxwarp, thresh
    bw = boxwarp
    tc = boxwarp * thresh
    device = xyz_unformatted.device
    # xyz = 0.5 * (xyz_unformatted+1) * torch.tensor([-1,1,-1]).to(device)[None,None,:]
    xyz = (xyz_unformatted) * torch.tensor([-1,1,-1]).to(device)[None,None,:]
    ans = (xyz[:,:,[0,2]].abs() <= (bw/2-tc)).all(dim=-1,keepdim=True)
    if allow_bottom:
        ans = ans | (
            (xyz[:,:,1:2] <= -(bw/2-tc)) &
            (xyz[:,:,[0,2]].abs() <= (bw/2-tc)).all(dim=-1,keepdim=True)
        )
    return ~ans
def cull_clouds_mask(denities, thresh):
    denities = torch.nn.functional.softplus(denities - 1) # activation bias of -1 makes things initialize better
    alpha = 1 - torch.exp(-denities)
    return alpha < thresh



class ImportanceRenderer(torch.nn.Module):
    def __init__(self, w_dim, num_ws,batch_size,thickness,box_warp,apply_deformation = True):
        super().__init__()
        self.ray_marcher = MipRayMarcher2()
        self.plane_axes = generate_planes()
        self.batch_size = batch_size
        self.num_betas = 10
        self.apply_deformation = apply_deformation
        if apply_deformation:
            body_model_smpl = smplx.create('./smplx_models',
                                            model_type='smpl',
                                            gender='neutral',
                                            use_compressed=False,
                                            use_face_contour=True,
                                            num_betas=self.num_betas,
                                            num_expression_coeffs=10,
                                            ext='npz',
                                            batch_size = batch_size
                                            ).cuda()
            self.aligned_SMPL = AlignedSMPL(model=body_model_smpl,batch_size=batch_size)



            shaped_smpl_data = self.aligned_SMPL.generate_shaped_smpl(
                        betas=None,
                        scale=None,  # shape_params['scale'],
                        transl=None,  # shape_params['transl']
                    )
            shaped_smpl = shaped_smpl_data['vertices'].detach().contiguous()
            align_points = shaped_smpl_data['align_joint_coordinate'].detach().contiguous()

            self.register_buffer('shaped_smpl', shaped_smpl)
            self.register_buffer('align_points', align_points)

            # shaped_smpl [B,N,3]
            # filter points that outside box
            box_side_length = box_warp
            # shaped_smpl: B,N,3
            point_mask = shaped_smpl[0:1,:,0] > -box_side_length/2  # 1,N
            point_mask = point_mask & (shaped_smpl[0:1,:,0] < box_side_length/2)
            point_mask = point_mask & (shaped_smpl[0:1,:,1] > -box_side_length/2)
            point_mask = point_mask & (shaped_smpl[0:1,:,1] < box_side_length/2)
            point_mask = point_mask & (shaped_smpl[0:1,:,2] > -box_side_length/2)
            point_mask = point_mask & (shaped_smpl[0:1,:,2] < box_side_length/2)
            point_mask = point_mask.squeeze(0).cuda() # N

            faces = self.aligned_SMPL.faces   # [20908, 3]
            face_mask = torch.ones(faces.shape[0],dtype=torch.bool).cuda() # [20908]
            for i in range(faces.shape[0]):
                face_mask[i] = point_mask[faces[i,0]] and point_mask[faces[i,1]] and point_mask[faces[i,2]]
            self.register_buffer('face_mask', face_mask)

            self.thickness = thickness

            # shaped_smpl [B,N,3]
            # filter points that not on the head
            # shaped_smpl: B,N,3

            #
            # point_mask = shaped_smpl[0:1, :, 1] > 0  # 1,N

            point_mask = shaped_smpl[0:1, :, 1] > 0.06  # 1,N
            point_mask = point_mask & (shaped_smpl[0:1, :, 2] < -0.0)

            point_mask = point_mask.squeeze(0).cuda()  # N

            faces = self.aligned_SMPL.faces  # [20908, 3]
            head_face_mask = torch.ones(faces.shape[0], dtype=torch.bool).cuda()  # [20908]
            for i in range(faces.shape[0]):
                head_face_mask[i] = point_mask[faces[i, 0]] and point_mask[faces[i, 1]] and point_mask[faces[i, 2]]
            self.register_buffer('head_face_mask', head_face_mask)

            self.back_head_depth = None
        #
        # print('head_face_mask shape:',head_face_mask.shape)


    def set_batch_size(self,batch_size):
        self.batch_size = batch_size
        body_model_smpl = smplx.create('./smplx_models',
                                       model_type='smpl',
                                       gender='neutral',
                                       use_compressed=False,
                                       use_face_contour=True,
                                       num_betas=self.num_betas,
                                       num_expression_coeffs=10,
                                       ext='npz',
                                       batch_size=batch_size
                                       ).to(self.aligned_SMPL.model.shapedirs.device)
        self.aligned_SMPL.set_model(body_model_smpl)
        self.aligned_SMPL.set_batch_size(batch_size)
        shaped_smpl_data = self.aligned_SMPL.generate_shaped_smpl(
            betas=None,
            scale=None,  # shape_params['scale'],
            transl=None,  # shape_params['transl']
        )
        shaped_smpl = shaped_smpl_data['vertices'].detach().contiguous()
        align_points = shaped_smpl_data['align_joint_coordinate'].detach().contiguous()
        self.register_buffer('shaped_smpl', shaped_smpl)
        self.register_buffer('align_points', align_points)


    def render_meshes(self, shape_pose_params,resolution,cameras):
        images = self.aligned_SMPL.get_visualization(shape_pose_params, resolution, cameras)
        return images


    def get_deformed_coordinate(self, ws, pose_params, original_coordinate):


        posed_smpl = self.aligned_SMPL.generate_posed_smpl(betas=None,
                                                              body_pose=pose_params,
                                                              scale=None, # shape_params['scale'],
                                                              transl=None, # shape_params['transl'],
                                                              align_joint_coordinate=self.align_points)['vertices']
        # misc.assert_shape(posed_smpl, [None, 10475, 3])


        mode = 'kaolin'
        if mode == 'pytorch3d':
            raise NotImplementedError
            import pytorch3d.ops
            #raise NotImplementedError
            with torch.no_grad():

                smpl_def_on_mesh = self.shaped_smpl - posed_smpl  # [B, , 3]

                # find the nearest face in posed_smpl for each vertex in original_coordinate
                knn_res = pytorch3d.ops.knn_points(p1=original_coordinate, p2=posed_smpl, K=1)
                distance = knn_res[0] # [B, N, 1]
                p1_index = knn_res[1].repeat(1, 1, 3) # [B, N, 3]
                misc.assert_shape(p1_index, [original_coordinate.shape[0], original_coordinate.shape[1],3])


                DistToMesh = distance.squeeze(-1) # [B, N]

                SmplDef = smpl_def_on_mesh.gather(1, p1_index)  # [B, N, 3]
                mask = DistToMesh < self.thickness# [B, N]


                scale = 5.
                SmplDef1 = SmplDef / torch.exp(DistToMesh.unsqueeze(-1) * scale)  # [B, N, 3]

                scale = DistToMesh.unsqueeze(-1) / (self.thickness * 2) * 20
                SmplDef2 = torch.zeros_like(SmplDef).to(SmplDef.device)

                SmplDef = torch.where(mask.unsqueeze(-1), SmplDef1, SmplDef2)  # [B, N, 3]
        elif mode == 'kaolin':
            faces = self.aligned_SMPL.faces.clone()  # [20908, 3]
            faces = faces[self.face_mask, :]
            # find the nearest face in shaped_smplx for each vertex in original_coordinate
            vertex_faces = posed_smpl.clone()  # [B, 6085, 3]

            with torch.no_grad():
                face_vertices = index_vertices_by_faces(vertex_faces, faces)
                distance, index, dist_type = point_to_mesh_distance(original_coordinate, face_vertices)  # B, N
                distance = torch.sqrt(distance)  # [B, N, 1]
            selected_posed_smpl_vertices = []
            selected_shaped_smpl_vertices = []

            for i in range(original_coordinate.shape[0]):
                selected_face = faces[index[i]]
                selected_posed_smpl_vertices.append(index_vertices_by_faces(posed_smpl[i:i + 1],
                                                                            selected_face))  # [1, N, 3, 3]
                selected_shaped_smpl_vertices.append(index_vertices_by_faces(self.shaped_smpl[i:i + 1],
                                                                             selected_face))  # [1, N, 3, 3]

            selected_posed_smpl_vertices = torch.cat(selected_posed_smpl_vertices, dim=0)  # [B, N, 3, 3]
            selected_shaped_smpl_vertices = torch.cat(selected_shaped_smpl_vertices, dim=0)  # [B, N, 3, 3]

            y_axes = torch.cross(selected_posed_smpl_vertices[:, :, 1, :] - selected_posed_smpl_vertices[:, :, 0, :],
                                 selected_posed_smpl_vertices[:, :, 2, :] - selected_posed_smpl_vertices[:, :, 0,
                                                                            :])  # [B, N, 3]
            y_axes = y_axes / torch.norm(y_axes, dim=2, keepdim=True)  # [B, N, 3]

            x_axes = selected_posed_smpl_vertices[:, :, 1, :] - selected_posed_smpl_vertices[:, :, 0, :]  # [B, N, 3]
            x_axes = x_axes / torch.norm(x_axes, dim=2, keepdim=True)  # [B, N, 3]

            z_axes = torch.cross(x_axes, y_axes)  # [B, N, 3]

            posed_smpl_coordinate = torch.stack(
                [torch.sum((original_coordinate - selected_posed_smpl_vertices[:, :, 0, :]) * x_axes, dim=2),
                 torch.sum((original_coordinate - selected_posed_smpl_vertices[:, :, 0, :]) * y_axes, dim=2),
                 torch.sum((original_coordinate - selected_posed_smpl_vertices[:, :, 0, :]) * z_axes, dim=2)],
                dim=2)  # [B, N, 3]
            del x_axes, y_axes, z_axes
            y_axes = torch.cross(selected_shaped_smpl_vertices[:, :, 1, :] - selected_shaped_smpl_vertices[:, :, 0, :],
                                 selected_shaped_smpl_vertices[:, :, 2, :] - selected_shaped_smpl_vertices[:, :, 0, :])
            y_axes = y_axes / torch.norm(y_axes, dim=2, keepdim=True)

            x_axes = selected_shaped_smpl_vertices[:, :, 1, :] - selected_shaped_smpl_vertices[:, :, 0, :]
            x_axes = x_axes / torch.norm(x_axes, dim=2, keepdim=True)

            z_axes = torch.cross(x_axes, y_axes)

            new_coordinate = posed_smpl_coordinate[:, :, 0:1] * x_axes + \
                             posed_smpl_coordinate[:, :, 1:2] * y_axes + \
                             posed_smpl_coordinate[:, :, 2:3] * z_axes + \
                             selected_shaped_smpl_vertices[:, :, 0, :]  # [B, N, 3]

            SmplDef = new_coordinate - original_coordinate  # [B, N, 3]

            DistToMesh = distance.unsqueeze(-1)  # [B, N, 1]

            mask = DistToMesh < self.thickness  # [B, N,1]

            SmplDef2 = torch.zeros_like(SmplDef).to(SmplDef.device)
            SmplDef = torch.where(mask, SmplDef, SmplDef2)  # [B, N, 3]

        else:
            raise NotImplementedError

        original_coordinate = original_coordinate + SmplDef
        return original_coordinate

    def forward(self, planes, decoder, ray_origins, ray_directions, rendering_options, apply_def = False, ws = None, pose_params = None, triplane_crop=0.1, cull_clouds=None, binarize_clouds=None ):
        _ = ws
        if apply_def:
            assert pose_params is not None
        else:
            assert pose_params is None

        self.plane_axes = self.plane_axes.to(ray_origins.device)

        # check if grad = 0

        if rendering_options['ray_start'] == rendering_options['ray_end'] == 'auto':
            ray_start, ray_end = math_utils.get_ray_limits_box(ray_origins, ray_directions, box_side_length=rendering_options['box_warp'])
            is_ray_valid = ray_end > ray_start
            if torch.any(is_ray_valid).item():
                ray_start[~is_ray_valid] = ray_start[is_ray_valid].min()
                ray_end[~is_ray_valid] = ray_start[is_ray_valid].max()
            depths_coarse = self.sample_stratified(ray_origins, ray_start, ray_end, rendering_options['depth_resolution'], rendering_options['disparity_space_sampling'])
        else:
            # Create stratified depth samples
            depths_coarse = self.sample_stratified(ray_origins, rendering_options['ray_start'], rendering_options['ray_end'], rendering_options['depth_resolution'], rendering_options['disparity_space_sampling'])

        batch_size, num_rays, samples_per_ray, _ = depths_coarse.shape

        # Coarse Pass
        sample_coordinates = (ray_origins.unsqueeze(-2) + depths_coarse * ray_directions.unsqueeze(-2)).reshape(batch_size, -1, 3)
        sample_directions = ray_directions.unsqueeze(-2).expand(-1, -1, samples_per_ray, -1).reshape(batch_size, -1, 3)
        # deform the sample_coordinates
        if apply_def:
            sample_coordinates = self.get_deformed_coordinate(None, pose_params, sample_coordinates)


        out = self.run_model(planes, decoder, sample_coordinates, sample_directions, rendering_options)
        colors_coarse = out['rgb']
        densities_coarse = out['sigma']

        xyz_coarse = out['xyz']

        if triplane_crop:
            # print(xyz_fine.amin(dim=(0,1)))
            # print(xyz_fine.amax(dim=(0,1)))
            cropmask = triplane_crop_mask(xyz_coarse, triplane_crop, rendering_options['box_warp'])
            densities_coarse[cropmask] = -1e3
        if binarize_clouds:
            ccmask = cull_clouds_mask(densities_coarse, binarize_clouds)
            densities_coarse[ccmask] = -1e3
            densities_coarse[~ccmask] = 1e3
        elif cull_clouds:
            ccmask = cull_clouds_mask(densities_coarse, cull_clouds)
            densities_coarse[ccmask] = -1e3

        colors_coarse = colors_coarse.reshape(batch_size, num_rays, samples_per_ray, colors_coarse.shape[-1])
        densities_coarse = densities_coarse.reshape(batch_size, num_rays, samples_per_ray, 1)
        xyz_coarse = xyz_coarse.reshape(batch_size, num_rays, samples_per_ray, xyz_coarse.shape[-1])

        # Fine Pass
        N_importance = rendering_options['depth_resolution_importance']
        if N_importance > 0:
            _, _, weights = self.ray_marcher(colors_coarse, densities_coarse, depths_coarse, rendering_options)

            depths_fine = self.sample_importance(depths_coarse, weights, N_importance)

            sample_directions = ray_directions.unsqueeze(-2).expand(-1, -1, N_importance, -1).reshape(batch_size, -1, 3)
            sample_coordinates = (ray_origins.unsqueeze(-2) + depths_fine * ray_directions.unsqueeze(-2)).reshape(batch_size, -1, 3)
            # deform the sample_coordinates
            if apply_def:
                sample_coordinates = self.get_deformed_coordinate(None, pose_params, sample_coordinates)

            out = self.run_model(planes, decoder, sample_coordinates, sample_directions, rendering_options)
            colors_fine = out['rgb']
            densities_fine = out['sigma']
            xyz_fine = out['xyz']
            if triplane_crop:
                # print(xyz_fine.amin(dim=(0,1)))
                # print(xyz_fine.amax(dim=(0,1)))
                cropmask = triplane_crop_mask(xyz_fine, triplane_crop, rendering_options['box_warp'])
                densities_fine[cropmask] = -1e3
            if binarize_clouds:
                ccmask = cull_clouds_mask(densities_fine, binarize_clouds)
                densities_fine[ccmask] = -1e3
                densities_fine[~ccmask] = 1e3
            elif cull_clouds:
                ccmask = cull_clouds_mask(densities_fine, cull_clouds)
                densities_fine[ccmask] = -1e3
            xyz_fine = xyz_fine.reshape(batch_size, num_rays, N_importance, xyz_fine.shape[-1])
            colors_fine = colors_fine.reshape(batch_size, num_rays, N_importance, colors_fine.shape[-1])
            densities_fine = densities_fine.reshape(batch_size, num_rays, N_importance, 1)

            # all_depths, all_colors, all_densities = self.unify_samples(depths_coarse, colors_coarse, densities_coarse,
            #                                                       depths_fine, colors_fine, densities_fine)
            all_depths, all_colors, all_densities, all_xyz = self.unify_samples(
                depths_coarse, colors_coarse, densities_coarse, xyz_coarse,
                depths_fine, colors_fine, densities_fine, xyz_fine,
            )

            # Aggregate
            # rgb_final, depth_final, weights = self.ray_marcher(all_colors, all_densities, all_depths, rendering_options)

            all_colors_ = torch.cat([all_colors, all_xyz], dim=-1)
            rgb_final_, depth_final, weights = self.ray_marcher(all_colors_, all_densities, all_depths, rendering_options)
            rgb_final = rgb_final_[...,:-3]
            xyz_final = rgb_final_[...,-3:]
        else:
            # rgb_final, depth_final, weights = self.ray_marcher(colors_coarse, densities_coarse, depths_coarse, rendering_options)
            colors_coarse_ = torch.cat([colors_coarse, xyz_coarse], dim=-1)
            rgb_final_, depth_final, weights = self.ray_marcher(colors_coarse_, densities_coarse, depths_coarse, rendering_options)
            rgb_final = rgb_final_[...,:-3]
            xyz_final = rgb_final_[...,-3:]


        output = {'rgb_final': rgb_final, 'depth_final': depth_final, 'weights': weights}

        return output

    def run_model(self, planes, decoder, sample_coordinates, sample_directions, options):
        self.plane_axes = self.plane_axes.to(planes[list(planes.keys())[0]].device)
        sampled_features = sample_from_planes(self.plane_axes, planes, sample_coordinates, padding_mode='zeros',
                                              box_warp=options['box_warp'], triplane_depth=options['triplane_depth'])

        out = decoder(sampled_features, sample_directions)

        if options.get('density_noise', 0) > 0:
            out['sigma'] += torch.randn_like(out['sigma']) * options['density_noise']
        out['xyz'] = sample_coordinates#.permute(0,2,1)[...,None]
        return out

    def sort_samples(self, all_depths, all_colors, all_densities):
        _, indices = torch.sort(all_depths, dim=-2)
        all_depths = torch.gather(all_depths, -2, indices)
        all_colors = torch.gather(all_colors, -2, indices.expand(-1, -1, -1, all_colors.shape[-1]))
        all_densities = torch.gather(all_densities, -2, indices.expand(-1, -1, -1, 1))
        return all_depths, all_colors, all_densities

    # def unify_samples(self, depths1, colors1, densities1, depths2, colors2, densities2):
    #     all_depths = torch.cat([depths1, depths2], dim = -2)
    #     all_colors = torch.cat([colors1, colors2], dim = -2)
    #     all_densities = torch.cat([densities1, densities2], dim = -2)

    #     _, indices = torch.sort(all_depths, dim=-2)
    #     all_depths = torch.gather(all_depths, -2, indices)
    #     all_colors = torch.gather(all_colors, -2, indices.expand(-1, -1, -1, all_colors.shape[-1]))
    #     all_densities = torch.gather(all_densities, -2, indices.expand(-1, -1, -1, 1))

    #     return all_depths, all_colors, all_densities
    def unify_samples(self, depths1, colors1, densities1, xyz1, depths2, colors2, densities2, xyz2):
        all_depths = torch.cat([depths1, depths2], dim = -2)
        all_colors = torch.cat([colors1, colors2], dim = -2)
        all_xyz = torch.cat([xyz1, xyz2], dim = -2)
        all_densities = torch.cat([densities1, densities2], dim = -2)

        _, indices = torch.sort(all_depths, dim=-2)
        all_depths = torch.gather(all_depths, -2, indices)
        all_colors = torch.gather(all_colors, -2, indices.expand(-1, -1, -1, all_colors.shape[-1]))
        all_xyz = torch.gather(all_xyz, -2, indices.expand(-1, -1, -1, all_xyz.shape[-1]))
        all_densities = torch.gather(all_densities, -2, indices.expand(-1, -1, -1, 1))

        return all_depths, all_colors, all_densities, all_xyz

    def sample_stratified(self, ray_origins, ray_start, ray_end, depth_resolution, disparity_space_sampling=False):
        """
        Return depths of approximately uniformly spaced samples along rays.
        """
        N, M, _ = ray_origins.shape
        if disparity_space_sampling:
            depths_coarse = torch.linspace(0,
                                    1,
                                    depth_resolution,
                                    device=ray_origins.device).reshape(1, 1, depth_resolution, 1).repeat(N, M, 1, 1)
            depth_delta = 1/(depth_resolution - 1)
            depths_coarse += torch.rand_like(depths_coarse) * depth_delta
            depths_coarse = 1./(1./ray_start * (1. - depths_coarse) + 1./ray_end * depths_coarse)
        else:
            if type(ray_start) == torch.Tensor:
                depths_coarse = math_utils.linspace(ray_start, ray_end, depth_resolution).permute(1,2,0,3)
                depth_delta = (ray_end - ray_start) / (depth_resolution - 1)
                depths_coarse += torch.rand_like(depths_coarse) * depth_delta[..., None]
            else:
                depths_coarse = torch.linspace(ray_start, ray_end, depth_resolution, device=ray_origins.device).reshape(1, 1, depth_resolution, 1).repeat(N, M, 1, 1)
                depth_delta = (ray_end - ray_start)/(depth_resolution - 1)
                depths_coarse += torch.rand_like(depths_coarse) * depth_delta

        return depths_coarse

    def sample_importance(self, z_vals, weights, N_importance):
        """
        Return depths of importance sampled points along rays. See NeRF importance sampling for more.
        """
        with torch.no_grad():
            batch_size, num_rays, samples_per_ray, _ = z_vals.shape

            z_vals = z_vals.reshape(batch_size * num_rays, samples_per_ray)
            weights = weights.reshape(batch_size * num_rays, -1) # -1 to account for loss of 1 sample in MipRayMarcher

            # smooth weights
            weights = torch.nn.functional.max_pool1d(weights.unsqueeze(1).float(), 2, 1, padding=1)
            weights = torch.nn.functional.avg_pool1d(weights, 2, 1).squeeze()
            weights = weights + 0.01

            z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:])
            importance_z_vals = self.sample_pdf(z_vals_mid, weights[:, 1:-1],
                                             N_importance).detach().reshape(batch_size, num_rays, N_importance, 1)
        return importance_z_vals

    def sample_pdf(self, bins, weights, N_importance, det=False, eps=1e-5):
        """
        Sample @N_importance samples from @bins with distribution defined by @weights.
        Inputs:
            bins: (N_rays, N_samples_+1) where N_samples_ is "the number of coarse samples per ray - 2"
            weights: (N_rays, N_samples_)
            N_importance: the number of samples to draw from the distribution
            det: deterministic or not
            eps: a small number to prevent division by zero
        Outputs:
            samples: the sampled samples
        """
        N_rays, N_samples_ = weights.shape
        weights = weights + eps # prevent division by zero (don't do inplace op!)
        pdf = weights / torch.sum(weights, -1, keepdim=True) # (N_rays, N_samples_)
        cdf = torch.cumsum(pdf, -1) # (N_rays, N_samples), cumulative distribution function
        cdf = torch.cat([torch.zeros_like(cdf[: ,:1]), cdf], -1)  # (N_rays, N_samples_+1)
                                                                   # padded to 0~1 inclusive

        if det:
            u = torch.linspace(0, 1, N_importance, device=bins.device)
            u = u.expand(N_rays, N_importance)
        else:
            u = torch.rand(N_rays, N_importance, device=bins.device)
        u = u.contiguous()

        inds = torch.searchsorted(cdf, u, right=True)
        below = torch.clamp_min(inds-1, 0)
        above = torch.clamp_max(inds, N_samples_)

        inds_sampled = torch.stack([below, above], -1).view(N_rays, 2*N_importance)
        cdf_g = torch.gather(cdf, 1, inds_sampled).view(N_rays, N_importance, 2)
        bins_g = torch.gather(bins, 1, inds_sampled).view(N_rays, N_importance, 2)

        denom = cdf_g[...,1]-cdf_g[...,0]
        denom[denom<eps] = 1 # denom equals 0 means a bin has weight 0, in which case it will not be sampled
                             # anyway, therefore any value for it is fine (set to 1 here)

        samples = bins_g[...,0] + (u-cdf_g[...,0])/denom * (bins_g[...,1]-bins_g[...,0])
        return samples
