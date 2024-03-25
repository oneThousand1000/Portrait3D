
import os.path as osp

import numpy as np
import torch
from nerf.torch_utils import misc

import trimesh
import pickle


import os
# os.environ["PYOPENGL_PLATFORM"] = "egl"
# os.environ["PYOPENGL_PLATFORM"] = "osmesa"
import pyrender

class AlignedSMPL(torch.nn.Module):
    def __init__(self, model,batch_size):
        super().__init__()
        self.batch_size = batch_size
        smpl_joint_regressor = torch.from_numpy(
            np.load('transfer_data/smpl_joint_regressor.npy')).float().cuda().contiguous()
        self.register_buffer('smpl_joint_regressor', smpl_joint_regressor)

        self.model = model
        faces = torch.from_numpy(self.model.faces.astype(np.int32)).cuda().long().contiguous()
        self.register_buffer('faces', faces)
        
        
    def set_model(self, model):
        self.model = model
    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def get_align_coordinate(self, vertices):
        # 30 x 6890
        batch_size = vertices.shape[0]
        smpl_joints = torch.bmm(self.smpl_joint_regressor[None, :, :].repeat(batch_size, 1, 1), vertices)
        align_joint_coordinate = smpl_joints[:,12, None, :]  # Neck
        return align_joint_coordinate

    def render_mesh(self, img, mesh, face, cam_param, color=(1.0, 1.0, 0.9, 1.0), cam_pose=None):
        # mesh
        mesh = trimesh.Trimesh(mesh, face)
        rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
        mesh.apply_transform(rot)
        material = pyrender.MetallicRoughnessMaterial(metallicFactor=0.0, alphaMode='OPAQUE', baseColorFactor=color)
        mesh = pyrender.Mesh.from_trimesh(mesh, material=material, smooth=False)
        scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=(0.3, 0.3, 0.3))
        scene.add(mesh, 'mesh')

        focal, princpt = cam_param['focal'], cam_param['princpt']
        camera = pyrender.IntrinsicsCamera(fx=focal[0], fy=focal[1], cx=princpt[0], cy=princpt[1])

        if cam_pose is not None:
            scene.add(camera, pose=cam_pose)
        else:
            scene.add(camera)
        # scene.add(camera)
        # print('camera pose in scene ', scene.get_pose(scene._main_camera_node))
        # renderer
        renderer = pyrender.OffscreenRenderer(viewport_width=img.shape[1], viewport_height=img.shape[0], point_size=1.0)

        # light
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=0.8)
        # light_pose = np.eye(4)
        # light_pose[:3, 3] = np.array([0, -1, 1])
        # scene.add(light, pose=light_pose)
        # light_pose[:3, 3] = np.array([0, 1, 1])
        # scene.add(light, pose=light_pose)
        # light_pose[:3, 3] = np.array([1, 1, 2])
        # scene.add(light, pose=light_pose)

        light_pose = np.eye(4)
        light_pose[:3, 3] = np.array([0, 0, -1])
        scene.add(light, pose=light_pose)

        scene.add(light, pose=cam_pose)
        scene.add(light, pose=cam_pose)
        scene.add(light, pose=cam_pose)
        light_pose[:3, 3] = np.array([1, 1, -4])
        scene.add(light, pose=light_pose)
        light_pose[:3, 3] = np.array([-1, 0, -1])
        scene.add(light, pose=light_pose)
        light_pose[:3, 3] = np.array([0.2469, 1.8828, -2.4473])
        scene.add(light, pose=light_pose)

        # render
        rgb, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        rgb = rgb[:, :, :3].astype(np.float32)
        valid_mask = (depth > 0)[:, :, None]

        # save to image
        img = rgb * valid_mask + img * (1 - valid_mask)
        return img.astype(np.uint8)

    def render_depth(self, img, mesh, face, cam_param, color=(1.0, 1.0, 0.9, 1.0), cam_pose=None):
        # mesh
        mesh = trimesh.Trimesh(mesh, face)
        rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
        mesh.apply_transform(rot)
        material = pyrender.MetallicRoughnessMaterial(metallicFactor=0.0, alphaMode='OPAQUE', baseColorFactor=color)
        mesh = pyrender.Mesh.from_trimesh(mesh, material=material, smooth=False)
        scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=(0.3, 0.3, 0.3))
        scene.add(mesh, 'mesh')

        focal, princpt = cam_param['focal'], cam_param['princpt']
        camera = pyrender.IntrinsicsCamera(fx=focal[0], fy=focal[1], cx=princpt[0], cy=princpt[1])

        if cam_pose is not None:
            scene.add(camera, pose=cam_pose)
        else:
            scene.add(camera)
        # scene.add(camera)
        # print('camera pose in scene ', scene.get_pose(scene._main_camera_node))
        # renderer
        renderer = pyrender.OffscreenRenderer(viewport_width=img.shape[1], viewport_height=img.shape[0], point_size=1.0)

        # render
        rgb, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        #rgb = rgb[:, :, :3].astype(np.float32)
        valid_mask = (depth > 0)[:, :, None]

        # save to image
        depth = depth * valid_mask + img * (1 - valid_mask)
        return depth.astype(np.uint8)


    def get_projected_vertex(self, mesh, world2screen_matrix):
        # mesh = np.concatenate([mesh, np.ones((mesh.shape[0], 1))], axis=1)  # N x 4
        mesh = torch.cat([mesh, torch.ones((mesh.shape[0], 1)).to(mesh.device)], dim=1)  # N x 4
        points_image = world2screen_matrix @ mesh.T  # 4,N
        points_image = points_image[:3, :]  # 3,N

        points_on_input_image = points_image / points_image[2, :]
        points_on_input_image = points_on_input_image[:2, :].T  # 30,2

        return points_on_input_image


    def generate_shaped_smpl(self, betas, scale, transl):
        if betas is not None:
            raise NotImplementedError
        else:
            betas = None
        if scale is not None:
            raise NotImplementedError
            misc.assert_shape(scale, [self.batch_size, 1])
        else:
            scale = torch.ones([self.batch_size, 1]).to(self.model.shapedirs.device)
        if transl is not None:
            raise NotImplementedError
            misc.assert_shape(transl, [self.batch_size, 3])
        else:
            transl = torch.zeros([self.batch_size, 3]).to(self.model.shapedirs.device)

        # body_pose_fill = torch.zeros((self.batch_size, 23, 3)).to(self.model.shapedirs.device)
        # # 15 16 for shoulder, we hope the Hands naturally sagging
        # body_pose_fill[:, 15, :] = torch.tensor([0.0, 0.0, -np.pi / 2]).to(self.model.shapedirs.device)

        # body_pose_fill[:, 16, :] = torch.tensor([0.0, 0.0, np.pi / 2]).to(self.model.shapedirs.device)
        # body_pose_fill = body_pose_fill.reshape(self.batch_size, -1)
        # apply beta, alignment, translation and scale
        shaed_output = self.model(betas=betas,
                                  expression=None,
                                  return_verts=True,
                                  body_pose=None,
                                  return_shaped=False
                                  )
        vertices_no_pose = shaed_output.vertices
        joints_no_pose = shaed_output.joints


        align_joint_coordinate = self.get_align_coordinate(vertices_no_pose) # B,1,3
        vertices_no_pose -= align_joint_coordinate
        joints_no_pose -= align_joint_coordinate

        vertices_no_pose += transl.view(self.batch_size, 1, 3)
        joints_no_pose += transl.view(self.batch_size, 1, 3)

        vertices_no_pose *= scale.view(self.batch_size, 1, 1)
        joints_no_pose *= scale.view(self.batch_size, 1, 1)

        nose_2d = joints_no_pose[:,86:90,:]  # B, 4, 3
        eye_right_2d = joints_no_pose[:,95: 101,:]  # B, 6, 3
        eye_left_2d = joints_no_pose[:,101: 107,:]  # B, 6, 3

        # points_3d = np.concatenate([nose_2d, eye_right_2d, eye_left_2d], axis=0)  # 16
        face_points = torch.cat([nose_2d, eye_right_2d, eye_left_2d], dim=1)  # B, 16, 3

        #transformation_matrix = self.compute_transformation_matrix(face_points)

        res = {
            'vertices': vertices_no_pose,
            'align_joint_coordinate': align_joint_coordinate,
            'face_points': face_points,
        }
        return res

    def generate_posed_smpl(self, betas, scale, transl, body_pose, align_joint_coordinate):
        batch_size = body_pose.shape[0]
        if betas is not None:
            raise NotImplementedError
        else:
            betas = None
        if scale is not None:
            raise NotImplementedError
            misc.assert_shape(scale, [self.batch_size, 1])
        else:
            scale = torch.ones([self.batch_size, 1]).to(self.model.shapedirs.device)
        if transl is not None:
            raise NotImplementedError
            misc.assert_shape(transl, [self.batch_size, 3])
        else:
            transl = torch.zeros([self.batch_size, 3]).to(self.model.shapedirs.device)
        misc.assert_shape(body_pose, [self.batch_size, 6])

        # apply beta, alignment, translation and scale

        # apply beta, pose, alignment, translation and scale
        # mask pose except 11 and 14
        body_pose_fill = torch.zeros((self.batch_size, 23, 3)).to(self.model.shapedirs.device)
        body_pose_fill[:, 11, :] = body_pose[:, :3]
        body_pose_fill[:, 14, :] = body_pose[:, 3:]

        # # 15 16 for shoulder, we hope the Hands naturally sagging
        # body_pose_fill[:, 15, :] = torch.tensor([0.0, 0.0, -np.pi / 2]).to(self.model.shapedirs.device)
        # body_pose_fill[:, 16, :] = torch.tensor([0.0, 0.0, np.pi / 2]).to(self.model.shapedirs.device)


        body_pose_fill = body_pose_fill.reshape(self.batch_size, -1)

        output = self.model(betas=betas,
                            expression=None,
                            return_verts=True,
                            body_pose=body_pose_fill,
                            return_shaped=True
                            )
        vertices = output.vertices
        joints = output.joints

        # align vertices and joints
        vertices -= align_joint_coordinate
        joints -= align_joint_coordinate

        # additional translation
        vertices += transl.view(self.batch_size, 1, 3)
        joints += transl.view(self.batch_size, 1, 3)

        # additional scale
        vertices *= scale.view(self.batch_size, 1, 1)
        joints *= scale.view(self.batch_size, 1, 1)

        nose_2d = joints[:, 86:90, :]  # B, 4, 3
        eye_right_2d = joints[:, 95: 101, :]  # B, 6, 3
        eye_left_2d = joints[:, 101: 107, :]  # B, 6, 3

        # points_3d = np.concatenate([nose_2d, eye_right_2d, eye_left_2d], axis=0)  # 16
        face_points = torch.cat([nose_2d, eye_right_2d, eye_left_2d], dim=1)  # B, 16, 3

        res = {
            'vertices': vertices,
            'face_points': face_points
        }

        return res



    def get_depth(self,vert, resolution=256, cameras=None):

        faces = self.model.faces
        # compute the transformation matrix with eg3d
        intrisics_standard_dict = {"focal": [5000.0 / 1024 * resolution / 0.75, 5000.0 / 1024 * resolution / 0.75],
                                   "princpt": [resolution / 2, resolution / 2]}
        # intrisics_standard = np.array( [[5000.0, 0.0, resolution/2, 0.0], [0.0, 5000.0, resolution/2.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
        # normalized_transformation_in_realworld = np.array(render_kwargs['world2camera_matrix'])
        R = np.eye(3)
        angle = np.pi
        R[1, 1] = np.cos(angle)
        R[1, 2] = -np.sin(angle)
        R[2, 1] = np.sin(angle)
        R[2, 2] = np.cos(angle)

        R = torch.from_numpy(R).float().to(self.model.shapedirs.device).unsqueeze(0).repeat(self.batch_size, 1,
                                                                                            1)  # self.batch_size x 3 x 3

        vertices_pyrender = torch.matmul(vert, R)  # 1 x 6890 x 3
        # normalized_camerapose_in_pyrender = np.array(render_kwargs['normalized_camerapose_in_pyrender'])

        # color = colorsys.hsv_to_rgb(np.random.rand(), 0.5, 1.0)
        images = []
        for i in range(self.batch_size):
            camera_pose = cameras[i, :16].reshape(4, 4)

            camerapose_in_pyrender = np.linalg.inv(camera_pose)
            camerapose_in_pyrender[[1, 2]] *= -1
            camerapose_in_pyrender = np.linalg.inv(camerapose_in_pyrender)

            # print(vertices_pyrender.shape, vertices_pyrender[i].shape,camerapose_in_pyrender.shape)
            image_camera_rotate = self.render_depth(np.ones((resolution, resolution, 3)) * 255,
                                                   vertices_pyrender[i].detach().cpu().numpy(), faces,
                                                   intrisics_standard_dict,
                                                   color=(0.4, 0.5, 0.9, 1.0),
                                                   cam_pose=camerapose_in_pyrender)

            image_camera_rotate = image_camera_rotate[None, :, :, :]  # 1 x 256 x 256 x 3
            image_camera_rotate = np.transpose(image_camera_rotate, (0, 3, 1, 2))  # 1 x 3 x 256 x 256
            images.append(image_camera_rotate)

        images = np.concatenate(images, axis=0)
        return images
    #
    def get_visualization(self, shape_pose_params, resolution=256, cameras=None):
        # apply beta, alignment, translation and scale
        if 'betas' in shape_pose_params:
            raise NotImplementedError
            betas = shape_pose_params['betas']
            misc.assert_shape(betas, [self.batch_size, self.num_betas])
        else:
            betas = None
        # scale = shape_pose_params['scale']
        # transl = shape_pose_params['transl']
        if 'scale' in shape_pose_params:
            raise NotImplementedError
            scale = shape_pose_params['scale']
            misc.assert_shape(scale, [self.batch_size, 1])
        else:
            scale = torch.ones([self.batch_size, 1]).to(self.model.shapedirs.device)
        if 'transl' in shape_pose_params:
            raise NotImplementedError
            transl = shape_pose_params['transl']
            misc.assert_shape(transl, [self.batch_size, 3])
        else:
            transl = torch.zeros([self.batch_size, 3]).to(self.model.shapedirs.device)


        body_pose = shape_pose_params['pose']


        misc.assert_shape(scale, [self.batch_size, 1])
        misc.assert_shape(transl, [self.batch_size, 3])
        misc.assert_shape(body_pose, [self.batch_size, 6])

        cameras = cameras.detach().cpu().numpy() # N, 25

        shaed_output = self.model(betas=betas,
                                  expression=None,
                                  return_verts=True,
                                  body_pose=None,
                                  return_shaped=False
                                  )
        vertices_no_pose = shaed_output.vertices
        faces = self.model.faces

        align_joint_coordinate = self.get_align_coordinate(vertices_no_pose)
        vertices_no_pose = vertices_no_pose
        vertices_no_pose -= align_joint_coordinate

        vertices_no_pose += transl.view(self.batch_size, 1, 3)
        vertices_no_pose *= scale.view(self.batch_size, 1, 1)

        # apply beta, pose, alignment, translation and scale
        # mask pose except 11 and 14
        body_pose_fill = torch.zeros((self.batch_size, 23, 3)).to(self.model.shapedirs.device)
        body_pose_fill[:, 11, :] = body_pose[:, :3]
        body_pose_fill[:, 14, :] = body_pose[:, 3:]

        # # 15 16 for shoulder, we hope the Hands naturally sagging
        # body_pose_fill[:, 15, :] = torch.tensor([0.0, 0.0, -np.pi / 2]).to(self.model.shapedirs.device)
        # body_pose_fill[:, 16, :] = torch.tensor([0.0, 0.0, np.pi / 2]).to(self.model.shapedirs.device)



        body_pose_fill = body_pose_fill.reshape(self.batch_size, -1)

        output = self.model(betas=betas,
                            expression=None,
                            return_verts=True,
                            body_pose=body_pose_fill,
                            return_shaped=True
                            )
        vertices = output.vertices
        joints = output.joints

        # align vertices and joints
        vertices -= align_joint_coordinate
        joints -= align_joint_coordinate

        # additional translation
        vertices += transl.view(self.batch_size, 1, 3)
        joints += transl.view(self.batch_size, 1, 3)

        # additional scale
        vertices *= scale.view(self.batch_size, 1, 1)
        joints *= scale.view(self.batch_size, 1, 1)

        # print(vertices[:,0].min(),vertices[:,0].max(),vertices[:,0].max() - vertices[:,0].min())
        # print(vertices[:,1].min(),vertices[:,1].max(),vertices[:,1].max() - vertices[:,1].min())
        # print(vertices[:,2].min(),vertices[:,2].max(),vertices[:,2].max() - vertices[:,2].min())

        # nose_2d = joints[86:90]  # 4
        # eye_right_2d = joints[95: 101]  # 6
        # eye_left_2d = joints[101: 107]  # 6

        #points_3d = np.concatenate([nose_2d, eye_right_2d, eye_left_2d], axis=0)  # 16
        #points_3d = torch.cat([nose_2d, eye_right_2d, eye_left_2d], dim=0)  # 16

        # compute the transformation matrix with eg3d
        intrisics_standard_dict = {"focal": [5000.0/1024*resolution/0.75, 5000.0/1024*resolution/0.75], "princpt": [resolution/2, resolution/2]}
        # intrisics_standard = np.array( [[5000.0, 0.0, resolution/2, 0.0], [0.0, 5000.0, resolution/2.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
        # normalized_transformation_in_realworld = np.array(render_kwargs['world2camera_matrix'])
        R = np.eye(3)
        angle = np.pi
        R[1, 1] = np.cos(angle)
        R[1, 2] = -np.sin(angle)
        R[2, 1] = np.sin(angle)
        R[2, 2] = np.cos(angle)

        R = torch.from_numpy(R).float().to(self.model.shapedirs.device).unsqueeze(0).repeat(self.batch_size, 1, 1) # self.batch_size x 3 x 3

        vertices_pyrender = torch.matmul(vertices, R) # 1 x 6890 x 3
        #normalized_camerapose_in_pyrender = np.array(render_kwargs['normalized_camerapose_in_pyrender'])

        # color = colorsys.hsv_to_rgb(np.random.rand(), 0.5, 1.0)
        images = []
        for i in range(self.batch_size):
            camera_pose = cameras[i,:16].reshape(4,4)

            camerapose_in_pyrender = np.linalg.inv(camera_pose)
            camerapose_in_pyrender[[1,2]] *= -1
            camerapose_in_pyrender = np.linalg.inv(camerapose_in_pyrender)

            #print(vertices_pyrender.shape, vertices_pyrender[i].shape,camerapose_in_pyrender.shape)
            image_camera_rotate = self.render_mesh(np.ones((resolution, resolution, 3)) * 255,
                                                   vertices_pyrender[i].detach().cpu().numpy(), faces,
                                                   intrisics_standard_dict,
                                                   color=(0.4, 0.5, 0.9, 1.0),
                                                   cam_pose=camerapose_in_pyrender)

            image_camera_rotate = image_camera_rotate[None, :, :, :] # 1 x 256 x 256 x 3
            image_camera_rotate = np.transpose(image_camera_rotate, (0, 3, 1, 2)) # 1 x 3 x 256 x 256
            images.append(image_camera_rotate)

        images = np.concatenate(images, axis=0)
        return images
