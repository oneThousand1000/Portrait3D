import os
import cv2
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl
import trimesh
#os.environ['PYOPENGL_PLATFORM'] = 'egl' # comment it if use windows
import pyrender


def vis_bbox(img, bbox, alpha=1):

    kp_mask = np.copy(img)
    bbox = bbox.astype(np.int32) # x, y, w, h

    b1 = bbox[0], bbox[1]
    b2 = bbox[0] + bbox[2], bbox[1]
    b3 = bbox[0] + bbox[2], bbox[1] + bbox[3]
    b4 = bbox[0], bbox[1] + bbox[3]

    cv2.line(kp_mask, b1, b2, color=(255, 255, 0), thickness=1, lineType=cv2.LINE_AA)
    cv2.line(kp_mask, b2, b3, color=(255, 255, 0), thickness=1, lineType=cv2.LINE_AA)
    cv2.line(kp_mask, b3, b4, color=(255, 255, 0), thickness=1, lineType=cv2.LINE_AA)
    cv2.line(kp_mask, b4, b1, color=(255, 255, 0), thickness=1, lineType=cv2.LINE_AA)

    return cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)


def vis_coco_skeleton(img, kps, kps_lines, alpha=1):
    colors = [
            # face
            (255/255, 153/255, 51/255),
            (255/255, 153/255, 51/255),
            (255/255, 153/255, 51/255),
            (255/255, 153/255, 51/255),

            # left arm
            (102/255, 255/255, 102/255),
            (51/255, 255/255, 51/255),

            # right leg
            (255 / 255, 102 / 255, 255 / 255),
            (255 / 255, 51 / 255, 255 / 255),


            # left leg

            (255 / 255, 102 / 255, 102 / 255),
            (255 / 255, 51 / 255, 51 / 255),

            # shoulder-thorax, hip-pevlis,
            (153/255, 255/255, 153/255), # l shoulder - thorax
            (153/255, 204/255, 255/255), # r shoulder - thorax
            (255/255, 153/255, 153/255), # l hip - pelvis
            (255/255, 153/255, 255/255), # r hip -pelvis

            # center body line
            (255/255, 204/255, 153/255),
            (255/255, 178/255, 102/255),

            # right arm
            (102 / 255, 178 / 255, 255 / 255),
            (51 / 255, 153 / 255, 255 / 255),
            ]

    colors = [[c[2]*255,c[1]*255,c[0]*255] for c in colors]

    # Perform the drawing on a copy of the image, to allow for blending.
    kp_mask = np.copy(img)

    line_thick = 5 #13
    circle_rad = 5 #10
    circle_thick = 5 #7

    # Draw the keypoints.
    for l in range(len(kps_lines)):
        i1 = kps_lines[l][0]
        i2 = kps_lines[l][1]
        p1 = kps[0, i1].astype(np.int32), kps[1, i1].astype(np.int32)
        p2 = kps[0, i2].astype(np.int32), kps[1, i2].astype(np.int32)
        cv2.line(
            kp_mask, p1, p2,
            color=colors[l], thickness=line_thick, lineType=cv2.LINE_AA)
        cv2.circle(
            kp_mask, p1,
            radius=circle_rad, color=colors[l], thickness=circle_thick, lineType=cv2.LINE_AA)
        cv2.circle(
            kp_mask, p2,
            radius=circle_rad, color=colors[l], thickness=circle_thick, lineType=cv2.LINE_AA)

    # Blend the keypoints.
    return cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)


def vis_keypoints_with_skeleton(img, kps, kps_lines, kp_thresh=0.4, alpha=1, kps_scores=None):
    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kps_lines) + 2)]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    # Perform the drawing on a copy of the image, to allow for blending.
    kp_mask = np.copy(img)

    # Draw the keypoints.
    for l in range(len(kps_lines)):
        i1 = kps_lines[l][0]
        i2 = kps_lines[l][1]
        p1 = kps[0, i1].astype(np.int32), kps[1, i1].astype(np.int32)
        p2 = kps[0, i2].astype(np.int32), kps[1, i2].astype(np.int32)
        if kps[2, i1] > kp_thresh and kps[2, i2] > kp_thresh:
            cv2.line(
                kp_mask, p1, p2,
                color=colors[l], thickness=2, lineType=cv2.LINE_AA)
        if kps[2, i1] > kp_thresh:
            cv2.circle(
                kp_mask, p1,
                radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)
            cv2.putText(kp_mask, str([l][0]), p1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[l])
        if kps[2, i2] > kp_thresh:
            cv2.circle(
                kp_mask, p2,
                radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)
            cv2.putText(kp_mask, str([l][1]), p2, cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[l])
            if kps_scores is not None:
                cv2.putText(kp_mask, str(kps_scores[i2, 0]), p2, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

    # Blend the keypoints.
    return cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)

def vis_keypoints(img, kps, alpha=1, kps_vis=None):
    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kps) + 2)]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    # Perform the drawing on a copy of the image, to allow for blending.
    kp_mask = np.copy(img)

    # Draw the keypoints.
    for i in range(len(kps)):
        p = kps[i][0].astype(np.int32), kps[i][1].astype(np.int32)
        cv2.circle(kp_mask, p, radius=3, color=colors[i], thickness=-1, lineType=cv2.LINE_AA)
        if kps_vis is not None:
            cv2.putText(kp_mask, str(kps_vis[i, 0]), p, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
        else:
            cv2.putText(kp_mask, str(i), p, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

    # Blend the keypoints.
    return cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)

def vis_mesh(img, mesh_vertex, alpha=0.5):
    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(mesh_vertex))]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    # Perform the drawing on a copy of the image, to allow for blending.
    mask = np.copy(img)

    # Draw the mesh
    for i in range(len(mesh_vertex)):
        p = mesh_vertex[i][0].astype(np.int32), mesh_vertex[i][1].astype(np.int32)
        cv2.circle(mask, p, radius=1, color=colors[i], thickness=-1, lineType=cv2.LINE_AA)

    # Blend the keypoints.
    return cv2.addWeighted(img, 1.0 - alpha, mask, alpha, 0)

def vis_3d_skeleton(kpt_3d, kpt_3d_vis, kps_lines, filename=None):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kps_lines) + 2)]
    colors = [np.array((c[2], c[1], c[0])) for c in colors]

    for l in range(len(kps_lines)):
        i1 = kps_lines[l][0]
        i2 = kps_lines[l][1]
        x = np.array([kpt_3d[i1,0], kpt_3d[i2,0]])
        y = np.array([kpt_3d[i1,1], kpt_3d[i2,1]])
        z = np.array([kpt_3d[i1,2], kpt_3d[i2,2]])

        if kpt_3d_vis[i1,0] > 0 and kpt_3d_vis[i2,0] > 0:
            ax.plot(x, z, -y, c=colors[l], linewidth=2)
        if kpt_3d_vis[i1,0] > 0:
            ax.scatter(kpt_3d[i1,0], kpt_3d[i1,2], -kpt_3d[i1,1], c=colors[l], marker='o')
        if kpt_3d_vis[i2,0] > 0:
            ax.scatter(kpt_3d[i2,0], kpt_3d[i2,2], -kpt_3d[i2,1], c=colors[l], marker='o')

    if filename is None:
        ax.set_title('3D vis')
    else:
        ax.set_title(filename)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Z Label')
    ax.set_zlabel('Y Label')
    ax.legend()

    plt.show()
    cv2.waitKey(0)

def save_obj(v, f, file_name='output.obj'):
    obj_file = open(file_name, 'w')
    for i in range(len(v)):
        obj_file.write('v ' + str(v[i][0]) + ' ' + str(v[i][1]) + ' ' + str(v[i][2]) + '\n')
    for i in range(len(f)):
        obj_file.write('f ' + str(f[i][0]+1) + '/' + str(f[i][0]+1) + ' ' + str(f[i][1]+1) + '/' + str(f[i][1]+1) + ' ' + str(f[i][2]+1) + '/' + str(f[i][2]+1) + '\n')
    obj_file.close()

def render_mesh(img, mesh, face, cam_param, color=(1.0, 1.0, 0.9, 1.0),cam_pose = None):
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
    light_pose[:3, 3] = np.array([0.2469,  1.8828, -2.4473])
    scene.add(light, pose=light_pose)

    # render
    rgb, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    rgb = rgb[:, :, :3].astype(np.float32)
    valid_mask = (depth > 0)[:, :, None]

    # save to image
    img = rgb * valid_mask + img * (1 - valid_mask)
    return img.astype(np.uint8)