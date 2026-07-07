#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
import torch
import random
import torch.nn.functional as F
from PIL import Image
from typing import NamedTuple
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from utils.pointe_utils import init_from_pointe
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from utils.general_utils import inverse_sigmoid_np
from scene.gaussian_model import BasicPointCloud
import utils3d
from scipy.spatial.transform import Rotation as R
from trellis.representations.gaussian import Gaussian
from LGM.gaussian_model import GaussianRenderer

class RandCameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    width: int
    height: int
    delta_polar: np.array
    delta_azimuth: np.array
    delta_radius: np.array
    c2w: np.array


class TrellisGaussians(NamedTuple):
    xyz: torch.Tensor
    features_dc: torch.Tensor
    scaling: torch.Tensor
    rotation: torch.Tensor
    opacity: torch.Tensor

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str


class RSceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    test_cameras: list
    ply_path: str


class GraphSceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    test_cameras: list
    points_per_obj: list
    volumes: int
    num_objs: int
    ply_path: str

def fetchPly(path, num_objs):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'],
                       vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions.reshape(num_objs, -1, 3), colors=colors.reshape(num_objs, -1, 3), normals=normals.reshape(num_objs, -1, 3))

def loadPly(path, num_points):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T

    # Center and scale to unit sphere
    positions = positions - positions.mean(axis=0)
    max_norm = np.max(np.linalg.norm(positions, axis=1))
    if max_norm > 0:
        positions = positions / max_norm  # normalize to unit radius

    n_pts = positions.shape[0]

    if n_pts >= num_points:
        indices = np.random.choice(n_pts, num_points, replace=False)
        return positions[indices]
    else:
        # Need to upsample: duplicate + jitter in small sphere
        repeat_times = int(np.ceil(num_points / n_pts))
        positions_tiled = np.tile(positions, (repeat_times, 1))[:num_points]

        # Jitter points in a small sphere
        thetas = np.random.rand(num_points) * np.pi
        phis = np.random.rand(num_points) * 2 * np.pi
        radius = np.random.rand(num_points) * 0.05

        jitter = np.stack([
            radius * np.sin(thetas) * np.sin(phis),
            radius * np.sin(thetas) * np.cos(phis),
            radius * np.cos(thetas),
        ], axis=-1)

        return (positions_tiled + jitter)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def rotate_point_cloud(point_cloud, angles):
    angle_x, angle_y, angle_z = angles

    # Rotation matrix around the X-axis
    rotation_x = np.array([
        [1, 0, 0],
        [0, np.cos(angle_x), -np.sin(angle_x)],
        [0, np.sin(angle_x), np.cos(angle_x)]
    ])

    # Rotation matrix around the Y-axis
    rotation_y = np.array([
        [np.cos(angle_y), 0, np.sin(angle_y)],
        [0, 1, 0],
        [-np.sin(angle_y), 0, np.cos(angle_y)]
    ])

    # Rotation matrix around the Z-axis
    rotation_z = np.array([
        [np.cos(angle_z), -np.sin(angle_z), 0],
        [np.sin(angle_z), np.cos(angle_z), 0],
        [0, 0, 1]
    ])

    # Combined rotation matrix
    # Order of multiplication: Z -> Y -> X
    rotation_matrix = rotation_z @ rotation_y @ rotation_x

    # Apply the rotation matrix to the point cloud
    if isinstance(point_cloud, torch.Tensor):
        point_cloud = point_cloud.cpu().numpy()
    return np.dot(point_cloud, rotation_matrix.T)

def zero_pad_tensor(tensor_list, pad_size, num_objs):
    x = list(tensor_list[0].shape)
    x[0] = pad_size
    for i in range(num_objs):
        xyz_pad = torch.zeros((x), device=tensor_list[i].device, dtype=tensor_list[i].dtype)
        xyz_pad[:tensor_list[i].shape[0]] = tensor_list[i]
        tensor_list[i] = xyz_pad
    
    return torch.cat(tensor_list, dim=0)


def readCircleCamInfo(path, opt):
    print("Reading Test Transforms")
    test_cam_infos = GenerateCircleCameras(opt, render45=opt.render_45)
    ply_path = os.path.join(path, "init_points3d.ply")
    lengths_path = os.path.join(path, "init_points3d_length.npy")
    volumes_path = os.path.join(path, "inint_points3d_volume.npy")
    num_pts = opt.init_num_pts
    num_objs = opt.num_objs
    reinit = opt.init_shape == "mix"

    if reinit or not os.path.exists(ply_path) or not os.path.exists(lengths_path) or not os.path.exists(volumes_path):
        # Since this data set has no colmap data, we start with random points
        points = num_pts // num_objs
        if opt.init_shape == 'sphere':
            xyz = []
            lengths = []
            for i in range(num_objs):
                thetas = np.random.rand(points)*np.pi
                phis = np.random.rand(points)*2*np.pi
                radius = np.random.rand(points)*opt.radius_params[i]
                # We create random points inside the bounds of sphere
                obj_xyz = np.stack([
                    radius * np.sin(thetas) * np.sin(phis),
                    radius * np.sin(thetas) * np.cos(phis),
                    radius * np.cos(thetas),
                ], axis=-1)  # [B, 3]
                obj_xyz = obj_xyz + opt.center_params[i]
                xyz.append(obj_xyz)
                lengths.append(points)
            xyz = np.array(xyz)
        elif opt.init_shape == 'box':
            xyz = np.random.random((num_pts, 3)) * 1.0 - 0.5
        elif opt.init_shape == 'rectangle_x':
            xyz = np.random.random((num_pts, 3))
            xyz[:, 0] = xyz[:, 0] * 0.6 - 0.3
            xyz[:, 1] = xyz[:, 1] * 1.2 - 0.6
            xyz[:, 2] = xyz[:, 2] * 0.5 - 0.25
        elif opt.init_shape == 'rectangle_z':
            xyz = np.random.random((num_pts, 3))
            xyz[:, 0] = xyz[:, 0] * 0.8 - 0.4
            xyz[:, 1] = xyz[:, 1] * 0.6 - 0.3
            xyz[:, 2] = xyz[:, 2] * 1.2 - 0.6
        elif opt.init_shape == 'pointe':
            xyz = []
            rgb = []
            lengths = []
            volumes = []
            for i in range(num_objs):
                num_pts = int(points/5000)
                obj_xyz, obj_rgb = init_from_pointe(opt.init_prompt[i])
                obj_xyz[:, 1] = - obj_xyz[:, 1]
                obj_xyz[:, 2] = obj_xyz[:, 2] + 0.15
                thetas = np.random.rand(num_pts) * np.pi
                phis = np.random.rand(num_pts) * 2 * np.pi
                radius = np.random.rand(num_pts) * 0.05
                # We create random points inside the bounds of sphere
                obj_xyz_ball = np.stack([
                    radius * np.sin(thetas) * np.sin(phis),
                    radius * np.sin(thetas) * np.cos(phis),
                    radius * np.cos(thetas),
                ], axis=-1)  # [B, 3]expend_dims
                obj_rgb_ball = np.random.random((4096, num_pts, 3))*0.0001
                obj_rgb = (np.expand_dims(obj_rgb, axis=1) +
                           obj_rgb_ball).reshape(-1, 3)
                if i in opt.rotate_angles.keys():
                    obj_xyz = rotate_point_cloud(
                        obj_xyz, (opt.rotate_angles[i][0], opt.rotate_angles[i][1], opt.rotate_angles[i][2]))
                obj_xyz = (np.expand_dims(obj_xyz, axis=1) * opt.radius_params[i] + np.expand_dims(
                    obj_xyz_ball, axis=0) * opt.radius_params[i]).reshape(-1, 3)
                obj_xyz = obj_xyz * 1. + opt.center_params[i]
                num_pts = obj_xyz.shape[0]
                x_min, y_min, z_min = np.min(obj_xyz, axis=0)
                x_max, y_max, z_max = np.max(obj_xyz, axis=0)
                x_val, y_val, z_val = np.abs([x_max - x_min, y_max - y_min, z_max - z_min])
                volume = [x_val, y_val, z_val]
                xyz.append(obj_xyz)
                rgb.append(obj_rgb)
                lengths.append(num_pts)
                volumes.append(volume)
            num_pts = num_objs * lengths[0]
            xyz = np.array(xyz)
            rgb = np.array(rgb)
        elif opt.init_shape == 'mix':
            xyz = []
            rgb = []
            lengths = []
            volumes = []
            for i in range(num_objs):
                num_pts = int(points/5000)
                if opt.init_list[i] == 'ply':
                    obj_xyz = loadPly(opt.init_prompt[i], 4096 * num_pts)
                    obj_rgb = np.random.random((4096, num_pts, 3))*0.0001
                    obj_rgb = obj_rgb.reshape(-1, 3)
                else:
                    obj_xyz, obj_rgb = init_from_pointe(opt.init_prompt[i])
                    obj_xyz[:, 1] = - obj_xyz[:, 1]
                    obj_xyz[:, 2] = obj_xyz[:, 2] + 0.15
                thetas = np.random.rand(num_pts) * np.pi
                phis = np.random.rand(num_pts) * 2 * np.pi
                radius = np.random.rand(num_pts) * 0.05
                # We create random points inside the bounds of sphere
                if i in opt.rotate_angles.keys():
                    obj_xyz = rotate_point_cloud(
                        obj_xyz, (opt.rotate_angles[i][0], opt.rotate_angles[i][1], opt.rotate_angles[i][2]))
                if opt.init_list[i] == 'ply':
                    obj_xyz = obj_xyz * opt.radius_params[i]
                else:
                    obj_xyz_ball = np.stack([
                        radius * np.sin(thetas) * np.sin(phis),
                        radius * np.sin(thetas) * np.cos(phis),
                        radius * np.cos(thetas),
                    ], axis=-1)  # [B, 3]expend_dims
                    obj_rgb_ball = np.random.random((4096, num_pts, 3))*0.0001
                    obj_rgb = (np.expand_dims(obj_rgb, axis=1) +
                            obj_rgb_ball).reshape(-1, 3)
                    obj_xyz = (np.expand_dims(obj_xyz, axis=1) * opt.radius_params[i] + np.expand_dims(
                        obj_xyz_ball, axis=0) * opt.radius_params[i]).reshape(-1, 3)
                obj_xyz = obj_xyz * 1. + opt.center_params[i]
                num_pts = obj_xyz.shape[0]
                x_min, y_min, z_min = np.min(obj_xyz, axis=0)
                x_max, y_max, z_max = np.max(obj_xyz, axis=0)
                x_val, y_val, z_val = np.abs([x_max - x_min, y_max - y_min, z_max - z_min])
                volume = [x_val, y_val, z_val]
                xyz.append(obj_xyz)
                rgb.append(obj_rgb)
                lengths.append(num_pts)
                volumes.append(volume)
            num_pts = num_objs * lengths[0]
            xyz = np.array(xyz)
            rgb = np.array(rgb)
        elif opt.init_shape == 'trellis':
            init_xyz = []
            init_features_dc = []
            init_scaling = []
            init_rotation = []
            init_opacity = []
            lengths = []

            for i in range(num_objs):
                num_pts = int(points/5000) * 4096
                
                # Initialize helper class
                data = Gaussian(
                    sh_degree=0,
                    aabb=[-0.5, -0.5, -0.5, 1.0, 1.0, 1.0],
                    mininum_kernel_size = 9e-4,
                    scaling_bias = 4e-3,
                    opacity_bias = 0.1,
                    scaling_activation = "softplus"
                )

                # Load the point cloud
                data.load_ply(opt.init_prompt[int(i)], num_pts)
                
                # --- 1. PREPARE PARAMS ---
                scale_val = opt.radius_params[i]
                center_val = opt.center_params[i]
                # rotate_point_cloud expects (x, y, z) angles
                angles = (opt.rotate_angles[i][0], opt.rotate_angles[i][1], opt.rotate_angles[i][2])

                # --- 2. TRANSFORM POSITIONS (XYZ) ---
                # Rotate positions (returns numpy)
                xyz_np = rotate_point_cloud(data._xyz, angles)
                # Scale positions (spread points apart) + Translate
                xyz_np = xyz_np * scale_val + center_val

                # --- 3. TRANSFORM SCALES (SIZE) ---
                # We must scale the size of the gaussians, otherwise they look like tiny dots.
                # We get the linear scale, multiply it, and re-encode it to hidden format.
                current_scales = data.get_scaling # Returns linear scales (not log)
                new_scales = current_scales * scale_val
                
                # Re-encode back to storage format (inverse softplus or log)
                if data.scaling_activation == "softplus":
                    hidden_scales = torch.log(torch.exp(new_scales) - 1 + 1e-6)
                else: 
                    hidden_scales = torch.log(new_scales + 1e-6)

                # --- 4. TRANSFORM ROTATIONS (ORIENTATION) ---
                # We must rotate the individual gaussians so they face the right way.
                # Construct the rotation matrix R_global matching 'rotate_point_cloud' logic (Z@Y@X)
                ax, ay, az = angles
                Rx = np.array([[1, 0, 0], [0, np.cos(ax), -np.sin(ax)], [0, np.sin(ax), np.cos(ax)]])
                Ry = np.array([[np.cos(ay), 0, np.sin(ay)], [0, 1, 0], [-np.sin(ay), 0, np.cos(ay)]])
                Rz = np.array([[np.cos(az), -np.sin(az), 0], [np.sin(az), np.cos(az), 0], [0, 0, 1]])
                R_global = torch.tensor(Rz @ Ry @ Rx, dtype=torch.float32, device=data.device)

                # Convert Quaternions -> Matrix -> Apply Transform -> Quaternions
                qs = data._rotation
                Rs_local = utils3d.torch.quaternion_to_matrix(qs)
                Rs_new = torch.matmul(R_global.unsqueeze(0), Rs_local) # Broadcast global rot over all local rots
                new_rotations = utils3d.torch.matrix_to_quaternion(Rs_new)

                # --- 5. STORE ---
                init_xyz.append(torch.tensor(xyz_np, dtype=torch.float32, device=data.device))
                init_features_dc.append(data._features_dc)
                init_scaling.append(hidden_scales)
                init_rotation.append(new_rotations)
                init_opacity.append(data._opacity)
                lengths.append(xyz_np.shape[0])
            print(lengths)

            # Concatenate all attributes
            # init_xyz = torch.cat(init_xyz, dim=0)
            # init_features_dc = torch.cat(init_features_dc, dim=0)
            # init_scaling = torch.cat(init_scaling, dim=0)
            # init_rotation = torch.cat(init_rotation, dim=0)
            # init_opacity = torch.cat(init_opacity, dim=0)
            init_xyz = zero_pad_tensor(init_xyz, pad_size=max(lengths), num_objs=num_objs)
            init_features_dc = zero_pad_tensor(init_features_dc, pad_size=max(lengths), num_objs=num_objs)
            init_scaling = zero_pad_tensor(init_scaling, pad_size=max(lengths), num_objs=num_objs)
            init_rotation = zero_pad_tensor(init_rotation, pad_size=max(lengths), num_objs=num_objs)
            init_opacity = zero_pad_tensor(init_opacity, pad_size=max(lengths), num_objs=num_objs)
            volumes = None

            pcd = TrellisGaussians(
                xyz=init_xyz.clone().detach().cpu(),
                features_dc=init_features_dc.clone().detach().cpu(),
                scaling=init_scaling.clone().detach().cpu(),
                rotation=init_rotation.clone().detach().cpu(),
                opacity=init_opacity.clone().detach().cpu()
            )
        elif opt.init_shape == 'LGM':
            init_xyz = []
            init_features_dc = []
            init_scaling = []
            init_rotation = []
            init_opacity = []
            lengths = []

            # Initialize the renderer once to access the load_ply method
            renderer = GaussianRenderer(opt)

            for i in range(num_objs):
                # Use the specified number of points from params or default calculation
                num_pts = opt.init_num_pts # Or keep your int(points/5000)*4096 logic
                
                # Load dictionary from LGM renderer
                # Returns keys: 'means3D', 'opacity', 'scales', 'rotations', 'shs'
                data_dict = renderer.load_ply(opt.init_prompt[int(i)], num_pts)
                
                # Extract Tensors (on GPU)
                means3D = data_dict["means3D"]
                scales = data_dict["scales"]       # Linear scale (because we used compatible=True)
                rotations = data_dict["rotations"] # Quaternions [w, x, y, z] or [x, y, z, w] depending on lib
                opacity = data_dict["opacity"]     # Linear opacity [0-1]
                shs = data_dict["shs"]             # RGB/SH features

                # --- 1. PREPARE PARAMS ---
                scale_val = opt.radius_params[i]
                center_val = torch.tensor(opt.center_params[i], device=means3D.device, dtype=torch.float32)
                angles = (opt.rotate_angles[i][0], opt.rotate_angles[i][1], opt.rotate_angles[i][2])

                # --- 2. TRANSFORM POSITIONS (XYZ) ---
                # Convert to numpy for rotate_point_cloud if strictly required, 
                # otherwise standard PyTorch matrix multiplication is preferred.
                # Assuming rotate_point_cloud takes numpy:
                xyz_np = means3D.detach().cpu().numpy()
                xyz_np = rotate_point_cloud(xyz_np, angles)
                
                # Convert back to tensor for scaling/translation
                xyz_tensor = torch.tensor(xyz_np, dtype=torch.float32, device=means3D.device)
                xyz_tensor = xyz_tensor * scale_val + center_val

                # --- 3. TRANSFORM SCALES (SIZE) ---
                # LGM loaded scales are linear. We multiply by the object scale.
                new_scales = scales * scale_val
                
                # Convert to "Hidden" Log-Space for GaussianModel storage
                # (Standard 3DGS optimization stores log-scales)
                hidden_scales = torch.log(new_scales + 1e-8)

                # --- 4. TRANSFORM ROTATIONS (ORIENTATION) ---
                # Construct Global Rotation Matrix (Z@Y@X)
                ax, ay, az = angles
                Rx = np.array([[1, 0, 0], [0, np.cos(ax), -np.sin(ax)], [0, np.sin(ax), np.cos(ax)]])
                Ry = np.array([[np.cos(ay), 0, np.sin(ay)], [0, 1, 0], [-np.sin(ay), 0, np.cos(ay)]])
                Rz = np.array([[np.cos(az), -np.sin(az), 0], [np.sin(az), np.cos(az), 0], [0, 0, 1]])
                R_global = torch.tensor(Rz @ Ry @ Rx, dtype=torch.float32, device=means3D.device)

                # Rotate Quaternions: Quat -> Mat -> MatMul -> Quat
                # Ensure utils3d matches your library (e.g. pytorch3d)
                Rs_local = utils3d.torch.quaternion_to_matrix(rotations)
                Rs_new = torch.matmul(R_global.unsqueeze(0), Rs_local) 
                new_rotations = utils3d.torch.matrix_to_quaternion(Rs_new)

                # --- 5. STORE ---
                init_xyz.append(xyz_tensor)
                
                # Handle Feature Dimensions: LGM is [N, 3], 3DGS often wants [N, 1, 3] for DC
                if shs.dim() == 2:
                    shs = shs.unsqueeze(1)
                init_features_dc.append(shs)
                
                init_scaling.append(hidden_scales)
                init_rotation.append(new_rotations)
                
                # Handle Opacity: Convert linear opacity back to logits (inverse sigmoid) for storage
                # This allows the optimizer to apply sigmoid() during forward pass
                hidden_opacity = kiui.op.inverse_sigmoid(opacity) 
                init_opacity.append(hidden_opacity)
                
                lengths.append(xyz_tensor.shape[0])

            print(f"Initialized LGM Objects with point counts: {lengths}")
            
            # Pad and stack
            init_xyz = zero_pad_tensor(init_xyz, pad_size=max(lengths), num_objs=num_objs)
            init_features_dc = zero_pad_tensor(init_features_dc, pad_size=max(lengths), num_objs=num_objs)
            init_scaling = zero_pad_tensor(init_scaling, pad_size=max(lengths), num_objs=num_objs)
            init_rotation = zero_pad_tensor(init_rotation, pad_size=max(lengths), num_objs=num_objs)
            init_opacity = zero_pad_tensor(init_opacity, pad_size=max(lengths), num_objs=num_objs)
            volumes = None

            pcd = TrellisGaussians(
                xyz=init_xyz.clone().detach().cpu(),
                features_dc=init_features_dc.clone().detach().cpu(),
                scaling=init_scaling.clone().detach().cpu(),
                rotation=init_rotation.clone().detach().cpu(),
                opacity=init_opacity.clone().detach().cpu()
            )
        elif opt.init_shape == 'scene':
            thetas = np.random.rand(num_pts)*np.pi
            phis = np.random.rand(num_pts)*2*np.pi
            radius = np.random.rand(num_pts) + opt.radius_range[-1]*3
            # We create random points inside the bounds of sphere
            xyz = np.stack([
                radius * np.sin(thetas) * np.sin(phis),
                radius * np.sin(thetas) * np.cos(phis),
                radius * np.cos(thetas),
            ], axis=-1)  # [B, 3]
        else:
            raise NotImplementedError()
        print(f"Generating random point cloud ({num_pts})...")

        shs = np.random.random((num_objs, num_pts // num_objs, 3)) / 255.0

        if opt.init_shape == 'pointe' and opt.use_pointe_rgb:
            pcd = BasicPointCloud(points=xyz, colors=rgb, normals=np.zeros(
                (num_objs, num_pts // num_objs, 3)))
            storePly(ply_path, xyz, rgb * 255)
            np.save(lengths_path, np.array(lengths))
            np.save(volumes_path, np.array(volumes))
        elif opt.init_shape == "trellis":
            pcd = pcd
            np.save(lengths_path, np.array(lengths))
            np.save(volumes_path, np.array(volumes))
        else:
            pcd = BasicPointCloud(points=xyz, colors=SH2RGB(
                shs), normals=np.zeros((num_objs, num_pts // num_objs, 3)))
            storePly(ply_path, np.vstack(xyz), np.vstack(SH2RGB(shs) * 255))
            np.save(lengths_path, np.array(lengths))
            np.save(volumes_path, np.array(volumes))

    if opt.init_shape != 'trellis':
        try:
            pcd = fetchPly(ply_path, num_objs)
            lengths = np.load(lengths_path).tolist()
            volumes = np.load(volumes_path).tolist()
        except:
            pcd = None
            lengths = None
            volumes = None
    else:
        lengths = np.load(lengths_path).tolist()
        pcd = pcd
        volumes = None

    scene_info = GraphSceneInfo(point_cloud=pcd,
                                points_per_obj=lengths,
                                num_objs=num_objs,
                                volumes=volumes,
                                test_cameras=test_cam_infos,
                                ply_path=ply_path)

    return scene_info

def safe_normalize(x, eps=1e-20):
    return x / torch.sqrt(torch.clamp(torch.sum(x * x, -1, keepdim=True), min=eps))


def circle_poses(radius=torch.tensor([3.2]), theta=torch.tensor([60]), phi=torch.tensor([0]), angle_overhead=30, angle_front=60):

    theta = theta / 180 * np.pi
    phi = phi / 180 * np.pi
    angle_overhead = angle_overhead / 180 * np.pi
    angle_front = angle_front / 180 * np.pi

    centers = torch.stack([
        radius * torch.sin(theta) * torch.sin(phi),
        radius * torch.sin(theta) * torch.cos(phi),
        radius * torch.cos(theta),
    ], dim=-1)  # [B, 3]

    # lookat
    forward_vector = safe_normalize(centers)
    up_vector = torch.FloatTensor(
        [0, 0, 1]).unsqueeze(0).repeat(len(centers), 1)
    right_vector = safe_normalize(
        torch.cross(forward_vector, up_vector, dim=-1))
    up_vector = safe_normalize(torch.cross(
        right_vector, forward_vector, dim=-1))

    poses = torch.eye(4, dtype=torch.float).unsqueeze(
        0).repeat(len(centers), 1, 1)
    poses[:, :3, :3] = torch.stack(
        (-right_vector, up_vector, forward_vector), dim=-1)
    poses[:, :3, 3] = centers

    return poses.numpy()


def gen_random_pos(size, param_range, gamma=1):
    lower, higher = param_range[0], param_range[1]

    mid = lower + (higher - lower) * 0.5
    radius = (higher - lower) * 0.5

    rand_ = torch.rand(size)  # 0, 1
    sign = torch.where(torch.rand(size) > 0.5,
                       torch.ones(size) * -1., torch.ones(size))
    rand_ = sign * (rand_ ** gamma)

    return (rand_ * radius) + mid


def rand_poses(size, opt, radius_range=[1, 1.5], theta_range=[0, 120], phi_range=[0, 360], angle_overhead=30, angle_front=60, uniform_sphere_rate=0.5, rand_cam_gamma=1):
    ''' generate random poses from an orbit camera
    Args:
            size: batch size of generated poses.
            device: where to allocate the output.
            radius: camera radius
            theta_range: [min, max], should be in [0, pi]
            phi_range: [min, max], should be in [0, 2 * pi]
    Return:
            poses: [size, 4, 4]
    '''

    theta_range = np.array(theta_range) / 180 * np.pi
    phi_range = np.array(phi_range) / 180 * np.pi
    angle_overhead = angle_overhead / 180 * np.pi
    angle_front = angle_front / 180 * np.pi

    radius = gen_random_pos(size, radius_range)

    if random.random() < uniform_sphere_rate:
        unit_centers = F.normalize(
            torch.stack([
                torch.randn(size),
                torch.abs(torch.randn(size)),
                torch.randn(size),
            ], dim=-1), p=2, dim=1
        )
        thetas = torch.acos(unit_centers[:, 1])
        phis = torch.atan2(unit_centers[:, 0], unit_centers[:, 2])
        phis[phis < 0] += 2 * np.pi
        centers = unit_centers * radius.unsqueeze(-1)
    else:
        thetas = gen_random_pos(size, theta_range, rand_cam_gamma)
        phis = gen_random_pos(size, phi_range, rand_cam_gamma)
        phis[phis < 0] += 2 * np.pi

        centers = torch.stack([
            radius * torch.sin(thetas) * torch.sin(phis),
            radius * torch.sin(thetas) * torch.cos(phis),
            radius * torch.cos(thetas),
        ], dim=-1)  # [B, 3]

    targets = 0

    # jitters
    if opt.jitter_pose:
        jit_center = opt.jitter_center  # 0.015  # was 0.2
        jit_target = opt.jitter_target
        centers += torch.rand_like(centers) * jit_center - jit_center/2.0
        targets += torch.randn_like(centers) * jit_target

    # lookat
    forward_vector = safe_normalize(centers - targets)
    up_vector = torch.FloatTensor([0, 0, 1]).unsqueeze(0).repeat(size, 1)
    # up_vector = torch.FloatTensor([0, 0, 1]).unsqueeze(0).repeat(size, 1)
    right_vector = safe_normalize(
        torch.cross(forward_vector, up_vector, dim=-1))

    if opt.jitter_pose:
        up_noise = torch.randn_like(up_vector) * opt.jitter_up
    else:
        up_noise = 0

    up_vector = safe_normalize(torch.cross(
        right_vector, forward_vector, dim=-1) + up_noise)  # forward_vector

    poses = torch.eye(4, dtype=torch.float).unsqueeze(0).repeat(size, 1, 1)
    poses[:, :3, :3] = torch.stack(
        (-right_vector, up_vector, forward_vector), dim=-1)  # up_vector
    poses[:, :3, 3] = centers

    # back to degree
    thetas = thetas / np.pi * 180
    phis = phis / np.pi * 180

    return poses.numpy(), thetas.numpy(), phis.numpy(), radius.numpy()


def rand_poses_orthogonal(
    size,
    opt,
    radius_range=[1, 1.5],
    theta_range=[0, 120],
    phi_range=[0, 360],
    angle_overhead=30,
    angle_front=60,
    uniform_sphere_rate=0.5,
    rand_cam_gamma=1,
):
    # Convert angle ranges to radians
    theta_range = np.array(theta_range) / 180 * np.pi
    phi_range = np.array(phi_range) / 180 * np.pi
    angle_overhead = angle_overhead / 180 * np.pi
    angle_front = angle_front / 180 * np.pi

    # Sample radii
    radius = gen_random_pos(size, radius_range)

    # Sample elevation (theta) angles
    if random.random() < 0.5:
        # Uniform sampling within the theta range
        thetas = torch.rand(
            size) * (theta_range[1] - theta_range[0]) + theta_range[0]
    else:
        # Uniform sampling on the sphere (biased toward poles)
        theta_percent = [
            (theta_range[0] + np.pi / 2) / np.pi,
            (theta_range[1] + np.pi / 2) / np.pi,
        ]
        thetas = torch.asin(
            2 * (torch.rand(size) *
                 (theta_percent[1] - theta_percent[0]) + theta_percent[0]) - 1.0
        )

    # Sample orthogonal azimuth (phi) angles
    views_per_batch = 4
    phis = (
        torch.rand(size // views_per_batch).reshape(-1, 1)
        + torch.arange(views_per_batch).reshape(1, -1)
    ).reshape(-1) / views_per_batch * (phi_range[1] - phi_range[0]) + phi_range[0]
    phis[phis < 0] += 2 * np.pi  # Ensure phis are in [0, 2*pi]

    # Generate centers
    centers = torch.stack(
        [
            radius * torch.sin(thetas) * torch.sin(phis),
            radius * torch.sin(thetas) * torch.cos(phis),
            radius * torch.cos(thetas),
        ],
        dim=-1,
    )
    # Look-at target and vectors
    targets = torch.zeros_like(centers)
    forward_vector = safe_normalize(centers - targets)
    up_vector = torch.FloatTensor([0, 0, 1]).unsqueeze(0).repeat(size, 1)
    right_vector = safe_normalize(
        torch.cross(forward_vector, up_vector, dim=-1))
    up_vector = safe_normalize(torch.cross(
        right_vector, forward_vector, dim=-1))

    # Construct camera-to-world (c2w) matrices
    poses = torch.eye(4, dtype=torch.float).unsqueeze(0).repeat(size, 1, 1)
    poses[:, :3, :3] = torch.stack(
        (-right_vector, up_vector, forward_vector), dim=-1)
    poses[:, :3, 3] = centers

    # Convert angles back to degrees
    thetas_deg = thetas / np.pi * 180
    phis_deg = phis / np.pi * 180

    return poses.numpy(), thetas_deg.numpy(), phis_deg.numpy(), radius.numpy()


def GenerateCircleCameras(opt, size=8, render45=False):
    # random focal
    fov = opt.default_fovy
    cam_infos = []
    # generate specific data structure
    for idx in range(size):
        thetas = torch.FloatTensor([opt.default_polar])
        phis = torch.FloatTensor([(idx / size) * 360])
        radius = torch.FloatTensor([opt.default_radius])
        # random pose on the fly
        poses = circle_poses(radius=radius, theta=thetas, phi=phis,
                             angle_overhead=opt.angle_overhead, angle_front=opt.angle_front)
        matrix = np.linalg.inv(poses[0])
        R = -np.transpose(matrix[:3, :3])
        R[:, 0] = -R[:, 0]
        T = -matrix[:3, 3]
        fovy = focal2fov(fov2focal(fov, opt.image_h), opt.image_w)
        FovY = fovy
        FovX = fov

        # delta polar/azimuth/radius to default view
        delta_polar = thetas - opt.default_polar
        delta_azimuth = phis - opt.default_azimuth
        delta_azimuth[delta_azimuth > 180] -= 360  # range in [-180, 180]
        delta_radius = radius - opt.default_radius
        cam_infos.append(RandCameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, width=opt.image_w,
                                        height=opt.image_h, delta_polar=delta_polar, delta_azimuth=delta_azimuth, delta_radius=delta_radius, c2w=poses[0]))
    if render45:
        for idx in range(size):
            thetas = torch.FloatTensor([opt.default_polar*2//3])
            phis = torch.FloatTensor([(idx / size) * 360])
            radius = torch.FloatTensor([opt.default_radius])
            # random pose on the fly
            poses = circle_poses(radius=radius, theta=thetas, phi=phis,
                                 angle_overhead=opt.angle_overhead, angle_front=opt.angle_front)
            matrix = np.linalg.inv(poses[0])
            R = -np.transpose(matrix[:3, :3])
            R[:, 0] = -R[:, 0]
            T = -matrix[:3, 3]
            fovy = focal2fov(fov2focal(fov, opt.image_h), opt.image_w)
            FovY = fovy
            FovX = fov

            # delta polar/azimuth/radius to default view
            delta_polar = thetas - opt.default_polar
            delta_azimuth = phis - opt.default_azimuth
            delta_azimuth[delta_azimuth > 180] -= 360  # range in [-180, 180]
            delta_radius = radius - opt.default_radius
            cam_infos.append(RandCameraInfo(uid=idx+size, R=R, T=T, FovY=FovY, FovX=FovX, width=opt.image_w,
                                            height=opt.image_h, delta_polar=delta_polar, delta_azimuth=delta_azimuth, delta_radius=delta_radius, c2w=poses[0]))
    return cam_infos


def get_dynamic_fovy_range(fovy_range, cam_scale):
    # Reference points
    x1, x2 = 0.1, 1.0
    y1 = [1.46, 1.98]     # FOV at smallest scale
    y2 = fovy_range       # FOV at default scale

    # Compute slopes and intercepts
    m_min = (y2[0] - y1[0]) / (x2 - x1)
    m_max = (y2[1] - y1[1]) / (x2 - x1)

    b_min = y1[0] - m_min * x1
    b_max = y1[1] - m_max * x1

    # Interpolate for current scale
    min_fov = m_min * cam_scale + b_min
    max_fov = m_max * cam_scale + b_max

    return [round(min_fov, 4), round(max_fov, 4)]


def GenerateRandomCameras(opt, size=2000, cam_scale=1.0, SSAA=True):
    # random pose on the fly
    poses, thetas, phis, radius = rand_poses_orthogonal(size, opt, radius_range=opt.radius_range, theta_range=opt.theta_range, phi_range=opt.phi_range,
                                             angle_overhead=opt.angle_overhead, angle_front=opt.angle_front, uniform_sphere_rate=opt.uniform_sphere_rate,
                                             rand_cam_gamma=opt.rand_cam_gamma)
    # poses, thetas, phis, radius = rand_poses_orthogonal(size, opt, radius_range=[x * cam_scale for x in opt.radius_range], theta_range=opt.theta_range, phi_range=opt.phi_range,
    #                                          angle_overhead=opt.angle_overhead, angle_front=opt.angle_front, uniform_sphere_rate=opt.uniform_sphere_rate,
    #                                          rand_cam_gamma=opt.rand_cam_gamma)
    # delta polar/azimuth/radius to default view
    delta_polar = thetas - opt.default_polar
    delta_azimuth = phis - opt.default_azimuth
    delta_azimuth[delta_azimuth > 180] -= 360  # range in [-180, 180]
    delta_radius = radius - opt.default_radius
    # print(radius, delta_radius, opt.radius_range, [x * cam_scale for x in opt.radius_range])
    # random focal
    # fovy_range = get_dynamic_fovy_range(opt.fovy_range, cam_scale)
    # fov = random.random() * (fovy_range[1] - fovy_range[0]) + fovy_range[0]

    fov = random.random() * \
        (opt.fovy_range[1] - opt.fovy_range[0]) + opt.fovy_range[0]


    cam_infos = []

    if SSAA:
        ssaa = opt.SSAA
    else:
        ssaa = 1

    image_h = opt.image_h * ssaa
    image_w = opt.image_w * ssaa

    # generate specific data structure
    for idx in range(size):
        matrix = np.linalg.inv(poses[idx])
        R = -np.transpose(matrix[:3, :3])
        R[:, 0] = -R[:, 0]
        T = -matrix[:3, 3]
        fovy = focal2fov(fov2focal(fov, image_h), image_w)
        FovY = fovy
        FovX = fov

        cam_infos.append(RandCameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, width=image_w,
                                        height=image_h, delta_polar=delta_polar[idx],
                                        delta_azimuth=delta_azimuth[idx], delta_radius=delta_radius[idx], c2w=poses[idx]))
    return cam_infos

def GenerateCameraAtZeroAzimuth(opt, radius_range, SSAA=True):
    # Generate a single pose at 0-degree azimuth
    size = 4  # Only one camera
    fixed_azimuth = 90  # Azimuth at 0 degrees
    

    # Generate a pose with fixed azimuth
    poses, thetas, phis, radius = rand_poses_orthogonal(
        size, opt,
        radius_range=radius_range,
        theta_range=opt.theta_range,
        phi_range=(fixed_azimuth, fixed_azimuth),  # Force azimuth to 0 degrees
        angle_overhead=0,
        angle_front=opt.angle_front,
        uniform_sphere_rate=opt.uniform_sphere_rate,
        rand_cam_gamma=opt.rand_cam_gamma
    )

    # delta polar/azimuth/radius to default view
    delta_polar = thetas - opt.default_polar
    delta_azimuth = phis - opt.default_azimuth
    delta_azimuth[delta_azimuth > 180] -= 360  # Ensure range in [-180, 180]
    delta_radius = radius - opt.default_radius

    # Random focal length
    fov = random.random() * (opt.fovy_range[1] - opt.fovy_range[0]) + opt.fovy_range[0]

    if SSAA:
        ssaa = opt.SSAA
    else:
        ssaa = 1

    image_h = opt.image_h * ssaa
    image_w = opt.image_w * ssaa
    cam_infos = []
    for idx in range(size):
        matrix = np.linalg.inv(poses[idx])
        R = -np.transpose(matrix[:3, :3])
        R[:, 0] = -R[:, 0]
        T = -matrix[:3, 3]
        fovy = focal2fov(fov2focal(fov, image_h), image_w)
        FovY = fovy
        FovX = fov

        cam_infos.append(RandCameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, width=image_w,
                                        height=image_h, delta_polar=delta_polar[idx],
                                        delta_azimuth=delta_azimuth[idx], delta_radius=delta_radius[idx], c2w=poses[idx]))

    return cam_infos  # Return the single camera

def GeneratePurnCameras(opt, size=300):
    # random pose on the fly
    poses, thetas, phis, radius = rand_poses(size, opt, radius_range=[opt.default_radius, opt.default_radius+0.1], theta_range=opt.theta_range,
                                             phi_range=opt.phi_range, angle_overhead=opt.angle_overhead, angle_front=opt.angle_front, uniform_sphere_rate=opt.uniform_sphere_rate)
    # delta polar/azimuth/radius to default view
    delta_polar = thetas - opt.default_polar
    delta_azimuth = phis - opt.default_azimuth
    delta_azimuth[delta_azimuth > 180] -= 360  # range in [-180, 180]
    delta_radius = radius - opt.default_radius

    fov = opt.default_fovy
    cam_infos = []
    # generate specific data structure
    for idx in range(size):
        matrix = np.linalg.inv(poses[idx])
        R = -np.transpose(matrix[:3, :3])
        R[:, 0] = -R[:, 0]
        T = -matrix[:3, 3]

        fovy = focal2fov(fov2focal(fov, opt.image_h), opt.image_w)
        FovY = fovy
        FovX = fov

        cam_infos.append(RandCameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, width=opt.image_w,
                                        height=opt.image_h, delta_polar=delta_polar[idx], delta_azimuth=delta_azimuth[idx], delta_radius=delta_radius[idx]))
    return cam_infos

def GenerateSphericalCameras(opt, size=150, obj_azimuth_adjustments=None, edge_azimuth_adjustments=None, SSAA=True):
    """
    Generate cameras using spherical Hammersley sequence with different radius parameters
    and azimuth adjustments for objects and edges.

    Args:
        opt: Options containing camera parameters
        size: Number of base cameras to generate (default 150)
        radius_params: List of fixed radius values to use
        obj_azimuth_adjustments: Dict {obj_idx: azimuth_offset} for objects
        edge_azimuth_adjustments: Dict {edge_idx: azimuth_offset} for edges
        SSAA: Whether to apply super sampling

    Returns:
        Dictionary containing different camera sets
    """
    from dataset_toolkits.utils import sphere_hammersley_sequence

    if obj_azimuth_adjustments is None:
        obj_azimuth_adjustments = {}

    if edge_azimuth_adjustments is None:
        edge_azimuth_adjustments = {}

    # SSAA settings
    if SSAA:
        ssaa = opt.SSAA
    else:
        ssaa = 1

    image_h = opt.image_h * ssaa
    image_w = opt.image_w * ssaa

    # Random focal length
    fov = random.random() * (opt.fovy_range[1] - opt.fovy_range[0]) + opt.fovy_range[0]

    def create_camera_from_spherical(phi, theta, radius, uid_offset=0, azimuth_adjustment=0):
        """Create camera info from spherical coordinates"""
        # Adjust azimuth
        phi_adjusted = phi + np.radians(azimuth_adjustment)

        # Convert spherical to Cartesian
        x = radius * np.cos(theta) * np.cos(phi_adjusted)
        y = radius * np.cos(theta) * np.sin(phi_adjusted)
        z = radius * np.sin(theta)

        # Create pose matrix
        center = torch.tensor([x, y, z], dtype=torch.float32)
        target = torch.zeros(3, dtype=torch.float32)

        # Look-at vectors
        forward_vector = safe_normalize((center - target).unsqueeze(0))
        up_vector = torch.tensor([[0, 0, 1]], dtype=torch.float32)
        right_vector = safe_normalize(torch.cross(forward_vector, up_vector, dim=-1))
        up_vector = safe_normalize(torch.cross(right_vector, forward_vector, dim=-1))

        # Create pose matrix
        pose = torch.eye(4, dtype=torch.float32)
        pose[:3, :3] = torch.stack([-right_vector[0], up_vector[0], forward_vector[0]], dim=-1)
        pose[:3, 3] = center

        # Convert to camera parameters
        matrix = np.linalg.inv(pose.numpy())
        R = -np.transpose(matrix[:3, :3])
        R[:, 0] = -R[:, 0]
        T = -matrix[:3, 3]

        # Calculate FOV
        fovy = focal2fov(fov2focal(fov, image_h), image_w)
        FovY = fovy
        FovX = fov

        # Calculate deltas for compatibility
        theta_deg = np.degrees(theta)
        phi_deg = np.degrees(phi_adjusted)

        delta_polar = theta_deg - opt.default_polar
        delta_azimuth = phi_deg - opt.default_azimuth
        delta_azimuth = delta_azimuth if delta_azimuth <= 180 else delta_azimuth - 360
        delta_azimuth = delta_azimuth if delta_azimuth >= -180 else delta_azimuth + 360
        delta_radius = radius - opt.default_radius

        return RandCameraInfo(
            uid=uid_offset,
            R=R, T=T, FovY=FovY, FovX=FovX,
            width=image_w, height=image_h,
            delta_polar=delta_polar,
            delta_azimuth=delta_azimuth,
            delta_radius=delta_radius,
            c2w=pose.numpy()
        )

    spherical_coords = []

    for i in range(size):
        phi, theta = sphere_hammersley_sequence(i, size)
        spherical_coords.append((phi, theta))

    result = {}

    # Generate object-specific cameras
    if obj_azimuth_adjustments:
        result['objects'] = {}
        for obj_idx, azimuth_offset in obj_azimuth_adjustments.items():
            result['objects'][obj_idx] = {}

            # Radius = 2 cameras
            obj_cameras_r2 = []
            for i, (phi, theta) in enumerate(spherical_coords):
                cam_info = create_camera_from_spherical(
                    phi, theta, 3.5,
                    uid_offset=i,  # Unique UID scheme
                    azimuth_adjustment=azimuth_offset
                )
                obj_cameras_r2.append(cam_info)
            result['objects'][obj_idx]['gt'] = obj_cameras_r2

            # Radius = 4 cameras
            obj_cameras_r4 = []
            for i, (phi, theta) in enumerate(spherical_coords):
                cam_info = create_camera_from_spherical(
                    phi, theta, 3.5,
                    uid_offset=i,  # Unique UID scheme
                    azimuth_adjustment=0
                )
                obj_cameras_r4.append(cam_info)
            result['objects'][obj_idx]['pred'] = obj_cameras_r4

    # Generate edge-specific cameras
    if edge_azimuth_adjustments:
        result['edges'] = {}
        for edge_idx, azimuth_offset in edge_azimuth_adjustments.items():
            result['edges'][edge_idx] = {}

            # Radius = 2 cameras
            edge_cameras_r2 = []
            for i, (phi, theta) in enumerate(spherical_coords):
                cam_info = create_camera_from_spherical(
                    phi, theta, 3.5,
                    uid_offset=i,  # Unique UID scheme
                    azimuth_adjustment=azimuth_offset
                )
                edge_cameras_r2.append(cam_info)
            result['edges'][edge_idx]['gt'] = edge_cameras_r2

            # Radius = 4 cameras
            edge_cameras_r4 = []
            for i, (phi, theta) in enumerate(spherical_coords):
                cam_info = create_camera_from_spherical(
                    phi, theta, 3.5,
                    uid_offset=i,  # Unique UID scheme
                    azimuth_adjustment=0
                )
                edge_cameras_r4.append(cam_info)
            result['edges'][edge_idx]['pred'] = edge_cameras_r4

    return result

sceneLoadTypeCallbacks = {
    "RandomCam": readCircleCamInfo
}
