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

    points = np.random.choice(positions.shape[0], num_points, replace=False)
    return positions[points]

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
    return np.dot(point_cloud, rotation_matrix.T)

# only test_camera


def readCircleCamInfo(path, opt):
    print("Reading Test Transforms")
    test_cam_infos = GenerateCircleCameras(opt, render45=opt.render_45)
    ply_path = os.path.join(path, "init_points3d.ply")
    lengths_path = os.path.join(path, "init_points3d_length.npy")
    volumes_path = os.path.join(path, "inint_points3d_volume.npy")
    num_pts = opt.init_num_pts
    num_objs = opt.num_objs

    if not os.path.exists(ply_path) or not os.path.exists(lengths_path) or not os.path.exists(volumes_path):
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
                volume = x_val * y_val * z_val
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
                volume = x_val * y_val * z_val
                xyz.append(obj_xyz)
                rgb.append(obj_rgb)
                lengths.append(num_pts)
                volumes.append(volume)
            num_pts = num_objs * lengths[0]
            xyz = np.array(xyz)
            rgb = np.array(rgb)
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
        else:
            pcd = BasicPointCloud(points=xyz, colors=SH2RGB(
                shs), normals=np.zeros((num_objs, num_pts // num_objs, 3)))
            storePly(ply_path, np.vstack(xyz), np.vstack(SH2RGB(shs) * 255))
            np.save(lengths_path, np.array(lengths))
            np.save(volumes_path, np.array(volumes))
    try:
        pcd = fetchPly(ply_path, num_objs)
        lengths = np.load(lengths_path).tolist()
        volumes = np.load(volumes_path).tolist()
    except:
        pcd = None
        lengths = None
        volumes = None

    scene_info = GraphSceneInfo(point_cloud=pcd,
                                points_per_obj=lengths,
                                num_objs=num_objs,
                                volumes=volumes,
                                test_cameras=test_cam_infos,
                                ply_path=ply_path)
    return scene_info
# borrow from https://github.com/ashawkey/stable-dreamfusion


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


def GenerateRandomCameras(opt, size=2000, SSAA=True):
    # random pose on the fly
    poses, thetas, phis, radius = rand_poses_orthogonal(size, opt, radius_range=opt.radius_range, theta_range=opt.theta_range, phi_range=opt.phi_range,
                                             angle_overhead=opt.angle_overhead, angle_front=opt.angle_front, uniform_sphere_rate=opt.uniform_sphere_rate,
                                             rand_cam_gamma=opt.rand_cam_gamma)
    # delta polar/azimuth/radius to default view
    delta_polar = thetas - opt.default_polar
    delta_azimuth = phis - opt.default_azimuth
    delta_azimuth[delta_azimuth > 180] -= 360  # range in [-180, 180]
    delta_radius = radius - opt.default_radius
    # random focal
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

def GenerateCameraAtZeroAzimuth(opt, SSAA=True):
    # Generate a single pose at 0-degree azimuth
    size = 4  # Only one camera
    fixed_azimuth = 0  # Azimuth at 0 degrees
    
    # Generate a pose with fixed azimuth
    poses, thetas, phis, radius = rand_poses_orthogonal(
        size, opt, 
        radius_range=opt.radius_range, 
        theta_range=opt.theta_range, 
        phi_range=(fixed_azimuth, fixed_azimuth),  # Force azimuth to 0 degrees
        angle_overhead=opt.angle_overhead, 
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


sceneLoadTypeCallbacks = {
    "RandomCam": readCircleCamInfo
}
