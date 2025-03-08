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

import torch
import traceback
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh, SH2RGB
from utils.graphics_utils import fov2focal
import random

def zero_pad_tensor(tensor_list, pad_size, num_objs):
    x = list(tensor_list[0].shape)
    x[0] = pad_size
    for i in range(num_objs):
        xyz_pad = torch.zeros(
            (x), device=tensor_list[i].device, dtype=tensor_list[i].dtype)
        xyz_pad[:tensor_list[i].shape[0]] = tensor_list[i]
        tensor_list[i] = xyz_pad.unsqueeze(0)

    return torch.cat(tensor_list, dim=0)


def render(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, objs: list, scaling_modifier=1.0, black_video=False,
           override_color=None, sh_deg_aug_ratio=0.1, bg_aug_ratio=0.3, shs_aug_ratio=1.0, scale_aug_ratio=1.0, test=False):
    """
    Render the scene. 

    Background tensor (bg_color) must be on GPU!
    """
    # for i in range(4):
    #     print(pc.get_xyz[i].isnan().sum())
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(
        pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    if black_video:
        bg_color = torch.zeros_like(bg_color)
    # Aug
    if random.random() < sh_deg_aug_ratio and not test:
        act_SH = 0
    else:
        act_SH = pc.active_sh_degree

    if random.random() < bg_aug_ratio and not test:
        if random.random() < 0.5:
            bg_color = torch.rand_like(bg_color)
        else:
            bg_color = torch.zeros_like(bg_color)
        # bg_color = torch.zeros_like(bg_color)

    # bg_color = torch.zeros_like(bg_color)
    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    try:
        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=act_SH,
            campos=viewpoint_camera.camera_center,
            prefiltered=False
        )
    except TypeError as e:
        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=act_SH,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=False
        )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = []
    means2D = []
    opacity = []

    for i in objs:
        means3D.append(pc.get_xyz[i][:pc.points_per_obj[i]])
        means2D.append(screenspace_points[i][:pc.points_per_obj[i]])
        opacity.append(pc.get_opacity[i][:pc.points_per_obj[i]])

    means3D = torch.cat(means3D, dim=0)
    means2D = torch.cat(means2D, dim=0)
    opacity = torch.cat(opacity, dim=0)

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = []
        rotations = []
        for i in objs:
            scales.append(
                pc.get_scaling[i][:pc.points_per_obj[i]].reshape(-1, 3))
            rotations.append(
                pc.get_rotation[i][:pc.points_per_obj[i]].reshape(-1, 4))
        scales = torch.cat(scales, dim=0)
        rotations = torch.cat(rotations, dim=0)

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if colors_precomp is None:
        if pipe.convert_SHs_python:
            rgb = []
            for i in objs:
                raw_rgb = pc.get_features[i][:pc.points_per_obj[i]].transpose(
                    2, 3).reshape(-1, (pc.max_sh_degree + 1) ** 2, 3).view(-1, 3, (pc.max_sh_degree+1)**2).squeeze()[:, :3]
                rgb.append(torch.sigmoid(raw_rgb))
            colors_precomp = torch.cat(rgb, dim=0)
        else:
            shs = []
            for i in objs:
                shs.append(pc.get_features[i][:pc.points_per_obj[i]
                                              ].reshape(-1, (pc.max_sh_degree + 1) ** 2, 3))
            shs = torch.cat(shs, dim=0)
    else:
        colors_precomp = override_color

    if random.random() < shs_aug_ratio and not test:
        variance = (0.2 ** 0.5) * shs
        shs = shs + (torch.randn_like(shs) * variance)

    # add noise to scales
    if random.random() < scale_aug_ratio and not test:
        variance = (0.2 ** 0.5) * scales / 4
        scales = torch.clamp(
            scales + (torch.randn_like(scales) * variance), 0.0)

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_image, radii = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp)

    radiis = []
    start = 0
    l = 0
    for i in range(pc.num_objs):
        l = max(l, pc.points_per_obj[i])

    for i in objs:
        end = start + pc.points_per_obj[i]
        radiis.append(radii[start: end])
        start = end

    radii = zero_pad_tensor(radiis, l, len(objs))
    visibility_filter = []
    for i in range(len(objs)):
        visibility_filter.append((radii[i] > 0).unsqueeze(0))

    visibility_filter = torch.cat(visibility_filter, dim=0)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter": visibility_filter,
            "radii": radii,
            "scales": scales}


def render_obj(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, obj: int, scaling_modifier=1.0, black_video=False,
               override_color=None, sh_deg_aug_ratio=0.1, bg_aug_ratio=0.3, shs_aug_ratio=1.0, scale_aug_ratio=1.0, test=False):
    """
    Render the scene. 

    Background tensor (bg_color) must be on GPU!
    """
    # for i in range(4):
    #     print(pc.get_xyz[i].isnan().sum())
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(
        pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    if black_video:
        bg_color = torch.zeros_like(bg_color)
    # Aug
    if random.random() < sh_deg_aug_ratio and not test:
        act_SH = 0
    else:
        act_SH = pc.active_sh_degree

    if random.random() < bg_aug_ratio and not test:
        if random.random() < 0.5:
            bg_color = torch.rand_like(bg_color)
        else:
            bg_color = torch.zeros_like(bg_color)
        # bg_color = torch.zeros_like(bg_color)

    # bg_color = torch.zeros_like(bg_color)
    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    try:
        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=act_SH,
            campos=viewpoint_camera.camera_center,
            prefiltered=False
        )
    except TypeError as e:
        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=act_SH,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=False
        )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = []
    means2D = []
    opacity = []

    means3D.append(pc.get_xyz[obj][:pc.points_per_obj[obj]])
    means2D.append(screenspace_points[obj][:pc.points_per_obj[obj]])
    opacity.append(pc.get_opacity[obj][:pc.points_per_obj[obj]])

    means3D = torch.cat(means3D, dim=0)
    means2D = torch.cat(means2D, dim=0)
    opacity = torch.cat(opacity, dim=0)

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = []
        rotations = []
        scales.append(pc.get_scaling[obj]
                      [:pc.points_per_obj[obj]].reshape(-1, 3))
        rotations.append(
            pc.get_rotation[obj][:pc.points_per_obj[obj]].reshape(-1, 4))
        scales = torch.cat(scales, dim=0)
        rotations = torch.cat(rotations, dim=0)

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if colors_precomp is None:
        if pipe.convert_SHs_python:
            rgb = []
            raw_rgb = pc.get_features[obj][:pc.points_per_obj[obj]].transpose(
                2, 3).reshape(-1, (pc.max_sh_degree + 1) ** 2, 3).view(-1, 3, (pc.max_sh_degree+1)**2).squeeze()[:, :3]
            rgb.append(torch.sigmoid(raw_rgb))
            colors_precomp = torch.cat(rgb, dim=0)
        else:
            shs = []
            shs.append(pc.get_features[obj][:pc.points_per_obj[obj]
                                            ].reshape(-1, (pc.max_sh_degree + 1) ** 2, 3))
            shs = torch.cat(shs, dim=0)
    else:
        colors_precomp = override_color

    if random.random() < shs_aug_ratio and not test:
        variance = (0.2 ** 0.5) * shs
        shs = shs + (torch.randn_like(shs) * variance)

    # add noise to scales
    if random.random() < scale_aug_ratio and not test:
        variance = (0.2 ** 0.5) * scales / 4
        scales = torch.clamp(
            scales + (torch.randn_like(scales) * variance), 0.0)

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_image, radii = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
            "scales": scales}