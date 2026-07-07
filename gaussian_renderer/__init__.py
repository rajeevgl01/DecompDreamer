import torch
import traceback
import math
from gsplat import rasterization
from scene.gaussian_model import GaussianModel
import random


def pad_packed_tensor(packed_tensor, points_per_obj, pad_value=0):
	"""
	Vectorized padding of a packed tensor into per-object chunks.
	
	Input: packed_tensor [B, Total_Points, ...]
	Output: padded_tensor [B, Num_Objs, Max_Points, ...]
	"""
	# Handle unbatched input by adding a dim
	if packed_tensor.dim() == 1:
		packed_tensor = packed_tensor.unsqueeze(0)
	
	# If input is [Total_Points, ...], add batch dim 1
	if packed_tensor.dim() == 2 and isinstance(points_per_obj, list) and len(points_per_obj) > 1:
		# Heuristic check: if dim 0 matches batch size elsewhere, this might be fine,
		# but usually packed_tensor comes from rasterizer as [B, N, ...]
		pass

	B = packed_tensor.shape[0]
	num_objs = len(points_per_obj)
	max_len = max(points_per_obj) if num_objs > 0 else 0
	remaining_dims = packed_tensor.shape[2:]
	
	# 1. Pre-allocate the full output tensor on GPU (One allocation!)
	out_shape = (B, num_objs, max_len) + remaining_dims
	padded = torch.full(out_shape, pad_value, device=packed_tensor.device, dtype=packed_tensor.dtype)

	# 2. Split the packed dimension (dim 1) into per-object chunks
	if num_objs > 0:
		splits = torch.split(packed_tensor, points_per_obj, dim=1)

		# 3. Fill the pre-allocated tensor
		for i, split_tensor in enumerate(splits):
			current_len = split_tensor.shape[1]
			padded[:, i, :current_len] = split_tensor

	return padded


def render(
	viewpoint_camera: list, pc: GaussianModel, pipe, bg_color: torch.Tensor, objs: list, 
	scaling_modifier=1.0, sh_deg_aug_ratio=0.1, bg_aug_ratio=0.3, shs_aug_ratio=1.0, 
	scale_aug_ratio=1.0, object_scale=1.0, test=False):
	"""
	Render the scene.
	Background tensor (bg_color) must be on GPU!
	"""
	# Set up rasterization configuration
	Ks = []
	B = len(viewpoint_camera)
	for i in range(B):
		tanfovx = math.tan(viewpoint_camera[i].FoVx * 0.5)
		tanfovy = math.tan(viewpoint_camera[i].FoVy * 0.5)
		focal_length_x = viewpoint_camera[i].image_width / (2 * tanfovx)
		focal_length_y = viewpoint_camera[i].image_height / (2 * tanfovy)
		K = torch.tensor(
			[
				[focal_length_x, 0, viewpoint_camera[i].image_width / 2.0],
				[0, focal_length_y, viewpoint_camera[i].image_height / 2.0],
				[0, 0, 1],
			],
			device="cuda",
			dtype=torch.float32,
		)
		Ks.append(K)
	Ks = torch.stack(Ks, dim=0)

	means3D_list = []
	opacity_list = []

	# Collect 3D means and opacities for each object
	# We need the specific point counts for the objects currently being rendered
	current_points_per_obj = []
	
	for i in objs:
		n_points = pc.points_per_obj[i]
		current_points_per_obj.append(n_points)
		means3D_list.append(pc.get_xyz[i][:n_points] + pc.get_pos_delta[i])
		opacity_list.append(pc.get_opacity[i][:n_points])

	# Concatenate all object data
	means3D = torch.cat(means3D_list, dim=0)
	opacity = torch.cat(opacity_list, dim=0)

	# Center 3D means around origin
	shift_to_origin = means3D.mean(dim=0, keepdim=True).detach()
	means3D = means3D - shift_to_origin
	means3D = means3D.float()
	# means3D = means3D / object_scale

	# Scales and Rotations
	scales = []
	rotations = []
	for i in objs:
		scales.append(
			pc.get_scaling[i][:pc.points_per_obj[i]].reshape(-1, 3) * scaling_modifier)
		rotations.append(
			pc.get_rotation[i][:pc.points_per_obj[i]].reshape(-1, 4))
	scales = torch.cat(scales, dim=0)
	rotations = torch.cat(rotations, dim=0)

	shs = []
	for i in objs:
		shs.append(pc.get_features[i][:pc.points_per_obj[i]
									].reshape(-1, (pc.max_sh_degree + 1) ** 2, 3))
	shs = torch.cat(shs, dim=0)

	if random.random() < shs_aug_ratio and not test:
		variance = (0.2 ** 0.5) * shs
		shs = shs + (torch.randn_like(shs) * variance)

	# add noise to scales
	if random.random() < scale_aug_ratio and not test:
		variance = (0.2 ** 0.5) * scales / 4
		scales = torch.clamp(
			scales + (torch.randn_like(scales) * variance), 0.0)

	sh_degree = pc.active_sh_degree

	viewmats = []
	for i in range(len(viewpoint_camera)):
		viewmat = viewpoint_camera[i].world_view_transform.transpose(0, 1)
		viewmats.append(viewmat)
	viewmats = torch.stack(viewmats, dim=0)
	width = int(viewpoint_camera[0].image_width)
	height = int(viewpoint_camera[0].image_height)

	bg_color = bg_color[None, :].repeat(B, 1)
	
	render_colors, render_alphas, info = rasterization(
		means=means3D,  # [N, 3]
		quats=rotations,  # [N, 4]
		scales=scales,  # [N, 3]
		opacities=opacity.squeeze(-1),  # [N,]
		colors=shs,
		viewmats=viewmats,  # [B, 4, 4]
		Ks=Ks,  # [B, 3, 3]
		backgrounds=bg_color,
		width=width,
		height=height,
		packed=False,
		sh_degree=sh_degree,
		absgrad=True,
	)
	  
	rendered_image = render_colors.permute(0, 3, 1, 2)  # [B, 3, H, W]
	
	# --- OPTIMIZED POST-PROCESSING ---
	
	# 1. Handle Radii shape (N, 2) -> (N)
	# info['radii'] shape: [B, Total_N, 2]
	b_radii = info['radii'].max(dim=-1).values # [B, Total_N]

	# 2. Vectorized Padding for Radii
	# shape: [B, Num_Objs, Max_Points]
	radii_padded = pad_packed_tensor(b_radii, current_points_per_obj, pad_value=0)

	# 3. Create Visibility Filter
	visibility_filter = (radii_padded > 0)

	if not test:
		info['means2d'].retain_grad()

	return {
		"render": rendered_image,
		"viewspace_points": info['means2d'],
		"visibility_filter": visibility_filter,
		"radii": radii_padded,
		"scales": scales,
		"points_per_obj_list": current_points_per_obj,
	}


def render_obj(
	viewpoint_camera: list, pc: GaussianModel, pipe, bg_color: torch.Tensor, obj: int,
	scaling_modifier=1.0, black_video=False, override_color=None,
	sh_deg_aug_ratio=0.1, bg_aug_ratio=0.3, shs_aug_ratio=1.0,
	scale_aug_ratio=1.0, test=False
):
	"""Render a single object using gsplat rasterization (BATCHED like `render`)."""
	# Camera intrinsics (batched)
	Ks = []
	B = len(viewpoint_camera)
	for i in range(B):
		tanfovx = math.tan(viewpoint_camera[i].FoVx * 0.5)
		tanfovy = math.tan(viewpoint_camera[i].FoVy * 0.5)
		fx = viewpoint_camera[i].image_width  / (2 * tanfovx)
		fy = viewpoint_camera[i].image_height / (2 * tanfovy)
		K = torch.tensor([[fx, 0, viewpoint_camera[i].image_width  / 2.0],
						  [0,  fy, viewpoint_camera[i].image_height / 2.0],
						  [0,   0, 1]], device="cuda", dtype=torch.float32)
		Ks.append(K)
	Ks = torch.stack(Ks, dim=0)  # [B,3,3]

	if black_video:
		bg_color = torch.zeros_like(bg_color)
	if random.random() < sh_deg_aug_ratio and not test:
		act_SH = 0
	else:
		act_SH = pc.active_sh_degree
	if random.random() < bg_aug_ratio and not test:
		bg_color = torch.rand_like(bg_color) if random.random() < 0.5 else torch.zeros_like(bg_color)

	# Gather single-object data
	n_pts = pc.points_per_obj[obj]
	means3D = pc.get_xyz[obj][:n_pts].clone()
	means3D = (means3D - means3D.mean(dim=0, keepdim=True).detach()).float()
	opacity  = pc.get_opacity[obj][:n_pts]
	scales   = (pc.get_scaling[obj][:n_pts].reshape(-1, 3) * scaling_modifier).float()
	rotations= pc.get_rotation[obj][:n_pts].reshape(-1, 4).float()
	shs      = pc.get_features[obj][:n_pts].reshape(-1, (pc.max_sh_degree + 1)**2, 3)

	if random.random() < shs_aug_ratio and not test:
		shs = shs + (torch.randn_like(shs) * (0.2 ** 0.5) * shs)
	if random.random() < scale_aug_ratio and not test:
		scales = torch.clamp(scales + torch.randn_like(scales) * 0.05, 0.0)

	# View matrices (batched)
	viewmats = []
	for i in range(B):
		viewmats.append(viewpoint_camera[i].world_view_transform.transpose(0, 1).float())
	viewmats = torch.stack(viewmats, dim=0)  # [B,4,4]

	width  = int(viewpoint_camera[0].image_width)
	height = int(viewpoint_camera[0].image_height)

	# Batched backgrounds [B,3]
	bg_batched = bg_color[None, :].repeat(B, 1)

	render_colors, render_alphas, info = rasterization(
		means=means3D,
		quats=rotations,
		scales=scales,
		opacities=opacity.squeeze(-1),
		colors=shs,
		viewmats=viewmats,
		Ks=Ks,
		backgrounds=bg_batched,
		width=width,
		height=height,
		packed=False,
		sh_degree=act_SH,
	)

	rendered_image = render_colors.permute(0, 3, 1, 2)  # [B,3,H,W]

	# --- OPTIMIZED POST-PROCESSING ---

	# 1. Handle Radii shape (N, 2) -> (N)
	# info['radii'] shape: [B, N, 2]
	b_radii = info["radii"].max(dim=-1).values # [B, N]

	# 2. Vectorized Padding
	# Even for a single object, we use the helper for consistency.
	# We pass a list containing the single point count.
	radii_padded = pad_packed_tensor(b_radii, [n_pts], pad_value=0) # [B, 1, N]

	# 3. Visibility
	visibility_filter = (radii_padded > 0)

	# 4. Means2d
	if not test:
		info['means2d'].retain_grad()

	return {
		"render": rendered_image,
		"viewspace_points": info['means2d'],
		"visibility_filter": visibility_filter,
		"radii": radii_padded,
		"scales": scales,
		"points_per_obj_list": [n_pts],
	}