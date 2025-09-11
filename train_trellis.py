import random
import imageio
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from random import randint
from utils.loss_utils import l1_loss, ssim, tv_loss
from gaussian_renderer import render, render_obj, render_trellis, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, GenerateCamParams, GuidanceParams
import math
from torchvision.utils import save_image
import torchvision.transforms as T
import wandb
import numpy as np
import datetime
from omegaconf import OmegaConf
os.environ['SPCONV_ALGO'] = 'native'
import imageio
from trellis.pipelines import TrellisTextTo3DPipeline, TrellisImageTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils
from diffusers import FluxPipeline, QwenImageEditPipeline
from PIL import Image

try:
	from torch.utils.tensorboard import SummaryWriter
	TENSORBOARD_FOUND = True
except ImportError:
	TENSORBOARD_FOUND = False

os.environ["TOKENIZERS_PARALLELISM"] = "false"

hashmap = {}

def forward(opt, objs, obj_azimuth_offsets, edge_azimuth_offsets, iteration, viewpoint_stack, scene, debug_from, gaussians, obj_gaussians, edge_gaussians, pipe, background, dataset):
	# Pick a random Camera
	if not viewpoint_stack:
		viewpoint_stack = scene.getTrellisCamera(obj_azimuth_offsets, edge_azimuth_offsets).copy()

	# Render
	if (iteration - 1) == debug_from:
		pipe.debug = True

	try:
		if len(objs) == 1:
			rand_idx = randint(0, len(viewpoint_stack[str(objs[0])]['pred']) - 1)
			pred_cam = viewpoint_stack[str(objs[0])]['pred'].pop(rand_idx)
			gt_cam = viewpoint_stack[str(objs[0])]['gt'].pop(rand_idx)
		else:
			rand_idx = randint(0, len(viewpoint_stack[f"{objs[0]}_{objs[1]}"]['pred']) - 1)
			pred_cam = viewpoint_stack[f"{objs[0]}_{objs[1]}"]['pred'].pop(rand_idx)
			gt_cam = viewpoint_stack[f"{objs[0]}_{objs[1]}"]['gt'].pop(rand_idx)
	except:
		viewpoint_stack = scene.getTrellisCamera(obj_azimuth_offsets, edge_azimuth_offsets).copy()
		if len(objs) == 1:
			rand_idx = randint(0, len(viewpoint_stack[str(objs[0])]['pred']) - 1)
			pred_cam = viewpoint_stack[str(objs[0])]['pred'].pop(rand_idx)
			gt_cam = viewpoint_stack[str(objs[0])]['gt'].pop(rand_idx)
		else:
			rand_idx = randint(0, len(viewpoint_stack[f"{objs[0]}_{objs[1]}"]['pred']) - 1)
			pred_cam = viewpoint_stack[f"{objs[0]}_{objs[1]}"]['pred'].pop(rand_idx)
			gt_cam = viewpoint_stack[f"{objs[0]}_{objs[1]}"]['gt'].pop(rand_idx)

	render_pkg_pred = render(pred_cam, gaussians, pipe, background, objs,
						sh_deg_aug_ratio=dataset.sh_deg_aug_ratio,
						bg_aug_ratio=dataset.bg_aug_ratio,
						shs_aug_ratio=dataset.shs_aug_ratio,
						scale_aug_ratio=dataset.scale_aug_ratio,
						scaling_modifier=1.0,)
	image, viewspace_point_tensor, visibility_filter, radii = render_pkg_pred["render"], render_pkg_pred[
		"viewspace_points"], render_pkg_pred["visibility_filter"], render_pkg_pred["radii"]

	if len(objs) == 1:
		if hashmap.get(str(objs[0])) and hashmap[str(objs[0])].get(gt_cam):
			image_gt = hashmap[str(objs[0])][gt_cam]
		else:
			render_pkg = render_trellis(gt_cam, obj_gaussians[str(objs[0])], pipe, background,sh_deg_aug_ratio=dataset.sh_deg_aug_ratio,
							bg_aug_ratio=dataset.bg_aug_ratio,
							shs_aug_ratio=dataset.shs_aug_ratio,
							scale_aug_ratio=dataset.scale_aug_ratio,
							scaling_modifier=1.0,)
			hashmap[str(objs[0])] = {gt_cam: {}}
			image_gt = render_pkg["render"]
			hashmap[str(objs[0])][gt_cam] = image_gt
	else:
		edge_key = f"{objs[0]}_{objs[1]}"
		if hashmap.get(edge_key) and hashmap[edge_key].get(gt_cam):
			image_gt = hashmap[edge_key][gt_cam]
		else:
			render_pkg = render_trellis(gt_cam, edge_gaussians[edge_key], pipe, background,sh_deg_aug_ratio=dataset.sh_deg_aug_ratio,
							bg_aug_ratio=dataset.bg_aug_ratio,
							shs_aug_ratio=dataset.shs_aug_ratio,
							scale_aug_ratio=dataset.scale_aug_ratio,
							scaling_modifier=1.0,)
			hashmap[edge_key] = {gt_cam: {}}
			image_gt = render_pkg["render"]
			hashmap[edge_key][gt_cam] = image_gt
	
	Ll1 = l1_loss(image, image_gt)
	Lssim = 1 - ssim(image, image_gt)
	# to_pil = T.ToPILImage()
	# save_pred = to_pil(image)
	# save_gt = to_pil(image_gt)
	# if len(objs) == 1:
	# 	save_pred.save(f"out_{objs[0]}_pred.png")
	# 	save_gt.save(f"out_{objs[0]}_gt.png")
	# else:
	# 	save_pred.save(f"out_{objs[0]}_{objs[1]}_pred.png")
	# 	save_gt.save(f"out_{objs[0]}_{objs[1]}_gt.png")
	loss = (1 - opt.ssim_lambda) * Ll1 + opt.ssim_lambda * Lssim
	return loss, image, viewspace_point_tensor, visibility_filter, radii, viewpoint_stack


def weighting_function(current_iter, total_iters, num_objs, method="linear"):
	if current_iter <= 1:
		return torch.tensor(0.0)
	if current_iter >= total_iters / 2:
		return torch.tensor(1.0)

	current_iter = current_iter // num_objs
	total_iters = total_iters // num_objs
	progress = current_iter / (total_iters / 2)
	
	return 2 * torch.tensor(progress)**2

def initialize(guidance_opt, save_dir):
	prompt = guidance_opt.text
	pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.bfloat16
    ).to("cuda")

	prompt = "a high quality photo of " + prompt
	image = pipe(prompt).images[0]

	image.save(os.path.join(save_dir, f"{prompt[:50]}.png"))
	
	del pipe
	torch.cuda.empty_cache()

	qwen_pipeline = QwenImageEditPipeline.from_pretrained(
		"Qwen/Qwen-Image-Edit",
		torch_dtype=torch.bfloat16
	).to("cuda")
	qwen_pipeline.set_progress_bar_config(disable=None)

	obj_text = guidance_opt.dino_obj_text
	image_keys = []
	for key, value in obj_text.items():
		image = Image.open(os.path.join(save_dir, f"{prompt[:50]}.png")).convert("RGB")
		for k in range(guidance_opt.num_objs):
			if k != key:
				removal_prompt = "remove the " + obj_text[k]
				inputs = {
					"image": image,
					"prompt": removal_prompt,
					"generator": torch.manual_seed(0),
					"true_cfg_scale": 4.0,
					"negative_prompt": " ",
					"num_inference_steps": 50,
				}

				with torch.inference_mode():
					output = qwen_pipeline(**inputs)
					output_image = output.images[0]
					output_image.save(os.path.join(save_dir, f"out_{key}.png"))
				image = output_image
				image_keys.append(str(key))

	for edge in guidance_opt.edge_list:
		image = Image.open(os.path.join(save_dir, f"{prompt[:50]}.png")).convert("RGB")
		for obj in range(guidance_opt.num_objs):
			if obj not in edge:
				removal_prompt = "remove the " + obj_text[obj]
				inputs = {
					"image": image,
					"prompt": removal_prompt,
					"generator": torch.manual_seed(0),
					"true_cfg_scale": 4.0,
					"negative_prompt": " ",
					"num_inference_steps": 50,
				}

				with torch.inference_mode():
					output = qwen_pipeline(**inputs)
					output_image = output.images[0]
			else:
				output_image = image
			output_image.save(os.path.join(save_dir, f"out_{edge[0]}_{edge[1]}.png"))
			image = output_image
		image_keys.append(f"{edge[0]}_{edge[1]}")
	
	pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
	pipeline.cuda()

	object_gaussians = {}
	edge_gaussians = {}
	for key in image_keys:
		image = Image.open(os.path.join(save_dir, f"out_{key}.png"))
		outputs = pipeline.run(image, seed=1)
		outputs['gaussian'][0].save_ply(os.path.join(save_dir, f"out_{key}.ply"))
		if "_" in key:
			edge_gaussians[key] = outputs['gaussian'][0]
		else:
			object_gaussians[key] = outputs['gaussian'][0]

	del pipeline
	torch.cuda.empty_cache()

	return object_gaussians, edge_gaussians

def generate_gt(opt, guidance_opt, gaussians, save_dir):
	import kaolin.ops.conversions as kconv
	pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
	pipeline.cuda()
	obj_gaussians = {}
	for i in range(opt.num_objs):
		voxel_grid = kconv.gs_to_voxelgrid(
			xyz=gaussians.get_xyz[i][:gaussians.points_per_obj[i]].cuda(),
			scales=gaussians.get_scaling[i][:gaussians.points_per_obj[i]].cuda(),
			rots=gaussians.get_rotation[i][:gaussians.points_per_obj[i]].cuda(),
			opacities=gaussians.get_opacity[i][:gaussians.points_per_obj[i]].cuda(),
			tol=1/64,
			level=6,
			step=100
		)
		coordinates, opacities = voxel_grid
		mask = torch.where(opacities > 0)
		coords = coordinates[mask].int()
		coords = torch.cat([torch.zeros(coords.shape[0], 1, device=coords.device), coords], dim=1).int()
		output = pipeline.run_gt(coords, Image.open(os.path.join(save_dir, f"segmented_{i}.png")))
		output['gaussian'][0].save_ply(os.path.join("./assets/test", f"gt_{i}.ply"))
		obj_gaussians[f"{i}"] = output['gaussian'][0]

	edge_gaussians = {}
	for edge in opt.edge_list:
		xyz = []
		scales = []
		rots = []
		opacities = []
		for obj in edge:
			xyz.append(gaussians.get_xyz[obj][:gaussians.points_per_obj[obj]].cuda())
			scales.append(gaussians.get_scaling[obj][:gaussians.points_per_obj[obj]].cuda())
			rots.append(gaussians.get_rotation[obj][:gaussians.points_per_obj[obj]].cuda())
			opacities.append(gaussians.get_opacity[obj][:gaussians.points_per_obj[obj]].cuda())
		xyz = torch.cat(xyz, dim=0)
		scales = torch.cat(scales, dim=0)
		rots = torch.cat(rots, dim=0)
		opacities = torch.cat(opacities, dim=0)

		voxel_grid = kconv.gs_to_voxelgrid(
			xyz=xyz,
			scales=scales,
			rots=rots,
			opacities=opacities,
			tol=1/64,
			level=6,
			step=100
		)
		coordinates, opacities = voxel_grid
		mask = torch.where(opacities > 0)
		coords = coordinates[mask]
		coords = torch.cat([torch.zeros(coords.shape[0], 1, device=coords.device), coords], dim=1).int()
		output = pipeline.run_gt(coords, Image.open(os.path.join(save_dir, f"segmented_{edge[0]}_{edge[1]}.png")))
		output['gaussian'][0].save_ply(os.path.join("./assets/test", f"gt_{edge[0]}_{edge[1]}.ply"))
		edge_gaussians[f"{edge[0]}_{edge[1]}"] = output['gaussian'][0]

	xyz = []
	scales = []
	rots = []
	opacities = []
	for i in range(opt.num_objs):
		xyz.append(gaussians.get_xyz[i][:gaussians.points_per_obj[i]].cuda())
		scales.append(gaussians.get_scaling[i][:gaussians.points_per_obj[i]].cuda())
		rots.append(gaussians.get_rotation[i][:gaussians.points_per_obj[i]].cuda())
		opacities.append(gaussians.get_opacity[i][:gaussians.points_per_obj[i]].cuda())
	xyz = torch.cat(xyz, dim=0)
	scales = torch.cat(scales, dim=0)
	rots = torch.cat(rots, dim=0)
	opacities = torch.cat(opacities, dim=0)

	# voxel_grid = kconv.gs_to_voxelgrid(
	# 	xyz=xyz,
	# 	scales=scales,
	# 	rots=rots,
	# 	opacities=opacities,
	# 	tol=1/64,
	# 	level=6,
	# 	step=100
	# )
	# coordinates, opacities = voxel_grid
	# mask = torch.where(opacities > 0)
	# coords = coordinates[mask]
	# coords = torch.cat([torch.zeros(coords.shape[0], 1, device=coords.device), coords], dim=1).int()
	# output = pipeline.run_gt(coords, Image.open(guidance_opt.text), seed=1)
	# output['gaussian'][0].save_ply(os.path.join("./assets/test", f"gt_scene.ply"))

	return obj_gaussians, edge_gaussians, None

def training(dataset, opt, pipe, gcams, guidance_opt, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, save_video, cfg):
	first_iter = 0
	tb_writer = prepare_output_and_logger(dataset)
	gaussians = GaussianModel(dataset.sh_degree)
	obj_gaussians, edge_gaussians = initialize(guidance_opt, save_dir="./assets/test")
	gcams.init_prompt = [os.path.join("./assets/test", f"out_{i}.ply") for i in range(opt.num_objs)]
	scene = Scene(dataset, gcams, gaussians)
	gaussians.training_setup(opt)
	if checkpoint:
		(model_params, first_iter) = torch.load(checkpoint)
		gaussians.restore(model_params, opt)

	wandb.tensorboard.patch(str(dataset._model_path))
	timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
	wandb.init(
		project="decompdreamer",
		name=f"{guidance_opt.wandb_name}_{timestamp}",
		sync_tensorboard=True,
		config=cfg,
		dir=dataset._model_path
	)

	bg_color = [1, 1, 1] if dataset._white_background else [0, 0, 0]
	background = torch.tensor(
		bg_color, dtype=torch.float32, device=dataset.data_device)
	iter_start = torch.cuda.Event(enable_timing=True)
	iter_end = torch.cuda.Event(enable_timing=True)

	save_folder = os.path.join(dataset._model_path, "train_process/")
	if not os.path.exists(save_folder):
		os.makedirs(save_folder)  # makedirs
		print('train_process is in :', save_folder)

	edges = opt.edge_list
	num_objs = opt.num_objs
	idx_list = [i for i in range(num_objs)]
	# obj_gaussians, edge_gaussians, scene_gaussians = generate_gt(opt, guidance_opt, gaussians, save_dir="./assets/test")

	viewpoint_stack = None
	ema_loss_for_log = 0.0
	progress_bar = tqdm(range(first_iter, opt.iterations),
						desc="Training progress")
	first_iter += 1

	if opt.save_process:
		save_folder_proc = os.path.join(
			scene.args._model_path, "process_videos/")
		if not os.path.exists(save_folder_proc):
			os.makedirs(save_folder_proc)  # makedirs
		process_view_points = scene.getCameraAtZeroAzimuth().copy()
		# process_view_points = scene.getRandTrainCameras().copy()
		save_process_iter = opt.iterations // len(process_view_points)
		pro_img_frames = []
	
	for i in range(len(process_view_points)):
		viewpoint_cam_p = process_view_points[0]
		render_p = render(viewpoint_cam_p, gaussians,
							pipe, background, idx_list, test=True)
		img_p = torch.clamp(render_p["render"], 0.0, 1.0)
		img_p = img_p.detach().cpu().permute(1, 2, 0).numpy()
		img_p = (img_p * 255).round().astype('uint8')
		pro_img_frames.append(img_p)

	imageio.mimwrite(os.path.join(save_folder_proc,
									  "video_rgb.mp4"), pro_img_frames, fps=30, quality=8)

	process_view_points = scene.getCameraAtZeroAzimuthTrellis().copy()
	for key in obj_gaussians.keys():
		pro_img_frames = []
		for i in range(len(process_view_points)):
			viewpoint_cam_p = process_view_points[0]
			render_p = render_trellis(viewpoint_cam_p, obj_gaussians[key],
							pipe, background)
			img_p = torch.clamp(render_p["render"], 0.0, 1.0)
			img_p = img_p.detach().cpu().permute(1, 2, 0).numpy()
			img_p = (img_p * 255).round().astype('uint8')
			pro_img_frames.append(img_p)

		imageio.mimwrite(os.path.join(save_folder_proc,
										f"video_rgb_obj_{key}.mp4"), pro_img_frames, fps=30, quality=8)
	
	for key in edge_gaussians.keys():
		pro_img_frames = []
		for i in range(len(process_view_points)):
			viewpoint_cam_p = process_view_points[0]
			render_p = render_trellis(viewpoint_cam_p, edge_gaussians[key],
							pipe, background)
			img_p = torch.clamp(render_p["render"], 0.0, 1.0)
			img_p = img_p.detach().cpu().permute(1, 2, 0).numpy()
			img_p = (img_p * 255).round().astype('uint8')
			pro_img_frames.append(img_p)

		imageio.mimwrite(os.path.join(save_folder_proc,
										f"video_rgb_edge_{key}.mp4"), pro_img_frames, fps=30, quality=8)

	def optimize_edges_and_scene(first_iter, total_iterations, stage_2_iters, num_edges=4):
		iteration = first_iter

		# First stage: Run the initial optimization until stage_2_iters
		while iteration < stage_2_iters:
			for edge_idx in range(num_edges):
				if iteration >= stage_2_iters:
					break
				yield (edge_idx, iteration)
				iteration += 1

		# Second stage: Run the second optimization until total_iterations
		while iteration < total_iterations:
			for edge_idx in range(num_edges):
				for obj_idx in range(2):
					if iteration >= total_iterations:
						break
					yield (edge_idx, obj_idx, iteration - stage_2_iters)
					iteration += 1

	state_reset = False
	for values in optimize_edges_and_scene(first_iter, opt.iterations + 1, opt.stage_2_iters, len(edges)):
		if len(values) == 2:
			edge_index, iteration = values
			stage = 1
		else:
			edge_index, obj_index, iteration = values
			stage = 2

		if not state_reset and stage == 2:
			opt.densify_until_iter -= opt.stage_2_iters
			opt.iterations -= opt.stage_2_iters
			gaussians.training_setup(opt)
			state_reset = True

		# TODO: DEBUG NETWORK_GUI
		if network_gui.conn == None:
			network_gui.try_connect()
		while network_gui.conn != None:
			try:
				net_image_bytes = None
				custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
				if custom_cam != None:
					net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)[
						"render"]
					net_image_bytes = memoryview((torch.clamp(
						net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
				network_gui.send(net_image_bytes, guidance_opt.text)
				if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
					break
			except Exception as e:
				network_gui.conn = None

		iter_start.record()

		gaussians.update_learning_rate(iteration)
		gaussians.update_feature_learning_rate(iteration)
		gaussians.update_rotation_learning_rate(iteration)
		gaussians.update_scaling_learning_rate(iteration)
		# Every 500 its we increase the levels of SH up to a maximum degree
		if iteration % 500 == 0:
			gaussians.oneupSHdegree()

		kwargs = {
			"obj_azimuth_offsets": opt.obj_azimuth_offset,
			"edge_azimuth_offsets": opt.edge_azimuth_offset,
			"iteration": iteration,
			"viewpoint_stack": viewpoint_stack,
			"scene": scene,
			"debug_from": debug_from,
			"gaussians": gaussians,
			"obj_gaussians": obj_gaussians,
			"edge_gaussians": edge_gaussians,
			"pipe": pipe,
			"background": background,
			"dataset": dataset
		}

		if stage == 1:
			edge = edges[edge_index]
			edge_loss, image, viewspace_point_tensor, visibility_filter, radii, viewpoint_stack = forward(opt, objs=edge, **kwargs)
			wandb.log({"loss/edge_loss": edge_loss.item()})
			obj_weight = weighting_function(iteration,
										opt.iterations, num_objs, "quadratic")
			loss = edge_loss
			selected_objs = [edge]
			vpt = [viewspace_point_tensor]
			vf = [visibility_filter]
			radiis = [radii]
			edge = random.sample(edge, 2)
			for obj in edge:
				obj_loss, _, viewspace_point_tensor_obj, visibility_filter_obj, radii_obj, viewpoint_stack = forward(opt, objs=[obj], **kwargs)
				loss += obj_weight * obj_loss
				selected_objs += [[obj]]
				vpt += [viewspace_point_tensor_obj]
				vf += [visibility_filter_obj]
				radiis += [radii_obj]
				wandb.log({"loss/obj_loss": obj_loss.item()})
			# if iteration == 2:
			# 	exit()
		else:
			edge = edges[edge_index]
			obj = edge[obj_index]
			edge_loss, image, viewspace_point_tensor, visibility_filter, radii, viewpoint_stack = forward(opt, objs=edge, **kwargs)
			wandb.log({"loss/edge_loss": edge_loss.item()})
			loss = edge_loss
			selected_objs = [edge]
			vpt = [viewspace_point_tensor]
			vf = [visibility_filter]
			radiis = [radii]
			obj_loss, _, viewspace_point_tensor_obj, visibility_filter_obj, radii_obj, viewpoint_stack = forward(opt, objs=[obj], **kwargs)
			loss += obj_loss
			selected_objs += [[obj]]
			vpt += [viewspace_point_tensor_obj]
			vf += [visibility_filter_obj]
			radiis += [radii_obj]
			wandb.log({"loss/obj_loss": obj_loss.item()})

		wandb.log({"loss/graph_loss": loss.item()})
		if stage == 2:
			previous_state = gaussians.capture()
		loss.backward()
		iter_end.record()

		with torch.no_grad():
			# Progress bar
			if stage == 2:
				iteration += opt.stage_2_iters
			ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
			if opt.save_process:
				if iteration % save_process_iter == 0 and len(process_view_points) > 0:
					viewpoint_cam_p = process_view_points.pop(0)
					render_p = render(viewpoint_cam_p, gaussians,
									  pipe, background, idx_list, test=True)
					img_p = torch.clamp(render_p["render"], 0.0, 1.0)
					img_p = img_p.detach().cpu().permute(1, 2, 0).numpy()
					img_p = (img_p * 255).round().astype('uint8')
					pro_img_frames.append(img_p)

			if iteration % 10 == 0:
				progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
				progress_bar.update(10)
			if iteration == opt.iterations:
				progress_bar.close()

			# Log and save
			training_report(tb_writer, iteration, iter_start.elapsed_time(
				iter_end), testing_iterations, scene, render, (pipe, background, idx_list))
			if (iteration in testing_iterations):
				if save_video:
					video_inference(iteration, scene, render,
									(pipe, background, idx_list), tb_writer)
					for i in range(num_objs):
						video_inference_obj(
							iteration, scene, render_obj, (pipe, background, i), i, tb_writer)

			if (iteration in saving_iterations):
				print("\n[ITER {}] Saving Gaussians".format(iteration))
				scene.save(iteration)

			# Densification
			if iteration < opt.densify_until_iter:
				for k in range(len(vf)):
					visibility_filter = vf[k]
					viewspace_point_tensor = vpt[k]
					radii = radiis[k]
					for j, i in enumerate(selected_objs[k]):
						gaussians.max_radii2D[i, :gaussians.points_per_obj[i]][visibility_filter[j][:gaussians.points_per_obj[i]]] = torch.max(
							gaussians.max_radii2D[i, :gaussians.points_per_obj[i]][visibility_filter[j][:gaussians.points_per_obj[i]]], radii[j, :gaussians.points_per_obj[i]][visibility_filter[j][:gaussians.points_per_obj[i]]])
					gaussians.add_densification_stats(
						viewspace_point_tensor, visibility_filter, selected_objs[k])

				if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
					size_threshold = 20 if iteration > opt.opacity_reset_interval else None
					gaussians.densify_and_prune(
						opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)

				if iteration % opt.opacity_reset_interval == 0:
					gaussians.reset_opacity()
			# Optimizer step
			if iteration < opt.iterations:
				gaussians.optimizer.step()
				gaussians.optimizer.zero_grad(set_to_none=True)

			if stage == 2:
				gaussians.reinit_from_state_dict(previous_state, obj)

			if (iteration in checkpoint_iterations):
				print("\n[ITER {}] Saving Checkpoint".format(iteration))
				torch.save((gaussians.capture(), iteration), os.path.join(scene.args._model_path, f"chkpnt_{str(iteration)}.pth"))

	if opt.save_process:
		imageio.mimwrite(os.path.join(save_folder_proc,
									  "video_rgb.mp4"), pro_img_frames, fps=30, quality=8)


def prepare_output_and_logger(args):
	if not args._model_path:
		if os.getenv('OAR_JOB_ID'):
			unique_str = os.getenv('OAR_JOB_ID')
		else:
			unique_str = str(uuid.uuid4())
		args._model_path = os.path.join("./output/", args.workspace)

	# Set up output folder
	print("Output folder: {}".format(args._model_path))
	os.makedirs(args._model_path, exist_ok=True)

	# copy configs
	if args.opt_path is not None:
		os.system(
			' '.join(['cp', args.opt_path, os.path.join(args._model_path, 'config.yaml')]))

	with open(os.path.join(args._model_path, "cfg_args"), 'w') as cfg_log_f:
		cfg_log_f.write(str(Namespace(**vars(args))))

	# Create Tensorboard writer
	tb_writer = None
	if TENSORBOARD_FOUND:
		tb_writer = SummaryWriter(args._model_path)
	else:
		print("Tensorboard not available: not logging progress")
	return tb_writer


def training_report(tb_writer, iteration, elapsed, testing_iterations, scene: Scene, renderFunc, renderArgs):
	if tb_writer:
		tb_writer.add_scalar('iter_time', elapsed, iteration)
	# Report test and samples of training set
	if iteration in testing_iterations:
		save_folder = os.path.join(
			scene.args._model_path, "test_six_views/{}_iteration".format(iteration))
		if not os.path.exists(save_folder):
			os.makedirs(save_folder)
			print('test views is in :', save_folder)
		torch.cuda.empty_cache()
		config = ({'name': 'test', 'cameras': scene.getTestCameras()})
		if config['cameras'] and len(config['cameras']) > 0:
			for iteration, viewpoint in enumerate(config['cameras']):
				render_out = renderFunc(
					viewpoint, scene.gaussians, *renderArgs, test=True)
				rgb = render_out["render"]

				image = torch.clamp(rgb, 0.0, 1.0)
				save_image(image, os.path.join(
					save_folder, "render_view_{}.png".format(viewpoint.uid)))
				if tb_writer:
					tb_writer.add_images(config['name'] + "_view_{}/render".format(
						viewpoint.uid), image[None], global_step=iteration)
			print("\n[ITER {}] Eval Done!".format(iteration))
		if tb_writer:
			tb_writer.add_histogram(
				"scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
			tb_writer.add_scalar(
				'total_points', scene.gaussians.get_xyz.shape[0], iteration)
		torch.cuda.empty_cache()


def video_inference(iteration, scene: Scene, renderFunc, renderArgs, tb_writer):
	sharp = T.RandomAdjustSharpness(3, p=1.0)

	save_folder = os.path.join(
		scene.args._model_path, "videos/{}_iteration".format(iteration))
	if not os.path.exists(save_folder):
		os.makedirs(save_folder)  # makedirs
		print('videos is in :', save_folder)
	torch.cuda.empty_cache()
	config = ({'name': 'test', 'cameras': scene.getCircleVideoCameras()})
	if config['cameras'] and len(config['cameras']) > 0:
		img_frames = []
		print("Generating Video using", len(
			config['cameras']), "different view points")
		for idx, viewpoint in enumerate(config['cameras']):
			render_out = renderFunc(
				viewpoint, scene.gaussians, *renderArgs, test=True)
			rgb = render_out["render"]

			image = torch.clamp(rgb, 0.0, 1.0)
			image = image.detach().cpu().permute(1, 2, 0).numpy()
			image = (image * 255).round().astype('uint8')
			img_frames.append(image)

		imageio.mimwrite(os.path.join(save_folder, "video_rgb_{}.mp4".format(
			iteration)), img_frames, fps=30, quality=8)

		wandb.log(
			{"video/rgb_graph": wandb.Video(np.array(img_frames).transpose(0, 3, 1, 2), fps=30)})

		print("\n[ITER {}] Video Save Done!".format(iteration))
	torch.cuda.empty_cache()


def video_inference_obj(iteration, scene: Scene, renderFunc, renderArgs, obj, tb_writer):
	sharp = T.RandomAdjustSharpness(3, p=1.0)

	save_folder = os.path.join(
		scene.args._model_path, "videos/{}_iteration".format(iteration))
	if not os.path.exists(save_folder):
		os.makedirs(save_folder)  # makedirs
		print('videos is in :', save_folder)
	torch.cuda.empty_cache()
	config = ({'name': 'test', 'cameras': scene.getCircleVideoCameras()})
	if config['cameras'] and len(config['cameras']) > 0:
		img_frames = []
		print("Generating Video using", len(
			config['cameras']), "different view points")
		for idx, viewpoint in enumerate(config['cameras']):
			render_out = renderFunc(
				viewpoint, scene.gaussians, *renderArgs, test=True)
			rgb = render_out["render"]

			image = torch.clamp(rgb, 0.0, 1.0)
			image = image.detach().cpu().permute(1, 2, 0).numpy()
			image = (image * 255).round().astype('uint8')
			img_frames.append(image)

		imageio.mimwrite(os.path.join(save_folder, "video_rgb_obj_{}_{}.mp4".format(
			obj, iteration)), img_frames, fps=30, quality=8)

		wandb.log(
			{f"video/rgb_obj_{obj}": wandb.Video(np.array(img_frames).transpose(0, 3, 1, 2), fps=30)})

		print("\n[ITER {}] Video Save Done!".format(iteration))
	torch.cuda.empty_cache()


if __name__ == "__main__":
	import yaml

	# Set up command line argument parser
	parser = ArgumentParser(description="Training script parameters")

	parser.add_argument('--opt', type=str, default=None)
	parser.add_argument('--ip', type=str, default="127.0.0.1")
	parser.add_argument('--port', type=int, default=6009)
	parser.add_argument('--debug_from', type=int, default=-1)
	parser.add_argument('--seed', type=int, default=0)
	parser.add_argument('--detect_anomaly', action='store_true', default=False)
	parser.add_argument("--test_ratio", type=int, default=10)
	parser.add_argument("--save_ratio", type=int, default=2)
	parser.add_argument("--save_video", type=bool, default=False)
	parser.add_argument("--quiet", action="store_true")
	parser.add_argument("--checkpoint_ratio", type=int, default=10)
	parser.add_argument("--start_checkpoint", type=str, default=None)

	lp = ModelParams(parser)
	op = OptimizationParams(parser)
	pp = PipelineParams(parser)
	gcp = GenerateCamParams(parser)
	gp = GuidanceParams(parser)

	args = parser.parse_args(sys.argv[1:])

	if args.opt is not None:
		with open(args.opt) as f:
			opts = yaml.load(f, Loader=yaml.FullLoader)
		lp.load_yaml(opts.get('ModelParams', None))
		op.load_yaml(opts.get('OptimizationParams', None))
		pp.load_yaml(opts.get('PipelineParams', None))
		gcp.load_yaml(opts.get('GenerateCamParams', None))
		gp.load_yaml(opts.get('GuidanceParams', None))

		lp.opt_path = args.opt
		args.port = opts['port']
		args.save_video = opts.get('save_video', True)
		args.seed = opts.get('seed', 0)
		args.device = opts.get('device', 'cuda')

		# override device
		gp.g_device = args.device
		lp.data_device = args.device
		gcp.device = args.device

	# save iterations
	test_iter = [1] + [k * op.iterations //
					   args.test_ratio for k in range(1, args.test_ratio)] + [op.iterations]
	args.test_iterations = test_iter

	save_iter = [k * op.iterations //
				 args.save_ratio for k in range(1, args.save_ratio)] + [op.iterations]
	args.save_iterations = save_iter
	args.checkpoint_iterations = [
		k * op.iterations // args.checkpoint_ratio for k in range(1, args.checkpoint_ratio)] + [op.iterations]

	print('Test iter:', args.test_iterations)
	print('Save iter:', args.save_iterations)

	print("Optimizing " + lp._model_path)

	# Initialize system state (RNG)
	safe_state(args.quiet, seed=args.seed)
	# Start GUI server, configure and run training
	network_gui.init(args.ip, args.port)
	torch.autograd.set_detect_anomaly(args.detect_anomaly)
	training(lp, op, pp, gcp, gp, args.test_iterations, args.save_iterations,
			 args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.save_video, opts)

	# All done
	print("\nTraining complete.")
