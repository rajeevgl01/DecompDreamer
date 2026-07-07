import random
import imageio
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from random import randint
from utils.loss_utils import l1_loss, ssim, tv_loss
from gaussian_renderer import render, render_obj, network_gui
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


try:
	from torch.utils.tensorboard import SummaryWriter
	TENSORBOARD_FOUND = True
except ImportError:
	TENSORBOARD_FOUND = False

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def adjust_text_embeddings(embeddings, azimuth, guidance_opt):
	# Compute text embeddings and weights
	text_z, text_z_pooled, weights = get_pos_neg_text_embeddings(
		embeddings, azimuth, guidance_opt)

	# Determine the max sequence length K
	K = weights.shape[0]

	# Pad embeddings if necessary
	def pad_sequence(seq, length):
		return [seq[i] if i < len(seq) else seq[0] for i in range(length)]

	# Stack text embeddings
	text_embeddings = torch.stack(pad_sequence(text_z, K), dim=0)
	text_embeddings_pooled = torch.stack(pad_sequence(text_z_pooled, K), dim=0)

	# Stack weights
	weights_padded = torch.stack(pad_sequence(weights, K), dim=0)  # [B * K]

	return text_embeddings, text_embeddings_pooled, weights_padded


def get_pos_neg_text_embeddings(embeddings, azimuth_val, opt):
	if -90 <= azimuth_val < 90:
		r = 1 - abs(azimuth_val) / 90
		start_key, end_key = 'front', 'side'
	else:
		r = 1 - abs(azimuth_val - 90) / \
			90 if azimuth_val >= 0 else 1 + (azimuth_val + 90) / 90
		start_key, end_key = 'side', 'back'

	# Compute interpolated embeddings
	pos_z = r * embeddings[start_key] + (1 - r) * embeddings[end_key]
	text_z = torch.cat([pos_z, embeddings['front'], embeddings['side']], dim=0)

	pos_z_pooled = r * \
		embeddings[f"{start_key}_pooled"] + \
		(1 - r) * embeddings[f"{end_key}_pooled"]
	text_z_pooled = torch.cat(
		[pos_z_pooled, embeddings['front_pooled'], embeddings['side_pooled']], dim=0)

	# Compute negative weights
	front_neg_w = 0.0 if (r > 0.8 and start_key == 'front') else math.exp(
		-r * opt.front_decay_factor) * opt.negative_w
	side_neg_w = 0.0 if (r > 0.8 and start_key == 'side') else math.exp(-(1 - r)
																		* opt.side_decay_factor) * opt.negative_w / (2 if start_key == 'side' else 1)

	# Set weight tensor
	weights = torch.tensor(
		[1.0, side_neg_w, front_neg_w], device=text_z.device)

	return text_z, text_z_pooled, weights


def prepare_embeddings(guidance_opt, text, guidance, neg_text=None):
	embeddings = {}

	negative_prompt = [f'{neg_text}, {guidance_opt.negative}'] if neg_text else [
		guidance_opt.negative]

	embeddings['default'], embeddings['uncond'], embeddings['default_pooled'], embeddings['uncond_pooled'] = guidance.get_text_embeds(
		prompt=[text], negative_prompt=negative_prompt)

	# Directional embeddings
	for d in ['front', 'side', 'back']:
		embeddings[d], _, embeddings[f'{d}_pooled'], _ = guidance.get_text_embeds(
			prompt=[f"{text}, {d} view"], negative_prompt=negative_prompt)

	return embeddings

def guidance_setup(guidance_opt):
	from guidance.sd3_utils import StableDiffusion

	# Initialize Stable Diffusion guidance
	guidance = StableDiffusion(
		device=guidance_opt.g_device,
		fp16=guidance_opt.fp16,
		t_range=guidance_opt.t_range,
		max_t_range=guidance_opt.max_t_range,
		num_train_timesteps=guidance_opt.num_train_timesteps,
		guidance_opt=guidance_opt
	)

	# Freeze model parameters
	for param in guidance.parameters():
		param.requires_grad = False

	# Prepare embeddings
	embeddings = prepare_embeddings(guidance_opt, guidance_opt.text, guidance)

	obj_embeddings = [
		prepare_embeddings(guidance_opt, text, guidance, neg_text)
		for text, neg_text in zip(guidance_opt.prompt_obj, guidance_opt.prompt_obj_neg)
	]

	edge_embeddings = {
		edge: prepare_embeddings(guidance_opt, text, guidance)
		for edge, text in guidance_opt.prompt_edge.items()
	}

	return guidance, embeddings, obj_embeddings, edge_embeddings


def forward(opt, embeddings, objs, azimuth_offset, iteration, viewpoint_stack, scene, guidance_opt, debug_from, gaussians, pipe, background, dataset, guidance, use_control_net, save_folder, gcams):
	# progressively relaxing view range
	if not opt.use_progressive:
		if iteration >= opt.progressive_view_iter and iteration % opt.scale_up_cameras_iter == 0:
			scene.pose_args.fovy_range[0] = max(
				scene.pose_args.max_fovy_range[0], scene.pose_args.fovy_range[0] * opt.fovy_scale_up_factor[0])
			scene.pose_args.fovy_range[1] = min(
				scene.pose_args.max_fovy_range[1], scene.pose_args.fovy_range[1] * opt.fovy_scale_up_factor[1])

			scene.pose_args.radius_range[1] = max(
				scene.pose_args.max_radius_range[1], scene.pose_args.radius_range[1] * opt.scale_up_factor)
			scene.pose_args.radius_range[0] = max(
				scene.pose_args.max_radius_range[0], scene.pose_args.radius_range[0] * opt.scale_up_factor)

			scene.pose_args.theta_range[1] = min(
				scene.pose_args.max_theta_range[1], scene.pose_args.theta_range[1] * opt.phi_scale_up_factor)
			scene.pose_args.theta_range[0] = max(
				scene.pose_args.max_theta_range[0], scene.pose_args.theta_range[0] * 1/opt.phi_scale_up_factor)

			# opt.reset_resnet_iter = max(500, opt.reset_resnet_iter // 1.25)
			scene.pose_args.phi_range[0] = max(
				scene.pose_args.max_phi_range[0], scene.pose_args.phi_range[0] * opt.phi_scale_up_factor)
			scene.pose_args.phi_range[1] = min(
				scene.pose_args.max_phi_range[1], scene.pose_args.phi_range[1] * opt.phi_scale_up_factor)

			print('scale up theta_range to:', scene.pose_args.theta_range)
			print('scale up radius_range to:', scene.pose_args.radius_range)
			print('scale up phi_range to:', scene.pose_args.phi_range)
			print('scale up fovy_range to:', scene.pose_args.fovy_range)

	if len(objs) == 1:
		cam_scale = gcams.radius_params[objs[0]]
	elif len(objs) == 2:
		cam_scale = max(gcams.radius_params[objs[0]], gcams.radius_params[objs[1]])
	else:
		cam_scale = 1.0

	# Pick a random Camera
	if not viewpoint_stack:
		viewpoint_stack = scene.getRandTrainCameras(cam_scale).copy()

	C_batch_size = guidance_opt.C_batch_size
	viewpoint_cams = []
	images = []
	text_z_ = []
	text_z_pooled_ = []
	weights_ = []
	scales = []

	for i in range(C_batch_size):
		try:
			viewpoint_cam = viewpoint_stack.pop()
		except:
			viewpoint_stack = scene.getRandTrainCameras(cam_scale).copy()
			viewpoint_cam = viewpoint_stack.pop()

		# Predict text embeddings
		azimuth = (viewpoint_cam.delta_azimuth + azimuth_offset + 180) % 360 - 180

		# Initialize text embeddings with unconditional embeddings
		text_z = [embeddings['uncond']]
		text_z_pooled = [embeddings['uncond_pooled']]

		if guidance_opt.perpneg:
			# Compute adjusted text embeddings
			text_z_comp, text_z_pooled_comp, weights = adjust_text_embeddings(
				embeddings, azimuth, guidance_opt)

			# Append computed embeddings and weights
			text_z.extend([text_z_comp])
			text_z_pooled.extend([text_z_pooled_comp])
			weights_.append(weights)
		else:
			if -90 <= azimuth < 90:
				r = 1 - abs(azimuth) / 90
				start_key, end_key = 'front', 'side'
			else:
				r = 1 - abs(azimuth - 90) / \
					90 if azimuth >= 0 else 1 + (azimuth + 90) / 90
				start_key, end_key = 'side', 'back'

			text_z.append(r * embeddings[start_key] +
						  (1 - r) * embeddings[end_key])
			text_z_pooled.append(
				r * embeddings[f"{start_key}_pooled"] + (1 - r) * embeddings[f"{end_key}_pooled"])

		# Concatenate embeddings
		text_z_.append(torch.cat(text_z, dim=0))
		text_z_pooled_.append(torch.cat(text_z_pooled, dim=0))

		# Render
		if (iteration - 1) == debug_from:
			pipe.debug = True

		render_pkg = render(viewpoint_cam, gaussians, pipe, background, objs,
							sh_deg_aug_ratio=dataset.sh_deg_aug_ratio,
							bg_aug_ratio=dataset.bg_aug_ratio,
							shs_aug_ratio=dataset.shs_aug_ratio,
							scale_aug_ratio=dataset.scale_aug_ratio,
							object_scale=cam_scale)
		image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg[
			"viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

		scales.append(render_pkg["scales"])
		images.append(image)
		# to_pil = T.ToPILImage()
		# save = to_pil(image)
		# save.save(f"out_{iteration}_{i}_{len(objs)}_without.png")

		viewpoint_cams.append(viewpoint_cams)

	images = torch.stack(images, dim=0)

	# Loss
	warm_up_rate = 1. - min(iteration/opt.warmup_iter, 1.)
	guidance_scale = guidance_opt.guidance_scale
	if iteration > opt.use_control_net_iter:
		use_control_net = True

	train_step_fn = guidance.train_step_perpneg if guidance_opt.perpneg else guidance.train_step
	weights = torch.stack(weights_, dim=1) if guidance_opt.perpneg else None

	loss, actual_loss = train_step_fn(
		text_embeddings=torch.stack(text_z_, dim=1),
		pooled_text_embeddings=torch.stack(text_z_pooled_, dim=1),
		pred_rgb=images,
		grad_scale=guidance_opt.lambda_guidance,
		use_control_net=use_control_net,
		save_folder=save_folder,
		iteration=iteration,
		warm_up_rate=warm_up_rate,
		weights=weights,
		resolution=(gcams.image_h, gcams.image_w),
		guidance_opt=guidance_opt
	)

	scales = torch.stack(scales, dim=0)

	loss_scale = torch.mean(scales, dim=-1).mean()
	loss_tv = tv_loss(images)
	loss = loss + opt.lambda_tv * loss_tv + opt.lambda_scale * loss_scale
	return loss, actual_loss, image, viewspace_point_tensor, visibility_filter, radii


def weighting_function(current_iter, total_iters, num_objs, method="linear"):
	if current_iter <= 1:
		return torch.tensor(0.0)
	if current_iter >= total_iters / 2:
		return torch.tensor(1.0)

	current_iter = current_iter // num_objs
	total_iters = total_iters // num_objs
	progress = current_iter / (total_iters / 2)

	return 2 * torch.tensor(progress)**2

def training(dataset, opt, pipe, gcams, guidance_opt, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, save_video, cfg):
	first_iter = 0
	tb_writer = prepare_output_and_logger(dataset)
	gaussians = GaussianModel(dataset.sh_degree)
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

	#
	save_folder = os.path.join(dataset._model_path, "train_process/")
	if not os.path.exists(save_folder):
		os.makedirs(save_folder)  # makedirs
		print('train_process is in :', save_folder)
	# controlnet
	use_control_net = False
	edges = opt.edge_list
	azimuth_offsets = opt.azimuth_offsets
	gt_volumes = scene.gt_volumes
	# set up pretrain diffusion models and text_embedings
	guidance, embeddings, obj_embeddings, edge_embeddings = guidance_setup(
		guidance_opt)
	num_objs = opt.num_objs
	idx_list = [i for i in range(num_objs)]

	viewpoint_stack = None
	viewpoint_stack_around = None
	ema_loss_for_log = 0.0
	progress_bar = tqdm(range(first_iter, opt.iterations),
						desc="Training progress")
	first_iter += 1

	if opt.save_process:
		save_folder_proc = os.path.join(
			scene.args._model_path, "process_videos/")
		if not os.path.exists(save_folder_proc):
			os.makedirs(save_folder_proc)  # makedirs
		process_view_points = scene.getCircleVideoCameras(
			batch_size=opt.pro_frames_num, render45=opt.pro_render_45).copy()
		save_process_iter = opt.iterations // len(process_view_points)
		pro_img_frames = []

	def optimize_edges_and_scene(first_iter, total_iterations, stage_2_iters, num_edges=4, stage_1_scene_optimizations=2, stage_2_scene_optimizations=2):
		iteration = first_iter

		# First stage: Run the initial optimization until stage_2_iters
		while iteration < stage_2_iters:
			for edge_idx in range(num_edges):
				if iteration >= stage_2_iters:
					break
				yield ('edge', edge_idx, iteration)
				iteration += 1

			for scene_iter in range(stage_1_scene_optimizations):
				if iteration >= stage_2_iters:
					break
				yield ('scene', scene_iter, iteration)
				iteration += 1

		# Second stage: Run the second optimization until total_iterations
		while iteration < total_iterations:
			for edge_idx in range(num_edges):
				for obj_idx in range(2):
					if iteration >= total_iterations:
						break
					yield ('edge', edge_idx, obj_idx, iteration - stage_2_iters)
					iteration += 1

			for scene_iter in range(stage_2_scene_optimizations):
				if iteration >= total_iterations:
					break
				yield ('scene', scene_iter, 0, iteration - stage_2_iters)
				iteration += 1

	state_reset = False
	for values in optimize_edges_and_scene(first_iter, opt.iterations + 1, opt.stage_2_iters, len(edges), opt.scene_iter_1, opt.scene_iter_2):
		if len(values) == 3:
			operation, edge_index, iteration = values
			stage = 1
		else:
			operation, edge_index, obj_index, iteration = values
			stage = 2

		if not state_reset and stage == 2:
			opt.warmup_iter = int((opt.iterations - opt.stage_2_iters) * 0.3)
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
			"iteration": iteration,
			"viewpoint_stack": viewpoint_stack,
			"scene": scene,
			"guidance_opt": guidance_opt,
			"debug_from": debug_from,
			"gaussians": gaussians,
			"pipe": pipe,
			"background": background,
			"dataset": dataset,
			"guidance": guidance,
			"use_control_net": use_control_net,
			"save_folder": save_folder,
			"gcams": gcams
		}

		if stage == 1:
			if operation == 'edge':
				full_scene = False
				edge = edges[edge_index]
				edge_loss, actual_edge_loss, image, viewspace_point_tensor, visibility_filter, radii = forward(opt,
																							embeddings=edge_embeddings[str(
																								edge)], objs=edge, azimuth_offset=azimuth_offsets[str(edge)], **kwargs)
				wandb.log({f"loss/{edge}/edge_loss": edge_loss.item()})
				wandb.log({f"loss/{edge}/actual_edge_loss": actual_edge_loss.item()})
				obj_weight = weighting_function(iteration,
											opt.iterations, num_objs, "quadratic")
				loss = edge_loss
				selected_objs = [edge]
				vpt = [viewspace_point_tensor]
				vf = [visibility_filter]
				radiis = [radii]
				edge = random.sample(edge, 2)
				for obj in edge:
					obj_loss, actual_obj_loss, _, viewspace_point_tensor_obj, visibility_filter_obj, radii_obj = forward(opt,
																										embeddings=obj_embeddings[obj], objs=[obj], azimuth_offset=azimuth_offsets[str(obj)], **kwargs)
					if opt.size_lambda and opt.num_objs == 6:
						size_loss = gaussians.get_volume_loss(obj, gt_volumes[obj])
						wandb.log({"loss/obj_size_loss": size_loss.item()})
						loss += obj_weight * obj_loss + float(opt.size_lambda) * (1 - (iteration // num_objs) / (opt.iterations // num_objs)) * size_loss
					else:
						loss += obj_weight * obj_loss
					selected_objs += [[obj]]
					vpt += [viewspace_point_tensor_obj]
					vf += [visibility_filter_obj]
					radiis += [radii_obj]
					wandb.log({f"loss/{obj}/obj_loss": obj_loss.item()})
					wandb.log({f"loss/{obj}/actual_obj_loss": actual_obj_loss.item()})
			else:
				random.shuffle(edges)
				full_scene = True
				# full scene rendering
				loss,_, image, viewspace_point_tensor, visibility_filter, radii = forward(opt,
																						embeddings=embeddings, objs=idx_list, azimuth_offset=azimuth_offsets['global'], **kwargs)
				selected_objs = idx_list
		else:
			if operation == 'edge':
				full_scene = False
				edge = edges[edge_index]
				obj = edge[obj_index]
				edge_loss, actual_edge_loss, image, viewspace_point_tensor, visibility_filter, radii = forward(opt,
																								embeddings=edge_embeddings[str(
																									edge)], objs=edge, azimuth_offset=azimuth_offsets[str(edge)], **kwargs)
				wandb.log({f"loss/{edge}/edge_loss": edge_loss.item()})
				wandb.log({f"loss/{edge}/actual_edge_loss": actual_edge_loss.item()})
				loss = edge_loss
				selected_objs = [edge]
				vpt = [viewspace_point_tensor]
				vf = [visibility_filter]
				radiis = [radii]
				obj_loss, actual_obj_loss, _, viewspace_point_tensor_obj, visibility_filter_obj, radii_obj = forward(opt,
																									embeddings=obj_embeddings[obj], objs=[
																										obj], azimuth_offset=azimuth_offsets[str(obj)], **kwargs)
				if opt.size_lambda and opt.num_objs == 6:
					size_loss = gaussians.get_volume_loss(obj, gt_volumes[obj])
					wandb.log({"loss/obj_size_loss": size_loss.item()})
					loss += obj_loss + float(opt.size_lambda) * (1 - (iteration // num_objs) / (opt.iterations // num_objs)) * size_loss
				else:
					loss += obj_loss
				# loss += obj_loss
				selected_objs += [[obj]]
				vpt += [viewspace_point_tensor_obj]
				vf += [visibility_filter_obj]
				radiis += [radii_obj]
				wandb.log({f"loss/{obj}/obj_loss": obj_loss.item()})
				wandb.log({f"loss/{obj}/actual_obj_loss": actual_obj_loss.item()})
			else:
				random.shuffle(edges)
				full_scene = True
				# full scene rendering
				loss, actual_loss, image, viewspace_point_tensor, visibility_filter, radii = forward(opt,
																						embeddings=embeddings, objs=idx_list, azimuth_offset=azimuth_offsets['global'], **kwargs)
				selected_objs = idx_list

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

			# # Log and save
			# training_report(tb_writer, iteration, iter_start.elapsed_time(
			# 	iter_end), testing_iterations, scene, render, (pipe, background, idx_list))
			# if (iteration in testing_iterations):
			# 	if save_video:
			# 		video_inference(iteration, scene, render,
			# 						(pipe, background, idx_list), tb_writer)
			# 		for i in range(num_objs):
			# 			video_inference_obj(
			# 				iteration, scene, render_obj, (pipe, background, i), i, tb_writer)

			# if (iteration in saving_iterations):
			# 	print("\n[ITER {}] Saving Gaussians".format(iteration))
			# 	scene.save(iteration)

			# Densification
			if iteration < opt.densify_until_iter:
				# Keep track of max radii in image-space for pruning
				if full_scene:
					for j, i in enumerate(selected_objs):
						gaussians.max_radii2D[i, :gaussians.points_per_obj[i]][visibility_filter[j][:gaussians.points_per_obj[i]]] = torch.max(
							gaussians.max_radii2D[i, :gaussians.points_per_obj[i]][visibility_filter[j][:gaussians.points_per_obj[i]]], radii[j, :gaussians.points_per_obj[i]][visibility_filter[j][:gaussians.points_per_obj[i]]])
					gaussians.add_densification_stats(
						viewspace_point_tensor, visibility_filter, selected_objs)
				else:
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

			if stage == 2 and operation == 'edge':
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
	parser.add_argument("--test_ratio", type=int, default=5)
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
