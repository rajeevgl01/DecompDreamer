from .perpneg_utils import weighted_perpendicular_aggregator
from torch.amp import custom_bwd, custom_fwd
from torchvision.utils import save_image
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
from transformers import logging
from diffusers import StableDiffusion3Pipeline, FlowMatchEulerDiscreteScheduler, SD3ControlNetModel
from pathlib import Path
import os
import random

import torchvision.transforms as T
# suppress partial model loading warning
logging.set_verbosity_error()


class SpecifyGradient(torch.autograd.Function):
    @staticmethod
    @custom_fwd(device_type='cuda')
    def forward(ctx, input_tensor, gt_grad):
        ctx.save_for_backward(gt_grad)
        # we return a dummy value 1, which will be scaled by amp's scaler so we get the scale in backward.
        return torch.ones([1], device=input_tensor.device, dtype=input_tensor.dtype)

    @staticmethod
    @custom_bwd(device_type='cuda')
    def backward(ctx, grad_scale):
        gt_grad, = ctx.saved_tensors
        gt_grad = gt_grad * grad_scale
        return gt_grad, None


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


class StableDiffusion(nn.Module):
    def __init__(self, device, fp16, t_range=[0.02, 0.98], max_t_range=0.98, num_train_timesteps=None,
                 use_control_net=False, guidance_opt=None):
        super().__init__()

        self.device = device
        self.precision_t = torch.float16

        model_key = "stabilityai/stable-diffusion-3.5-medium" if guidance_opt.model_key is None else guidance_opt.model_key

        pipe = StableDiffusion3Pipeline.from_pretrained(
            model_key, torch_dtype=self.precision_t)

        if use_control_net:
            controlnet = SD3ControlNetModel.from_pretrained(
                "stabilityai/stable-diffusion-3.5-large-controlnet-depth", torch_dtype=self.precision_t).to(self.device)
            controlnet.pos_embed = controlnet._get_pos_embed_from_transformer(pipe.transformer).to(self.precision_t).to(self.device)
            self.controlnet_cond_scale = 0.5 if guidance_opt.controlnet_cond_scale is None else guidance_opt.controlnet_cond_scale
            self.controlnet = controlnet.eval()

        self.scheduler = pipe.scheduler

        pipe = pipe.to(self.device)

        self.pipe = pipe

        self.vae = pipe.vae
        self.transformer = pipe.transformer

        self.num_train_timesteps = num_train_timesteps if num_train_timesteps is not None else self.scheduler.config.num_train_timesteps
        self.scheduler.set_timesteps(self.num_train_timesteps, device=device)

        self.timesteps = torch.flip(self.scheduler.timesteps, dims=(0, ))
        self.sigmas = torch.flip(
            self.scheduler.sigmas.to(self.device), dims=(0, ))
        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])
        self.warmup_step = int(
            self.num_train_timesteps*(max_t_range-t_range[1]))

        self.noise_gen = torch.Generator(self.device)
        self.noise_gen.manual_seed(guidance_opt.noise_seed)

        print(f'[INFO] loaded stable diffusion!')

    @torch.no_grad()
    def get_text_embeds(self, prompt, negative_prompt):
        return self.pipe.encode_prompt(prompt=prompt, prompt_2=None, prompt_3=None, negative_prompt=negative_prompt)

    def train_step_perpneg(self, text_embeddings, pooled_text_embeddings, pred_rgb, control_image=None,
                           grad_scale=1, use_control_net=False,
                           save_folder: Path = None, iteration=0, warm_up_rate=0, weights=0,
                           resolution=(512, 512), guidance_opt=None):
        B = pred_rgb.shape[0]
        K = text_embeddings.shape[0] - 1

        latents = self.encode_imgs(pred_rgb.to(self.precision_t))

        weights = weights.reshape(-1)
        noise = torch.randn(latents.shape, dtype=latents.dtype, device=latents.device,
                            generator=self.noise_gen) + 0.1 * torch.randn((1, 16, 1, 1), device=latents.device).repeat(latents.shape[0], 1, 1, 1)
        
        text_embeddings = text_embeddings.reshape(
            -1, text_embeddings.shape[-2], text_embeddings.shape[-1])
        pooled_text_embeddings = pooled_text_embeddings.reshape(
            -1, pooled_text_embeddings.shape[-1])

        ind_t = torch.randint(self.min_step, self.max_step + int(self.warmup_step * warm_up_rate),
                              (1, ), dtype=torch.long, generator=self.noise_gen, device=self.device)[0]

        sigma = self.sigmas[ind_t]
        t = self.timesteps[ind_t]

        with torch.no_grad():
            latents_noisy = (1 - sigma) * latents + sigma * noise
            target = noise - latents
            latent_model_input = latents_noisy[None, :, ...].repeat(
                1 + K, 1, 1, 1, 1).reshape(-1, 16, resolution[0] // 8, resolution[1] // 8, )
            tt = t.reshape(1, 1).repeat(
                latent_model_input.shape[0], 1).reshape(-1)

            control_block_samples = None
            if use_control_net:
                control_image = self.encode_imgs(control_image.to(self.precision_t))
                control_image = control_image[None, :, ...].repeat(
                    1 + K, 1, 1, 1, 1).reshape(-1, 16, resolution[0] // 8, resolution[1] // 8,)
                
                if self.controlnet.config.force_zeros_for_pooled_projection:
                    controlnet_pooled_projections = torch.zeros_like(pooled_text_embeddings)
                else:
                    controlnet_pooled_projections = pooled_text_embeddings

                control_block_samples = self.controlnet(
                    hidden_states=latent_model_input.to(self.precision_t),
                    timestep=tt.to(self.precision_t),
                    pooled_projections=controlnet_pooled_projections.to(
                        self.precision_t),
                    controlnet_cond=control_image,
                    conditioning_scale=self.controlnet_cond_scale,
                    return_dict=False,
                )[0]
            noise_pred = self.transformer(
                hidden_states=latent_model_input.to(self.precision_t),
                timestep=tt.to(self.precision_t),
                encoder_hidden_states=text_embeddings.to(self.precision_t),
                pooled_projections=pooled_text_embeddings.to(self.precision_t),
                block_controlnet_hidden_states=control_block_samples,
                return_dict=False,
            )[0]

            noise_pred = noise_pred.reshape(
                1 + K, -1, 16, resolution[0] // 8, resolution[1] // 8, )
            noise_pred_uncond, noise_pred_text = noise_pred[:1].reshape(
                -1, 16, resolution[0] // 8, resolution[1] // 8, ), noise_pred[1:].reshape(-1, 16, resolution[0] // 8, resolution[1] // 8, )
            delta_noise_preds = noise_pred_text - \
                noise_pred_uncond.repeat(K, 1, 1, 1)
            delta_DSD = weighted_perpendicular_aggregator(delta_noise_preds,
                                                          weights,
                                                          B)

        pred_noise = noise_pred_uncond + guidance_opt.guidance_scale * delta_DSD

        grad = sigma * (pred_noise - target)

        grad = torch.nan_to_num(grad_scale * grad)
        loss = SpecifyGradient.apply(latents, grad)
        return loss

    def train_step(self, text_embeddings, pooled_text_embeddings, pred_rgb, control_image=None,
                   grad_scale=1, use_control_net=False,
                   save_folder: Path = None, iteration=0, warm_up_rate=0, weights=0,
                   resolution=(512, 512), guidance_opt=None):
        B = pred_rgb.shape[0]

        latents = self.encode_imgs(pred_rgb.to(self.precision_t))

        noise = torch.randn(latents.shape, dtype=latents.dtype, device=latents.device,
                            generator=self.noise_gen)

        text_embeddings = text_embeddings.reshape(
            -1, text_embeddings.shape[-2], text_embeddings.shape[-1])
        pooled_text_embeddings = pooled_text_embeddings.reshape(
            -1, pooled_text_embeddings.shape[-1])

        ind_t = torch.randint(self.min_step, self.max_step + int(self.warmup_step*warm_up_rate),
                              (1, ), dtype=torch.long, generator=self.noise_gen, device=self.device)[0]

        sigma = self.sigmas[ind_t]
        t = self.timesteps[ind_t]

        # Add noise according to flow matching
        latents_noisy = (1. - sigma) * latents + sigma * noise

        target = noise - latents

        with torch.no_grad():
            latent_model_input = latents_noisy[None, :, ...].repeat(
                2, 1, 1, 1, 1).reshape(-1, 16, resolution[0] // 8, resolution[1] // 8, )
            tt = t.reshape(1, 1).repeat(
                latent_model_input.shape[0], 1).reshape(-1)

            control_block_samples = None
            if use_control_net:
                control_image = self.encode_imgs(control_image.to(self.precision_t))
                control_image = control_image[None, :, ...].repeat(
                    2, 1, 1, 1, 1).reshape(-1, 16, resolution[0] // 8, resolution[1] // 8,)
                
                control_block_samples = self.controlnet(
                    hidden_states=latent_model_input.to(self.precision_t),
                    timestep=tt.to(self.precision_t),
                    pooled_projections=pooled_text_embeddings.to(
                        self.precision_t),
                    controlnet_cond=control_image,
                    conditioning_scale=self.controlnet_cond_scale,
                    return_dict=False,
                )[0]
            noise_pred = self.transformer(
                hidden_states=latent_model_input.to(self.precision_t),
                timestep=tt.to(self.precision_t),
                encoder_hidden_states=text_embeddings.to(self.precision_t),
                pooled_projections=pooled_text_embeddings.to(self.precision_t),
                block_controlnet_hidden_states=control_block_samples,
                return_dict=False,
            )[0]

            noise_pred = noise_pred.reshape(
                2, -1, 16, resolution[0] // 8, resolution[1] // 8, )
            noise_pred_uncond, noise_pred_text = noise_pred[:1].reshape(
                -1, 16, resolution[0] // 8, resolution[1] // 8, ), noise_pred[1:].reshape(-1, 16, resolution[0] // 8, resolution[1] // 8, )
            delta_DSD = noise_pred_text - noise_pred_uncond

        pred_noise = noise_pred_uncond + guidance_opt.guidance_scale * delta_DSD

        grad = sigma * (pred_noise - target)

        grad = torch.nan_to_num(grad_scale * grad)
        loss = SpecifyGradient.apply(latents, grad)
        return loss

    def encode_imgs(self, images):
        images = 2 * images - 1
        latents = [
            self.vae.encode(image.unsqueeze(0)).latent_dist.sample(
            ) * self.vae.config.scaling_factor
            for image in images
        ]
        return torch.cat(latents, dim=0).to(images.dtype)