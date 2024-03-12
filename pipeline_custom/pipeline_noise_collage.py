# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import numpy as np
import torch
import torch.nn.functional as F

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer
from PIL import Image

from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.utils import (
    PIL_INTERPOLATION,
    is_accelerate_available,
    is_accelerate_version,
    is_compiled_module,
    logging,
    randn_tensor,
    replace_example_docstring,
)
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers import KarrasDiffusionSchedulers, DDIMScheduler, UniPCMultistepScheduler
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput, StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from scripts.convert_lora_safetensor_to_diffusers import convert_and_add_pipe


SD_MODELS = {
    "sd15_original": "runwayml/stable-diffusion-v1-5",
    "sd21_base": "stabilityai/stable-diffusion-2-1-base",
    "sd15_anything-v4" :"andite/anything-v4.0",
    "sd15_dreamlike-v20": "dreamlike-art/dreamlike-photoreal-2.0", #768x768 recommended
    "sdxl_base1.0": "stabilityai/stable-diffusion-xl-base-1.0", #1024x1024 recommended

    "sd15_epicphotogasm_safetensors": "../pre-train/epicphotogasm_v1.safetensors",
    "sd15_realisticvision-v51_safetensors": "../pre-train/realisticVisionV51_v51VAE.safetensors",
    "sd15_nextphoto_safetensors": "../pre-train/nextphoto_v30.safetensors",
}

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name
EXAMPLE_DOC_STRING = "Sorry, No example docstring for this method."


def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg



class NoiseCollageStableDiffusionPipeline(StableDiffusionPipeline):
    def __init__(
            self, 
            vae: AutoencoderKL, 
            text_encoder: CLIPTextModel, 
            tokenizer: CLIPTokenizer, 
            unet: UNet2DConditionModel, 
            scheduler: UniPCMultistepScheduler, 
            safety_checker: StableDiffusionSafetyChecker, 
            feature_extractor: CLIPImageProcessor, 
            requires_safety_checker: bool = True
        ):
        super().__init__(vae, text_encoder, tokenizer, unet, scheduler, safety_checker, feature_extractor, requires_safety_checker)
    
    
    def check_inputs(self, **kwargs):
        pass


    def squarize(self, target_tensor, **bg_kwargs):
        batch, ch, height, width = target_tensor.size()
        if width == height:
            return target_tensor, (0, 0, width, height)
        elif width > height:
            top, bottom, left, right = (width-height)//2, (width-height)//2+height, 0, width
            bg_tensor = self.scheduler.add_noise(
                torch.ones((batch, ch, width, width), device=target_tensor.device),
                torch.randn((batch, ch, width, width), device=target_tensor.device),
                bg_kwargs['timestep'].repeat(batch),
            )
            bg_tensor[:, :, top:bottom, left:right] = target_tensor
            return bg_tensor, (left, top, right, bottom)
        else:
            top, bottom, left, right = 0, height, (height-width)//2, (height-width)//2+width
            bg_tensor = self.scheduler.add_noise(
                torch.ones((batch, ch, height, height), device=target_tensor.device),
                torch.randn((batch, ch, height, height), device=target_tensor.device),
                bg_kwargs['timestep'].repeat(batch),
            )
            bg_tensor[:, :, top:bottom, left:right] = target_tensor
            return bg_tensor, (left, top, right, bottom)

    def trim_by_mask(self, target_tensor:torch.Tensor, mask_img:Image.Image):
        """
        target_tensor: Image tensor to be trimmed
        mask_img: Image of mask
        out_res: Resolution of output image
        fill: Fill value of output image
        """
        ## Get Bounding Box
        bbox = mask_img.getbbox()
        if bbox is None:
            raise Exception("Invalid mask image")
        
        ## Crop Image with bounding box
        target_tensor = target_tensor[:, :, bbox[1]:bbox[3], bbox[0]:bbox[2]]  
        return target_tensor, bbox


    @torch.no_grad()
    def __call__(
        self,
        only_semantic_attention = False,
        conditions: List[Dict[str, Any]] = None,
        height: Optional[int] = 512,
        width: Optional[int] = 512,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        guidance_rescale: float = 0.0,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = {},
        mode_select_kwargs: Optional[Dict[str, Any]] = {},
    ):
        # 1. Check inputs. Raise error if not correct
        self.check_inputs() # skip for debugging...

        # 2. Define call parameters
        batch_size = num_images_per_prompt
        num_channels_latents = self.unet.config.in_channels
        device = self._execution_device

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        for cond in conditions:
            cond["prompt_embeds"] = self._encode_prompt(
                cond["prompt"],
                device,
                num_images_per_prompt,
                do_classifier_free_guidance,
                cond["negative_prompt"],
                prompt_embeds=cond["prompt_embeds"],
                negative_prompt_embeds=cond["negative_prompt_embeds"],
            )
        
        null_prompt_embeds = self._encode_prompt(
            "",
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            "",
            prompt_embeds=None,
            negative_prompt_embeds=None,
        )

        # 4. Mask Normalization and Mask squarize
        dtype = conditions[0]["prompt_embeds"].dtype
        for cond in conditions:
            mask_image_resized = cond['bbox_mask_image'].resize((width//self.vae_scale_factor, height//self.vae_scale_factor)) 
            # mask_image_resized = cond['mask_image'].resize((width//self.vae_scale_factor, height//self.vae_scale_factor)) 

            arr = np.array(mask_image_resized)
            arr = arr.astype(np.float32) / 255.0
            cond['attention_mask'] = torch.from_numpy(arr).to(device=device, dtype=dtype)
            cond['mask_tensor'] = cond['attention_mask'] * cond['mask_weight']
        mask_images = torch.concat([cond['mask_tensor'].unsqueeze(dim=0) for cond in conditions], dim=0)
        total_mask = mask_images.sum(dim=0)
        for cond in conditions:
            cond['mask_tensor'] = (cond['mask_tensor'] / total_mask
                                ).reshape(1, 1, height//self.vae_scale_factor, width//self.vae_scale_factor
                                ).repeat(1, num_channels_latents, 1, 1)
            
        # 5. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 6. Prepare latent variables
        latents = self.prepare_latents(
            batch_size,
            num_channels_latents,
            height,
            width,
            dtype,
            device,
            generator,
            latents,
        )

        # 7. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                final_noise_pred = torch.zeros_like(latents)
                if only_semantic_attention:
                    cross_attention_kwargs.update({
                        'hidden_states_semantic': [cond['attention_mask'] for cond in conditions],
                    })
                    encoder_hidden_states = [cond['prompt_embeds'] for cond in conditions]

                    # predict the noise residual
                    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=encoder_hidden_states,
                        cross_attention_kwargs=cross_attention_kwargs,
                        return_dict=False,
                    )[0]

                    # perform guidance
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                    if do_classifier_free_guidance and guidance_rescale > 0.0:
                        # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                        noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

                    # apply the mask to noise_pred
                    final_noise_pred = final_noise_pred + noise_pred

                else:
                    for idx, cond in enumerate(conditions):
                        # register smantics for cross-attention
                        if cond['label'] == 'PALL':
                            cross_attention_kwargs.update({
                                'hidden_states_semantic': [cond['attention_mask']],
                            })
                            encoder_hidden_states = [cond['prompt_embeds']]                      
                        
                        elif mode_select_kwargs.get('drop_rate', None) is not None: # w/ dropout
                            drop_rate = mode_select_kwargs['drop_rate']

                            bg_attention_mask = 1.0-cond['attention_mask']
                            random_mask = torch.rand_like(bg_attention_mask)
                            bg_attention_mask_dropped = torch.where((bg_attention_mask != 0) & (random_mask < drop_rate), 0, bg_attention_mask)
                            bg_attention_mask_null = torch.where((bg_attention_mask != 0) & (random_mask >= drop_rate), 0, bg_attention_mask)

                            cross_attention_kwargs.update({
                                'hidden_states_semantic': [cond['attention_mask'], bg_attention_mask_dropped, bg_attention_mask_null]
                            })
                            encoder_hidden_states = [cond['prompt_embeds'], conditions[0]['prompt_embeds'], null_prompt_embeds]

                        elif mode_select_kwargs.get('nullpad_scale', None) is not None: # w/ null padding
                            padding_scale = mode_select_kwargs['nullpad_scale']
                            latents_w, latents_h = latents.shape[2], latents.shape[3]

                            mask_bbox = cond['bbox_mask_image'].resize((width//self.vae_scale_factor, height//self.vae_scale_factor)).getbbox()
                            mask_w, mask_h = mask_bbox[2]-mask_bbox[0], mask_bbox[3]-mask_bbox[1]
                            padding_w, padding_h = int(mask_w * padding_scale), int(mask_h * padding_scale)
                            mask_bbox = (
                                max(mask_bbox[0] - padding_w, 0), 
                                max(mask_bbox[1] - padding_h, 0),
                                min(mask_bbox[2] + padding_w, latents_w),
                                min(mask_bbox[3] + padding_h, latents_h)
                            )

                            bg_attention_mask_padded = torch.zeros_like(cond['attention_mask'])
                            bg_attention_mask_padded[mask_bbox[1]:mask_bbox[3], mask_bbox[0]:mask_bbox[2]] = 1.0
                            bg_attention_mask_null = 1.0 - bg_attention_mask_padded
                            bg_attention_mask_padded = bg_attention_mask_padded - cond['attention_mask']
                            
                            cross_attention_kwargs.update({
                                'hidden_states_semantic': [cond['attention_mask'], bg_attention_mask_padded, bg_attention_mask_null]
                            })
                            encoder_hidden_states = [cond['prompt_embeds'], conditions[0]['prompt_embeds'], null_prompt_embeds]

                        else:
                            cross_attention_kwargs.update({
                                'hidden_states_semantic': [cond['attention_mask'], (1.0-cond['attention_mask'])],
                            })
                            encoder_hidden_states = [cond['prompt_embeds'], conditions[0]['prompt_embeds']]


                        # predict the noise residual
                        noise_pred = self.unet(
                            latent_model_input,
                            t,
                            encoder_hidden_states=encoder_hidden_states,
                            cross_attention_kwargs=cross_attention_kwargs,
                            return_dict=False,
                        )[0]

                        # perform guidance
                        if do_classifier_free_guidance:
                            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                        if do_classifier_free_guidance and guidance_rescale > 0.0:
                            # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                            noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

                        # apply the mask to noise_pred
                        final_noise_pred = final_noise_pred + noise_pred * cond['mask_tensor']

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(final_noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        # If we do sequential model offloading, let's offload unet and controlnet
        # manually for max memory savings
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.unet.to("cpu")
            self.controlnet.to("cpu")
            torch.cuda.empty_cache()

        ## 9. Decode latents with VAE
        image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
        
        ## 10. Postprocess image
        return image

        # do_denormalize = [True] * image.shape[0]
        # image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        # # Offload last model to CPU
        # if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
        #     self.final_offload_hook.offload()

        # if not return_dict:
        #     return (image, None)

        # return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=None)



def make_pipe(args, cls=StableDiffusionPipeline):
    ## Load pre-trained stable-diffusion model
    pipe = cls.from_pretrained(
        SD_MODELS[f"{args.model_version}_{args.model_base}"],
        torch_dtype=torch.float32
    )

    ## Load LoRA (attention layer only)
    if args.use_lora:
        pipe = convert_and_add_pipe(
            pipe, 
            args.lora_path,
            alpha=args.lora_alpha
        )

    pipe = pipe.to(f'cuda:{args.GPU_IDX}')
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    return pipe


def make_pipe_from_file(args, cls=StableDiffusionPipeline):
    ## Load pre-trained stable-diffusion model
    pipe = cls.from_single_file(
        SD_MODELS[f"{args.model_version}_{args.model_base}"],
        torch_dtype=torch.float32
    )

    ## Load LoRA (attention layer only)
    if args.use_lora:
        pipe = convert_and_add_pipe(
            pipe, 
            args.lora_path,
            alpha=args.lora_alpha
        )

    pipe = pipe.to(f'cuda:{args.GPU_IDX}')
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    return pipe


def format_condition_mscoco(coco_data_id, args, label):
    cond_img = None #Image.open(f"{args.ms_coco_path}/{coco_data_id}/{label}/cond_image.png")
    mask_img = Image.open(f"{args.ms_coco_path}/{coco_data_id}/{label}/mask_image.png")
    bbox_mask_img = Image.open(f"{args.ms_coco_path}/{coco_data_id}/{label}/bbox_mask_image.png").convert('L')
    prompt = json.load(open(f"{args.ms_coco_path}/{coco_data_id}/{label}/{args.prompt_type}.json"))

    condition = {
        'label': label,
        'cond_image': cond_img,
        'mask_image': mask_img,
        'mask_weight': 1.0,
        'bbox_mask_image': bbox_mask_img,
        'prompt': prompt + args.prompt,
        'negative_prompt': args.n_prompt,
        'prompt_embeds': None,
        'negative_prompt_embeds': None,
    }

    if label == 'PALL':
        condition['mask_weight'] = args.bg_blending_weight
        condition['margin_mask_image'] = Image.open(f"{args.ms_coco_path}/{coco_data_id}/{label}/margin_mask_image.png")

    return condition