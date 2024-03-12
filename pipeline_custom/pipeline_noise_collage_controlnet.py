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



from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F
import json
from PIL import Image
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer
from scripts.convert_lora_safetensor_to_diffusers import convert_and_add_pipe

from diffusers.image_processor import VaeImageProcessor
from diffusers.loaders import TextualInversionLoaderMixin, LoraLoaderMixin, FromSingleFileMixin
from diffusers.models import AutoencoderKL, UNet2DConditionModel, ControlNetModel
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
from diffusers.schedulers import KarrasDiffusionSchedulers, UniPCMultistepScheduler
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput, StableDiffusionPipeline
from diffusers.pipelines.controlnet import StableDiffusionControlNetPipeline
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel

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


class NoiseCollageControlnetStableDiffusionPipeline(StableDiffusionControlNetPipeline):
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        controlnet: Union[ControlNetModel, List[ControlNetModel], Tuple[ControlNetModel], MultiControlNetModel],
        scheduler: UniPCMultistepScheduler,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPImageProcessor,
        requires_safety_checker: bool = True,
        ):
        super().__init__(vae, text_encoder, tokenizer, unet, controlnet, scheduler, safety_checker, feature_extractor, requires_safety_checker)
    
    
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

    def trim_by_mask(self, target_tensor:torch.Tensor, mask_img:Image.Image=None, bbox=None):
        """
        target_tensor: Image tensor to be trimmed
        mask_img: Image of mask
        out_res: Resolution of output image
        fill: Fill value of output image
        """
        ## Get Bounding Box
        if bbox is None:
            if mask_img is None:
                raise ValueError("Either mask_img or bbox must be provided.")
            bbox = mask_img.getbbox()
        
        ## Crop Image with bounding box
        target_tensor = target_tensor[:, :, bbox[1]:bbox[3], bbox[0]:bbox[2]]  
        return target_tensor, bbox


    @torch.no_grad()
    def __call__(
        self,
        collage_mode:str = "flex",
        collage_mode_kwargs:Dict[str, Any] = {},
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
        controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
        guess_mode: bool = False,
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

        # 2.1 Controlnet settings
        controlnet = self.controlnet._orig_mod if is_compiled_module(self.controlnet) else self.controlnet

        if isinstance(controlnet, MultiControlNetModel) and isinstance(controlnet_conditioning_scale, float):
            controlnet_conditioning_scale = [controlnet_conditioning_scale] * len(controlnet.nets)

        global_pool_conditions = (
            controlnet.config.global_pool_conditions
            if isinstance(controlnet, ControlNetModel)
            else controlnet.nets[0].config.global_pool_conditions
        )
        guess_mode = guess_mode or global_pool_conditions


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

        
        # 4. Prepare image
        if isinstance(controlnet, ControlNetModel):
            for cond in conditions:
                cond['cond_image'] = self.prepare_image(
                    image=cond['cond_image'],
                    width=width,
                    height=height,
                    batch_size=batch_size,
                    num_images_per_prompt=num_images_per_prompt,
                    device=device,
                    dtype=controlnet.dtype,
                    do_classifier_free_guidance=do_classifier_free_guidance,
                    guess_mode=guess_mode,
                )
        elif isinstance(controlnet, MultiControlNetModel):
            for cond in conditions:
                images = []
                for image_ in cond['cond_image']:
                    image_ = self.prepare_image(
                        image=image_,
                        width=width,
                        height=height,
                        batch_size=batch_size,
                        num_images_per_prompt=num_images_per_prompt,
                        device=device,
                        dtype=controlnet.dtype,
                        do_classifier_free_guidance=do_classifier_free_guidance,
                        guess_mode=guess_mode,
                    )

                    images.append(image_)

                cond['cond_image'] = images
        else:
            assert False


        # 4.1 Mask Normalization and Mask squarize
        dtype = conditions[0]["prompt_embeds"].dtype
        for cond in conditions:
            mask_image_resized = cond['bbox_mask_image'].resize((width//self.vae_scale_factor, height//self.vae_scale_factor)) 
            arr = np.array(mask_image_resized)
            arr = arr.astype(np.float32) / 255.0 * cond["mask_weight"]
            cond['mask_tensor'] = torch.from_numpy(arr).to(device=device, dtype=dtype)
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
                final_noise_pred = torch.zeros_like(latents)
                for cond in conditions:
                    ## prepare sub_latents
                    if collage_mode == "flex":
                        sub_latents, mask_bbox = self.trim_by_mask(latents, mask_img=cond['bbox_mask_image'].resize((width//self.vae_scale_factor, height//self.vae_scale_factor)))
                        
                        sub_cond_image, _ = self.trim_by_mask(cond['cond_image'], cond['bbox_mask_image'])
                        cond_width, cond_height = (mask_bbox[2]-mask_bbox[0])*self.vae_scale_factor, (mask_bbox[3]-mask_bbox[1])*self.vae_scale_factor
                        sub_cond_image = F.interpolate(sub_cond_image, size=(cond_height,cond_width), mode="nearest")

                    elif collage_mode == "padflex":
                        padding_scale = collage_mode_kwargs["padding_scale"]

                        pad_bbox = cond['bbox_mask_image'].getbbox()
                        pad_bbox_w, pad_bbox_h = pad_bbox[2]-pad_bbox[0], pad_bbox[3]-pad_bbox[1]
                        padding_w, padding_h = int(pad_bbox_w * padding_scale), int(pad_bbox_h * padding_scale)
                        pad_bbox = (
                            max(pad_bbox[0] - padding_w, 0), 
                            max(pad_bbox[1] - padding_h, 0),
                            min(pad_bbox[2] + padding_w, width),
                            min(pad_bbox[3] + padding_h, height)
                        )

                        sub_latents, mask_bbox = self.trim_by_mask(latents, bbox=[val//self.vae_scale_factor for val in pad_bbox])
                        
                        sub_cond_image, _ = self.trim_by_mask(cond['cond_image'], bbox=pad_bbox)
                        cond_width, cond_height = (mask_bbox[2]-mask_bbox[0])*self.vae_scale_factor, (mask_bbox[3]-mask_bbox[1])*self.vae_scale_factor
                        sub_cond_image = F.interpolate(sub_cond_image, size=(cond_height,cond_width), mode="nearest")

                    
                    elif collage_mode == "fixed":
                        sub_latents = latents.clone()
                        sub_cond_image = cond['cond_image']
                    
                    sub_latent_model_input = torch.cat([sub_latents] * 2) if do_classifier_free_guidance else sub_latents
                    sub_latent_model_input = self.scheduler.scale_model_input(sub_latent_model_input, t)

                    # controlnet(s) inference
                    if guess_mode and do_classifier_free_guidance:
                        # Infer ControlNet only for the conditional batch.
                        control_model_input = sub_latents
                        control_model_input = self.scheduler.scale_model_input(control_model_input, t)
                        controlnet_prompt_embeds = cond['prompt_embeds'].chunk(2)[1]
                    else:
                        control_model_input = sub_latent_model_input
                        controlnet_prompt_embeds = cond['prompt_embeds']

                    down_block_res_samples, mid_block_res_sample = self.controlnet(
                        control_model_input,
                        t,
                        encoder_hidden_states=controlnet_prompt_embeds,
                        controlnet_cond=sub_cond_image,
                        conditioning_scale=controlnet_conditioning_scale,
                        guess_mode=guess_mode,
                        return_dict=False,
                        cross_attention_kwargs=cross_attention_kwargs,
                    )

                    if guess_mode and do_classifier_free_guidance:
                        # Infered ControlNet only for the conditional batch.
                        # To apply the output of ControlNet to both the unconditional and conditional batches,
                        # add 0 to the unconditional batch to keep it unchanged.
                        down_block_res_samples = [torch.cat([torch.zeros_like(d), d]) for d in down_block_res_samples]
                        mid_block_res_sample = torch.cat([torch.zeros_like(mid_block_res_sample), mid_block_res_sample])

                    # predict the noise residual
                    noise_pred = self.unet(
                        sub_latent_model_input,
                        t,
                        encoder_hidden_states=cond['prompt_embeds'],
                        cross_attention_kwargs=cross_attention_kwargs,
                        down_block_additional_residuals=down_block_res_samples,
                        mid_block_additional_residual=mid_block_res_sample,
                        return_dict=False,
                    )[0]


                    # perform guidance
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                    if do_classifier_free_guidance and guidance_rescale > 0.0:
                        # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                        noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

                    ## arrange the estimated noise on original size image
                    if collage_mode == "flex" or collage_mode == "padflex":
                        noise_pred_base = torch.zeros_like(latents)
                        noise_pred_base[:, :, mask_bbox[1]:mask_bbox[3], mask_bbox[0]:mask_bbox[2]] = noise_pred
                    elif collage_mode == "fixed":
                        noise_pred_base = noise_pred

                    # apply the mask to noise_pred
                    final_noise_pred = final_noise_pred + noise_pred_base * cond['mask_tensor']

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




SD_MODELS = {
    "sd15_original": "runwayml/stable-diffusion-v1-5",
    "sd21_base": "stabilityai/stable-diffusion-2-1-base",
    "sd15_anything-v4" :"andite/anything-v4.0",
    "sd15_dreamlike-v20": "dreamlike-art/dreamlike-photoreal-2.0", #768x768 recommended

    "sd15_epicphotogasm_safetensors": "../pre-train/epicphotogasm_v1.safetensors",
    "sd15_realisticvision-v51_safetensors": "../pre-train/realisticVisionV51_v51VAE.safetensors",
    "sd15_nextphoto_safetensors": "../pre-train/nextphoto_v30.safetensors",
}

CONTROLNET_MODELS = {
    "sd15_pose": "lllyasviel/control_v11p_sd15_openpose",
    "sd15_pose_01d": "lllyasviel/sd-controlnet-openpose",
    "sd15_canny": "lllyasviel/control_v11p_sd15_canny",
    "sd15_sketch": "lllyasviel/control_v11p_sd15_scribble",

    "sd21_pose": "thibaud/controlnet-sd21-openposev2-diffusers",
    "sd21_canny": "thepowefuldeez/sd21-controlnet-canny",
}


def make_pipe(args, cls=StableDiffusionControlNetPipeline):
    ## Load pre-trained controler model
    controlnet = ControlNetModel.from_pretrained(
        CONTROLNET_MODELS[f"{args.model_version}_{args.model_controlnet}"],
        torch_dtype=torch.float32
    )

    ## Load pre-trained stable-diffusion model
    pipe = cls.from_pretrained(
        SD_MODELS[f"{args.model_version}_{args.model_base}"],
        controlnet=controlnet, 
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


def make_pipe_from_file(args, cls=StableDiffusionControlNetPipeline):
    ## Load pre-trained controler model
    controlnet = ControlNetModel.from_pretrained(
        CONTROLNET_MODELS[f"{args.model_version}_{args.model_controlnet}"],
        torch_dtype=torch.float32
    )

    ## Load pre-trained stable-diffusion model
    pipe = cls.from_single_file(
        SD_MODELS[f"{args.model_version}_{args.model_base}"],
        controlnet=controlnet,
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


def format_condition_mscoco(coco_data_id, args, label, controlent=True):
    if controlent:
        cond_img = Image.open(f"{args.ms_coco_path}/{coco_data_id}/{label}/{args.model_controlnet}_image.png")
    else:
        cond_img = None
    
    mask_img = Image.open(f"{args.ms_coco_path}/{coco_data_id}/{label}/mask_image.png")
    if label == 'PALL':
        bbox_mask_img = Image.new("L", mask_img.size, color=255)
    else:
        bbox_mask_img = Image.open(f"{args.ms_coco_path}/{coco_data_id}/{label}/bbox_mask_image.png")
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
