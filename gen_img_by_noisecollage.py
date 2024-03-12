import glob
import json
import os
import torch

from dataclasses import asdict, dataclass, field
from omegaconf import OmegaConf

from tools.tools import torch_fix_seed
from pipeline_custom.pipeline_noise_collage import NoiseCollageStableDiffusionPipeline
from pipeline_custom.pipeline_noise_collage import make_pipe, make_pipe_from_file, format_condition_mscoco
from evaluation.script import EvalModule

NUM_DATA = 9999
torch_fix_seed(True, seed=334)

## CONFIGS
@dataclass(init=True)
class ExpConfigs:
    ## Model Settings
    model_version:str = "sd15"
    model_base:str = "epicphotogasm_safetensors" 
    
    use_lora:bool = False
    lora_path:str = ""
    lora_alpha:float = None

    ## MS-COCO settings
    ms_coco_path:str = "./sample_inputs/normal"
    prompt_type:str = 'self_caption' # <-- 'self_caption' or 'blip_caption'

    ## Generation Settings
    batch_size:int = 4
    resolution:int = 512
    guidance_scale:float = 7.5
    num_inference_steps:int = 50 # <-- total denoise steps
    prompt:str = "" #", high quality, photorealistic, 4K, 8K, ultra detailed" <-- optional 
    n_prompt:str = ", low quality, noisy, artifact, blurry, watermark" # <-- optional
    drop_rate:float = 0.75 # <-- replace randomly caption for background into empty in Masked Cross-Attention. 
    bg_blending_weight:float = 0.1

    ## Experiment Settings
    GPU_IDX:int = 0
    result_dir:str = "./sample_outputs"
    exp_name:str = "nc"


def run_pipe(pipe, coco_data_id:str, args:ExpConfigs):
    ## setup conditions
    conditions = []
    cond = format_condition_mscoco(coco_data_id, args, 'PALL')
    conditions.append(cond)

    PNUM = len(glob.glob(f"{args.ms_coco_path}/{coco_data_id}/P*")) - 1 
    for i in range(PNUM):
        if os.path.exists(f"{args.ms_coco_path}/{coco_data_id}/P{i:03d}"):
            cond = format_condition_mscoco(coco_data_id, args, f'P{i:03d}')
            conditions.append(cond)

    mode_select_kwargs = {
        'drop_rate': args.drop_rate,
    }

    ## Generate
    result_tensor = pipe.__call__(
        conditions=conditions,
        width=args.resolution,
        height=args.resolution,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        num_images_per_prompt=args.batch_size,
        mode_select_kwargs=mode_select_kwargs,
        latents=args.latents
    ) #__call__() is called.

    return result_tensor, conditions


def eval_result(pipe, coco_data_id, args, result_tensor, conditions, eval_module):
    result_imgs = pipe.image_processor.postprocess(result_tensor)
    results_all, results_details = eval_module.eval(result_imgs, conditions, None)

    return results_all, results_details


def main():
    args:ExpConfigs = ExpConfigs()
    os.makedirs(f"{args.result_dir}/{args.exp_name}", exist_ok=True)
    OmegaConf.save(config=asdict(args), f=f"{args.result_dir}/{args.exp_name}/configs.yaml")

    ## Make Pipeline for StableDiffusion ControlNet
    model_name = f"{args.model_version}_{args.model_base}"
    if '_safetensors' in model_name:
        pipe:NoiseCollageStableDiffusionPipeline = make_pipe_from_file(args, cls=NoiseCollageStableDiffusionPipeline)
    else:
        pipe:NoiseCollageStableDiffusionPipeline = make_pipe(args, cls=NoiseCollageStableDiffusionPipeline)

    ## experimnet setting.
    eval_options = {
        'return_masked_img':True,
        'drawtext': False
    }
    eval_module = EvalModule(eval_options)
    coco_data_id_list = glob.glob(f"{args.ms_coco_path}/testcase_*")

    num_coco_data = len(coco_data_id_list)
    setattr(args, 'latents', torch.randn((args.batch_size,4,64,64), device=f'cuda:{args.GPU_IDX}')) # fix latents


    ## experiment start.
    with torch.no_grad():
        for progress, coco_data_path in enumerate(coco_data_id_list[:NUM_DATA]):
            coco_data_id = coco_data_path.split('/')[-1]
            save_path = f"{args.result_dir}/{args.exp_name}/{coco_data_id}"
            os.makedirs(save_path, exist_ok=True)

            ## Run pipeline
            result_tensor, conditions = run_pipe(pipe, coco_data_id, args) 

            result_tensor_cat = torch.cat(tuple(result_tensor), dim=2).unsqueeze(0)
            output_imgs = pipe.image_processor.postprocess(result_tensor_cat)[0]
            output_imgs.save(f"{save_path}/result_all.png")
            for i in range(args.batch_size):
                output_img = pipe.image_processor.postprocess(result_tensor[i:i+1])[0]
                output_img.save(f"{save_path}/result_{i:03d}.png")

            
            ## Update prompts for evaluation when generating images with the catname-caption prompt.
            if args.prompt_type == 'catname_caption':
                for cond in conditions:
                    if cond['label'] != 'PALL':
                        cond['prompt'] = "a photo of " + cond['prompt']

            ## Evaluate the result
            results_all, results_details = eval_result(pipe, coco_data_id, args, result_tensor, conditions, eval_module)

            with open(f"{save_path}/score_result.json", 'w') as file:
                json.dump(results_all, file, indent=4)
            results_details[0].save(f"{save_path}/result_all_details.png")

            ## progress report
            print(f"Progress: {progress+1:5d}/{num_coco_data:5d}")

if __name__ == '__main__':
    main()

