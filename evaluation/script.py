from typing import Any, Dict, List
from PIL import Image, ImageDraw
from tools.tools import (
    concatenate_images_horizontally, 
    concatenate_images_vertically,
    AddText2Image
)

from .util import trim_by_mask
from .model import EvalCLIP, EvalLPIPS, EvalMSE

class EvalModule:
    def __init__(self, eval_option:dict={}) -> None:
        self.drawtext = eval_option.get('drawtext', False)
        self.return_masked_img = eval_option.get('return_masked_img', False)

        self.eval_clip = EvalCLIP()
        self.eval_lpips = EvalLPIPS()
        self.eval_mse = EvalMSE()

    def eval_to_img_per_mask(self, masked_x, masked_y, eval_type='clip'):
        if eval_type == 'clip':
            return self.eval_clip.img_img(masked_x, masked_y)
        elif eval_type == 'mse':
            return self.eval_mse.img_img(masked_x, masked_y)
        elif eval_type == 'lpips':
            return self.eval_lpips.img_img(masked_x, masked_y)
        else:
            print(f"[Warning!] Invalid eval_type: {eval_type}")
            return -1.0
    
    def eval_to_text_per_mask(self, masked_x, text_y, eval_type='clip'):
        if eval_type == 'clip':
            return self.eval_clip.img_text(masked_x, text_y)
        else:
            print(f"[Warning!] Invalid eval_type: {eval_type}")
            return -1.0

    def eval_to_features_per_mask(self, masked_x, features_y, eval_type='clip'):
        if eval_type == 'clip':
            return self.eval_clip.img_features(masked_x, features_y)
        else:
            print(f"[Warning!] Invalid eval_type: {eval_type}")
            return -1.0

    def eval_per_mask(
        self,
        x:Image.Image, 
        y:Image.Image=None, 
        cond:Dict[str, Any]=None, 
        eval_list:list=['clip', 'lpips', 'mse'],
    ):
        if cond['label'] == 'PALL':
            mask_img = cond['margin_mask_image']
            bbox_mask_img = cond['margin_mask_image']
        else:
            mask_img = cond['mask_image']
            bbox_mask_img = cond['bbox_mask_image']
        
        prompt = cond['prompt']
        prompt_embeds = cond['prompt_embeds']

        if x is not None:
            masked_x = trim_by_mask(x, mask_img)
            bbox_masked_x = trim_by_mask(x, bbox_mask_img)
        if y is not None:
            masked_y = trim_by_mask(y, mask_img)
            bbox_masked_y = trim_by_mask(y, bbox_mask_img)
        else:
            masked_y = Image.new('RGB', masked_x.size, (0, 0, 0))
            bbox_masked_y = Image.new('RGB', bbox_masked_x.size, (0, 0, 0))

        results = {}
        scroe_text, scroe_text_bbox = f"{cond['label']}:\n", f"{cond['label']}:\n"
        if y is not None:
            for eval_type in eval_list:
                results[eval_type] = self.eval_to_img_per_mask(masked_x, masked_y, eval_type=eval_type)
                scroe_text += f"img_img[{eval_type}]: {results[eval_type]:.3e}\n"

            for eval_type in eval_list:
                results[f'bbox_{eval_type}'] = self.eval_to_img_per_mask(bbox_masked_x, bbox_masked_y, eval_type=eval_type)
                scroe_text_bbox += f"bbox:img_img[{eval_type}]: {results[f'bbox_{eval_type}']:.3e}\n"

        if prompt is not None:
            results['clip_prompt'] = self.eval_to_text_per_mask(masked_x, prompt, eval_type='clip')
            scroe_text += f"img_[{prompt}]: {results['clip_prompt']:.3e}\n"

            results['bbox_clip_prompt'] = self.eval_to_text_per_mask(bbox_masked_x, prompt, eval_type='clip')
            scroe_text_bbox += f"bbox:img_[{prompt}]: {results['bbox_clip_prompt']:.3e}\n"

        if prompt_embeds is not None:
            pass
            # results['clip_token'] = eval_to_features_per_mask(masked_x, prompt_embeds, eval_type='clip')
            # scroe_text += f"img_features: {results['clip_token']:.3e}\n"

            # results['bbox_clip_token'] = eval_to_features_per_mask(bbox_masked_x, prompt_embeds, eval_type='clip')
            # scroe_text_bbox += f"img_features: {results['bbox_clip_token']:.3e}\n"
        
        if self.drawtext:
            masked_x = AddText2Image.drawText(img=masked_x, text=scroe_text, pos=(0,0))
            bbox_masked_x = AddText2Image.drawText(img=bbox_masked_x, text=scroe_text_bbox, pos=(0,0))

        if self.return_masked_img:
            return results, (masked_x, masked_y), (bbox_masked_x, bbox_masked_y)
        else:
            return results, None, None

    def eval(
        self,
        result_imgs:List[Image.Image], 
        conditions:List[Dict[str, Any]], 
        gt_img:Image.Image=None,
    ):
        results_all, masked_imgs_x_all, masked_imgs_y_all = [], [], []
        for img_iter, result_img in enumerate(result_imgs):
            results_per_img, masked_imgs_x, masked_imgs_y = {}, [], []
            for cond_iter, cond in enumerate(conditions):
                if len(cond['label']) == 4: # label='PALL'or'P{*}'
                    results_per_img[cond['label']], (masked_img_x, masked_img_y), (bbox_masked_img_x, bbox_masked_img_y) = self.eval_per_mask(
                        x=result_img, 
                        y=gt_img, 
                        cond=cond, 
                        eval_list=['clip', 'lpips', 'mse'],
                    )

                    masked_imgs_x.append(masked_img_x)
                    masked_imgs_x.append(bbox_masked_img_x)
                    masked_imgs_y.append(masked_img_y)
                    masked_imgs_y.append(bbox_masked_img_y)
            
            results_all.append(results_per_img)
            if self.return_masked_img:
                masked_imgs_x_all.append(concatenate_images_horizontally(masked_imgs_x))
                masked_imgs_y_all.append(concatenate_images_horizontally(masked_imgs_y))
        
        if self.return_masked_img:
            return results_all, (concatenate_images_vertically(masked_imgs_x_all), concatenate_images_vertically(masked_imgs_y_all))
        else:
            return results_all, None