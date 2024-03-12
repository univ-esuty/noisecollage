import sys, os
import datetime
from typing import Any
import cv2
import imageio
from PIL import Image, ImageDraw
from omegaconf import OmegaConf
from dataclasses import asdict, dataclass

from tqdm import tqdm

def concatenate_images_horizontally(image_list):
    """
    Return concatenated image from a list of images.
    """
    # Get the widths and heights of the input images
    widths = [image.size[0] for image in image_list]
    heights = [image.size[1] for image in image_list]

    # Compute the total width and maximum height of the concatenated image
    concatenated_width = sum(widths)
    concatenated_height = max(heights)

    # Create a new blank image with the concatenated dimensions
    concatenated_image = Image.new('RGB', (concatenated_width, concatenated_height))

    # Initialize the starting position for pasting the images
    x_offset = 0

    # Paste each image horizontally
    for image in image_list:
        concatenated_image.paste(image, (x_offset, 0))
        x_offset += image.size[0]

    return concatenated_image

def concatenate_images_vertically(image_list):
    """
    Return concatenated image from a list of images.
    """
    # Get the widths and heights of the input images
    widths = [image.size[0] for image in image_list]
    heights = [image.size[1] for image in image_list]

    # Compute the total width and maximum height of the concatenated image
    concatenated_width = max(widths)
    concatenated_height = sum(heights)

    # Create a new blank image with the concatenated dimensions
    concatenated_image = Image.new('RGB', (concatenated_width, concatenated_height))

    # Initialize the starting position for pasting the images
    y_offset = 0

    # Paste each image horizontally
    for image in image_list:
        concatenated_image.paste(image, (0, y_offset))
        y_offset += image.size[1]

    return concatenated_image

def create_args(args_configs):
    """
    return test_args, train_args
    """
    from utils import logger
    
    args = OmegaConf.create(asdict(args_configs))
    if args.exp_name is None:
        args.exp_name = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        if os.path.exists(f'{args.result_dir}/{args.exp_name}'):
            raise Exception(f"Exp_name directory already exists: {args.exp_name}")
    
    logger.configure(dir=f'{args.result_dir}/{args.exp_name}')
    OmegaConf.save(args, f'{logger.get_dir()}/exp_configs.yaml')
    
    dir = os.path.expanduser(f'{logger.get_dir()}/samples')
    os.makedirs(os.path.expanduser(dir), exist_ok=True)
    
    return args


def video_crop(video_path, crop_size=[0,0,512,512]):
    video = cv2.VideoCapture(video_path)
    output_frames = []
    
    for _ in tqdm(range(int(video.get(cv2.CAP_PROP_FRAME_COUNT)))):
        _, frame = video.read()
        frame = frame[crop_size[0]:crop_size[2], crop_size[1]:crop_size[3]]
        output_frames.append(frame)
    
    out_path = video_path.replace('.mp4', '_crop.mp4')
    imageio.mimsave(out_path, output_frames, fps=video.get(cv2.CAP_PROP_FPS))


def video_2_imgs(video_path, save_per_iter=4, pre_processor=None):
    video = cv2.VideoCapture(video_path)
    img_dir = video_path.replace('.mp4', '')
    os.makedirs(img_dir, exist_ok=True)
    print("fps:", video.get(cv2.CAP_PROP_FPS))
    
    for i in tqdm(range(int(video.get(cv2.CAP_PROP_FRAME_COUNT)))):
        _, frame = video.read()
        
        if i % save_per_iter == 0:
            frame.save(f'{img_dir}/org_{i:06d}.png')
            
            if pre_processor is not None:
                frame_p = pre_processor(frame)
                frame_p.save(f'{img_dir}/enc_{i:06d}.png')


class AddText2Image:
    def __init__(self):
        pass

    @classmethod
    def split_text(cls, text, text_in_one_line=50):
        text_list = [text[i: i + text_in_one_line] for i in range(0, len(text), text_in_one_line)]
        return '\n'.join(text_list)

    @classmethod
    def drawText(cls, img, text, pos=(0,0), color=(255,0,0)):
        text = cls.split_text(text)
        draw = ImageDraw.Draw(img)
        draw.text(pos, text, color, spacing=0, align='left')
        return draw._image


def torch_fix_seed(fix_flag, seed=334):
    if not fix_flag:
        return

    # Python random
    import random; random.seed(seed)
    # Numpy
    import numpy; numpy.random.seed(seed)
    # PyTorch
    import torch
    torch.manual_seed(seed)
    # PyTorch with GPU
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True


"""
You can tool directly
"""
if __name__ == '__main__':
    pass

    
    