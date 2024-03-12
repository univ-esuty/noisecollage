import numpy as np
import math
import random
import blobfile as bf

from glob import glob
from PIL import Image
from mpi4py import MPI
from torch.utils.data import DataLoader, Dataset

def load_image_dataset(
    data_dir,
    batch_size,
    image_size,
    class_cond=False,
    deterministic=False,
    to_gray=False,
):
    if not data_dir:
        raise ValueError("unspecified data directory")
    
    all_files = []
    for type in ["jpg", "jpeg", "png", "gif"]:
        all_files += glob(f'{data_dir}/*/*.{type}')

    classes = {}
    if class_cond:
        class_labels = sorted(glob(f'{data_dir}/*'))
        num = 0
        for label in class_labels:
            classes[label.split('/')[-1]] = num
            num += 1
    
    dataset = SimpleImageDataset(
        image_size,
        all_files,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
        classes=classes,
        random_flip=False,
        random_crop=False,
        to_gray=to_gray,
    )
    
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    
    while True:
        yield from loader    
    
    
class SimpleImageDataset(Dataset):
    def __init__(
        self,
        resolution,
        image_paths,
        shard=0,
        num_shards=1,
        classes={},
        random_crop=False,
        random_flip=False,
        to_gray=False,
    ):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.classese = classes
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.to_gray = to_gray

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        class_label = path.split('/')[-2]
        
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
        pil_image = pil_image.convert("RGB")

        if self.random_crop:
            arr = self.random_crop_arr(pil_image, self.resolution)
        else:
            pil_image = pil_image.resize((self.resolution, self.resolution))
            arr = self.center_crop_arr(pil_image, self.resolution)

        if self.random_flip and random.random() < 0.5:
            arr = arr[:, ::-1]

        arr = arr.astype(np.float32) / 127.5 - 1
        
        if self.to_gray:
            arr = (arr[:, :, 0] + arr[:, :, 1] + arr[:, :, 2]) / 3.0
            arr = np.reshape(arr, (arr.shape[0], arr.shape[1], 1))

        out_dict = {}
        if self.classese != {}:
            out_dict["y"] = np.array(self.classese[class_label], dtype=np.int64)
        return np.transpose(arr, [2, 0, 1]), out_dict
    
    def center_crop_arr(self, pil_image, image_size):
        # We are not on a new enough PIL to support the `reducing_gap`
        # argument, which uses BOX downsampling at powers of two first.
        # Thus, we do it by hand to improve downsample quality.
        while min(*pil_image.size) >= 2 * image_size:
            pil_image = pil_image.resize(
                tuple(x // 2 for x in pil_image.size), resample=Image.BOX
            )

        scale = image_size / min(*pil_image.size)
        pil_image = pil_image.resize(
            tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
        )

        arr = np.array(pil_image)
        crop_y = (arr.shape[0] - image_size) // 2
        crop_x = (arr.shape[1] - image_size) // 2
        return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


    def random_crop_arr(self, pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
        min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
        max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
        smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

        # We are not on a new enough PIL to support the `reducing_gap`
        # argument, which uses BOX downsampling at powers of two first.
        # Thus, we do it by hand to improve downsample quality.
        while min(*pil_image.size) >= 2 * smaller_dim_size:
            pil_image = pil_image.resize(
                tuple(x // 2 for x in pil_image.size), resample=Image.BOX
            )

        scale = smaller_dim_size / min(*pil_image.size)
        pil_image = pil_image.resize(
            tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
        )

        arr = np.array(pil_image)
        crop_y = random.randrange(arr.shape[0] - image_size + 1)
        crop_x = random.randrange(arr.shape[1] - image_size + 1)
        return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]  
