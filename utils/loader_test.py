import torch
from torch.utils.data import Dataset
import torchvision
from torch.utils.data import DataLoader
import os
import numpy as np
from PIL import Image

from utils.config import *

class KneeMRIDataset(Dataset):
    def __init__(self, images,image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index])
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path), dtype=np.float32)
        n_classes = 5
        one_hot = np.zeros((mask.shape[0], mask.shape[1], n_classes))
        for i, unique_value in enumerate(np.unique(mask/200)):
            one_hot[:, :, i][mask/200 == unique_value] = 1
        mask = one_hot
        
        
        if self.transform is not None:

            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]


        return image, mask

def get_test_dataloader(
    images,
    test_dir,
    test_maskdir,
    batch_size,
    test_transform,
    num_workers=4,
    pin_memory=True,
):
    test_ds = KneeMRIDataset(
        images = images, 
        image_dir=test_dir,
        mask_dir=test_maskdir,
        transform=test_transform,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=test_batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )
    return test_loader
