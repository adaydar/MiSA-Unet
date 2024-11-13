import torch
from torch.utils.data import Dataset
import torchvision
from torch.utils.data import DataLoader
import os
import numpy as np
from PIL import Image

#From local directory
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

def get_loaders(
    train_images_path,
    val_images_path,
    train_dir,
    train_maskdir,
    batch_size,
    train_transform,
    val_transform,
    num_workers=4,
    pin_memory=True,
):
    train_ds = KneeMRIDataset(
        images = train_images_path,
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=train_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = KneeMRIDataset(
        images = val_images_path,
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=val_transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader

def train_test_split(images,splitSize):
    imageLen = len(images)
    val_len = int(splitSize*imageLen)
    train_len = imageLen - val_len
    train_images,val_images = images[:train_len],images[train_len:]
    return train_images,val_images
    
images = os.listdir(TRAIN_IMG_DIR)
masks = os.listdir(TRAIN_MASK_DIR)
train_images_path,val_images_path = train_test_split(images,SPLIT)






