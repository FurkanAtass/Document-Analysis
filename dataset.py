import numpy as np   
import utils as U   
from patchify import patchify
import torch
from torch.utils.data import Dataset
from PIL import Image
import random

def pad_image_to_patch_size(img, patch_h, patch_w, pad_value=0):
    h, w = img.shape
    pad_h = (patch_h - (h % patch_h)) % patch_h
    pad_w = (patch_w - (w % patch_w)) % patch_w
    if pad_h == 0 and pad_w == 0:
        return img, (0, 0)
    padded = np.pad(img, ((0, pad_h), (0, pad_w)), constant_values=pad_value)
    return padded, (pad_h, pad_w)

def image_to_patches(img, patch_h, patch_w):
    padded, pad = pad_image_to_patch_size(img, patch_h, patch_w)
    patches = patchify(padded, (patch_h, patch_w), step=patch_h)
    patches = patches.reshape(-1, patch_h, patch_w)
    return patches, padded.shape, pad

def recompose_patches(patches, padded_shape, patch_h, patch_w):
    n_h = padded_shape[0] // patch_h
    n_w = padded_shape[1] // patch_w
    patches = patches.reshape(n_h, n_w, patch_h, patch_w)
    patches = patches.transpose(0, 2, 1, 3)
    full = patches.reshape(padded_shape)
    return full 

def get_train_val_data(
        dataset_dir,
        patch_size=(256, 256),
        train_transform=None,
        valid_transform=None,
        Test_year = 2009, 
        val_split_size = 0.2
    ):
    images, masks = U.get_train_data(dataset_dir, Test_year)
    val_idxs = random.sample(range(len(images)), int(len(images) * val_split_size))
    train_idxs = [i for i in range(len(images)) if i not in val_idxs]

    train_images = [images[i] for i in train_idxs]
    train_masks = [masks[i] for i in train_idxs]

    val_images = [images[i] for i in val_idxs]
    val_masks = [masks[i] for i in val_idxs]

    train_dataset = DIBCODataset(train_images, train_masks, patch_size=patch_size, transform=train_transform)
    val_dataset = DIBCODataset(val_images, val_masks, patch_size=patch_size, transform=valid_transform)

    return train_dataset, val_dataset

class DIBCODataset(Dataset):
    def __init__(self, images, masks, patch_size=(256, 256), transform=None, test_year=2009):
        self.patch_size = patch_size
        self.transform = transform
        self.full_images = images
        self.full_masks = masks
        
        # Compute number of patches per image to map flat indices to image/patch pairs
        self.patch_counts = []
        self.cumulative_patches = [0]
        for img in images:
            padded, _ = pad_image_to_patch_size(img, patch_size[0], patch_size[1])
            n_patches = (padded.shape[0] // patch_size[0]) * (padded.shape[1] // patch_size[1])
            self.patch_counts.append(n_patches)
            self.cumulative_patches.append(self.cumulative_patches[-1] + n_patches)

    def __len__(self):
        return self.cumulative_patches[-1]

    def __getitem__(self, idx):
        # Find which image this patch belongs to
        img_idx = 0
        for i, cumsum in enumerate(self.cumulative_patches[:-1]):
            if idx >= cumsum and idx < self.cumulative_patches[i+1]:
                img_idx = i
                break
        
        # Get the patch index within this image
        patch_idx = idx - self.cumulative_patches[img_idx]
        
        # Get patches for this image on-demand
        img_patches, _, _ = image_to_patches(self.full_images[img_idx], self.patch_size[0], self.patch_size[1])
        msk_patches, _, _ = image_to_patches(self.full_masks[img_idx], self.patch_size[0], self.patch_size[1])
        
        image = img_patches[patch_idx]
        mask = msk_patches[patch_idx]
        
        if self.transform is not None:
            image = Image.fromarray(image.astype(np.uint8))
            mask = Image.fromarray(mask.astype(np.uint8))
            image, mask = self.transform(image, mask)
        else:
            image = image.astype(np.float32) / 255.0
            image = torch.from_numpy(image).unsqueeze(0)
            
            mask = mask.astype(np.float32) / 255.0
            mask = torch.from_numpy(mask).unsqueeze(0)

        return image, mask

        