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
    def __init__(self, images, masks,patch_size=(256, 256), transform=None, test_year=2009):
        self.patch_size = patch_size
        self.transform = transform
        self.images, self.masks = self._prepare_patches(images, masks)

    def _prepare_patches(self, full_images, full_masks):
        all_image_patches = []
        all_mask_patches = []
        for img, msk in zip(full_images, full_masks):
            img_patches, _, _ = image_to_patches(img, self.patch_size, self.patch_size)
            msk_patches, _, _ = image_to_patches(msk, self.patch_size, self.patch_size)
            all_image_patches.append(img_patches)
            all_mask_patches.append(msk_patches)

        all_image_patches = np.vstack(all_image_patches)
        all_mask_patches = np.vstack(all_mask_patches)
        return all_image_patches, all_mask_patches

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]
        
        if self.transform is not None:
            image = Image.fromarray(image.astype(np.uint8))
            mask = Image.fromarray(mask.astype(np.uint8))

            image, mask = self.transform(image, mask)
        else:
            image = self.images[idx].astype(np.float32) / 255.0
            image = torch.from_numpy(image).unsqueeze(0)
            
            mask = self.masks[idx].astype(np.float32) / 255.0
            mask = torch.from_numpy(mask).unsqueeze(0)

        return image, mask

        