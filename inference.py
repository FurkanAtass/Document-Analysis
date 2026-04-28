import os
import torch
import numpy as np
from PIL import Image
from dataset import image_to_patches, recompose_patches

MODEL_PATH = 'unet_dibco_last.pth'
BATCH_SIZE = 16
PATCH_SIZE = 256
THRESHOLD = 0.5
THRESHOLD_MODE = 'mean'  # options: 'fixed', 'mean'
IMAGE_PATH = "DAVU-UE1/dibco2009/DIBC02009_Test_images-handwritten/dibco_img0001.tif"
OUTPUT_DIR = "output"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(checkpoint_path, device):
    model = torch.hub.load(
        'mateuszbuda/brain-segmentation-pytorch',
        'unet',
        in_channels=1,
        out_channels=1,
        init_features=32,
        pretrained=False,
    )
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model


def compute_threshold(prediction, mode='mean', fixed_threshold=0.5):
    if mode == 'fixed':
        return fixed_threshold
    if mode == 'mean':
        return float(prediction.mean())
    raise ValueError(f"Unsupported threshold mode: {mode}")


def predict_full_mask(model, image, patch_size=256, device='cpu', batch_size=16, threshold=0.5, threshold_mode='fixed'):
    patches, padded_shape, pad = image_to_patches(image, patch_size, patch_size)
    patches = patches.astype(np.float32) / 255.0
    patches = patches[:, None, :, :]

    outputs = []
    with torch.no_grad():
        for start in range(0, patches.shape[0], batch_size):
            batch = torch.from_numpy(patches[start:start + batch_size]).to(device)
            pred = model(batch)
            pred = torch.sigmoid(pred).squeeze(1).cpu().numpy()
            outputs.append(pred)

    outputs = np.vstack(outputs)
    full_padded = recompose_patches(outputs, padded_shape, patch_size, patch_size)
    original_h, original_w = image.shape
    full = full_padded[:original_h, :original_w]

    threshold_value = compute_threshold(full, mode=threshold_mode, fixed_threshold=threshold)
    mask = (full >= threshold_value).astype(np.uint8) * 255
    return mask


def save_mask(mask, output_path):
    Image.fromarray(mask).save(output_path)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model = load_model(MODEL_PATH, device)
    image = np.asarray(Image.open(IMAGE_PATH).convert('L'))
    print(f"Loaded image {IMAGE_PATH} with shape {image.shape}")
    mask = predict_full_mask(
        model,
        image,
        patch_size=PATCH_SIZE,
        device=device,
        batch_size=BATCH_SIZE,
        threshold=THRESHOLD,
        threshold_mode=THRESHOLD_MODE,
    )
    print(f"Predicted mask shape: {mask.shape}, dtype: {mask.dtype}, min: {mask.min()}, max: {mask.max()}")
    save_mask(mask, f"{OUTPUT_DIR}/output_mask.png")
    print(f'Saved padded reconstructed mask to {OUTPUT_DIR}/{IMAGE_PATH.replace(IMAGE_PATH.split(".")[-1], "png")}')


if __name__ == '__main__':
    main()
