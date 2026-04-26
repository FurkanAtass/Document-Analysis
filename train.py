import tqdm
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from dice_loss import BinaryDiceLoss
from torch.utils.data import DataLoader
from preprocess import get_train_val_data

Dataset_dir = './DIBCO/'   
patch_size  = 256
N_patches   = 100
test_year   = 2009
seed = 42
val_split_size = 0.2

BATCH_SIZE = 16
NUM_EPOCHS = 100
LR = 5e-4

torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_transform(img, mask):
    if random.random() > 0.5:
        img = transforms.functional.hflip(img)
        mask = transforms.functional.hflip(mask)
    if random.random() > 0.5:
        img = transforms.functional.vflip(img)
        mask = transforms.functional.vflip(mask)
    if random.random() > 0.5:
        gaussian = transforms.GaussianBlur(3, sigma=(0.1, 2.0))
        img = gaussian(img)

    augmentation_pipeline = transforms.Compose([
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue = 0.25),
        transforms.ToTensor()
    ])

    img = augmentation_pipeline(img)
    mask = transforms.ToTensor()(mask)
    return img, mask

def valid_transform(img, mask):
    img = transforms.ToTensor()(img)
    mask = transforms.ToTensor()(mask)
    return img, mask

def show_image_mask_pair(image, mask, title=None):
    image_np = image.squeeze(0).cpu().detach().numpy()
    mask_np = mask.squeeze(0).cpu().detach().numpy()
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(image_np, cmap='gray')
    axes[0].set_title('Image')
    axes[0].axis('off')
    axes[1].imshow(mask_np, cmap='gray')
    axes[1].set_title('Mask')
    axes[1].axis('off')
    if title:
        fig.suptitle(title)
    plt.savefig(f"{title}.png")

train_dataset, val_dataset = get_train_val_data(
    Dataset_dir, 
    patch_size=patch_size, 
    train_transform=train_transform, 
    valid_transform=valid_transform, 
    Test_year=test_year, 
    val_split_size=val_split_size
    )

print(f"Train dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")

# for i in range(3):
#     idx = random.randint(0, len(train_dataset)-1)
#     img, msk = train_dataset[idx]
#     show_image_mask_pair(img, msk, title=f'Patch {i}')

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
    in_channels=1, out_channels=1, init_features=32, pretrained=False)

model.to(device)
model.train()

bceloss = torch.nn.BCELoss()
dice_loss = BinaryDiceLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
train_losses = []
val_losses = []
for epoch in range(NUM_EPOCHS):
    model.train()
    epoch_loss = 0
    for images, masks in tqdm.tqdm(train_loader, desc="Training"):
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = bceloss(outputs, masks) + dice_loss(outputs, masks)
        loss.backward()
        optimizer.step()
        scheduler.step()
        epoch_loss += loss.item()

    avg_epoch_loss = epoch_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {avg_epoch_loss:.4f}")
    train_losses.append(avg_epoch_loss)

    model.eval()
    with torch.no_grad():
        val_loss = 0
        for images, masks in tqdm.tqdm(val_loader, desc="Validation"):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = bceloss(outputs, masks) + dice_loss(torch.sigmoid(outputs), masks)
            val_loss += loss.item()
    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Validation Loss: {avg_val_loss:.4f}")

model.eval()
model.cpu()
torch.save(model.state_dict(), "unet_dibco_last.pth")

plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.savefig("loss_curve.png")
