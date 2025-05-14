import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# ------- Define U-Net from Scratch -------
class UNetScratch(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()

        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.ReLU(inplace=True)
            )

        self.down1 = conv_block(3, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = conv_block(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.middle = conv_block(64, 128)

        self.up1 = nn.Upsample(scale_factor=2)
        self.upc1 = conv_block(128 + 64, 64)
        self.up2 = nn.Upsample(scale_factor=2)
        self.upc2 = conv_block(64 + 32, 32)

        self.final = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(self.pool1(d1))
        mid = self.middle(self.pool2(d2))

        u1 = self.up1(mid)
        u1 = self.upc1(torch.cat([u1, d2], dim=1))
        u2 = self.up2(u1)
        u2 = self.upc2(torch.cat([u2, d1], dim=1))

        return torch.sigmoid(self.final(u2))


# ------- Define Dataset for Segmentation -------
class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        valid_ext = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

        image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(valid_ext)]
        mask_files = [f for f in os.listdir(mask_dir) if f.lower().endswith(valid_ext)]

        image_bases = {os.path.splitext(f)[0]: f for f in image_files}
        mask_bases = {os.path.splitext(f)[0]: f for f in mask_files}

        # Only keep pairs that have both image and mask
        common_keys = sorted(list(image_bases.keys() & mask_bases.keys()))

        self.image_paths = [os.path.join(image_dir, image_bases[k]) for k in common_keys]
        self.mask_paths = [os.path.join(mask_dir, mask_bases[k]) for k in common_keys]

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx]).convert("L")
        image = self.transform(image)
        mask = self.transform(mask)
        mask = (mask > 0.5).float()
        return image, mask


# ------- Dice Loss Function -------
def dice_loss(pred, target, epsilon=1e-6):
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    intersection = (pred * target).sum()
    return 1 - (2. * intersection + epsilon) / (pred.sum() + target.sum() + epsilon)


# ------- Training the U-Net -------
def train_unet():
    model = UNetScratch()
    dataset = SegmentationDataset("data/sampled_rice_images/images", "data/sampled_rice_masks")
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(20):
        total_loss = 0
        for img, mask in loader:
            pred = model(img)
            loss = F.binary_cross_entropy(pred, mask) + dice_loss(pred, mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1} | Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), "checkpoints/unet_sampled/model.pth")
    return model


# ------- Generate Binary Masks and Save -------
def generate_masks(model, image_dir="data/rice_images", mask_dir="data/generated_rice_masks"):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    model.eval()
    with torch.no_grad():
        for root, _, files in os.walk(image_dir):
            for fname in files:
                if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue  # ✅ Skip non-image files like .DS_Store

                img_path = os.path.join(root, fname)
                relative_path = os.path.relpath(img_path, image_dir)
                save_path = os.path.join(mask_dir, relative_path)

                os.makedirs(os.path.dirname(save_path), exist_ok=True)

                try:
                    image = Image.open(img_path).convert("RGB")
                    img_tensor = transform(image).unsqueeze(0)

                    pred_mask = model(img_tensor).squeeze().cpu()
                    binary_mask = (pred_mask > 0.5).float()
                    binary_mask_img = transforms.ToPILImage()(binary_mask)
                    binary_mask_img.save(save_path)
                except Exception as e:
                    print(f"[ERROR] Skipping {img_path}: {e}")


if __name__ == "__main__":
    os.makedirs("checkpoints/unet", exist_ok=True)
    model = UNetScratch()
    model.load_state_dict(torch.load("checkpoints/unet/best_model.pth", map_location="cpu"))
    generate_masks(model)

# if __name__ == "__main__":
#     os.makedirs("checkpoints/unet_sampled", exist_ok=True)
#     model = train_unet()
#     generate_masks(model)