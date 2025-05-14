import os
import pandas as pd
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image

from src.models.baseline import Baseline
from src.models.resnet_age import ResNetAge
from src.models.cbam import ResNetAgeWithCBAM  
from scripts.utils import get_device

# Custom dataset for unlabelled test images
class TestImageDataset(Dataset):
    def __init__(self, image_dir, csv_path, transform=None):
        self.image_dir = image_dir
        self.data = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_id = row['image_id']
        image_path = os.path.join(self.image_dir, image_id)
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image_id, image

def eval_model(config):
    # Constants
    AGE_MIN = 45.0
    AGE_MAX = 82.0

    # Config
    image_dir = config['test']['test_dir']
    csv_path = config['test']['prediction_csv_path']
    batch_size = config['test']['test_batch_size']
    image_size = config['test']['test_image_size']
    model_path = config['test']['test_model_path']
    model = config['test']['model']


    # Transform
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Dataset & Dataloader
    test_dataset = TestImageDataset(image_dir=image_dir, csv_path=csv_path, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Load model
    model_map = {
        "baseline": Baseline,
        "resnet": ResNetAge,
        "cbam": ResNetAgeWithCBAM
    }

    model_class = model_map.get(model, ResNetAge)

    model = model_class()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    device = get_device(force_cpu=False)
    model.to(device)
    model.eval()

    # Predict
    predictions = []
    with torch.no_grad():
        for image_ids, images in test_loader:
            images = images.to(device)
            outputs = model(images).view(-1)

            # Denormalize
            ages = outputs * (AGE_MAX - AGE_MIN) + AGE_MIN
            for img_id, age in zip(image_ids, ages):
                predictions.append((img_id, round(age.item(), 2)))

    # Write to CSV
    df = pd.read_csv(csv_path)
    pred_dict = {img_id: age for img_id, age in predictions}
    df['age'] = df['image_id'].map(pred_dict)
    df.to_csv(csv_path, index=False)
    print(f"[✓] Predictions saved to {csv_path}")