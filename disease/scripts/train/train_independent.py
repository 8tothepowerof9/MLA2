import torch
from utils.datamodule import get_dataloaders
from src.classify_diseases.models.independent import FineTunedResNet34
from src.classify_diseases.trainer import train_model
from src.classify_diseases.eval import evaluate_model
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

if __name__ == "__main__":
    # === Device Setup ==
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Using device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    # === Load Data (includes internal split) ===
    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        csv_path="data/meta_train.csv",
        image_dir="data/train_images",
        label_col="label",
        batch_size=32,
        strategy="weights",  # "oversample", "undersample", or None
        num_workers=4,
        pin_memory=True
    )

    # === Init Model ===
    model = FineTunedResNet34(num_classes=len(class_names)).to(device)

    # === Optimizer & Loss ===
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)

    loss_fn = torch.nn.CrossEntropyLoss()

    # === Train Model ===
    train_model(model, train_loader, val_loader, optimizer, 
                loss_fn, device, epochs=100, scheduler=scheduler, 
                mixup=False, cutmix=False, save_path="checkpoints/resnet34_paper.pt", history_path_file="history_paper.json")

    # === Evaluate Model on val_loader directly ===
    evaluate_model(model, test_loader, class_names, device)
    