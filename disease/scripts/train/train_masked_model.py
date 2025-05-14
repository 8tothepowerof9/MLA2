import torch
from utils.datamodule import get_dataloaders
from src.classify_diseases.models.simple_cnn import SimpleCNN
from src.classify_diseases.models.resnet_model import ResNet18Classifier
from src.classify_diseases.models.cbam import ResNet34CBAMClassifier
from src.classify_diseases.trainer import train_model
from src.classify_diseases.eval import evaluate_model
from utils.inspectors import plot_all_histories_in_dir
from utils.focal_loss import FocalLoss
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
        image_dir="data/masked_images",
        label_col="label",
        batch_size=32,
        strategy="weights",  # "oversample", "undersample", or None
        num_workers=4,
        pin_memory=True
    )

    # === Init Model ===
    model = ResNet34CBAMClassifier(num_classes=len(class_names)).to(device)

    # === Optimizer & Loss ===
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)

    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    # Use Focal Loss instead of CrossEntropyLoss
    # loss_fn = FocalLoss(gamma=0.5, alpha=1.0)  

    # === Train Model ===
    train_model(model, train_loader, val_loader, optimizer, 
                loss_fn, device, epochs=100, scheduler=scheduler, 
                mixup=False, cutmix=True, save_path="checkpoints/resnet34_cbam_masked.pt", history_path_file="history_cbam_34_masked.json")

    # === Evaluate Model on val_loader directly ===
    evaluate_model(model, test_loader, class_names, device)
    