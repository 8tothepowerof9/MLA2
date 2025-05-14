import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import classification_report
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from collections import Counter
import os

def fine_tune_model_full(
    model_class,
    model_path,
    dataset_dir,
    batch_size=32,
    device="cuda" if torch.cuda.is_available() else "cpu",
    epochs=15,
    lr=1e-4,
    label_smoothing=0.1,
    weight_decay=0.0
):
    train_ratio, val_ratio = 0.6, 0.2
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
        transforms.ToTensor()
    ])

    dataset = datasets.ImageFolder(dataset_dir, transform=transform)
    class_names = dataset.classes

    labels = [label for _, label in dataset.samples]
    class_counts = Counter(labels)
    total = sum(class_counts.values())
    weights = [total / class_counts[i] for i in range(len(class_counts))]
    class_weights = torch.tensor(weights).float().to(device)

    total_size = len(dataset)
    train_len = int(train_ratio * total_size)
    val_len = int(val_ratio * total_size)
    test_len = total_size - train_len - val_len
    train_set, val_set, test_set = random_split(dataset, [train_len, val_len, test_len])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    model = model_class(num_classes=len(class_names))
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict({k: v for k, v in state_dict.items() if "fc" not in k}, strict=False)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3)

    best_val_acc = 0.0
    for epoch in range(epochs):
        model.train()
        train_loss, train_correct = 0.0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
            train_correct += (outputs.argmax(dim=1) == labels).sum().item()

        model.eval()
        val_loss, val_correct = 0.0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * imgs.size(0)
                val_correct += (outputs.argmax(dim=1) == labels).sum().item()

        val_acc = val_correct / len(val_set)
        best_val_acc = max(best_val_acc, val_acc)
        scheduler.step(val_loss)
        print(f"[Epoch {epoch+1}/{epochs}] Val Acc: {val_acc*100:.2f}%")


    return best_val_acc  # for Optuna to optimize


import optuna
from src.classify_diseases.models.cbam import ResNet34CBAMClassifier

def objective(trial):
    params = {
        "lr": trial.suggest_float("lr", 1e-5, 5e-4, log=True),
        "label_smoothing": trial.suggest_float("label_smoothing", 0.0, 0.15),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
        "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.01)
    }

    val_acc = fine_tune_model_full(
        model_class=ResNet34CBAMClassifier,
        model_path="checkpoints/resnet34_cbam_masked.pt",
        dataset_dir="data/masked_rice_images",
        lr=params["lr"],
        label_smoothing=params["label_smoothing"],
        batch_size=params["batch_size"],
        weight_decay=params["weight_decay"],
        epochs=15
    )
    print(f"Trial {trial.number} — Val Acc: {val_acc:.4f} | Params: {trial.params}")
    return val_acc

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    optuna.logging.set_verbosity(optuna.logging.INFO)
    study.optimize(objective, n_trials=20, show_progress_bar=True)

    print("✅ Best parameters:", study.best_params)
    print("🏆 Best validation accuracy:", study.best_value)
