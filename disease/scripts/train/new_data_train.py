import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch.nn as nn
from sklearn.metrics import confusion_matrix, classification_report
from torchvision import datasets, transforms
from src.classify_diseases.models.cbam import ResNet34CBAMClassifier
from torch.utils.data import DataLoader, random_split
from collections import Counter
import os
import json
def fine_tune_model_full(
    model_class,
    model_path,
    dataset_dir="",
    batch_size=64,
    device="cuda" if torch.cuda.is_available() else "cpu",
    epochs=25,
    lr= 8.792645881654775e-05,
    model_save_path="",
    model_history_path="",
):
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": []
    }
    # Define split proportions
    train_ratio = 0.6
    val_ratio = 0.2
    # 1. Dataset setup
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
        transforms.ToTensor()
    ])


    dataset = datasets.ImageFolder(dataset_dir, transform=transform)
    class_names = dataset.classes

    # Compute class weights for balancing
    labels = [label for _, label in dataset.samples]
    class_counts = Counter(labels)
    total = sum(class_counts.values())
    weights = [total / class_counts[i] for i in range(len(class_counts))]
    class_weights = torch.tensor(weights).float().to(device)

    total_size = len(dataset)
    train_len = int(train_ratio * total_size)
    val_len = int(val_ratio * total_size)
    test_len = total_size - train_len - val_len  # Remaining goes to test to ensure full coverage

    # Perform the split
    train_set, val_set, test_set = random_split(dataset, [train_len, val_len, test_len])

    # DataLoaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    # 2. Model setup
    model = model_class(num_classes=len(class_names))
    state_dict = torch.load(model_path, map_location=device)
    filtered_state_dict = {k: v for k, v in state_dict.items() if "fc" not in k}
    model.load_state_dict(filtered_state_dict, strict=False)
    model.to(device)

    # No freezing — train all layers
    for param in model.parameters():
        param.requires_grad = True

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0032532739689025893)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.14696613186872526)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3)
    # === 3. Training loop (with val tracking) ===
    print("Fine-tuning entire model...")
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)
            train_correct += (outputs.argmax(dim=1) == labels).sum().item()
            train_total += labels.size(0)

        avg_train_loss = train_loss / train_total
        train_acc = train_correct / train_total

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * imgs.size(0)
                val_correct += (outputs.argmax(dim=1) == labels).sum().item()
                val_total += labels.size(0)

        avg_val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        # Log metrics
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        print(f"[Epoch {epoch+1}/{epochs}] "
              f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc*100:.2f}% || "
              f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")

        scheduler.step(avg_val_loss)
    checkpoint_dir = os.path.dirname(model_save_path)
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    # === Save training history ===
    history_path = os.path.join(checkpoint_dir, model_history_path)
    with open(history_path, "w") as f:
        json.dump(history, f)
    print(f"Training history saved to {history_path}")
    # === Save model state dict ===
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
    # === 4. Final Evaluation on Test Set ===
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    test_acc = np.mean(np.array(all_preds) == np.array(all_labels))
    print(f"\nFinal Accuracy on test set: {test_acc * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix (Test)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()


fine_tune_model_full(
    model_class=ResNet34CBAMClassifier,
    model_path="checkpoints/resnet34_cbam.pt",
    dataset_dir="data/rice_images",
    device="cuda" if torch.cuda.is_available() else "cpu",
    epochs=30, 
    model_save_path="checkpoints/resnet34_cbam_finetuned_new_data.pt",
    model_history_path="history_finetune_new_data.json",
)