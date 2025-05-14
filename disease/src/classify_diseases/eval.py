import json
import matplotlib.pyplot as plt
import os
from sklearn.metrics import confusion_matrix
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
def plot_training_history(history_path: str):
    """
    Plots training and validation loss/accuracy curves from a saved training history file.

    Args:
        history_path (str): Path to the JSON file containing training history.
    """
    if not os.path.exists(history_path):
        print(f"❌ File not found: {history_path}")
        return

    with open(history_path, "r") as f:
        history = json.load(f)

    epochs = range(1, len(history["train_loss"]) + 1)

    # === Plot Loss ===
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss", marker="o")
    plt.plot(epochs, history["val_loss"], label="Val Loss", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.grid(True)

    # === Plot Accuracy ===
    plt.subplot(1, 2, 2)
    plt.plot(
        epochs, [x * 100 for x in history["train_acc"]], label="Train Acc", marker="o"
    )
    plt.plot(epochs, [x * 100 for x in history["val_acc"]], label="Val Acc", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training & Validation Accuracy")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names, title="Confusion Matrix", cmap="Blues", normalize=False):
    """
    Plots a confusion matrix with optional normalization.

    Args:
        y_true (list or array): Ground truth labels
        y_pred (list or array): Predicted labels
        class_names (list): List of class names
        title (str): Title of the plot
        cmap (str): Matplotlib colormap
        normalize (bool): If True, normalize the confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt=".2f" if normalize else "d", cmap=cmap,
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title(title + (" (Normalized)" if normalize else ""))
    plt.tight_layout()
    plt.show()


def evaluate_model(model, dataloader, class_names, device):
    """
    Evaluates a PyTorch model on a given dataloader.
    
    Args:
        model (torch.nn.Module): Trained model
        dataloader (DataLoader): Validation/test DataLoader
        class_names (list): List of class names, sorted
        device (torch.device): The device to run evaluation on
    """
    print(f"🖥️ Evaluating on device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    model.eval()
    
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    print(f"\n Accuracy: {acc * 100:.2f}%")
    print("\n Classification Report:")
    cls_report = classification_report(all_labels, all_preds, target_names=class_names)
    print(cls_report)
    plot_confusion_matrix(all_labels, all_preds, class_names)

    return {
        "accuracy": acc,
        "predictions": all_preds,
        "labels": all_labels,
        "class_names": class_names
    } 
    
    
def evaluate_model_tune(model, dataloader, device):
    """
    Evaluates a PyTorch model and returns only overall accuracy.

    Args:
        model (torch.nn.Module): Trained model
        dataloader (DataLoader): Validation/test DataLoader
        device (torch.device): The device to run evaluation on

    Returns:
        float: accuracy (0.0 to 1.0)
    """
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    return acc