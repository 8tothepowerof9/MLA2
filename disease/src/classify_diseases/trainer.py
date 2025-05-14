import torch
import os
import json
from utils.early_stopping import EarlyStopping
import torch
import numpy as np

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def mixup_data(x, y, alpha=0.4):
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size()[0]
    index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def cutmix_data(x, y, alpha=0.4):
    lam = np.random.beta(alpha, alpha)
    rand_index = torch.randperm(x.size()[0])
    y_a = y
    y_b = y[rand_index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bby1:bby2, bbx1:bbx2] = x[rand_index, :, bby1:bby2, bbx1:bbx2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    return x, y_a, y_b, lam


def train_model(model, train_loader, val_loader, optimizer, loss_fn, device, epochs=10, mixup_alpha=0.4, patience=10, save_path="", scheduler=None, cutmix=False, mixup=False, history_path_file=""):
    early_stopping = EarlyStopping(patience=patience, verbose=True, path=save_path)
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": []
    }

    for epoch in range(epochs):
        # === Training Phase ===
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            # === Apply MixUp or CutMix if enabled ===
            if mixup:
                images, targets_a, targets_b, lam = mixup_data(images, labels, alpha=mixup_alpha)
                outputs = model(images)
                loss = lam * loss_fn(outputs, targets_a) + (1 - lam) * loss_fn(outputs, targets_b)
                _, preds = outputs.max(1)
                correct += (lam * preds.eq(targets_a).sum().item() + (1 - lam) * preds.eq(targets_b).sum().item())

            elif cutmix:
                images, targets_a, targets_b, lam = cutmix_data(images, labels, alpha=mixup_alpha)
                outputs = model(images)
                loss = lam * loss_fn(outputs, targets_a) + (1 - lam) * loss_fn(outputs, targets_b)
                _, preds = outputs.max(1)
                correct += (lam * preds.eq(targets_a).sum().item() + (1 - lam) * preds.eq(targets_b).sum().item())

            else:
                outputs = model(images)
                loss = loss_fn(outputs, labels)
                _, preds = outputs.max(1)
                correct += preds.eq(labels).sum().item()

            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            total += labels.size(0)

        epoch_train_loss = train_loss / total
        epoch_train_acc = correct / total

        # === Validation Phase ===
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = loss_fn(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, preds = outputs.max(1)
                correct += preds.eq(labels).sum().item()
                total += labels.size(0)

        epoch_val_loss = val_loss / total
        epoch_val_acc = correct / total
        scheduler.step(epoch_val_loss)

        # Log metrics
        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_acc"].append(epoch_val_acc)

        print(f"[{epoch+1}/{epochs}] Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | "
              f"Train Acc: {epoch_train_acc*100:.2f}% | Val Acc: {epoch_val_acc*100:.2f}%")

        # === Early Stopping Check ===
        early_stopping(epoch_val_loss, model)
        if early_stopping.early_stop:
            print("🛑 Early stopping triggered.")
            break
    # === Save history ===
    if history_path_file:
        checkpoint_dir = os.path.dirname(save_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # === Save training history ===
        history_path = os.path.join(checkpoint_dir, history_path_file)
        with open(history_path, "w") as f:
            json.dump(history, f)
        print(f"📊 Training history saved to {history_path}")
    # === Save model state dict ===
    torch.save(model.state_dict(), save_path)
    print(f"💾 Model saved to {save_path}")
    return history

