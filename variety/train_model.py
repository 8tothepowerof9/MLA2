import sys
import os
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from torchvision import transforms
from torchmetrics.classification import MulticlassF1Score, MultilabelF1Score
from torch.nn import functional as F
from dataloader import get_dataloaders
from trainer import Trainer
from models import *


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        inputs: logits of shape (B, C)
        targets: soft labels of shape (B, C)
        """
        log_probs = F.log_softmax(inputs, dim=1)
        probs = log_probs.exp()
        focal_weight = (1 - probs) ** self.gamma
        loss = -targets * focal_weight * log_probs

        if self.reduction == "mean":
            return loss.sum(dim=1).mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


if __name__ == "__main__":
    device = torch.device("cuda")

    # ----- Define transforms -----
    train_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
            transforms.ToTensor(),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    # ----- Load data -----
    train_dataloader, val_dataloader = get_dataloaders(
        "data/train_images",
        "data/meta_train.csv",
        "variety",
        train_transform=train_transform,
        val_transform=val_transform,
        batch_size=32,
        oversample=False,
    )

    # ----- Initialize model -----
    model = PaperCNN(num_classes=10)
    model = model.to(device)

    # ----- Define loss, optimizer, scheduler, and metric -----
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)
    # loss_fn = FocalLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    metric = MulticlassF1Score(num_classes=10, average="weighted").to(device)

    # ----- Run training -----
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        metric=metric,
        device=device,
        model_name="PaperCNN",
        save=True,
        mixup=False,
    )

    trainer.fit(train_dataloader, val_dataloader, epochs=20)
