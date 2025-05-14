import optuna
import torch.nn.functional as F
from torchmetrics.classification import MulticlassF1Score
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


def plot_optuna_trials(study):
    trial_ids = []
    f1_scores = []

    for trial in study.trials:
        if trial.value is not None:
            trial_ids.append(trial.number)
            f1_scores.append(trial.value)

    plt.figure(figsize=(10, 6))
    plt.scatter(trial_ids, f1_scores, color="blue", s=60)
    plt.title("Optuna Trial F1 Scores")
    plt.xlabel("Trial ID")
    plt.ylabel("Validation F1 Score")
    plt.ylim(0.68, 0.92)  # consistent with previous mock plots
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def objective(trial):
    # ----- Suggest hyperparameters -----
    lr = trial.suggest_categorical("lr", [1e-5, 5e-5, 1e-4, 5e-4])
    weight_decay = trial.suggest_categorical("weight_decay", [1e-6, 1e-5, 1e-4, 1e-3])
    step_size = trial.suggest_categorical("step_size", [5, 7, 10, 15])
    gamma = trial.suggest_categorical("gamma", [0.3, 0.5, 0.7, 0.9])
    use_mixup = trial.suggest_categorical("mixup", [True, False])

    # ----- Setup everything -----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3),
            transforms.ToTensor(),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    train_dataloader, val_dataloader = get_dataloaders(
        "data/train_images",
        "data/meta_train.csv",
        "variety",
        train_transform=train_transform,
        val_transform=val_transform,
        batch_size=32,
    )

    model = CBAMResNet18(num_classes=10, weights="DEFAULT").to(device)

    loss_fn = FocalLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=step_size, gamma=gamma
    )
    metric = MulticlassF1Score(num_classes=10, average="weighted").to(device)

    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        metric=metric,
        device=device,
        model_name=f"trial_model_{trial.number}",
        save=False,
        mixup=use_mixup,
    )

    trainer.fit(train_dataloader, val_dataloader, epochs=10)  # Fewer epochs for tuning

    val_score = trainer.history["val_metric"][-1]  # Use final val F1
    return val_score


if __name__ == "__main__":
    study = optuna.create_study(
        direction="maximize",  # We're maximizing F1-score
        study_name="cbam_variety_tuning",
    )
    study.optimize(objective, n_trials=25, timeout=60 * 60)  # 25 trials or 1 hour

    print("Best trial:")
    print(study.best_trial)
    print("Best params:", study.best_params)

    plot_optuna_trials(study)
