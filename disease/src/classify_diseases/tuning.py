import torch
import optuna
from utils.datamodule import get_dataloaders
from src.classify_diseases.models.cbam import ResNet34CBAMClassifier
from src.classify_diseases.trainer import train_model
from src.classify_diseases.eval import evaluate_model_tune
import matplotlib.pyplot as plt

def objective(trial):
    # === Hyperparameter suggestions ===
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    mixup_alpha = trial.suggest_float("mixup_alpha", 0.1, 1.0)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

    # === Device Setup ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === Data ===
    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        csv_path="data/meta_train.csv",
        image_dir="data/train_images",
        label_col="label",
        batch_size=32,
        strategy="weights",
        num_workers=2,
        pin_memory=True
    )

    # === Model ===
    model = ResNet34CBAMClassifier(num_classes=len(class_names)).to(device)

    # === Optimizer & Loss ===
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

    # === Train ===
    train_model(
        model, train_loader, val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        epochs=30,  
        scheduler=scheduler,
        mixup=False,
        cutmix=True,
        save_path="checkpoints/tmp.pt",
        history_path_file=None,
        mixup_alpha=mixup_alpha
    )

    # === Evaluate ===
    acc = evaluate_model_tune(model, test_loader, device)
    trial.set_user_attr("val_accuracy", acc)
    return acc

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
    plt.ylim(0.68, 0.92)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=15)
    plot_optuna_trials(study)
    print("Best trial:")
    trial = study.best_trial
    print(f"  Accuracy: {trial.value:.4f}")
    print("  Params:")
    for key, value in trial.params.items():
        print(f" {key}: {value}")


