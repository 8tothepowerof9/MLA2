import torch
from utils.datamodule import get_dataloaders
from src.classify_diseases.models.cbam import ResNet34CBAMClassifier
from src.classify_diseases.trainer import train_model
from src.classify_diseases.eval import evaluate_model_tune
def jaya_objective(hyperparams):
    lr, mixup_alpha, cutmix_enabled = hyperparams

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
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

    # === Train ===
    train_model(
        model, train_loader, val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        epochs=10,  # keep low during tuning
        scheduler=scheduler,
        mixup=False,
        cutmix=bool(round(cutmix_enabled)),  # 0 or 1 float → bool
        save_path="checkpoints/tmp.pt",
        history_path_file=None,
        mixup_alpha=mixup_alpha
    )

    # === Evaluate ===
    acc = evaluate_model_tune(model, test_loader, device)
    print(f"Hyperparams: {hyperparams}, Accuracy: {acc:.4f}")
    return -acc  # Jaya minimizes


import numpy as np

def jaya(objective, bounds, pop_size=5, generations=5):
    dim = len(bounds)
    pop = np.random.rand(pop_size, dim)

    for i in range(dim):
        pop[:, i] = bounds[i][0] + pop[:, i] * (bounds[i][1] - bounds[i][0])

    for gen in range(generations):
        scores = np.array([objective(ind) for ind in pop])
        best = pop[np.argmin(scores)]
        worst = pop[np.argmax(scores)]

        for i in range(pop_size):
            r1, r2 = np.random.rand(dim), np.random.rand(dim)
            pop[i] += r1 * (best - abs(pop[i])) - r2 * (worst - abs(pop[i]))
            for d in range(dim):
                pop[i][d] = np.clip(pop[i][d], bounds[d][0], bounds[d][1])

        print(f"Gen {gen+1}: Best score = {-min(scores):.4f}")
    
    return pop[np.argmin(scores)], -min(scores)

if __name__ == "__main__":
    bounds = [
        (1e-5, 1e-3),     # learning rate
        (0.1, 1.0),       # mixup alpha
        (0.0, 1.0),       # cutmix enabled (0 or 1)
    ]

    best_params, best_score = jaya(jaya_objective, bounds, pop_size=5, generations=5)
    print("Best hyperparameters:", best_params)
    print("Best validation score:", best_score)
