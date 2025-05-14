import optuna
import sys
import os
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models.cbam import ResNetAgeWithCBAM
from src.trainer.trainer import AgeTrainer
from src.dataloader import get_dataloaders
from torchvision import transforms

def objective(trial):
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-2)

    config = {
        "train": {
            "lr": lr,
            "lr_step": 10,
            "lr_gamma": 0.5,
            "epochs": 10,
            "model": "cbam",
            "model_name": f"cbam_trial_{trial.number}",
            "checkpoint_dir": "age/checkpoints/hyperparameter",
            "weighted_sampling": False,
            "force_cpu": False,
            "early_stopping": True,
            "patience": 3
        },
        "dataset": {
            "image_dir": "age/data/train_images",
            "csv_path": "age/data/meta_train.csv",
            "batch_size": 64,
            "val_size": 0.2,
            "random_seed": 42,
            "oversample": False
        }
    }

    image_size = config['dataset'].get("image_size", 224)
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

    train_loader, val_loader = get_dataloaders(
        image_dir=config['dataset']['image_dir'],
        labels_path=config['dataset']['csv_path'],
        batch_size=config['dataset']['batch_size'],
        val_size=config['dataset']['val_size'],
        random_seed=config['dataset']['random_seed'],
        train_transform=train_transform,
        val_transform=val_transform,
        oversample=config['dataset']['oversample'],
        weighted_sampling=config['train']['weighted_sampling']
    )

    model = ResNetAgeWithCBAM()
    trainer = AgeTrainer(model=model, config=config)
    trainer.fit(train_loader, val_loader)

    return min(trainer.log['val_mae'])

if __name__ == '__main__':
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)

    print(study.best_value)
    print("Best trial:")
    trial = study.best_trial

    for key, value in trial.params.items():
        print(f"  {key}: {value}")

    # Save study results to plot
    optuna.visualization.matplotlib.plot_optimization_history(study)
    plt.title("Optimization History (MAE)")
    plt.tight_layout()
    os.makedirs("age/checkpoints/hyperparameter", exist_ok=True)
    plt.savefig("age/checkpoints/hyperparameter/optuna_mae_history.png")
    plt.close()

    print("Saved optimization history plot.")
