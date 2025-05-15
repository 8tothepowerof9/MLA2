
from src.dataloader import get_dataloaders
from src.models.resnet_age import ResNetAge
from src.trainer.trainer import AgeTrainer
from src.models.baseline import Baseline
from src.models.cbam import ResNetAgeWithCBAM
from src.models.other_model import CNNAgeRegressor
from src.models.cbam import EfficientNetAgeWithCBAM

from torchvision import transforms
from src.utils import ModelVisualizer
import torch
import os

def train_model(config):
    image_size = config['dataset']['image_size']
    model_name = config['train']['model_name']
    checkpoint_dir = os.path.join(config['train']['checkpoint_dir'], model_name)
    log_path = os.path.join(checkpoint_dir, f"{model_name}.csv")
    checkpoint_path = os.path.join(checkpoint_dir, "best_model.pt")
    csv_path = config['dataset']['csv_path']

    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Dataloaders
    train_loader, val_loader = get_dataloaders(
        image_dir=config['dataset']['image_dir'],
        labels_path=config['dataset']['csv_path'],
        batch_size=config['dataset']['batch_size'],
        val_size=config['dataset']['val_size'],
        random_seed=config['dataset'].get('random_seed', 42),
        train_transform=train_transform,
        val_transform=val_transform,
        oversample=config['dataset']['oversample'],
        weighted_sampling=config['train']['weighted_sampling'],
        force_cpu=config['train']['force_cpu']
    )

    # Model selection
    model_map = {
        'baseline': Baseline,
        'resnet': ResNetAge,
        'cbam': ResNetAgeWithCBAM,
        'other': CNNAgeRegressor,
        'efficientnet': EfficientNetAgeWithCBAM,
    }
    model_class = model_map.get(config['train']['model'])
    model = model_class()

    # Trainer
    trainer = AgeTrainer(model=model, config=config)
    if config['train']['fitting'] == True:
        trainer.fit(train_loader, val_loader)

    # Visualizations
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    visualizer = ModelVisualizer(output_dir=checkpoint_dir, model_name=model_name)
    if config['train']['plotting'] == True:
        visualizer.plot_metrics(log_path)
        visualizer.plot_residuals(trainer.model, val_loader, trainer.device)
    
    if config['train']['grad-camp'] == True:
        model = model_class(use_gradcam=True)

        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        model.to(trainer.device)

        sample = val_loader.dataset[config['train']['grad_camp_sample']]
        visualizer.plot_gradcam(model, sample, trainer.device, csv_path=csv_path)
