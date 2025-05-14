import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

class RiceDataset(Dataset):
    def __init__(
        self,
        image_dir: str,
        labels_path: str,
        split: str = "train",
        transform=None,
        target_transform=None,
        val_size: float = 0.2,
        random_seed: int = 42,
        oversample: bool = False,
    ):
        self.image_dir = image_dir
        self.transform = transform
        self.target_transform = target_transform

        df = pd.read_csv(labels_path)
        train_df, val_df = train_test_split(
            df, test_size=val_size, random_state=random_seed
        )
        self.metadata = train_df if split == "train" else val_df

        if oversample and split == "train":
            age_counts = self.metadata['age'].value_counts()
            max_size = age_counts.max()
            class_dfs = [
                resample(group, replace=True, n_samples=max_size, random_state=random_seed)
                for _, group in self.metadata.groupby("age")
            ]
            self.metadata = pd.concat(class_dfs).sample(frac=1, random_state=random_seed).reset_index(drop=True)

        self.image_paths = [
            os.path.join(image_dir, row["label"], row["image_id"])
            for _, row in self.metadata.iterrows()
        ]
        self.targets = [row["age"] for _, row in self.metadata.iterrows()]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)

        target = torch.tensor(float(self.targets[idx]), dtype=torch.float)
        if self.target_transform:
            target = self.target_transform(target)

        image_id = os.path.basename(self.image_paths[idx])
        return image, target, image_id


def get_dataloaders(
    image_dir: str,
    labels_path: str,
    batch_size: int = 32,
    val_size: float = 0.2,
    random_seed: int = 42,
    train_transform=None,
    val_transform=None,
    target_transform=None,
    oversample: bool = False,
    weighted_sampling: bool = False,
    force_cpu: bool = False,
):
    train_ds = RiceDataset(
        image_dir=image_dir,
        labels_path=labels_path,
        split="train",
        transform=train_transform,
        target_transform=target_transform,
        val_size=val_size,
        random_seed=random_seed,
        oversample=oversample,
    )

    val_ds = RiceDataset(
        image_dir=image_dir,
        labels_path=labels_path,
        split="val",
        transform=val_transform,
        target_transform=target_transform,
        val_size=val_size,
        random_seed=random_seed,
    )

    cpu_cores = os.cpu_count()
    use_cuda = torch.cuda.is_available() and not force_cpu

    loader_args = {
        "batch_size": batch_size,
        "num_workers": cpu_cores // 2,
        "pin_memory": use_cuda,
        "persistent_workers": True,
        "prefetch_factor": 2
    }

    if weighted_sampling:
        age_series = pd.Series(train_ds.targets)
        age_counts = age_series.value_counts()
        weights = age_series.map(lambda x: 1.0 / age_counts[x])
        sampler = WeightedRandomSampler(weights.values, num_samples=len(weights), replacement=True)

        train_loader = DataLoader(
            train_ds,
            sampler=sampler,
            **loader_args
        )
    else:
        train_loader = DataLoader(
            train_ds,
            shuffle=True,
            **loader_args
        )

    val_loader = DataLoader(
        val_ds,
        shuffle=False,
        **loader_args
    )

    return train_loader, val_loader
