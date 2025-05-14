import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms
from test.test_rebalance import get_balanced_dataframe, ImageCSVLoader

def get_dataloaders(
    csv_path,
    image_dir,
    label_col="label",
    batch_size=32,
    strategy="weights",  # "oversample", "undersample", or None
    num_workers=4,
    pin_memory=True,
    shuffle=True,
    seed=42
):
    # === Step 1: Load original dataframe ===
    df = pd.read_csv(csv_path)

    # === Step 2: First split (train vs temp 60/40) ===
    df_train, df_temp = train_test_split(
        df,
        test_size=0.4,
        stratify=df[label_col],
        random_state=seed
    )

    # === Step 3: Split temp into val and test (each 20%) ===
    df_val, df_test = train_test_split(
        df_temp,
        test_size=0.5,
        stratify=df_temp[label_col],
        random_state=seed
    )

    # === Step 4: Balance train only ===
    _, df_train_balanced = get_balanced_dataframe(df_train, label_col, strategy=strategy)

    # === Step 5: Define transforms ===
    transform = transforms.Compose([
                    transforms.Resize(256),  # Resize so shorter side is 256, keeps aspect ratio
                    transforms.CenterCrop(224),  # Crop center to 224x224 (no distortion)
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
                    transforms.RandomRotation(degrees=10),
                    transforms.ToTensor(),
                    transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
                ])
    # === Step 6: Create datasets ===
    train_ds = ImageCSVLoader(df_train_balanced, image_dir, transform, label_col)
    val_ds = ImageCSVLoader(df_val, image_dir, transform, label_col)
    test_ds = ImageCSVLoader(df_test, image_dir, transform, label_col)

    # === Step 7: Create dataloaders ===
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle,
                              num_workers=num_workers, pin_memory=pin_memory,
                              persistent_workers=True, prefetch_factor=2)

    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory,
                            persistent_workers=True, prefetch_factor=2)

    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin_memory,
                             persistent_workers=True, prefetch_factor=2)

    return train_loader, val_loader, test_loader, train_ds.class_names
