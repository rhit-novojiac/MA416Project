"""
Data loading and augmentation pipeline for DeFungi microscopic image dataset.
"""

import sys
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Sized, cast

# Ensure current directory is in sys.path
_MODULE_DIR = Path(__file__).resolve().parent
if str(_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULE_DIR))

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


def find_dataset_dir(custom_path: Optional[str] = None) -> Path:
    """Locate the defungi dataset directory."""
    if custom_path:
        p = Path(custom_path).resolve()
        if p.exists() and p.is_dir():
            return p
        raise FileNotFoundError(f"Provided dataset directory does not exist: {custom_path}")

    # Search current working directory and parents
    curr = Path.cwd().resolve()
    while not (curr / "defungi").exists() and curr.parent != curr:
        curr = curr.parent

    target = curr / "defungi"
    if target.exists() and target.is_dir():
        return target

    # Search relative to this script's directory and parents
    file_dir = Path(__file__).resolve().parent
    while not (file_dir / "defungi").exists() and file_dir.parent != file_dir:
        file_dir = file_dir.parent

    target = file_dir / "defungi"
    if target.exists() and target.is_dir():
        return target

    raise FileNotFoundError("Could not locate 'defungi' dataset folder in repository hierarchy.")


class DeFungiDataset(Dataset):
    """PyTorch Dataset for DeFungi microscopic image classification."""

    def __init__(self, df: pd.DataFrame, transform: Optional[transforms.Compose] = None):
        self.filepaths: List[str] = [str(p) for p in df["filepath"].tolist()]
        self.labels: List[int] = [int(l) for l in df["label"].tolist()]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.filepaths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        img_path = self.filepaths[index]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[index]

        if self.transform is not None:
            tensor_image = self.transform(image)
        else:
            tensor_image = transforms.functional.to_tensor(image)

        return cast(torch.Tensor, tensor_image), int(label)


def get_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Returns train and evaluation image transformations.
    Uses standard ImageNet normalization and microscopic-appropriate augmentations.
    """
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=30),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    return train_transform, eval_transform


def load_dataset_dataframe(dataset_dir: Path) -> Tuple[pd.DataFrame, List[str], Dict[str, int]]:
    """Scan dataset folder and load image records with labels."""
    records = []
    classes = sorted([d.name for d in dataset_dir.iterdir() if d.is_dir()])
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}

    for cls_name in classes:
        cls_dir = dataset_dir / cls_name
        for img_file in cls_dir.iterdir():
            if img_file.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                records.append({
                    "filepath": str(img_file),
                    "class": cls_name,
                    "label": class_to_idx[cls_name]
                })

    df = pd.DataFrame(records)
    return df, classes, class_to_idx


def get_dataloaders(
    dataset_dir: Optional[str] = None,
    batch_size: int = 32,
    num_workers: int = 2,
    test_size: float = 0.20,
    random_state: int = 42
) -> Tuple[DataLoader, DataLoader, List[str], torch.Tensor]:
    """
    Prepares train and test DataLoaders with stratified train/test split.
    
    Returns:
        train_loader: DataLoader for training set
        test_loader: DataLoader for testing set
        classes: List of class names (e.g. ['H1', 'H2', 'H3', 'H5', 'H6'])
        class_weights: Tensor of inverse-frequency weights for CrossEntropyLoss
    """
    root_dir = find_dataset_dir(dataset_dir)
    df, classes, _ = load_dataset_dataframe(root_dir)

    # Stratified split matching Baseline.ipynb
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df["class"],
        random_state=random_state
    )

    # Compute inverse-frequency class weights: N / (num_classes * N_c)
    num_classes = len(classes)
    class_counts = train_df["label"].value_counts().sort_index().values
    weights = len(train_df) / (num_classes * class_counts)
    class_weights = torch.tensor(weights, dtype=torch.float32)

    train_transform, eval_transform = get_transforms()

    train_dataset = DeFungiDataset(train_df, transform=train_transform)
    test_dataset = DeFungiDataset(test_df, transform=eval_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, test_loader, classes, class_weights


if __name__ == "__main__":
    train_l, test_l, cls_list, c_weights = get_dataloaders(batch_size=16, num_workers=0)
    print(f"Loaded classes: {cls_list}")
    print(f"Train samples: {len(cast(Sized, train_l.dataset)):,}, Batches: {len(train_l)}")
    print(f"Test samples: {len(cast(Sized, test_l.dataset)):,}, Batches: {len(test_l)}")
    print(f"Inverse class weights: {c_weights.tolist()}")

