from typing import List, Tuple
import os
import random

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


IMG_SIZE_H = 240
IMG_SIZE_W = 320
BATCH_SIZE = 32


def build_transforms():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE_H, IMG_SIZE_W)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def load_dataset(root: str) -> datasets.ImageFolder:
    """Load dataset from root containing class subfolders `daninha/` and `nao_daninha/`."""
    transform = build_transforms()
    dataset = datasets.ImageFolder(root=root, transform=transform)
    return dataset


def partition_dataset(dataset: datasets.ImageFolder, num_clients: int, seed: int = 42) -> List[Subset]:
    """Randomly partition dataset indices into `num_clients` subsets."""
    rng = random.Random(seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    splits = [indices[i::num_clients] for i in range(num_clients)]
    return [Subset(dataset, idxs) for idxs in splits]


def get_loaders_for_subset(subset: Subset, batch_size: int = BATCH_SIZE) -> Tuple[DataLoader, DataLoader]:
    """Split subset into train/val (80/20) and return dataloaders."""
    indices = list(range(len(subset)))
    split = int(0.8 * len(indices))
    train_indices = indices[:split]
    val_indices = indices[split:]
    train_subset = Subset(subset, train_indices)
    val_subset = Subset(subset, val_indices)
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, val_loader
