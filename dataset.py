"""Data loading utilities built on torchvision.datasets.ImageFolder.

Expected directory layout:

    data/
      train/
        benign/
        malignant/
      val/
        benign/
        malignant/
"""
import os

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

IMAGE_SIZE = 224

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(image_size: int = IMAGE_SIZE):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def get_eval_transforms(image_size: int = IMAGE_SIZE):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def get_dataloaders(data_dir: str, batch_size: int = 32, num_workers: int = 2, image_size: int = IMAGE_SIZE):
    """Build train/val DataLoaders from an ImageFolder-structured directory.

    Returns:
        train_loader, val_loader, class_names (list[str])
    """
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    if not os.path.isdir(train_dir) or not os.path.isdir(val_dir):
        raise FileNotFoundError(
            f"Expected '{train_dir}' and '{val_dir}' to exist. "
            "Organize your data as data/train/<class>/*.jpg and data/val/<class>/*.jpg"
        )

    train_dataset = datasets.ImageFolder(train_dir, transform=get_train_transforms(image_size))
    val_dataset = datasets.ImageFolder(val_dir, transform=get_eval_transforms(image_size))

    if train_dataset.classes != val_dataset.classes:
        raise ValueError(
            f"Train classes {train_dataset.classes} do not match val classes {val_dataset.classes}"
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, train_dataset.classes
