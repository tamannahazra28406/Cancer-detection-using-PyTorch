"""Model factory: pretrained torchvision backbones with a custom classifier head."""
import torch.nn as nn
from torchvision import models


SUPPORTED_BACKBONES = ("resnet18", "resnet34", "resnet50", "efficientnet_b0")


def build_model(backbone: str = "resnet18", num_classes: int = 2, freeze_backbone: bool = False) -> nn.Module:
    """Create a pretrained classifier.

    Args:
        backbone: one of SUPPORTED_BACKBONES
        num_classes: number of output classes (2 for benign/malignant)
        freeze_backbone: if True, only the new classifier head is trainable
    """
    backbone = backbone.lower()

    if backbone == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_features, num_classes))
    elif backbone == "resnet34":
        model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_features, num_classes))
    elif backbone == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_features, num_classes))
    elif backbone == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unsupported backbone '{backbone}'. Choose from {SUPPORTED_BACKBONES}")

    if freeze_backbone:
        for name, param in model.named_parameters():
            # Keep the newly-added classifier head trainable; freeze everything else.
            if "fc" not in name and "classifier" not in name:
                param.requires_grad = False

    return model
