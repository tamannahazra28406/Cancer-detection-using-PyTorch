"""Evaluate a trained checkpoint on the validation (or test) set.

Example:
    python src/evaluate.py --data-dir data --checkpoint models/best_model.pth
"""
import argparse

import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

from dataset import get_dataloaders
from model import build_model
from utils import get_device, load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained cancer image classifier")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--checkpoint", type=str, default="models/best_model.pth")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--model", type=str, default="resnet18")
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device()

    checkpoint = load_checkpoint(args.checkpoint, map_location=device)
    class_names = checkpoint["class_names"]

    _, val_loader, _ = get_dataloaders(args.data_dir, batch_size=args.batch_size)

    model = build_model(args.model, num_classes=len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = probs.argmax(dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.tolist())
            # Probability of the positive (index 1) class, for binary ROC-AUC
            if probs.shape[1] == 2:
                all_probs.extend(probs[:, 1].cpu().tolist())

    print(f"Classes: {class_names}\n")
    print(f"Accuracy:  {accuracy_score(all_labels, all_preds):.4f}")
    print(f"Precision: {precision_score(all_labels, all_preds, average='macro', zero_division=0):.4f}")
    print(f"Recall:    {recall_score(all_labels, all_preds, average='macro', zero_division=0):.4f}")
    print(f"F1:        {f1_score(all_labels, all_preds, average='macro', zero_division=0):.4f}")

    if len(class_names) == 2 and all_probs:
        try:
            auc = roc_auc_score(all_labels, all_probs)
            print(f"ROC-AUC:   {auc:.4f}")
        except ValueError:
            pass

    print("\nConfusion matrix (rows=true, cols=predicted):")
    print(confusion_matrix(all_labels, all_preds))

    print("\nClassification report:")
    print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))


if __name__ == "__main__":
    main()
