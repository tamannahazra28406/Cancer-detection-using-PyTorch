"""Train a cancer image classifier.

Example:
    python src/train.py --data-dir data --epochs 20 --model resnet18
"""
import argparse
import time

import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from tqdm import tqdm

from dataset import get_dataloaders
from model import build_model, SUPPORTED_BACKBONES
from utils import set_seed, get_device, save_checkpoint, EarlyStopper


def parse_args():
    parser = argparse.ArgumentParser(description="Train a cancer image classifier")
    parser.add_argument("--data-dir", type=str, default="data", help="Root data directory")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--model", type=str, default="resnet18", choices=SUPPORTED_BACKBONES)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--output", type=str, default="models/best_model.pth")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()

    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, labels in tqdm(loader, desc="train" if train else "val", leave=False):
            images, labels = images.to(device), labels.to(device)

            if train:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / total
    accuracy = correct / total
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, accuracy, f1


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    print(f"Using device: {device}")

    train_loader, val_loader, class_names = get_dataloaders(
        args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers
    )
    print(f"Classes: {class_names}")
    print(f"Train samples: {len(train_loader.dataset)} | Val samples: {len(val_loader.dataset)}")

    model = build_model(args.model, num_classes=len(class_names), freeze_backbone=args.freeze_backbone)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    early_stopper = EarlyStopper(patience=args.patience, mode="max")

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss, train_acc, train_f1 = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc, val_f1 = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step(val_f1)
        elapsed = time.time() - start

        print(
            f"Epoch {epoch}/{args.epochs} ({elapsed:.1f}s) | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} train_f1={train_f1:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f}"
        )

        is_best = early_stopper.step(val_f1)
        if is_best:
            save_checkpoint(model, optimizer, epoch, val_f1, class_names, args.output)
            print(f"  -> New best model saved to {args.output} (val_f1={val_f1:.4f})")

        if early_stopper.should_stop:
            print(f"Early stopping triggered after {epoch} epochs (no improvement for {args.patience} epochs).")
            break

    print("Training complete.")


if __name__ == "__main__":
    main()
