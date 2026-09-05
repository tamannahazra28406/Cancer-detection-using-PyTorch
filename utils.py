"""Shared utility helpers: reproducibility, device selection, checkpoints."""
import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Fix random seeds across libraries for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Pick the best available device: CUDA > Apple MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_checkpoint(model, optimizer, epoch, best_metric, class_names, path):
    """Save a training checkpoint including model/optimizer state and metadata."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
            "best_metric": best_metric,
            "class_names": class_names,
        },
        path,
    )


def load_checkpoint(path, map_location=None):
    """Load a checkpoint dictionary saved by `save_checkpoint`."""
    return torch.load(path, map_location=map_location)


class EarlyStopper:
    """Stops training when a monitored metric stops improving."""

    def __init__(self, patience: int = 5, mode: str = "max"):
        self.patience = patience
        self.mode = mode
        self.best = None
        self.counter = 0
        self.should_stop = False

    def step(self, value: float) -> bool:
        """Update state with the latest metric value. Returns True if this is a new best."""
        is_better = (
            self.best is None
            or (self.mode == "max" and value > self.best)
            or (self.mode == "min" and value < self.best)
        )
        if is_better:
            self.best = value
            self.counter = 0
            return True
        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
        return False
