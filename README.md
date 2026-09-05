# Cancer Image Detection with PyTorch

A transfer-learning image classifier (ResNet18 backbone) for detecting cancer
from medical images (e.g., skin lesion, histopathology, or mammography
patches). Built with PyTorch, ready to train on your own labeled dataset.

> **Disclaimer:** This project is for educational/research purposes only.
> It is **not** a medical device and must not be used for real clinical
> diagnosis. Always consult a qualified medical professional.

## Features

- Transfer learning with a pretrained ResNet18 (swap-in ready for other
  torchvision backbones)
- Handles binary classification out of the box (e.g., `benign` vs
  `malignant`), extendable to multi-class
- Train/validation split, data augmentation, early stopping
- Metrics: accuracy, precision, recall, F1, confusion matrix, ROC-AUC
- Saves the best checkpoint automatically
- Simple CLI for training and single-image prediction
- GPU-aware (CUDA / Apple MPS / CPU fallback)

## Project structure

```
cancer-image-detection/
├── data/
│   ├── train/
│   │   ├── benign/
│   │   └── malignant/
│   └── val/
│       ├── benign/
│       └── malignant/
├── models/                 # saved checkpoints land here
├── src/
│   ├── dataset.py          # data loading & transforms
│   ├── model.py            # model definition
│   ├── train.py            # training loop / CLI entry point
│   ├── evaluate.py         # metrics & confusion matrix
│   ├── predict.py          # inference on a single image
│   └── utils.py            # helpers (seed, device, checkpoints)
├── requirements.txt
└── README.md
```

## Dataset format

Organize your images using the standard `torchvision.datasets.ImageFolder`
layout — one subfolder per class:

```
data/train/benign/img001.png
data/train/malignant/img002.png
data/val/benign/img101.png
data/val/malignant/img102.png
```

Public datasets you can use to try this out:
- [ISIC Skin Cancer Dataset](https://www.isic-archive.com/)
- [BreakHis (breast histopathology)](https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/)
- [Kaggle Histopathologic Cancer Detection](https://www.kaggle.com/c/histopathologic-cancer-detection)

## Setup

```bash
git clone https://github.com/<your-username>/cancer-image-detection.git
cd cancer-image-detection
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Training

```bash
python src/train.py \
    --data-dir data \
    --epochs 20 \
    --batch-size 32 \
    --lr 1e-4 \
    --model resnet18 \
    --output models/best_model.pth
```

Key arguments:

| Flag | Default | Description |
|------|---------|-------------|
| `--data-dir` | `data` | Root folder containing `train/` and `val/` |
| `--epochs` | `20` | Number of training epochs |
| `--batch-size` | `32` | Batch size |
| `--lr` | `1e-4` | Learning rate |
| `--model` | `resnet18` | Backbone (`resnet18`, `resnet34`, `resnet50`, `efficientnet_b0`) |
| `--freeze-backbone` | `False` | Freeze pretrained layers, train only the classifier head |
| `--patience` | `5` | Early stopping patience (epochs without improvement) |
| `--output` | `models/best_model.pth` | Path to save the best checkpoint |

Training prints per-epoch loss/accuracy and saves the best-performing
checkpoint (by validation F1) automatically.

## Evaluation

```bash
python src/evaluate.py --data-dir data --checkpoint models/best_model.pth
```

Outputs accuracy, precision, recall, F1, ROC-AUC, and a confusion matrix.

## Predicting on a single image

```bash
python src/predict.py --image path/to/image.jpg --checkpoint models/best_model.pth
```

```
Prediction: malignant (confidence: 0.87)
```

## How it works

1. **`dataset.py`** loads images via `ImageFolder`, applies augmentation
   (random flips, rotation, color jitter) for training and simple
   resize/normalize for validation.
2. **`model.py`** loads a pretrained torchvision backbone and replaces the
   final fully-connected layer with a new head sized to your number of
   classes.
3. **`train.py`** runs the training loop with `CrossEntropyLoss` and Adam,
   tracks validation F1 for early stopping, and checkpoints the best model.
4. **`evaluate.py`** reloads a checkpoint and reports classification metrics
   on the validation/test set.
5. **`predict.py`** loads a checkpoint and classifies a single new image.

## Extending this project

- Swap in a different backbone in `model.py` (e.g., `efficientnet_b0`,
  `densenet121`, or a Vision Transformer via `timm`)
- Add Grad-CAM visualization to see which regions influenced the prediction
- Add k-fold cross-validation for small datasets
- Export to ONNX/TorchScript for deployment

## License

MIT — see [LICENSE](LICENSE).
