"""Run inference on a single image using a trained checkpoint.

Example:
    python src/predict.py --image sample.jpg --checkpoint models/best_model.pth
"""
import argparse

import torch
from PIL import Image

from dataset import get_eval_transforms
from model import build_model
from utils import get_device, load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Predict cancer class for a single image")
    parser.add_argument("--image", type=str, required=True, help="Path to an image file")
    parser.add_argument("--checkpoint", type=str, default="models/best_model.pth")
    parser.add_argument("--model", type=str, default="resnet18")
    return parser.parse_args()


def predict(image_path: str, checkpoint_path: str, backbone: str = "resnet18"):
    device = get_device()
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    class_names = checkpoint["class_names"]

    model = build_model(backbone, num_classes=len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    transform = get_eval_transforms()
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1).squeeze(0)
        pred_idx = int(probs.argmax())

    return class_names[pred_idx], float(probs[pred_idx])


def main():
    args = parse_args()
    label, confidence = predict(args.image, args.checkpoint, args.model)
    print(f"Prediction: {label} (confidence: {confidence:.2f})")


if __name__ == "__main__":
    main()
