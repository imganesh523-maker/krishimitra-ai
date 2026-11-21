"""Model definition and image preprocessing shared by training and the app."""
import torch
import torch.nn as nn
from torchvision import models, transforms

IMG_SIZE = 160
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def build_model(num_classes: int = 38, pretrained: bool = True) -> nn.Module:
    """MobileNetV2 backbone (ImageNet weights) with a new classification head."""
    weights = models.MobileNet_V2_Weights.IMAGENET1K_V2 if pretrained else None
    model = models.mobilenet_v2(weights=weights)
    in_features = model.classifier[1].in_features  # 1280
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes),
    )
    return model


def train_transform():
    return transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def eval_transform():
    return transforms.Compose([
        transforms.Resize(176),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def load_model(path: str, num_classes: int = 38) -> nn.Module:
    model = build_model(num_classes, pretrained=False)
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model
