"""Inference + Grad-CAM for KrishiMitra AI."""
import json
import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.cm as cm

from .classes import CLASS_NAMES
from .model import eval_transform, load_model

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT, "models", "krishimitra_mobilenetv2.pt")
KB_PATH = os.path.join(ROOT, "data", "knowledge_base.json")


@dataclass
class Prediction:
    label: str
    confidence: float
    top3: list  # [(label, prob), ...]
    probs: np.ndarray


class GradCAM:
    """Grad-CAM (Selvaraju et al., 2017) on the last convolutional block of MobileNetV2."""

    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._save)

    def _save(self, module, inp, out):
        self.activations = out
        if out.requires_grad:  # only during the Grad-CAM pass (plain predictions run under no_grad)
            out.register_hook(lambda g: setattr(self, "gradients", g))

    def __call__(self, x, class_idx=None):
        self.model.zero_grad()
        logits = self.model(x)
        if class_idx is None:
            class_idx = int(logits.argmax(1))
        logits[0, class_idx].backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)          # GAP of gradients -> channel weights
        cam = F.relu((weights * self.activations).sum(1, keepdim=True))  # weighted sum of feature maps
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.detach().numpy(), torch.softmax(logits, 1)[0].detach().numpy()


class Predictor:
    def __init__(self, model_path: str = MODEL_PATH, kb_path: str = KB_PATH):
        self.model = load_model(model_path, len(CLASS_NAMES))
        self.tf = eval_transform()
        self.kb = json.load(open(kb_path, encoding="utf-8"))
        self.cam = GradCAM(self.model, self.model.features[-1])

    def _tensor(self, img: Image.Image):
        return self.tf(img.convert("RGB")).unsqueeze(0)

    def predict(self, img: Image.Image) -> Prediction:
        with torch.no_grad():
            probs = torch.softmax(self.model(self._tensor(img)), 1)[0].numpy()
        top = probs.argsort()[::-1][:3]
        return Prediction(CLASS_NAMES[top[0]], float(probs[top[0]]),
                          [(CLASS_NAMES[i], float(probs[i])) for i in top], probs)

    def gradcam(self, img: Image.Image, alpha: float = 0.45):
        """Returns (overlay PIL image, raw heatmap 0-1, probs)."""
        x = self._tensor(img).requires_grad_(False)
        with torch.enable_grad():
            cam, probs = self.cam(x)
        # overlay on the same centre-cropped view the model saw
        view = img.convert("RGB").resize((176, 176)).crop((8, 8, 168, 168))
        heat = (cm.jet(cam)[..., :3] * 255).astype(np.uint8)
        overlay = (np.asarray(view) * (1 - alpha) + heat * alpha).astype(np.uint8)
        return Image.fromarray(overlay).resize((320, 320), Image.BICUBIC), cam, probs

    def info(self, label: str) -> dict:
        return self.kb[label]
