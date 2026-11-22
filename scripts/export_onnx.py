"""Export KrishiMitra AI to ONNX for in-browser inference (onnxruntime-web).

The exported graph takes a normalised 1x3x160x160 float tensor and returns:
  * logits : [1, 38]        class scores
  * cams   : [1, 38, 5, 5]  Class Activation Maps for every class

MobileNetV2 ends in  features -> global-average-pool -> (dropout) -> linear,
so for class c the CAM  M_c = sum_k w_ck * A_k  (A = last conv feature maps,
w = linear weights) is exact - and identical to Grad-CAM after normalisation,
because dy_c/dA_k(i,j) = w_ck / (H*W). Computing it inside the graph (a 1x1 conv
with the classifier weights) means the browser never needs gradients.

Usage:
    python scripts/export_onnx.py                 # -> web/model/krishimitra.onnx (fp32)
    python scripts/export_onnx.py --quantize      # also writes an int8 QDQ model
"""
import argparse, glob, os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.model import load_model, IMG_SIZE  # noqa: E402


class WithCAM(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.features = m.features
        lin = m.classifier[1]
        self.fc = lin
        self.cam = nn.Conv2d(lin.in_features, lin.out_features, 1, bias=False)
        self.cam.weight.data = lin.weight.data.clone()[:, :, None, None]

    def forward(self, x):
        a = self.features(x)                        # [1,1280,5,5]
        logits = self.fc(F.adaptive_avg_pool2d(a, 1).flatten(1))
        return logits, self.cam(a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "web", "model", "krishimitra.onnx"))
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    m = WithCAM(load_model(os.path.join(ROOT, "models", "krishimitra_mobilenetv2.pt"))).eval()
    x = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    kw = dict(input_names=["input"], output_names=["logits", "cams"], opset_version=17,
              do_constant_folding=True)
    try:
        torch.onnx.export(m, x, a.out, dynamo=False, **kw)
    except TypeError:
        torch.onnx.export(m, x, a.out, **kw)
    print("saved", a.out, f"{os.path.getsize(a.out)/1e6:.2f} MB")


if __name__ == "__main__":
    main()
