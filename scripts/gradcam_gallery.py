"""Grad-CAM gallery: original leaf vs. heatmap for a few held-out test images (assets/gradcam_examples.png)."""
import glob, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.classes import pretty  # noqa: E402
from src.predictor import Predictor  # noqa: E402

samples = sorted(glob.glob("assets/samples/*.jpg"))[:6]
P = Predictor()
fig, ax = plt.subplots(2, len(samples), figsize=(3 * len(samples), 7.4))
for i, s in enumerate(samples):
    img = Image.open(s).convert("RGB")
    ov, _, _ = P.gradcam(img)
    p = P.predict(img)
    true = os.path.basename(s).rsplit(".", 1)[0].split("_", 1)[1].replace("_", " ")
    ax[0, i].imshow(img); ax[0, i].set_title(f"True: {true}", fontsize=8.5); ax[0, i].axis("off")
    ax[1, i].imshow(ov); ax[1, i].set_title(f"Pred: {pretty(p.label)}\n{p.confidence*100:.1f}%", fontsize=8.5, color="#15803d"); ax[1, i].axis("off")
plt.suptitle("KrishiMitra AI - Grad-CAM on held-out test leaves (top: input, bottom: where the model looked)", fontweight="bold")
plt.tight_layout(); plt.subplots_adjust(hspace=0.28); plt.savefig("assets/gradcam_examples.png", dpi=120); plt.close()
print("ok")
