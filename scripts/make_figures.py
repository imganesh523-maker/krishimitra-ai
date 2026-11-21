"""Create training-curve and confusion-matrix figures from reports/ (real outputs of train.py)."""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.classes import CLASS_NAMES, pretty  # noqa: E402

os.makedirs("assets", exist_ok=True)
h = json.load(open("reports/history.json"))
ep = [r["epoch"] for r in h]
n_head = sum(r["phase"] == "head" for r in h)
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
for a, key, title in [(ax[0], "acc", "Accuracy"), (ax[1], "loss", "Loss (label-smoothed CE)")]:
    a.plot(ep, [r["train_" + key] for r in h], "o-", label="train", color="#15803d")
    a.plot(ep, [r["val_" + key] for r in h], "s-", label="validation", color="#ea580c")
    a.axvline(n_head + 0.5, ls="--", color="grey"); a.text(n_head + 0.6, a.get_ylim()[0] + 0.02 * (a.get_ylim()[1] - a.get_ylim()[0]), "fine-tuning starts", color="grey")
    a.set_title(title); a.set_xlabel("epoch"); a.grid(alpha=.3); a.legend()
plt.suptitle("KrishiMitra AI - MobileNetV2 training curves", fontweight="bold")
plt.tight_layout(); plt.savefig("assets/training_curves.png", dpi=130); plt.close()

cm = np.load("reports/confusion_matrix.npy")
cmn = cm / cm.sum(1, keepdims=True)
names = [pretty(c) for c in CLASS_NAMES]
fig, a = plt.subplots(figsize=(15, 13))
im = a.imshow(cmn, cmap="Greens", vmin=0, vmax=1)
a.set_xticks(range(len(names))); a.set_yticks(range(len(names)))
a.set_xticklabels(names, rotation=90, fontsize=7); a.set_yticklabels(names, fontsize=7)
for i in range(len(names)):
    for j in range(len(names)):
        if cm[i, j]:
            a.text(j, i, cm[i, j], ha="center", va="center", fontsize=5.5, color="white" if cmn[i, j] > .5 else "black")
a.set_xlabel("Predicted"); a.set_ylabel("True")
m = json.load(open("reports/metrics.json"))
a.set_title(f"Confusion matrix on held-out test set (n={m['n_test']}, accuracy={m['test_accuracy']*100:.2f}%)", fontweight="bold")
plt.colorbar(im, fraction=0.035); plt.tight_layout(); plt.savefig("assets/confusion_matrix.png", dpi=130); plt.close()
print("figures written")
