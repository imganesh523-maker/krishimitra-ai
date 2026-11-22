"""Dark-theme versions of the training curves and confusion matrix for the web demo (docs/img/).
Same data as scripts/make_figures.py (reports/history.json, reports/confusion_matrix.npy)."""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.classes import CLASS_NAMES, pretty  # noqa: E402

OUT = os.path.join(ROOT, "docs", "img"); os.makedirs(OUT, exist_ok=True)
FG, MUTED, GRID = "#e6f4ec", "#8aa597", "#ffffff18"
EM, AMB = "#34d399", "#fbbf24"
plt.rcParams.update({"text.color": FG, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.edgecolor": "#ffffff22", "font.family": "DejaVu Sans", "font.size": 10})

h = json.load(open(os.path.join(ROOT, "reports", "history.json")))
ep = [r["epoch"] for r in h]; n_head = sum(r["phase"] == "head" for r in h)
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
for a, key, title in [(ax[0], "acc", "Accuracy"), (ax[1], "loss", "Loss (label-smoothed CE)")]:
    a.set_facecolor("none")
    a.plot(ep, [r["train_" + key] for r in h], "o-", label="train", color=EM, lw=2.2, ms=5)
    a.plot(ep, [r["val_" + key] for r in h], "s-", label="validation", color=AMB, lw=2.2, ms=5)
    a.axvline(n_head + 0.5, ls="--", color=MUTED, lw=1)
    lo, hi = a.get_ylim(); a.text(n_head + 0.65, lo + 0.04 * (hi - lo), "fine-tuning starts", color=MUTED, fontsize=9)
    a.set_title(title, color=FG, fontsize=12, pad=10); a.set_xlabel("epoch"); a.grid(color=GRID)
    for s in ("top", "right"): a.spines[s].set_visible(False)
    a.legend(frameon=False, labelcolor=FG)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "training_curves.png"), dpi=140, transparent=True); plt.close()

cm = np.load(os.path.join(ROOT, "reports", "confusion_matrix.npy")); cmn = cm / cm.sum(1, keepdims=True)
names = [pretty(c) for c in CLASS_NAMES]
cmap = LinearSegmentedColormap.from_list("em", ["#0b1410", "#064e3b", "#059669", "#6ee7b7"])
fig, a = plt.subplots(figsize=(13, 11.5))
im = a.imshow(cmn, cmap=cmap, vmin=0, vmax=1)
a.set_xticks(range(len(names))); a.set_yticks(range(len(names)))
a.set_xticklabels(names, rotation=90, fontsize=7); a.set_yticklabels(names, fontsize=7)
for i in range(len(names)):
    for j in range(len(names)):
        if cm[i, j]:
            a.text(j, i, cm[i, j], ha="center", va="center", fontsize=5.5, color="#04120b" if cmn[i, j] > .6 else FG)
a.set_xlabel("Predicted"); a.set_ylabel("True")
cb = plt.colorbar(im, fraction=0.035); cb.outline.set_visible(False)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "confusion_matrix.png"), dpi=130, transparent=True); plt.close()
print("web figures written to", OUT)
