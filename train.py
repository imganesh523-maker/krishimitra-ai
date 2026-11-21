"""Train KrishiMitra AI: MobileNetV2 transfer learning on a balanced PlantVillage subset (CPU-friendly).

Data layout expected (built by scripts/build_subset.py):
    DATA_DIR/{train,val,test}/<class_id 00..37>/*.jpg

Two phases:
  1. Feature extraction - backbone frozen, only the new classifier head is trained.
  2. Fine-tuning      - last MobileNetV2 blocks unfrozen and trained with a small learning rate.
Outputs: models/krishimitra_mobilenetv2.pt, reports/*.json|txt, assets/*.png
"""
import argparse, json, os, time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets

from src.classes import CLASS_NAMES, pretty
from src.model import build_model, train_transform, eval_transform

torch.manual_seed(42); np.random.seed(42)


def run_epoch(model, loader, criterion, optimizer=None):
    train = optimizer is not None
    model.train(train)
    total, correct, loss_sum = 0, 0, 0.0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            out = model(x)
            loss = criterion(out, y)
            if train:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            loss_sum += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += x.size(0)
    return loss_sum / total, correct / total


@torch.no_grad()
def predict_all(model, loader):
    model.eval(); ys, ps, probs = [], [], []
    for x, y in loader:
        p = torch.softmax(model(x), 1)
        ys.append(y.numpy()); ps.append(p.argmax(1).numpy()); probs.append(p.numpy())
    return np.concatenate(ys), np.concatenate(ps), np.concatenate(probs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/plantvillage_subset")
    ap.add_argument("--head-epochs", type=int, default=4)
    ap.add_argument("--ft-epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    os.makedirs("models", exist_ok=True); os.makedirs("reports", exist_ok=True)

    tr = datasets.ImageFolder(f"{args.data}/train", train_transform())
    va = datasets.ImageFolder(f"{args.data}/val", eval_transform())
    te = datasets.ImageFolder(f"{args.data}/test", eval_transform())
    assert len(tr.classes) == len(CLASS_NAMES)
    print(f"train={len(tr)} val={len(va)} test={len(te)} classes={len(tr.classes)}", flush=True)
    dl = lambda ds, sh: DataLoader(ds, batch_size=args.batch, shuffle=sh, num_workers=args.workers)
    trl, val, tel = dl(tr, True), dl(va, False), dl(te, False)

    model = build_model(len(CLASS_NAMES))
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    history, best = [], 0.0
    ckpt = "models/krishimitra_mobilenetv2.pt"

    def fit(epochs, optimizer, scheduler, phase):
        nonlocal best
        for ep in range(epochs):
            t = time.time()
            trl_loss, tr_acc = run_epoch(model, trl, criterion, optimizer)
            va_loss, va_acc = run_epoch(model, val, criterion)
            scheduler.step()
            history.append(dict(phase=phase, epoch=len(history) + 1, train_loss=trl_loss, train_acc=tr_acc,
                                val_loss=va_loss, val_acc=va_acc, secs=round(time.time() - t)))
            print(history[-1], flush=True)
            if va_acc > best:
                best = va_acc; torch.save(model.state_dict(), ckpt)
            json.dump(history, open("reports/history.json", "w"), indent=2)

    # Phase 1: frozen backbone
    for p in model.features.parameters(): p.requires_grad = False
    opt = torch.optim.AdamW(model.classifier.parameters(), lr=1e-3, weight_decay=1e-4)
    fit(args.head_epochs, opt, torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.head_epochs), "head")

    # Phase 2: unfreeze the last blocks of MobileNetV2 (features[13:])
    for p in model.features[13:].parameters(): p.requires_grad = True
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=3e-4, weight_decay=1e-4)
    fit(args.ft_epochs, opt, torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.ft_epochs), "finetune")

    # Evaluate the best checkpoint on the held-out test split
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    y, p, prob = predict_all(model, tel)
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, top_k_accuracy_score
    names = [pretty(c) for c in CLASS_NAMES]
    metrics = dict(
        test_accuracy=float(accuracy_score(y, p)),
        test_top3_accuracy=float(top_k_accuracy_score(y, prob, k=3, labels=list(range(len(names))))),
        macro_f1=float(f1_score(y, p, average="macro")),
        weighted_f1=float(f1_score(y, p, average="weighted")),
        best_val_accuracy=float(best),
        n_train=len(tr), n_val=len(va), n_test=len(te), n_classes=len(names),
        image_size=160, backbone="MobileNetV2 (ImageNet)",
        model_size_mb=round(os.path.getsize(ckpt) / 1e6, 2),
    )
    json.dump(metrics, open("reports/metrics.json", "w"), indent=2)
    rep = classification_report(y, p, target_names=names, digits=3)
    open("reports/classification_report.txt", "w").write(rep)
    np.save("reports/confusion_matrix.npy", confusion_matrix(y, p))
    print(json.dumps(metrics, indent=2)); print(rep)


if __name__ == "__main__":
    main()
