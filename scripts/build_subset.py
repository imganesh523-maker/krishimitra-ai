"""Build the balanced PlantVillage subset used to train KrishiMitra AI.

Source: Hugging Face dataset `BrandonFors/Plant-Diseases-PlantVillage-Dataset`
(PlantVillage, Hughes & Salathe / Mohanty et al. 2016; 38 classes, 80/20 train/test parquet files).

Per class (seed 42):
  * train : up to 200 images (from the HF *train* split)
  * val   : up to 40 images  (from the HF *train* split, disjoint from train)
  * test  : up to 60 images  (from the HF *test* split - never seen in training)
Images are resized to 176x176 JPEG (the model trains on random 160x160 crops).

Usage:
    python scripts/build_subset.py --out data/plantvillage_subset
"""
import argparse, collections, io, json, os, random, urllib.request

import pyarrow.parquet as pq
from PIL import Image

BASE = "https://huggingface.co/datasets/BrandonFors/Plant-Diseases-PlantVillage-Dataset/resolve/main/data/"
FILES = ["train-00000-of-00002", "train-00001-of-00002", "test-00000-of-00001"]
TR, VA, TE = 200, 40, 60


def download(cache):
    os.makedirs(cache, exist_ok=True)
    for f in FILES:
        p = f"{cache}/{f}.parquet"
        if not os.path.exists(p):
            print("downloading", f); urllib.request.urlretrieve(BASE + f + ".parquet", p)


def load(cache, files):
    rows = collections.defaultdict(list)
    for f in files:
        t = pq.read_table(f"{cache}/{f}.parquet")
        for lab, im in zip(t.column("label").to_pylist(), t.column("image").to_pylist()):
            rows[lab].append(im["bytes"])
    return rows


def save(b, path):
    Image.open(io.BytesIO(b)).convert("RGB").resize((176, 176), Image.BILINEAR).save(path, quality=92)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/raw")
    ap.add_argument("--out", default="data/plantvillage_subset")
    a = ap.parse_args()
    random.seed(42)
    download(a.cache)
    stats = {}
    tr = load(a.cache, FILES[:2])
    for lab, lst in tr.items():
        random.shuffle(lst)
        ntr = min(TR, int(len(lst) * 0.8)); nva = min(VA, len(lst) - ntr)
        for split, sub in [("train", lst[:ntr]), ("val", lst[ntr:ntr + nva])]:
            d = f"{a.out}/{split}/{lab:02d}"; os.makedirs(d, exist_ok=True)
            for i, b in enumerate(sub): save(b, f"{d}/{i}.jpg")
        stats[lab] = [ntr, nva]
    del tr
    te = load(a.cache, FILES[2:])
    for lab, lst in te.items():
        random.shuffle(lst); sub = lst[:TE]
        d = f"{a.out}/test/{lab:02d}"; os.makedirs(d, exist_ok=True)
        for i, b in enumerate(sub): save(b, f"{d}/{i}.jpg")
        stats[lab].append(len(sub))
    json.dump(stats, open(f"{a.out}/stats.json", "w"))
    print("train/val/test:", *[sum(v[i] for v in stats.values()) for i in range(3)])


if __name__ == "__main__":
    main()
