"""Smoke test: the model loads, predicts every bundled sample correctly-shaped, and Grad-CAM returns a valid heatmap."""
import glob, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PIL import Image
from src.classes import CLASS_NAMES, pretty
from src.predictor import Predictor


def main():
    P = Predictor()
    samples = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "..", "assets", "samples", "*.jpg")))
    assert samples, "no sample images found"
    for s in samples:
        img = Image.open(s)
        p = P.predict(img)
        overlay, cam, _ = P.gradcam(img)
        assert p.label in CLASS_NAMES and 0 <= p.confidence <= 1 and len(p.top3) == 3
        assert cam.shape == (160, 160) and 0 <= cam.min() and cam.max() <= 1.0001
        info = P.info(p.label)
        assert info["treatment_hi"] and info["treatment_en"]
        print(f"{os.path.basename(s):40s} -> {pretty(p.label):45s} {p.confidence*100:5.1f}%")
    print("OK:", len(samples), "samples")


if __name__ == "__main__":
    main()
