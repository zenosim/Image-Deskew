"""Run the real remove_shine and dump its internal glare mask via monkeypatched warp."""
import sys
sys.path.insert(0, ".")
import numpy as np
import cv2
from PIL import Image
from deskew_pipeline.shine_remover import ShineRemover

captured = {}
orig_cc = cv2.connectedComponentsWithStats


def spy(img, connectivity=8):
    ok, labels, stats, centroids = orig_cc(img, connectivity)
    if img.dtype == np.uint8 and img.max() <= 1 and img.mean() < 0.5:
        captured.setdefault("bright", (labels, stats))
    return ok, labels, stats, centroids


cv2.connectedComponentsWithStats = spy
sr = ShineRemover()
g = Image.open("tests/fixtures/glare_55.png").convert("RGB")
res = sr.remove_shine(g, strength=75)
cv2.connectedComponentsWithStats = orig_cc
if "bright" in captured:
    labels, stats = captured["bright"]
    sizes = stats[:, 4]
    print("bright comps:", len(sizes) - 1, "largest:", int(sizes[1:].max()))
changed = int((np.abs(np.array(g).astype(int) - np.array(res.cleaned_image).astype(int)).sum(2) > 20).sum())
print("changed px:", changed)
res.cleaned_image.save("/tmp/shine_removed.png")
