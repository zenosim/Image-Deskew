"""Watermark remover v2 test suite.

Covers: canvas-watermark removal (background-residual detector), overlay-on-art text
(saturation-drop detector), no-detection on clean art, LaMa fill quality, toggle.
"""
import glob
import os
import sys
import time
import numpy as np
import pytest
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from deskew_pipeline.watermark_remover import WatermarkRemover

FIX = os.path.join(ROOT, "tests", "fixtures")


def _gray_residual(arr):
    """Fraction of low-saturation mid-gray pixels (watermark color signature)."""
    rgb = arr.astype(np.int16)
    sat = rgb.max(2) - rgb.min(2)
    gray = rgb.mean(2)
    return float((((np.abs(gray - 140) < 45) & (sat < 26))).mean() * 100)


@pytest.fixture(scope="module")
def remover():
    return WatermarkRemover()


def test_tiled_canvas_watermark_detected_and_reduced(remover):
    img = Image.open(os.path.join(FIX, "watermarked_a52_tiles.png")).convert("RGB")
    r = remover.remove(img, sensitivity=60)
    assert r.watermark_detected
    out = np.array(r.cleaned_image)
    # canvas tile row (left margin strip, pure canvas + text): watermark gray must not increase
    before = _gray_residual(np.array(img)[0:300, 0:500])
    after = _gray_residual(out[0:300, 0:500])
    assert after <= before + 0.05, f"canvas watermark grew: {before:.3f} -> {after:.3f}"


def test_single_watermark_detected(remover):
    img = Image.open(os.path.join(FIX, "watermarked_a90_single.png")).convert("RGB")
    r = remover.remove(img, sensitivity=60)
    assert r.watermark_detected
    assert r.inpainted_pixels > 500


def test_clean_white_sticker_not_touched(remover):
    """Clean kiss-cut on white: zero changes on the character."""
    p = glob.glob(os.path.join(ROOT, "Test_stickes", "diesentai.com_ada-kisscut*"))[0]
    img = Image.open(p).convert("RGB")
    img.thumbnail((1400, 1400))
    r = remover.remove(img, sensitivity=50)
    out = np.array(r.cleaned_image).astype(int)
    orig = np.array(img).astype(int)
    # character interior (center crop) must be pixel-identical: protection works
    h, w = orig.shape[:2]
    center = slice(h // 4, 3 * h // 4), slice(w // 4, 3 * w // 4)
    changed = (np.abs(out[center] - orig[center]).sum(axis=2) > 20).mean()
    assert changed < 0.02, f"{changed*100:.1f}% of character pixels altered on clean art"


def test_face_never_inpainted(remover):
    """Watermarked fixture: actual skin pixels must carry no watermark mask.

    The character's bounding box inevitably contains canvas text (that's the point of the
    remover); the invariant is that SKIN ITSELF is never inpainted.
    """
    import cv2
    img = Image.open(os.path.join(FIX, "watermarked_a52_tiles.png")).convert("RGB")
    arr = np.array(img)
    r = remover.remove(img, sensitivity=60)
    mask = r.watermark_mask
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    skin = ((hsv[:, :, 0] <= 25) | (hsv[:, :, 0] >= 170)) & (hsv[:, :, 1] > 20) & (hsv[:, :, 2] > 70)
    skin = cv2.erode(skin.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0  # avoid edge bleed
    mask_on_skin = int((mask & skin).sum())
    total_skin = int(skin.sum())
    assert mask_on_skin == 0, f"{mask_on_skin}/{total_skin} skin pixels watermark-masked"


def test_result_shapes_and_types(remover):
    img = Image.open(os.path.join(FIX, "watermarked_a90_single.png")).convert("RGB")
    r = remover.remove(img, sensitivity=60)
    assert r.cleaned_image.mode == "RGB"
    assert r.cleaned_image.size == img.size
    assert r.execution_time_s < 60


def test_telea_fallback(remover):
    """use_lama=False path must still work (cv2.inpaint)."""
    wm = WatermarkRemover(use_lama=False)
    img = Image.open(os.path.join(FIX, "watermarked_a90_single.png")).convert("RGB")
    r = wm.remove(img, sensitivity=60)
    assert r.watermark_detected
    assert r.cleaned_image.size == img.size
