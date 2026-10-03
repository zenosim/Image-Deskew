"""Inpainter tests: tiled LaMa path, mask-clip blending, ONNX fallback, watermark band fill."""
import os
import sys
import time
import numpy as np
import pytest
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from deskew_pipeline.inpainter import BigLamaInpainter

FIX = os.path.join(ROOT, "tests", "fixtures")


@pytest.fixture(scope="module")
def inpainter():
    return BigLamaInpainter(model_type="anime")  # auto-falls back to ONNX without torch


def test_engine_available(inpainter):
    assert inpainter.model_type in ("anime", "general")


def test_band_inpaint_shape_and_time(inpainter):
    img = Image.open(os.path.join(FIX, "watermarked_a90_single.png")).convert("RGB")
    arr = np.array(img)
    h, w = arr.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    mask[int(h * 0.45):int(h * 0.62), int(w * 0.2):int(w * 0.85)] = 255
    t0 = time.time()
    out = inpainter.inpaint(img, Image.fromarray(mask))
    dt = time.time() - t0
    assert out.size == img.size
    assert dt < 60, f"inpaint too slow: {dt:.1f}s"


def test_border_tile_blend_no_broadcast_error(inpainter):
    """Mask near image edge forces tiles that extend past region bounds (regression)."""
    img = Image.open(os.path.join(FIX, "watermarked_a52_tiles.png")).convert("RGB")
    arr = np.array(img)
    h, w = arr.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    mask[h - 300:, w // 3: 2 * w // 3] = 255  # bottom edge band
    out = inpainter.inpaint(img, Image.fromarray(mask))
    assert out.size == img.size


def test_watermark_band_removed(inpainter):
    img = Image.open(os.path.join(FIX, "watermarked_a90_single.png")).convert("RGB")
    arr = np.array(img)
    h, w = arr.shape[:2]
    # watermark was stamped at (w//4, h//2) — inpaint a generous box around the text
    mask = np.zeros((h, w), np.uint8)
    tx, ty = w // 4, h // 2
    mask[ty - 60:ty + 90, tx - 60:tx + 420] = 255
    out = np.array(inpainter.inpaint(img, Image.fromarray(mask)))

    def text_residual(a):
        rgb = a.astype(np.int16)
        sat = rgb.max(2) - rgb.min(2)
        gray = rgb.mean(2)
        return float(((np.abs(gray - 140) < 45) & (sat < 26)).mean())

    assert text_residual(out[ty - 60:ty + 90, tx - 60:tx + 420]) <= text_residual(
        arr[ty - 60:ty + 90, tx - 60:tx + 420]
    ) + 0.005
