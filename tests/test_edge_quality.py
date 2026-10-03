"""Edge-quality regression tests on complex-background cutouts.

Wolf-girl (nekodecal 4th-wall, pink patterned bg) and maid (roller-maid, same bg style):
- no green background hue leaking into the cutout (objective hue histogram)
- alpha band thin (<=2% of opaque area) after postprocessing
"""
import glob
import os
import sys
import numpy as np
import pytest
import cv2
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from deskew_pipeline import StickerPipeline


def _hue_hist_pct(rgba, sat_min=40):
    hsv = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2HSV)
    m = rgba[:, :, 3] > 200
    hh, ss = hsv[:, :, 0][m], hsv[:, :, 1][m]
    sig = ss > sat_min
    if sig.sum() == 0:
        return np.zeros(9)
    hist = np.histogram(hh[sig], bins=9, range=(0, 180))[0]
    return hist / hist.sum() * 100


@pytest.mark.parametrize("pattern,label", [
    ("nekodecal.com_4th-wall*", "wolf-girl"),
    ("nekodecal.com_roller-maid*", "maid"),
])
def test_no_green_background_leak(pattern, label):
    path = glob.glob(os.path.join(ROOT, "Test_stickes", pattern))[0]
    p = StickerPipeline(segmentor_model="isnet-anime", extract_mode="character_only",
                        deskew_mode="none")
    r = p.process(path)
    arr = np.array(r.final_rgba)
    hist = _hue_hist_pct(arr)
    # green (hue 80-120 = bins 4,5) must be a trace only; art is orange/pink/white
    green = float(hist[4] + hist[5])
    assert green < 1.0, f"{label}: green background leak {green:.1f}% of saturated pixels"


@pytest.mark.parametrize("pattern,label", [
    ("nekodecal.com_4th-wall*", "wolf-girl"),
    ("nekodecal.com_roller-maid*", "maid"),
])
def test_alpha_band_thin(pattern, label):
    path = glob.glob(os.path.join(ROOT, "Test_stickes", pattern))[0]
    p = StickerPipeline(segmentor_model="isnet-anime", extract_mode="character_only",
                        deskew_mode="none")
    r = p.process(path)
    a = np.array(r.final_rgba)[:, :, 3]
    opaque = (a >= 245).sum()
    semi = ((a > 10) & (a < 245)).sum()
    ratio = semi / max(1, opaque)
    assert ratio < 0.05, f"{label}: soft band {ratio * 100:.1f}% of opaque (want <5%)"


def test_shadowed_fixture_clean():
    """Shadowed fixture: no gray semi-transparent shadow blob survives."""
    f = os.path.join(ROOT, "tests", "fixtures", "shadowed_4.png")
    p = StickerPipeline(segmentor_model="isnet-anime", extract_mode="character_only",
                        deskew_mode="none")
    r = p.process(f)
    arr = np.array(r.final_rgba)
    a = arr[:, :, 3]
    gray = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2GRAY)
    # shadow = semi-transparent AND neutral gray (low sat, mid value)
    semi = (a > 10) & (a < 250)
    shadow_like = semi & (cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2HSV)[:, :, 1] < 30) & (gray > 60) & (gray < 220)
    assert shadow_like.sum() < 500, f"{int(shadow_like.sum())} gray semi-transparent shadow px survived"
