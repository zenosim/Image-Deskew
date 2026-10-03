"""Deskew regression tests: upright art must never be rotated; quad cards and peekers must."""
import os
import sys
import numpy as np
import pytest
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from deskew_pipeline import StickerPipeline

UPRIGHT = [
    "Test_stickes/diesentai.com_ada-kisscut_01_ASDSADASASD.png",
    "Test_stickes/cowf.ee_shadowheart-kiss-cut-1_03_shadowheart-kiss-cut-972301.jpg",
    "Test_stickes/cowf.ee_frieren-pout-kiss-cut_02_frieren-pout-kiss-cut-201601.jpg",
    "Test_stickes/nekodecal.com_roller-maid-stickers_12_3392.jpg",
    "Test_stickes/nekodecal.com_4th-wall-waifu-stickers-1_06_1941_3bc62463-b513-4809-b3f0-b4aba1644bc5.jpg",
]


def _applied_rot(path, **kw):
    p = StickerPipeline(segmentor_model="isnet-anime", extract_mode="character_only", deskew_mode="auto", **kw)
    r = p.process(path)
    h = r.metadata.get("homography_matrix")
    return float(np.degrees(np.arctan2(h[1][0], h[0][0]))) if h is not None else 0.0


@pytest.mark.parametrize("rel", UPRIGHT)
def test_upright_not_rotated(rel):
    rot = _applied_rot(os.path.join(ROOT, rel))
    assert abs(rot) < 1.0, f"upright art phantom-deskewed by {rot:.2f} deg: {rel}"


def test_rotated_card_deskewed():
    f = os.path.join(ROOT, "tests", "fixtures", "rotated_6.png")
    img = Image.open(f).convert("RGB")
    # rotate a WHITE CARD (quad) for the quad strategy: paste sticker? use synthetic card
    card = Image.new("RGB", (600, 400), (240, 240, 245))
    from PIL import ImageDraw
    d = ImageDraw.Draw(card)
    d.rectangle([50, 50, 550, 350], fill=(60, 80, 200))
    d.ellipse([260, 160, 340, 240], fill=(250, 220, 90))
    rot = card.rotate(6, expand=True, resample=Image.BICUBIC, fillcolor="white")
    p = StickerPipeline(segmentor_model="isnet-anime", extract_mode="full_sticker", deskew_mode="auto", target_aspect_ratio=1.5)
    r = p.process(rot)
    h = r.metadata.get("homography_matrix")
    applied = float(np.degrees(np.arctan2(h[1][0], h[0][0]))) if h is not None else 0.0
    assert abs(applied) >= 3.0, f"quad card tilt {6} deg not corrected (applied {applied:.2f})"
