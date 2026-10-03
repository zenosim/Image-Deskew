"""Visual fidelity tests: deskewed output must LOOK like the source image.

Known-angle rotations of a clean sticker give ground truth. The output alpha mask is
de-rotated by the input angle before comparison (deskew may legitimately not auto-apply
on pose-ambiguous art), then compared to the reference cutout:
- alpha IoU > 0.97  -> geometry preserved, nothing cropped/invented
- color MAE < 25    -> pixels identical to source (no channel/brightness shift)
- homography pure rotation -> no shear/scale distortion
"""
import os
import sys
import numpy as np
import pytest
import cv2
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from deskew_pipeline import StickerPipeline
from rembg import remove, new_session

ADA = os.path.join(ROOT, "Test_stickes", "diesentai.com_ada-kisscut_01_ASDSADASASD.png")


def _crop_alpha(arr):
    ys, xs = np.nonzero(arr[:, :, 3] > 128)
    return arr[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def _aligned_pair(angle):
    """Returns (reference RGBA, output RGBA) both tight-cropped and de-rotated, same size."""
    orig = Image.open(ADA).convert("RGB")
    orig.thumbnail((1400, 1400), Image.LANCZOS)
    rot = orig.rotate(angle, expand=True, resample=Image.BICUBIC, fillcolor="white")

    s = new_session("isnet-anime")
    ref = _crop_alpha(np.array(remove(orig, session=s)))

    p = StickerPipeline(
        segmentor_model="isnet-anime",
        extract_mode="character_only",
        deskew_mode="auto",
        pca_align=False,
    )
    r = p.process(rot)
    out = np.array(r.final_rgba)
    ys, xs = np.nonzero(out[:, :, 3] > 128)
    out = out[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

    # De-rotate by the input angle so both are upright (deskew may or may not have applied;
    # if it did, the output is already upright and de-rotation by the same angle is a no-op
    # only when applied == -angle. Handle both by measuring which of the two matches better.)
    candidates = []
    for extra in (0.0, -angle):
        der = Image.fromarray(out).rotate(extra, expand=True, resample=Image.BICUBIC,
                                          fillcolor=(0, 0, 0, 0))
        a = _crop_alpha(np.array(der))
        common = (min(ref.shape[0], a.shape[0]), min(ref.shape[1], a.shape[1]))
        ar = np.array(Image.fromarray(a).resize((common[1], common[0]), Image.LANCZOS))
        rr = np.array(Image.fromarray(ref).resize((common[1], common[0]), Image.LANCZOS))
        core = (rr[:, :, 3] > 200) & (ar[:, :, 3] > 200)
        core = cv2.erode(core.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
        iou = np.count_nonzero((rr[:, :, 3] > 128) & (ar[:, :, 3] > 128)) / max(
            1, np.count_nonzero((rr[:, :, 3] > 128) | (ar[:, :, 3] > 128))
        )
        candidates.append((iou, ar, rr, core))
    candidates.sort(key=lambda t: -t[0])
    return ref, candidates[0]


@pytest.mark.parametrize("angle", [4.0, 6.0])
def test_deskew_geometry_matches_source(angle):
    ref, (iou, ar, rr, core) = _aligned_pair(angle)
    assert iou > 0.97, f"geometry mismatch after deskew: alpha IoU {iou:.3f} (angle {angle})"


@pytest.mark.parametrize("angle", [4.0, 6.0])
def test_deskew_colors_match_source(angle):
    ref, (iou, ar, rr, core) = _aligned_pair(angle)
    assert core.any(), "no common character core"
    mae = float(np.abs(ar[:, :, :3].astype(np.int16) - rr[:, :, :3].astype(np.int16)).sum(axis=2)[core].mean())
    assert mae < 25, f"color drift vs source: MAE {mae:.1f} (angle {angle})"


def test_deskew_does_not_distort_aspect():
    orig = Image.open(ADA).convert("RGB")
    orig.thumbnail((1400, 1400), Image.LANCZOS)
    rot = orig.rotate(6, expand=True, resample=Image.BICUBIC, fillcolor="white")
    p = StickerPipeline(segmentor_model="isnet-anime", extract_mode="character_only", deskew_mode="auto")
    r = p.process(rot)
    h = r.metadata.get("homography_matrix")
    if h is None:
        pytest.skip("no homography recorded (deskew not applied)")
    m = np.array(h)[:2, :2]
    scale0 = float(np.linalg.norm(m[:, 0]))
    scale1 = float(np.linalg.norm(m[:, 1]))
    dot = float(np.dot(m[:, 0], m[:, 1]))
    assert abs(scale0 - scale1) < 0.05, f"non-uniform scale: {scale0:.3f} vs {scale1:.3f}"
    assert abs(dot) < 0.05, f"shear present: dot={dot:.3f}"
