"""Fix fidelity comparison: aspect-ratio-normalized IoU (rotation changes bbox aspect).

The output bbox after +6deg rotation is taller/narrower (1338x685) than the reference
(1321x804). Naive resize-to-ref-shape stretches it. Correct: rotate the OUTPUT back by
the known input angle before comparing, or compare aspect-normalized masks.
Here: since deskew didn't apply (mode=none), the output is still rotated by ~6deg.
Rotate the output mask by -6deg then compare IoU. This tests content preservation
independent of the (intentionally conservative) deskew behavior.
"""
import os
import sys
import numpy as np
import cv2
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from deskew_pipeline import StickerPipeline
from rembg import remove, new_session

ADA = os.path.join(ROOT, "Test_stickes", "diesentai.com_ada-kisscut_01_ASDSADASASD.png")
ANGLE = 6.0


def crop_alpha(arr):
    ys, xs = np.nonzero(arr[:, :, 3] > 128)
    return arr[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


orig = Image.open(ADA).convert("RGB")
orig.thumbnail((1400, 1400), Image.LANCZOS)
rot = orig.rotate(ANGLE, expand=True, resample=Image.BICUBIC, fillcolor="white")

s = new_session("isnet-anime")
ref = crop_alpha(np.array(remove(orig, session=s)))

p = StickerPipeline(segmentor_model="isnet-anime", extract_mode="character_only",
                    deskew_mode="auto", pca_align=False)
r = p.process(rot)
out_c = crop_alpha(np.array(r.final_rgba))

# de-rotate the output mask by the input angle, then crop tight
out_mask = (out_c[:, :, 3] > 128).astype(np.uint8) * 255
derot = Image.fromarray(out_mask).rotate(-ANGLE, expand=True, resample=Image.BICUBIC, fillcolor=0)
derot = np.array(derot)
ys, xs = np.nonzero(derot > 128)
derot_tight = derot[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

ref_mask = (ref[:, :, 3] > 128).astype(np.uint8) * 255
common = (min(ref_mask.shape[0], derot_tight.shape[0]),
          min(ref_mask.shape[1], derot_tight.shape[1]))
a = np.array(Image.fromarray(ref_mask).resize((common[1], common[0]), Image.LANCZOS)) > 128
b = np.array(Image.fromarray(derot_tight).resize((common[1], common[0]), Image.LANCZOS)) > 128
iou = np.count_nonzero(a & b) / max(1, np.count_nonzero(a | b))
print("de-rotated alpha IoU:", round(iou, 3))

# color check on common core
oc_res = np.array(Image.fromarray(out_c).resize((ref.shape[1], ref.shape[0]), Image.LANCZOS))
core = (ref[:, :, 3] > 200) & (oc_res[:, :, 3] > 200)
core = cv2.erode(core.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
mae = float(np.abs(oc_res[:, :, :3].astype(np.int16) - ref[:, :, :3].astype(np.int16)).sum(axis=2)[core].mean())
print("color MAE on core:", round(mae, 1))
