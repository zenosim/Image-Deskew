"""Diagnose fidelity IoU: save side-by-side overlay of ref vs output masks."""
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


def crop_alpha(arr):
    ys, xs = np.nonzero(arr[:, :, 3] > 128)
    return arr[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


orig = Image.open(ADA).convert("RGB")
orig.thumbnail((1400, 1400), Image.LANCZOS)
rot = orig.rotate(6, expand=True, resample=Image.BICUBIC, fillcolor="white")

s = new_session("isnet-anime")
ref = crop_alpha(np.array(remove(orig, session=s)))

p = StickerPipeline(segmentor_model="isnet-anime", extract_mode="character_only", deskew_mode="auto", pca_align=False)
r = p.process(rot)
out_c = crop_alpha(np.array(r.final_rgba))
print("ref shape:", ref.shape, "out shape:", out_c.shape)
out_res = np.array(Image.fromarray(out_c).resize((ref.shape[1], ref.shape[0]), Image.LANCZOS))

ra = ref[:, :, 3] > 128
oa = out_res[:, :, 3] > 128
vis = np.zeros((*ra.shape, 3), np.uint8)
vis[ra & ~oa] = [255, 0, 0]    # ref only (missing in output)
vis[oa & ~ra] = [0, 255, 0]    # output only (extra)
vis[ra & oa] = [255, 255, 255]  # both
Image.fromarray(vis).save("/tmp/fidelity_overlay.png")
iou = np.count_nonzero(ra & oa) / max(1, np.count_nonzero(ra | oa))
print("IoU:", round(iou, 3))
