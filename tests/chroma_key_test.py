"""Chroma-key matte extraction from a synthetic flat-background Grok output."""
import sys
sys.path.insert(0, ".")
import numpy as np
import cv2
from PIL import Image


def chroma_key_matte(img_rgb, tolerance=0.28, softness=0.18, despill=True):
    arr = np.asarray(img_rgb).astype(np.float32) / 255.0
    h, w = arr.shape[:2]
    border = np.concatenate([
        arr[:8].reshape(-1, 3), arr[-8:].reshape(-1, 3),
        arr[:, :8].reshape(-1, 3), arr[:, -8:].reshape(-1, 3),
    ])
    bg = np.median(border, axis=0)
    dist = np.linalg.norm(arr - bg[None, None, :], axis=-1)
    t0 = tolerance * (1 - softness)
    t1 = tolerance * (1 + softness)
    alpha = np.clip((dist - t0) / max(1e-6, (t1 - t0)), 0, 1)
    hard = (alpha > 0.5).astype(np.uint8)
    num, lbl, st, _ = cv2.connectedComponentsWithStats(hard, connectivity=8)
    if num > 1:
        largest = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
        keep = lbl == largest
        alpha = alpha * keep
        inv = (~keep).astype(np.uint8)
        ff = inv.copy()
        mask2 = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(ff, mask2, (0, 0), 1)
        holes = (inv == 1) & (ff == 0)
        alpha[holes] = 1.0
    if despill:
        if bg[1] >= bg[0] and bg[1] >= bg[2]:
            pass  # green: handled below via min clamp on partial alpha
        else:
            over = np.minimum(arr[..., 0], arr[..., 2]) - arr[..., 1]
            fix = np.clip(over, 0, None) * 0.5
            arr[..., 0] = np.where(alpha < 1, arr[..., 0] - fix, arr[..., 0])
            arr[..., 2] = np.where(alpha < 1, arr[..., 2] - fix, arr[..., 2])
    rgba = np.dstack([(np.clip(arr, 0, 1) * 255).astype(np.uint8), (alpha * 255).astype(np.uint8)])
    return Image.fromarray(rgba, "RGBA"), bg


def comp_over(img_rgba, bg_rgb=(240, 240, 245), out=None):
    comp = Image.new("RGB", img_rgba.size, bg_rgb)
    comp.paste(img_rgba, (0, 0), img_rgba)
    if out:
        comp.save(out)
    return comp


if __name__ == "__main__":
    for name in ("green", "magenta"):
        img = Image.open(f"/tmp/grok_{name}_out.png").convert("RGB")
        matte, bg = chroma_key_matte(img)
        print(name, "bg sampled:", (bg * 255).round().astype(int))
        matte.save(f"/tmp/chroma_{name}_rgba.png")
        comp_over(matte, out=f"/tmp/chroma_{name}_cut.png")
    print("saved chroma cuts")
