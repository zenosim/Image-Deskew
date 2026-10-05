"""Measure chroma-key matte quality on the existing Grok outputs (green vs magenta)."""
import sys
sys.path.insert(0, "/root/workspace/Image-Deskew")
import numpy as np
from PIL import Image
from deskew_pipeline.chroma_key import chroma_key_matte

for name in ("green", "magenta"):
    img = Image.open(f"/tmp/grok_{name}_out.png").convert("RGB")
    arr = np.asarray(img).astype(np.float32) / 255.0
    border = np.concatenate([
        arr[:8].reshape(-1, 3), arr[-8:].reshape(-1, 3),
        arr[:, :8].reshape(-1, 3), arr[:, -8:].reshape(-1, 3)])
    bg = np.median(border, axis=0)
    print(f"--- {name} ---")
    print("bg color:", (bg * 255).round().astype(int),
          "border std:", border.std(axis=0).round(4),
          "max dev from median:", np.abs(border - bg).max().round(4))
    matte, _ = chroma_key_matte(img)
    a = np.asarray(matte)[:, :, 3] / 255.0
    print(f"alpha>0.5 frac: {(a > 0.5).mean():.3f}  "
          f"partial: {((a > 0.02) & (a < 0.98)).mean() * 100:.2f}%  "
          f"zero: {(a < 0.02).mean():.3f}")
    rgb = np.asarray(matte)[:, :, :3].astype(np.float32)
    pm = (a > 0.05) & (a < 0.95)
    if pm.sum() > 0:
        spill = (rgb[..., 1] - np.maximum(rgb[..., 0], rgb[..., 2]))[pm]
        print(f"partial px: {pm.sum()}, mean green excess on edges: {spill.mean():.1f} "
              f"(want <=0), frac still green: {(spill > 20).mean():.3f}")
    matte.save(f"/tmp/ck_{name}_rgba.png")
print("done")
