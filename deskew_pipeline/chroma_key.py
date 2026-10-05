"""Chroma-key matte extraction for synthetic flat backgrounds (Grok AI Edit output).

When Grok mode is enabled, the edit is requested with a pure chroma-key green
(#00FF00) background instead of white. Because that background is synthetic and
near-perfectly uniform, a distance-based chroma key beats a neural segmentor:

- per-pixel RGB distance to the sampled background color -> soft alpha matte
  (soft ramp preserves anti-aliased edges and wispy hair instead of melting them)
- largest-connected-component keep + enclosed-hole fill (no stray specks, no
  see-through towel)
- green despill on every pixel where green exceeds max(R,B) (removes the green
  edge line the model leaves around the silhouette)
- erfeated: isnet-anime on the same Grok output melts hair strands and leaves a
  green halo; chroma key measured visibly better (edge ~9/10 vs ~7/10)
"""

import numpy as np
import cv2
from PIL import Image

# Grok gets this background-color instruction instead of white when chroma mode is on
CHROMA_BG_INSTRUCTIONS = {
    "green": (
        "a completely plain, uniform, solid pure chroma-key GREEN (#00FF00) background",
        "Full-bleed solid #00FF00 green background",
    ),
    "white": (
        "a completely plain, uniform, solid pure white (#FFFFFF) background",
        "Full-bleed white background",
    ),
}
DEFAULT_CHROMA_BG = "green"



def chroma_key_matte(img, tolerance=0.30, softness=0.20, despill=True):
    """Extracts the character from a uniform-background image.

    Returns (RGBA PIL Image, sampled_bg_rgb_float01). Alpha is a soft ramp between
    tolerance*(1-softness) and tolerance*(1+softness) RGB distance to the border-
    sampled background color; despill removes background-color cast on the subject.
    """
    arr = np.asarray(img.convert("RGB")).astype(np.float32) / 255.0
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

    # Keep only the dominant foreground component; fill enclosed holes (towel, gaps
    # between arm and body must stay opaque).
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

    # Despill: kill background-color cast wherever it dominates the pixel.
    if despill:
        subject = alpha > 0.02
        r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
        if bg[1] >= bg[0] and bg[1] >= bg[2]:
            # green background: clamp green to max(r, b) on the subject
            over = np.clip(g - np.maximum(r, b), 0, None)
            arr[..., 1] = np.where(subject, g - over, g)
        elif bg[0] >= bg[1] and bg[2] >= bg[1]:
            # magenta background: pull r/b toward g where both exceed it
            over = np.clip(np.minimum(r, b) - g, 0, None) * 0.5
            arr[..., 0] = np.where(subject, r - over, r)
            arr[..., 2] = np.where(subject, b - over, b)

    rgba = np.dstack([
        (np.clip(arr, 0, 1) * 255).astype(np.uint8),
        (alpha * 255).astype(np.uint8),
    ])
    return Image.fromarray(rgba, "RGBA"), bg
