"""Chroma-key matte extraction for synthetic flat backgrounds (Grok AI Edit output).

When Grok mode is enabled, the edit is requested with a pure chroma-key green
(#00FF00) background instead of white. Because that background is synthetic and
near-perfectly uniform, a distance-based chroma key beats a neural segmentor:

- per-pixel RGB distance to the sampled background color -> soft alpha matte
  (soft ramp preserves anti-aliased edges and wispy hair instead of melting them)
- top-3 connected-component keep (detached hair strands / floating props survive;
  specks under 0.01% of canvas dropped) + enclosed-hole fill (no see-through towel)
- despill clamps the background channel wherever it dominates: green g>max(r,b),
  magenta min(r,b)>g at FULL strength (Grok bakes the cast into opaque art pixels
  on the silhouette — half-strength leaves a visible pink line; measured)
- defringe: semi-transparent edge pixels are un-premultiplied against the sampled
  background (F = (C - (1-a)B) / a) so no background tint stays baked into RGB
- green beats magenta on warm-palette anime art: magenta's red cast is
  indistinguishable from the art's own red lineart once baked in (measured ~7/10
  vs ~9/10 edges). isnet-anime on the same Grok output leaves a green halo and
  5x more partial-alpha edge pixels; chroma key is visibly crisper.
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



def chroma_key_matte(img, tolerance=0.30, softness=0.20, despill=True,
                     keep_components=3, defringe=True):
    """Extracts the character from a uniform-background image.

    Returns (RGBA PIL Image, sampled_bg_rgb_float01). Alpha is a soft ramp between
    tolerance*(1-softness) and tolerance*(1+softness) RGB distance to the border-
    sampled background color; despill removes background-color cast on the subject.
    keep_components: how many largest connected foreground blobs to keep (catches
    detached hair strands / floating props; 1 = old behaviour).
    defringe: un-premultiply semi-transparent edge pixels against the sampled
    background color so no background tint remains baked into their RGB.
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

    # Keep the dominant foreground component plus up to keep_components-1 smaller
    # ones (detached hair strands, floating props); drop tiny specks. Then fill
    # enclosed holes (towel, gaps between arm and body must stay opaque).
    hard = (alpha > 0.5).astype(np.uint8)
    num, lbl, st, _ = cv2.connectedComponentsWithStats(hard, connectivity=8)
    if num > 1:
        areas = st[1:, cv2.CC_STAT_AREA]
        order = np.argsort(areas)[::-1]
        min_area = max(30, int(0.0001 * h * w))
        keep = np.zeros((h, w), dtype=bool)
        for idx in order[:max(1, keep_components)]:
            if areas[idx] < min_area:
                break
            keep |= lbl == (idx + 1)
        alpha = alpha * keep
        inv = (~keep).astype(np.uint8)
        ff = inv.copy()
        mask2 = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(ff, mask2, (0, 0), 1)
        holes = (inv == 1) & (ff == 0)
        alpha[holes] = 1.0

    # Despill: kill background-color cast wherever it dominates the pixel
    # (fully-opaque interior pixels included — defringe below only sees edges).
    if despill:
        subject = alpha > 0.02
        r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
        if bg[1] >= bg[0] and bg[1] >= bg[2]:
            # green background: clamp green to max(r, b) on the subject
            over = np.clip(g - np.maximum(r, b), 0, None)
            arr[..., 1] = np.where(subject, g - over, g)
        elif min(bg) < 0.5:
            if bg[0] >= bg[1] and bg[0] >= bg[2] and bg[2] > 0.5 * bg[0]:
                # magenta family: both r and b elevated -> clamp them to g.
                # Full strength: Grok bakes the cast into opaque art pixels on
                # the silhouette, half-strength leaves a visible pink line.
                over = np.clip(np.minimum(arr[..., 0], arr[..., 2]) - arr[..., 1], 0, None)
                arr[..., 0] = np.where(subject, arr[..., 0] - over, arr[..., 0])
                arr[..., 2] = np.where(subject, arr[..., 2] - over, arr[..., 2])
            elif bg[0] >= bg[1] and bg[0] >= bg[2]:
                # pure red: clamp r to max(g, b)
                over = np.clip(arr[..., 0] - np.maximum(arr[..., 1], arr[..., 2]), 0, None)
                arr[..., 0] = np.where(subject, arr[..., 0] - over, arr[..., 0])
            elif bg[2] >= bg[1] and bg[2] >= bg[0]:
                # blue-dominant: clamp b to max(r, g)
                over = np.clip(arr[..., 2] - np.maximum(arr[..., 0], arr[..., 1]), 0, None)
                arr[..., 2] = np.where(subject, arr[..., 2] - over, arr[..., 2])

    # Defringe: un-premultiply semi-transparent edge pixels. A pixel on the
    # silhouette is C = a*F + (1-a)*B; solve for the true foreground F so no
    # background color stays baked into the RGB when composited elsewhere.
    if defringe:
        partial = (alpha > 0.02) & (alpha < 0.98)
        if partial.any():
            a_safe = np.clip(alpha, 0.05, 1.0)[..., None]
            f_est = (arr - (1.0 - a_safe) * bg[None, None, :]) / a_safe
            arr = np.where(partial[..., None], np.clip(f_est, 0.0, 1.0), arr)

    rgba = np.dstack([
        (np.clip(arr, 0, 1) * 255).astype(np.uint8),
        (alpha * 255).astype(np.uint8),
    ])
    return Image.fromarray(rgba, "RGBA"), bg
