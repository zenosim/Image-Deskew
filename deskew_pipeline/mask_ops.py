"""Shared vectorized helpers for connected-component label maps, colour features, and character hint masks."""

import math
from typing import Dict, Optional, Tuple

import cv2
import numpy as np


def label_lut(num_labels: int, touched_labels: np.ndarray) -> np.ndarray:
    """Boolean lookup table marking every label that appears in `touched_labels` (background label 0 excluded).

    `label_lut(n, labels[mask])[labels]` is a fast replacement for
    `np.isin(labels, list(set(labels[mask]) - {0}))`.
    """
    lut = np.bincount(touched_labels.ravel(), minlength=num_labels)[:num_labels] > 0
    lut[0] = False
    return lut


def components_min_area(labels: np.ndarray, stats: np.ndarray, min_area: int) -> np.ndarray:
    """Pixel mask of every foreground component whose area is >= min_area."""
    lut = stats[:, cv2.CC_STAT_AREA] >= min_area
    lut[0] = False
    return lut[labels]


def dilate_bool(mask: np.ndarray, ksize: int) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    return cv2.dilate(mask.astype(np.uint8), kernel) > 0


def color_distance_below(rgb: np.ndarray, reference: np.ndarray, threshold: float) -> np.ndarray:
    """Equivalent to `np.linalg.norm(rgb - reference, axis=2) < threshold` using float32 squared distances."""
    diff = rgb.astype(np.float32) - np.asarray(reference, dtype=np.float32)
    return np.einsum("ijk,ijk->ij", diff, diff) < np.float32(threshold * threshold)


class ColorFeatures:
    """Lazily computed per-pixel colour features shared by the segmentor and the character extractor."""

    def __init__(self, rgb: np.ndarray):
        self.rgb = rgb
        self._cache: Dict[str, np.ndarray] = {}

    def _cached(self, key, compute):
        if key not in self._cache:
            self._cache[key] = compute()
        return self._cache[key]

    @property
    def hsv(self) -> np.ndarray:
        return self._cached("hsv", lambda: cv2.cvtColor(self.rgb, cv2.COLOR_RGB2HSV))

    @property
    def gray(self) -> np.ndarray:
        return self._cached("gray", lambda: cv2.cvtColor(self.rgb, cv2.COLOR_RGB2GRAY))

    @property
    def rgb16(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._cached("rgb16", lambda: tuple(self.rgb[:, :, c].astype(np.int16) for c in range(3)))

    @property
    def grad_mag(self) -> np.ndarray:
        def compute():
            grad_x = cv2.Sobel(self.gray, cv2.CV_16S, 1, 0, ksize=3)
            grad_y = cv2.Sobel(self.gray, cv2.CV_16S, 0, 1, ksize=3)
            return cv2.addWeighted(cv2.convertScaleAbs(grad_x), 0.5, cv2.convertScaleAbs(grad_y), 0.5, 0)
        return self._cached("grad_mag", compute)

    @property
    def is_chroma(self) -> np.ndarray:
        return self._cached("is_chroma", lambda: (self.hsv[:, :, 1] > 25) & (self.hsv[:, :, 2] > 40))

    @property
    def is_skin(self) -> np.ndarray:
        def compute():
            r, g, b = self.rgb16
            hue, sat, val = self.hsv[:, :, 0], self.hsv[:, :, 1], self.hsv[:, :, 2]
            return (r > b + 8) & (r >= g) & ((hue <= 22) | (hue >= 175)) & (sat > 15) & (val > 70)
        return self._cached("is_skin", compute)

    @property
    def is_body_skin(self) -> np.ndarray:
        """Skin regions of at least 40px after removing 1-2px highlight noise."""
        def compute():
            k_3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            is_real_skin = cv2.morphologyEx(self.is_skin.astype(np.uint8), cv2.MORPH_OPEN, k_3) > 0
            _, sk_lbl, sk_stats, _ = cv2.connectedComponentsWithStats(is_real_skin.astype(np.uint8), connectivity=8)
            return components_min_area(sk_lbl, sk_stats, 40)
        return self._cached("is_body_skin", compute)


def prepare_hint_mask(hint: Optional[np.ndarray], shape: Tuple[int, int]) -> Optional[np.ndarray]:
    """Normalizes a user-painted character highlight into a boolean mask of `shape`; None when empty."""
    if hint is None:
        return None
    hint = np.asarray(hint)
    if hint.ndim == 3:
        hint = hint.max(axis=2)
    if hint.shape != shape:
        hint = cv2.resize(hint.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    hint = hint > 0
    return hint if hint.any() else None


def apply_character_hint(mask: np.ndarray, hint: np.ndarray) -> np.ndarray:
    """Keeps only the foreground components the user highlighted, trimmed to a generous reach around the strokes.

    Brush strokes are rough, so parts of a highlighted component that stick out slightly past the
    strokes (hair tips, fingers) are kept, while background objects fused to the character are cut.
    Also ensures any foreground pixel directly under the user's positive hint strokes is solidly included.
    """
    if hint is None or not np.any(hint):
        return mask
    hint_bool = hint > 0 if hint.dtype != bool else hint
    num, labels = cv2.connectedComponents((mask > 0).astype(np.uint8), connectivity=8)
    keep = label_lut(num, labels[hint_bool])[labels]
    _, _, bw, bh = cv2.boundingRect(hint_bool.astype(np.uint8))
    reach = max(45.0, 0.15 * math.hypot(bw, bh))
    dist_from_hint = cv2.distanceTransform((~hint_bool).astype(np.uint8), cv2.DIST_L2, 5)
    keep &= (dist_from_hint <= reach)
    keep |= hint_bool & (mask > 0)
    out = mask.copy()
    out[~keep] = 0
    # Drop weakly-attached components: background patches whose overlap with the user's hint
    # is tiny relative to their size are residue stuck to the crop/hint boundary (measured:
    # hint-rect edges kept full background slivers as opaque px).
    if np.any(out):
        num2, lab2, st2, _ = cv2.connectedComponentsWithStats((out > 0).astype(np.uint8), connectivity=8)
        for i in range(1, num2):
            comp = lab2 == i
            overlap = int(np.count_nonzero(comp & hint_bool))
            comp_area = int(st2[i, cv2.CC_STAT_AREA])
            if overlap < 0.20 * comp_area and comp_area > 50:
                out[comp] = 0
    return out


def apply_negative_mask(mask: np.ndarray, negative_mask: Optional[np.ndarray]) -> np.ndarray:
    """Strictly eliminates any pixels covered by the negative mask.

    Dilates the negative mask slightly (2-3px) so edge halos and bleeding of unwanted elements are completely pruned.
    """
    if negative_mask is None:
        return mask
    neg = np.asarray(negative_mask)
    if neg.ndim == 3:
        neg = neg.max(axis=2)
    if neg.shape != mask.shape:
        neg = cv2.resize(neg.astype(np.uint8), (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST)
    neg_bool = neg > 0
    if not neg_bool.any():
        return mask

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    neg_dilated = cv2.dilate(neg_bool.astype(np.uint8), k) > 0
    out = mask.copy()
    out[neg_dilated] = 0
    return out


def detect_cast_shadow(
    rgb: np.ndarray,
    mask: np.ndarray,
    feats: ColorFeatures,
    bg_rgb: np.ndarray,
    min_area: int = 60,
    soft_touch_frac: float = 0.15,
    reach_px: int = 3,
    confidence: Optional[np.ndarray] = None,
    max_confidence: float = 170.0,
) -> np.ndarray:
    """Finds cast drop shadows the cutout kept, by what a shadow physically is rather than by connectivity.

    A shadow pixel is the canvas colour darkened (same chromaticity, lower brightness), un-inked and not skin
    or colour, and the shadow meets the exterior canvas directly, with no ink line in between (sticker mockup
    shadows are often hard-edged offset silhouettes, so edge softness is not required). Dark clothing, stockings
    and boots only meet the canvas across their ink outline, which blocks the reach test, so they are kept.
    Returns a boolean mask of shadow pixels (inside `mask`) to remove.
    """
    h, w = mask.shape
    empty = np.zeros((h, w), dtype=bool)
    fg = mask > 40
    bg = np.asarray(bg_rgb, dtype=np.float32)
    bg_norm = float(np.linalg.norm(bg))
    if not fg.any() or bg_norm < 150.0:
        return empty

    px = rgb.astype(np.float32)
    p_norm = np.sqrt(np.einsum("ijk,ijk->ij", px, px)) + 1e-3
    cos = np.einsum("ijk,k->ij", px, bg) / (p_norm * bg_norm)
    ratio = p_norm / bg_norm
    gray = feats.gray
    # Only true black ink blocks: offset mockup shadows are often dark grey (60-100) right next to the outline
    ink = gray < 45
    # Colour is judged by absolute channel spread: HSV saturation calls slightly tinted dark greys "colourful"
    # (e.g. (69,75,77) has S=26), which fragmented shadows into pieces that were never removed
    channel_spread = rgb.max(axis=2).astype(np.int16) - rgb.min(axis=2).astype(np.int16)
    art = (channel_spread > 22) | feats.is_skin

    darkened = fg & (cos > 0.985) & ~art & ~ink
    cand = darkened & (ratio > 0.15) & (ratio < 0.93)
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cand = cv2.morphologyEx(cand.astype(np.uint8), cv2.MORPH_OPEN, k3) > 0
    if not cand.any():
        return empty

    # Exterior canvas: pixels the model calls background (or canvas-coloured and uncertain), connected to the
    # image border. Confident foreground (e.g. white capes) never counts as canvas, even if it is white.
    canvas = (mask < 128) | (color_distance_below(rgb, bg, 18.0) & (mask < 200))
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    n_c, c_lbl = cv2.connectedComponents(canvas.astype(np.uint8), connectivity=8)
    exterior = label_lut(n_c, c_lbl[canvas & border])[c_lbl]
    if not exterior.any():
        return empty

    # Geodesic reach from the exterior that cannot jump across ink outlines
    passable = (~ink).astype(np.uint8)
    reach = exterior.astype(np.uint8)
    for _ in range(reach_px):
        reach = cv2.dilate(reach, k3) & passable
    reach = reach > 0

    n, lbl, stats, _ = cv2.connectedComponentsWithStats(cand.astype(np.uint8), connectivity=8)
    boundary = cand & ~(cv2.erode(cand.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)
    b_total = np.bincount(lbl[boundary], minlength=n)
    b_exterior = np.bincount(lbl[boundary & reach], minlength=n)
    area = stats[:, cv2.CC_STAT_AREA]
    touches_exterior = b_exterior >= soft_touch_frac * np.maximum(b_total, 1)
    keep = (area >= min_area) & touches_exterior
    if confidence is not None:
        # Kept shadows are where the segmentation model itself was unsure; light-grey outline shading on white
        # fabric or grey stockings is confident foreground and must never be removed
        conf_mean = np.bincount(lbl.ravel(), weights=confidence.ravel().astype(np.float64), minlength=n) / np.maximum(area, 1)
        # Small shadow slivers are also removed when the model was clearly unsure about them
        keep = ((area >= min_area) | ((area >= 12) & (conf_mean < 130.0))) & touches_exterior & (conf_mean < max_confidence)
    keep[0] = False
    shadow = keep[lbl]
    if not shadow.any():
        return empty

    # Absorb the lighter falloff around the shadow body without crossing ink or artwork
    fade = ((darkened & (ratio < 0.995)) | shadow) & dilate_bool(shadow, 13)
    n_f, f_lbl = cv2.connectedComponents(fade.astype(np.uint8), connectivity=8)
    return label_lut(n_f, f_lbl[shadow])[f_lbl]

