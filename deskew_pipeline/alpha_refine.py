"""Edge and colour finishing helpers shared by the pipeline stages.

- guided_filter / guided_filter_color: He et al. edge-preserving filters built on box filters.
- refine_alpha_edges: snaps a soft matte to colour edges inside a narrow band around the outline.
- signed_distance / resize_alpha_sdf / smooth_alpha_sdf: scale-independent, stair-step-free alpha edges.
- fill_transparent_rgb: extends edge colours under transparent pixels so resamplers never mix in black.
- denoise_chroma: removes compression chroma noise while leaving luma detail untouched.
"""

from typing import Optional

import cv2
import numpy as np


def _box(img: np.ndarray, radius: int) -> np.ndarray:
    k = 2 * int(radius) + 1
    return cv2.boxFilter(img, cv2.CV_32F, (k, k), normalize=True, borderType=cv2.BORDER_REFLECT)


def guided_filter(guide: np.ndarray, src: np.ndarray, radius: int, eps: float) -> np.ndarray:
    """Grayscale-guide guided filter. `guide` and `src` are float32 HxW arrays."""
    guide = guide.astype(np.float32)
    src = src.astype(np.float32)
    mean_i = _box(guide, radius)
    mean_p = _box(src, radius)
    cov_ip = _box(guide * src, radius) - mean_i * mean_p
    var_i = _box(guide * guide, radius) - mean_i * mean_i
    a = cov_ip / (var_i + eps)
    b = mean_p - a * mean_i
    return _box(a, radius) * guide + _box(b, radius)


def guided_filter_color(guide_rgb: np.ndarray, src: np.ndarray, radius: int, eps: float) -> np.ndarray:
    """Colour-guide guided filter (He et al. 2010). `guide_rgb` float32 HxWx3 in [0, 1], `src` float32 HxW."""
    guide = guide_rgb.astype(np.float32)
    src = src.astype(np.float32)
    mean_i = np.dstack([_box(guide[..., c], radius) for c in range(3)])
    mean_p = _box(src, radius)
    cov_ip = np.dstack([_box(guide[..., c] * src, radius) for c in range(3)]) - mean_i * mean_p[..., None]

    sigma = np.empty(guide.shape[:2] + (3, 3), np.float32)
    for i in range(3):
        for j in range(i, 3):
            v = _box(guide[..., i] * guide[..., j], radius) - mean_i[..., i] * mean_i[..., j]
            sigma[..., i, j] = v
            sigma[..., j, i] = v
    sigma += np.eye(3, dtype=np.float32) * eps
    a = np.linalg.solve(sigma, cov_ip[..., None])[..., 0]
    b = mean_p - np.einsum("hwc,hwc->hw", a, mean_i)
    mean_a = np.dstack([_box(a[..., c], radius) for c in range(3)])
    return np.einsum("hwc,hwc->hw", mean_a, guide) + _box(b, radius)


def refine_alpha_edges(rgb: np.ndarray, alpha: np.ndarray, band_px: int = 2, radius: int = 1, eps: float = 2e-3) -> np.ndarray:
    """Snaps a soft or hard uint8 matte to the colour edge of `rgb`, only within `band_px` of the outline.

    Solid interior and clear exterior are left exactly as they were; only the uncertain band is re-estimated.
    """
    if alpha.max() == 0:
        return alpha
    solid = alpha >= 128
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * band_px + 1, 2 * band_px + 1))
    outer = cv2.dilate(solid.astype(np.uint8), k) > 0
    inner = cv2.erode(solid.astype(np.uint8), k) > 0
    band = outer & ~inner
    if not band.any():
        return alpha

    ys, xs = np.nonzero(outer)
    pad = radius * 2 + 2
    y0, y1 = max(0, ys.min() - pad), min(alpha.shape[0], ys.max() + pad + 1)
    x0, x1 = max(0, xs.min() - pad), min(alpha.shape[1], xs.max() + pad + 1)

    guide = rgb[y0:y1, x0:x1].astype(np.float32) / 255.0
    a = alpha[y0:y1, x0:x1].astype(np.float32) / 255.0
    refined = np.clip(guided_filter_color(guide, a, radius, eps), 0.0, 1.0)

    # Only trust the guide where the outline actually has colour contrast. White fabric on a white canvas has
    # almost none, and snapping there would smear the edge into a soft, fuzzy fade.
    gray = cv2.cvtColor(rgb[y0:y1, x0:x1], cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    local_var = _box(gray * gray, radius) - _box(gray, radius) ** 2
    trust = np.clip((local_var - 0.002) / (0.010 - 0.002), 0.0, 1.0)
    blended = a * (1.0 - trust) + refined * trust

    out = alpha.copy()
    sub_band = band[y0:y1, x0:x1]
    region = out[y0:y1, x0:x1]
    region[sub_band] = np.round(blended[sub_band] * 255.0).astype(np.uint8)
    return out


def signed_distance(alpha: np.ndarray) -> np.ndarray:
    """Subpixel signed distance (px, positive inside) from a uint8 matte, using coverage near the edge."""
    a = alpha.astype(np.float32) / 255.0
    inside = (a >= 0.5).astype(np.uint8)
    d_in = cv2.distanceTransform(inside, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    d_out = cv2.distanceTransform(1 - inside, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    sdf = np.where(inside > 0, d_in - 0.5, 0.5 - d_out).astype(np.float32)
    soft = (a > 0.02) & (a < 0.98) & (np.abs(sdf) < 1.5)
    sdf[soft] = a[soft] - 0.5
    return sdf


def _render_sdf(sdf: np.ndarray) -> np.ndarray:
    return np.round(np.clip(sdf + 0.5, 0.0, 1.0) * 255.0).astype(np.uint8)


def resize_alpha_sdf(alpha: np.ndarray, out_w: int, out_h: int, smooth_sigma: float = 0.6) -> np.ndarray:
    """Resizes a matte by resampling its signed distance field, so outlines stay smooth at any scale."""
    h, w = alpha.shape[:2]
    if out_w <= w and out_h <= h:
        return cv2.resize(alpha, (out_w, out_h), interpolation=cv2.INTER_AREA)
    sdf = signed_distance(alpha)
    if smooth_sigma > 0:
        sdf = cv2.GaussianBlur(sdf, (0, 0), smooth_sigma)
    scale = 0.5 * (out_w / w + out_h / h)
    up = cv2.resize(sdf, (out_w, out_h), interpolation=cv2.INTER_CUBIC) * scale
    return _render_sdf(up)


def smooth_alpha_sdf(alpha: np.ndarray, smooth_sigma: float = 0.7, supersample: int = 4) -> np.ndarray:
    """Removes stair-steps at the current resolution: smooth the SDF, render at `supersample`x, area-average back."""
    h, w = alpha.shape[:2]
    if alpha.max() == 0:
        return alpha
    up = resize_alpha_sdf(alpha, w * supersample, h * supersample, smooth_sigma=smooth_sigma)
    out = cv2.resize(up, (w, h), interpolation=cv2.INTER_AREA)
    out[alpha == 255] = np.maximum(out[alpha == 255], cv2.erode(alpha, np.ones((3, 3), np.uint8))[alpha == 255])
    return out


def fill_transparent_rgb(rgb: np.ndarray, alpha: np.ndarray, band_px: int = 16, solid_threshold: int = 200) -> np.ndarray:
    """Copies the nearest solid colour into transparent/semi-transparent pixels within `band_px` of the art.

    Beyond the band RGB is set to 0 to keep files compact. Prevents dark halos when the image is upscaled
    or shrunk by tools that do not premultiply alpha.
    """
    known = alpha >= solid_threshold
    out = rgb.copy()
    if not known.any():
        out[alpha == 0] = 0
        return out
    unknown = (~known).astype(np.uint8)
    dist, labels = cv2.distanceTransformWithLabels(unknown, cv2.DIST_L2, cv2.DIST_MASK_5, labelType=cv2.DIST_LABEL_PIXEL)
    ys, xs = np.nonzero(unknown == 0)  # zero pixels are labelled 1..N in raster order
    near = (~known) & (dist <= band_px)
    idx = labels[near] - 1
    out[near] = rgb[ys[idx], xs[idx]]
    out[(~known) & (dist > band_px)] = 0
    return out


def denoise_chroma(rgb: np.ndarray, alpha: Optional[np.ndarray] = None, radius: int = 2, eps: float = 4e-4) -> np.ndarray:
    """Luma-guided filtering of Cr/Cb: removes compression chroma blocks/noise, keeps luma detail untouched."""
    ycc = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb).astype(np.float32) / 255.0
    y = ycc[..., 0]
    for c in (1, 2):
        ycc[..., c] = guided_filter(y, ycc[..., c], radius, eps)
    out = cv2.cvtColor(np.clip(ycc * 255.0 + 0.5, 0, 255).astype(np.uint8), cv2.COLOR_YCrCb2RGB)
    out[..., :] = np.where(out.sum(axis=2, keepdims=True) == 0, rgb, out) if alpha is None else out
    return out
