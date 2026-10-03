"""System 1: Background & Foreground Segmentor
Isolates the foreground sticker/object from digital background canvases and drop shadows.
Supports:
- SOTA Deep Learning models: BiRefNet ('birefnet-general') and U2-Net ('u2net') via rembg
- High-speed digital background detector (fast color & shadow unmixing)
"""

import numpy as np
import cv2
from PIL import Image
from dataclasses import dataclass, field
from typing import Any, Optional, Tuple

from .mask_ops import (
    ColorFeatures,
    apply_character_hint,
    apply_negative_mask,
    color_distance_below,
    detect_cast_shadow,
    dilate_bool,
    label_lut,
    prepare_hint_mask,
)


@dataclass
class SegmentationResult:
    rgba: Image.Image
    mask: np.ndarray          # 2D uint8 mask (0-255)
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    confidence: float = 1.0
    features: Optional[Any] = field(default=None, repr=False, compare=False)  # ColorFeatures of the input canvas


_GLOBAL_SESSIONS = {}

class BackgroundSegmentor:
    def __init__(self, model_name: str = "isnet-anime", fallback_to_digital: bool = True):
        """
        Args:
            model_name: 'isnet-anime', 'birefnet-general', 'u2net', or 'digital'
            fallback_to_digital: If true, falls back to digital thresholding if model download is unavailable.
        """
        self.model_name = model_name
        self.fallback_to_digital = fallback_to_digital

    def _get_session(self):
        global _GLOBAL_SESSIONS
        if self.model_name == "digital":
            return None
        if self.model_name not in _GLOBAL_SESSIONS:
            try:
                from rembg import new_session
                print(f"[BackgroundSegmentor] Initializing model session '{self.model_name}'...")
                _GLOBAL_SESSIONS[self.model_name] = new_session(self.model_name)
            except Exception as e:
                print(f"[BackgroundSegmentor] Warning: Could not initialize model '{self.model_name}': {e}")
                if self.fallback_to_digital:
                    print("[BackgroundSegmentor] Falling back to high-precision digital segmentor.")
                    self.model_name = "digital"
                else:
                    raise
        return _GLOBAL_SESSIONS.get(self.model_name)

    def segment(
        self,
        image: Image.Image,
        alpha_threshold: int = 15,
        clean_shadow_smudges: bool = True,
        smudge_sensitivity: int = 50,
        edge_inset_px: int = 0,
        character_bbox: Optional[Tuple[int, int, int, int]] = None,
        character_hint_mask: Optional[np.ndarray] = None,
        character_negative_mask: Optional[np.ndarray] = None
    ) -> SegmentationResult:
        """Removes background and drop shadows, returning isolated RGBA and mask.

        Args:
            image: Input PIL Image (RGB)
            alpha_threshold: Alpha cutoff threshold (0-255)
            clean_shadow_smudges: Removes drop-shadow smudges outside sticker border
            smudge_sensitivity: Aggressiveness of shadow pruning (0-100)
            edge_inset_px: Optional pixel shave from edge
            character_bbox: Optional (x, y, w, h) ROI bounding box to focus segmentation strictly on the character
            character_hint_mask: Optional HxW mask painted over the character; only highlighted parts are kept
            character_negative_mask: Optional HxW mask painted over areas that must be strictly excluded/removed
        """
        if image.mode != "RGB":
            image = image.convert("RGB")

        orig_w, orig_h = image.size
        hint = prepare_hint_mask(character_hint_mask, (orig_h, orig_w))
        neg_hint = prepare_hint_mask(character_negative_mask, (orig_h, orig_w))
        if hint is not None and character_bbox is None:
            character_bbox = cv2.boundingRect(hint.astype(np.uint8))

        # Handle user-provided character bounding box ROI
        crop_box = None
        inference_img = image
        if character_bbox is not None and len(character_bbox) == 4:
            bx, by, bw, bh = character_bbox
            if bw > 10 and bh > 10:
                # Add safe padding (10% of width/height or at least 20px) so boundary features aren't clipped
                pad_x = max(20, int(bw * 0.10))
                pad_y = max(20, int(bh * 0.10))
                x0 = max(0, bx - pad_x)
                y0 = max(0, by - pad_y)
                x1 = min(orig_w, bx + bw + pad_x)
                y1 = min(orig_h, by + bh + pad_y)
                crop_box = (x0, y0, x1, y1)
                inference_img = image.crop(crop_box)

        if self.model_name != "digital":
            try:
                from rembg import remove
                session = self._get_session()
                # Run ML background removal on (cropped or full) image
                result_rgba = remove(
                    inference_img,
                    session=session,
                    alpha_matting=False
                )
                np_sub_rgba = np.array(result_rgba)
                sub_mask = np_sub_rgba[:, :, 3]

                rgb_full = np.array(image)
                if crop_box is not None:
                    # Map cropped mask back onto full canvas coordinates
                    full_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
                    x0, y0, x1, y1 = crop_box
                    full_mask[y0:y1, x0:x1] = sub_mask
                    mask = full_mask

                    np_rgba = np.zeros((orig_h, orig_w, 4), dtype=np.uint8)
                    np_rgba[:, :, :3] = rgb_full
                    np_rgba[:, :, 3] = mask
                else:
                    np_rgba = np_sub_rgba
                    mask = sub_mask

                # Colour features are shared by the recovery and smudge stages (and later by the character extractor)
                feats = ColorFeatures(rgb_full)

                # 1. Apply Alpha Cutoff / Threshold
                if alpha_threshold > 0:
                    mask[mask < alpha_threshold] = 0

                # 1b. Smart Artwork & Stocking Recovery (resolves model 'mixed signals' on sheer lace, stockings, & dark fabrics)
                # Uses topological border-seeded flood fill to distinguish true canvas background from interior white clothing.
                # Only recovers pixels where the raw model provided some support (raw_mask >= 15).
                raw_mask = mask.copy()  # preserve original model output before modifications
                border_px = np.vstack([
                    rgb_full[:12, :].reshape(-1, 3),
                    rgb_full[-12:, :].reshape(-1, 3),
                    rgb_full[:, :12].reshape(-1, 3),
                    rgb_full[:, -12:].reshape(-1, 3)
                ])
                bg_median = np.median(border_px, axis=0)
                bg_std = np.std(border_px, axis=0)
                if np.mean(bg_std) < 40.0:
                    is_bg_color = color_distance_below(rgb_full, bg_median, 28.0)

                    # Topological true canvas BG: only pixels connected to the 4 outer image borders
                    # This protects interior white shirts, pale skin, and white frills from being
                    # mistaken for background on white canvases
                    border_mask = np.zeros((orig_h, orig_w), dtype=bool)
                    border_mask[0, :] = True; border_mask[-1, :] = True
                    border_mask[:, 0] = True; border_mask[:, -1] = True
                    n_bg_lbl, bg_lbls, _, _ = cv2.connectedComponentsWithStats(
                        is_bg_color.astype(np.uint8), connectivity=8
                    )
                    border_lut = label_lut(n_bg_lbl, bg_lbls[is_bg_color & border_mask])
                    is_true_canvas_bg = border_lut[bg_lbls] if border_lut.any() else is_bg_color

                    sat, val = feats.hsv[:, :, 1], feats.hsv[:, :, 2]
                    r, g, b = feats.rgb16
                    is_chroma = feats.is_chroma
                    is_skin = feats.is_skin
                    gray = feats.gray
                    grad_mag = feats.grad_mag
                    is_ink = (grad_mag > 32) & (gray < 90)

                    # Dynamic shadow brightness threshold based on canvas brightness
                    bg_val = float(np.mean(bg_median))
                    min_shadow_val = max(20, int(bg_val * 0.40))

                    diff_rg = np.abs(r - g)
                    diff_gb = np.abs(g - b)
                    is_neutral = (diff_rg <= 14) & (diff_gb <= 14) & (sat <= 18)
                    # Drop shadow candidates must have brightness above min_shadow_val (separates from black stockings)
                    is_drop_shadow_candidate = (
                        is_neutral & (grad_mag < 28) & (~is_skin) & (~is_chroma) & (~is_ink)
                        & (val >= min_shadow_val)
                    )

                    char_core = (mask > 210) | is_skin | is_chroma | is_ink
                    is_any_bg = is_bg_color | is_drop_shadow_candidate | is_true_canvas_bg
                    non_bg = ~is_any_bg
                    if crop_box is not None:
                        # Restrict recovery within the user-specified ROI
                        roi_bound = np.zeros((orig_h, orig_w), dtype=bool)
                        roi_bound[crop_box[1]:crop_box[3], crop_box[0]:crop_box[2]] = True
                        non_bg &= roi_bound

                    num_lbls, lbls, _, _ = cv2.connectedComponentsWithStats(non_bg.astype(np.uint8), connectivity=8)
                    is_recovered_core = label_lut(num_lbls, lbls[char_core])[lbls]
                    # Only solidify pixels where the raw model already had some support (>= 15 alpha)
                    # This prevents resurrecting zero-alpha drop shadow regions
                    # Solidify only the interior of recovered regions; their outline keeps the model's soft alpha
                    recovered_interior = cv2.erode(is_recovered_core.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
                    mask[recovered_interior & (raw_mask >= 15)] = 255

                    # Prune peripheral shadows that touch true canvas background
                    shadow_near_bg = dilate_bool(is_true_canvas_bg, 5)
                    peripheral_shadow = is_drop_shadow_candidate & shadow_near_bg & (mask > 0) & (~is_recovered_core)
                    mask[peripheral_shadow] = 0

                # 1c. Cast drop shadows fused to the outline (canvas colour darkened, soft outer falloff)
                if clean_shadow_smudges and smudge_sensitivity > 0 and np.mean(bg_std) < 40.0:
                    mask[detect_cast_shadow(rgb_full, mask, feats, bg_median, confidence=raw_mask)] = 0

                # 2. Edge-Grounded Shadow & Smudge Cleaner
                if clean_shadow_smudges and smudge_sensitivity > 0:
                    h, w = orig_h, orig_w
                    r, g, b = feats.rgb16
                    diff_rg = np.abs(r - g)
                    diff_gb = np.abs(g - b)
                    val = feats.hsv[:, :, 2]
                    sat = feats.hsv[:, :, 1]
                    gray = feats.gray
                    grad_mag = feats.grad_mag

                    # Character artwork barriers: protect fine lineart, eyelashes, and skin
                    is_chroma = feats.is_chroma
                    is_body_skin = feats.is_body_skin
                    is_lineart = (grad_mag > 35) & (gray < 85)

                    # Topological character limb & clothing connectivity:
                    # Legitimate dark clothing, sheer lace stockings, and boots connect to the character body
                    # Flood fill from skin/chroma through dark pixels to identify full clothing
                    char_core = is_chroma | is_body_skin | is_lineart
                    core_bridged = dilate_bool(char_core, 7)
                    # Identify all mask components connected to character core
                    num_comp, comp_lbls, _, _ = cv2.connectedComponentsWithStats((mask > 50).astype(np.uint8), connectivity=8)
                    is_char_body = label_lut(num_comp, comp_lbls[core_bridged])[comp_lbls]

                    # High edge-gradient cartoon ink barriers (edges surrounding clothing/stockings)
                    # The entire character body (clothing, stockings, dress, shoes, hair) is an inviolable barrier
                    is_barrier = is_chroma | is_body_skin | is_lineart | is_char_body

                    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
                    dist_from_outside = cv2.distanceTransform(binary, cv2.DIST_L2, 5)

                    # Drop shadows: monochromatic neutral grey/black, diffuse falloff, 2D patches
                    max_diff = max(8, int(smudge_sensitivity * 0.22))
                    max_sat = max(10, int(smudge_sensitivity * 0.32))
                    is_neutral = (diff_rg <= max_diff) & (diff_gb <= max_diff) & (sat <= max_sat)
                    # Low-gradient requirement: shadows are diffuse/soft falloff, NOT sharp cartoon fills
                    is_soft_shadow = is_neutral & (grad_mag < 28) & (val > 25) & (val < 185) & (~is_barrier) & (mask > 50)

                    # Morphological opening strips thin lineart, preserving only diffuse 2D shadow patches
                    k_blob = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                    shadow_blobs = cv2.morphologyEx(is_soft_shadow.astype(np.uint8), cv2.MORPH_OPEN, k_blob)

                    num_lbl, lbls, stats, _ = cv2.connectedComponentsWithStats(shadow_blobs, connectivity=8)
                    is_drop_shadow = np.zeros((h, w), dtype=bool)

                    if num_lbl > 1:
                        # Reach depth scales with sensitivity
                        reach_dist = max(12, int(min(h, w) * 0.04 * (smudge_sensitivity / 50.0)))

                        # Per-component statistics in one pass instead of a full-image scan per component
                        in_blob = lbls > 0
                        blob_labels = lbls[in_blob]
                        area = stats[:, cv2.CC_STAT_AREA]
                        min_dist = np.full(num_lbl, np.inf, dtype=np.float32)
                        np.minimum.at(min_dist, blob_labels, dist_from_outside[in_blob])
                        body_count = np.bincount(lbls[in_blob & is_char_body], minlength=num_lbl)
                        # A dilated component touches art <=> the component touches dilated art (symmetric kernel)
                        touches_art = label_lut(num_lbl, lbls[dilate_bool(is_chroma | is_body_skin | is_lineart, 5)])

                        # A peripheral drop shadow must contact or be within reach_dist of the outer boundary,
                        # must NOT be an internal character limb, and must cling to character art
                        is_shadow_comp = (area >= 60) & (min_dist <= reach_dist) & (body_count != area) & touches_art
                        is_shadow_comp[0] = False
                        is_drop_shadow = is_shadow_comp[lbls]

                    # Prune identified drop shadow patches from alpha mask without touching character barriers
                    is_drop_shadow &= (~is_barrier)
                    mask[is_drop_shadow] = 0

                    # Decide coverage on a cleaned binary mask (seal pinholes, drop 1px specks), but keep the
                    # model's soft anti-aliased alpha along the outline instead of hard-cutting it to 0/255
                    soft_mask = np.ascontiguousarray(mask, dtype=np.uint8)
                    micro_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                    decision = cv2.morphologyEx(soft_mask, cv2.MORPH_CLOSE, micro_k)
                    decision = cv2.morphologyEx(decision, cv2.MORPH_OPEN, micro_k)
                    decision = cv2.GaussianBlur(decision, (3, 3), 0)
                    decision = np.where(decision >= 128, 255, 0).astype(np.uint8)
                    mask = np.minimum(soft_mask, cv2.dilate(decision, micro_k))
                    mask = np.maximum(mask, cv2.erode(decision, micro_k))

                # 2b. User character highlight: keep only the painted character
                if hint is not None:
                    mask = apply_character_hint(mask, hint)

                # 2c. Negative mask: strictly eliminate excluded unwanted elements
                if neg_hint is not None:
                    mask = apply_negative_mask(mask, neg_hint)

                # 3. Edge Inset / Shave
                if edge_inset_px > 0:
                    erode_k = cv2.getStructuringElement(
                        cv2.MORPH_ELLIPSE,
                        (edge_inset_px * 2 + 1, edge_inset_px * 2 + 1)
                    )
                    mask = cv2.erode(mask, erode_k)

                np_rgba[:, :, 3] = mask
                result_rgba = Image.fromarray(np_rgba, "RGBA")

                contours, _ = cv2.findContours((mask > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    largest = max(contours, key=cv2.contourArea)
                    x, y, w, h = cv2.boundingRect(largest)
                    return SegmentationResult(rgba=result_rgba, mask=mask, bbox=(x, y, w, h), features=feats)
            except Exception as e:
                print(f"[BackgroundSegmentor] ML segmentation failed: {e}. Using digital segmentor.")

        # Digital segmentor (optimized for digital canvases, gradients, and drop shadows)
        res = self._segment_digital(image)
        if hint is not None or neg_hint is not None or edge_inset_px > 0:
            if hint is not None:
                res.mask = apply_character_hint(res.mask, hint)
            if neg_hint is not None:
                res.mask = apply_negative_mask(res.mask, neg_hint)
            if edge_inset_px > 0:
                erode_k = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (edge_inset_px * 2 + 1, edge_inset_px * 2 + 1)
                )
                res.mask = cv2.erode(res.mask, erode_k)
            rgba_np = np.array(res.rgba)
            rgba_np[:, :, 3] = res.mask
            res.rgba = Image.fromarray(rgba_np, "RGBA")
        return res

    def _segment_digital(self, image: Image.Image) -> SegmentationResult:
        """Digital image background extractor with mathematical drop-shadow suppression."""
        np_rgb = np.array(image)
        h, w, _ = np_rgb.shape

        # Sample 4 corners to estimate the background color
        corners = [
            np_rgb[:15, :15],
            np_rgb[:15, -15:],
            np_rgb[-15:, :15],
            np_rgb[-15:, -15:]
        ]
        corner_pixels = np.concatenate([c.reshape(-1, 3) for c in corners], axis=0)
        bg_mean = np.median(corner_pixels, axis=0).astype(np.float32)
        bg_norm = np.linalg.norm(bg_mean)

        # Compute per-pixel intensity and vector norms
        rgb_f = np_rgb.astype(np.float32)
        p_norm = np.linalg.norm(rgb_f, axis=2)
        p_norm = np.maximum(p_norm, 1e-5)

        # Cosine similarity between pixel color and background color
        # Shadows have the exact same color chromaticity (cos_sim > 0.985) but lower brightness!
        dot_product = np.sum(rgb_f * bg_mean, axis=2)
        cos_sim = dot_product / (p_norm * bg_norm)

        # Color difference
        diff = rgb_f - bg_mean
        dist = np.sqrt(np.sum(diff ** 2, axis=2))

        # Gradient magnitude (stickers have sharp boundary edges; drop shadows have soft blurry falloff)
        gray = cv2.cvtColor(np_rgb, cv2.COLOR_RGB2GRAY)
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)

        # Classify pixels:
        # A pixel is a drop shadow if it's darker than background, has very high cosine similarity with background,
        # and has low edge gradient.
        is_darker = p_norm < (bg_norm - 5.0)
        is_shadow_chroma = cos_sim > 0.982
        is_drop_shadow = is_darker & is_shadow_chroma & (grad_mag < 25.0)

        # Foreground is anything with significant color difference AND NOT a drop shadow
        is_fg = (dist > 18.0) & (~is_drop_shadow)

        # Find the sticker boundary using Canny edges to seal the contour
        canny = cv2.Canny(gray, 30, 90)
        combined = ((is_fg.astype(np.uint8) * 255) | canny)

        # Morphological close to bridge edge contours
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)

        # Keep largest connected component (the sticker)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros((h, w), dtype=np.uint8)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            cv2.drawContours(mask, [largest], -1, 255, -1)
            # Remove any residual shadow boundary by intersecting with non-shadow
            mask[is_drop_shadow] = 0
            # Smooth mask
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
            x, y, bw, bh = cv2.boundingRect(largest)
            bbox = (x, y, bw, bh)
        else:
            bbox = (0, 0, w, h)

        # Construct RGBA output
        rgba = np.dstack([np_rgb, mask])
        return SegmentationResult(rgba=Image.fromarray(rgba, "RGBA"), mask=mask, bbox=bbox)
