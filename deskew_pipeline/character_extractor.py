"""System 2: Character / Artwork Extractor
Specialized for die-cut stickers:
Distinguishes the inner character artwork from the outer white/clear die-cut vinyl border.
Allows extracting 'just the character' with zero white border remnants.
"""

import numpy as np
import cv2
from PIL import Image
from dataclasses import dataclass
from typing import Tuple, Optional

if __package__ is None or __package__ == "":
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from deskew_pipeline.mask_ops import (
        ColorFeatures, color_distance_below, dilate_bool, label_lut,
        apply_character_hint, apply_negative_mask, prepare_hint_mask, detect_cast_shadow
    )
else:
    from .mask_ops import (
        ColorFeatures, color_distance_below, dilate_bool, label_lut,
        apply_character_hint, apply_negative_mask, prepare_hint_mask, detect_cast_shadow
    )


@dataclass
class CharacterExtractionResult:
    rgba: Image.Image
    character_mask: np.ndarray      # Mask of only the inner character (0-255)
    margin_mask: np.ndarray         # Mask of the stripped die-cut border
    mode: str                       # 'character_only', 'full_sticker', or 'custom_margin'


class CharacterExtractor:
    def __init__(
        self,
        mode: str = "character_only",
        margin_inset_px: int = 2,
        ensure_connected: bool = True,
        connectivity_mode: str = "bridge",
        bridge_thickness_px: int = 10,
        clean_hair_gaps: bool = True,
        clean_shadows: bool = True
    ):
        """
        Args:
            mode: 'character_only' (strips white vinyl border),
                  'full_sticker' (keeps entire sticker with border),
                  'custom_margin' (erodes border by margin_inset_px)
            margin_inset_px: Extra pixels to shave off boundary to avoid white fringing.
            ensure_connected: If True, guarantees the extracted asset is a single continuous piece.
            connectivity_mode: 'bridge' (connects floating parts to main body) or 'largest' (keeps only main body).
            bridge_thickness_px: Thickness of connecting bridge between disjoint components.
            clean_hair_gaps: If True, clears trapped background and shadows from hair loops & limb gaps.
            clean_shadows: If True, detects and removes exterior cast drop shadows.
        """
        self.mode = mode
        self.margin_inset_px = margin_inset_px
        self.ensure_connected = ensure_connected
        self.connectivity_mode = connectivity_mode
        self.bridge_thickness_px = bridge_thickness_px
        self.clean_hair_gaps = clean_hair_gaps
        self.clean_shadows = clean_shadows

    def extract(
        self,
        sticker_rgba: Image.Image,
        sticker_mask: Optional[np.ndarray] = None,
        raw_image: Optional[Image.Image] = None,
        character_bbox: Optional[Tuple[int, int, int, int]] = None,
        features: Optional[ColorFeatures] = None,
        character_hint_mask: Optional[np.ndarray] = None,
        character_negative_mask: Optional[np.ndarray] = None
    ) -> CharacterExtractionResult:
        """
        Extracts the character artwork from the sticker RGBA image.
        Supports stripping the outer vinyl die-cut border, clearing drop shadows,
        and cleanly removing background trapped inside internal hair loops.

        `features` may carry colour features already computed by the segmentor for the same raw canvas.
        """
        np_rgba = np.array(sticker_rgba)
        h, w, _ = np_rgba.shape

        if sticker_mask is None:
            sticker_mask = np_rgba[:, :, 3]

        hint = prepare_hint_mask(character_hint_mask, (h, w))
        neg_hint = prepare_hint_mask(character_negative_mask, (h, w))

        if self.mode == "full_sticker":
            character_mask = sticker_mask.copy()
            if self.clean_hair_gaps and raw_image is not None:
                try:
                    raw_np = np.array(raw_image.convert("RGB"))
                    if raw_np.shape[:2] == (h, w):
                        border_px = np.vstack([
                            raw_np[:12, :].reshape(-1, 3),
                            raw_np[-12:, :].reshape(-1, 3),
                            raw_np[:, :12].reshape(-1, 3),
                            raw_np[:, -12:].reshape(-1, 3)
                        ])
                        bg_med = np.median(border_px, axis=0)
                        diff = np.linalg.norm(raw_np.astype(float) - bg_med.astype(float), axis=2)
                        gray = cv2.cvtColor(raw_np, cv2.COLOR_RGB2GRAY)
                        hsv = cv2.cvtColor(raw_np, cv2.COLOR_RGB2HSV)
                        sat = hsv[:, :, 1]

                        is_bg = (diff < 32.0)
                        is_shadow = (sat < 35) & (diff < 95.0) & (gray > 75)
                        is_neutral_light = (sat < 25) & (gray > 140) & (diff < 75.0)

                        # Real art: high saturation, skin tones, or dark ink lines
                        is_chroma = (sat > 40)
                        h_deg = hsv[:, :, 0] * 2
                        is_skin = ((h_deg <= 30) | (h_deg >= 340)) & (sat >= 15) & (sat <= 85) & (gray >= 115)
                        is_dark_line = (gray < 85)
                        is_art = is_chroma | is_skin | is_dark_line

                        # Candidate gap pixels inside sticker foreground
                        cand_gap = (character_mask > 40) & (is_bg | is_shadow | is_neutral_light) & (~is_art)

                        # Connected components of candidate gap pixels
                        num_g, g_lbls, g_stats, _ = cv2.connectedComponentsWithStats(cand_gap.astype(np.uint8), connectivity=8)
                        for gi in range(1, num_g):
                            g_area = g_stats[gi, cv2.CC_STAT_AREA]
                            if 15 <= g_area <= 35000:
                                pts = (g_lbls == gi)
                                touches_bg = np.any(is_bg[pts])
                                mean_conf = np.mean(sticker_mask[pts])
                                if not touches_bg and mean_conf > 235:
                                    continue
                                if mean_conf < 220 or touches_bg:
                                    perim = dilate_bool(pts, 3) & (~pts)
                                    perim_total = np.count_nonzero(perim)
                                    if perim_total > 0:
                                        skin_ratio = np.count_nonzero(perim & is_skin) / float(perim_total)
                                        if skin_ratio < 0.65:
                                            character_mask[pts] = 0
                except Exception as e:
                    print(f"[FullSticker] Hair gap clean exception: {e}")

            if character_bbox is not None and len(character_bbox) == 4:
                bx, by, bw, bh = character_bbox
                roi_mask = np.zeros((h, w), dtype=bool)
                pad_x = max(15, int(bw * 0.08))
                pad_y = max(15, int(bh * 0.08))
                x0, y0 = max(0, bx - pad_x), max(0, by - pad_y)
                x1, y1 = min(w, bx + bw + pad_x), min(h, by + bh + pad_y)
                roi_mask[y0:y1, x0:x1] = True
                character_mask[~roi_mask] = 0

            if hint is not None:
                character_mask = apply_character_hint(character_mask, hint)
            if neg_hint is not None:
                character_mask = apply_negative_mask(character_mask, neg_hint)

            out_rgba = np_rgba.copy()
            if self.ensure_connected:
                character_mask = self._ensure_single_connected_component(character_mask, out_rgba, neg_hint=neg_hint)
            margin_mask = np.zeros_like(sticker_mask)
            if neg_hint is not None:
                character_mask = apply_negative_mask(character_mask, neg_hint)
            out_rgba[:, :, 3] = character_mask
            return CharacterExtractionResult(
                rgba=Image.fromarray(out_rgba, "RGBA"),
                character_mask=character_mask,
                margin_mask=margin_mask,
                mode=self.mode
            )

        if self.mode == "custom_margin":
            character_mask = sticker_mask.copy()
            if self.margin_inset_px > 0:
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.margin_inset_px * 2 + 1, self.margin_inset_px * 2 + 1))
                character_mask = cv2.erode(character_mask, k)
            if hint is not None:
                character_mask = apply_character_hint(character_mask, hint)
            if neg_hint is not None:
                character_mask = apply_negative_mask(character_mask, neg_hint)
            margin_mask = cv2.subtract(sticker_mask, character_mask)
            out_rgba = np_rgba.copy()
            out_rgba[:, :, 3] = character_mask
            return CharacterExtractionResult(
                rgba=Image.fromarray(out_rgba, "RGBA"),
                character_mask=character_mask,
                margin_mask=margin_mask,
                mode=self.mode
            )

        # Get RGB from raw_image if provided (for pristine background and color sampling),
        # otherwise use the RGB channels of sticker_rgba.
        if features is not None and features.rgb.shape[:2] == (h, w):
            feats = features
        else:
            if raw_image is not None:
                raw_rgb = np.array(raw_image.convert("RGB"))
                if raw_rgb.shape[:2] != (h, w):
                    raw_rgb = cv2.resize(raw_rgb, (w, h), interpolation=cv2.INTER_LANCZOS4)
                rgb = raw_rgb
            else:
                rgb = np.ascontiguousarray(np_rgba[:, :, :3])
            feats = ColorFeatures(rgb)
        rgb = feats.rgb

        # HSV and Grayscale color spaces
        s_chan, v_chan = feats.hsv[:, :, 1], feats.hsv[:, :, 2]
        gray = feats.gray
        r, g, b = feats.rgb16

        sticker_fg = sticker_mask > 50
        outside_sticker = ~sticker_fg

        # 1. Sample canvas background distribution strictly from the RAW image borders
        is_canvas_bg = np.zeros((h, w), dtype=bool)
        diff_from_bg = np.full((h, w), 50.0)
        bg_median = np.array([255.0, 255.0, 255.0])
        has_consistent_canvas_bg = False
        if raw_image is not None:
            border_px = np.vstack([
                rgb[:15, :].reshape(-1, 3),
                rgb[-15:, :].reshape(-1, 3),
                rgb[:, :15].reshape(-1, 3),
                rgb[:, -15:].reshape(-1, 3)
            ])
            bg_median = np.median(border_px, axis=0)
            bg_std = np.std(border_px, axis=0)
            diff_from_bg = np.linalg.norm(rgb.astype(float) - bg_median.astype(float), axis=2)
            if np.mean(bg_std) < 45.0:
                is_canvas_bg = (diff_from_bg < 24.0)
                has_consistent_canvas_bg = True

        # 2. Inviolable Character Artwork Features
        # High-chroma colors (colored hair, eyes, costumes, accessories)
        is_chroma = feats.is_chroma
        # Anime skin tones (peach / blush / tan)
        is_skin = feats.is_skin
        is_body_skin = feats.is_body_skin
        # Crisp cartoon ink lines (Sobel & gradient magnitude)
        grad_mag = feats.grad_mag
        is_ink = (grad_mag > 32) & (gray < 100)

        # 3. Exterior cast drop shadows: canvas colour darkened, soft outer falloff, touching the true canvas
        cleaned_mask = sticker_mask.copy()
        if self.clean_shadows and has_consistent_canvas_bg:
            shadow = detect_cast_shadow(rgb, sticker_mask, feats, bg_median, confidence=sticker_mask)
            if hint is not None:
                shadow &= ~hint
            cleaned_mask[shadow] = 0

        # 4. Internal Negative Space & Hair Loop Cleaning:
        # Clears trapped background from hair loops, between arms, and between legs.
        # CRITICAL FIX: Only clear if pixels strictly match the exterior canvas background!
        # White shirts, pale legs, and white frills have artwork gradients and do NOT match canvas background.
        if self.clean_hair_gaps and has_consistent_canvas_bg:
            cand_holes = (cleaned_mask > 30) & (diff_from_bg < 24.0) & (s_chan < 22) & (~is_skin) & (~is_chroma) & (~is_ink)
            num_h, h_lbls, h_stats, _ = cv2.connectedComponentsWithStats(cand_holes.astype(np.uint8), connectivity=8)
            if num_h > 1:
                area = h_stats[:, cv2.CC_STAT_AREA].astype(np.float64)
                safe_area = np.maximum(area, 1.0)
                flat_lbls = h_lbls.ravel()
                # Model confidence: flat white clothing (capes, scarves, boots) is the canvas colour too, but the
                # segmentation model is sure it is foreground. Only uncertain regions are real see-through gaps.
                mean_conf = np.bincount(flat_lbls, weights=sticker_mask.ravel().astype(np.float64), minlength=num_h) / safe_area
                # Interior texture: clothing has variance, see-through canvas is flat
                rgb_f = rgb.reshape(-1, 3).astype(np.float64)
                chan_std = []
                for c in range(3):
                    s1 = np.bincount(flat_lbls, weights=rgb_f[:, c], minlength=num_h) / safe_area
                    s2 = np.bincount(flat_lbls, weights=rgb_f[:, c] ** 2, minlength=num_h) / safe_area
                    chan_std.append(np.sqrt(np.maximum(s2 - s1 ** 2, 0.0)))
                pts_std = np.where(area > 4, np.mean(chan_std, axis=0), 0.0)
                # Surrounding ring: if mostly skin (cleavage, inner thighs), never peel
                ring_lbls = cv2.dilate(h_lbls.astype(np.float32), np.ones((3, 3), np.uint8)).astype(np.int32)
                ring = (ring_lbls > 0) & (h_lbls == 0)
                ring_total = np.bincount(ring_lbls[ring], minlength=num_h)
                ring_skin = np.bincount(ring_lbls[ring & (is_body_skin | is_skin)], minlength=num_h)
                skin_ratio = ring_skin / np.maximum(ring_total, 1)
                clear = ((area >= 8) & (area <= 30000) & (ring_total > 0) & (skin_ratio <= 0.38)
                         & (pts_std < 32.0) & (mean_conf < 235.0))
                clear[0] = False
                cleaned_mask[clear[h_lbls]] = 0

        character_mask = cleaned_mask

        # If user specified character bounding box, zero out anything outside the expanded ROI
        if character_bbox is not None and len(character_bbox) == 4:
            bx, by, bw, bh = character_bbox
            pad_x = max(15, int(bw * 0.08))
            pad_y = max(15, int(bh * 0.08))
            x0, y0 = max(0, bx - pad_x), max(0, by - pad_y)
            x1, y1 = min(w, bx + bw + pad_x), min(h, by + bh + pad_y)
            roi_outside = np.ones((h, w), dtype=bool)
            roi_outside[y0:y1, x0:x1] = False
            character_mask[roi_outside] = 0

        # Inset / erode by user-defined margin if custom mode
        if self.mode == "custom_margin" and self.margin_inset_px > 0:
            erode_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.margin_inset_px * 2 + 1, self.margin_inset_px * 2 + 1))
            character_mask = cv2.erode(character_mask, erode_k)

        # Intersect with original sticker mask to ensure bounds are strictly respected
        character_mask = cv2.bitwise_and(character_mask, sticker_mask)

        # Discard detached floating non-art specks or residual border/shadow fragments
        clean_bin = (character_mask > 50).astype(np.uint8)
        num_c, c_lbls, c_stats, _ = cv2.connectedComponentsWithStats(clean_bin, connectivity=8)
        if num_c > 2:
            c_area = c_stats[:, cv2.CC_STAT_AREA]
            largest_c_idx = 1 + np.argmax(c_area[1:])
            is_art_comp = label_lut(num_c, c_lbls[is_chroma | is_body_skin])
            discard = (c_area < 40) | (~is_art_comp & (c_area < 150))
            discard[0] = False
            discard[largest_c_idx] = False
            character_mask[discard[c_lbls]] = 0

        if neg_hint is not None:
            character_mask = apply_negative_mask(character_mask, neg_hint)

        # Apply Ensure Connected Component if requested
        out_rgba = np_rgba.copy()
        if self.ensure_connected:
            character_mask = self._ensure_single_connected_component(character_mask, out_rgba, neg_hint=neg_hint)

        if hint is not None:
            character_mask = apply_character_hint(character_mask, hint)
        if neg_hint is not None:
            character_mask = apply_negative_mask(character_mask, neg_hint)

        # Compute the stripped margin mask
        margin_mask = cv2.subtract(sticker_mask, character_mask)

        # Construct character RGBA output
        out_rgba[:, :, 3] = character_mask

        return CharacterExtractionResult(
            rgba=Image.fromarray(out_rgba, "RGBA"),
            character_mask=character_mask,
            margin_mask=margin_mask,
            mode=self.mode
        )

    def _ensure_single_connected_component(self, mask: np.ndarray, rgba: np.ndarray, neg_hint: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Analyzes the mask connectivity. If disjoint islands exist:
        - In 'bridge' mode: automatically creates organic bridges between floating elements and main body.
        - In 'largest' mode: removes stray disjoint islands, keeping only the primary body.
        Bridge pixels are colour-filled in `rgba` in place.
        """
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

        # Label 0 is background; labels 1..num_labels-1 are foreground components
        if num_labels <= 2:
            # Already 0 or 1 component
            return mask

        # Find largest component index (excluding background 0)
        component_indices = list(range(1, num_labels))
        component_indices.sort(key=lambda i: stats[i, cv2.CC_STAT_AREA], reverse=True)
        largest_idx = component_indices[0]

        if self.connectivity_mode == "largest":
            # Keep only the largest component (with its soft anti-aliased edge alpha)
            return np.where(labels == largest_idx, mask, 0).astype(np.uint8)

        # 'bridge' mode: connect all valid components to the main body
        connected_mask = np.where(labels == largest_idx, mask, 0).astype(np.uint8)

        # Precompute HSV for color analysis of floating islands
        hsv_img = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2HSV)
        bridge_w = max(4, self.bridge_thickness_px)
        bridged_any = False

        for comp_idx in component_indices[1:]:
            area = stats[comp_idx, cv2.CC_STAT_AREA]
            if area < 40:
                # Ignore microscopic single-pixel noise
                continue

            comp_pixels = (labels == comp_idx)
            if neg_hint is not None and np.any(comp_pixels & neg_hint):
                # Never bridge back components that intersect the negative mask
                continue

            mean_sat = np.mean(hsv_img[comp_pixels, 1])
            mean_val = np.mean(hsv_img[comp_pixels, 2])

            # Discard detached drop shadow smudges or neutral background noise
            # Do NOT bridge back shadow blobs that were deliberately peeled/pruned
            if area < 600 and mean_sat < 28 and mean_val < 180:
                continue
            if area < 250 and mean_sat < 20:
                continue

            # Find closest point of this component to the current main body
            dist_to_main = cv2.distanceTransform((connected_mask == 0).astype(np.uint8), cv2.DIST_L2, 5)
            y_pts, x_pts = np.nonzero(comp_pixels)
            min_dist_idx = np.argmin(dist_to_main[y_pts, x_pts])
            p_comp = (int(x_pts[min_dist_idx]), int(y_pts[min_dist_idx]))

            # Find closest point on main_body boundary
            dist_to_comp = cv2.distanceTransform((~comp_pixels).astype(np.uint8), cv2.DIST_L2, 5)
            y_main, x_main = np.nonzero(connected_mask)
            min_main_idx = np.argmin(dist_to_comp[y_main, x_main])
            p_main = (int(x_main[min_main_idx]), int(y_main[min_main_idx]))

            # Draw smooth bridge line between p_comp and p_main, then add the component itself
            cv2.line(connected_mask, p_comp, p_main, 255, thickness=bridge_w)
            connected_mask = np.maximum(connected_mask, np.where(comp_pixels, mask, 0).astype(np.uint8))
            bridged_any = True

        # Colour all bridge pixels from neighbouring art in a single inpaint pass
        # (Telea ignores the prior values of masked pixels, so one pass over the final mask
        # matches inpainting after every bridge)
        if bridged_any:
            bridge_pixels = (connected_mask > 0) & (rgba[:, :, 3] == 0)
            if np.any(bridge_pixels):
                inpaint_mask = bridge_pixels.astype(np.uint8) * 255
                rgba[:, :, :3] = cv2.inpaint(rgba[:, :, :3], inpaint_mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)

        return connected_mask


if __name__ == "__main__":
    import os
    import sys

    print("=" * 60)
    print("      Character & Artwork Extractor - Standalone Demo")
    print("=" * 60)

    workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(workspace_dir, "output_results")
    os.makedirs(out_dir, exist_ok=True)

    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        test_img = sys.argv[1]
    else:
        print("[Usage] python character_extractor.py <image_path>")
        sys.exit(0)

    print(f"Loading image: {test_img}")
    input_img = Image.open(test_img).convert("RGBA")

    extractor = CharacterExtractor(mode="character_only", ensure_connected=True)
    res = extractor.extract(input_img)

    out_file = os.path.join(out_dir, "standalone_extracted_character.png")
    res.rgba.save(out_file)
    print(f"[Success] Successfully extracted character artwork!")
    print(f"Saved result to: {out_file}")
    print(f"Dimensions: {res.rgba.size}")
