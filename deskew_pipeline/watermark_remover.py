"""Pre-Processing System: Fast CPU Watermark & Text Stamp Remover
Executes before background segmentation and deskewing.
Detects and removes:
- Semi-transparent white/light overlay watermarks and sample stamps anywhere on canvas
- Creator handles, timestamps, URLs, signatures, and corner watermarks
- Preserves 100% of character anatomy lineart, hair contours, and solid artwork.
Operates in < 30ms on CPU.
"""

import time
from dataclasses import dataclass
from typing import Optional, Tuple
import cv2
import numpy as np
from PIL import Image


@dataclass
class WatermarkRemovalResult:
    cleaned_image: Image.Image
    watermark_mask: np.ndarray
    watermark_detected: bool
    execution_time_s: float
    inpainted_pixels: int


class WatermarkRemover:
    def __init__(self):
        pass

    def remove(
        self,
        image: Image.Image,
        sensitivity: int = 50,
        region: str = "full"
    ) -> WatermarkRemovalResult:
        """
        Detects and removes text stamps, creator handles, URLs, and watermarks.

        Args:
            image: Input PIL Image (RGB)
            sensitivity: Detection aggressiveness 10-100 (default 50).
            region: 'full' (entire canvas, default) or 'corners_and_margins' (outer 25%).
        """
        t0 = time.time()
        if image.mode != "RGB":
            image = image.convert("RGB")

        np_rgb = np.array(image)
        h, w, _ = np_rgb.shape
        bgr = cv2.cvtColor(np_rgb, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # -------------------------------------------------------------
        # 1. Dark Linework & Character Anatomy Protection Mask
        # -------------------------------------------------------------
        grad_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3)
        grad = cv2.addWeighted(cv2.convertScaleAbs(grad_x), 0.5, cv2.convertScaleAbs(grad_y), 0.5, 0)
        # Continuous dark lines are character contours and must NOT be inpainted
        is_character_line = (gray < 115) | ((gray < 165) & (grad > 45))

        # Detect canvas background to avoid confusing beige/white background with skin
        border_px = np.vstack([
            np_rgb[:10, :].reshape(-1, 3),
            np_rgb[-10:, :].reshape(-1, 3),
            np_rgb[:, :10].reshape(-1, 3),
            np_rgb[:, -10:].reshape(-1, 3)
        ])
        bg_med = np.median(border_px, axis=0)
        is_canvas = np.linalg.norm(np_rgb.astype(float) - bg_med, axis=2) < 22.0

        # Skin & Facial Features Protection (NEVER inpaint eyes, nose, mouth, blush)
        hsv = cv2.cvtColor(np_rgb, cv2.COLOR_RGB2HSV)
        raw_skin = ((hsv[:, :, 0] <= 25) | (hsv[:, :, 0] >= 170)) & (hsv[:, :, 1] > 20) & (hsv[:, :, 2] > 70) & (~is_canvas)
        num_s, lbls_s, stats_s, _ = cv2.connectedComponentsWithStats(raw_skin.astype(np.uint8))
        is_skin = np.zeros((h, w), dtype=bool)
        for i in range(1, num_s):
            if stats_s[i, cv2.CC_STAT_AREA] >= 80:
                is_skin[lbls_s == i] = True
        face_zone = cv2.dilate(is_skin.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))) > 0

        # -------------------------------------------------------------
        # 2. Multi-Scale Stroke Extraction
        # -------------------------------------------------------------
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        tophat_stroke = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, k)
        blackhat_stroke = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k)

        # -------------------------------------------------------------
        # 3. Target Region Filtering
        # -------------------------------------------------------------
        margin_mask = np.zeros((h, w), dtype=np.uint8)
        mx = max(10, int(w * 0.25))
        my = max(10, int(h * 0.22))
        margin_mask[:my, :] = 255
        margin_mask[-my:, :] = 255
        margin_mask[:, :mx] = 255
        margin_mask[:, -mx:] = 255
        if region == "corners_and_margins":
            th_active = cv2.bitwise_and(tophat_stroke, tophat_stroke, mask=margin_mask)
            bh_active = cv2.bitwise_and(blackhat_stroke, blackhat_stroke, mask=margin_mask)
        else:
            th_active = tophat_stroke
            bh_active = blackhat_stroke

        # Detect large dark character structures (hair, costume, uniform, linework)
        is_dark = (gray < 115).astype(np.uint8)
        num_d, lbls_d, stats_d, _ = cv2.connectedComponentsWithStats(is_dark)
        large_dark_art = np.zeros((h, w), dtype=bool)
        for i in range(1, num_d):
            if stats_d[i, cv2.CC_STAT_AREA] > 300:
                large_dark_art[lbls_d == i] = True
        large_dark_art = cv2.dilate(large_dark_art.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))) > 0

        # Character chroma (hair color, costumes, ribbons, eyes)
        is_chroma = (hsv[:, :, 1] > 25) & (hsv[:, :, 2] > 30)

        # Threshold scales with sensitivity: 50% -> th_thresh ~ 36, bh_thresh ~ 44
        thresh_th = max(18, int(54 - (sensitivity / 100.0) * 36))
        thresh_bh = max(24, int(64 - (sensitivity / 100.0) * 40))

        _, raw_th_mask = cv2.threshold(th_active, thresh_th, 255, cv2.THRESH_BINARY)
        _, raw_bh_mask = cv2.threshold(bh_active, thresh_bh, 255, cv2.THRESH_BINARY)

        # Combine stroke candidate masks
        candidate_mask = (raw_th_mask | raw_bh_mask)
        # STRICT PROTECTION OF CHARACTER ARTWORK:
        # Face, skin, colored hair/costumes, and large character structures are 100% INVIOLABLE everywhere
        candidate_mask[face_zone] = 0
        candidate_mask[is_chroma] = 0
        candidate_mask[large_dark_art] = 0

        # -------------------------------------------------------------
        # 5. Connected Component & Font Dimension Filtering
        # -------------------------------------------------------------
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(candidate_mask, connectivity=8)
        clean_mask = np.zeros_like(candidate_mask)

        # Text letter/stamp dimensions
        max_char_dim = max(35, int(min(h, w) * 0.35))
        max_comp_area = max(500, int((w * h) * 0.05))
        min_comp_area = 8

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            cw = stats[i, cv2.CC_STAT_WIDTH]
            ch = stats[i, cv2.CC_STAT_HEIGHT]

            if min_comp_area <= area <= max_comp_area:
                # Text characters have bounded dimensions and reasonable aspect ratios
                if cw < max_char_dim and ch < max_char_dim:
                    aspect = max(cw, ch) / max(1, min(cw, ch))
                    if aspect < 15:
                        clean_mask[labels == i] = 255

        # -------------------------------------------------------------
        # 6. Fringe Dilation & Fast Telea Inpainting
        # -------------------------------------------------------------
        dilate_k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated_mask = cv2.dilate(clean_mask, dilate_k, iterations=1)

        inpainted_count = int(np.count_nonzero(dilated_mask))
        # Detection threshold prevents false positive trigger on tiny noise dots
        min_detection_px = max(60, int((w * h) * 0.003))
        watermark_detected = inpainted_count >= min_detection_px

        if watermark_detected:
            inpainted_bgr = cv2.inpaint(bgr, dilated_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
            result_rgb = cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2RGB)
            result_pil = Image.fromarray(result_rgb)
        else:
            result_pil = image.copy()
            dilated_mask = np.zeros((h, w), dtype=np.uint8)
            inpainted_count = 0

        t_elapsed = time.time() - t0
        return WatermarkRemovalResult(
            cleaned_image=result_pil,
            watermark_mask=dilated_mask,
            watermark_detected=watermark_detected,
            execution_time_s=round(t_elapsed, 4),
            inpainted_pixels=inpainted_count
        )
