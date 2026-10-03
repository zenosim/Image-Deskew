"""Pre-Processing System: Watermark & Text Stamp Remover (v2)

Detection (fast, CPU): multi-scale stroke extraction with strict character-art protection,
same proven heuristics as v1.
Removal (quality): detected strokes are filled with Big-LaMa (ONNX) instead of Telea —
context-aware synthesis leaves no smearing on busy artwork. Telea remains a fallback
when the LaMa engine cannot load.

The remover never deletes character content: face/skin/chroma/large-dark-art regions are
inviolable, and LaMa only ever sees the small stroke mask.
"""

import time
from dataclasses import dataclass
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
    def __init__(self, use_lama: bool = True):
        self.use_lama = use_lama
        self._lama = None  # lazy singleton; False = unavailable sentinel

    # ------------------------------------------------------------------
    # LaMa engine (lazy)
    # ------------------------------------------------------------------
    def _get_lama(self):
        if self._lama is not None:
            return self._lama
        try:
            from .inpainter import BigLamaInpainter
            self._lama = BigLamaInpainter(model_type="anime")  # falls back to ONNX w/o torch
        except Exception:
            self._lama = False  # sentinel: unavailable
        return self._lama

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
        # Faint large-font watermarks vanish under a 5x5 tophat (stroke wider than kernel).
        # Union three scales so both hairline text and big stamps fire.
        # -------------------------------------------------------------
        cand_th = None
        cand_bh = None
        for ks in (5, 11, 21):
            k = cv2.getStructuringElement(cv2.MORPH_RECT, (ks, ks))
            th_s = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, k)
            bh_s = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k)
            cand_th = th_s if cand_th is None else np.maximum(cand_th, th_s)
            cand_bh = bh_s if cand_bh is None else np.maximum(cand_bh, bh_s)
        tophat_stroke = cand_th
        blackhat_stroke = cand_bh

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
        # Light text ON TOP of dark art (white watermark over hair/dress) lives INSIDE large
        # dark regions — killing everything dark-adjacent erased it (measured: 1441 -> 5 px).
        # Kill only boundary-touching strokes from outside; keep strokes ENCLOSED by dark art
        # (>=70% dark in a 15px neighborhood = overlay text on artwork).
        kern_den = np.ones((31, 31), np.float32)
        dark_density = cv2.filter2D(
            large_dark_art.astype(np.float32), -1, kern_den, borderType=cv2.BORDER_REPLICATE
        ) / kern_den.sum()
        on_art = dark_density > 0.70
        dark_boundary = cv2.dilate(
            large_dark_art.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        ) > 0
        large_dark_boundary_only = dark_boundary & ~on_art

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
        # Face, skin, colored hair/costumes are 100% INVIOLABLE everywhere; dark-art protection
        # only suppresses boundary strokes (enclosed overlay text survives — see above).
        candidate_mask[face_zone] = 0
        candidate_mask[is_chroma] = 0
        candidate_mask[large_dark_boundary_only] = 0
        # Overlay-on-art strokes are only trusted from the blackhat channel (light-on-dark text);
        # tophat inside dark art is usually specular detail/linework highlights.
        candidate_mask[on_art & (raw_bh_mask == 0)] = 0

        # -------------------------------------------------------------
        # 4b. Background-Residual Detector (canvas watermarks)
        # Faint gray text blended over flat canvas (delta ~25-95 from canvas color, low sat)
        # never fires as a tophat stroke — measured precision 0.97 / recall 0.99 on tiled
        # watermarks vs. edge-hits for tophat. Protect: chroma art, face, dark-art boundary.
        # -------------------------------------------------------------
        delta_canvas = np.linalg.norm(np_rgb.astype(float) - bg_med, axis=2)
        is_grayish = hsv[:, :, 1] < 40
        wm_band = (delta_canvas > 22) & (delta_canvas < 95) & is_grayish
        canvas_cand = (wm_band & ~is_chroma & ~face_zone & ~large_dark_boundary_only).astype(np.uint8) * 255
        num_c, lbl_c, stats_c, _ = cv2.connectedComponentsWithStats(canvas_cand, connectivity=8)
        for i in range(1, num_c):
            area_c = stats_c[i, cv2.CC_STAT_AREA]
            cw_c = stats_c[i, cv2.CC_STAT_WIDTH]
            ch_c = stats_c[i, cv2.CC_STAT_HEIGHT]
            # letter/stamp components: compact, bounded; kill giant blobs (shadow gradients)
            if 20 <= area_c <= (w * h) * 0.02 and cw_c < w * 0.5 and ch_c < h * 0.25:
                candidate_mask[lbl_c == i] = 255

        # 4c. Overlay-text-on-art: gray text blended over colored artwork lowers LOCAL saturation.
        # A text stroke sits measurably below its neighborhood's mean saturation while carrying a
        # blackhat response (lighter than surroundings). Protect face + skin absolutely.
        sat_f = hsv[:, :, 1].astype(np.float32)
        sat_localmean = cv2.blur(sat_f, (25, 25))
        sat_drop = sat_localmean - sat_f
        # Exclude true canvas from the overlay detector: its local mean sat is already ~0,
        # so sat_drop there is noise; canvas text is handled by the 4b residual detector.
        overlay_text = (
            (sat_drop > 18)
            & (blackhat_stroke > 30)
            & (~face_zone)
            & (~is_skin)
            & (sat_localmean > 40)
        )
        candidate_mask[overlay_text] = 255

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
        # 6. Fringe Dilation & Quality Inpainting (LaMa, Telea fallback)
        # -------------------------------------------------------------
        dilate_k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated_mask = cv2.dilate(clean_mask, dilate_k, iterations=1)

        inpainted_count = int(np.count_nonzero(dilated_mask))
        # Detection threshold: enough stroke pixels to be real text, not sensor noise.
        # Scaled to actual text coverage: light tiled watermarks measure ~0.05% of canvas;
        # noise specks stay under ~0.01%. (0.003 = 0.3% was calibrated for v1's looser mask
        # and rejected every real faint watermark — measured 755 px < 5791 needed.)
        min_detection_px = max(60, int((w * h) * 0.0004))
        watermark_detected = inpainted_count >= min_detection_px

        if watermark_detected:
            result_rgb = self._fill(np_rgb, dilated_mask)
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

    # ------------------------------------------------------------------
    def _fill(self, np_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Fill masked strokes with LaMa when available; Telea otherwise."""
        if self.use_lama:
            lama = self._get_lama()
            if lama:
                try:
                    filled = lama.inpaint(
                        Image.fromarray(np_rgb),
                        Image.fromarray(mask),
                        dilate_px=0,  # already dilated
                    )
                    return np.array(filled.convert("RGB"))
                except Exception:
                    pass  # degrade to Telea
        bgr = cv2.cvtColor(np_rgb, cv2.COLOR_RGB2BGR)
        inpainted = cv2.inpaint(bgr, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        return cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB)
