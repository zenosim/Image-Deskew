"""Pre-Processing / Post-Processing System: Anime De-Shine & True Cel Color Restorer
Eliminates specular glare hotspots and photographic sheen from figures, stickers, and scans.
Converts noisy photo/screen textures into clean, authentic 2D anime cel-shaded colors.

Technique:
  1. Safe Specular Glare Mitigation:
     - Detects genuine blown-out specular highlights (V > 245, S < 25) without black-level erosion.
     - Blends hotspots smoothly into local surface color rather than clamping to black.
  2. Edge-Preserving Surface Smoothing:
     - Bilateral filtering smooths paper/plastic grain while keeping cartoon line art razor-sharp.
  3. Hue-Preserved Cel Quantization:
     - Keeps Hue 100% untouched so skin stays skin, hair stays blue, and clothes retain true colors.
     - Quantizes Value/Luminance into clean 2D anime cel bands (Highlight, Midtone, Shadow, Deep Shadow).
  4. Crisp Cartoon Ink Line Enhancement:
     - Enhances line art contours without creating noisy dark blotches or grain.
  5. Alpha-Aware:
     - Strictly operates on opaque pixels, leaving transparent pixels and backgrounds untouched.
"""

import time
from dataclasses import dataclass
from typing import Optional
import cv2
import numpy as np
from PIL import Image

from .alpha_refine import fill_transparent_rgb


@dataclass
class ShineRemovalResult:
    cleaned_image: Image.Image
    glare_mask: np.ndarray
    shine_detected: bool
    execution_time_s: float


class ShineRemover:
    def __init__(self):
        pass

    def remove_shine(
        self,
        image: Image.Image,
        strength: int = 60,
        color_tiers: int = 40,
        flat_cel_look: bool = False
    ) -> ShineRemovalResult:
        """
        Removes specular glare reflections and restores authentic, ultra-detailed 2D anime cel colors
        with original digital master source fidelity.

        Accepts RGB or RGBA input. When RGBA, strictly processes opaque pixels and
        preserves the alpha channel untouched.

        Args:
            image: Input PIL Image (RGB or RGBA)
            strength: Glare suppression and cel flattening aggressiveness 10-100 (default 60).
            color_tiers: Number of discrete anime cel tonal bands (default 40, range 16-64).
            flat_cel_look: Whether to quantize tones into flat posterized cel steps (default False).
        """
        t0 = time.time()
        has_alpha = image.mode == "RGBA"
        alpha_channel = None

        if has_alpha:
            np_full = np.array(image)
            alpha_channel = np_full[:, :, 3].copy()
            np_rgb = np_full[:, :, :3]
        else:
            if image.mode != "RGB":
                image = image.convert("RGB")
            np_rgb = np.array(image)

        h, w, _ = np_rgb.shape
        bgr = cv2.cvtColor(np_rgb, cv2.COLOR_RGB2BGR)

        # Build an opaque pixel mask (strictly process visible pixels)
        if alpha_channel is not None:
            opaque_mask = (alpha_channel > 50)
            # Inpaint/extend edge colors into near-transparent border zone so bilateral filter
            # never samples black [0, 0, 0] or background colors across character silhouette edges
            if np.any(opaque_mask):
                fringe_bleed_mask = (
                    (alpha_channel <= 50) &
                    (cv2.dilate(opaque_mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))) > 0)
                ).astype(np.uint8)
                if np.any(fringe_bleed_mask):
                    bgr = cv2.inpaint(bgr, fringe_bleed_mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
        else:
            opaque_mask = np.ones((h, w), dtype=bool)

        if not np.any(opaque_mask):
            return ShineRemovalResult(
                cleaned_image=image,
                glare_mask=np.zeros((h, w), dtype=np.uint8),
                shine_detected=False,
                execution_time_s=0.0
            )

        # -----------------------------------------------------------------
        # 1. Safe Specular Glare Detection via Normalized Convolution
        # -----------------------------------------------------------------
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        h_chan, s_chan, v_chan = cv2.split(hsv)

        # Normalized convolution ensures background NEVER bleeds into character limbs/skin
        mask_f = opaque_mask.astype(np.float32)
        k_blur = max(31, int(min(h, w) * 0.20) | 1)
        denom = cv2.GaussianBlur(mask_f, (k_blur, k_blur), 0) + 1e-6
        v_local_mean = cv2.GaussianBlur(v_chan.astype(np.float32) * mask_f, (k_blur, k_blur), 0) / denom
        s_local_mean = cv2.GaussianBlur(s_chan.astype(np.float32) * mask_f, (k_blur, k_blur), 0) / denom

        # Natural warm anime skin tones (peach, blush, tan) - NEVER treat skin as glare!
        is_skin = ((h_chan <= 25) | (h_chan >= 170)) & (s_chan > 15) & (v_chan > 70) & opaque_mask

        # Glare is a localized blown-out highlight on a colored/saturated surface:
        # Requires high value, low saturation, and being on a saturated surface (s_local_mean > 28)
        glare_mask_raw = (
            (v_chan > 240) &
            (s_chan < 22) &
            (
                ((v_chan.astype(float) - v_local_mean) > 12.0) |
                ((s_local_mean - s_chan.astype(float)) > 14.0)
            ) &
            (
                # Small hotspot on saturated art: classic gate. Wide photo-glare bands wash out
                # their OWN neighborhood (s_local collapses), so also accept strong brightness
                # deficits vs local mean (measured: band core went 135 -> 3346 detected px).
                (s_local_mean > 28.0) |
                ((v_chan.astype(float) - v_local_mean) > 40.0)
            ) &
            opaque_mask
        )
        # Canvas exclusion for the relaxed branch: at canvas/art boundaries the dark art drags
        # v_local_mean down, so bright canvas fired (v - v_loc) > 40 and the remover brightened
        # the whole canvas ring (measured 156k px on clean art). Canvas pixels only pass the
        # strict gate.
        border_px_g = np.vstack([
            np_rgb[:15, :].reshape(-1, 3),
            np_rgb[-15:, :].reshape(-1, 3),
            np_rgb[:, :15].reshape(-1, 3),
            np_rgb[:, -15:, :].reshape(-1, 3),
        ])
        bg_median_g = np.median(border_px_g, axis=0)
        is_canvas_g = np.linalg.norm(np_rgb.astype(float) - bg_median_g.astype(float), axis=2) < 30.0
        relaxed_only = (
            (s_local_mean <= 28.0)
            & ((v_chan.astype(float) - v_local_mean) > 40.0)
            & is_canvas_g
        )
        glare_mask_raw[relaxed_only] = False

        # Specular glare is a small hotspot surrounded by a coloured surface. Large bright regions (white capes,
        # scarves, boots next to coloured art) passed the pixel test above and were being tinted and darkened
        # into blotches, so keep only compact components whose surrounding ring is clearly saturated.
        if np.any(glare_mask_raw):
            # Judge whole bright regions (the hotspot plus its white halo), not just the detected core.
            # Exclude canvas-colored pixels from the component map: the canvas is one giant bright
            # component, and a glare band crossing the character edge bridges into it, making
            # b_area exceed any sane cap (measured: 1.37M px component -> all band glare dropped).
            border_px = np.vstack([
                np_rgb[:15, :].reshape(-1, 3),
                np_rgb[-15:, :].reshape(-1, 3),
                np_rgb[:, :15].reshape(-1, 3),
                np_rgb[:, -15:, :].reshape(-1, 3),
            ])
            bg_median = np.median(border_px, axis=0)
            is_canvas_px = np.linalg.norm(np_rgb.astype(float) - bg_median.astype(float), axis=2) < 30.0

            bright = (v_chan > 235) & (s_chan < 30) & opaque_mask & (~is_canvas_px)
            n_b, b_lbl, b_stats, _ = cv2.connectedComponentsWithStats(bright.astype(np.uint8), connectivity=8)
            b_area = b_stats[:, cv2.CC_STAT_AREA]
            contains_glare = np.bincount(b_lbl[glare_mask_raw], minlength=n_b) > 0
            max_glare_area = max(40, int(0.10 * np.count_nonzero(opaque_mask)))
            # Ring = neighborhood of the bright region that is NOT itself bright. (The old
            # dilate(labels)>0 & labels==0 ring was empty by construction — dilating labels
            # spreads component labels over the ring, so labels==0 never held near any
            # component and EVERY component was dropped: measured legacy fixture core 360 -> 0.)
            ring_lbl = cv2.dilate(bright.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
            # Canvas ring exclusion only applies to actual canvas-like color, not to art whose
            # border pixels happen to define the median (synthetic/flat art: border median ==
            # art color, flagging 93% of the art as 'canvas' and emptying the ring). Require
            # BOTH proximity to border color AND low saturation: canvas is white/neutral.
            bg_sat = int(cv2.cvtColor(
                np.clip(bg_median, 0, 255).astype(np.uint8)[None, None, :],
                cv2.COLOR_RGB2HSV
            )[0, 0, 1])
            if bg_sat < 40:
                ring = ring_lbl & (b_lbl == 0) & opaque_mask & (v_chan > 60) & (~is_canvas_px)
            else:
                # colored border median = the 'canvas' is itself colored art; skip exclusion
                ring = ring_lbl & (b_lbl == 0) & opaque_mask & (v_chan > 60)
            ring_count = np.bincount(ring_lbl[ring], minlength=n_b)
            ring_coloured = np.bincount(ring_lbl[ring & (s_chan > 35)], minlength=n_b)
            coloured_frac = ring_coloured / np.maximum(ring_count, 1)
            keep_bright = contains_glare & (b_area <= max_glare_area) & (ring_count > 0) & (coloured_frac >= 0.7)
            keep_bright[0] = False
            # Label 0 = pixels in NO bright component (e.g. glare on canvas, which bright excludes
            # by design). The pixel test already validated those — the compaction applies only to
            # real components. Without this, every canvas-region glare pixel was dropped
            # (measured: core 29806 -> 0, all on label 0).
            glare_mask_raw = glare_mask_raw & (keep_bright[b_lbl] | (b_lbl == 0))

        glare_pixels = int(np.count_nonzero(glare_mask_raw))
        shine_detected = glare_pixels > 20

        deglared_bgr = bgr.copy()
        dilated_glare = np.zeros((h, w), dtype=np.uint8)

        if shine_detected:
            dilated_glare = cv2.dilate(
                glare_mask_raw.astype(np.uint8) * 255,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                iterations=1
            )
            v_repaired = v_chan.copy()
            s_repaired = s_chan.copy()
            core_px = int(np.count_nonzero(glare_mask_raw))
            # Wide bands (photo glare) need deeper suppression than small hotspots.
            att_cap = 0.85 if core_px > 3000 else 0.65
            attenuation = min(att_cap, (strength / 100.0) * 0.7 * (1.3 if core_px > 3000 else 1.0))
            target_v = np.clip(v_local_mean * 0.95, 160, 235).astype(np.uint8)
            v_repaired[dilated_glare > 0] = (
                (1.0 - attenuation) * v_chan[dilated_glare > 0] +
                attenuation * target_v[dilated_glare > 0]
            ).astype(np.uint8)
            # Also restore saturation from surrounding context to prevent desaturated gray spots
            target_s = np.clip(s_local_mean * 0.90, 0, 255).astype(np.uint8)
            s_repaired[dilated_glare > 0] = (
                (1.0 - attenuation) * s_chan[dilated_glare > 0] +
                attenuation * target_s[dilated_glare > 0]
            ).astype(np.uint8)

            deglared_hsv = cv2.merge([h_chan, s_repaired, v_repaired])
            deglared_bgr = cv2.cvtColor(deglared_hsv, cv2.COLOR_HSV2BGR)

        # -----------------------------------------------------------------
        # 2. Edge-Preserving Surface Smoothing (Bilateral Filter)
        # -----------------------------------------------------------------
        # Flattens photo/scanner grain while keeping line art and iris detail razor-sharp
        norm_strength = max(0.1, min(1.0, strength / 100.0))
        smoothed = cv2.bilateralFilter(deglared_bgr, d=5, sigmaColor=26, sigmaSpace=10)

        # -----------------------------------------------------------------
        # 3. Authentic Anime Cel Color Quantization & Vibrance
        # -----------------------------------------------------------------
        lab = cv2.cvtColor(smoothed, cv2.COLOR_BGR2LAB)
        L, A, B = cv2.split(lab)

        # Unsharp mask on Luminance channel: restores razor-sharp digital master lineart
        blur_L = cv2.GaussianBlur(L, (0, 0), 1.2)
        sharp_L = cv2.addWeighted(L, 1.25, blur_L, -0.25, 0)
        sharp_L = np.clip(sharp_L, 0, 255).astype(np.uint8)

        # Edge-preserving flat cel tonal clustering
        cel_L = cv2.bilateralFilter(sharp_L, d=5, sigmaColor=26, sigmaSpace=10)
        smooth_L = cv2.addWeighted(sharp_L, 0.50, cel_L, 0.50, 0)

        # Discrete Anime Cel Tonal Bands (Quantization) - only if explicitly requested
        if flat_cel_look and color_tiers > 0 and color_tiers < 48:
            tier_step = max(1.0, 255.0 / float(max(10, min(64, color_tiers))))
            quant_L = (np.round(smooth_L.astype(np.float32) / tier_step) * tier_step).clip(0, 255).astype(np.uint8)
            cel_blend_weight = min(0.80, 0.20 + (norm_strength * 0.50))
            final_L = cv2.addWeighted(quant_L, cel_blend_weight, smooth_L, 1.0 - cel_blend_weight, 0)
        else:
            final_L = smooth_L

        # Multi-Chroma Anime Vibrance: restores authentic anime digital master brilliance
        chroma_boost = 1.0  # saturation belongs to Color Pop; boosting here also amplified compression noise
        A_f = (A.astype(np.float32) - 128.0) * chroma_boost + 128.0
        B_f = (B.astype(np.float32) - 128.0) * chroma_boost + 128.0
        final_A = np.clip(A_f, 0, 255).astype(np.uint8)
        final_B = np.clip(B_f, 0, 255).astype(np.uint8)

        final_lab = cv2.merge([final_L, final_A, final_B])
        cel_bgr = cv2.cvtColor(final_lab, cv2.COLOR_LAB2BGR)

        # -----------------------------------------------------------------
        # 4. Crisp Cartoon Ink Line Enhancement (Difference of Gaussians)
        # -----------------------------------------------------------------
        gray = cv2.cvtColor(smoothed, cv2.COLOR_BGR2GRAY)
        g1 = cv2.GaussianBlur(gray, (0, 0), 1.0)
        g2 = cv2.GaussianBlur(gray, (0, 0), 2.2)
        dog = cv2.subtract(g2, g1)

        inner_mask = cv2.erode(
            opaque_mask.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        ) > 0

        is_ink_line = (dog > 14) & (gray < 85) & inner_mask
        ink_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        is_ink_line = cv2.morphologyEx(is_ink_line.astype(np.uint8), cv2.MORPH_OPEN, ink_k) > 0

        hsv_check = cv2.cvtColor(cel_bgr, cv2.COLOR_BGR2HSV)
        can_darken = is_ink_line & (hsv_check[:, :, 2] >= 40) & (hsv_check[:, :, 1] < 55)

        darkening_factor = 0.93 - (norm_strength * 0.05)
        for c in range(3):
            ch = cel_bgr[:, :, c].astype(float)
            ch[can_darken] = ch[can_darken] * darkening_factor
            cel_bgr[:, :, c] = np.clip(ch, 0, 255).astype(np.uint8)

        # -----------------------------------------------------------------
        # 5. Strict Alpha Protection & Zero-Background
        # -----------------------------------------------------------------
        final_rgb = cv2.cvtColor(cel_bgr, cv2.COLOR_BGR2RGB)

        if alpha_channel is not None:
            final_rgb = fill_transparent_rgb(final_rgb, alpha_channel, band_px=16)
            final_rgba = np.dstack([final_rgb, alpha_channel])
            result_pil = Image.fromarray(final_rgba, "RGBA")
        else:
            result_pil = Image.fromarray(final_rgb, "RGB")

        # -----------------------------------------------------------------
        # Wide-band glare pass: photo glare bands are elongated; spot attenuation
        # cannot restore them. When the core is band-shaped, LaMa-inpaint the
        # full band (measured on fixture: band MAE 82.9 -> 63.4, band visually gone).
        # -----------------------------------------------------------------
        try:
            from .glare_band import extend_glare_band

            # AUTO BAND PASS DISABLED (2026-10-03): canvas glare scatter collapses the PCA fit
            # (mask 653k px) and LaMa then smears the entire character. The standalone
            # extend_glare_band() helper remains available for an explicit/manual path once a
            # reliable on-art-only core exists. Spot attenuation above remains the safe default.
            pass
        except Exception:
            pass

        t_elapsed = time.time() - t0
        return ShineRemovalResult(
            cleaned_image=result_pil,
            glare_mask=dilated_glare,
            shine_detected=shine_detected,
            execution_time_s=round(t_elapsed, 4)
        )

    def analyze_for_auto_cel(
        self,
        image: Image.Image,
        character_mask: Optional[np.ndarray] = None
    ) -> dict:
        """
        Analyzes image characteristics (glare ratio, edge density, palette entropy)
        and calculates optimal parameters for the Cel Converter & De-Shine.
        If character_mask is provided, analysis is strictly bounded to the character artwork.
        """
        if image.mode != "RGBA" and image.mode != "RGB":
            image = image.convert("RGBA")
        np_img = np.array(image)
        has_alpha = (np_img.ndim == 3 and np_img.shape[2] == 4)
        if character_mask is not None:
            mask = (character_mask > 40)
            rgb = np_img[:, :, :3] if np_img.ndim == 3 and np_img.shape[2] >= 3 else np_img
        elif has_alpha:
            mask = np_img[:, :, 3] > 40
            rgb = np_img[:, :, :3]
        else:
            rgb = np_img
            # Estimate foreground: exclude uniform canvas background from glare count!
            h, w = rgb.shape[:2]
            border_px = np.vstack([
                rgb[:15, :].reshape(-1, 3),
                rgb[-15:, :].reshape(-1, 3),
                rgb[:, :15].reshape(-1, 3),
                rgb[:, -15:].reshape(-1, 3)
            ])
            bg_median = np.median(border_px, axis=0)
            bg_diff = np.linalg.norm(rgb.astype(float) - bg_median.astype(float), axis=2)
            mask = (bg_diff > 25.0)
            if not np.any(mask):
                mask = np.ones((h, w), dtype=bool)

        total_px = max(1, int(np.count_nonzero(mask)))
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        v = hsv[:, :, 2]
        s = hsv[:, :, 1]

        # Glare ratio: high V, low S on opaque character pixels
        glare_px = np.count_nonzero((v > 238) & (s < 28) & mask)
        glare_ratio = float(glare_px) / float(total_px)

        # Average saturation on colored regions
        colored_px = (s > 25) & mask
        mean_sat = float(np.mean(s[colored_px])) if np.any(colored_px) else 80.0

        # Edge density (lineart complexity)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_px = np.count_nonzero((edges > 0) & mask)
        edge_ratio = float(edge_px) / float(total_px)

        # Optimal calculations: bounded gracefully
        rec_strength = int(round(np.clip(45 + glare_ratio * 400.0, 40, 75)))

        if edge_ratio > 0.08:
            rec_tiers = 40
        elif edge_ratio > 0.04:
            rec_tiers = 32
        else:
            rec_tiers = 24

        summary = (
            f"Analyzed: {round(glare_ratio * 100, 1)}% specular glare detected, "
            f"{round(edge_ratio * 100, 1)}% lineart density. "
            f"Optimal: {rec_tiers} cel tiers, {rec_strength}% glare suppression."
        )

        return {
            "success": True,
            "glare_ratio_pct": round(glare_ratio * 100, 2),
            "edge_density_pct": round(edge_ratio * 100, 2),
            "mean_saturation": round(mean_sat, 1),
            "recommended_strength": rec_strength,
            "recommended_tiers": rec_tiers,
            "recommended_color_tiers": rec_tiers,
            "metrics": {
                "glare_ratio_pct": round(glare_ratio * 100, 2),
                "edge_density_pct": round(edge_ratio * 100, 2),
                "mean_saturation": round(mean_sat, 1)
            },
            "summary": summary
        }
