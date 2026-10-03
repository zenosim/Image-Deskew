"""System 4: Edge Finishing & Asset Packaging (Post-Processor)
Implements:
1. Color Defringing (Color Unmixing): Eliminates background halos/fringes at semi-transparent edges.
2. Signed Distance Field (SDF) Alpha Antialiasing: Produces silky smooth sub-pixel curves.
3. Bounding Box Auto-Cropping: Tight transparent crop with user-defined margin padding.
4. Lossless RGBA Export.
"""

import numpy as np
import cv2
from PIL import Image
from dataclasses import dataclass
from typing import Tuple, Optional

from .alpha_refine import fill_transparent_rgb, smooth_alpha_sdf


@dataclass
class PostProcessResult:
    rgba: Image.Image
    cropped_bbox: Tuple[int, int, int, int]
    original_size: Tuple[int, int]
    final_size: Tuple[int, int]


class PostProcessor:
    def __init__(self, padding_px: int = 15, defringe: bool = True, antialias_radius: float = 0.8):
        """
        Args:
            padding_px: Transparent padding added around tight bounding box.
            defringe: If true, removes background color bleed at boundary pixels.
            antialias_radius: Sub-pixel Gaussian smoothing radius for the alpha channel.
        """
        self.padding_px = padding_px
        self.defringe = defringe
        self.antialias_radius = antialias_radius

    def process(self, image: Image.Image) -> PostProcessResult:
        """Runs the complete post-processing pipeline on an RGBA image."""
        if image.mode != "RGBA":
            image = image.convert("RGBA")

        np_rgba = np.array(image)
        h_orig, w_orig, _ = np_rgba.shape
        rgb = np_rgba[:, :, :3].copy()
        alpha = np_rgba[:, :, 3].copy()

        # Step 1: Color Defringing (Unmixing border color contamination)
        if self.defringe:
            rgb = self._defringe_colors(rgb, alpha)

        # Step 2: Alpha Antialiasing via 4x Super-Sampled Area Coverage (SSAA)
        if self.antialias_radius > 0:
            alpha = self._smooth_alpha_edges(alpha, radius=self.antialias_radius)

        # Extend edge colours under transparent pixels (zero beyond a narrow band): resizing in any tool,
        # premultiplied or not, then blends with the artwork's own colour instead of black
        rgb = fill_transparent_rgb(rgb, alpha, band_px=16)

        # Combine back into RGBA
        processed_rgba = np.dstack([rgb, alpha])

        # Step 3: Tight Bounding Box Crop with Padding
        cropped_rgba, bbox = self._crop_with_padding(processed_rgba, padding=self.padding_px)
        final_h, final_w, _ = cropped_rgba.shape

        return PostProcessResult(
            rgba=Image.fromarray(cropped_rgba, "RGBA"),
            cropped_bbox=bbox,
            original_size=(w_orig, h_orig),
            final_size=(final_w, final_h)
        )

    def _defringe_colors(self, rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        """
        Propagates solid foreground colors outward into semi-transparent boundary pixels
        without sampling unmasked background or creating cyan/dark halos.
        """
        core_mask = (alpha >= 220).astype(np.uint8)
        if not np.any(core_mask):
            core_mask = (alpha > 80).astype(np.uint8)
        if not np.any(core_mask):
            return rgb

        fringe_mask = ((alpha > 0) & (alpha < 220)).astype(np.uint8)
        if not np.any(fringe_mask):
            return rgb

        clean_rgb = rgb.copy()
        known = core_mask.copy()
        unknown = fringe_mask.copy()

        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        for _ in range(4):
            if not np.any(unknown):
                break
            dilated = cv2.dilate(known, k)
            frontier = (dilated > 0) & (unknown > 0)
            if not np.any(frontier):
                break

            sum_rgb = cv2.boxFilter((clean_rgb.astype(np.float32) * known[:, :, None]), cv2.CV_32F, (3, 3), normalize=False)
            sum_mask = cv2.boxFilter(known.astype(np.float32), cv2.CV_32F, (3, 3), normalize=False)
            valid = (sum_mask > 0) & frontier
            clean_rgb[valid] = np.clip(sum_rgb[valid] / sum_mask[valid, None], 0, 255).astype(np.uint8)

            known[frontier] = 1
            unknown[frontier] = 0

        return clean_rgb

    def _smooth_alpha_edges(self, alpha: np.ndarray, radius: float = 1.0) -> np.ndarray:
        """
        Computes silky smooth sub-pixel antialiasing using 4x Super-Sampled Area Coverage
        Anti-Aliasing (SSAA) combined with continuous area integration and Hermite smoothstep.
        Completely eliminates jagged stair-steps while keeping hair strands, curves, and tips razor-sharp.
        """
        if not np.any(alpha > 0):
            return alpha

        # Signed-distance smoothing: stair-steps are removed as a geometric curve, then re-rendered with
        # exact area coverage at 4x supersampling (keeps thin strands, no fuzzy halo)
        return smooth_alpha_sdf(alpha, smooth_sigma=max(0.5, 1.25 * float(radius)), supersample=4)

    def _crop_with_padding(self, rgba: np.ndarray, padding: int = 15) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """Crops image to tight bounding box of non-zero alpha, plus padding."""
        alpha = rgba[:, :, 3]
        contours, _ = cv2.findContours(alpha, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return rgba, (0, 0, rgba.shape[1], rgba.shape[0])

        # Combined bounding box of all contours
        all_pts = np.concatenate(contours, axis=0)
        x, y, w, h = cv2.boundingRect(all_pts)
        img_h, img_w, _ = rgba.shape

        # Expand with padding
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(img_w, x + w + padding)
        y2 = min(img_h, y + h + padding)

        cropped = rgba[y1:y2, x1:x2]
        return cropped, (x1, y1, x2 - x1, y2 - y1)
