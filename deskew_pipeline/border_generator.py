"""Die-Cut Sticker Border Generator
Generates a crisp, smooth, rounded die-cut sticker border (stroke) around any RGBA asset.
Uses 2D Euclidean Distance Transform with curvature-flow smoothing for vector-like rounded offset corners (no fuzziness).
"""

import numpy as np
import cv2
from PIL import Image
from typing import Tuple, Optional

from .alpha_refine import fill_transparent_rgb


class DiecutBorderGenerator:
    def __init__(
        self,
        thickness_px: int = 14,
        color: Tuple[int, int, int] = (255, 255, 255),
        smoothing_radius: float = 1.0,
        die_cut_smoothing: int = 50,
        mode: str = "solid",
        glow_radius: int = 20,
        box_padding: int = 16,
        box_corner_radius: int = 12
    ):
        """
        Args:
            thickness_px: Thickness in pixels of the border/highlight stroke.
            color: (R, G, B) tuple for the highlight color (default: white).
            smoothing_radius: Antialiasing radius factor for curvature smoothing.
            die_cut_smoothing: Die-cut smoothing percentage [0-100%]:
                              - 0%: Sharp, tight contour following intricate silhouette details.
                              - 50%: Balanced, organic rounded vinyl cutline.
                              - 100%: Ultra-smooth sweeping arcs bridging crevices and rounding corners.
            mode: Highlighting mode:
                  - 'solid': Vector-smooth solid vinyl border.
                  - 'glow': Soft neon/vibrant radial bloom halo.
                  - 'outline': Crisp accent contour stroke (1-4px).
                  - 'box': Outer framing bounding box with rounded corners.
                  - 'both': Character contour highlight + outer box frame.
            glow_radius: Radius in pixels for neon glow halo spread.
            box_padding: Padding around character bounding box for box frame.
            box_corner_radius: Corner roundness for the bounding box.
        """
        self.thickness_px = thickness_px
        self.color = color
        self.smoothing_radius = smoothing_radius
        self.die_cut_smoothing = die_cut_smoothing
        self.mode = mode
        self.glow_radius = glow_radius
        self.box_padding = box_padding
        self.box_corner_radius = box_corner_radius

    def add_border(self, image: Image.Image) -> Image.Image:
        """Adds clean, vector-smooth die-cut vinyl border or neon highlight around the RGBA image."""
        if self.thickness_px <= 0 and self.mode != "box":
            return image

        if image.mode != "RGBA":
            image = image.convert("RGBA")

        np_img = np.array(image)
        h, w, _ = np_img.shape
        alpha = np_img[:, :, 3]

        if not np.any(alpha > 0):
            return image

        # Determine normalized smoothing factor [0.0, 1.0]
        if self.die_cut_smoothing is not None:
            s_norm = max(0.0, min(1.0, float(self.die_cut_smoothing) / 100.0))
        else:
            s_norm = max(0.0, min(1.0, (self.smoothing_radius - 0.2) / 2.8))

        # Pad canvas to accommodate outer border, smoothing spread, and glow
        pad = self.thickness_px + (self.glow_radius if self.mode in ("glow", "both") else 16) + (self.box_padding if "box" in self.mode else 0) + int(s_norm * 16)
        padded_alpha = np.pad(alpha, ((pad, pad), (pad, pad)), mode="constant")
        padded_rgb = np.pad(np_img[:, :, :3], ((pad, pad), (pad, pad), (0, 0)), mode="constant")
        pad_h, pad_w = padded_alpha.shape

        # 1. Supersampled Distance Field for Sub-Pixel Contour Highlights
        up_w = pad_w * 2
        up_h = pad_h * 2
        up_alpha = cv2.resize(padded_alpha, (up_w, up_h), interpolation=cv2.INTER_LANCZOS4)
        _, up_bin = cv2.threshold(up_alpha, 128, 255, cv2.THRESH_BINARY)

        # Morphological crevice relaxation (bridges deep spiky hair clefts and needle notches)
        if s_norm > 0.05:
            close_sz = int(round(3 + s_norm * 14)) | 1
            morph_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_sz, close_sz))
            up_bin = cv2.morphologyEx(up_bin, cv2.MORPH_CLOSE, morph_k)
        else:
            morph_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            up_bin = cv2.morphologyEx(up_bin, cv2.MORPH_CLOSE, morph_k)

        # 2. Euclidean Distance Transform (L2 metric for circular curvature expansion)
        up_dist = cv2.distanceTransform(255 - up_bin, cv2.DIST_L2, 5)

        # 3. Curvature-preserving Gaussian smoothing on distance field (mean curvature flow)
        sigma = 1.6 + s_norm * 7.5
        k_size = int(round(sigma * 4)) | 1
        up_dist_smooth = cv2.GaussianBlur(up_dist, (k_size, k_size), sigma)

        border_alpha = np.zeros((pad_h, pad_w), dtype=np.uint8)

        if self.mode in ("solid", "outline", "both"):
            # Solid vinyl stroke or crisp outline
            eff_thickness = 3 if self.mode == "outline" else self.thickness_px
            t_up = float(eff_thickness * 2)
            w_aa = max(1.8, 2.0 + s_norm * 1.5)
            u = np.clip((t_up + 1.0 - up_dist_smooth) / w_aa, 0.0, 1.0)
            up_border_alpha = u * u * (3.0 - 2.0 * u) * 255.0

            # Fill interior small void holes based on smoothing level (seals tiny needle voids)
            void_threshold = int(round(200 + s_norm * 3000))
            up_border_bin = (up_border_alpha > 128).astype(np.uint8)
            num_c, lbls_c, stats_c, _ = cv2.connectedComponentsWithStats(1 - up_border_bin)
            for i in range(1, num_c):
                if stats_c[i, cv2.CC_STAT_AREA] < void_threshold:
                    up_border_alpha[lbls_c == i] = 255.0

            down_solid = cv2.resize(up_border_alpha, (pad_w, pad_h), interpolation=cv2.INTER_AREA).astype(np.uint8)
            border_alpha = np.maximum(border_alpha, down_solid)

        if self.mode in ("glow", "both"):
            # Neon / bloom radial falloff glow
            dist_1x = cv2.resize(up_dist_smooth / 2.0, (pad_w, pad_h), interpolation=cv2.INTER_LINEAR)
            glow_rad = float(max(6, self.glow_radius))
            glow_falloff = np.exp(-1.8 * (dist_1x / glow_rad)) * 240.0
            glow_alpha = np.clip(glow_falloff, 0, 255).astype(np.uint8)
            glow_alpha[dist_1x > glow_rad * 1.5] = 0
            border_alpha = np.maximum(border_alpha, glow_alpha)

        if "box" in self.mode:
            # Highlight around the bounding box (card framing)
            ys, xs = np.where(padded_alpha > 50)
            if len(xs) > 0:
                bx0 = max(4, int(xs.min()) - self.box_padding)
                by0 = max(4, int(ys.min()) - self.box_padding)
                bx1 = min(pad_w - 5, int(xs.max()) + self.box_padding)
                by1 = min(pad_h - 5, int(ys.max()) + self.box_padding)

                box_mask = np.zeros((pad_h, pad_w), dtype=np.uint8)
                radius = min(self.box_corner_radius, (bx1 - bx0) // 4, (by1 - by0) // 4)
                cv2.rectangle(box_mask, (bx0 + radius, by0), (bx1 - radius, by1), 255, -1)
                cv2.rectangle(box_mask, (bx0, by0 + radius), (bx1, by1 - radius), 255, -1)
                cv2.circle(box_mask, (bx0 + radius, by0 + radius), radius, 255, -1)
                cv2.circle(box_mask, (bx1 - radius, by0 + radius), radius, 255, -1)
                cv2.circle(box_mask, (bx0 + radius, by1 - radius), radius, 255, -1)
                cv2.circle(box_mask, (bx1 - radius, by1 - radius), radius, 255, -1)

                stroke_w = max(2, min(self.thickness_px, 8))
                box_inner = cv2.erode(box_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (stroke_w * 2 + 1, stroke_w * 2 + 1)))
                box_stroke = cv2.subtract(box_mask, box_inner)
                border_alpha = np.maximum(border_alpha, box_stroke)

        # Composite output RGBA canvas
        out_rgba = np.zeros((pad_h, pad_w, 4), dtype=np.uint8)
        out_rgba[border_alpha > 0] = [*self.color, 255]
        out_rgba[:, :, 3] = border_alpha

        # Composite original character on top of border with sub-pixel color blending
        char_alpha = (padded_alpha.astype(np.float32) / 255.0)[:, :, None]
        char_rgb = padded_rgb.astype(np.float32)
        border_rgb = out_rgba[:, :, :3].astype(np.float32)

        composite_rgb = (char_rgb * char_alpha + border_rgb * (1.0 - char_alpha)).clip(0, 255).astype(np.uint8)
        out_rgba[:, :, :3] = composite_rgb
        out_rgba[:, :, 3] = np.maximum(out_rgba[:, :, 3], padded_alpha)
        out_rgba[:, :, :3] = fill_transparent_rgb(out_rgba[:, :, :3], out_rgba[:, :, 3], band_px=16)

        return Image.fromarray(out_rgba, "RGBA")
