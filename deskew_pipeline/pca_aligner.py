"""PCA Orientation & Bottom Alignment System
Performs 2D Principal Component Analysis (PCA) on extracted sticker/character artwork.
Ensures the sticker orientation is correctly aligned with the bottom (upright vertical or flat baseline).
"""

import math
import numpy as np
import cv2
from PIL import Image
from typing import Tuple, Dict, Any, Optional


class PCAOrientationAligner:
    def __init__(
        self,
        mode: str = "bottom_upright",
        detect_inverted: bool = True,
        snap_tolerance_deg: float = 45.0
    ):
        """
        Args:
            mode: Alignment strategy:
                  - 'bottom_upright': Aligns major axis vertically with base at bottom.
                  - 'bottom_baseline': Fits a horizontal baseline to bottom contact points.
                  - 'auto': Automatically decides vertical vs horizontal based on aspect ratio.
                  - 'horizontal': Aligns major axis parallel to bottom (for wide stickers/banners).
                  - 'snap_orthogonal': Snaps to the nearest 90-degree angle.
            detect_inverted: If True, checks mass distribution to ensure character isn't upside-down.
            snap_tolerance_deg: Angular threshold for axis classification.
        """
        self.mode = mode
        self.detect_inverted = detect_inverted
        self.snap_tolerance_deg = snap_tolerance_deg

    def align(self, image: Image.Image, mask: Optional[np.ndarray] = None) -> Tuple[Image.Image, Dict[str, Any]]:
        """
        Analyzes the 2D principal components and rotates the image to align with the bottom.
        """
        if image.mode != "RGBA":
            image = image.convert("RGBA")

        np_rgba = np.array(image)
        h, w, _ = np_rgba.shape

        if mask is None:
            alpha = np_rgba[:, :, 3]
            mask = (alpha > 127).astype(np.uint8) * 255
        else:
            alpha = mask

        ys, xs = np.where(mask > 127)
        if len(xs) < 15:
            return image, {"status": "skipped_too_few_pixels"}

        # 1. Compute 2D PCA on foreground coordinate point cloud
        pts = np.column_stack((xs, ys)).astype(np.float32)
        mean, eigenvectors, eigenvalues = cv2.PCACompute2(pts, mean=None)

        cx, cy = float(mean[0][0]), float(mean[0][1])
        e0 = eigenvectors[0]  # Major principal axis
        e1 = eigenvectors[1]  # Minor principal axis
        lambda0 = float(eigenvalues[0][0])
        lambda1 = float(eigenvalues[1][0])
        elongation = lambda0 / max(1e-5, lambda1)

        # Angle of the major principal axis relative to positive X-axis in degrees [-180, 180]
        # In screen coords: 0 deg = right, 90 deg = down (bottom), -90 deg = up (top), 180 = left
        major_angle = float(np.degrees(np.arctan2(e0[1], e0[0])))

        # 2. Determine target rotation angle based on mode
        rot_deg = 0.0
        target_axis = "vertical"

        # Angular distance to vertical (90 or -90) vs horizontal (0 or 180)
        dist_to_vert = min(abs(major_angle - 90.0), abs(major_angle - (-90.0)), abs(major_angle - 270.0))
        dist_to_horiz = min(abs(major_angle - 0.0), abs(major_angle - 180.0), abs(major_angle - (-180.0)))

        is_vertical = dist_to_vert <= dist_to_horiz

        if self.mode == "bottom_baseline":
            # Baseline mode: first check for a dominant straight cut edge in the lower half
            # (common for peeker stickers and flat-bottomed character cutouts)
            gray = cv2.cvtColor(np_rgba[:, :, :3], cv2.COLOR_RGB2GRAY)
            y_mid = int(h * 0.45)
            lower_gray = gray[y_mid:, :]
            lower_alpha = mask[y_mid:, :]
            edges = cv2.Canny(lower_gray, 40, 120)
            edges = edges & (lower_alpha > 0)

            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=35, minLineLength=45, maxLineGap=12)
            cut_angles = []
            if lines is not None:
                for l in lines:
                    x1, y1, x2, y2 = l.ravel()
                    angle = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
                    if angle > 90:
                        angle -= 180
                    elif angle < -90:
                        angle += 180
                    if abs(angle) <= 35:  # nearly horizontal cut edge
                        cut_angles.append(angle)

            if len(cut_angles) >= 2:
                # Median angle of detected straight cut edge
                baseline_angle = float(np.median(cut_angles))
                rot_deg = -baseline_angle
                target_axis = "bottom_baseline"
            else:
                # Fallback to bottom 15% point cloud line fitting
                thresh_y = np.percentile(ys, 85)
                bottom_pts = pts[pts[:, 1] >= thresh_y]
                if len(bottom_pts) >= 10:
                    line = cv2.fitLine(bottom_pts, cv2.DIST_L2, 0, 0.01, 0.01)
                    vx, vy = float(line[0][0]), float(line[1][0])
                    baseline_angle = float(np.degrees(np.arctan2(vy, vx)))
                    if baseline_angle > 90:
                        baseline_angle -= 180
                    elif baseline_angle < -90:
                        baseline_angle += 180
                    rot_deg = -baseline_angle
                    target_axis = "bottom_baseline"
                else:
                    self.mode = "bottom_upright"

        if self.mode == "bottom_upright" or (self.mode == "auto" and is_vertical):
            # Target is vertical axis (90 degrees, pointing down towards bottom)
            if major_angle >= 0:
                rot_deg = 90.0 - major_angle
            else:
                rot_deg = -90.0 - major_angle
            target_axis = "vertical"

        elif self.mode == "horizontal" or (self.mode == "auto" and not is_vertical):
            # Target is horizontal axis (0 degrees, parallel to bottom)
            if abs(major_angle) <= 90:
                rot_deg = -major_angle
            elif major_angle > 90:
                rot_deg = 180.0 - major_angle
            else:
                rot_deg = -180.0 - major_angle
            target_axis = "horizontal"

        elif self.mode == "snap_orthogonal":
            # Snap to closest 0, 90, 180, or 270
            snap_angles = [0.0, 90.0, 180.0, -90.0, -180.0]
            closest = min(snap_angles, key=lambda a: abs(major_angle - a))
            rot_deg = closest - major_angle
            target_axis = f"orthogonal_{int(closest)}"

        # 3. Apply rotation with canvas expansion to avoid clipping corners
        M = cv2.getRotationMatrix2D((cx, cy), -rot_deg, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        M[0, 2] += (new_w / 2.0) - cx
        M[1, 2] += (new_h / 2.0) - cy

        rotated_rgba = cv2.warpAffine(
            np_rgba, M, (new_w, new_h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0)
        )

        # 3b. For bottom_baseline mode, detect the leveled horizontal bottom cut line
        # and clip any residual drop shadow beneath it
        if target_axis == "bottom_baseline":
            rot_gray = cv2.cvtColor(rotated_rgba[:, :, :3], cv2.COLOR_RGB2GRAY)
            rot_alpha = rotated_rgba[:, :, 3]
            rys, rxs = np.where(rot_alpha > 127)
            if len(rys) > 50:
                max_ry = int(np.max(rys))
                edges = cv2.Canny(rot_gray, 40, 120)
                h_edges = np.sum((edges > 0) & (rot_alpha > 0), axis=1)
                search_start = max(0, max_ry - 65)
                search_range = h_edges[search_start:max_ry]
                if len(search_range) > 0 and np.max(search_range) >= 20:
                    cut_y = search_start + int(np.argmax(search_range))
                    rotated_rgba[cut_y + 1:, :, 3] = 0

        # 4. Inverted / Upside-down check (if vertical orientation)
        was_flipped_180 = False
        if self.detect_inverted and target_axis == "vertical":
            rot_alpha = rotated_rgba[:, :, 3]
            ry, rx = np.where(rot_alpha > 127)
            if len(ry) > 20:
                mid_y = np.median(ry)
                # Count mass in top half vs bottom half
                top_half_mass = np.sum(ry < mid_y)
                bot_half_mass = np.sum(ry >= mid_y)
                # If top half is significantly heavier/denser than bottom (inverted mascot), flip 180
                # In character stickers, the body/paws/base are almost always denser than ears/head
                if top_half_mass > bot_half_mass * 1.55:
                    rotated_rgba = cv2.rotate(rotated_rgba, cv2.ROTATE_180)
                    was_flipped_180 = True

        metadata = {
            "major_angle_deg": round(major_angle, 2),
            "applied_rotation_deg": round(rot_deg, 2),
            "target_axis": target_axis,
            "elongation_ratio": round(elongation, 3),
            "centroid": [round(cx, 1), round(cy, 1)],
            "was_flipped_180": was_flipped_180,
            "original_size": [w, h],
            "aligned_size": [new_w, new_h]
        }

        return Image.fromarray(rotated_rgba, "RGBA"), metadata
