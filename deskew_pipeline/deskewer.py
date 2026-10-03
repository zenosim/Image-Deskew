"""System 3: 3D Perspective Plane & Affine Deskewer
Solves 3D perspective distortion and skew for both:
1. Polygonal / Square / Rectangular stickers (4-corner DLT homography)
2. Die-cut / Organic character stickers (Spatial moment tensor & 3D parametric tilt solvers)
"""

import math
import numpy as np
import cv2
from PIL import Image
from dataclasses import dataclass
from typing import Tuple, Optional, List, Dict, Any


@dataclass
class DeskewResult:
    rgba: Image.Image
    homography: np.ndarray             # 3x3 transformation matrix
    detected_corners: Optional[np.ndarray] = None  # 4x2 corners if quad detected
    aspect_ratio: float = 1.0
    detected_angles: Optional[Dict[str, float]] = None # {'pitch', 'yaw', 'roll', 'shear'}


class PerspectiveDeskewer:
    def __init__(self, default_fov_deg: float = 50.0):
        self.default_fov_deg = default_fov_deg

    def deskew_quadrilateral(self, image: Image.Image, mask: np.ndarray, target_aspect_ratio: Optional[float] = None) -> DeskewResult:
        """
        Deskews a square, rectangular, or rounded-rect sticker by detecting its 4 perspective corners.
        """
        np_rgba = np.array(image)
        h_img, w_img = mask.shape

        # Extract external contour of the sticker
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise ValueError("No sticker contour found in mask for quadrilateral deskewing.")

        largest = max(contours, key=cv2.contourArea)

        # 1. Approximate 4-corner polygon using convex hull and approxPolyDP
        hull = cv2.convexHull(largest)
        peri = cv2.arcLength(hull, True)
        approx = None

        for eps in np.linspace(0.01, 0.08, 20):
            cand = cv2.approxPolyDP(hull, eps * peri, True)
            if len(cand) == 4:
                approx = cand
                break

        if approx is None or len(approx) != 4:
            # Fallback: compute minimum rotated rectangle or extreme points
            rect = cv2.minAreaRect(largest)
            box = cv2.boxPoints(rect)
            src_corners = np.array(box, dtype=np.float32)
        else:
            src_corners = approx.reshape(4, 2).astype(np.float32)

        # Order corners: Top-Left, Top-Right, Bottom-Right, Bottom-Left
        src_corners = self._order_points(src_corners)

        # Calculate side lengths
        tl, tr, br, bl = src_corners
        w_top = np.linalg.norm(tr - tl)
        w_bot = np.linalg.norm(br - bl)
        h_left = np.linalg.norm(bl - tl)
        h_right = np.linalg.norm(br - tr)

        avg_w = (w_top + w_bot) / 2.0
        avg_h = (h_left + h_right) / 2.0

        if target_aspect_ratio is not None:
            # Enforce user ratio (e.g. 1.0 for square)
            dst_w = int(max(avg_w, avg_h))
            dst_h = int(dst_w / target_aspect_ratio)
        else:
            # Keep estimated rectangular aspect ratio without arbitrary 1:1 snapping
            dst_w = max(10, int(round(avg_w)))
            dst_h = max(10, int(round(avg_h)))
            target_aspect_ratio = dst_w / dst_h

        # Destination corners (orthographic rectangle)
        dst_corners = np.array([
            [0, 0],
            [dst_w - 1, 0],
            [dst_w - 1, dst_h - 1],
            [0, dst_h - 1]
        ], dtype=np.float32)

        # Compute Homography: mapping from distorted src to rectified dst
        H = cv2.getPerspectiveTransform(src_corners, dst_corners)

        # Warp RGBA image
        warped = cv2.warpPerspective(
            np_rgba, H, (dst_w, dst_h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0)
        )

        return DeskewResult(
            rgba=Image.fromarray(warped, "RGBA"),
            homography=H,
            detected_corners=src_corners,
            aspect_ratio=target_aspect_ratio
        )

    def detect_flat_baseline_angle(self, mask: np.ndarray) -> Optional[float]:
        """
        Detects if the sticker has a dominant straight flat cut (e.g. car peeker baseline or sliced thighs/boots).
        Supports split/segmented cuts (e.g. separated thighs or clothing folds) via collinear angle clustering.
        Returns the angle in degrees needed to make the baseline horizontal, or None if no dominant straight baseline.
        """
        if mask is None or np.sum(mask > 0) == 0:
            return None

        h, w = mask.shape
        # Edge detection on the contour mask
        edges = cv2.Canny(mask, 50, 150)
        min_line_len = max(25, int(min(w, h) * 0.05))
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180,
            threshold=15,
            minLineLength=min_line_len,
            maxLineGap=20
        )

        if lines is None or len(lines) == 0:
            return None

        candidates = []
        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            avg_y = (y1 + y2) / 2.0
            # Sliced baselines are located in the lower portion of the character
            if avg_y < h * 0.55:
                continue
            dx = float(x2 - x1)
            dy = float(y2 - y1)
            angle = math.degrees(math.atan2(dy, dx))
            # Normalize angle to [-90, 90]
            if angle > 90:
                angle -= 180
            elif angle < -90:
                angle += 180

            # Baselines are tilted cuts within +/- 35 deg of horizontal
            if abs(angle) <= 35.0:
                length = math.hypot(dx, dy)
                # Strongly prioritize lines near the bottom
                y_weight = (avg_y / float(h)) ** 3
                candidates.append((angle, length * y_weight, length, min(x1, x2), max(x1, x2)))

        if not candidates:
            return None

        # Cluster candidate lines by angle (3-degree bins) to detect collinear/parallel cuts across gaps
        bins = {}
        for angle, weight, raw_len, xmin, xmax in candidates:
            key = round(angle / 3.0) * 3.0
            if key not in bins:
                bins[key] = {'total_w': 0.0, 'weighted_a': 0.0, 'raw_len': 0.0, 'xs': []}
            bins[key]['total_w'] += weight
            bins[key]['weighted_a'] += angle * weight
            bins[key]['raw_len'] += raw_len
            bins[key]['xs'].extend([xmin, xmax])

        best_bin = max(bins.values(), key=lambda b: b['total_w'])
        span_x = max(best_bin['xs']) - min(best_bin['xs'])
        min_span = max(40, int(w * 0.25))
        min_len = max(45, int(w * 0.12))

        if span_x >= min_span and best_bin['raw_len'] >= min_len:
            return float(best_bin['weighted_a'] / best_bin['total_w'])
        return None

    def deskew_flat_baseline(self, image: Image.Image, mask: np.ndarray) -> DeskewResult:
        """
        Deskews a peeker or flat-cut sticker by detecting its dominant straight baseline
        and leveling it to 0 degrees horizontal, expanding canvas to prevent clipping.
        """
        angle = self.detect_flat_baseline_angle(mask)
        if angle is None or abs(angle) < 0.2:
            return DeskewResult(rgba=image, homography=np.eye(3), detected_angles={"baseline_angle": 0.0, "correction": 0.0})

        np_rgba = np.array(image)
        h, w, _ = np_rgba.shape
        cx, cy = w / 2.0, h / 2.0
        # In image coordinates (Y-down), positive slope dy/dx > 0 tilts clockwise;
        # counter-clockwise OpenCV rotation (rot_correction = +angle) levels it to 0 deg.
        rot_correction = angle

        rad = math.radians(rot_correction)
        sin_a = abs(math.sin(rad))
        cos_a = abs(math.cos(rad))
        new_w = int(math.ceil(h * sin_a + w * cos_a))
        new_h = int(math.ceil(h * cos_a + w * sin_a))

        M_rot = cv2.getRotationMatrix2D((cx, cy), rot_correction, 1.0)
        M_rot[0, 2] += (new_w / 2.0) - cx
        M_rot[1, 2] += (new_h / 2.0) - cy
        H_rot = np.vstack([M_rot, [0, 0, 1]])

        warped = cv2.warpPerspective(
            np_rgba, H_rot, (new_w, new_h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0)
        )

        return DeskewResult(
            rgba=Image.fromarray(warped, "RGBA"),
            homography=H_rot,
            detected_angles={"baseline_angle": angle, "correction": rot_correction}
        )

    def deskew_diecut_moments(self, image: Image.Image, mask: np.ndarray) -> DeskewResult:
        """
        Deskews a die-cut character sticker using 2nd-order central spatial moments and shear analysis.
        Corrects in-plane rotation and affine shear while preserving the organic character shape,
        expanding canvas to prevent cropping tips.
        """
        np_rgba = np.array(image)
        h, w = mask.shape

        # Calculate central spatial moments of the character mask
        moments = cv2.moments(mask)
        if moments["m00"] == 0:
            return DeskewResult(rgba=image, homography=np.eye(3))

        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]

        mu20 = moments["mu20"] / moments["m00"]
        mu02 = moments["mu02"] / moments["m00"]
        mu11 = moments["mu11"] / moments["m00"]

        # Principal orientation angle
        theta = 0.5 * math.atan2(2.0 * mu11, mu20 - mu02)

        # Shear component: ratio of cross-moment
        denom = math.sqrt(max(1e-5, mu20 * mu02))
        shear = mu11 / denom

        # In-plane angle in degrees
        deg = math.degrees(theta)

        # Respect object aspect: if taller than wide (mu02 > mu20), align to vertical (90 deg)
        # If wider than tall (mu20 >= mu02), align to horizontal (0 deg)
        if mu02 > mu20:
            rot_correction = 90.0 - deg if deg >= 0 else -90.0 - deg
        else:
            rot_correction = -deg

        # Canvas expansion to avoid clipping tips/corners
        rad = math.radians(rot_correction)
        sin_a = abs(math.sin(rad))
        cos_a = abs(math.cos(rad))
        new_w = int(math.ceil(h * sin_a + w * cos_a))
        new_h = int(math.ceil(h * cos_a + w * sin_a))

        M_rot = cv2.getRotationMatrix2D((cx, cy), rot_correction, 1.0)
        M_rot[0, 2] += (new_w / 2.0) - cx
        M_rot[1, 2] += (new_h / 2.0) - cy
        H_rot = np.vstack([M_rot, [0, 0, 1]])

        # Apply transformation with expanded bounds
        warped = cv2.warpPerspective(
            np_rgba, H_rot, (new_w, new_h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0)
        )

        return DeskewResult(
            rgba=Image.fromarray(warped, "RGBA"),
            homography=H_rot,
            detected_angles={"roll": deg, "correction": rot_correction, "shear": shear}
        )

    def detect_character_skew_angle(self, mask: Any, rgb: Optional[np.ndarray] = None) -> Optional[float]:
        """
        Detects physical tilt/skew angle in anime character stickers and cutouts.
        Examines:
        1. Straight baseline cuts (peeker bottom, waist cut, sticker edge).
        2. Oriented minimum-bounding-box deviation.
        3. Bilateral horizontal facial / eye-line edges.
        Returns the rotation correction angle in degrees needed to make the artwork level.
        """
        if mask is None:
            return None

        if isinstance(mask, Image.Image):
            if mask.mode == "RGBA":
                mask = np.array(mask.split()[-1])
            else:
                mask = np.array(mask.convert("L"))
        elif isinstance(mask, np.ndarray) and mask.ndim == 3:
            if mask.shape[2] == 4:
                mask = mask[:, :, 3]
            else:
                mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)

        if not isinstance(mask, np.ndarray) or np.sum(mask > 0) == 0:
            return None

        h, w = mask.shape

        # Strategy 1: Check for bottom cut baseline (peeker or sliced sticker base)
        baseline_angle = self.detect_flat_baseline_angle(mask)
        if baseline_angle is not None and abs(baseline_angle) >= 0.3:
            return baseline_angle

        # Strategy 2: Oriented Minimum Bounding Box of the sticker contour
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > 500:
                rect = cv2.minAreaRect(largest)
                (cx, cy), (rw, rh), ang = rect
                if rw < rh:
                    dev = ang
                else:
                    dev = ang + 90.0 if ang < 0 else ang - 90.0

                # If within a reasonable tilt window (0.4 to 28 degrees)
                if 0.4 <= abs(dev) <= 28.0:
                    return -dev

        # Strategy 3: Horizontal line segments (eyes, bangs, collar lines)
        edges = cv2.Canny(mask, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=15, minLineLength=max(18, int(min(w, h) * 0.05)), maxLineGap=15)
        if lines is not None and len(lines) > 0:
            horiz_angles = []
            for x1, y1, x2, y2 in lines.reshape(-1, 4):
                dx = float(x2 - x1)
                dy = float(y2 - y1)
                ang = math.degrees(math.atan2(dy, dx))
                if ang > 90:
                    ang -= 180
                elif ang < -90:
                    ang += 180
                if abs(ang) <= 25.0:
                    horiz_angles.append(ang)
            if len(horiz_angles) >= 3:
                median_ang = float(np.median(horiz_angles))
                if abs(median_ang) >= 0.4:
                    return median_ang

        return None

    def deskew_character_artwork(self, image: Image.Image, mask: Optional[Any] = None, rgb: Optional[np.ndarray] = None) -> DeskewResult:
        """
        Physically deskews and levels a tilted anime character sticker or cutout.
        """
        if mask is None:
            mask = image
        angle = self.detect_character_skew_angle(mask, rgb)
        if angle is None or abs(angle) < 0.25:
            return DeskewResult(rgba=image, homography=np.eye(3), detected_angles={"skew_angle": 0.0, "correction": 0.0})

        return self.deskew_manual_angle(image, angle_deg=angle)

    def deskew_manual_angle(self, image: Image.Image, angle_deg: float) -> DeskewResult:
        """
        Physically deskews an image by an exact angle with seamless bounds expansion.
        """
        if abs(angle_deg) < 0.01:
            return DeskewResult(rgba=image, homography=np.eye(3), detected_angles={"angle": 0.0, "correction": 0.0})

        if image.mode != "RGBA":
            image = image.convert("RGBA")

        np_rgba = np.array(image)
        h, w, _ = np_rgba.shape
        cx, cy = w / 2.0, h / 2.0

        rot_correction = angle_deg
        rad = math.radians(rot_correction)
        sin_a = abs(math.sin(rad))
        cos_a = abs(math.cos(rad))
        new_w = int(math.ceil(h * sin_a + w * cos_a))
        new_h = int(math.ceil(h * cos_a + w * sin_a))

        M_rot = cv2.getRotationMatrix2D((cx, cy), rot_correction, 1.0)
        M_rot[0, 2] += (new_w / 2.0) - cx
        M_rot[1, 2] += (new_h / 2.0) - cy
        H_rot = np.vstack([M_rot, [0, 0, 1]])

        warped = cv2.warpPerspective(
            np_rgba, H_rot, (new_w, new_h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0)
        )

        return DeskewResult(
            rgba=Image.fromarray(warped, "RGBA"),
            homography=H_rot,
            detected_angles={"angle": angle_deg, "correction": rot_correction}
        )

    def deskew_parametric_3d(
        self,
        image: Image.Image,
        pitch_deg: float = 0.0,
        yaw_deg: float = 0.0,
        roll_deg: float = 0.0,
        scale: float = 1.0
    ) -> DeskewResult:
        """
        Parametric 3D un-tilting: inverts virtual 3D camera angles (Pitch, Yaw, Roll).
        Specifically built for digital mockups where stickers are tilted at known or estimated angles.
        """
        np_rgba = np.array(image)
        h, w, _ = np_rgba.shape

        # Reverse angles to un-tilt
        pitch = math.radians(-pitch_deg)
        yaw = math.radians(-yaw_deg)
        roll = math.radians(-roll_deg)

        Rx = np.array([
            [1, 0, 0],
            [0, math.cos(pitch), -math.sin(pitch)],
            [0, math.sin(pitch), math.cos(pitch)]
        ])
        Ry = np.array([
            [math.cos(yaw), 0, math.sin(yaw)],
            [0, 1, 0],
            [-math.sin(yaw), 0, math.cos(yaw)]
        ])
        Rz = np.array([
            [math.cos(roll), -math.sin(roll), 0],
            [math.sin(roll), math.cos(roll), 0],
            [0, 0, 1]
        ])
        R = Rz @ Ry @ Rx

        # Virtual camera matrix
        f = (w / 2.0) / math.tan(math.radians(self.default_fov_deg / 2.0))
        K = np.array([
            [f, 0, w / 2.0],
            [0, f, h / 2.0],
            [0, 0, 1]
        ], dtype=np.float64)

        # Homography: H = K * R * K^-1
        H = K @ R @ np.linalg.inv(K)
        H /= H[2, 2]

        # Apply scaling if requested
        if scale != 1.0:
            S = np.array([[scale, 0, (1 - scale) * w / 2], [0, scale, (1 - scale) * h / 2], [0, 0, 1]])
            H = S @ H

        warped = cv2.warpPerspective(
            np_rgba, H, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0)
        )

        return DeskewResult(
            rgba=Image.fromarray(warped, "RGBA"),
            homography=H,
            detected_angles={"pitch": pitch_deg, "yaw": yaw_deg, "roll": roll_deg}
        )

    def apply_horizontal_skew(self, image: Image.Image, skew_deg: float) -> Image.Image:
        """
        Applies a horizontal shear / skew angle to rectify slanted perspective or customize angle.
        Matches CSS skewX(skew_deg) with symmetric center preservation and non-clipping padding.
        """
        if abs(skew_deg) < 1e-3:
            return image
        np_rgba = np.array(image.convert("RGBA"))
        h, w = np_rgba.shape[:2]
        theta = math.radians(skew_deg)
        tan_theta = math.tan(theta)

        # Calculate symmetric width expansion so slanted corners don't get clipped
        pad_w = int(math.ceil(abs(h * tan_theta)))
        new_w = w + pad_w

        # Center stays fixed: x' = x + tan_theta * (y - h/2) + pad_w / 2
        M = np.float32([
            [1, tan_theta, (pad_w / 2.0) - tan_theta * (h / 2.0)],
            [0, 1, 0]
        ])
        warped = cv2.warpAffine(
            np_rgba, M, (new_w, h),
            flags=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0)
        )
        return Image.fromarray(warped, "RGBA")

    def apply_vertical_skew(self, image: Image.Image, skew_deg: float) -> Image.Image:
        """
        Applies a vertical shear / skew angle to rectify slanted perspective or customize angle.
        Matches CSS skewY(skew_deg) with symmetric center preservation and non-clipping padding.
        """
        if abs(skew_deg) < 1e-3:
            return image
        np_rgba = np.array(image.convert("RGBA"))
        h, w = np_rgba.shape[:2]
        theta = math.radians(skew_deg)
        tan_theta = math.tan(theta)

        # Calculate symmetric height expansion so slanted corners don't get clipped
        pad_h = int(math.ceil(abs(w * tan_theta)))
        new_h = h + pad_h

        # Center stays fixed: y' = y + tan_theta * (x - w/2) + pad_h / 2
        M = np.float32([
            [1, 0, 0],
            [tan_theta, 1, (pad_h / 2.0) - tan_theta * (w / 2.0)]
        ])
        warped = cv2.warpAffine(
            np_rgba, M, (w, new_h),
            flags=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0)
        )
        return Image.fromarray(warped, "RGBA")

    def apply_skew(self, image: Image.Image, h_skew_deg: float = 0.0, v_skew_deg: float = 0.0) -> Image.Image:
        """
        Applies both horizontal and vertical skew shears sequentially with non-clipping padding.
        """
        out = image
        if abs(h_skew_deg) >= 1e-3:
            out = self.apply_horizontal_skew(out, h_skew_deg)
        if abs(v_skew_deg) >= 1e-3:
            out = self.apply_vertical_skew(out, v_skew_deg)
        return out

    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        """
        Sorts 4 points in consistent clockwise order:
        Top-Left, Top-Right, Bottom-Right, Bottom-Left.
        Robust to arbitrary 2D/3D rotations (0-360 degrees).
        """
        pts = pts.astype(np.float32)
        center = pts.mean(axis=0)

        # Find the point closest to top-left (smallest distance to min_x, min_y)
        min_xy = pts.min(axis=0)
        dist_to_tl = np.sum((pts - min_xy) ** 2, axis=1)
        tl_idx = int(np.argmin(dist_to_tl))
        tl = pts[tl_idx]

        ref_angle = math.atan2(tl[1] - center[1], tl[0] - center[0])

        def clockwise_diff(pt):
            a = math.atan2(pt[1] - center[1], pt[0] - center[0])
            diff = (ref_angle - a) % (2.0 * math.pi)
            return diff

        sorted_indices = sorted(range(4), key=lambda i: clockwise_diff(pts[i]))
        return pts[sorted_indices]
