# pipeline/staff.py

import cv2
import numpy as np

# These are OpenCV hue units (0-179), NOT degrees (0-360).
# Under STORE_BLR_002 in store_layout.json, the range is [95, 115].
STAFF_HUE_RANGE = (95, 115)   
STAFF_SAT_MIN   = 0.40        # rules out grey/white non-uniforms
STAFF_PIX_PCT   = 0.60        # 60% of upper-body pixels must match hue range


def classify_staff(
    crop_bgr: np.ndarray,
    hue_range: tuple = STAFF_HUE_RANGE
) -> tuple[bool, float]:
    """
    Classify whether a bounding box crop belongs to a staff member.

    Args:
        crop_bgr: Upper-body crop in BGR format (top 40% of bounding box).
        hue_range: (low, high) in OpenCV hue units (0-179).

    Returns:
        (is_staff, confidence) where confidence is the fraction of
        upper-body pixels that fall within the hue range.
        Never raises — returns (False, 0.0) on any error.
    """
    try:
        if crop_bgr is None or crop_bgr.size == 0:
            return False, 0.0
            
        h, w, _ = crop_bgr.shape
        if h == 0 or w == 0:
            return False, 0.0

        hsv  = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
        
        # We need to build a mask for Hue range and Saturation threshold
        h_min, h_max = hue_range
        # OpenCV Hue ranges 0-179, Saturation 0-255, Value 0-255
        # SAT_MIN = 0.40 * 255 = 102
        # VAL_MIN = 50 (to rule out very dark shadows)
        lower_bound = np.array([h_min, int(STAFF_SAT_MIN * 255), 50])
        upper_bound = np.array([h_max, 255, 255])
        
        # Handle wraparound if necessary
        if h_min <= h_max:
            mask = cv2.inRange(hsv, lower_bound, upper_bound)
        else:
            # Hue wraparound
            mask1 = cv2.inRange(hsv, np.array([h_min, int(STAFF_SAT_MIN * 255), 50]), np.array([179, 255, 255]))
            mask2 = cv2.inRange(hsv, np.array([0, int(STAFF_SAT_MIN * 255), 50]), np.array([h_max, 255, 255]))
            mask = cv2.bitwise_or(mask1, mask2)
            
        conf = float(mask.sum() / 255) / max(1, mask.size)
        return conf >= STAFF_PIX_PCT, round(conf, 3)
    except Exception:
        return False, 0.0
