import math


def get_body_angle(s_x, s_y, h_x, h_y) -> float:
    """
    Arguments:
        s_x: Shoulder x
        s_y: Shoulder y
        h_x: Hip x
        h_y: Hip y
    """
    # Default values if keypoints are missing
    angle_deg = -1.0

    # Ensure both shoulder and hip are validly detected
    if s_x > 0 and s_y > 0 and h_x > 0 and h_y > 0:
        # --- 1. Calculate the vector differences ---
        dx = h_x - s_x
        dy = h_y - s_y  # Note: Image coordinates grow downwards

        # --- 2. Calculate angle relative to vertical axis ---
        # Using atan2(dx, dy) automatically aligns 0 degrees to a purely vertical line
        angle_rad = math.atan2(dx, dy)
        angle_deg = abs(math.degrees(angle_rad))
    
    return angle_deg


def get_box_ratio(box_width, box_heigh) -> float:
    return box_heigh / box_width