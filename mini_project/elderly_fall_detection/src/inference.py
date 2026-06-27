from utils import get_body_angle, get_box_ratio


def infer_pose_status(box, person_kp, track_history):
    bb_x1, bb_y1, bb_x2, bb_y2 = box

    # Extract Left (Index 5) and Right (Index 6) shoulders
    left_shoulder = person_kp[5]   # [x, y]
    right_shoulder = person_kp[6]  # [x, y]
    left_hip = person_kp[11]       # [x, y]
    right_hip = person_kp[12]      # [x, y]
    left_knee = person_kp[13]
    right_knee = person_kp[14]
    left_ankle = person_kp[15]
    right_ankle = person_kp[16]

    # Unpack coordinates
    ls_x, ls_y = int(left_shoulder[0]), int(left_shoulder[1])
    rs_x, rs_y = int(right_shoulder[0]), int(right_shoulder[1])
    lh_x, lh_y = int(left_hip[0]), int(left_hip[1])
    rh_x, rh_y = int(right_hip[0]), int(right_hip[1])
    lk_x, lk_y = int(left_knee[0]), int(left_knee[1])
    rk_x, rk_y = int(right_knee[0]), int(right_knee[1])
    la_x, la_y = int(left_ankle[0]), int(left_ankle[1])
    ra_x, ra_y = int(right_ankle[0]), int(right_ankle[1])

    body_angle = get_body_angle(
        (ls_x, ls_y, lh_x, lh_y,),
        (rs_x, rs_y, rh_x, rh_y,),
    )

    box_ratio = get_box_ratio(bb_x2 - bb_x1, bb_y2 - bb_y1)
    is_lying = False

    if body_angle > 65 and box_ratio < 1:
        is_lying = True
    else:
        shoulder_y = ls_y if ls_y > 0 else rs_y
        if shoulder_y > 0:
            for _y in (lk_y, la_y, rk_y, ra_y,):
                if _y > 0 and _y <= shoulder_y:
                    is_lying = True
                    break
    
    if is_lying is False and box_ratio < 1:
        is_lying = track_history[-1]['is_lying'] if track_history else True
    
    return is_lying, body_angle, box_ratio


def infer_fall(history, is_lying):
    is_falling_event = False
    alert_score = 0
    
    # We need a minimum window of history to calculate structural drops accurately
    if len(history) >= 15:
        
        # 1. Fetch historical states (e.g., 5-7 frames ago)
        current_state = history[-1]
        past_state = history[-4]  # 3-4 frames ago for instantaneous speed delta
        context_state = history[-14] # Reach back 14 frames (~0.55s) to guarantee standing context
        
        # 2. Calculate Hip Drop Velocity (Y increases downwards in pixels)
        if current_state['hip_y'] > 0 and past_state['hip_y'] > 0:
            hip_drop_velocity = current_state['hip_y'] - past_state['hip_y']
        else:
            hip_drop_velocity = 0
            
        # 3. Calculate Angular Velocity (Using absolute value to track rotation intensity)
        angular_velocity = abs(current_state['body_angle'] - past_state['body_angle'])

        # Tracks how fast the box ratio changes from tall/upright to wide/flat
        box_ratio_change = abs(past_state['box_ratio'] - current_state['box_ratio'])

        # Condition A: Standard high-fidelity pose velocity triggers
        kp_triggered = (hip_drop_velocity > 18 or angular_velocity > 22)

        # Condition B: Fallback bounding box collapse (Used if joints are missing or tracking behaves wildly)
        # A positive box_ratio_change > 0.4 implies the box flattened significantly in < 0.2 seconds
        bbox_triggered = (box_ratio_change > 0.45 and current_state['box_ratio'] < 0.95)
        
        # 4. Establish Pre-Event Context
        # Was the person explicitly upright/standing 10 frames ago?
        was_upright = context_state['is_lying'] is False

        # print((hip_drop_velocity, angular_velocity, box_ratio_change, was_upright,))

        if (kp_triggered or bbox_triggered) and was_upright:
            # We caught them in mid-air crashing downwards!
            current_state['fall_latched'] = True 
        else:
            # Carry over the latched flag from the previous frame if it exists
            current_state['fall_latched'] = history[-2].get('fall_latched', False) if len(history) > 1 else False
        
        if is_lying:
            # Look back across the last 5 frames to make sure this wasn't an accidental single-frame glitch
            recent_latch_activity = any(h.get('fall_latched', False) for h in tuple(history)[-5:])
            
            # If they are down on the ground, check if they got there via a latched crash spike
            if current_state.get('fall_latched', False) or recent_latch_activity:
                is_falling_event = True
                alert_score = 90
            else:
                is_falling_event = False
                alert_score = 20  # Controlled transition / already lying down
        else:
            alert_score = 0
    
    return is_falling_event, alert_score