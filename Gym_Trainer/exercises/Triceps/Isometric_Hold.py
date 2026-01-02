from utils.pose_math import calculate_angle

LEFT_SHOULDER = 11
LEFT_ELBOW = 13
LEFT_WRIST = 15

def run(landmarks, state):
    s = landmarks[LEFT_SHOULDER]
    e = landmarks[LEFT_ELBOW]
    w = landmarks[LEFT_WRIST]

    if e.visibility < 0.5:
        return None

    angle = calculate_angle(s, e, w)

    if angle > 150:
        state["hold"] += 1
    else:
        state["hold"] = 0

    if state["hold"] > 40:
        state["count"] += 1
        state["hold"] = 0

    return angle, (s, e, w)
