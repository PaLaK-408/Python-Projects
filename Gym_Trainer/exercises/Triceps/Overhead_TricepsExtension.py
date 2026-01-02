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

    if angle < 60:
        state["down"] = True

    if angle > 160 and state["down"]:
        state["count"] += 1
        state["down"] = False

    return angle, (s, e, w)
