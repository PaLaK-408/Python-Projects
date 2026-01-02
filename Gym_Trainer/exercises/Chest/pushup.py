from utils.pose_math import calculate_angle

LEFT_SHOULDER = 11
LEFT_ELBOW = 13
LEFT_WRIST = 15

def run(landmarks, state):
    shoulder = landmarks[LEFT_SHOULDER]
    elbow = landmarks[LEFT_ELBOW]
    wrist = landmarks[LEFT_WRIST]

    if elbow.visibility < 0.5:
        return None

    angle = calculate_angle(shoulder, elbow, wrist)

    if angle < 90:
        state["down"] = True

    if angle > 160 and state["down"]:
        state["count"] += 1
        state["down"] = False

    return angle, (shoulder, elbow, wrist)
