from utils.pose_math import calculate_angle

RIGHT_SHOULDER = 12
RIGHT_ELBOW = 14
RIGHT_WRIST = 16

def run(landmarks, state):
    shoulder = landmarks[RIGHT_SHOULDER]
    elbow = landmarks[RIGHT_ELBOW]
    wrist = landmarks[RIGHT_WRIST]

    if shoulder.visibility < 0.5 or elbow.visibility < 0.5:
        return None

    angle = calculate_angle(shoulder, elbow, wrist)

    if angle < 70:
        state["hold"] = True

    if angle > 140 and state.get("hold"):
        state["count"] += 1
        state["hold"] = False

    return angle, (shoulder, elbow, wrist)
