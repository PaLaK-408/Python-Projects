from utils.pose_math import calculate_angle

LEFT_HIP = 23
LEFT_KNEE = 25
LEFT_ANKLE = 27

def run(landmarks, state):
    hip = landmarks[LEFT_HIP]
    knee = landmarks[LEFT_KNEE]
    ankle = landmarks[LEFT_ANKLE]

    # Visibility safety
    if hip.visibility < 0.5 or knee.visibility < 0.5:
        return None

    angle = calculate_angle(hip, knee, ankle)

    if angle < 90:
        state["down"] = True

    if angle > 160 and state["down"]:
        state["count"] += 1
        state["down"] = False

    return angle, (hip, knee, ankle)

