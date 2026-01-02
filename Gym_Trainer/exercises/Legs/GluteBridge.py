from utils.pose_math import calculate_angle

LEFT_SHOULDER = 11
LEFT_HIP = 23
LEFT_KNEE = 25

def run(landmarks, state):
    shoulder = landmarks[LEFT_SHOULDER]
    hip = landmarks[LEFT_HIP]
    knee = landmarks[LEFT_KNEE]

    if hip.visibility < 0.5:
        return None

    angle = calculate_angle(shoulder, hip, knee)

    if angle > 160:
        state["up"] = True
    if angle < 120 and state["up"]:
        state["count"] += 1
        state["up"] = False

    return angle, (shoulder, hip, knee)
