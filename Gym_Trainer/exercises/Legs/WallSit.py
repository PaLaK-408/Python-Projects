from utils.pose_math import calculate_angle

LEFT_HIP = 23
LEFT_KNEE = 25
LEFT_ANKLE = 27

def run(landmarks, state):
    hip = landmarks[LEFT_HIP]
    knee = landmarks[LEFT_KNEE]
    ankle = landmarks[LEFT_ANKLE]

    if hip.visibility < 0.5:
        return None

    angle = calculate_angle(hip, knee, ankle)

    if 80 <= angle <= 100:
        state["hold"] += 1

    return angle, (hip, knee, ankle)

