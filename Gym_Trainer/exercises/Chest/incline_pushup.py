LEFT_SHOULDER = 11
LEFT_HIP = 23

def run(landmarks, state):
    shoulder = landmarks[LEFT_SHOULDER]
    hip = landmarks[LEFT_HIP]

    if shoulder.visibility < 0.5:
        return None

    y_diff = hip.y - shoulder.y

    if y_diff > 0.20:
        state["down"] = True

    if y_diff < 0.15 and state["down"]:
        state["count"] += 1
        state["down"] = False

    return y_diff, (shoulder, hip, hip)

