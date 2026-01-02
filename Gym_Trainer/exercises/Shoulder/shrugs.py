LEFT_SHOULDER = 11

def run(landmarks, state):
    shoulder = landmarks[LEFT_SHOULDER]

    if shoulder.visibility < 0.5:
        return None

    y = shoulder.y

    if y > 0.55:
        state["down"] = True

    if y < 0.50 and state["down"]:
        state["count"] += 1
        state["down"] = False

    return y, (shoulder,)