LEFT_WRIST = 15
RIGHT_WRIST = 16

def run(landmarks, state):
    lw = landmarks[LEFT_WRIST]
    rw = landmarks[RIGHT_WRIST]

    if lw.visibility < 0.5 or rw.visibility < 0.5:
        return None

    dist = abs(lw.x - rw.x)

    if dist > 0.35:
        state["down"] = True

    if dist < 0.15 and state["down"]:
        state["count"] += 1
        state["down"] = False

    return dist, (lw, rw, rw)

