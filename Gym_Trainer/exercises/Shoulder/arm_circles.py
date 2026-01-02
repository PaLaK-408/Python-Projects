RIGHT_WRIST = 16

def run(landmarks, state):
    wrist = landmarks[RIGHT_WRIST]

    if wrist.visibility < 0.5:
        return None

    if "prev_y" not in state:
        state["prev_y"] = wrist.y
        return None

    if wrist.y < state["prev_y"] - 0.03:
        state["count"] += 1

    state["prev_y"] = wrist.y
    return wrist.y, (wrist,)
