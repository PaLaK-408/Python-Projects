LEFT_WRIST = 15
RIGHT_WRIST = 16

def run(landmarks, state):
    left = landmarks[LEFT_WRIST]
    right = landmarks[RIGHT_WRIST]

    if left.visibility < 0.5 or right.visibility < 0.5:
        return None

    distance = abs(left.x - right.x)

    if distance < 0.05:
        state["down"] = True

    if distance > 0.10 and state["down"]:
        state["count"] += 1
        state["down"] = False

    return distance, (left, right, right)

