LEFT_ANKLE = 27
LEFT_FOOT_INDEX = 31

def run(landmarks, state):
    ankle = landmarks[LEFT_ANKLE]
    foot = landmarks[LEFT_FOOT_INDEX]

    if ankle.visibility < 0.5:
        return None

    height = foot.y - ankle.y

    if height < -0.03:
        state["up"] = True

    if height > 0 and state["up"]:
        state["count"] += 1
        state["up"] = False

    return height * 100, (ankle, foot, ankle)
