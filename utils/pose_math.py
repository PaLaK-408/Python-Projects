import math

def calculate_angle(a, b, c):
    a = (a.x, a.y)
    b = (b.x, b.y)
    c = (c.x, c.y)

    radians = math.atan2(c[1]-b[1], c[0]-b[0]) - \
              math.atan2(a[1]-b[1], a[0]-b[0])

    angle = abs(radians * 180 / math.pi)
    if angle > 180:
        angle = 360 - angle

    return angle
