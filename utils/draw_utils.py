import cv2

def to_pixel(landmark, width, height):
    return int(landmark.x * width), int(landmark.y * height)

def draw_point(frame, landmark, color=(0, 255, 0), r=6):
    h, w, _ = frame.shape
    cx, cy = to_pixel(landmark, w, h)
    cv2.circle(frame, (cx, cy), r, color, -1)

def draw_line(frame, lm1, lm2, color=(255, 0, 0), thickness=2):
    h, w, _ = frame.shape
    x1, y1 = to_pixel(lm1, w, h)
    x2, y2 = to_pixel(lm2, w, h)
    cv2.line(frame, (x1, y1), (x2, y2), color, thickness)

def draw_angle(frame, angle, landmark, color=(0, 255, 255)):
    h, w, _ = frame.shape
    x, y = to_pixel(landmark, w, h)
    cv2.putText(
        frame,
        f"{int(angle)}°",
        (x - 20, y - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2
    )
