import cv2
import tkinter as tk
from PIL import Image, ImageTk

from utils.pose_engine import PoseEngine
from utils.draw_utils import draw_point, draw_line, draw_angle
from exercises.Legs import Squat, lunge, WallSit, GluteBridge, CalfRaise
from exercises.Biceps import Bicep_curl, Hammer_curl, Isometric_hold, concentration_curl, Alternating_curl
from exercises.Triceps import DiamondPush_Ups, Kickbacks,Isometric_Hold,Overhead_TricepsExtension,Triceps_Dips
from exercises.Shoulder import arm_circles, arm_raise, lateral_raise, shoulder_press, shrugs
from exercises.Chest import arm_fly, chest_squeeze, pushup, wide_pushup, incline_pushup


EXERCISE_MAP = {
    "Squat": Squat,
    "Lunge": lunge,
    "Wall Sit": WallSit,
    "Glute Bridge": GluteBridge,
    "Calf Raise": CalfRaise,

    #Arm Exercises
    "Bicep_curl": Bicep_curl,
    "Hammer_curl": Hammer_curl,
    "Isometric_hold": Isometric_hold,
    "concentration_curl": concentration_curl,
    "Alternating_curl": Alternating_curl,

    #Triceps Exercises
    "DiamondPush_Ups": DiamondPush_Ups,
    "Kickbacks": Kickbacks,
    "Isometric_Hold": Isometric_Hold,
    "Overhead_TricepsExtension": Overhead_TricepsExtension,
    "Triceps_Dips": Triceps_Dips,

    #Shoulder
    "arm_raise": arm_raise,
    "Lateral Raise": lateral_raise,
    "Arm Circles": arm_circles,
    "Shoulder Press": shoulder_press,
    "Shrugs": shrugs,

    # Chest
    "Push-ups": pushup,
    "Wide Push-ups": wide_pushup,
    "Incline Push-ups": incline_pushup,
    "Chest Squeeze": chest_squeeze,
    "Arm Fly": arm_fly

}

class WorkoutScreen:
    def __init__(self, exercise):
        self.exercise = exercise
        self.state = {
            "count": 0,
            "down": False,
            "up": False,
            "hold": 0
        }

        self.root = tk.Tk()
        self.root.title(exercise)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.label = tk.Label(self.root)
        self.label.pack()

        self.cap = cv2.VideoCapture(0)
        self.pose = PoseEngine()

        self.update()
        self.root.mainloop()

    def update(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        landmarks = self.pose.process(frame)

        if landmarks:
            result = EXERCISE_MAP[self.exercise].run(landmarks, self.state)

            if result:
                angle, (a, b, c) = result

                # 🔴 Draw joints
                draw_point(frame, a)
                draw_point(frame, b)
                draw_point(frame, c)

                # 🔵 Draw bones
                draw_line(frame, a, b)
                draw_line(frame, b, c)

                # 🟡 Draw angle text
                draw_angle(frame, angle, b)

        cv2.putText(
            frame,
            f"Reps: {self.state['count']}",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),  
            2
        )

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = ImageTk.PhotoImage(Image.fromarray(frame))
        self.label.configure(image=img)
        self.label.image = img

        self.root.after(10, self.update)

    def on_close(self):
            self.cap.release()
            self.root.destroy()

