import tkinter as tk

EXERCISES = {
    "Legs": ["Squat", "Lunge", "Wall Sit", "Glute Bridge", "Calf Raise"],
    "Biceps": ["Bicep_curl","Hammer_curl","concentration_curl","Isometric_hold","Alternating_curl"],
    "Triceps": ["DiamondPush_Ups","Isometric_Hold","Kickbacks","Overhead_TricepsExtension","Triceps_Dips"],
    "Shoulder": ["Arm Raise","Lateral Raise","Arm Circles","Shoulder Press","Shrugs"],
    "Chest": ["Arm Fly","Chest Squeeze","Incline Push-ups","Push-ups","Wide Push-ups" ]
}


class ExerciseMenu:
    def __init__(self, body_part):
        self.root = tk.Tk()
        self.root.title(body_part + " Exercises")

        tk.Label(self.root, text=body_part,
                 font=("Arial", 20)).pack(pady=20)

        for ex in EXERCISES[body_part]:
            tk.Button(
                self.root,
                text=ex,
                width=25,
                height=2,
                command=lambda e=ex: self.start_workout(e)
            ).pack(pady=5)

        self.root.mainloop()

    def start_workout(self, exercise):
        self.root.destroy()
        from gui.workout_screen import WorkoutScreen
        WorkoutScreen(exercise)
