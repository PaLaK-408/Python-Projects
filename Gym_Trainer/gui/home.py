import tkinter as tk

def open_exercises(body_part, root):
    root.destroy()
    from gui.exercise_menu import ExerciseMenu
    ExerciseMenu(body_part)

class Home:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title(" Gym Trainer,Made by Palak ")

        tk.Label(self.root, text="Select Body Part",
                 font=("Arial", 20)).pack(pady=20)

        for part in ["Legs", "Biceps", "Triceps","Shoulder","Chest"]:
            tk.Button(
                self.root,
                text=part,
                width=20,
                height=2,
                command=lambda p=part: open_exercises(p, self.root)
            ).pack(pady=10)

        self.root.mainloop()
