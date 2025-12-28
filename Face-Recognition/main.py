import cv2
import numpy as np
import csv
from datetime import datetime
import os

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


# LBPH Face Recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

labels = {}
faces = []
ids = []
current_id = 0

# Training data
dataset_path = "faces"

for person in os.listdir(dataset_path):
    labels[current_id] = person
    person_path = os.path.join(dataset_path, person)

    for image_name in os.listdir(person_path):
        img_path = os.path.join(person_path, image_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        faces.append(img)
        ids.append(current_id)

    current_id += 1

recognizer.train(faces, np.array(ids))

# Attendance setup
students = list(labels.values())

now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

f = open(f"{current_date}.csv", "w", newline="")
writer = csv.writer(f)
writer.writerow(["Name", "Time"])

video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detected_faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in detected_faces:
        roi_gray = gray[y:y+h, x:x+w]
        id_, confidence = recognizer.predict(roi_gray)

        name = "Unknown"
        if confidence < 80:   # lower = better match
            name = labels[id_]

            if name in students:
                students.remove(name)
                time_now = datetime.now().strftime("%H:%M:%S")
                writer.writerow([name, time_now])

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, name, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Attendance System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
f.close()




