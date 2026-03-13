import cv2
import pickle
import numpy as np
from core.detector import detect_faces
from core.matcher import match_face
from core.decision import decide
from config import FACE_DISTANCE_THRESHOLD

with open("encodings.pkl", "rb") as f:
    db = pickle.load(f)

known_encodings = np.array(db["encodings"])
known_names = db["names"]

video = cv2.VideoCapture(0)

while True:
    ret, frame = video.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    locations, encodings = detect_faces(rgb)

    for enc, loc in zip(encodings, locations):
        dist, idx = match_face(known_encodings, enc)
        decision = decide(dist, True, FACE_DISTANCE_THRESHOLD)

        label = known_names[idx] if decision == "ACCEPT" else "Unknown"

        t, r, b, l = loc
        cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{label} | dist={dist:.2f}",
            (l, t - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    cv2.imshow("Industrial Face Biometrics", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()