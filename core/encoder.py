import face_recognition
import numpy as np

def encode_person(images):
    encodings = []
    for img in images:
        e = face_recognition.face_encodings(img)
        if len(e) == 1:
            encodings.append(e[0])

    if len(encodings) < 2:
        return None

    return np.mean(encodings, axis=0)