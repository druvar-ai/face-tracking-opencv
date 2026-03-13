import os
import pickle
import face_recognition
from core.encoder import encode_person

DATA_DIR = "D:\coding\Python\Industrial_level_face_recognition\data\enrolled_faces"
DB_FILE = "encodings.pkl"

db_encodings = []
db_names = []

for person in os.listdir(DATA_DIR):
    person_dir = os.path.join(DATA_DIR, person)
    if not os.path.isdir(person_dir):
        continue

    images = []
    for img_name in os.listdir(person_dir):
        img_path = os.path.join(person_dir, img_name)
        images.append(face_recognition.load_image_file(img_path))

    encoding = encode_person(images)
    if encoding is not None:
        db_encodings.append(encoding)
        db_names.append(person)
        print(f"[OK] Enrolled {person}")
    else:
        print(f"[SKIP] {person}")

with open(DB_FILE, "wb") as f:
    pickle.dump(
        {"encodings": db_encodings, "names": db_names}, f
    )

print("Enrollment completed.")