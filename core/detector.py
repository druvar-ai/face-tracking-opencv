import face_recognition

def detect_faces(rgb_frame):
    locations = face_recognition.face_locations(rgb_frame)
    encodings = face_recognition.face_encodings(rgb_frame, locations)
    return locations, encodings