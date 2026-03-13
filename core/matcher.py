import numpy as np

def match_face(known_encodings, query_encoding):
    distances = np.linalg.norm(known_encodings - query_encoding, axis=1)
    idx = distances.argmin()
    return distances[idx], idx