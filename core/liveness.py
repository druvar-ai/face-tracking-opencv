import numpy as np

def depth_liveness(depth_frame, bbox):
    top, right, bottom, left = bbox
    region = depth_frame[top:bottom, left:right]
    values = region[region > 0]

    if len(values) < 500:
        return False

    return np.std(values) > 15