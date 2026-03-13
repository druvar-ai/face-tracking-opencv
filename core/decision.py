def decide(identity_distance, live, threshold):
    if identity_distance < threshold and live:
        return "ACCEPT"
    return "REJECT"