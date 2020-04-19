import numpy as np


def haversine_dist(x, y):
    R = 6371.0088
    x, y = np.radians(x), np.radians(y)
    dlat = y[:, 0] - x[:, 0]
    dlon = y[:, 1] - x[:, 1]
    a = np.sin(dlat / 2) ** 2 + np.cos(x[:, 0]) * np.cos(y[:, 0]) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(a ** 0.5, (1 - a) ** 0.5)
    d = R * c
    return d