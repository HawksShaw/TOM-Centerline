import numpy as np

def make_endpoints(points):
    max_dist = 0
    start, end = 0, 0
    n = len(points)
    for i in range(n - 1):
        dists = np.linalg.norm(points[i+1:] - points[i], axis=1)
        if dists.size == 0:
            continue
        j = np.argmax(dists)
        if dists[j] > max_dist:
            max_dist = dists[j]
            start, end = i, i + 1 + j
    return start, end