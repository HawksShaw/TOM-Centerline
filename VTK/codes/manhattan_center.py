import numpy as np
from sklearn.cluster import DBSCAN
from scipy.ndimage import gaussian_filter1d

def compute_slice_centerline(points, axis=2, dz=1.0, eps=0.5, min_samples=5, max_jump=10.0, sigma=1.0):
    """
    1. Initialize slicing along an axis.
    2. Find points in a slice
    3. Cluster the points using DBSCAN
    4. Compute the centroids of the main cluster.
    5. Append the chosen centroid to centerline.
    6. Smoothen centerline using gaussian smoothing.
    """
    centerline = []
    coord = points[:, axis] #axis = 2 to go through transverse planes (Z)
    zmin, zmax = np.min(coord), np.max(coord) #zmin, zmax because we're on the Z plane :v
    slices = np.arange(zmin, zmax, dz) #dz should be the same as the voxel depth but I'm too lazy to check so we go with 1.0
    prev_center = None
    axes = [i for i in range(3) if i != axis] #We need the other two axes stored for later clustering
    for slice in slices:
        mask = (coord >= slice) & (coord < slice + dz)
        slice_pts = points[mask]
        if len(slice_pts) == 0:
            continue
        y = slice_pts[:, axes]
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(y)
        labels = clustering.labels_
        valid_labels = [label for label in np.unique(labels) if label != -1]
        if not valid_labels:
            continue
        centroids = []
        for label in valid_labels:
            cluster_pts = slice_pts[labels == label]
            center = np.mean(cluster_pts, axis=0)
            centroids.append(center)
        centroids = np.array(centroids)
        if prev_center is None:
            sizes = [np.sum(labels == label) for label in valid_labels]
            chosen = np.argmax(sizes)
        else:
            dists = np.linalg.norm(centroids - prev_center, axis=1)
            close_clusters = np.where(dists < max_jump)[0]
            if len(close_clusters) > 0:
                chosen = close_clusters[np.argmin(dists[close_clusters])]
        chosen_center = centroids[chosen]
        centerline.append(chosen_center)
        prev_center = chosen_center
    centerline = np.array(centerline)
    if sigma > 0 and len(centerline) > 1:
        centerline = gaussian_filter1d(centerline, sigma=sigma, axis=0)
    return centerline