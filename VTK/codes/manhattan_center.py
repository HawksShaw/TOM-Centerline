import numpy as np
from sklearn.cluster import DBSCAN
from scipy.ndimage import gaussian_filter1d
import vtk

def filter_points_inside_mesh(points, polydata):
    enclosed = vtk.vtkSelectEnclosedPoints()
    enclosed.SetSurfaceData(polydata)
    vtk_points = vtk.vtkPoints()
    for pt in points:
        vtk_points.InsertNextPoint(pt)
    test_poly = vtk.vtkPolyData()
    test_poly.SetPoints(vtk_points)
    enclosed.SetInputData(test_poly)
    enclosed.Update()
    mask = [enclosed.IsInside(i) for i in range(points.shape[0])]
    return points[mask]

def compute_slice_centerline(points, polydata=None, axis=2, dz=1.0, eps=0.5, min_samples=5, max_jump=10.0, sigma=1.0):
    centerline = []
    coord = points[:, axis]
    zmin, zmax = np.min(coord), np.max(coord)
    slices = np.arange(zmin, zmax, dz)
    prev_center = None
    axes = [i for i in range(3) if i != axis]
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
            else:
                chosen = np.argmin(dists)
        chosen_center = centroids[chosen]
        centerline.append(chosen_center)
        prev_center = chosen_center
    centerline = np.array(centerline)
    if sigma > 0 and len(centerline) > 1:
        centerline = gaussian_filter1d(centerline, sigma=sigma, axis=0)
    # Remove any points outside the mesh surface
    if polydata is not None and len(centerline) > 0:
        centerline = filter_points_inside_mesh(centerline, polydata)
    return centerline