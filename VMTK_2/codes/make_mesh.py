import numpy as np

def make_mesh(polydata):
    points = polydata.GetPoints()
    np_points = np.array([points.GetPoint(i) for i in range(points.GetNumberOfPoints())])
    return np_points