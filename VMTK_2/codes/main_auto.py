from read_file import read_file
from make_mesh import make_mesh
from make_endpoints import make_endpoints
from manhattan_center import compute_slice_centerline
from visualize_centerline import visualize_centerline


filename = r'C:\Users\robik\PyCharmMiscProject\VMTK_2\models\0157_0000.vtp'
polydata = read_file(filename)
print(f"[main] Polydata points: {polydata.GetNumberOfPoints()}")
points = make_mesh(polydata)
print(f"[main] Points array shape: {points.shape}")
start, end = make_endpoints(points)
centerline = compute_slice_centerline(points)
visualize_centerline(polydata, centerline)
