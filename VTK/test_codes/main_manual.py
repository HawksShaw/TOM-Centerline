from read_file import read_file
from make_mesh import make_mesh
from make_endpoints_manual import make_endpoints_manual
from manhattan_center import compute_slice_centerline
from visualize_centerline import visualize_centerline

filename = r'/VTK/models/0140_2001.vtp'
polydata = read_file(filename)
points = make_mesh(polydata)
start_id, end_id = make_endpoints_manual(polydata)

start_pt = points[start_id]
end_pt = points[end_id]

zmin, zmax = sorted([start_pt[2], end_pt[2]])
cropped_points = points[(points[:,2] >= zmin) & (points[:,2] <= zmax)]

centerline = compute_slice_centerline(cropped_points)
visualize_centerline(polydata, centerline)