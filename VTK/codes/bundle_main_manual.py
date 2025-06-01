import os
import glob
from read_file import read_file
from make_mesh import make_mesh
from make_endpoints_manual import make_endpoints_manual
from manhattan_center import compute_slice_centerline
from visualize_centerline import visualize_centerline
import numpy as np
import vtk

def save_centerline_csv(centerline, out_path):
    np.savetxt(out_path, centerline, delimiter=",", header="x,y,z", comments='')

def render_and_save_image(polydata, centerline, out_path_img):
    # ... (same as before, or use your previous function)
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetOpacity(0.5)

    points = vtk.vtkPoints()
    for pt in centerline:
        points.InsertNextPoint(*pt)
    lines = vtk.vtkCellArray()
    if len(centerline) > 1:
        line = vtk.vtkPolyLine()
        line.GetPointIds().SetNumberOfIds(len(centerline))
        for i in range(len(centerline)):
            line.GetPointIds().SetId(i, i)
        lines.InsertNextCell(line)

    center_poly = vtk.vtkPolyData()
    center_poly.SetPoints(points)
    center_poly.SetLines(lines)

    center_mapper = vtk.vtkPolyDataMapper()
    center_mapper.SetInputData(center_poly)
    center_actor = vtk.vtkActor()
    center_actor.SetMapper(center_mapper)
    center_actor.GetProperty().SetColor(0, 1, 0)
    center_actor.GetProperty().SetLineWidth(8)

    renderer = vtk.vtkRenderer()
    renderer.SetBackground(1, 1, 1)
    renderer.AddActor(actor)
    renderer.AddActor(center_actor)

    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetOffScreenRendering(1)

    # Camera setup as before...
    renderer.ResetCamera()
    camera = renderer.GetActiveCamera()
    bounds = polydata.GetBounds()
    center = [(bounds[1]+bounds[0])/2, (bounds[3]+bounds[2])/2, (bounds[5]+bounds[4])/2]
    camera.SetFocalPoint(*center)
    camera.SetPosition(center[0], center[1], center[2] + 300)
    camera.SetViewUp(0, 1, 0)
    renderer.ResetCameraClippingRange()

    render_window.Render()

    window_to_image = vtk.vtkWindowToImageFilter()
    window_to_image.SetInput(render_window)
    window_to_image.Update()

    writer = vtk.vtkPNGWriter()
    writer.SetFileName(out_path_img)
    writer.SetInputConnection(window_to_image.GetOutputPort())
    writer.Write()

def main(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    vtp_files = glob.glob(os.path.join(input_folder, "*.vtp"))
    print(f"Found {len(vtp_files)} .vtp files in {input_folder}")
    for vtp_file in vtp_files:
        print(f"\nProcessing (manual selection): {vtp_file}")
        polydata = read_file(vtp_file)
        points = make_mesh(polydata)
        start_id, end_id = make_endpoints_manual(polydata)
        start_pt = points[start_id]
        end_pt = points[end_id]
        zmin, zmax = sorted([start_pt[2], end_pt[2]])
        cropped_points = points[(points[:,2] >= zmin) & (points[:,2] <= zmax)]
        centerline = compute_slice_centerline(cropped_points)

        # Visualize interactively
        visualize_centerline(polydata, centerline)

        basename = os.path.splitext(os.path.basename(vtp_file))[0]
        out_csv = os.path.join(output_folder, f"{basename}_centerline.csv")
        out_img = os.path.join(output_folder, f"{basename}_centerline.png")
        save_centerline_csv(centerline, out_csv)
        render_and_save_image(polydata, centerline, out_img)

if __name__ == "__main__":
    input_folder = r"C:\Users\robik\PyCharmMiscProject\VTK\models"
    output_folder = r"C:\Users\robik\PyCharmMiscProject\VTK\centerlines_manual"
    main(input_folder, output_folder)