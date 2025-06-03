import vtk
import numpy as np

def load_vtk_model(filename):
    if filename.endswith('.vtp'):
        reader = vtk.vtkXMLPolyDataReader()
    elif filename.endswith('.stl'):
        reader = vtk.vtkSTLReader()
    else:
        raise ValueError("Unsupported model file type")
    reader.SetFileName(filename)
    reader.Update()
    return reader.GetOutput()

def load_csv_centerline(filename):
    return np.loadtxt(filename, delimiter=',', skiprows=1)  # skip header

import xml.etree.ElementTree as ET
def load_pth_centerline(pth_file):
    with open(pth_file, 'r', encoding='utf-8') as f:
        content = f.read()
    start = content.find("<path")
    end = content.rfind("</path>") + len("</path>")
    xml_str = content[start:end]
    root = ET.fromstring(xml_str)
    points = []
    for path_point in root.findall(".//path_points/path_point"):
        pos = path_point.find("pos")
        x = float(pos.attrib['x'])
        y = float(pos.attrib['y'])
        z = float(pos.attrib['z'])
        points.append([x, y, z])
    return np.array(points)

def make_polydata_from_points(pts):
    vtk_points = vtk.vtkPoints()
    for pt in pts:
        vtk_points.InsertNextPoint(pt)
    lines = vtk.vtkCellArray()
    lines.InsertNextCell(len(pts))
    for j in range(len(pts)):
        lines.InsertCellPoint(j)
    poly_data = vtk.vtkPolyData()
    poly_data.SetPoints(vtk_points)
    poly_data.SetLines(lines)
    return poly_data

def show_model_with_centerlines(model_file, auto_csv, pth_gt):
    # Load model surface
    model = load_vtk_model(model_file)
    model_mapper = vtk.vtkPolyDataMapper()
    model_mapper.SetInputData(model)
    model_actor = vtk.vtkActor()
    model_actor.SetMapper(model_mapper)
    model_actor.GetProperty().SetColor(0.85, 0.85, 0.85)
    model_actor.GetProperty().SetOpacity(0.25)

    auto_pts = load_csv_centerline(auto_csv)
    auto_poly = make_polydata_from_points(auto_pts)
    auto_mapper = vtk.vtkPolyDataMapper()
    auto_mapper.SetInputData(auto_poly)
    auto_actor = vtk.vtkActor()
    auto_actor.SetMapper(auto_mapper)
    auto_actor.GetProperty().SetColor(1, 0, 0)  # RED
    auto_actor.GetProperty().SetLineWidth(4)

    gt_pts = load_pth_centerline(pth_gt)
    gt_poly = make_polydata_from_points(gt_pts)
    gt_mapper = vtk.vtkPolyDataMapper()
    gt_mapper.SetInputData(gt_poly)
    gt_actor = vtk.vtkActor()
    gt_actor.SetMapper(gt_mapper)
    gt_actor.GetProperty().SetColor(0, 0, 1)  # BLUE
    gt_actor.GetProperty().SetLineWidth(4)

    renderer = vtk.vtkRenderer()
    renderer.AddActor(model_actor)
    renderer.AddActor(auto_actor)
    renderer.AddActor(gt_actor)
    renderer.SetBackground(1, 1, 1)

    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(1000, 800)

    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)
    interactor.Initialize()
    render_window.Render()
    interactor.Start()

# --------- Example usage ---------
if __name__ == "__main__":
    # Provide your file paths here
    model_file = r"C:\Users\robik\PyCharmMiscProject\VTK\models\0161_0001.vtp"
    auto_csv = r"C:\Users\robik\PyCharmMiscProject\VTK\centerlines_auto\0161_0001_centerline.csv"
    pth_gt = r"C:\Users\robik\PyCharmMiscProject\VTK\pths\0161_0001\paths\aorta.pth"
    show_model_with_centerlines(model_file, auto_csv, pth_gt)