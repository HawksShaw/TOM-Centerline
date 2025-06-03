import vtk
import numpy as np
import xml.etree.ElementTree as ET
import glob
import os

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

def load_pth_centerline(pth_file):
    with open(pth_file, 'r', encoding='utf-8') as f:
        content = f.read()
    start = content.find("<path")
    end = content.rfind("</path>") + len("</path>")
    if start == -1 or end == -1:
        raise ValueError(f"Could not find <path> block in {pth_file}")
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

def show_model_with_centerlines(model_file, manual_csv, pth_gt_dir):
    model = load_vtk_model(model_file)
    model_mapper = vtk.vtkPolyDataMapper()
    model_mapper.SetInputData(model)
    model_actor = vtk.vtkActor()
    model_actor.SetMapper(model_mapper)
    model_actor.GetProperty().SetColor(0.85, 0.85, 0.85)
    model_actor.GetProperty().SetOpacity(0.35)

    manual_pts = load_csv_centerline(manual_csv)
    manual_poly = make_polydata_from_points(manual_pts)
    manual_mapper = vtk.vtkPolyDataMapper()
    manual_mapper.SetInputData(manual_poly)
    manual_actor = vtk.vtkActor()
    manual_actor.SetMapper(manual_mapper)
    manual_actor.GetProperty().SetColor(0, 1, 0)
    manual_actor.GetProperty().SetLineWidth(4)

    gt_actors = []
    pth_files = sorted(glob.glob(os.path.join(pth_gt_dir, "*.pth")))
    if not pth_files:
        print(f"No .pth files found in {pth_gt_dir}")
    for idx, pth_file in enumerate(pth_files):
        try:
            gt_pts = load_pth_centerline(pth_file)
        except Exception as e:
            print(f"Failed to load {pth_file}: {e}")
            continue
        gt_poly = make_polydata_from_points(gt_pts)
        gt_mapper = vtk.vtkPolyDataMapper()
        gt_mapper.SetInputData(gt_poly)
        gt_actor = vtk.vtkActor()
        gt_actor.SetMapper(gt_mapper)
        gt_actor.GetProperty().SetColor(1, 0, 0)
        gt_actor.GetProperty().SetLineWidth(4)
        gt_actors.append(gt_actor)

    renderer = vtk.vtkRenderer()
    renderer.AddActor(model_actor)
    renderer.AddActor(manual_actor)
    for actor in gt_actors:
        renderer.AddActor(actor)
    renderer.SetBackground(1, 1, 1)

    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(1000, 800)

    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)
    interactor.Initialize()
    render_window.Render()
    interactor.Start()

if __name__ == "__main__":
    model_file = r"C:\Users\robik\PyCharmMiscProject\VTK\models\0140_2001.vtp"
    manual_csv = r"C:\Users\robik\PyCharmMiscProject\VTK\centerlines_manual\0140_2001_centerline.csv"
    pth_gt_dir = r"C:\Users\robik\PyCharmMiscProject\VTK\pths\0140_2001\paths"
    show_model_with_centerlines(model_file, manual_csv, pth_gt_dir)