import xml.etree.ElementTree as ET
import numpy as np
import vtk

def parse_path_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    start = content.find("<path")
    end = content.rfind("</path>") + len("</path>")
    xml_str = content[start:end]
    root = ET.fromstring(xml_str)

    points = []
    for point in root.findall(".//control_points/point"):
        x = float(point.attrib['x'])
        y = float(point.attrib['y'])
        z = float(point.attrib['z'])
        points.append([x, y, z])
    return np.array(points)

paths = [
    r"C:\Users\robik\PyCharmMiscProject\VTK\pths\0140_2001\paths\aorta.pth",
]

renderer = vtk.vtkRenderer()
render_window = vtk.vtkRenderWindow()
render_window.AddRenderer(renderer)
interactor = vtk.vtkRenderWindowInteractor()
interactor.SetRenderWindow(render_window)

for path in paths:
    pts = parse_path_file(path)

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

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(poly_data)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(1, 0, 0)
    actor.GetProperty().SetLineWidth(3)

    renderer.AddActor(actor)

renderer.SetBackground(1, 1, 1)
render_window.SetSize(800, 600)

interactor.Initialize()
render_window.Render()
interactor.Start()