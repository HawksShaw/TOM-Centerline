import vtk

def visualize_centerline(polydata, centerline):
    print(f"Polydata points: {polydata.GetNumberOfPoints()}")
    print(f"Centerline length: {len(centerline)}")
    if len(centerline) > 0:
        print(f"First centerline point: {centerline[0]}")

    # Aorta mesh
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetOpacity(0.5)  # Or 1.0 for fully opaque

    # Centerline as a polyline
    points = vtk.vtkPoints()
    for path in centerline:
        points.InsertNextPoint(*path)
    print(f"Centerline vtkPoints: {points.GetNumberOfPoints()}")

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
    center_actor.GetProperty().SetColor(0, 1, 0)  # Green
    center_actor.GetProperty().SetLineWidth(8)

    renderer = vtk.vtkRenderer()
    renderer.SetBackground(1, 1, 1)  # Bright background
    renderer.AddActor(actor)
    renderer.AddActor(center_actor)
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)
    render_window.Render()
    interactor.Start()