import vtk

class RightClickPickerStyle(vtk.vtkInteractorStyleTrackballCamera):
    def __init__(self, polydata, picked_points, renderer, parent=None):
        super().__init__()
        self.AddObserver("RightButtonPressEvent", self.right_click_event)
        self.polydata = polydata
        self.picked_points = picked_points
        self.point_picker = vtk.vtkPointPicker()
        self.renderer = renderer
        self.sphere_radius = 0.5

    def add_sphere(self, position, color):
        sphere = vtk.vtkSphereSource()
        sphere.SetCenter(*position)
        sphere.SetRadius(self.sphere_radius)
        sphere.SetPhiResolution(30)
        sphere.SetThetaResolution(30)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(sphere.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*color)
        actor.GetProperty().SetDiffuse(0.8)
        actor.GetProperty().SetSpecular(0.5)
        self.renderer.AddActor(actor)
        self.renderer.GetRenderWindow().Render()

    def right_click_event(self, obj, event):
        click_pos = self.GetInteractor().GetEventPosition()
        renderer = self.GetDefaultRenderer()
        self.point_picker.Pick(click_pos[0], click_pos[1], 0, renderer)
        point_id = self.point_picker.GetPointId()
        if point_id >= 0:
            point = self.polydata.GetPoint(point_id)
            self.picked_points.append((point_id, point))
            print(f"Picked point {len(self.picked_points)}: ID={point_id}, Coord={point}")
            # Visual indicator: red for first, blue for second
            color = (1, 0, 0)
            self.add_sphere(point, color)
        if len(self.picked_points) == 2:
            self.GetInteractor().GetRenderWindow().Finalize()
            self.GetInteractor().TerminateApp()

def make_endpoints_manual(polydata):
    picked_points = []

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetOpacity(0.7)

    renderer = vtk.vtkRenderer()
    renderer.SetBackground(1, 1, 1)
    renderer.AddActor(actor)

    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)

    style = RightClickPickerStyle(polydata, picked_points, renderer)
    style.SetDefaultRenderer(renderer)
    interactor.SetInteractorStyle(style)

    print("Right-click two points on the mesh to select start and end endpoints.")
    render_window.Render()
    interactor.Start()

    start_id, start_pt = picked_points[0]
    end_id, end_pt = picked_points[1]
    return start_id, end_id