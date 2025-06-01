import vtk

filename = (r'C:\Users\robik\PyCharmMiscProject\VTK'r'\models\0157_0000.vtp')
reader = vtk.vtkXMLPolyDataReader()
reader.SetFileName(filename)
reader.Update()
polydata = reader.GetOutput()
print("Number of points:", polydata.GetNumberOfPoints())