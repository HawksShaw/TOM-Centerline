import vtk
import os

def read_file(filename):
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(filename)
    reader.Update()
    polydata = reader.GetOutput()
    print(f"[read_file] Polydata points: {polydata.GetNumberOfPoints()}")
    return polydata