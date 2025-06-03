import numpy as np

def load_pth_centerline(pth_file):
    import xml.etree.ElementTree as ET
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