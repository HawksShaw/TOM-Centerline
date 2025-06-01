import os
import glob
from read_file import read_file
from make_mesh import make_mesh
from make_endpoints import make_endpoints
from manhattan_center import compute_slice_centerline
# from visualize_centerline import visualize_centerline  # Comment out for batch mode

def save_centerline_csv(centerline, out_path):
    import numpy as np
    np.savetxt(out_path, centerline, delimiter=",", header="x,y,z", comments='')

def main(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    vtp_files = glob.glob(os.path.join(input_folder, "*.vtp"))
    print(f"Found {len(vtp_files)} .vtp files in {input_folder}")
    for vtp_file in vtp_files:
        print(f"Processing: {vtp_file}")
        polydata = read_file(vtp_file)
        points = make_mesh(polydata)
        start, end = make_endpoints(points)
        centerline = compute_slice_centerline(points)
        # visualize_centerline(polydata, centerline)  # Uncomment if you want to visualize each interactively
        basename = os.path.splitext(os.path.basename(vtp_file))[0]
        out_csv = os.path.join(output_folder, f"{basename}_centerline.csv")
        save_centerline_csv(centerline, out_csv)
        print(f"Saved centerline to {out_csv}")

if __name__ == "__main__":
    input_folder = r"C:\Users\robik\PyCharmMiscProject\VTK\models"  # Change as needed
    output_folder = r"C:\Users\robik\PyCharmMiscProject\VTK\centerlines_auto"
    main(input_folder, output_folder)