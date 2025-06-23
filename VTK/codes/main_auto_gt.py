import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from read_file import read_file
from make_mesh import make_mesh
from make_endpoints import make_endpoints
from manhattan_center import compute_slice_centerline
import xml.etree.ElementTree as ET

from centerline_scoring import (
    resample_line,
    mean_closest_distance,
    hausdorff_distance,
    average_symmetric_distance,
    hausdorff95_distance,
    accuracy_over_tolerance,
)

def save_centerline_csv(centerline, out_path):
    np.savetxt(out_path, centerline, delimiter=",", header="x,y,z", comments='')

def load_pth_centerline(pth_file):
    """
    load .pth files as ground truth centerline.
    """
    with open(pth_file, 'r', encoding='utf-8') as f:
        content = f.read()
    start = content.find("<path")
    end = content.rfind("</path>") + len("</path>")
    if start == -1 or end == -1:
        raise ValueError(f"Could not find <path> in {pth_file}")
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

def load_all_segments(model_pth_dir):
    """
    Load and concatenate all .pth files in a directory.
    Return a single (N,3) array of all .pth points.
    """
    segment_files = sorted(glob.glob(os.path.join(model_pth_dir, "*.pth")))
    if not segment_files:
        raise FileNotFoundError(f"No .pth files found in {model_pth_dir}")
    all_points = []
    for seg in segment_files:
        try:
            seg_points = load_pth_centerline(seg)
            all_points.append(seg_points)
        except Exception as e:
            print(f"Failed to load {seg}: {e}")
    if not all_points:
        raise ValueError(f"No valid centerline points loaded from {model_pth_dir}")
    return np.concatenate(all_points, axis=0)

def main(input_folder, pth_folder, output_folder, output_scores_csv):
    """
    1. Create an output folder
    2. Find all .vtp files in the 'models' directory'.
    3. Make mesh of a chosen file.
    4. Get endpoints.
    5. Compute the centerline and save to CSV.
    6. Locate and compute corresponding ground truth.
    7. Resample both centerlines to match in num_points.
    8. Compare using mean_closest_distance, hausdorff_distance, average_symmetric_distance, hausdorff95.
    9. Plot accuracy curves.
    10. Add the scores to CSV.
    11. Display average of all distances after computing each .vtp file.
    """
    os.makedirs(output_folder, exist_ok=True)
    vtp_files = glob.glob(os.path.join(input_folder, "*.vtp"))
    print(f"Found {len(vtp_files)} .vtp files in {input_folder}")

    all_mean = []
    all_haus = []
    all_avg_sym = []
    all_hd95 = []
    all_accuracy_curves = []
    n_scored = 0


    tolerances = np.linspace(0.5, 10, 20)  # 0.5mm to 10mm

    with open(output_scores_csv, 'w') as score_file:
        score_file.write("filename,mean_closest,hausdorff,avg_symmetric,hausdorff95\n")
        for vtp_file in vtp_files:
            print(f"Processing: {vtp_file}")
            polydata = read_file(vtp_file)
            points = make_mesh(polydata)
            start, end = make_endpoints(points)
            centerline = compute_slice_centerline(points)
            basename = os.path.splitext(os.path.basename(vtp_file))[0]

            out_csv = os.path.join(output_folder, f"{basename}_centerline.csv")
            save_centerline_csv(centerline, out_csv)

            model_pth_dir = os.path.join(pth_folder, basename, "paths")
            if not os.path.exists(model_pth_dir):
                print(f"Ground truth dir not found for {basename}")
                score_file.write(f"{basename},,,,\n")
                continue
            try:
                gt_centerline = load_all_segments(model_pth_dir)
            except Exception as e:
                print(f"Failed to load segments for {basename}: {e}")
                score_file.write(f"{basename},,,,\n")
                continue

            gt_csv = os.path.join(output_folder, f"{basename}_centerline_gt.csv")
            save_centerline_csv(gt_centerline, gt_csv)

            num_points = 100
            try:
                pred_rs = resample_line(centerline, num_points)
                gt_rs = resample_line(gt_centerline, num_points)

                mean_c = mean_closest_distance(pred_rs, gt_rs)
                haus = hausdorff_distance(pred_rs, gt_rs)
                avg_sym = average_symmetric_distance(pred_rs, gt_rs)
                hd95 = hausdorff95_distance(pred_rs, gt_rs)


                tols, accs = accuracy_over_tolerance(pred_rs, gt_rs, tolerances)
                all_accuracy_curves.append(accs)

                print(f"Scores for {basename}: mean={mean_c:.3f}, hausdorff={haus:.3f}, avg_sym={avg_sym:.3f}, hd95={hd95:.3f}")
                score_file.write(f"{basename},{mean_c},{haus},{avg_sym},{hd95}\n")
                all_mean.append(mean_c)
                all_haus.append(haus)
                all_avg_sym.append(avg_sym)
                all_hd95.append(hd95)
                n_scored += 1

                # Uncomment this if you need to check the acc over tolerance plot for every model
                # plt.figure()
                # plt.plot(tols, accs, marker='o')
                # plt.xlabel('Tolerance (mm)')
                # plt.ylabel('Accuracy (fraction within tolerance)')
                # plt.title(f'{basename}: Accuracy vs. Tolerance')
                # plt.grid(True)
                # plt.show()

            except Exception as e:
                print(f"Scoring failed for {vtp_file}: {e}")
                score_file.write(f"{basename},,,,\n")

        if n_scored:
            avg_mean = np.mean(all_mean)
            avg_haus = np.mean(all_haus)
            avg_avg_sym = np.mean(all_avg_sym)
            avg_hd95 = np.mean(all_hd95)

            avg_acc_curve = np.mean(all_accuracy_curves, axis=0) * 100

            print(f"\nAverage scores for all models:")
            print(f"  Mean Closest Distance: {avg_mean:.3f} mm")
            print(f"  Hausdorff Distance:    {avg_haus:.3f} mm")
            print(f"  Avg Symmetric Dist:    {avg_avg_sym:.3f} mm")
            print(f"  Hausdorff95:           {avg_hd95:.3f} mm")
            score_file.write(f"\nAVERAGE,{avg_mean},{avg_haus},{avg_avg_sym},{avg_hd95} [mm]\n")


            plt.figure()
            plt.plot(tolerances, avg_acc_curve, marker='o')
            plt.xlabel('Promień zakresu tolerancji [mm]')
            plt.ylabel('Średnia dokładność [%]')
            plt.title('Średnia dokładność ścieżki centralnej w zależności od obszaru tolerancji')
            plt.grid(True)
            plt.show()
        else:
            print("\nNo models were scored (no valid ground truth found).")

if __name__ == "__main__":
    input_folder = r"C:\Users\robik\PyCharmMiscProject\VTK\models"
    pth_folder = r"C:\Users\robik\PyCharmMiscProject\VTK\pths"
    output_folder = r"C:\Users\robik\PyCharmMiscProject\VTK\centerlines_auto"
    output_scores_csv = os.path.join(output_folder, "accuracy_scores_vs_pth.csv")
    main(input_folder, pth_folder, output_folder, output_scores_csv)