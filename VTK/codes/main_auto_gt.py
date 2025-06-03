import os
import glob
import numpy as np
from read_file import read_file
from make_mesh import make_mesh
from make_endpoints import make_endpoints
from manhattan_center import compute_slice_centerline
import xml.etree.ElementTree as ET
from scipy.spatial import cKDTree
import time

def save_centerline_csv(centerline, out_path):
    np.savetxt(out_path, centerline, delimiter=",", header="x,y,z", comments='')

def resample_line(line, num=100):
    from scipy.interpolate import interp1d
    if len(line) < 2:
        return line
    dists = np.cumsum(np.linalg.norm(np.diff(line, axis=0), axis=1))
    dists = np.insert(dists, 0, 0)
    interp = interp1d(dists, line, axis=0, kind='linear')
    new_dists = np.linspace(0, dists[-1], num)
    return interp(new_dists)

def mean_closest_distance(pred, gt):
    tree = cKDTree(gt)
    dists, _ = tree.query(pred)
    return np.mean(dists)

def hausdorff_distance(pred, gt):
    tree_gt = cKDTree(gt)
    dists_pred_to_gt, _ = tree_gt.query(pred)
    tree_pred = cKDTree(pred)
    dists_gt_to_pred, _ = tree_pred.query(gt)
    return max(np.max(dists_pred_to_gt), np.max(dists_gt_to_pred))

def average_symmetric_distance(pred, gt):
    tree_gt = cKDTree(gt)
    dists_pred_to_gt, _ = tree_gt.query(pred)
    tree_pred = cKDTree(pred)
    dists_gt_to_pred, _ = tree_pred.query(gt)
    return (np.mean(dists_pred_to_gt) + np.mean(dists_gt_to_pred)) / 2

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

def load_all_segments(model_pth_dir):
    """
    Loads and concatenates all segment .pth files in a directory.
    Returns a single (N,3) array of all path points.
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
    os.makedirs(output_folder, exist_ok=True)
    vtp_files = glob.glob(os.path.join(input_folder, "*.vtp"))
    print(f"Found {len(vtp_files)} .vtp files in {input_folder}")

    all_mean = []
    all_haus = []
    all_avg_sym = []
    n_scored = 0

    with open(output_scores_csv, 'w') as score_file:
        score_file.write("filename,mean_closest,hausdorff,avg_symmetric\n")
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
                score_file.write(f"{basename},,,\n")
                continue
            try:
                gt_centerline = load_all_segments(model_pth_dir)
            except Exception as e:
                print(f"Failed to load segments for {basename}: {e}")
                score_file.write(f"{basename},,,\n")
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
                print(f"Scores for {basename}: mean={mean_c:.3f}, hausdorff={haus:.3f}, avg_sym={avg_sym:.3f}")
                score_file.write(f"{basename},{mean_c},{haus},{avg_sym}\n")
                all_mean.append(mean_c)
                all_haus.append(haus)
                all_avg_sym.append(avg_sym)
                n_scored += 1
            except Exception as e:
                print(f"Scoring failed for {vtp_file}: {e}")
                score_file.write(f"{basename},,,\n")

        if n_scored:
            avg_mean = np.mean(all_mean)
            avg_haus = np.mean(all_haus)
            avg_avg_sym = np.mean(all_avg_sym)
            print(f"\nAverage scores for all models:")
            print(f"  Mean Closest Distance: {avg_mean:.3f} mm")
            print(f"  Hausdorff Distance:    {avg_haus:.3f} mm")
            print(f"  Avg Symmetric Dist:    {avg_avg_sym:.3f} mm")
            score_file.write(f"\nAVERAGE,{avg_mean},{avg_haus},{avg_avg_sym} [mm]\n")
        else:
            print("\nNo models were scored (no valid ground truth found).")

if __name__ == "__main__":
    input_folder = r"C:\Users\robik\PyCharmMiscProject\VTK\models"
    pth_folder = r"C:\Users\robik\PyCharmMiscProject\VTK\pths"
    output_folder = r"C:\Users\robik\PyCharmMiscProject\VTK\centerlines_auto"
    output_scores_csv = os.path.join(output_folder, "accuracy_scores_vs_pth.csv")
    main(input_folder, pth_folder, output_folder, output_scores_csv)