import numpy as np
from scipy.spatial import cKDTree
from scipy.interpolate import interp1d

def resample_line(line, num=100):
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

def hausdorff95_distance(pred, gt):
    tree_gt = cKDTree(gt)
    dists_pred_to_gt, _ = tree_gt.query(pred)
    tree_pred = cKDTree(pred)
    dists_gt_to_pred, _ = tree_pred.query(gt)
    hd95_pred = np.percentile(dists_pred_to_gt, 95)
    hd95_gt = np.percentile(dists_gt_to_pred, 95)
    return max(hd95_pred, hd95_gt)

def accuracy_over_tolerance(pred, gt, tolerances):
    tree_gt = cKDTree(gt)
    dists_pred_to_gt, _ = tree_gt.query(pred)
    accuracies = []
    for tol in tolerances:
        acc = np.mean(dists_pred_to_gt <= tol)
        accuracies.append(acc)
    return np.array(tolerances), np.array(accuracies)