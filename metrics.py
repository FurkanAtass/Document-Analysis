from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage import generic_filter, distance_transform_cdt, binary_erosion
import os
from skimage.morphology import skeletonize
import numpy as np

def compute_confusion_counts(pred_binary, gt_binary):
    pred_binary = pred_binary.astype(bool)
    gt_binary = gt_binary.astype(bool)

    tp = np.sum(pred_binary & gt_binary)
    fp = np.sum(pred_binary & ~gt_binary)
    fn = np.sum(~pred_binary & gt_binary)
    tn = np.sum(~pred_binary & ~gt_binary)

    return tp, fp, fn, tn

def compute_precision(pred_binary, gt_binary):
    tp, fp, _, _ = compute_confusion_counts(pred_binary, gt_binary)
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0

def compute_recall(pred_binary, gt_binary):
    tp, _, fn, _ = compute_confusion_counts(pred_binary, gt_binary)
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0

def compute_f_measure(pred_binary, gt_binary):
    precision = compute_precision(pred_binary, gt_binary)
    recall = compute_recall(pred_binary, gt_binary)
    return (2 * recall * precision) / (recall + precision) if (recall + precision) > 0 else 0.0

def compute_pseudo_f_measure(pred_binary, gt_binary):
    precision = compute_precision(pred_binary, gt_binary)
    gt_skeleton = skeletonize(gt_binary.astype(bool))
    gt_skeleton_sum = np.sum(gt_skeleton)

    pseudo_recall = (
        np.sum(gt_skeleton & pred_binary.astype(bool)) / gt_skeleton_sum
        if gt_skeleton_sum > 0 else 0.0
    )

    return (
        (2 * pseudo_recall * precision) / (pseudo_recall + precision)
        if (pseudo_recall + precision) > 0 else 0.0
    )

def compute_nrm(pred_binary, gt_binary):
    tp, fp, fn, tn = compute_confusion_counts(pred_binary, gt_binary)

    nr_fn = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    nr_fp = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return (nr_fn + nr_fp) / 2

def compute_psnr(pred_binary, gt_binary):
    pred_binary = pred_binary.astype(bool)
    gt_binary = gt_binary.astype(bool)

    mse = np.mean(pred_binary != gt_binary)
    return 10 * np.log10(1.0 / mse) if mse > 0 else 0.0

def compute_mpm(pred_binary, gt_binary):
    pred_binary = pred_binary.astype(bool)
    gt_binary = gt_binary.astype(bool)

    structure = np.ones((3, 3), dtype=bool)
    gt_eroded = binary_erosion(gt_binary, structure=structure, border_value=0)
    gt_contour = gt_binary & ~gt_eroded

    dist = distance_transform_cdt(~gt_contour, metric='chessboard')

    fn = gt_binary & ~pred_binary
    fp = pred_binary & ~gt_binary

    D = dist[gt_binary].sum()

    if D == 0:
        return 0.0

    mp_fn = dist[fn].sum() / D
    mp_fp = dist[fp].sum() / D

    return (mp_fn + mp_fp) / 2

def print_results(title, metrics):
    f_m, p_f_m, p, r, psnr, nrm, mpm = metrics

    print(title)
    print('Overall:')
    print(f'F-Score: {f_m:.2f}\npseudo-F-Score: {p_f_m:.2f}\n'
          f'P: {p:.2f}\nR: {r}\nPSNR: {psnr:.2f}\n'
          f'NRM: {nrm*10**2:.3f} * 10^-2\nMPM: {mpm * 10**3:.2f} * 10^-3\n')