import os
import numpy as np
import torch
from inference import load_model, predict_full_mask
from dataset import load_test_gt
from metrics import (
    compute_f_measure,
    compute_mpm,
    compute_nrm,
    compute_precision,
    compute_pseudo_f_measure,
    compute_psnr,
    compute_recall,
    print_results,
)

MODEL_PATH = 'unet_dibco_last.pth'
BATCH_SIZE = 16
PATCH_SIZE = 256
THRESHOLD = 0.5
THRESHOLD_MODE = 'mean'


def evaluate_unet(model, test_images, gt_images, device, log=False):
    f_measures = []
    pseudo_f_measures = []
    precisions = []
    recalls = []
    psnrs = []
    nrms = []
    mpms = []

    assert len(test_images) == len(gt_images), "Mismatch between number of test and ground truth images"

    for i, (test_img, gt_img) in enumerate(zip(test_images, gt_images), start=1):
        pred = predict_full_mask(
            model,
            test_img,
            patch_size=PATCH_SIZE,
            device=device,
            batch_size=BATCH_SIZE,
            threshold=THRESHOLD,
            threshold_mode=THRESHOLD_MODE,
        )

        pred_binary_foreground = pred < 127
        gt_binary_foreground = gt_img < 127

        pred_binary_background = pred >= 127
        gt_binary_background = gt_img >= 127

        f_m = compute_f_measure(pred_binary_foreground, gt_binary_foreground)
        pseudo_f_m = compute_pseudo_f_measure(pred_binary_foreground, gt_binary_foreground)
        pr = compute_precision(pred_binary_foreground, gt_binary_foreground)
        rc = compute_recall(pred_binary_foreground, gt_binary_foreground)
        psnr = compute_psnr(pred_binary_foreground, gt_binary_foreground)
        nrm = compute_nrm(pred_binary_foreground, gt_binary_foreground)
        mpm = compute_mpm(pred_binary_background, gt_binary_background)

        f_measures.append(f_m)
        pseudo_f_measures.append(pseudo_f_m)
        precisions.append(pr)
        recalls.append(rc)
        psnrs.append(psnr)
        nrms.append(nrm)
        mpms.append(mpm)

        if log:
            print('-------------------------------')
            print(f'Image {i}:')
            print(f'F-Score: {f_m:.2f}\npseudo-F-Score: {pseudo_f_m:.2f}\n'
                  f'P: {pr:.2f}\nR: {rc}\nPSNR: {psnr:.2f}\n'
                  f'NRM: {nrm*10**2:.3f} * 10^-2\nMPM: {mpm * 10**3:.2f} * 10^-3')
            print('-------------------------------')

    return (
        np.mean(np.array(f_measures)),
        np.mean(np.array(pseudo_f_measures)),
        np.mean(np.array(precisions)),
        np.mean(np.array(recalls)),
        np.mean(np.array(psnrs)),
        np.mean(np.array(nrms)),
        np.mean(np.array(mpms)),
    )

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(MODEL_PATH, device)

    dibco_dir = os.path.join(os.path.dirname(__file__), 'DAVU-UE1', 'dibco2009')

    handwritten_dir = os.path.join(dibco_dir, 'DIBC02009_Test_images-handwritten')
    handwritten_test, handwritten_gt = load_test_gt(handwritten_dir)
    handwritten_metrics = evaluate_unet(model, handwritten_test, handwritten_gt, device, log=False)
    print_results('1. Handwritten', handwritten_metrics)

    printed_dir = os.path.join(dibco_dir, 'DIBCO2009_Test_images-printed')
    printed_test, printed_gt = load_test_gt(printed_dir)
    printed_metrics = evaluate_unet(model, printed_test, printed_gt, device, log=False)
    print_results('2. Printed', printed_metrics)


if __name__ == '__main__':
    main()
