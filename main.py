import os
import torch
from dataset import load_test_gt
from inference import load_model
from paper_implementation import evaluate as evaluate_paper
from test_unet import MODEL_PATH, evaluate_unet

def format_metrics(metrics):
    f_m, p_f_m, p, r, psnr, nrm, mpm = metrics
    return (
        f'F-Score: {f_m:.2f}\n'
        f'pseudo-F-Score: {p_f_m:.2f}\n'
        f'P: {p:.2f}\n'
        f'R: {r}\n'
        f'PSNR: {psnr:.2f}\n'
        f'NRM: {nrm * 10**2:.3f} * 10^-2\n'
        f'MPM: {mpm * 10**3:.2f} * 10^-3'
    )

def print_comparison(dataset_name, paper_metrics, unet_metrics):
    print(dataset_name)
    print('Paper Implementation:')
    print(format_metrics(paper_metrics))
    print()
    print('UNet:')
    print(format_metrics(unet_metrics))
    print()

def evaluate_dataset(dataset_name, dataset_dir, model, device):
    test_images, gt_images = load_test_gt(dataset_dir)

    paper_metrics = evaluate_paper(test_images, gt_images, log=False)
    unet_metrics = evaluate_unet(model, test_images, gt_images, device, log=False)

    print_comparison(dataset_name, paper_metrics, unet_metrics)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(MODEL_PATH, device)

    dibco_dir = os.path.join(os.path.dirname(__file__), 'DAVU-UE1', 'dibco2009')

    evaluate_dataset(
        '1. Handwritten',
        os.path.join(dibco_dir, 'DIBC02009_Test_images-handwritten'),
        model,
        device,
    )

    evaluate_dataset(
        '2. Printed',
        os.path.join(dibco_dir, 'DIBCO2009_Test_images-printed'),
        model,
        device,
    )

if __name__ == '__main__':
    main()
