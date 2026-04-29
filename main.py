import cv2
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage import generic_filter
import os
from skimage.morphology import skeletonize

EPSILON = 1e-8


# Uses integral image to compute sum of values of the original image in window around each pixel
# No padding required beforehand
def rect_sum(original_img: np.ndarray, window_size: int) -> np.ndarray:
    padded_img = np.pad(original_img, pad_width=window_size//2, mode='constant')
    integral_img = cv2.integral(padded_img)
    # I(D) - I(B) - I(C) + I(A)
    # https://en.wikipedia.org/wiki/Summed-area_table
    return (integral_img[:-window_size, :-window_size]
          - integral_img[window_size:, :-window_size]
          - integral_img[:-window_size, window_size:]
          + integral_img[window_size:, window_size:])


# Historical Document Thresholding, used by Su's binarization method
def hist_doc_threshold(contrast_img: np.ndarray, grayscale_img: np.ndarray,
                       window_size: int, n_min: int) -> np.ndarray:
    
    contrast_img = contrast_img.astype(np.float32)
    grayscale_img = grayscale_img.astype(np.float32)

    # Calculate N_e value for each window using integral image
    n_es = rect_sum(contrast_img, window_size)
    
    # Calculate E_mean values
    grayscale_masked = grayscale_img * contrast_img
    e_means = np.divide(rect_sum(grayscale_masked, window_size), n_es,
                        out=np.zeros_like(n_es), where=n_es>0)

    # Calculate E_std values
    # Use V[X] = E[X^2] - E[X]^2
    grayscale_masked_squared = grayscale_masked**2
    e_vars = np.divide(rect_sum(grayscale_masked_squared, window_size), n_es,
                       out=np.zeros_like(n_es), where=n_es>0) - e_means**2
    e_stds = np.sqrt(np.maximum(e_vars, 0)) # The max is for catastrophic cancellation

    threshold = e_means + e_stds/2
    output_img = np.where(
        (n_es >= n_min) &
        (grayscale_img <= threshold),
        1, 0
    )

    return output_img


# Estimates stroke width. Used to define window size for thresholding
def estimate_stroke_width(contrast_img: np.ndarray) -> int:
    # Apply additional threshold to find peak values
    contrast_img_thresholded = contrast_img.copy()
    contrast_img_thresholded[contrast_img < np.percentile(contrast_img, 90)] = 0
    
    distances = []
    for row in contrast_img_thresholded:
        peak_positions = []

        for i in range(1, len(row) - 1):
            center = row[i]
            window = row[i - 1 : i + 2]

            if center > 0 and center == window.max():
                if not peak_positions or i != peak_positions[-1]:
                    peak_positions.append(i)

        for i in range(len(peak_positions) - 1):
            distances.append(peak_positions[i+1] - peak_positions[i])
    
    distances = np.array(distances)
    hist, bin_edges = np.histogram(distances, bins=np.arange(1, distances.max()+2))
    hist[0] = 0
    stroke_width = int(bin_edges[np.argmax(hist)])
    return stroke_width if stroke_width%2 != 0 else stroke_width+1


# Implements Su's binarization method
def Su(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32) / 255.0

    element = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    erosion = cv2.erode(image, element)
    dilation = cv2.dilate(image, element)

    contrast_img = (dilation - erosion) / (dilation + erosion + EPSILON)

    contrast_img_uint8 = (contrast_img * 255.0).astype(np.uint8)
    _, otsu_result = cv2.threshold(contrast_img_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    otsu_binary = otsu_result > 127

    stroke_width = estimate_stroke_width(contrast_img)

    result = hist_doc_threshold(contrast_img=otsu_binary,
                                grayscale_img=image,
                                window_size=stroke_width,
                                n_min=stroke_width)
    
    result = 1 - result
    return (result * 255.0).astype(np.uint8)


# Load Test and Ground Truth images
def load_test_gt(test_dir):
    test_images, gt_images = [], []
    for filename in sorted([f for f in os.listdir(test_dir) if not f.endswith('_gt.tif')]):
        test_images.append(cv2.imread(os.path.join(test_dir, filename), cv2.IMREAD_GRAYSCALE))
    gt_dir = os.path.join(test_dir, 'gt')
    for filename in sorted([f for f in os.listdir(test_dir) if f.endswith('_gt.tif')]):
        gt_images.append(cv2.imread(os.path.join(test_dir, filename), cv2.IMREAD_GRAYSCALE))
    return (test_images, gt_images)


def evaluate(test_images, gt_images, log=False):
    f_measures = []
    pseudo_f_measures = []
    precisions = []
    recalls = []
    psnrs = []
    assert len(test_images) == len(gt_images), "Mismatch between number of test and ground truth images"
   
    i = 1
    for test_img, gt_img in zip(test_images, gt_images): 
        pred = Su(test_img)

        pred_binary = pred<127
        gt_binary = gt_img<127

        ctp = np.sum(pred_binary * gt_binary)
        cfp = np.sum(pred_binary * (1-gt_binary))
        cfn = np.sum((1-pred_binary) * gt_binary)

        rc = ctp / (ctp + cfn) if (ctp + cfn) > 0 else 0.0
        pr = ctp / (ctp + cfp) if (ctp + cfp) > 0 else 0.0

        f_m = (2 * rc * pr) / (rc + pr)
        f_measures.append(f_m)

        gt_skeleton = skeletonize(gt_binary)
        gt_skeleton_sum = np.sum(gt_skeleton)
        pseudo_rc = np.sum(gt_skeleton * pred_binary)/gt_skeleton_sum if gt_skeleton_sum > 0 else 0
        pseudo_f_m = (2 * pseudo_rc * pr)/(pseudo_rc + pr) if (pseudo_rc + pr) > 0 else 0.0
        pseudo_f_measures.append(pseudo_f_m)

        precisions.append(pr)
        recalls.append(rc)

        mse = np.mean(pred_binary != gt_binary)
        psnr = 10 * np.log10(1.0 / mse) if mse > 0 else 0.0
        psnrs.append(psnr) 

        if log==True:
            print('-------------------------------')
            print(f'Image {i}:')
            print(f'F-Score: {f_m:.2f}\npseudo-F-Score: {pseudo_f_m:.2f}\n'
                  f'P: {pr:.2f}\nR: {rc}\nPSNR: {psnr:.2f}')
            print('-------------------------------')
            i = i+1

    return (np.mean(np.array(f_measures)),
            np.mean(np.array(pseudo_f_measures)),
            np.mean(np.array(precisions)),
            np.mean(np.array(recalls)),
            np.mean(np.array(psnrs)))


def main():
    dibco_dir = os.path.join(os.path.dirname(__file__), 'DAVU-UE1', 'dibco2009')
    handwritten_dir = os.path.join(dibco_dir, 'DIBC02009_Test_images-handwritten')
    handwritten_test, handwritten_gt = load_test_gt(handwritten_dir)
    print('1. Handwritten')
    f_m, p_f_m, p, r, psnr = evaluate(handwritten_test, handwritten_gt, log=False)
    print('Overall:')
    print(f'F-Score: {f_m:.2f}\npseudo-F-Score: {p_f_m:.2f}\nP: {p:.2f}\nR: {r}\nPSNR: {psnr:.2f}\n')
    
    printed_dir = os.path.join(dibco_dir, 'DIBCO2009_Test_images-printed')
    printed_test, printed_gt = load_test_gt(printed_dir)
    print('2. Printed')
    f_m, p_f_m, p, r, psnr = evaluate(printed_test, printed_gt, log=False)
    print('Overall:')
    print(f'F-Score: {f_m:.2f}\npseudo-F-Score: {p_f_m:.2f}\nP: {p:.2f}\nR: {r}\nPSNR: {psnr:.2f}\n')



if __name__ == "__main__":
    main()