# Document Analysis Task 1 - Part 2

This part of the task compares two document image binarization methods on DIBCO 2009:

- a paper-based implementation of binarization method of Su et al.
- a trained U-Net model

The scripts report the same evaluation metrics for both methods: F-Score, pseudo-F-Score, precision, recall, PSNR, NRM, and MPM.

## Environment
Install uv and run 
```bash
uv sync
```
to install dependencies.

## Pre-Trained U-Net Model

Download the pre-trained U-Net checkpoint here:

[Download pre-trained U-Net model](https://drive.google.com/file/d/1jpZ5U3gfCsvJTiwrNbWJIePvEhZw1KQR/view?usp=sharing)

Place the downloaded checkpoint in this folder with the name:

```text
unet_dibco_last.pth
```

## Test Both Methods on DIBCO 2009

Run `main.py` to evaluate both the paper implementation and the U-Net model on the DIBCO 2009 handwritten and printed test images:

```bash
uv run python main.py
```

The expected test dataset structure is:

```text
part2/
  DAVU-UE1/
    dibco2009/
      DIBC02009_Test_images-handwritten/
        dibco_img0001.tif
        dibco_img0001_gt.tif
        ...
      DIBCO2009_Test_images-printed/
        dibco_img0006.tif
        dibco_img0006_gt.tif
        ...
```

Each folder should contain the test images and their corresponding ground-truth images. Ground-truth files are expected to end with `_gt.tif`.

## Train the U-Net

Use `train.py` to train the U-Net model:

```bash
uv run python train.py
```

The training dataset and preprocessing code are based on:

https://github.com/rezazad68/BCDUnet_DIBCO

Download that dataset and unzip it into this project folder. The training script expects the dataset directory defined by `Dataset_dir` at the top of `train.py`.

Training parameters such as patch size, number of epochs, batch size, learning rate, validation split, and test year are also defined near the top of `train.py`.

After training, the model is saved as:

```text
unet_dibco_last.pth
```

## Inference on a Custom Image

Use `inference.py` to run the trained U-Net model on a single custom image:

```bash
uv run python inference.py
```

The input image path, output directory, model path, threshold mode, patch size, and batch size are configured at the top of `inference.py`.

The predicted binary mask is saved to:

```text
output/output_mask.png
```



