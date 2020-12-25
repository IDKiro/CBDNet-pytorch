# CBDNet-pytorch

Unofficial PyTorch implementation of CBDNet.

[CBDNet in MATLAB](https://github.com/GuoShi28/CBDNet)

[CBDNet in Tensorflow](https://github.com/IDKiro/CBDNet-tensorflow)

#### Update

2020.12.04: We trained the CBDNet model using the dataset in [GMSNet](https://doi.org/10.1109/LSP.2020.3039726). PSNR (DND benchmark): 38.06 -> 39.63.

## Quick Start

Download the dataset and pretrained model from [GoogleDrive](https://drive.google.com/drive/folders/1-e2nPCr_eP1cTDhFFes27Rjj-QXzMk5u?usp=sharing).

Extract the files to `data` folder and `save_model` folder as follow:

```
~/
  data/
    SIDD_train/
      ... (scene id)
    Syn_train/
      ... (id)
    DND/
      images_srgb/
        ... (mat files)
      ... (mat files)
  save_model/
    checkpoint.pth.tar
```

Train the model:

```
python train.py
```

Predict using the trained model:

```
python predict.py input_filename output_filename
```

## Network Structure

![Image of Network](imgs/CBDNet_v13.png)

## Realistic Noise Model
Given a clean image `x`, the realistic noise model can be represented as:

![](http://latex.codecogs.com/gif.latex?\\textbf{y}=f(\\textbf{DM}(\\textbf{L}+n(\\textbf{L}))))

![](http://latex.codecogs.com/gif.latex?n(\\textbf{L})=n_s(\\textbf{L})+n_c)

Where `y` is the noisy image, `f(.)` is the CRF function and the irradiance ![](http://latex.codecogs.com/gif.latex?\\textbf{L}=\\textbf{M}f^{-1}(\\textbf{x})) , `M(.)` represents the function that convert sRGB image to Bayer image and `DM(.)` represents the demosaicing function.

If considering denosing on compressed images, 

![](http://latex.codecogs.com/gif.latex?\\textbf{y}=JPEG(f(\\textbf{DM}(\\textbf{L}+n(\\textbf{L})))))

## Result

![](imgs/results.png)
