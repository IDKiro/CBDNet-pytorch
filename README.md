# CBDNet-pytorch

An unofficial implementation of CBDNet by PyTorch.

[CBDNet in MATLAB](https://github.com/GuoShi28/CBDNet)

## Quick Start

Download the [dataset](https://zjueducn-my.sharepoint.com/:u:/g/personal/3140103306_zju_edu_cn/Ec0R2eTtC81Is0KvC5mbomsBw2oJ_oV7jahe1srpgVufcQ?e=aFIc7h) and pretrained ~~[model]()~~ and decompression the files to `dataset` and `checkpoint` folder as follow:

![](imgs/folder.png)

Train the model on synthetic noisy images :

```
python train_syn.py
```

Train the model on real noisy images:

```
python train_real.py
```

Test the trained model on DND dataset:

```
python test.py
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
