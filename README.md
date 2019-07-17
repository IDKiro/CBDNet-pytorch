# CBDNet-pytorch

An unofficial implementation of CBDNet by PyTorch.

[CBDNet in MATLAB](https://github.com/GuoShi28/CBDNet)

[CBDNet in Tensorflow](https://github.com/IDKiro/CBDNet-tensorflow)

## Quick Start

### Data

Download the dataset and pre-trained model: 
[[OneDrive](https://zjueducn-my.sharepoint.com/:f:/g/personal/3140103306_zju_edu_cn/EorD2T0_OHNEu_5rH6IpdzYB0l3SM9IfmyxWhHjyfVfFJA?e=YL4V99)]
[[Baidu Pan](https://pan.baidu.com/s/1ObvekJcPhtK9RUOC86vmNA) (8ko0)]
[[Mega](https://mega.nz/#F!uOZEVAYR!fbf-RCtnbUR7mlHZsgiL5g)]

Extract the files to `dataset` folder and `checkpoint` folder as follow:

![](imgs/folder.png)

### Train

Train the model on synthetic noisy images:

```
python train_syn.py
```

Train the model on real noisy images:

```
python train_real.py
```

Train the model on synthetic noisy images and real noisy images:

```
python train_all.py
```

**In order to reduce the time to read the images, it will save all the images in memory which requires large memory.**

### Test

Test the trained model on DND dataset:

```
python test.py
```

Optional:

```
--ckpt {all,real,synthetic}     checkpoint type
--cpu [CPU]                     Use CPU
```

Example:

```
python test.py --ckpt synthetic --cpu
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
