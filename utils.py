import numpy as np
import cv2
import os
import math
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def func(x, a):
    return np.power(x, a)

def CRF_curve_fit(I, B):
    popt, pcov = curve_fit(func, I, B)
    return popt

def CRF_function_transfer(x, y):
    para = []
    for crf in range(201):
        temp_x = np.array(x[crf, :])
        temp_y = np.array(y[crf, :])
        para.append(CRF_curve_fit(temp_x, temp_y))
    return para

def mosaic_bayer(rgb, pattern, noiselevel):

    w, h, c = rgb.shape
    if pattern == 1:
        num = [1, 2, 0, 1]
    elif pattern == 2:
        num = [1, 0, 2, 1]
    elif pattern == 3:
        num = [2, 1, 1, 0]
    elif pattern == 4:
        num = [0, 1, 1, 2]
    elif pattern == 5:
        return rgb
    
    mosaic = np.zeros((w, h, 3))
    mask = np.zeros((w, h, 3))
    B = np.zeros((w, h))

    B[0:w:2, 0:h:2] = rgb[0:w:2, 0:h:2, num[0]]
    B[0:w:2, 1:h:2] = rgb[0:w:2, 1:h:2, num[1]]
    B[1:w:2, 0:h:2] = rgb[1:w:2, 0:h:2, num[2]]
    B[1:w:2, 1:h:2] = rgb[1:w:2, 1:h:2, num[3]]

    gauss = np.random.normal(0, noiselevel/255.,(w, h))
    gauss = gauss.reshape(w, h)
    B = B + gauss

    return (B, mask, mosaic)

def ICRF_Map(Img, I, B):
    w, h, c = Img.shape
    output_Img = Img.copy()
    prebin = I.shape[0]
    tiny_bin = 9.7656e-04
    min_tiny_bin = 0.0039
    for i in range(w):
       for j in range(h):
           for k in range(c):
               temp = output_Img[i, j, k]
               start_bin = 0
               if temp > min_tiny_bin:
                   start_bin = math.floor(temp/tiny_bin - 1) - 1
               for b in range(start_bin, prebin):
                   tempB = B[b]
                   if tempB >= temp:
                       index = b
                       if index > 0:
                           comp1 = tempB - temp
                           comp2 = temp - B[index-1]
                           if comp2 < comp1:
                               index = index - 1
                       output_Img[i, j, k] = I[index]
                       break
               
    return output_Img

def CRF_Map(Img, I, B):
    w, h, c = Img.shape
    output_Img = Img.copy()
    prebin = I.shape[0]
    tiny_bin = 9.7656e-04
    min_tiny_bin = 0.0039
    for i in range(w):
        for j in range(h):
            for k in range(c):
                temp = output_Img[i, j, k]

                if temp < 0:
                    temp = 0
                    Img[i, j, k] = 0
                elif temp > 1:
                    temp = 1
                    Img[i, j, k] = 1
                start_bin = 0
                if temp > min_tiny_bin:
                    start_bin = math.floor(temp/tiny_bin - 1) - 1

                for b in range(start_bin, prebin):
                    tempB = I[b]
                    if tempB >= temp:
                        index = b
                        if index > 0:
                            comp1 = tempB - temp
                            comp2 = temp - B[index-1]
                            if comp2 < comp1:
                                index = index - 1
                        output_Img[i, j, k] = B[index]
                        break
    return output_Img 

def CRF_Map_opt(Img, popt):
    w, h, c = Img.shape
    output_Img = Img.copy()

    output_Img = func(output_Img, *popt)
    return output_Img 

def Demosaic(B_b, pattern):
     
    B_b = B_b * 255
    B_b = B_b.astype(np.uint16)

    if pattern == 1:
        lin_rgb = cv2.demosaicing(B_b, cv2.COLOR_BayerGB2BGR)
    elif pattern == 2:
        lin_rgb = cv2.demosaicing(B_b, cv2.COLOR_BayerGR2BGR)
    elif pattern == 3:
        lin_rgb = cv2.demosaicing(B_b, cv2.COLOR_BayerBG2BGR)
    elif pattern == 4:
        lin_rgb = cv2.demosaicing(B_b, cv2.COLOR_BayerRG2BGR)
    elif pattern == 5:
        lin_rgb = B_b

    lin_rgb = lin_rgb[:,:,::-1] / 255.  
    return lin_rgb

def AddNoiseMosai(x, CRF_para, iCRF_para, I, B, Iinv, Binv, sigma_s, sigma_c, crf_index, pattern, opt = 1):
    w, h, c = x.shape
    temp_x = CRF_Map_opt(x, iCRF_para[crf_index] )

    sigma_s = np.reshape(sigma_s, (1, 1, c))
    noise_s_map = np.multiply(sigma_s, temp_x)
    noise_s = np.random.randn(w, h, c) * noise_s_map
    temp_x_n = temp_x + noise_s

    noise_c = np.zeros((w, h, c))
    for chn in range(3):
        noise_c [:, :, chn] = np.random.normal(0, sigma_c[chn], (w, h)) 

    temp_x_n = temp_x_n + noise_c
    temp_x_n = np.clip(temp_x_n, 0.0, 1.0)
    temp_x_n = CRF_Map_opt(temp_x_n, CRF_para[crf_index])

    if opt == 1:
        temp_x = CRF_Map_opt(temp_x, CRF_para[crf_index])

    B_b_n = mosaic_bayer(temp_x_n[:,:,::-1], pattern, 0)[0]
    lin_rgb_n = Demosaic(B_b_n, pattern)
    result = lin_rgb_n 
    if opt == 1:
        B_b = mosaic_bayer(temp_x[:,:,::-1], pattern, 0)[0]
        lin_rgb = Demosaic(B_b, pattern)
        diff = lin_rgb_n - lin_rgb
        result = x + diff
    
    return result
