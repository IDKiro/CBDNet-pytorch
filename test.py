from __future__ import division
from __future__ import print_function
import os, time, scipy.io, shutil
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms, utils
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import glob
import re
import cv2

from utils import *
from model import *


def hwc_to_chw(img):
    return np.transpose(img, axes=[2, 0, 1])

def chw_to_hwc(img):
    return np.transpose(img, axes=[1, 2, 0])

input_dir = './dataset/test/'
checkpoint_dir = './checkpoint/'
result_dir = './result/'

test_fns = glob.glob(input_dir + '*.bmp')

# model load
if os.path.exists(checkpoint_dir + 'checkpoint.pth.tar'):
    # load existing model
    model_info = torch.load(checkpoint_dir + 'checkpoint.pth.tar')
    print('==> loading existing model:', checkpoint_dir + 'checkpoint.pth.tar')
    model = CBDNet()
    # model.cuda()
    model.load_state_dict(model_info['state_dict'])
else:
    print('Error: No trained model detected!')

if not os.path.isdir(result_dir + 'test/'):
    os.makedirs(result_dir + 'test/')


for ind, test_fn in enumerate(test_fns):
    model.eval()
    with torch.no_grad():
        print(test_fn)
        noisy_img = cv2.imread(test_fn)
        noisy_img = noisy_img[:,:,::-1] / 255.0
        noisy_img = np.array(noisy_img).astype('float32')
        temp_noisy_img = noisy_img
        temp_noisy_img_chw = hwc_to_chw(temp_noisy_img)

        input_var = torch.from_numpy(temp_noisy_img_chw.copy()).type(torch.FloatTensor).unsqueeze(0)
        # input_var = input_var.cuda()
        _, output = model(input_var)

        output_np = output.squeeze().cpu().detach().numpy()
        output_np = chw_to_hwc(np.clip(output_np, 0, 1))

        temp = np.concatenate((temp_noisy_img, output_np), axis=1)
        scipy.misc.toimage(temp*255, high=255, low=0, cmin=0, cmax=255).save(result_dir + 'test/test_%d.jpg'%(ind))

