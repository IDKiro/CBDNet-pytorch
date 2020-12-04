import os, time, scipy.io, shutil
import numpy as np
import torch
import torch.nn as nn
import argparse
import cv2

from model.cbdnet import Network
from utils import read_img, chw_to_hwc, hwc_to_chw

parser = argparse.ArgumentParser(description = 'Test')
parser.add_argument('input_filename', type=str)
parser.add_argument('output_filename', type=str)
args = parser.parse_args()

save_dir = './save_model/'

model = Network()
model.cuda()
model = nn.DataParallel(model)

model.eval()

if os.path.exists(os.path.join(save_dir, 'checkpoint.pth.tar')):
    # load existing model
    model_info = torch.load(os.path.join(save_dir, 'checkpoint.pth.tar'))
    model.load_state_dict(model_info['state_dict'])
else:
    print('Error: no trained model detected!')
    exit(1)

input_image = read_img(args.input_filename)
input_var =  torch.from_numpy(hwc_to_chw(input_image)).unsqueeze(0).cuda()

with torch.no_grad():
    _, output = model(input_var)

output_image = chw_to_hwc(output[0,...].cpu().numpy())
output_image = np.uint8(np.round(np.clip(output_image, 0, 1) * 255.))[: ,: ,::-1]

cv2.imwrite(args.output_filename, output_image)