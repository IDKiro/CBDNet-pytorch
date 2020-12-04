import os
import random, math
import torch
import numpy as np
import glob
import cv2
from tqdm import tqdm
from skimage import io

from ISP_implement import ISP


if __name__ == '__main__':
	isp = ISP()

	source_dir = './source/'
	target_dir = './target/'

	if not os.path.isdir(target_dir):
		os.makedirs(target_dir)

	fns = glob.glob(os.path.join(source_dir, '*.png'))

	patch_size = 256

	for fn in tqdm(fns):
		img_rgb = cv2.imread(fn)[:, :, ::-1] / 255.0

		H = img_rgb.shape[0]
		W = img_rgb.shape[1]

		H_s = H // patch_size
		W_s = W // patch_size

		patch_id = 0

		for i in range(H_s):
			for j in range(W_s):
	
				yy = i * patch_size
				xx = j * patch_size

				patch_img_rgb = img_rgb[yy:yy+patch_size, xx:xx+patch_size, :]

				gt, noise, sigma = isp.noise_generate_srgb(patch_img_rgb)

				sigma = np.uint8(np.round(np.clip(sigma * 15 , 0, 1) * 255))	# store in uint8

				filename = os.path.basename(fn)
				foldername = filename.split('.')[0]

				out_folder = os.path.join(target_dir, foldername)

				if not os.path.isdir(out_folder):
					os.makedirs(out_folder)

				io.imsave(os.path.join(out_folder, 'GT_SRGB_%d_%d.png' % (i, j)), gt)
				io.imsave(os.path.join(out_folder, 'NOISY_SRGB_%d_%d.png' % (i, j)), noise)
				io.imsave(os.path.join(out_folder, 'SIGMA_SRGB_%d_%d.png' % (i, j)), sigma)
