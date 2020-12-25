import os
import random
import torch
import numpy as np
import glob
from torch.utils.data import Dataset

from utils import read_img, hwc_to_chw


def get_patch(imgs, patch_size):
	H = imgs[0].shape[0]
	W = imgs[0].shape[1]

	ps_temp = min(H, W, patch_size)

	xx = np.random.randint(0, W-ps_temp) if W > ps_temp else 0
	yy = np.random.randint(0, H-ps_temp) if H > ps_temp else 0

	for i in range(len(imgs)):
		imgs[i] = imgs[i][yy:yy+ps_temp, xx:xx+ps_temp, :]

	if np.random.randint(2, size=1)[0] == 1:
		for i in range(len(imgs)):
			imgs[i] = np.flip(imgs[i], axis=1)
	if np.random.randint(2, size=1)[0] == 1: 
		for i in range(len(imgs)):
			imgs[i] = np.flip(imgs[i], axis=0)
	if np.random.randint(2, size=1)[0] == 1:
		for i in range(len(imgs)):
			imgs[i] = np.transpose(imgs[i], (1, 0, 2))

	return imgs


class Real(Dataset):
	def __init__(self, root_dir, sample_num, patch_size=128):
		self.patch_size = patch_size

		folders = glob.glob(root_dir + '/*')
		folders.sort()

		self.clean_fns = [None] * sample_num
		for i in range(sample_num):
			self.clean_fns[i] = []

		for ind, folder in enumerate(folders):
			clean_imgs = glob.glob(folder + '/*GT_SRGB*')
			clean_imgs.sort()

			for clean_img in clean_imgs:
				self.clean_fns[ind % sample_num].append(clean_img)

	def __len__(self):
		l = len(self.clean_fns)
		return l

	def __getitem__(self, idx):
		clean_fn = random.choice(self.clean_fns[idx])

		clean_img = read_img(clean_fn)
		noise_img = read_img(clean_fn.replace('GT_SRGB', 'NOISY_SRGB'))

		if self.patch_size > 0:
			[clean_img, noise_img] = get_patch([clean_img, noise_img], self.patch_size)

		return hwc_to_chw(noise_img), hwc_to_chw(clean_img), np.zeros((3, self.patch_size, self.patch_size)), np.zeros((3, self.patch_size, self.patch_size))


class Syn(Dataset):
	def __init__(self, root_dir, sample_num, patch_size=128):
		self.patch_size = patch_size

		folders = glob.glob(root_dir + '/*')
		folders.sort()

		self.clean_fns = [None] * sample_num
		for i in range(sample_num):
			self.clean_fns[i] = []

		for ind, folder in enumerate(folders):
			clean_imgs = glob.glob(folder + '/*GT_SRGB*')
			clean_imgs.sort()

			for clean_img in clean_imgs:
				self.clean_fns[ind % sample_num].append(clean_img)

	def __len__(self):
		l = len(self.clean_fns)
		return l

	def __getitem__(self, idx):
		clean_fn = random.choice(self.clean_fns[idx])

		clean_img = read_img(clean_fn)
		noise_img = read_img(clean_fn.replace('GT_SRGB', 'NOISY_SRGB'))
		sigma_img = read_img(clean_fn.replace('GT_SRGB', 'SIGMA_SRGB')) / 15.	# inverse scaling

		if self.patch_size > 0:
			[clean_img, noise_img, sigma_img] = get_patch([clean_img, noise_img, sigma_img], self.patch_size)

		return hwc_to_chw(noise_img), hwc_to_chw(clean_img), hwc_to_chw(sigma_img), np.ones((3, self.patch_size, self.patch_size))