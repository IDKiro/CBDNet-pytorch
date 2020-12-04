import numpy as np
import cv2


class AverageMeter(object):
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count


class ListAverageMeter(object):
	"""Computes and stores the average and current values of a list"""
	def __init__(self):
		self.len = 10000  # set up the maximum length
		self.reset()

	def reset(self):
		self.val = [0] * self.len
		self.avg = [0] * self.len
		self.sum = [0] * self.len
		self.count = 0

	def set_len(self, n):
		self.len = n
		self.reset()

	def update(self, vals, n=1):
		assert len(vals) == self.len, 'length of vals not equal to self.len'
		self.val = vals
		for i in range(self.len):
			self.sum[i] += self.val[i] * n
		self.count += n
		for i in range(self.len):
			self.avg[i] = self.sum[i] / self.count
			

def read_img(filename, if_gray=False):
	if if_gray:
		img = cv2.imread(filename, 0)
		img = np.expand_dims(img, 2) / 255.0
	else:
		img = cv2.imread(filename)
		img = img[:,:,::-1] / 255.0
		
	img = np.array(img).astype('float32')

	return img


def hwc_to_chw(img):
	return np.transpose(img, axes=[2, 0, 1]).astype('float32')


def chw_to_hwc(img):
	return np.transpose(img, axes=[1, 2, 0]).astype('float32')
	

def data_augmentation(image, mode):
	'''
	Performs data augmentation of the input image
	Input:
		image: a cv2 (OpenCV) image
		mode: int. Choice of transformation to apply to the image
				0 - no transformation
				1 - flip up and down
				2 - rotate counterwise 90 degree
				3 - rotate 90 degree and flip up and down
				4 - rotate 180 degree
				5 - rotate 180 degree and flip
				6 - rotate 270 degree
				7 - rotate 270 degree and flip
	'''
	if mode == 0:
		# original
		out = image
	elif mode == 1:
		# flip up and down
		out = np.flipud(image)
	elif mode == 2:
		# rotate counterwise 90 degree
		out = np.rot90(image)
	elif mode == 3:
		# rotate 90 degree and flip up and down
		out = np.rot90(image)
		out = np.flipud(out)
	elif mode == 4:
		# rotate 180 degree
		out = np.rot90(image, k=2)
	elif mode == 5:
		# rotate 180 degree and flip
		out = np.rot90(image, k=2)
		out = np.flipud(out)
	elif mode == 6:
		# rotate 270 degree
		out = np.rot90(image, k=3)
	elif mode == 7:
		# rotate 270 degree and flip
		out = np.rot90(image, k=3)
		out = np.flipud(out)
	else:
		raise Exception('Invalid choice of image transformation')

	return out


def inverse_data_augmentation(image, mode):
	'''
	Performs inverse data augmentation of the input image
	'''
	if mode == 0:
		# original
		out = image
	elif mode == 1:
		out = np.flipud(image)
	elif mode == 2:
		out = np.rot90(image, axes=(1,0))
	elif mode == 3:
		out = np.flipud(image)
		out = np.rot90(out, axes=(1,0))
	elif mode == 4:
		out = np.rot90(image, k=2, axes=(1,0))
	elif mode == 5:
		out = np.flipud(image)
		out = np.rot90(out, k=2, axes=(1,0))
	elif mode == 6:
		out = np.rot90(image, k=3, axes=(1,0))
	elif mode == 7:
		# rotate 270 degree and flip
		out = np.flipud(image)
		out = np.rot90(out, k=3, axes=(1,0))
	else:
		raise Exception('Invalid choice of image transformation')

	return out