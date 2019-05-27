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

class fixed_loss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, out_image, gt_image, est_noise, gt_noise):
        h_x = est_noise.size()[2]
        w_x = est_noise.size()[3]
        count_h = self._tensor_size(est_noise[:,:,1:,:])
        count_w = self._tensor_size(est_noise[:,:,:,1:])
        h_tv = torch.pow((est_noise[:,:,1:,:]-est_noise[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((est_noise[:,:,:,1:]-est_noise[:,:,:,:w_x-1]),2).sum()
        tvloss = h_tv/count_h + w_tv/count_w

        loss = torch.mean(torch.pow((out_image - gt_image), 2)) + \
                0.5 * torch.mean(torch.mul(torch.abs(0.3 - F.relu(gt_noise - est_noise)), torch.pow(est_noise - gt_noise, 2))) + \
                0.05 * tvloss
        return loss

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]

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

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
	torch.save(state, checkpoint_dir + 'checkpoint.pth.tar')
	if is_best:
		shutil.copyfile(checkpoint_dir + 'checkpoint.pth.tar',checkpoint_dir + 'model_best.pth.tar')

def adjust_learning_rate(optimizer, epoch, lr_update_freq):
	if not epoch % lr_update_freq and epoch:
		for param_group in optimizer.param_groups:
			param_group['lr'] = param_group['lr'] * 0.1
	return optimizer

def hwc_to_chw(img):
    return np.transpose(img, axes=[2, 0, 1])

def chw_to_hwc(img):
    return np.transpose(img, axes=[1, 2, 0])

input_dir = './dataset/real/'
checkpoint_dir = './checkpoint/'
result_dir = './result/'

ps = 512
save_freq = 100
lr_update_freq = 1000

train_fns = glob.glob(input_dir + 'Batch_*')

origin_imgs = [None] * len(train_fns)
noise_imgs = [None] * len(train_fns)

for i in range(len(train_fns)):
    origin_imgs[i] = []
    noise_imgs[i] = []

# model setting
if os.path.exists(checkpoint_dir + 'checkpoint.pth.tar'):
    # load existing model
    model_info = torch.load(checkpoint_dir + 'checkpoint.pth.tar')
    print('==> loading existing model:', checkpoint_dir + 'checkpoint.pth.tar')
    model = CBDNet()
    model.cuda()
    model.load_state_dict(model_info['state_dict'])
    optimizer = torch.optim.Adam(model.parameters())
    optimizer.load_state_dict(model_info['optimizer'])
    cur_epoch = model_info['epoch']
else:
    # create model
    model = CBDNet()
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    cur_epoch = 0

criterion = fixed_loss()
criterion = criterion.cuda()

for epoch in range(cur_epoch, 2001):
    cnt=0
    losses = AverageMeter()
    model.train()

    for ind in np.random.permutation(len(train_fns)):
        train_fn = train_fns[ind]

        if not len(origin_imgs[ind]):
            train_origin_fns = glob.glob(train_fn + '/*Reference.bmp')
            train_noise_fns = glob.glob(train_fn + '/*Noisy.bmp')

            origin_img = cv2.imread(train_origin_fns[0])
            origin_img = origin_img[:,:,::-1] / 255.0
            origin_imgs[ind] = np.array(origin_img).astype('float32')

            for train_noise_fn in train_noise_fns:
                noise_img = cv2.imread(train_noise_fn)
                noise_img = noise_img[:,:,::-1] / 255.0
                noise_img = np.array(noise_img).astype('float32')
                noise_imgs[ind].append(noise_img)


        st = time.time()
        for nind in np.random.permutation(len(noise_imgs[ind])):
            H = origin_imgs[ind].shape[0]
            W = origin_imgs[ind].shape[1]

            ps_temp = min(H, W, ps) - 1

            xx = np.random.randint(0, W-ps_temp)
            yy = np.random.randint(0, H-ps_temp)
            
            temp_origin_img = origin_imgs[ind][yy:yy+ps_temp, xx:xx+ps_temp, :]
            temp_noise_img = noise_imgs[ind][nind][yy:yy+ps_temp, xx:xx+ps_temp, :]

            if np.random.randint(2, size=1)[0] == 1:
                temp_origin_img = np.flip(temp_origin_img, axis=1)
                temp_noise_img = np.flip(temp_noise_img, axis=1)
            if np.random.randint(2, size=1)[0] == 1: 
                temp_origin_img = np.flip(temp_origin_img, axis=0)
                temp_noise_img = np.flip(temp_noise_img, axis=0)
            if np.random.randint(2, size=1)[0] == 1:
                temp_origin_img = np.transpose(temp_origin_img, (1, 0, 2))
                temp_noise_img = np.transpose(temp_noise_img, (1, 0, 2))
            
            noise_level = temp_noise_img - temp_origin_img

            temp_noise_img_chw = hwc_to_chw(temp_noise_img)
            temp_origin_img_chw = hwc_to_chw(temp_origin_img)
            noise_level_chw = hwc_to_chw(noise_level)

            cnt += 1
            st = time.time()

            optimizer = adjust_learning_rate(optimizer, epoch, lr_update_freq)

            input_var = torch.autograd.Variable(
                torch.from_numpy(temp_noise_img_chw.copy()).type(torch.FloatTensor).unsqueeze(0)
                )
            target_var = torch.autograd.Variable(
                torch.from_numpy(temp_origin_img_chw.copy()).type(torch.FloatTensor).unsqueeze(0)
                )
            noise_level_var = torch.autograd.Variable(
                torch.from_numpy(noise_level_chw.copy()).type(torch.FloatTensor).unsqueeze(0)
                )
            input_var, target_var, noise_level_var = input_var.cuda(), target_var.cuda(), noise_level_var.cuda()

            noise_level_est, output = model(input_var)

            loss = criterion(output, target_var, noise_level_est, noise_level_var)
            losses.update(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print('[{0}][{1}/{2}]\t'
                'lr: {lr:.5f}\t'
                'Loss: {loss.val:.4f} ({loss.avg:.4f})\t'
                'Time: {time:.3f}'.format(
                epoch, cnt, len(train_fns),
                lr=optimizer.param_groups[-1]['lr'],
                loss=losses,
                time=time.time()-st))

            if epoch % save_freq == 0:
                if not os.path.isdir(result_dir + '%04d'%epoch):
                    os.makedirs(result_dir + '%04d'%epoch)

                output_np = output.squeeze().cpu().detach().numpy()
                output_np = chw_to_hwc(np.clip(output_np, 0, 1))

                temp = np.concatenate((temp_origin_img, temp_noise_img, output_np), axis=1)
                scipy.misc.toimage(temp*255, high=255, low=0, cmin=0, cmax=255).save(result_dir + '%04d/train_%d_%d.jpg'%(epoch, ind, nind))
    
    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer' : optimizer.state_dict()}, is_best=0)