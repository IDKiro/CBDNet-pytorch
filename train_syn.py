from __future__ import division
from __future__ import print_function
import os, time, scipy.io, shutil
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import glob
import re
import cv2

from utils.noise import *
from utils.common import *
from model import *


def load_CRF():
    CRF = scipy.io.loadmat('matdata/201_CRF_data.mat')
    iCRF = scipy.io.loadmat('matdata/dorfCurvesInv.mat')
    B_gl = CRF['B']
    I_gl = CRF['I']
    B_inv_gl = iCRF['invB']
    I_inv_gl = iCRF['invI']

    if os.path.exists('matdata/201_CRF_iCRF_function.mat')==0:
        CRF_para = np.array(CRF_function_transfer(I_gl, B_gl))
        iCRF_para = 1. / CRF_para
        scipy.io.savemat('matdata/201_CRF_iCRF_function.mat', {'CRF':CRF_para, 'iCRF':iCRF_para})
    else:
        Bundle = scipy.io.loadmat('matdata/201_CRF_iCRF_function.mat')
        CRF_para = Bundle['CRF']
        iCRF_para = Bundle['iCRF']

    return CRF_para, iCRF_para, I_gl, B_gl, I_inv_gl, B_inv_gl

def DataAugmentation(temp_origin_img, temp_noise_img):
    if np.random.randint(2, size=1)[0] == 1:
        temp_origin_img = np.flip(temp_origin_img, axis=1)
        temp_noise_img = np.flip(temp_noise_img, axis=1)
    if np.random.randint(2, size=1)[0] == 1: 
        temp_origin_img = np.flip(temp_origin_img, axis=0)
        temp_noise_img = np.flip(temp_noise_img, axis=0)
    if np.random.randint(2, size=1)[0] == 1:
        temp_origin_img = np.transpose(temp_origin_img, (1, 0, 2))
        temp_noise_img = np.transpose(temp_noise_img, (1, 0, 2))
    
    return temp_origin_img, temp_noise_img

def load_checkpoint(checkpoint_dir):
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

    return model, optimizer, cur_epoch

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
	torch.save(state, checkpoint_dir + 'checkpoint.pth.tar')
	if is_best:
		shutil.copyfile(checkpoint_dir + 'checkpoint.pth.tar',checkpoint_dir + 'model_best.pth.tar')

def adjust_learning_rate(optimizer, epoch, lr_update_freq):
	if not epoch % lr_update_freq and epoch:
		for param_group in optimizer.param_groups:
			param_group['lr'] = param_group['lr'] * 0.1
	return optimizer


if __name__ == '__main__':
    input_dir = './dataset/synthetic/'
    checkpoint_dir = './checkpoint/synthetic/'
    result_dir = './result/synthetic/'

    save_freq = 100
    lr_update_freq = 100

    CRF_para, iCRF_para, I_gl, B_gl, I_inv_gl, B_inv_gl = load_CRF()

    train_fns = glob.glob(input_dir + '*.bmp')

    origin_imgs = [None] * len(train_fns)
    noise_imgs = [None] * len(train_fns)

    for i in range(len(train_fns)):
        origin_imgs[i] = []
        noise_imgs[i] = []

    model, optimizer, cur_epoch = load_checkpoint(checkpoint_dir)

    criterion = fixed_loss()
    criterion = criterion.cuda()

    for epoch in range(cur_epoch, 201):
        cnt=0
        losses = AverageMeter()
        optimizer = adjust_learning_rate(optimizer, epoch, lr_update_freq)
        model.train()
        
        for ind in np.random.permutation(len(train_fns)):
            train_fn = train_fns[ind]

            if not len(origin_imgs[ind]):
                origin_img = cv2.imread(train_fn)
                origin_img = origin_img[:,:,::-1] / 255.0
                origin_imgs[ind] = np.array(origin_img).astype('float32')

            # re-add noise
            if epoch % save_freq == 0:
                noise_imgs[ind] = []

            if len(noise_imgs[ind]) < 1:
                noise_img = AddRealNoise(origin_imgs[ind][:, :, :], CRF_para, iCRF_para, I_gl, B_gl, I_inv_gl, B_inv_gl)
                noise_imgs[ind].append(noise_img)

            st = time.time()
            for nind in np.random.permutation(len(noise_imgs[ind])):
                temp_origin_img = origin_imgs[ind]
                temp_noise_img = noise_imgs[ind][nind]
                temp_origin_img, temp_noise_img = DataAugmentation(temp_origin_img, temp_noise_img)
                noise_level = temp_noise_img - temp_origin_img

                temp_noise_img_chw = hwc_to_chw(temp_noise_img)
                temp_origin_img_chw = hwc_to_chw(temp_origin_img)
                noise_level_chw = hwc_to_chw(noise_level)

                cnt += 1
                st = time.time()

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

                loss = criterion(output, target_var, noise_level_est, noise_level_var, 1)
                losses.update(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                print('[{0}][{1}]\t'
                    'lr: {lr:.5f}\t'
                    'Loss: {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Time: {time:.3f}'.format(
                    epoch, cnt,
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
