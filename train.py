from __future__ import division
from __future__ import print_function
import os, time, scipy.io, shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import glob
import re
import cv2

from utils import *
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
    input_syn_dir = './dataset/synthetic/'
    input_real_dir = './dataset/real/'
    checkpoint_dir = './checkpoint/all/'
    result_dir = './result/all/'

    PS = 512
    REAPET = 10
    save_freq = 50
    lr_update_freq = 100

    CRF_para, iCRF_para, I_gl, B_gl, I_inv_gl, B_inv_gl = load_CRF()

    train_syn_fns = glob.glob(input_syn_dir + '*.bmp')
    train_real_fns = glob.glob(input_real_dir + 'Batch_*')

    origin_syn_imgs = [None] * len(train_syn_fns)
    noise_syn_imgs = [None] * len(train_syn_fns)
    noise_syn_levels = [None] * len(train_syn_fns)

    origin_real_imgs = [None] * len(train_real_fns)
    noise_real_imgs = [None] * len(train_real_fns)

    for i in range(len(train_syn_fns)):
        origin_syn_imgs[i] = []
        noise_syn_imgs[i] = []
        noise_syn_levels[i] = []

    for i in range(len(train_real_fns)):
        origin_real_imgs[i] = []
        noise_real_imgs[i] = []

    model, optimizer, cur_epoch = load_checkpoint(checkpoint_dir)

    criterion = fixed_loss()
    criterion = criterion.cuda()

    for epoch in range(cur_epoch, 201):
        cnt=0
        losses = AverageMeter()
        optimizer = adjust_learning_rate(optimizer, epoch, lr_update_freq)
        model.train()
        
        print('Training on synthetic noisy images...')
        for ind in np.random.permutation(len(train_syn_fns)):
            train_syn_fn = train_syn_fns[ind]

            if not len(origin_syn_imgs[ind]):
                origin_syn_img = cv2.imread(train_syn_fn)
                origin_syn_img = origin_syn_img[:,:,::-1] / 255.0
                origin_syn_imgs[ind] = np.array(origin_syn_img).astype('float32')

            # re-add noise
            if epoch % save_freq == 0:
                noise_syn_imgs[ind] = []
                noise_syn_levels[ind] = []

            if len(noise_syn_imgs[ind]) < 1:
                noise_syn_img, noise_syn_level = AddRealNoise(origin_syn_imgs[ind][:, :, :], CRF_para, iCRF_para, I_gl, B_gl, I_inv_gl, B_inv_gl)
                noise_syn_imgs[ind].append(noise_syn_img)
                noise_syn_levels[ind].append(noise_syn_level)

            st = time.time()
            for nind in np.random.permutation(len(noise_syn_imgs[ind])):
                temp_origin_img = origin_syn_imgs[ind]
                temp_noise_img = noise_syn_imgs[ind][nind]
                temp_noise_level = noise_syn_levels[ind][nind]

                if np.random.randint(2, size=1)[0] == 1:
                    temp_origin_img = np.flip(temp_origin_img, axis=1)
                    temp_noise_img = np.flip(temp_noise_img, axis=1)
                    temp_noise_level = np.flip(temp_noise_level, axis=1)
                if np.random.randint(2, size=1)[0] == 1: 
                    temp_origin_img = np.flip(temp_origin_img, axis=0)
                    temp_noise_img = np.flip(temp_noise_img, axis=0)
                    temp_noise_level = np.flip(temp_noise_level, axis=0)
                if np.random.randint(2, size=1)[0] == 1:
                    temp_origin_img = np.transpose(temp_origin_img, (1, 0, 2))
                    temp_noise_img = np.transpose(temp_noise_img, (1, 0, 2))
                    temp_noise_level = np.transpose(temp_noise_level, (1, 0, 2))

                temp_noise_img_chw = hwc_to_chw(temp_noise_img)
                temp_origin_img_chw = hwc_to_chw(temp_origin_img)
                temp_noise_level_chw = hwc_to_chw(temp_noise_level)

                cnt += 1
                st = time.time()

                input_var = torch.from_numpy(temp_noise_img_chw.copy()).type(torch.FloatTensor).unsqueeze(0)
                target_var = torch.from_numpy(temp_origin_img_chw.copy()).type(torch.FloatTensor).unsqueeze(0)
                noise_level_var = torch.from_numpy(temp_noise_level_chw.copy()).type(torch.FloatTensor).unsqueeze(0)
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
        
        print('Training on real noisy images...')
        for r in range(REAPET):
            for ind in np.random.permutation(len(train_real_fns)):
                train_real_fn = train_real_fns[ind]

                if not len(origin_real_imgs[ind]):
                    train_real_origin_fns = glob.glob(train_real_fn + '/*Reference.bmp')
                    train_real_noise_fns = glob.glob(train_real_fn + '/*Noisy.bmp')

                    origin_real_img = cv2.imread(train_real_origin_fns[0])
                    origin_real_img = origin_real_img[:,:,::-1] / 255.0
                    origin_real_imgs[ind] = np.array(origin_real_img).astype('float32')

                    for train_real_noise_fn in train_real_noise_fns:
                        noise_real_img = cv2.imread(train_real_noise_fn)
                        noise_real_img = noise_real_img[:,:,::-1] / 255.0
                        noise_real_img = np.array(noise_real_img).astype('float32')
                        noise_real_imgs[ind].append(noise_real_img)


                st = time.time()
                for nind in np.random.permutation(len(noise_real_imgs[ind])):
                    H = origin_real_imgs[ind].shape[0]
                    W = origin_real_imgs[ind].shape[1]

                    ps_temp = min(H, W, PS) - 1

                    xx = np.random.randint(0, W-ps_temp)
                    yy = np.random.randint(0, H-ps_temp)
                    
                    temp_origin_img = origin_real_imgs[ind][yy:yy+ps_temp, xx:xx+ps_temp, :]
                    temp_noise_img = noise_real_imgs[ind][nind][yy:yy+ps_temp, xx:xx+ps_temp, :]

                    if np.random.randint(2, size=1)[0] == 1:
                        temp_origin_img = np.flip(temp_origin_img, axis=1)
                        temp_noise_img = np.flip(temp_noise_img, axis=1)
                    if np.random.randint(2, size=1)[0] == 1: 
                        temp_origin_img = np.flip(temp_origin_img, axis=0)
                        temp_noise_img = np.flip(temp_noise_img, axis=0)
                    if np.random.randint(2, size=1)[0] == 1:
                        temp_origin_img = np.transpose(temp_origin_img, (1, 0, 2))
                        temp_noise_img = np.transpose(temp_noise_img, (1, 0, 2))

                    temp_noise_img_chw = hwc_to_chw(temp_noise_img)
                    temp_origin_img_chw = hwc_to_chw(temp_origin_img)

                    cnt += 1
                    st = time.time()

                    input_var = torch.from_numpy(temp_noise_img_chw.copy()).type(torch.FloatTensor).unsqueeze(0)
                    target_var = torch.from_numpy(temp_origin_img_chw.copy()).type(torch.FloatTensor).unsqueeze(0)
                    input_var, target_var = input_var.cuda(), target_var.cuda()

                    noise_level_est, output = model(input_var)

                    loss = criterion(output, target_var, noise_level_est, 0, 0)
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
                        scipy.misc.toimage(temp*255, high=255, low=0, cmin=0, cmax=255).save(result_dir + '%04d/train_%d_%d.jpg'%(epoch, ind + len(train_syn_fns) + r * len(train_real_fns), nind))

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict()}, is_best=0)
