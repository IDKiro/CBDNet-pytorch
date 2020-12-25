import random
import numpy as np
import cv2
import os
import json
import scipy.io
import math
import skimage

from modules import demosaicing_CFA_Bayer_Malvar2004, CRF_Map_Cython, ICRF_Map_Cython


class ISP:
    def __init__(self, curve_path='./'):
        filename = os.path.join(curve_path, 'metadata/201_CRF_data.mat')
        CRFs = scipy.io.loadmat(filename)
        self.I = CRFs['I']
        self.B = CRFs['B']
        filename = os.path.join(curve_path, 'metadata/dorfCurvesInv.mat')
        inverseCRFs = scipy.io.loadmat(filename)
        self.I_inv = inverseCRFs['invI']
        self.B_inv = inverseCRFs['invB']
        filename = os.path.join(curve_path, 'metadata/cameras.json')
        with open(filename, 'r') as load_f:
            self.cameras = json.load(load_f)

    def ICRF_Map(self, img):
        invI_temp = self.I_inv[self.icrf_index, :]
        invB_temp = self.B_inv[self.icrf_index, :]
        out = ICRF_Map_Cython(img.astype(np.double), invI_temp.astype(np.double), invB_temp.astype(np.double))
        return out

    def CRF_Map(self, img):
        I_temp = self.I[self.icrf_index, :]  # shape: (1024, 1)
        B_temp = self.B[self.icrf_index, :]  # shape: (1024, 1)
        out = CRF_Map_Cython(img.astype(np.double), I_temp.astype(np.double), B_temp.astype(np.double))
        return out

    def RGB2XYZ(self, img):
        xyz = skimage.color.rgb2xyz(img)
        return xyz

    def XYZ2RGB(self, img):
        rgb = skimage.color.xyz2rgb(img)
        return rgb

    def XYZ2CAM(self, img):
        M_xyz2cam = np.reshape(self.M_xyz2cam, (3, 3))
        M_xyz2cam = M_xyz2cam / np.tile(np.sum(M_xyz2cam, axis=1), [3, 1]).T
        cam = self.apply_cmatrix(img, M_xyz2cam)
        cam = np.clip(cam, 0, 1)
        return cam

    def CAM2XYZ(self, img):
        M_xyz2cam = np.reshape(self.M_xyz2cam, (3, 3))
        M_xyz2cam = M_xyz2cam / np.tile(np.sum(M_xyz2cam, axis=1), [3, 1]).T
        M_cam2xyz = np.linalg.inv(M_xyz2cam)
        xyz = self.apply_cmatrix(img, M_cam2xyz)
        xyz = np.clip(xyz, 0, 1)
        return xyz

    def apply_cmatrix(self, img, matrix):
        r = (matrix[0, 0] * img[:, :, 0] + matrix[0, 1] * img[:, :, 1]
             + matrix[0, 2] * img[:, :, 2])
        g = (matrix[1, 0] * img[:, :, 0] + matrix[1, 1] * img[:, :, 1]
             + matrix[1, 2] * img[:, :, 2])
        b = (matrix[2, 0] * img[:, :, 0] + matrix[2, 1] * img[:, :, 1]
             + matrix[2, 2] * img[:, :, 2])
        r = np.expand_dims(r, axis=2)
        g = np.expand_dims(g, axis=2)
        b = np.expand_dims(b, axis=2)
        results = np.concatenate((r, g, b), axis=2)
        return results

    def mosaic_bayer(self, rgb):
        # analysis pattern
        num = np.zeros(4, dtype=int)
        # the image store in OpenCV using BGR
        temp = list(self.find(self.pattern, 'R'))
        num[temp] = 0
        temp = list(self.find(self.pattern, 'G'))
        num[temp] = 1
        temp = list(self.find(self.pattern, 'B'))
        num[temp] = 2

        mosaic_img = np.zeros((rgb.shape[0], rgb.shape[1]), dtype=rgb.dtype)
        mosaic_img[0::2, 0::2] = rgb[0::2, 0::2, num[0]]
        mosaic_img[0::2, 1::2] = rgb[0::2, 1::2, num[1]]
        mosaic_img[1::2, 0::2] = rgb[1::2, 0::2, num[2]]
        mosaic_img[1::2, 1::2] = rgb[1::2, 1::2, num[3]]
        return mosaic_img

    def WB_Mask(self, img, fr_now, fb_now):
        wb_mask = np.ones(img.shape)
        if  self.pattern == 'RGGB':
            wb_mask[0::2, 0::2] = fr_now
            wb_mask[1::2, 1::2] = fb_now
        elif  self.pattern == 'BGGR':
            wb_mask[1::2, 1::2] = fr_now
            wb_mask[0::2, 0::2] = fb_now
        elif  self.pattern == 'GRBG':
            wb_mask[0::2, 1::2] = fr_now
            wb_mask[1::2, 0::2] = fb_now
        elif  self.pattern == 'GBRG':
            wb_mask[1::2, 0::2] = fr_now
            wb_mask[0::2, 1::2] = fb_now
        return wb_mask


    def find(self, str, ch):
        for i, ltr in enumerate(str):
            if ltr == ch:
                yield i

    def Demosaic(self, bayer):
        results = demosaicing_CFA_Bayer_Malvar2004(bayer, self.pattern)
        results = np.clip(results, 0, 1)
        return results
    
    def add_PG_noise(self, img):
        min_log = np.log([0.0001])
        max_log_s = np.log([0.01])

        log_sigma_s = min_log + np.random.rand(1) * (max_log_s - min_log)
        sigma_s = np.exp(log_sigma_s)

        line_c = 2.2 * log_sigma_s + 1.2
        offset_c = np.random.normal(0.0, 0.26)
        log_sigma_c = line_c + offset_c
        sigma_c = np.exp(log_sigma_c)

        # add noise
        sigma_total = np.sqrt(sigma_s * img + sigma_c)

        noisy_img = img +  \
            sigma_total * np.random.randn(img.shape[0], img.shape[1])
        return noisy_img, sigma_s, sigma_c

    def noise_generate_srgb(self, img, configs='DND'):
        # -------------------------- CAMERA SETTING --------------------------
        cameras = self.cameras[configs]
        camera = cameras[random.randint(0, len(cameras)-1)]

        self.icrf_index = random.randint(0, 200)

        try:
            self.pattern = camera['bayertype']
        except:
            self.pattern = random.choice(['GRBG', 'RGGB', 'GBRG', 'BGGR'])

        try:
            ColorMatrix1 = camera['ColorMatrix1']
            ColorMatrix2 = camera['ColorMatrix2']
            alpha = np.random.random_sample([1])
            self.M_xyz2cam = alpha * ColorMatrix1 + (1 - alpha) * ColorMatrix2
        except:
            cam_index = np.random.random((1, 4))
            cam_index = cam_index / np.sum(cam_index)
            self.M_xyz2cam = ([1.0234,-0.2969,-0.2266,-0.5625,1.6328,-0.0469,-0.0703,0.2188,0.6406] * cam_index[0, 0] + \
                            [0.4913,-0.0541,-0.0202,-0.613,1.3513,0.2906,-0.1564,0.2151,0.7183] * cam_index[0, 1] + \
                            [0.838,-0.263,-0.0639,-0.2887,1.0725,0.2496,-0.0627,0.1427,0.5438] * cam_index[0, 2] + \
                            [0.6596,-0.2079,-0.0562,-0.4782,1.3016,0.1933,-0.097,0.1581,0.5181] * cam_index[0, 3])

        try:
            min_offset = -0.05
            max_offset = 0.05
            AsShotNeutral = camera['AsShotNeutral']
            self.fr_now = AsShotNeutral[0] + random.uniform(min_offset, max_offset)
            self.fb_now = AsShotNeutral[2] + random.uniform(min_offset, max_offset)
        except:
            min_fc = 0.75
            max_fc = 1
            self.fr_now = random.uniform(min_fc, max_fc)
            self.fb_now = random.uniform(min_fc, max_fc)
        
        try:
            blacklevel = camera['blacklevel']
            whitelevel = camera['whitelevel']
        except:
            blacklevel = 254
            whitelevel = 4094
        
        # -------------------------- INVERSE ISP PROCESS --------------------------
        img_rgb = img
        # Step 1 : inverse tone mapping
        img_L = self.ICRF_Map(img_rgb)
        # Step 2 : from RGB to XYZ
        img_XYZ = self.RGB2XYZ(img_L)
        # Step 3: from XYZ to Cam
        img_Cam = self.XYZ2CAM(img_XYZ)
        # Step 4: Mosaic
        img_mosaic = self.mosaic_bayer(img_Cam)
        # Step 5: inverse White Balance
        wb_mask = self.WB_Mask(img_mosaic, self.fr_now, self.fb_now)
        img_mosaic = img_mosaic * wb_mask
        img_mosaic_gt = img_mosaic

        # -------------------------- POISSON-GAUSSIAN NOISE ON RAW --------------------------
        img_mosaic_noise, sigma_s, sigma_c = self.add_PG_noise(img_mosaic)

        # -------------------------- QUANTIZATION NOISE AND CLIPPING EFFECT ON RAW --------------------------
        upper_bound = math.pow(2, math.ceil(math.log(whitelevel + 1, 2))) - 1
        img_mosaic_noise = np.clip(np.floor(img_mosaic_noise * (whitelevel - blacklevel) + blacklevel), 0, upper_bound)
        img_mosaic_noise = (img_mosaic_noise - blacklevel) / (whitelevel - blacklevel)
        img_mosaic_gt = np.clip(np.floor(img_mosaic_gt * (whitelevel - blacklevel) + blacklevel), 0, upper_bound)
        img_mosaic_gt = (img_mosaic_gt - blacklevel) / (whitelevel - blacklevel)

        # -------------------------- ISP PROCESS --------------------------
        # Step 5 : White Balance
        wb_mask = self.WB_Mask(img_mosaic_noise, 1/self.fr_now, 1/self.fb_now)
        img_mosaic_noise = img_mosaic_noise * wb_mask
        img_mosaic_noise = np.clip(img_mosaic_noise, 0, 1)
        img_mosaic_gt = img_mosaic_gt * wb_mask
        img_mosaic_gt = np.clip(img_mosaic_gt, 0, 1)
        # Step 4 : Demosaic
        img_demosaic = self.Demosaic(img_mosaic_noise)
        img_demosaic_gt = self.Demosaic(img_mosaic_gt)
        # Step 3 : from Cam to XYZ
        img_IXYZ = self.CAM2XYZ(img_demosaic)
        img_IXYZ_gt = self.CAM2XYZ(img_demosaic_gt)
        # Step 2 : frome XYZ to RGB
        img_IL = self.XYZ2RGB(img_IXYZ)
        img_IL_gt = self.XYZ2RGB(img_IXYZ_gt)
        # Step 1 : tone mapping
        img_Irgb = self.CRF_Map(img_IL)
        img_Irgb_gt = self.CRF_Map(img_IL_gt)

        # -------------------------- QUANTIZATION NOISE AND CLIPPING EFFECT ON RGB --------------------------
        noise = np.clip(img_Irgb, 0, 1) - np.clip(img_Irgb_gt, 0, 1)
        img_Irgb_gt = np.clip(img_rgb, 0, 1)
        img_Irgb = np.clip((img_rgb + noise), 0, 1)

        sigma_total = np.sqrt(sigma_s * img + sigma_c)  # noise level map

        return np.uint8(np.round(img_Irgb_gt*255)), np.uint8(np.round(img_Irgb*255)), sigma_total
