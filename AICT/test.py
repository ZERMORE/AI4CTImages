import os
from datasets import read_dicom_file
import scipy
import numpy as np
import random
import torch.nn
# from datasets import dicom_file
from torch.utils.data import DataLoader
from torch.backends import cudnn
# from skimage.measure import compare_ssim as ssim
# from skimage.measure import compare_psnr as psnr
import time
import torch.optim as optim
import argparse
from losses import *
from models import *
from utils import *
from datetime import datetime
from pytools import *
import torch.nn.functional as F
from skimage import io

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


############TCIA Data###########
# results_save_dir = '/mnt/no3/yikun/ConvTrans/TCIA/Head/N284/12-23-2021-NA-NA-22739/UNet2d_AWMU/'
# make_dirs(results_save_dir)

# ldct_path = '/mnt/no3/yikun/ConvTrans/TCIA/Head/N284/12-23-2021-NA-NA-22739/1.000000-Low Dose Images-37908/'
# ldct_list = os.listdir(ldct_path)
# ldct_list.sort()

# results_save_dir = '/mnt/no3/yikun/ConvTrans/TCIA/Phantom/1mm B30/ACR_quarter_1mm/UNet2d_AWMU/'
# make_dirs(results_save_dir)

# ldct_path = '/mnt/no3/yikun/ConvTrans/TCIA/Phantom/1mm B30/ACR_quarter_1mm/quarter_1mm/'
# ldct_list = os.listdir(ldct_path)
# ldct_list.sort()

# results_save_dir = '/mnt/no3/yikun/ConvTrans/TCIA/Phantom/1mm D45/ACR_quarter_1mm_sharp/UNet2d_AWMU/'
# make_dirs(results_save_dir)

# ldct_path = '/mnt/no3/yikun/ConvTrans/TCIA/Phantom/1mm D45/ACR_quarter_1mm_sharp/quarter_1mm_sharp/'
# ldct_list = os.listdir(ldct_path)
# ldct_list.sort()

# results_save_dir = '/mnt/no3/yikun/ConvTrans/TCIA/Chest/C124/12-23-2021-NA-NA-73316/UNet2d_AWMU/'
# make_dirs(results_save_dir)

ldct_path = '/mnt/no3/yikun/ConvTrans/TCIA/C067/'
ldct_list = os.listdir(ldct_path)
ldct_list.sort()

results_save_dir = '/mnt/no3/yikun/ConvTrans/TCIA/C067UNet2D/'
make_dirs(results_save_dir)

epoch = 100
model_dir = '/mnt/no3/yikun/ConvTrans/Code/runs/UNet2d_AWMU/checkpoints/model_at_epoch_' + str(epoch).rjust(3, '0') + '.dat'
checkpoint = torch.load(model_dir)

model = UNet2d()
model = load_model(model, checkpoint).cuda()
model.eval()

for i, file_name in enumerate(ldct_list):


    ldct_name = ldct_path + file_name
    ldct_info = pydicom.read_file(ldct_name)
    ldct_slice = np.float32(ldct_info.pixel_array)
    ldct_slice[ldct_slice<0] = 0

    ldct_slice_tensor = ldct_slice[np.newaxis, np.newaxis, ...]

    ldCT = torch.FloatTensor(ldct_slice_tensor).cuda()
    with torch.no_grad():

        img_net = model(ldCT)
        pred_img = np.squeeze(img_net.data.cpu().numpy())
        pred_img[pred_img<0] = 0

        # pred_img = pred_img * 0.8 + ldct_slice * 0.2


        ldct_info.PixelData = np.uint16(pred_img)
        ldct_info.save_as(results_save_dir + file_name, write_like_original=True)






# ############UIH-C-arm Data###########

# root_dir = '/mnt/no3/yikun/ConvTrans/UIH-C-arm/'
# model_name = 'UNet2d_AWMU'
# results_save_dir = root_dir + 'UNet2d_AWMU_body/'
# make_dirs(results_save_dir)

# epoch = 100
# model_dir = './runs/' + model_name + '/checkpoints/model_at_epoch_' + str(epoch).rjust(3, '0') + '.dat'
# checkpoint = torch.load(model_dir)

# model = UNet2d()
# model = load_model(model, checkpoint).cuda()
# model.eval()

# ldct_path = root_dir + 'body_ld_fdk/'
# ldct_list = os.listdir(ldct_path)
# ldct_list.sort()

# for i, file_name in enumerate(ldct_list):

#     ldct = io.imread(ldct_path + file_name) + 1000
#     ldct[ldct<0] = 0

#     ldct_slice = ldct[np.newaxis, np.newaxis, ...]

#     ldCT = torch.FloatTensor(ldct_slice).cuda()

#     with torch.no_grad():

#         img_net = model(ldCT)
#         pred_img = np.squeeze(img_net.data.cpu().numpy())
#         pred_img[pred_img<0] = 0

#         # pred_img = pred_img * 0.7 + ldct * 0.3

#         io.imsave(results_save_dir+file_name, pred_img-1000)







# ############Generated Data###########
# results_save_dir = '/mnt/no3/yikun/ConvTrans/Gen_Data/UNet2d_AWMU/'
# make_dirs(results_save_dir)

# ldct_path = '/mnt/no3/yikun/ConvTrans/Gen_Data/'
# ldct_list = os.listdir(ldct_path)
# ldct_list.sort()

# epoch = 100
# model_dir = '/mnt/no3/yikun/ConvTrans/Code/runs/UNet2d_AWMU/checkpoints/model_at_epoch_' + str(epoch).rjust(3, '0') + '.dat'
# checkpoint = torch.load(model_dir)

# model = UNet2d()
# model = load_model(model, checkpoint).cuda()
# model.eval()

# for i, file_name in enumerate(ldct_list):


#     ldct_name = ldct_path + file_name
#     ldct_info = pydicom.read_file(ldct_name)
#     ldct_slice = np.float32(ldct_info.pixel_array)
#     ldct_slice[ldct_slice<0] = 0

#     ldct_slice_tensor = ldct_slice[np.newaxis, np.newaxis, ...]

#     ldCT = torch.FloatTensor(ldct_slice_tensor).cuda()
#     with torch.no_grad():

#         img_net = model(ldCT)
#         pred_img = np.squeeze(img_net.data.cpu().numpy())
#         pred_img[pred_img<0] = 0

#         # pred_img = pred_img * 0.8 + ldct_slice * 0.2


#         ldct_info.PixelData = np.uint16(pred_img)
#         ldct_info.save_as(results_save_dir + file_name, write_like_original=True)