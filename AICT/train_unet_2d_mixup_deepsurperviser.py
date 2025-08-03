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
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def train(train_loader, model, loss_mse, loss_cl, loss_ssim, optimizer, scheduler, writer, epoch):

    batch_time = AverageMeter()
    loss_mse_scalar = AverageMeter()
    loss2_scalar = AverageMeter()
    loss3_scalar = AverageMeter()
    loss4_scalar = AverageMeter()
    loss5_scalar = AverageMeter()
    model.train()
    end = time.time()

    step = 0

    for data in train_loader:

        RDCTImg = data["rdct"]
        RDCTImg = RDCTImg.cuda()

        LDCTImg = data["ldct"]
        LDCTImg = LDCTImg.cuda()

        if epoch>5:

            RDCTImg, LDCTImg = MixUp_AUG().aug(RDCTImg, LDCTImg)

        # print(LDCTImg.shape)
        # LDCTImg[LDCTImg < 0] = 0

        s5, s4, s3, s2, predictImg = model(LDCTImg)
        # fdkImgSave = fdkImg.data.cpu().numpy()
        # fdkImgSave.astype(np.float32).tofile('./fdk' + str(step) + '.raw')

        # print('#'*20)

        # print(s2.size())
        # print(s3.size())
        # print(s4.size())
        # print(s5.size())

        # print(F.interpolate(RDCTImg, scale_factor=0.5, mode='bilinear').repeat(1, s2.size(1), 1, 1).size())
        # print(F.interpolate(RDCTImg, scale_factor=0.25, mode='bilinear').repeat(1, s3.size(1), 1, 1).size())
        # print(F.interpolate(RDCTImg, scale_factor=0.125, mode='bilinear').repeat(1, s4.size(1), 1, 1).size())
        # print(F.interpolate(RDCTImg, scale_factor=0.0625, mode='bilinear').repeat(1, s5.size(1), 1, 1).size())

        loss1 = loss_mse(predictImg, RDCTImg)
        loss2 = 0.0001 * loss_mse(s2, F.interpolate(RDCTImg, scale_factor=0.5, mode='bilinear').repeat(1, s2.size(1), 1, 1))
        loss3 = 0.0001 * loss_mse(s3, F.interpolate(RDCTImg, scale_factor=0.25, mode='bilinear').repeat(1, s3.size(1), 1, 1))
        loss4 = 0.0001 * loss_mse(s4, F.interpolate(RDCTImg, scale_factor=0.125, mode='bilinear').repeat(1, s4.size(1), 1, 1))
        loss5 = 0.0001 * loss_mse(s5, F.interpolate(RDCTImg, scale_factor=0.0625, mode='bilinear').repeat(1, s5.size(1), 1, 1))

        loss = loss1 + loss2 + loss3 + loss4 + loss5 

        loss_mse_scalar.update(loss1.item(), LDCTImg.size(0))
        loss2_scalar.update(loss2.item(), LDCTImg.size(0))
        loss3_scalar.update(loss3.item(), LDCTImg.size(0))
        loss4_scalar.update(loss4.item(), LDCTImg.size(0))
        loss5_scalar.update(loss5.item(), LDCTImg.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        step += 1

    writer.add_scalars('loss/mse', {'train_mse_loss': loss_mse_scalar.avg}, epoch + 1)
    writer.add_scalars('loss/mse2', {'train_mse2_loss': loss2_scalar.avg}, epoch + 1)
    writer.add_scalars('loss/mse3', {'train_mse3_loss': loss3_scalar.avg}, epoch + 1)
    writer.add_scalars('loss/mse4', {'train_mse4_loss': loss4_scalar.avg}, epoch + 1)
    writer.add_scalars('loss/mse5', {'train_mse5_loss': loss5_scalar.avg}, epoch + 1)
    # print((len(train_loader.dataset) / len(scoutProj)) * (epoch) + step + 1)
    # writer.add_scalar('learning_rate', scheduler.get_last_lr()[0], (len(train_loader.dataset) / len(scoutProj)) * (epoch) + step + 1)
    writer.add_image('train img/reference img', normalization(RDCTImg[0, :, :, :]), epoch + 1)
    writer.add_image('train img/predict img', normalization(predictImg[0, :, :, :]), epoch + 1)
    writer.add_image('train img/ldct img', normalization(LDCTImg[0, :, :, :]), epoch + 1)
    writer.add_image('train img/residual img', normalization(torch.abs(RDCTImg[0, :, :, :] - predictImg[0, :, :, :])), epoch + 1)

    scheduler.step()
    print('Train Epoch: {}\t train_mse_loss: {:.6f}\t'.format(epoch + 1, loss_mse_scalar.avg))

def valid(valid_loader, model, loss_mse, loss_ssim, writer, epoch):

    batch_time = AverageMeter()
    loss_mse_scalar = AverageMeter()
    loss_ssim_scalar = AverageMeter()
    model.eval()
    end = time.time()

    step = 0

    for data in valid_loader:


        RDCTImg = data["rdct"]
        RDCTImg = RDCTImg.cuda()

        LDCTImg = data["ldct"]
        LDCTImg = LDCTImg.cuda()

        with torch.no_grad():

            _, _, _, _, predictImg = model(LDCTImg)
            loss1 = loss_mse(predictImg, RDCTImg)
            loss2 = loss_ssim(predictImg, RDCTImg)

        loss_mse_scalar.update(loss1.item(), LDCTImg.size(0))
        loss_ssim_scalar.update(loss2.item(), LDCTImg.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        step += 1

    writer.add_scalars('loss/mse', {'valid_mse_loss': loss_mse_scalar.avg}, epoch+1)
    writer.add_scalars('loss/ssim', {'valid_ssim_loss': loss_ssim_scalar.avg}, epoch+1) 
    writer.add_image('valid img/reference img', normalization(RDCTImg[0, :, :, :]), epoch + 1)
    writer.add_image('valid img/predict img', normalization(predictImg[0, :, :, :]), epoch + 1)
    writer.add_image('valid img/ldct img', normalization(LDCTImg[0, :, :, :]), epoch + 1)
    writer.add_image('valid img/residual img', normalization(torch.abs(RDCTImg[0, :, :, :]- predictImg[0, :, :, :])), epoch + 1)

    print('Valid Epoch: {}\t valid_mse_loss: {:.6f}\t'.format(epoch + 1, loss_mse_scalar.avg))

if __name__ == "__main__":


    cudnn.benchmark = True

    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)

    result_path = './runs/UNet2d_DeepSuperviser/logs/'
    save_dir = './runs/UNet2d_DeepSuperviser/checkpoints/'

    # Get dataset
    train_dataset = read_deicom_file.dicom_reader(paired_img_txt='./train_img.txt')
    # train_dataset = read_deicom_file.dicom_reader(paired_img_txt='./valid_img.txt')
    train_loader = DataLoader(train_dataset, batch_size=12, num_workers=16, shuffle=True)

    valid_dataset = read_deicom_file.dicom_reader(paired_img_txt='./valid_img.txt')
    valid_loader = DataLoader(valid_dataset, batch_size=1, num_workers=16, shuffle=True)

    model = UNet2dDeepSuperviser()
    loss_mse = torch.nn.MSELoss()
    loss_cl = CharbonnierLoss()
    loss_ssim = SSIM()
    # optimizer = optim.Adam(model.parameters(), lr=0.0001)
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.02)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.95)

    if os.path.exists(save_dir) is False:

        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)

        model = model.cuda()

    else:
        checkpoint_latest = torch.load(find_lastest_file(save_dir))
        model = load_model_data_parallel(model, checkpoint_latest).cuda()
        optimizer.load_state_dict(checkpoint_latest['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint_latest['lr_scheduler'])
        print('Latest checkpoint {0} loaded.'.format(find_lastest_file(save_dir)))

    now_time = datetime.now()
    time_str = datetime.strftime(now_time, '%m-%d_%H-%M-%S')
    log_dir = os.path.join(result_path, time_str)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    writer = SummaryWriter(log_dir=log_dir)

    print("*"*20 + "Start Train" + "*"*20)

    for epoch in range(0, 100):

        print("*" * 20 + "Epoch: " + str(epoch + 1).rjust(4, '0') + "*" * 20)

        train(train_loader, model, loss_mse, loss_cl, loss_ssim, optimizer, scheduler, writer, epoch)
        valid(valid_loader, model, loss_mse, loss_ssim, writer, epoch)

        save_model(model, optimizer, scheduler, epoch + 1, save_dir)