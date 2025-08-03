import os
from datasets import read_dicom_file
import scipy
import numpy as np
import torch.nn
# from datasets import dicom_file
from torch.utils.data import DataLoader
from torch.backends import cudnn
# from skimage.measure import compare_ssim as ssim
# from skimage.measure import compare_psnr as psnr
import time
import torch.optim as optim
import argparse

from models import *
from utils import *
from datetime import datetime
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def train(train_loader, loss_mse, writer, epoch, g_model, g_optm, g_lr, d_model, d_optm, d_lr):

    batch_time = AverageMeter()
    loss_mse_scalar = AverageMeter()
    loss_d_adv_scalar = AverageMeter()
    end = time.time()

    g_model.train()
    d_model.train()

    step = 0

    for data in train_loader:

        RDCTImg = data["rdct"]
        RDCTImg = RDCTImg.cuda()

        LDCTImg = data["ldct"]
        LDCTImg = LDCTImg.cuda()
        # print(LDCTImg.shape)
        # LDCTImg[LDCTImg < 0] = 0

        predictImg = g_model(LDCTImg)
        # fdkImgSave = fdkImg.data.cpu().numpy()
        # fdkImgSave.astype(np.float32).tofile('./fdk' + str(step) + '.raw')

        d_optm.zero_grad()

        real = d_model(extract_patches_online(RDCTImg, 4))
        fake = d_model(extract_patches_online(predictImg, 4))

        gradient_penalty = compute_gradient_penalty(d_model, extract_patches_online(RDCTImg, 4), extract_patches_online(predictImg, 4))
        d_loss_adv = -torch.mean(real) + torch.mean(fake) + 10 * gradient_penalty
        d_loss_adv.backward()
        d_optm.step()

        loss_d_adv_scalar.update(d_loss_adv.item(), RDCTImg.size(0))

        g_optm.zero_grad()

        if step % 1 == 0:
    
            predictImg = g_model(LDCTImg)

            fake = d_model(extract_patches_online(predictImg, 4))

            g_loss_adv = -torch.mean(fake)
            g_loss_recon = loss_mse(predictImg, RDCTImg)
            g_loss = g_loss_recon + 0.01 * g_loss_adv

            g_loss.backward()
            g_optm.step()

            loss_mse_scalar.update(g_loss_recon.item(), RDCTImg.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        step += 1

    g_lr.step()
    d_lr.step()

    writer.add_scalars('loss/mse', {'train_mse_loss': loss_mse_scalar.avg}, epoch + 1)
    writer.add_scalars('loss/d_adv', {'loss_d_adv': loss_d_adv_scalar.avg}, epoch + 1)
    writer.add_image('train img/reference img', normalization(gaussian_smooth_show(RDCTImg)[0, :, :, :]), epoch + 1)
    writer.add_image('train img/predict img', normalization(predictImg[0, :, :, :]), epoch + 1)
    writer.add_image('train img/ldct img', normalization(LDCTImg[0, :, :, :]), epoch + 1)
    writer.add_image('train img/residual img', normalization(torch.abs(RDCTImg[0, :, :, :] - predictImg[0, :, :, :])), epoch + 1)

    print('Train Epoch: {}\t train_mse_loss: {:.6f}\t'.format(epoch + 1, loss_mse_scalar.avg))

def valid(valid_loader, g_model, writer, epoch):

    batch_time = AverageMeter()
    loss_mse_scalar = AverageMeter()
    loss_ssim_scalar = AverageMeter()
    end = time.time()

    g_model.eval()

    step = 0

    for data in valid_loader:


        RDCTImg = data["rdct"]
        RDCTImg = RDCTImg.cuda()

        LDCTImg = data["ldct"]
        LDCTImg = LDCTImg.cuda()

        with torch.no_grad():

            predictImg = g_model(LDCTImg)
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

    result_path = './runs/UNet2D_GAN_Holder/logs/'
    save_dir = './runs/UNet2D_GAN_Holder/checkpoints/'

    # Get dataset
    # train_dataset = read_deicom_file.dicom_reader(paired_img_txt='./train_img.txt')
    train_dataset = read_deicom_file.dicom_reader(paired_img_txt='./valid_img.txt')
    train_loader = DataLoader(train_dataset, batch_size=12, num_workers=16, shuffle=True)

    valid_dataset = read_deicom_file.dicom_reader(paired_img_txt='./valid_img.txt')
    valid_loader = DataLoader(valid_dataset, batch_size=1, num_workers=16, shuffle=True)

    generator = UNet2d()
    discriminator = Discriminator2D()

    loss_mse = torch.nn.MSELoss()
    loss_ssim = SSIM()
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=0.0001)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.9))
    scheduler_g = torch.optim.lr_scheduler.StepLR(optimizer_g, step_size=100, gamma=0.1)
    scheduler_d = torch.optim.lr_scheduler.StepLR(optimizer_d, step_size=100, gamma=0.1)

    if os.path.exists(save_dir) is False:
    
        generator = generator.cuda()
        discriminator = discriminator.cuda()

    else:
        checkpoint_latest_g = torch.load(find_lastest_file(save_dir + 'G/'))
        generator = load_model_model_parallel(generator, checkpoint_latest_g)
        generator = generator.cuda()
        optimizer_g.load_state_dict(checkpoint_latest_g['optimizer_state_dict'])
        scheduler_g.load_state_dict(checkpoint_latest_g['lr_scheduler'])

        checkpoint_latest_d = torch.load(find_lastest_file(save_dir + 'D/'))
        discriminator = load_model_model_parallel(discriminator, checkpoint_latest_d)
        discriminator = discriminator.cuda()
        optimizer_d.load_state_dict(checkpoint_latest_d['optimizer_state_dict'])
        scheduler_d.load_state_dict(checkpoint_latest_d['lr_scheduler'])

        print('Latest checkpoint {0} loaded.'.format(find_lastest_file(save_dir)))

    now_time = datetime.now()
    time_str = datetime.strftime(now_time, '%m-%d_%H-%M-%S')
    log_dir = os.path.join(result_path, time_str)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    writer = SummaryWriter(log_dir=log_dir)

    print("*"*20 + "Start Train" + "*"*20)

    for epoch in range(0, 1000):
    
        print("*" * 20 + "Epoch: " + str(epoch + 1).rjust(4, '0') + "*" * 20)

        train(train_loader, loss_mse, writer, epoch, generator, optimizer_g, scheduler_g, discriminator, optimizer_d, scheduler_d)

        valid(valid_loader, generator, writer, epoch)

        save_model(generator, optimizer_g, scheduler_g, epoch + 1, save_dir + 'G/')
        save_model(discriminator, optimizer_d, scheduler_d, epoch + 1, save_dir + 'D/')