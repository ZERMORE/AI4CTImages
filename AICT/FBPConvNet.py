import os
from datasets import read_dicom_file
import scipy
import numpy as np
import torch.nn
from torch.utils.data import DataLoader
from torch.backends import cudnn
import time
import torch.optim as optim
import argparse
from losses import *
from models import *
from utils import *
from datetime import datetime
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm  # 导入 tqdm 库

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def train(train_loader, model, loss_mse, loss_cl, loss_ssim, optimizer, scheduler, writer, epoch):
    batch_time = AverageMeter()
    loss_mse_scalar = AverageMeter()
    model.train()
    end = time.time()

    # 使用 tqdm 创建进度条
    train_progress = tqdm(
        enumerate(train_loader),
        total=len(train_loader),
        desc=f"Train Epoch {epoch + 1}",
        leave=False,
    )

    for step, data in train_progress:
        RDCTImg = data["rdct"].cuda()
        LDCTImg = data["ldct"].cuda()

        predictImg = model(LDCTImg)
        loss1 = loss_mse(predictImg, RDCTImg)
        loss3 = loss_cl(predictImg, RDCTImg)
        loss = loss1

        loss_mse_scalar.update(loss1.item(), LDCTImg.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 更新进度条信息
        train_progress.set_postfix({
            "MSE Loss": f"{loss_mse_scalar.avg:.6f}",
            "Batch Time": f"{batch_time.avg:.3f}s",
        })

    writer.add_scalars('loss/mse', {'train_mse_loss': loss_mse_scalar.avg}, epoch + 1)
    writer.add_image('train img/reference img', normalization(RDCTImg[0, :, :, :]), epoch + 1)
    writer.add_image('train img/predict img', normalization(predictImg[0, :, :, :]), epoch + 1)
    writer.add_image('train img/ldct img', normalization(LDCTImg[0, :, :, :]), epoch + 1)
    writer.add_image('train img/residual img', normalization(torch.abs(RDCTImg[0, :, :, :] - predictImg[0, :, :, :])), epoch + 1)

    scheduler.step()
    print(f"Train Epoch: {epoch + 1}\t train_mse_loss: {loss_mse_scalar.avg:.6f}")


def valid(valid_loader, model, loss_mse, loss_ssim, writer, epoch):
    batch_time = AverageMeter()
    loss_mse_scalar = AverageMeter()
    loss_ssim_scalar = AverageMeter()
    model.eval()
    end = time.time()

    # 使用 tqdm 创建进度条
    valid_progress = tqdm(
        enumerate(valid_loader),
        total=len(valid_loader),
        desc=f"Valid Epoch {epoch + 1}",
        leave=False,
    )

    for step, data in valid_progress:
        RDCTImg = data["rdct"].cuda()
        LDCTImg = data["ldct"].cuda()

        with torch.no_grad():
            predictImg = model(LDCTImg)
            loss1 = loss_mse(predictImg, RDCTImg)
            loss2 = loss_ssim(predictImg, RDCTImg)

        loss_mse_scalar.update(loss1.item(), LDCTImg.size(0))
        loss_ssim_scalar.update(loss2.item(), LDCTImg.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        # 更新进度条信息
        valid_progress.set_postfix({
            "MSE Loss": f"{loss_mse_scalar.avg:.6f}",
            "SSIM Loss": f"{loss_ssim_scalar.avg:.6f}",
        })

    writer.add_scalars('loss/mse', {'valid_mse_loss': loss_mse_scalar.avg}, epoch + 1)
    writer.add_scalars('loss/ssim', {'valid_ssim_loss': loss_ssim_scalar.avg}, epoch + 1)
    writer.add_image('valid img/reference img', normalization(RDCTImg[0, :, :, :]), epoch + 1)
    writer.add_image('valid img/predict img', normalization(predictImg[0, :, :, :]), epoch + 1)
    writer.add_image('valid img/ldct img', normalization(LDCTImg[0, :, :, :]), epoch + 1)
    writer.add_image('valid img/residual img', normalization(torch.abs(RDCTImg[0, :, :, :] - predictImg[0, :, :, :])), epoch + 1)

    print(f"Valid Epoch: {epoch + 1}\t valid_mse_loss: {loss_mse_scalar.avg:.6f}")


if __name__ == "__main__":
    cudnn.benchmark = True

    result_path = './runs/FBPConv/logs/'
    save_dir = './runs/FBPConv/checkpoints/'

    # Get dataset
    train_dataset = read_dicom_file.dicom_reader(paired_img_txt='./train_img.txt')
    train_loader = DataLoader(train_dataset, batch_size=12, num_workers=4, shuffle=True)

    valid_dataset = read_dicom_file.dicom_reader(paired_img_txt='./valid_img.txt')
    valid_loader = DataLoader(valid_dataset, batch_size=1, num_workers=4, shuffle=False)

    model = DenseUNet2d()
    loss_mse = torch.nn.MSELoss()
    loss_cl = CharbonnierLoss()
    loss_ssim = SSIM()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
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
        print(f"Latest checkpoint {find_lastest_file(save_dir)} loaded.")

    now_time = datetime.now()
    time_str = datetime.strftime(now_time, '%m-%d_%H-%M-%S')
    log_dir = os.path.join(result_path, time_str)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    writer = SummaryWriter(log_dir=log_dir)

    print("*" * 20 + "Start Train" + "*" * 20)

    for epoch in range(0, 200):
        print("*" * 20 + f"Epoch: {epoch + 1:04d}" + "*" * 20)

        train(train_loader, model, loss_mse, loss_cl, loss_ssim, optimizer, scheduler, writer, epoch)
        valid(valid_loader, model, loss_mse, loss_ssim, writer, epoch)

        save_model(model, optimizer, scheduler, epoch + 1, save_dir)