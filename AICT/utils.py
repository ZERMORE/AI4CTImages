from cgi import print_arguments
import os
import torch
import numpy as np
from collections import OrderedDict
from scipy import ndimage
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import cv2

def load_model(net, checkpoint):
    net.load_state_dict(checkpoint['state_dict'])
    # if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    #     net = torch.nn.DataParallel(net).cuda()
    # elif torch.cuda.is_available() and torch.cuda.device_count() == 1:
    #     net = net.cuda()
    return net

def load_model_data_parallel(net, checkpoint):
    net.load_state_dict(checkpoint['state_dict'])
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net).cuda()
    else:
        print('Error: GPU is unavailable or you only have one GPU.')
    return net

def load_model_model_parallel(net, checkpoint):

    '''one or multi-gpu'''
    net.load_state_dict(checkpoint['state_dict'])
    # net = net.cuda()
    return net

def save_model(net, optimizer, scheduler, epoch, save_dir):
    '''save model'''

    if os.path.exists(save_dir) == False:
        os.makedirs(save_dir)

    if 'module' in dir(net):
        state_dict = net.module.state_dict()
    else:
        state_dict = net.state_dict()

    torch.save({
        'state_dict': state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'lr_scheduler': scheduler.state_dict()},
        os.path.join(save_dir, 'model_at_epoch_%03d.dat' % (epoch)))

    print(os.path.join(save_dir, 'model_at_epoch_%03d.dat' % (epoch)))

def find_lastest_file(file_dir):

    lists = os.listdir(file_dir)
    lists.sort(key=lambda x: os.path.getmtime((file_dir + x)))
    file_latest = os.path.join(file_dir, lists[-1])

    return file_latest

def normalization(tensor):

    return (tensor - torch.min(tensor)) / (torch.max(tensor) - torch.min(tensor))

def extract_patches_online(tensor, num=2):

    if tensor.ndim == 5:

        split_w = torch.chunk(tensor, chunks=num, dim=3)
        stack_w = torch.reshape(torch.stack(split_w, dim=0),
                                [num*tensor.shape[0], tensor.shape[1],
                                 tensor.shape[2], tensor.shape[3]//num, tensor.shape[4]])
        split_h = torch.chunk(stack_w, chunks=num, dim=4)
        stack_h = torch.reshape(torch.stack(split_h, dim=0),
                                [num*num*tensor.shape[0], tensor.shape[1],
                                 tensor.shape[2], tensor.shape[3]//num, tensor.shape[4]//num])

        return stack_h

    elif tensor.ndim == 4:

        split_w = torch.chunk(tensor, chunks=num, dim=2)
        # print('split_w', split_w.size())
        stack_w = torch.reshape(torch.stack(split_w, dim=0),
                                [num*tensor.shape[0], tensor.shape[1],
                                 tensor.shape[2]//num, tensor.shape[3]])
        # print('stack_w', stack_w.size())
        split_h = torch.chunk(stack_w, chunks=num, dim=3)
        # print('split_h', split_h.size())
        stack_h = torch.reshape(torch.stack(split_h, dim=0),
                                [num*num*tensor.shape[0], tensor.shape[1],
                                 tensor.shape[2]//num, tensor.shape[3]//num])
        # print('stack_h', stack_h.size())

        return stack_h

    else:
        print('Expect for the tensor with dim==5 or 4, other cases are not yet implemented.')

def gaussian_smooth(input, kernel_size=7, sigma=3.5):
    # inputs: batch, channel, width, height

    filter = np.float32(np.multiply(cv2.getGaussianKernel(kernel_size, sigma), np.transpose(cv2.getGaussianKernel(kernel_size, sigma))))
    filter = filter[np.newaxis, np.newaxis, ...]
    kernel = torch.FloatTensor(filter).cuda(input.get_device())
    # kernel = torch.reshape(torch.from_numpy(filter).cuda(input.get_device()), [1, 1, kernel_size, kernel_size])
    low = F.conv2d(input, kernel, padding=(kernel_size-1)//2)
    high = input - low

    return torch.cat([input, low, high], 1)

# def gaussian_smooth_show(input, kernel_size=5, sigma=3.5):
#     # inputs: batch, channel, width, height
#     filter = np.float32(np.multiply(cv2.getGaussianKernel(kernel_size, sigma), np.transpose(cv2.getGaussianKernel(kernel_size, sigma))))
#     kernel = torch.reshape(torch.from_numpy(filter).cuda(input.get_device()), [1, 1, kernel_size, kernel_size])
#     # kernel = torch.reshape(torch.from_numpy(filter, dtype=torch.float).cuda(input.get_device()), [1, 1, kernel_size, kernel_size])
#     low = F.conv2d(input, kernel, padding=(kernel_size-1)//2)
#     high = input - low

#     return torch.cat([input, low, high], 2)

class AverageMeter(object):
    """
    Computes and stores the average and current value
    Copied from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """

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

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):

    min_val = torch.min(torch.cat([img1, img2], 1))
    max_val = torch.max(torch.cat([img1, img2], 1))
    
    img1 = (img1 - min_val) / (max_val - min_val)
    img2 = (img2 - min_val) / (max_val - min_val)

    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel


        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

def ssim(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, size_average)

### mix two images
class MixUp_AUG:
    def __init__(self):
        self.dist = torch.distributions.beta.Beta(torch.tensor([1.2]), torch.tensor([1.2]))

    def aug(self, rgb_gt, rgb_noisy):
        bs = rgb_gt.size(0)
        indices = torch.randperm(bs)
        rgb_gt2 = rgb_gt[indices]
        rgb_noisy2 = rgb_noisy[indices]

        lam = self.dist.rsample((bs,1)).view(-1,1,1,1).cuda()

        rgb_gt    = lam * rgb_gt + (1-lam) * rgb_gt2
        rgb_noisy = lam * rgb_noisy + (1-lam) * rgb_noisy2

        return rgb_gt, rgb_noisy

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

class CutMix_AUG:

    def __init__(self):
        self.dist = torch.distributions.beta.Beta(torch.tensor([1.2]), torch.tensor([1.2]))
        # print('CutMix_AUG __init__')

    def aug(self, rgb_gt, rgb_noisy):

        lam = np.random.beta(1, 1)
        rand_index = torch.randperm(rgb_gt.size(0))
        bbx1, bby1, bbx2, bby2 = rand_bbox(rgb_gt.size(), lam)

        rgb_gt[:, :, bbx1:bbx2, bby1:bby2] = rgb_gt[rand_index, :, bbx1:bbx2, bby1:bby2]
        rgb_noisy[:, :, bbx1:bbx2, bby1:bby2] = rgb_noisy[rand_index, :, bbx1:bbx2, bby1:bby2]

        return rgb_gt, rgb_noisy

class CutMix_AUG_V2:
    
    def __init__(self):
        self.dist = torch.distributions.beta.Beta(torch.tensor([1.2]), torch.tensor([1.2]))
        # pass

    def aug(self, rgb_gt, rgb_noisy):

        bs = rgb_gt.size(0)
        indices = torch.randperm(bs)
        rgb_gt2 = rgb_gt[indices]
        rgb_noisy2 = rgb_noisy[indices]

        lam = np.random.beta(1, 1, bs)
        mask = np.zeros(rgb_gt.size(), dtype=np.float32)
        for i in range(bs):
            bbx1, bby1, bbx2, bby2 = rand_bbox(rgb_gt.size(), lam[i])
            mask[i, :, bbx1:bbx2, bby1:bby2] = 1
        
        mask = torch.FloatTensor(mask).cuda(rgb_gt.get_device())
        
        rgb_gt = rgb_gt * (1 - mask) + rgb_gt2 * mask
        rgb_noisy = rgb_noisy * (1 - mask) + rgb_noisy2 * mask

        return rgb_gt, rgb_noisy

        
# import torch as t
# import matplotlib.pyplot as plt
# import numpy as np
#
# img = np.load('/home/yikun/10T/Data/DataProcessed/fullFDK64x256x256/2019-11-11_155051.npy')
# tensor_img = t.as_tensor(img)
# tensor_img = t.reshape(tensor_img, [1, 1, 64, 256, 256])
# tensor_img = tensor_img.repeat(3, 1, 1, 1, 1)
# print(tensor_img.shape)
# patch_img = extract_patches_online(tensor_img, 4)
# print(patch_img.shape)
# plt.imshow(patch_img[17, 0, 32, :, :], cmap=plt.cm.gray)
# plt.show()