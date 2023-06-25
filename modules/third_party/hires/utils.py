import argparse
import math
import random
import os,time

import numpy as np
import torch
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
import torch.distributed as dist
import torchvision
from torchvision import transforms, utils
from tqdm import tqdm

from torch.autograd import Variable
import matplotlib as mlb

from criteria.lpips.lpips import LPIPS
from criteria import w_norm
import itertools
from tensorboardX import SummaryWriter


class SoftErosion(nn.Module):
    def __init__(self, kernel_size=15, threshold=0.6, iterations=1):
        super(SoftErosion, self).__init__()
        r = kernel_size // 2
        self.padding = r
        self.iterations = iterations
        self.threshold = threshold

        # Create kernel
        y_indices, x_indices = torch.meshgrid(torch.arange(0., kernel_size), torch.arange(0., kernel_size))
        dist = torch.sqrt((x_indices - r) ** 2 + (y_indices - r) ** 2)
        kernel = dist.max() - dist
        kernel /= kernel.sum()
        kernel = kernel.view(1, 1, *kernel.shape)
        self.register_buffer('weight', kernel)

    def forward(self, x):
        #print (x.shape)
        x = x.float()
        for i in range(self.iterations - 1):
            x = torch.min(x, F.conv2d(x, weight=self.weight, groups=x.shape[1], padding=self.padding))
        x = F.conv2d(x, weight=self.weight, groups=x.shape[1], padding=self.padding)

        mask = x >= self.threshold
        x[mask] = 1.0
        x[~mask] /= x[~mask].max()

        return x, mask


def encode_segmentation_rgb(segmentation, no_neck=True):
    parse = segmentation[:,:,0]

    face_part_ids = [1, 2, 3, 4, 5, 6, 10, 12, 13] if no_neck else [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 13, 14]
    mouth_id = 11
    hair_id = 17
    face_map = np.zeros([parse.shape[0], parse.shape[1]])
    mouth_map = np.zeros([parse.shape[0], parse.shape[1]])
    hair_map = np.zeros([parse.shape[0], parse.shape[1]])

    for valid_id in face_part_ids:
        valid_index = np.where(parse==valid_id)
        face_map[valid_index] = 255
    valid_index = np.where(parse==mouth_id)
    mouth_map[valid_index] = 255
    valid_index = np.where(parse==hair_id)
    hair_map[valid_index] = 255

    return np.stack([face_map, mouth_map, hair_map], axis=2)

    
def flip_video(x):
    num = random.randint(0, 1)
    if num == 0:
        return torch.flip(x, [2])
    else:
        return x


class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        # self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_data = next(self.loader)
        except StopIteration:
            self.next_input = None
            return
        # with torch.cuda.stream(self.stream):
        #     self.next_data = self.next_data.cuda(non_blocking=True)
            
    def next(self):
        # torch.cuda.current_stream().wait_stream(self.stream)
        data = self.next_data
        self.preload()
        return data

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

setup_seed(1984)

def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)

MISSING_VALUE = -1
def cords_to_map(_cords, img_size, sigma=6):
    results = []
    for j in range(_cords.shape[0]):
        cords = _cords[j]
        result = torch.zeros(img_size + cords.shape[0:1], dtype=torch.uint8)
        for i, point in enumerate(cords):
            if point[0] == MISSING_VALUE or point[1] == MISSING_VALUE:
                continue
            xx, yy = torch.meshgrid(torch.arange(img_size[1]), torch.arange(img_size[0]))
            x = torch.exp(-((yy - int(point[0])) ** 2 + (xx - int(point[1])) ** 2) / (2 * sigma ** 2))
            result[..., i] = x
        results.append(result)
    return torch.stack(results,dim=0)



def cords_to_map_np(_cords, img_size, sigma=6):
    results = []
    for j in range(_cords.shape[0]):
        cords = _cords[j]

        result = np.zeros(img_size + cords.shape[0:1], dtype='uint8')
        for i, point in enumerate(cords):
            if point[0] == MISSING_VALUE or point[1] == MISSING_VALUE:
                continue
            xx, yy = np.meshgrid(np.arange(img_size[1]), np.arange(img_size[0]))
            x = np.exp(-((yy - int(point[0])) ** 2 + (xx - int(point[1])) ** 2) / (2 * sigma ** 2))
            result[..., i] = x
        results.append(result)
    return np.array(results)


def set_requires_grad(nets, requires_grad=False):

    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

    
def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag
    

def sample_data(loader):
    while True:
        for batch in loader:
            yield batch
            

def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()

def d_logistic_loss_2(real_pred, fake_pred1, fake_pred2):
    real_loss = F.softplus(-real_pred)
    fake_loss1 = F.softplus(fake_pred1)
    fake_loss2 = F.softplus(fake_pred2)

    return real_loss.mean() + fake_loss1.mean() + fake_loss2.mean()


def d_r1_loss(real_pred, real_img):
    grad_real, = autograd.grad(
        outputs=real_pred.sum(), inputs=real_img, create_graph=True
    )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty


def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()

    return loss

def discriminator_r1_loss(real_pred, real_w):
    grad_real, = autograd.grad(
        outputs=real_pred.sum(), inputs=real_w, create_graph=True
    )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty