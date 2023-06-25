import os
import sys
import cv2
import glob
import random
import torch
import argparse
import torchvision
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# sys.path.append('/data1/gege.gao/projects/InfoSwap-master/preprocess')
from infoswap.preprocess.mtcnn import MTCNN

mtcnn = MTCNN()

class FaceEmbed(TensorDataset):
    def __init__(self, data_path_list, same_prob=0.8):
        super(FaceEmbed, self).__init__()

        datasets = []
        # embeds = []
        self.N = []
        self.same_prob = same_prob
        for data_path in data_path_list:
            image_list = glob.glob(f'{data_path}/*.*g')
            datasets.append(image_list)
            self.N.append(len(image_list))
            # with open(f'{data_path}/embed.pkl', 'rb') as f:
            #     embed = pickle.load(f)
            #     embeds.append(embed)
        self.datasets = datasets
        # self.embeds = embeds
        self.transforms = transforms.Compose([
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.01),
            transforms.Resize((512, 512), interpolation=2),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __getitem__(self, item):
        idx = 0
        while item >= self.N[idx]:
            item -= self.N[idx]
            idx += 1
        image_path_s = self.datasets[idx][item]
        Xs = cv2.imread(image_path_s)
        Xs = Image.fromarray(Xs)
        # Xs = Image.open(image_path_s)

        if random.random() > self.same_prob:
            image_path = random.choice(self.datasets[random.randint(0, len(self.datasets)-1)])
            Xt = cv2.imread(image_path)
            Xt = Image.fromarray(Xt)
            same_person = 0
        else:
            Xt = Xs.copy()
            same_person = 1
        return self.transforms(Xs), self.transforms(Xt), same_person  #, Xs_id, Xt_id

    def __len__(self):
        return sum(self.N)


class FaceEmbedSingle(TensorDataset):
    def __init__(self, data_path_list):
        super(FaceEmbedSingle, self).__init__()

        datasets = []
        # embeds = []
        self.N = []
        for data_path in data_path_list:
            image_list = glob.glob(f'{data_path}/*.*g')
            datasets.append(image_list)
            self.N.append(len(image_list))
            # with open(f'{data_path}/embed.pkl', 'rb') as f:
            #     embed = pickle.load(f)
            #     embeds.append(embed)
        self.datasets = datasets
        # self.embeds = embeds
        self.transforms = transforms.Compose([
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.01),
            transforms.CenterCrop(438),
            transforms.Resize((112, 112), interpolation=Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __getitem__(self, item):
        idx = 0
        while item >= self.N[idx]:
            item -= self.N[idx]
            idx += 1
        image_path_s = self.datasets[idx][item]
        Xs = cv2.imread(image_path_s)
        Xs = Image.fromarray(Xs)
        # Xs = Image.open(image_path_s)

        # if random.random() > self.same_prob:
        #     image_path = random.choice(self.datasets[random.randint(0, len(self.datasets)-1)])
        #     Xt = cv2.imread(image_path)
        #     Xt = Image.fromarray(Xt)
        #     same_person = 0
        # else:
        #     Xt = Xs.copy()
        #     same_person = 1
        return self.transforms(Xs)

    def __len__(self):
        return sum(self.N)


class FaceEmbedAlign(TensorDataset):
    def __init__(self, data_path_list, same_prob=0.0, crop_size=(512, 512), minSize=64., thresholds=[0.6, 0.6, 0.6], factor=0.707):
        super(FaceEmbedAlign, self).__init__()

        datasets = []
        # embeds = []
        self.N = []
        self.same_prob = same_prob
        self.crop_size = crop_size
        self.minSize = minSize
        self.thresholds = thresholds
        self.factor = factor

        for data_path in data_path_list:
            image_list = glob.glob(f'{data_path}/*.*g')
            datasets.append(image_list)
            self.N.append(len(image_list))
            # with open(f'{data_path}/embed.pkl', 'rb') as f:
            #     embed = pickle.load(f)
            #     embeds.append(embed)
        self.datasets = datasets
        # print(type(self.datasets))
        # self.embeds = embeds
        self.transforms = transforms.Compose([
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.01),
            transforms.Resize(crop_size, interpolation=2),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __getitem__(self, item):
        idx = 0
        # print(item)
        while item >= self.N[idx]:
            item -= self.N[idx]
            idx += 1

        image_path_s = self.datasets[idx][item]
        Xs = cv2.imread(image_path_s)
        Xs = Image.fromarray(Xs)
        # Xs = Image.open(image_path_s)
        with torch.no_grad():
            faces = mtcnn.align_multi(Xs, min_face_size=self.minSize, crop_size=self.crop_size)  # , thresholds=self.thresholds, factor=self.factor
        if faces is not None:
            Xs = faces[0]
            # print('source: ', image_path_s)
        else:
            Xs = None
            while Xs is None:
                image_path_s = random.choice(self.datasets[random.randint(0, len(self.datasets) - 1)])
                Xs = cv2.imread(image_path_s)
                Xs = Image.fromarray(Xs)
                # Xs = Image.open(image_path_s)
                with torch.no_grad():
                    faces = mtcnn.align_multi(Xs, min_face_size=self.minSize, crop_size=self.crop_size)  # , thresholds=self.thresholds, factor=self.factor
                if faces is not None:
                    Xs = faces[0]
                    # print('source new: ', image_path_s)
                    # print(np.array(Xs).shape)
                else:
                    Xs = None

        if random.random() > self.same_prob:
            Xt = None
            while Xt is None:
                image_path_t = random.choice(self.datasets[random.randint(0, len(self.datasets)-1)])
                Xt = cv2.imread(image_path_t)
                Xt = Image.fromarray(Xt)
                # Xt = Image.open(image_path)
                with torch.no_grad():
                    faces = mtcnn.align_multi(Xt, min_face_size=self.minSize, crop_size=self.crop_size)  # , thresholds=self.thresholds, factor=self.factor
                if faces is not None:
                    Xt = faces[0]
                    # print('target: ', image_path)
                    # print(np.array(Xt).shape)
                else:
                    Xt = None
            same_person = 0
        else:
            Xt = Xs.copy()
            # Xt_id = copy.deepcopy(Xs_id)
            same_person = 1

        return self.transforms(Xs), self.transforms(Xt), [
            image_path_s.split('/')[-1].split('.')[0], image_path_t.split('/')[-1].split('.')[0]]  #, Xs_id, Xt_id

    def __len__(self):
        return sum(self.N)


class FaceEmbedAlign1024(TensorDataset):
    def __init__(self, data_path_list, minSize=96., thresholds=[0.6, 0.6, 0.6], factor=0.707):
        super(FaceEmbedAlign1024, self).__init__()

        datasets = []
        self.N = []
        self.minSize = minSize
        self.thresholds = thresholds
        self.factor = factor

        for data_path in data_path_list:
            image_list = glob.glob(f'{data_path}/*.*g')
            datasets.append(image_list)
            self.N.append(len(image_list))
        self.datasets = datasets
        self.transforms = transforms.Compose([
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.01),
            transforms.Resize((1024, 1024), interpolation=2),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.transforms512 = transforms.Compose([
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.01),
            transforms.Resize((512, 512), interpolation=2),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __getitem__(self, item):
        idx = 0
        while item >= self.N[idx]:
            item -= self.N[idx]
            idx += 1

        image_path_s = self.datasets[idx][item]
        Xs = cv2.imread(image_path_s)
        Xs = Image.fromarray(Xs)
        with torch.no_grad():
            faces = mtcnn.align_multi(Xs, min_face_size=self.minSize, crop_size=(1024, 1024))
            if faces is not None:
                Xs = faces[0]
            else:
                Xs = None
                while Xs is None:
                    image_path_s = random.choice(self.datasets[random.randint(0, len(self.datasets) - 1)])
                    Xs = cv2.imread(image_path_s)
                    Xs = Image.fromarray(Xs)
                    faces = mtcnn.align_multi(Xs, min_face_size=self.minSize, crop_size=(1024, 1024))
                    if faces is not None:
                        Xs = faces[0]
                    else:
                        Xs = None

            # target:
            Xt = None
            while Xt is None:
                image_path_t = random.choice(self.datasets[random.randint(0, len(self.datasets)-1)])
                Xt = cv2.imread(image_path_t)
                Xt = Image.fromarray(Xt)
                faces = mtcnn.align_multi(Xt, min_face_size=self.minSize, crop_size=(1024, 1024))
                if faces is not None:
                    Xt = faces[0]
                else:
                    Xt = None

            Xs_1024, Xt_1024, Xs_512, Xt_512 = self.transforms(Xs), self.transforms(Xt), self.transforms512(Xs), self.transforms512(Xt)

        return Xs_1024, Xt_1024, Xs_512, Xt_512, [
            image_path_s.split('/')[-1].split('.')[0], image_path_t.split('/')[-1].split('.')[0]]

    def __len__(self):
        return sum(self.N)



class FaceEmbedAlign256(TensorDataset):
    def __init__(self, data_path_list, same_prob=0.0):
        super(FaceEmbedAlign256, self).__init__()

        datasets = []
        # embeds = []
        self.N = []
        self.same_prob = same_prob
        for data_path in data_path_list:
            image_list = glob.glob(f'{data_path}/*.*g')
            datasets.append(image_list)
            self.N.append(len(image_list))
        self.datasets = datasets
        self.transforms = transforms.Compose([
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.01),
            transforms.Resize((256, 256), interpolation=2),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __getitem__(self, item):
        idx = 0
        # print(item)
        while item >= self.N[idx]:
            item -= self.N[idx]
            idx += 1

        image_path_s = self.datasets[idx][item]
        Xs = cv2.imread(image_path_s)
        Xs = Image.fromarray(Xs)
        # Xs = Image.open(image_path_s)
        faces = mtcnn.align_multi(Xs, min_face_size=64, crop_size=(256, 256))
        if faces is not None:
            Xs = faces[0]
            # print('source: ', image_path_s)
        else:
            Xs = None
            while Xs is None:
                image_path_s = random.choice(self.datasets[random.randint(0, len(self.datasets) - 1)])
                Xs = cv2.imread(image_path_s)
                Xs = Image.fromarray(Xs)
                # Xs = Image.open(image_path_s)
                faces = mtcnn.align_multi(Xs, min_face_size=64, crop_size=(256, 256))
                if faces is not None:
                    Xs = faces[0]
                    # print('source new: ', image_path_s)
                    # print(np.array(Xs).shape)
                else:
                    Xs = None

        if random.random() > self.same_prob:
            Xt = None
            while Xt is None:
                image_path = random.choice(self.datasets[random.randint(0, len(self.datasets)-1)])
                Xt = cv2.imread(image_path)
                Xt = Image.fromarray(Xt)
                # Xt = Image.open(image_path)
                faces = mtcnn.align_multi(Xt, min_face_size=64, crop_size=(256, 256))
                if faces is not None:
                    Xt = faces[0]
                    # print('target: ', image_path)
                    # print(np.array(Xt).shape)
                else:
                    Xt = None
            same_person = 0
        else:
            Xt = Xs.copy()
            # Xt_id = copy.deepcopy(Xs_id)
            same_person = 1
        return self.transforms(Xs), self.transforms(Xt), [
            image_path_s.split('/')[-1].split('.')[0], image_path.split('/')[-1].split('.')[0]]  #, Xs_id, Xt_id

    def __len__(self):
        return sum(self.N)


class FaceEmbedAlignSingle112(TensorDataset):
    def __init__(self, data_path_list, same_prob=0.8):
        super(FaceEmbedAlignSingle112, self).__init__()

        datasets = []
        self.N = []
        self.same_prob = same_prob
        for data_path in data_path_list:
            image_list = glob.glob(f'{data_path}/*.*g')
            datasets.append(image_list)
            self.N.append(len(image_list))
        self.datasets = datasets
        self.transforms = transforms.Compose([
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.01),
            transforms.Resize((512, 512), interpolation=2),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __getitem__(self, item):
        idx = 0
        while item >= self.N[idx]:
            item -= self.N[idx]
            idx += 1
        image_path_s = self.datasets[idx][item]
        X = cv2.imread(image_path_s)
        X = Image.fromarray(X)
        faces = mtcnn.align_multi(X, min_face_size=64, crop_size=(512, 512))
        if faces is not None:
            X = faces[0]
        else:
            X = None
            while X is None:
                image_path_s = random.choice(self.datasets[random.randint(0, len(self.datasets) - 1)])
                X = cv2.imread(image_path_s)
                X = Image.fromarray(X)
                faces = mtcnn.align_multi(X, min_face_size=64, crop_size=(512, 512))
                if faces is not None:
                    X = faces[0]
                else:
                    X = None

        X = self.transforms(X).unsqueeze(0)
        return F.interpolate(X[:, :, 37:475, 37:475], size=[112, 112], mode='bilinear', align_corners=True)[0]

    def __len__(self):
        return sum(self.N)


class FaceEmbedAlignSingle128(TensorDataset):
    def __init__(self, data_path_list):
        super(FaceEmbedAlignSingle128, self).__init__()

        datasets = []
        self.N = []
        for data_path in data_path_list:
            image_list = glob.glob(f'{data_path}/*.*g')
            datasets.append(image_list)
            self.N.append(len(image_list))
        self.datasets = datasets
        self.transforms = transforms.Compose([
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.01),
            transforms.Resize((512, 512), interpolation=2),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __getitem__(self, item):
        idx = 0
        while item >= self.N[idx]:
            item -= self.N[idx]
            idx += 1
        image_path_s = self.datasets[idx][item]
        X = cv2.imread(image_path_s)
        X = Image.fromarray(X)
        faces = mtcnn.align_multi(X, min_face_size=64, crop_size=(512, 512))
        if faces is not None:
            X = faces[0]
        else:
            X = None
            while X is None:
                image_path_s = random.choice(self.datasets[random.randint(0, len(self.datasets) - 1)])
                X = cv2.imread(image_path_s)
                X = Image.fromarray(X)
                faces = mtcnn.align_multi(X, min_face_size=64, crop_size=(512, 512))
                if faces is not None:
                    X = faces[0]
                else:
                    X = None

        X = self.transforms(X).unsqueeze(0)
        return F.interpolate(X[:, :, 37:475, 37:475], size=[128, 128], mode='bilinear', align_corners=True)[0]

    def __len__(self):
        return sum(self.N)


class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        # X shape: B x C x H x W
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :]-x[:, :, :h_x-1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:]-x[:, :, :, :w_x-1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv/count_h + w_tv/count_w) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]


tv_loss = TVLoss()


def hinge_loss(X, positive=True):
    if positive:
        return torch.relu(1 - X).mean()
    else:
        return torch.relu(X + 1).mean()


def make_image(I_list, show_size):
    def get_grid_image(X):
        X = X[:show_size]
        X = (torchvision.utils.make_grid(X.detach().cpu(), nrow=X.shape[0]) * 0.5 + 0.5) * 255
        return X

    res = get_grid_image(I_list[0])
    if len(I_list) > 1:
        for i in I_list[1:]:
            res = torch.cat((res, get_grid_image(i)), dim=1)
    return res.numpy()


class ConvexUpsample(nn.Module):
    def __init__(self, cin, factor=2):
        super(ConvexUpsample, self).__init__()
        self.factor = factor
        self.mask = nn.Sequential(
            nn.Conv2d(cin, cin*2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(cin*2, factor * factor * 9, 1, padding=0))

    def get_mask(self, h):
        mask = .25 * self.mask(h)
        return mask

    def upsample(self, flow, mask):
        B, C, H, W = flow.shape
        mask = mask.view(B, 1, 9, self.factor, self.factor, H, W)
        mask = torch.softmax(mask, dim=2)
        # print(mask.shape)

        up_flow = F.unfold(8 * flow, [3, 3], padding=[1, 1])
        up_flow = up_flow.view(B, C, 9, 1, 1, H, W)
        # print(up_flow.shape)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(B, C, self.factor * H, self.factor * W)

    def forward(self, h):
        mask = self.get_mask(h)
        up_h = self.upsample(h, mask)
        return up_h


def frames_to_video(frames_dir, output_dir, name, fps=24, start=0, loop_num=None):
    import os
    from os.path import isfile, join
    pathIn = frames_dir
    pathOut = output_dir + name
    print(pathOut)

    frame_array = []
    if pathIn.endswith('.png') or pathIn.endswith('.jpg'):
        files = [pathIn]
    else:
        files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f)) and f.endswith('png')]
        files.sort()
    for i in range(start, len(files)):
        filename = os.path.join(pathIn, files[i])
        # reading each files
        # print(filename)
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)

        # inserting the frames into an image array
        if loop_num is None:
            frame_array.append(img)
        else:
            for _ in range(loop_num):
                frame_array.append(img)
    out = cv2.VideoWriter(pathOut, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()


def laplacian_blending(A, B, m, num_levels=8):
    """
    :param A: foreground
    :param B: background
    :param m: mask for A
    :param num_levels: for Pyramids
    :return: blended images
    """
    # assume mask is float32 [0,1]
    # generate Gaussian pyramid for A,B and mask
    assert A.shape == B.shape
    assert B.shape == m.shape
    # save image width and height
    height = m.shape[0]
    width = m.shape[1]
    # image size should be 2^n
    size_list = np.array([4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096])
    size = size_list[np.where(size_list > max(height, width))][0]
    GA = np.zeros((size, size, 3), dtype=np.float32)
    GA[:height, :width, :] = A
    GB = np.zeros((size, size, 3), dtype=np.float32)
    GB[:height, :width, :] = B
    GM = np.zeros((size, size, 3), dtype=np.float32)
    GM[:height, :width, :] = m

    gpA = [GA]
    gpB = [GB]
    gpM = [GM]
    for i in range(num_levels):
        GA = cv2.pyrDown(GA)
        GB = cv2.pyrDown(GB)
        GM = cv2.pyrDown(GM)
        gpA.append(np.float32(GA))
        gpB.append(np.float32(GB))
        gpM.append(np.float32(GM))

    # generate Laplacian Pyramids for A,B and masks
    lpA  = [gpA[num_levels-1]]  # the bottom of the Lap-pyr holds the last (smallest) Gauss level
    lpB  = [gpB[num_levels-1]]
    gpMr = [gpM[num_levels-1]]
    for i in range(num_levels-1,0,-1):
        # Laplacian: subtarct upscaled version of lower level from current level
        # to get the high frequencies
        LA = np.subtract(gpA[i-1], cv2.pyrUp(gpA[i]))
        LB = np.subtract(gpB[i-1], cv2.pyrUp(gpB[i]))
        lpA.append(LA)
        lpB.append(LB)
        gpMr.append(gpM[i-1]) # also reverse the masks

    # Now blend images according to mask in each level
    LS = []
    for la,lb,gm in zip(lpA,lpB,gpMr):
        ls = la * gm + lb * (1.0 - gm)
        LS.append(ls)

    # now reconstruct
    ls_ = LS[0]
    for i in range(1,num_levels):
        ls_ = cv2.pyrUp(ls_)
        ls_ = cv2.add(ls_, LS[i])
    ls_ = np.clip(ls_[:height, :width, :], 0, 255).astype(np.uint8)
    return ls_


def cut_video(vid_path, start, end, save_dir='/data1/gege.gao/datasets/Videos', name=None):
    from moviepy.editor import VideoFileClip
    # from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
    """
    Returns a clip playing the content of the current clip but
    skips the extract between ``ta`` and ``tb``, which can be
    expressed in seconds (15.35), in (min, sec), in (hour, min, sec),
    or as a string: '01:03:05.35'.
    If the original clip has a ``duration`` attribute set,
    the duration of the returned clip  is automatically computed as
    `` duration - (tb - ta)``.

    The resulting clip's ``audio`` and ``mask`` will also be cutout
    if they exist.
    """
    from moviepy.tools import subprocess_call
    from moviepy.config import get_setting

    def ffmpeg_extract_subclip(filename, t1, t2, targetname=None):
        """ Makes a new video file playing video file ``filename`` between
        the times ``t1`` and ``t2``. """
        name, ext = os.path.splitext(filename)
        if not targetname:
            T1, T2 = [int(1000 * t) for t in [t1, t2]]
            targetname = "%sSUB%d_%d.%s" % (name, T1, T2, ext)

        cmd = [get_setting("FFMPEG_BINARY"), "-y",
               "-ss", "%0.2f" % t1,
               "-i", filename,
               "-t", "%0.2f" % (t2 - t1),
               "-vcodec", "copy", "-acodec", "copy", targetname]

        subprocess_call(cmd)

    if ':' in start:
        h = start.split(':')[0]
        m = start.split(':')[1]
        s = start.split(':')[2]
        start_sec = 60 * (60*int(h) + int(m)) + s
        start = f"{h}-{m}-{s}"
    else:  # (hour, min, sec)
        start_sec = float(60 * (60 * int(start[0]) + int(start[1]))) + float(start[2])
        start = f"{int(start[0])}-{int(start[1])}-{start[2]}"

    if ':' in end:
        h = end.split(':')[0]
        m = end.split(':')[1]
        s = end.split(':')[2]
        end_sec = 60 * (60*int(h) + int(m)) + s
        end = f"{h}-{m}-{s}"
    else:
        end_sec = float(60 * (60 * int(end[0]) + int(end[1]))) + float(end[2])
        end = f"{int(end[0])}-{int(end[1])}-{end[2]}"
    if name is None:
        save_name = vid_path.split('/')[-1].split('.')[0] + f"new_{start}_{end}.mkv".replace(':', '-')
    else:
        save_name = name + f"{start}_{end}.mkv".replace(':', '-')

    ffmpeg_extract_subclip(vid_path, start_sec, end_sec, targetname=os.path.join(save_dir, save_name))
    # clip = VideoFileClip(vid_path, audio=False).cutout(start, end)
    # clip.write_videofile(os.path.join(save_dir, save_name), audio=False)


def frames_inter(f1_path, f2_path, save=False, save_dir=None):
    f1 = cv2.imread(f1_path)
    f2 = cv2.imread(f2_path)
    f = cv2.addWeighted(f1, 0.5, f2, 0.5, 1.)
    plt.imshow(cv2.cvtColor(f.astype(np.uint8), cv2.COLOR_RGB2BGR))
    if save:
        save_name = f2_path.split('/')[-1]
        plt.imsave(os.path.join(save_dir, f'i_{save_name}'), cv2.cvtColor(f.astype(np.uint8), cv2.COLOR_RGB2BGR))
    # plt.show()
    plt.close()

if __name__ == '__main__':
    # cut_video('/data1/gege.gao/datasets/Videos/Aladdin2019.mkv', start=(0, 19, 22.0), end=(0, 19, 30.3), name='Aladdin_')
    # cut_video('/data1/gege.gao/datasets/Videos/Titanic1997.mkv', start=(0, 46, 11.5), end=(0, 46, 21.0), name='Titanic_')
    # p = argparse.ArgumentParser(
    #     formatter_class=argparse.ArgumentDefaultsHelpFormatter
    # )
    # p.add_argument('--dir', type=str, default='/data1/gege.gao/projects/InfoSwap-master/demo/target_683_50_Emma-Watson_2020-11-18/epoch4_iter30000/frames_full_interx9')
    # p.add_argument('--name', type=str, default='')
    # fps = 50
    # p.add_argument('--ori_dir', type=str, default='')
    # p.add_argument('--start', type=int, default=0, help="how many interpolation had been made")
    # p.add_argument('--num', type=int, default=15)
    # args = p.parse_args()
    #
    # num = args.num
    # start = args.start
    # dir = args.dir
    # ori_dir = args.ori_dir
    # if not os.path.isdir(ori_dir):
    #     ori_dir = args.dir
    # ori_name = args.name +'_seamless-fps50'
    #
    # if start < num:
    #     save_dir = ori_dir + f'_interx{start}'
    #     print(save_dir)
    #     for idx in range(start, num+1):
    #         save_dir = ori_dir + f'_interx{idx}'
    #         os.makedirs(save_dir, exist_ok=True)
    #         f_list = sorted(os.listdir(dir))
    #         M = len(f_list)
    #         i = 0
    #         j = i + 1
    #         f1 = cv2.imread(os.path.join(dir, f_list[0]))
    #         plt.imsave(os.path.join(save_dir, f"i_{f_list[0]}"), cv2.cvtColor(f1.astype(np.uint8), cv2.COLOR_RGB2BGR))
    #         while j < M:
    #             print(i)
    #             f1_path = os.path.join(dir, f_list[i])
    #             f2_path = os.path.join(dir, f_list[j])
    #             frames_inter(f1_path, f2_path, save=True, save_dir=save_dir)
    #             i += 1
    #             j = i + 1
    #
    #         dir = save_dir
    # else:
    #     save_dir = dir
    #
    # save_name = ori_name + f'_interx{num}.mp4'
    # root = save_dir.replace(save_dir.split('/')[-1], '')
    # os.makedirs(save_dir, exist_ok=True)
    # frames_to_video(save_dir,
    #                 root,
    #                 name=save_name,
    #                 fps=fps)

    """
    ffmpeg -i /data1/gege.gao/datasets/Videos/Titanic1997.mkv -ss 00:46:11.5 -c copy -to 00:46:21.0 /data1/gege.gao/datasets/Videos/Titanic_0-46-11.5_0-46-21.0.mkv
    # --- 
    ffmpeg -i video.avi -vf "fps=50" frames/%04d.p'ng
    """
    frames_to_video('/data1/share/Nikki/Emma-Watson.jpg',
                 '/data1/gege.gao/projects/InfoSwap-master/demo/supp/',
                 'Emma-Watson.mp4', 50, loop_num=344)