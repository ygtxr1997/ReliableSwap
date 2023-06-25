import os
import numbers

import torch
import mxnet as mx
from PIL import Image
from torch.utils import data
from torchvision import transforms

import numpy as np
import PIL.Image as Image


""" Original mxnet dataset
"""
class MXFaceDataset(data.Dataset):
    def __init__(self, root_dir, crop_param=(0, 0, 112, 112)):
        super(MXFaceDataset, self,).__init__()
        self.transform = transforms.Compose([
             # transforms.ToPILImage(),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
             ])
        self.root_dir = root_dir
        self.crop_param = crop_param
        path_imgrec = os.path.join(root_dir, 'train.rec')
        path_imgidx = os.path.join(root_dir, 'train.idx')
        self.imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
        s = self.imgrec.read_idx(0)
        header, _ = mx.recordio.unpack(s)
        if header.flag > 0:
            self.header0 = (int(header.label[0]), int(header.label[1]))
            self.imgidx = np.array(range(1, int(header.label[0])))
        else:
            self.imgidx = np.array(list(self.imgrec.keys))

    def __getitem__(self, index):
        idx = self.imgidx[index]
        s = self.imgrec.read_idx(idx)
        header, img = mx.recordio.unpack(s)
        label = header.label
        if not isinstance(label, numbers.Number):
            label = label[0]
        label = torch.tensor(label, dtype=torch.long)
        sample = mx.image.imdecode(img).asnumpy()
        if self.transform is not None:
            sample: Image = transforms.ToPILImage()(sample)
            sample = sample.crop(self.crop_param)
            sample = self.transform(sample)
        return sample, label

    def __len__(self):
        return len(self.imgidx)


""" MXNet binary dataset reader. 
Refer to https://github.com/deepinsight/insightface.
"""
import pickle
from typing import List
from mxnet import ndarray as nd
class ReadMXNet(object):
    def __init__(self, val_targets, rec_prefix, image_size=(112, 112)):
        self.ver_list: List[object] = []
        self.ver_name_list: List[str] = []
        self.rec_prefix = rec_prefix
        self.val_targets = val_targets

    def init_dataset(self, val_targets, data_dir, image_size):
        for name in val_targets:
            path = os.path.join(data_dir, name + ".bin")
            if os.path.exists(path):
                data_set = self.load_bin(path, image_size)
                self.ver_list.append(data_set)
                self.ver_name_list.append(name)

    def load_bin(self, path, image_size):
        try:
            with open(path, 'rb') as f:
                bins, issame_list = pickle.load(f)  # py2
        except UnicodeDecodeError as e:
            with open(path, 'rb') as f:
                bins, issame_list = pickle.load(f, encoding='bytes')  # py3
        data_list = []
        # for flip in [0, 1]:
        #     data = torch.empty((len(issame_list) * 2, 3, image_size[0], image_size[1]))
        #     data_list.append(data)
        for idx in range(len(issame_list) * 2):
            _bin = bins[idx]
            img = mx.image.imdecode(_bin)
            if img.shape[1] != image_size[0]:
                img = mx.image.resize_short(img, image_size[0])
            img = nd.transpose(img, axes=(2, 0, 1))  # (C, H, W)

            img = nd.transpose(img, axes=(1, 2, 0))  # (H, W, C)
            import PIL.Image as Image
            fig = Image.fromarray(img.asnumpy(), mode='RGB')
            data_list.append(fig)
            # data_list[flip][idx][:] = torch.from_numpy(img.asnumpy())
            if idx % 1000 == 0:
                print('loading bin', idx)

            # # save img to '/home/yuange/dataset/LFW/rgb-arcface'
            # img = nd.transpose(img, axes=(1, 2, 0))  # (H, W, C)
            # # save_name = 'ind_' + str(idx) + '.bmp'
            # # import os
            # # save_name = os.path.join('/home/yuange/dataset/LFW/rgb-arcface', save_name)
            # import PIL.Image as Image
            # fig = Image.fromarray(img.asnumpy(), mode='RGB')
            # # fig.save(save_name)

        print('load finished', len(data_list))
        return data_list, issame_list


"""
Evaluation Benchmark
"""
class EvalDataset(data.Dataset):
    def __init__(self,
                 target: str = 'lfw',
                 rec_folder: str = '',
                 transform = None,
                 crop_param = (0, 0, 112, 112)
                 ):
        print("=> Pre-loading images ...")
        self.target = target
        self.rec_folder = rec_folder
        mx_reader = ReadMXNet(target, rec_folder)
        path = os.path.join(rec_folder, target + ".bin")
        all_img, issame_list = mx_reader.load_bin(path, (112, 112))
        self.all_img = all_img
        self.issame_list = []
        for i in range(len(issame_list)):
            flag = 0 if issame_list[i] else 1  # 0:is same
            self.issame_list.append(flag)

        self.transform = transform
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        self.crop_param = crop_param

    def __getitem__(self, index):
        img1 = self.all_img[index * 2]
        img2 = self.all_img[index * 2 + 1]
        same = self.issame_list[index]

        save_index = 11
        if index == save_index:
            img1.save('img1_ori.jpg')
            img2.save('img2_ori.jpg')

        img1 = img1.crop(self.crop_param)
        img2 = img2.crop(self.crop_param)
        if index == save_index:
            img1.save('img1_crop.jpg')
            img2.save('img2_crop.jpg')

        img1 = self.transform(img1)
        img2 = self.transform(img2)

        return img1, img2, same

    def __len__(self):
        return len(self.issame_list)


if __name__ == '__main__':

    import PIL.Image as Image
    import time

    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    mx.random.seed(1)

    is_gray = False

    train_set = FaceByRandOccMask(
        root_dir='/tmp/train_tmp/casia',
        local_rank=0,
        use_norm=True,
        is_gray=is_gray,
    )
    start = time.time()
    for idx in range(100):
        face, mask, label = train_set.__getitem__(idx)
        if idx < 15:
            face = ((face + 1) * 128).numpy().astype(np.uint8)
            face = np.transpose(face, (1, 2, 0))
            if is_gray:
                face = Image.fromarray(face[:, :, 0], mode='L')
            else:
                face = Image.fromarray(face, mode='RGB')
            face.save('face_{}.jpg'.format(idx))
    print('time cost: %d ms' % (int((time.time() - start) * 1000)))