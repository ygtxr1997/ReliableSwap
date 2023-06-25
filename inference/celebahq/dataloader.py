import os

import pickle
import time

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from inference.alignment import norm_crop, norm_crop_with_M, paste_back
from inference.utils import save, get_5_from_98, get_detector, get_lmk
from inference.PIPNet.lib.tools import get_lmk_model, demo_image


class CelebaHQDataset(Dataset):
    def __init__(self,
                 celebahq_pickle: str = '/gavin/datasets/celeba_hq/256x256.pickle',
                 image_size: int = 256,
                 transform=transforms.Compose([
                     transforms.ToTensor(),
                     transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                          std=(0.5, 0.5, 0.5)),
                 ]),
                 ):
        super(CelebaHQDataset, self).__init__()

    def _load_celebahq(self, celebahq_pickle: str):
        with open(celebahq_pickle, "rb") as handle:
            celebahq_list: list = pickle.load(handle)
        ''' Format, list:
            ['xxx/data256x256/00001.jpg',
             'xxx/data256x256/30000.jpg']
        '''

        self.imgs = []
        print('Loading Celeba-HQ dataset...')
        self.imgs = celebahq_list

        print('Celeba-HQ dataset loaded, total imgs = %d'
              % (len(self.imgs)))


class CelebaHQAlignCrop(CelebaHQDataset):
    def __init__(self,
                 celebahq_pickle: str = '/gavin/datasets/celeba_hq/256x256.pickle',
                 image_size: int = 256,
                 transform=transforms.Compose([
                     transforms.ToTensor(),
                     transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                          std=(0.5, 0.5, 0.5)),
                 ]),
                 align_mode: str = 'ffhq',
                 ):
        super(CelebaHQAlignCrop, self).__init__()

        self._load_celebahq(celebahq_pickle)

        self.image_size = image_size
        self.transform = transform
        self.align_mode = align_mode

        ''' face alignment '''
        self.net, self.detector = get_lmk_model()
        self.net.eval()
        print('alignment model loaded')

        ''' check face (warning: different align_mode will result different box result) '''
        # from tddfa_v2.FaceBoxes import FaceBoxes
        # self.face_boxes = FaceBoxes()
        # print('detection box model loaded')

    def __getitem__(self, index):
        t_id = index

        t_img, has_lmk_box = self._check_lmk_box(t_id)

        if self.transform is not None:
            t_img = self.transform(t_img)

        return {
            "target_image": t_img,
            "has_lmk_box": has_lmk_box,
        }

    def __len__(self):
        return len(self.imgs)

    def _check_lmk_box(self, t_id):
        full_img = np.array(Image.open(self.imgs[t_id]).convert("RGB")).astype(np.uint8)

        ''' face alignment and check landmarks '''
        lmks = demo_image(full_img, self.net, self.detector)
        if len(lmks) > 0:
            lmk = get_5_from_98(lmks[0])
            cropped_img = norm_crop(full_img, lmk, 256, mode=self.align_mode, borderValue=0.0)
            return cropped_img, True
        else:
            print('Landmarks checking failed @ (id=%d)' % t_id)
            return full_img, False


class CelebaHQEvalDataset(CelebaHQDataset):
    def __init__(self,
                 ts_list: list,
                 celebahq_pickle: str = '/gavin/datasets/celeba_hq/256x256_ffhq.pickle',
                 image_size: int = 256,
                 transform=transforms.Compose([
                     transforms.ToTensor(),
                     transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                          std=(0.5, 0.5, 0.5)),
                 ]),
                 dataset_len: int = 150,
                 ):
        super(CelebaHQEvalDataset, self).__init__()

        self._load_celebahq(celebahq_pickle)

        self.image_size = image_size
        self.transform = transform
        self.dataset_len = dataset_len
        self.ts_list = ts_list

    def __getitem__(self, index):
        t_id = self.ts_list[index][0]
        s_id = self.ts_list[index][1] #if index >= 50 else t_id

        t_img = np.array(Image.open(self.imgs[t_id]).convert("RGB")).astype(np.uint8)
        s_img = np.array(Image.open(self.imgs[s_id]).convert("RGB")).astype(np.uint8)

        if self.transform is not None:
            t_img = self.transform(t_img)
            s_img = self.transform(s_img)

        return {
            "target_image": t_img,
            "source_image": s_img,
        }

    def __len__(self):
        return self.dataset_len
        # return len(self.ts_list)
        # return 10000
        # return 400

class CelebaHQEvalDatasetHanbang(CelebaHQDataset):
    def __init__(self,
                 ts_list: list,
                 celebahq_target_pickle: str = '/gavin/datasets/celeba_hq/256x256_set1.pickle',
                 celebahq_source_pickle: str = '/gavin/datasets/celeba_hq/256x256_arcface.pickle',
                 image_size: int = 256,
                 transform=transforms.Compose([
                     transforms.ToTensor(),
                     transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                          std=(0.5, 0.5, 0.5)),
                 ]),
                 dataset_len: int = 150,
                 ):
        super(CelebaHQEvalDatasetHanbang, self).__init__()

        self._load_celebahq_ts(celebahq_target_pickle, celebahq_source_pickle)

        self.image_size = image_size
        self.transform = transform
        self.dataset_len = dataset_len
        self.ts_list = ts_list

    def __getitem__(self, index):
        t_id = index
        s_id = self.ts_list[t_id] if index >= 50 else t_id

        t_path = self.t_imgs[t_id]
        s_path = self.s_imgs[s_id]
        t_img = np.array(Image.open(t_path).convert("RGB")).astype(np.uint8)
        s_img = np.array(Image.open(s_path).convert("RGB")).astype(np.uint8)

        if self.transform is not None:
            t_img = self.transform(t_img)
            s_img = self.transform(s_img)

        return {
            "target_image": t_img,
            "source_image": s_img,
        }

    def __len__(self):
        return self.dataset_len

    def _load_celebahq_ts(self, celebahq_target_pickle: str, celebahq_source_pickle: str):
        with open(celebahq_target_pickle, "rb") as handle:
            celebahq_t_list: list = pickle.load(handle)
        with open(celebahq_source_pickle, "rb") as handle:
            celebahq_s_list: list = pickle.load(handle)
        ''' Format, list:
            ['xxx/data256x256/00001.jpg',
             'xxx/data256x256/30000.jpg']
        '''

        print('Loading Celeba-HQ dataset...')
        self.t_imgs = celebahq_t_list
        self.s_imgs = celebahq_s_list

        print('Celeba-HQ dataset loaded, total imgs = %d'
              % (len(self.t_imgs)))


def generate_aligned(
        aligned_folder: str = '/gavin/datasets/celeba_hq/256x256_aligned',
        pickle_path: str = '/gavin/datasets/celeba_hq/256x256_aligned.pickle',
        align_mode: str = 'ffhq'
        ):
    print('Start align and crop')
    celebahq_dataset = CelebaHQAlignCrop(
        align_mode=align_mode
    )
    batch_size = 1
    train_loader = DataLoader(celebahq_dataset,
                              batch_size=batch_size,
                              num_workers=1,
                              drop_last=False,
                              shuffle=False,
                              )

    if os.path.exists(aligned_folder):
        os.system('rm -rf %s' % aligned_folder)
    os.mkdir(aligned_folder)

    def tensor_to_arr(tensor):
        arr = ((tensor + 1.) * 127.5).permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
        return arr

    img_cnt = 0
    batch_idx = 0
    for batch in tqdm(train_loader):
        batch_idx += 1
        i_t = batch["target_image"]
        has_lmk_box = batch["has_lmk_box"]

        arr_t = tensor_to_arr(i_t)
        for b in range(batch_size):
            if not has_lmk_box[b]:
                print('skip image %d' % batch_idx - 1)
                continue
            name_t = '{:05}.jpg'.format(img_cnt)
            img_t = Image.fromarray(arr_t[b])
            img_t.save(os.path.join(aligned_folder, name_t))
            img_cnt += 1

    from inference.celebahq.pickle_gen import gen_celebahq
    gen_celebahq(
        in_root=aligned_folder,
        out_path=pickle_path
    )


def check_aligned(
        pickle_path: str = '/gavin/datasets/celeba_hq/256x256_aligned.pickle',
        ):
    print('Checking aligned dataset...')
    celebahq_ts_list = np.arange(25000)
    np.random.shuffle(celebahq_ts_list)
    ffplus_dataset = CelebaHQEvalDataset(
        ts_list=celebahq_ts_list,
        celebahq_pickle=pickle_path,
    )
    batch_size = 1
    train_loader = DataLoader(ffplus_dataset,
                              batch_size=batch_size,
                              num_workers=1,
                              drop_last=False,
                              shuffle=False,
                              )

    demo_folder = './demo'
    if os.path.exists(demo_folder):
        os.system('rm -rf %s' % demo_folder)
    os.mkdir(demo_folder)

    def tensor_to_arr(tensor):
        arr = ((tensor + 1.) * 127.5).permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
        return arr

    for batch_idx, batch in tqdm(enumerate(train_loader)):
        test_idx = 50
        if batch_idx > test_idx:
            print('Only test < %d batches.' % test_idx)
            exit(0)

        i_t = batch["target_image"]
        i_s = batch["source_image"]

        arr_t = tensor_to_arr(i_t)
        arr_s = tensor_to_arr(i_s)
        for b in range(batch_size):
            name_t = '{}_{}_t.jpg'.format(batch_idx, b)
            name_s = '{}_{}_s.jpg'.format(batch_idx, b)
            img_t = Image.fromarray(arr_t[b])
            img_s = Image.fromarray(arr_s[b])
            img_t.save(os.path.join(demo_folder, name_t))
            img_s.save(os.path.join(demo_folder, name_s))


def set_random_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    set_random_seed(0)

    from tqdm import tqdm
    from torch.utils.data.dataloader import DataLoader
    torch.multiprocessing.set_start_method('spawn')

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--align_mode", type=str, default="ffhq", choices=['ffhq', 'set1', 'arcface'])
    args = parser.parse_args()

    align_mode = args.align_mode
    assert align_mode in ('ffhq', 'set1', 'arcface')
    aligned_folder = '/gavin/datasets/celeba_hq/256x256_%s' % align_mode
    pickle_path = '/gavin/datasets/celeba_hq/256x256_%s.pickle' % align_mode

    ''' Op.1 Align and crop '''
    generate_aligned(
        aligned_folder, pickle_path, align_mode
    )

    ''' Op.2 Check '''
    check_aligned(
        pickle_path
    )
