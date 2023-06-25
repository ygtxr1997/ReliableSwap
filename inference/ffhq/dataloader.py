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


class FFHQDataset(Dataset):
    def __init__(self,
                 ffhq_pickle: str = '/gavin/datasets/ffhq/1024x1024.pickle',
                 image_size: int = 256,
                 transform=transforms.Compose([
                     transforms.ToTensor(),
                     transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                          std=(0.5, 0.5, 0.5)),
                 ]),
                 ):
        super(FFHQDataset, self).__init__()

    def _load_ffhq(self, ffhq_pickle: str):
        with open(ffhq_pickle, "rb") as handle:
            ffhq_list: list = pickle.load(handle)
        ''' Format, list:
            ['xxx/images1024x1024/00001.jpg',
             'xxx/images1024x1024/30000.jpg']
        '''

        self.imgs = []
        print('Loading FFHQ dataset...')
        self.imgs = ffhq_list

        print('FFHQ dataset loaded, total imgs = %d'
              % (len(self.imgs)))


class FFHQAlignCrop(FFHQDataset):
    def __init__(self,
                 ffhq_pickle: str = '/gavin/datasets/ffhq/1024x1024.pickle',
                 image_size: int = 256,
                 transform=transforms.Compose([
                     transforms.ToTensor(),
                     transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                          std=(0.5, 0.5, 0.5)),
                 ]),
                 ):
        super(FFHQAlignCrop, self).__init__()

        self._load_ffhq(ffhq_pickle)

        self.image_size = image_size
        self.transform = transform

        ''' face alignment '''
        self.net, self.detector = get_lmk_model()
        self.net.eval()
        print('alignment model loaded')

        ''' check face '''
        from tddfa_v2.FaceBoxes import FaceBoxes
        self.face_boxes = FaceBoxes()
        print('detection box model loaded')

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
            cropped_img = norm_crop(full_img, lmk, 1024, mode='ffhq', borderValue=0.0)
        else:
            print('Landmarks checking failed @ (id=%d)' % t_id)
            return full_img, False

        ''' check detection boxes '''
        boxes = self.face_boxes(cropped_img)
        if len(boxes) > 0:
            return cropped_img, True
        else:
            print('Detection boxes checking failed @ (id=%d)' % t_id)
            return full_img, False


class FFHQEvalDataset(FFHQDataset):
    def __init__(self,
                 ts_list: list,
                 ffhq_pickle: str = '/gavin/datasets/ffhq/1024x1024.pickle',
                 image_size: int = 256,
                 transform=transforms.Compose([
                     transforms.ToTensor(),
                     transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                          std=(0.5, 0.5, 0.5)),
                 ]),
                 dataset_len: int = 100,
                 ):
        super(FFHQEvalDataset, self).__init__()

        self._load_ffhq(ffhq_pickle)

        self.image_size = image_size
        self.transform = transform
        self.dataset_len = dataset_len
        self.ts_list = ts_list

    def __getitem__(self, index):
        t_id = index
        s_id = self.ts_list[t_id]

        size = (self.image_size, self.image_size)
        t_img = np.array(Image.open(self.imgs[t_id]).convert("RGB").resize(size)).astype(np.uint8)
        s_img = np.array(Image.open(self.imgs[s_id]).convert("RGB").resize(size)).astype(np.uint8)

        if self.transform is not None:
            t_img = self.transform(t_img)
            s_img = self.transform(s_img)

        return {
            "target_image": t_img,
            "source_image": s_img,
        }

    def __len__(self):
        return self.dataset_len


def generate_aligned():
    print('Start align and crop')
    ffhq_dataset = FFHQAlignCrop()
    batch_size = 1
    train_loader = DataLoader(ffhq_dataset,
                              batch_size=batch_size,
                              num_workers=1,
                              drop_last=False,
                              shuffle=False,
                              )

    aligned_folder = '/gavin/datasets/ffhq/1024x1024_aligned'
    if os.path.exists(aligned_folder):
        os.system('rm -rf %s' % aligned_folder)
    os.mkdir(aligned_folder)

    def tensor_to_arr(tensor):
        arr = ((tensor + 1.) * 127.5).permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
        return arr

    img_cnt = 0
    batch_idx = 0
    for batch in tqdm(train_loader):
        i_t = batch["target_image"]
        has_lmk_box = batch["has_lmk_box"]

        arr_t = tensor_to_arr(i_t)
        for b in range(batch_size):
            if not has_lmk_box[b]:
                print('skip image %d' % batch_idx)
                continue
            name_t = '{:05}.jpg'.format(img_cnt)
            img_t = Image.fromarray(arr_t[b])
            img_t.save(os.path.join(aligned_folder, name_t))
            img_cnt += 1

        batch_idx += 1

    from inference.ffhq.pickle_gen import gen_ffhq
    gen_ffhq(
        in_root=aligned_folder,
        out_path='/gavin/datasets/ffhq/1024x1024_aligned.pickle',
    )

    return img_cnt


def check_aligned(dataset_len: int):
    print('Checking aligned dataset...(total cnt = %d)' % dataset_len)
    ffhq_ts_list = np.arange(dataset_len)
    np.random.shuffle(ffhq_ts_list)
    ffplus_dataset = FFHQEvalDataset(
        ts_list=ffhq_ts_list
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


if __name__ == '__main__':
    from tqdm import tqdm
    from torch.utils.data.dataloader import DataLoader
    torch.multiprocessing.set_start_method('spawn')

    ''' Op.1 Align and crop (FFHQ doesn't need to be aligned) '''
    length = generate_aligned()

    ''' Op.2 Check '''
    check_aligned(length)
