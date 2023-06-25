import os

import pickle
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from inference.alignment import norm_crop, norm_crop_with_M, paste_back
from inference.utils import save, get_5_from_98, get_detector, get_lmk
from inference.PIPNet.lib.tools import get_lmk_model, demo_image


class FFPlusDataset(Dataset):
    def __init__(self,
                 ffplus_pickle: str = '/gavin/datasets/ff+/original_sequences/youtube/c23/images.pickle',
                 frames_per_id: int = 10,
                 ts_pairs_pickle: str = '/gavin/datasets/ff+/original_sequences/youtube/c23/ts_pairs.pickle',
                 image_size: int = 256,
                 transform=transforms.Compose([
                     transforms.ToTensor(),
                     transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                          std=(0.5, 0.5, 0.5)),
                 ]),
                 ):
        super(FFPlusDataset, self).__init__()

    def _load_ffplus(self, ffplus_pickle: str, frames_per_id: int):
        with open(ffplus_pickle, "rb") as handle:
            ffplus_dict: dict = pickle.load(handle)
        ''' Format, dict:
            {0: ['xxx/000/0000.png', 'xxx/000/0395.png'],
             999: ['xxx/999/0000.png', 'xxx/999/0334.png']}
        '''

        self.dict = {}
        self.ids = []
        print('Loading FF+ dataset...')
        for folder_id, full_frames in ffplus_dict.items():
            self.ids.append(folder_id)
            step = 1
            if frames_per_id != -1:  # -1 means using all frames
                step = (len(full_frames) + frames_per_id - 1) // frames_per_id
            self.dict[folder_id] = full_frames[::step]

        print('FF+ dataset loaded, total id = %d, frames_per_id = %d (-1 means using full frames)'
              % (len(self.dict), frames_per_id))

    def _load_aligned(self, aligned_pickle: str, frames_per_id: int):
        pass

    def _load_ts_pairs(self, ts_pairs_pickle: str):
        with open(ts_pairs_pickle, "rb") as handle:
            ts_pairs_dict: list = pickle.load(handle)
        ''' Format, list[[target, source], ...]:
            {0: [0, 3],
             999: [999, 960]}
        '''

        self.ts_list = [0] * len(ts_pairs_dict)
        for target_id, source_id in ts_pairs_dict:
            self.ts_list[target_id] = source_id


class FFPlusAlignCrop(FFPlusDataset):
    def __init__(self,
                 ffplus_pickle: str = '/gavin/datasets/ff+/original_sequences/youtube/c23/images.pickle',
                 frames_per_id: int = 10,
                 ts_pairs_pickle: str = '/gavin/datasets/ff+/original_sequences/youtube/c23/ts_pairs.pickle',
                 image_size: int = 256,
                 transform=transforms.Compose([
                     transforms.ToTensor(),
                     transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                          std=(0.5, 0.5, 0.5)),
                 ]),
                 ):
        super(FFPlusAlignCrop, self).__init__()

        self.frames_per_id = frames_per_id
        self._load_ffplus(ffplus_pickle, frames_per_id=-1)
        self._load_ts_pairs(ts_pairs_pickle)

        self.image_size = image_size
        self.transform = transform

        ''' face alignment '''
        self.net, self.detector = get_lmk_model()
        self.net.eval()
        # self.t_skip = np.zeros((len(self.ts_list)), dtype=np.uint32)  # skip noise frames
        # self.s_skip = np.zeros((len(self.ts_list)), dtype=np.uint32)  # skip noise frames
        print('alignment model loaded')

        ''' check face '''
        from tddfa_v2.FaceBoxes import FaceBoxes
        self.face_boxes = FaceBoxes()
        print('detection box model loaded')

    def __getitem__(self, index):
        t_id = index // self.frames_per_id
        offset = index % self.frames_per_id
        step = (len(self.dict[t_id]) + self.frames_per_id - 1) // self.frames_per_id
        offset *= step
        s_id = self.ts_list[t_id]

        t_img = None
        skip = 0
        while t_img is None:
            t_img = self._check_lmk_box(t_id, offset=offset + skip)
            skip += 1 if t_img is None else 0

        s_img = None
        skip = 0
        while s_img is None:
            s_img = self._check_lmk_box(s_id, offset=0 + skip)
            skip += 1 if s_img is None else 0

        # t_img = np.array(Image.open(self.dict[t_id][offset]).convert("RGB")).astype(np.uint8)
        # s_img = np.array(Image.open(self.dict[s_id][0]).convert("RGB")).astype(np.uint8)  # use first or random image?
        # # t_lmk = get_5_from_98(demo_image(t_img, self.net, self.detector)[0])
        # t_img = norm_crop(t_img, t_lmk, 256, mode='ffhq', borderValue=0.0)
        # s_lmk = get_5_from_98(demo_image(s_img, self.net, self.detector)[0])
        # s_img = norm_crop(s_img, s_lmk, 256, mode='ffhq', borderValue=0.0)

        if self.transform is not None:
            t_img = self.transform(t_img)
            s_img = self.transform(s_img)

        return {
            "target_image": t_img,
            "source_image": s_img,
        }

    def __len__(self):
        return len(self.ids) * self.frames_per_id

    def _check_lmk_box(self, ts_id, offset):
        max_offset = len(self.dict[ts_id])
        offset %= max_offset
        full_img = np.array(Image.open(self.dict[ts_id][offset]).convert("RGB")).astype(np.uint8)

        ''' face alignment and check landmarks '''
        lmks = demo_image(full_img, self.net, self.detector)
        if len(lmks) > 0:
            lmk = get_5_from_98(lmks[0])
            cropped_img = norm_crop(full_img, lmk, 256, mode='ffhq', borderValue=0.0)
        else:
            print('Landmarks checking failed @ (id=%d, offset=%d, max=%d)' % (ts_id, offset, max_offset))
            return None

        ''' check detection boxes '''
        boxes = self.face_boxes(cropped_img)
        if len(boxes) > 0:
            return cropped_img
        else:
            print('Detection boxes checking failed @ (id=%d, offset=%d, max=%d)' % (ts_id, offset, max_offset))
            return None


class FFPlusEvalDataset(FFPlusDataset):
    def __init__(self,
                 ffplus_root: str = '/gavin/datasets/ff+/',
                 ffplus_pickle: str = '/gavin/datasets/ff+/aligned.pickle',
                 frames_per_id: int = 10,
                 ts_pairs_pickle: str = '/gavin/datasets/ff+/original_sequences/youtube/c23/ts_pairs.pickle',
                 image_size: int = 256,
                 transform=transforms.Compose([
                     transforms.ToTensor(),
                     transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                          std=(0.5, 0.5, 0.5)),
                 ]),
                 ):
        super(FFPlusEvalDataset, self).__init__()

        default_ffplus_root = '/gavin/datasets/ff+/'
        ffplus_pickle = ffplus_pickle.replace(default_ffplus_root, ffplus_root)
        ts_pairs_pickle = ts_pairs_pickle.replace(default_ffplus_root, ffplus_root)

        self.frames_per_id = frames_per_id
        self._load_ffplus(ffplus_pickle, frames_per_id)
        self._load_ts_pairs(ts_pairs_pickle)

        for key in self.dict.keys():
            self.dict[key] = [x.replace(default_ffplus_root, ffplus_root) for x in self.dict[key]]

        self.image_size = image_size
        self.transform = transform

    def __getitem__(self, index):
        t_id = index // self.frames_per_id
        offset = index % self.frames_per_id
        s_id = self.ts_list[t_id]

        t_img = np.array(Image.open(self.dict[t_id][offset]).convert("RGB")).astype(np.uint8)
        s_img = np.array(Image.open(self.dict[s_id][0]).convert("RGB")).astype(np.uint8)  # use first or random image?

        if self.transform is not None:
            t_img = self.transform(t_img)
            s_img = self.transform(s_img)

        return {
            "target_image": t_img,
            "source_image": s_img,
        }

    def __len__(self):
        # return len(self.ids) * self.frames_per_id
        return 400 * self.frames_per_id


def generate_aligned():
    ffplus_dataset = FFPlusAlignCrop()
    batch_size = 1
    train_loader = DataLoader(ffplus_dataset,
                              batch_size=batch_size,
                              num_workers=1,
                              drop_last=False,
                              shuffle=False,
                              )
    ts_list = ffplus_dataset.ts_list

    aligned_folder = '/gavin/datasets/ff+/aligned'
    if os.path.exists(aligned_folder):
        os.system('rm -rf %s' % aligned_folder)
    os.mkdir(aligned_folder)

    def tensor_to_arr(tensor):
        arr = ((tensor + 1.) * 127.5).permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
        return arr

    for batch_idx, batch in tqdm(enumerate(train_loader)):
        # test_idx = 5080
        # if batch_idx < test_idx:
        #     print('Only test >= %d batches.' % test_idx)
        #     continue

        i_t = batch["target_image"]
        i_s = batch["source_image"]

        id_t = batch_idx // 10
        offset = batch_idx % 10
        id_s = ts_list[id_t]

        id_folder = '{:03}'.format(id_t)
        if not os.path.exists(os.path.join(aligned_folder, id_folder)):
            os.mkdir(os.path.join(aligned_folder, id_folder))

        arr_t = tensor_to_arr(i_t)
        # arr_s = tensor_to_arr(i_s)
        for b in range(batch_size):
            name_t = '{:04}.png'.format(offset)
            # name_s = '{}_{}_s.jpg'.format(batch_idx, b)
            img_t = Image.fromarray(arr_t[b])
            # img_s = Image.fromarray(arr_s[b])
            img_t.save(os.path.join(aligned_folder, id_folder, name_t))
            # img_s.save(os.path.join(demo_folder, name_s))

    # from inference.ffplus.pickle_gen import gen_ffplus
    # gen_ffplus(
    #     in_root='/gavin/datasets/ff+/aligned',
    #     out_path='/gavin/datasets/ff+/aligned.pickle',
    # )


def check_aligned():
    ffplus_dataset = FFPlusEvalDataset()
    batch_size = 1
    train_loader = DataLoader(ffplus_dataset,
                              batch_size=batch_size,
                              num_workers=1,
                              drop_last=False,
                              shuffle=False,
                              )
    ts_list = ffplus_dataset.ts_list

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

        id_t = batch_idx // 10
        offset = batch_idx % 10
        id_s = ts_list[id_t]

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

    ''' Op.1 Generate aligned ffplus '''
    # generate_aligned()

    ''' Op.2 Check aligned ffplus and save to demo folder '''
    check_aligned()
