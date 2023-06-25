import os.path
import random
import numpy as np
from PIL import Image
import pickle
import cv2

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from pytorch_lightning.core.datamodule import LightningDataModule

# PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True

class Image512QualityDataModule(LightningDataModule):
    def __init__(self,
                 batch_size=12,
                 same_rate=10,
                 image_size=256,
                 ):
        super(Image512QualityDataModule, self).__init__()
        self.batch_size = batch_size
        self.same_rate = same_rate
        self.image_size = image_size

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: str = None) -> None:
        pass

    def test_dataloader(self):
        image512quality_test = BatchTrainDataset(
            same_rate=10,
            image_size=256,
            top_k=1500000,
        )
        return DataLoader(
            image512quality_test,
            batch_size=self.batch_size,
            num_workers=32,
            drop_last=True,
            shuffle=True,
        )


class BatchTrainDataset(Dataset):
    def __init__(
        self,
        img_root: str = "/gavin/datasets/original/image_512_quality.pickle",
        same_rate: int = 50,
        image_size: int = 256,
        top_k: int = 1500000,
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]),
    ):
        super(BatchTrainDataset, self).__init__()
        with open(img_root, "rb") as handle:
            picke_list = pickle.load(handle)
        picke_list = sorted(picke_list, key=lambda x: x[1])[::-1]
        full_len = len(picke_list)

        picke_list = picke_list[:top_k]  # 4m out of 5m
        random.shuffle(picke_list)

        pickle_dict = {}
        for img, iqa in picke_list:
            id = img.split("/")[-2]
            if id not in pickle_dict:
                pickle_dict[id] = [img]
            else:
                pickle_dict[id] += [img]

        """
        Format, key-sublist:
        { 'n001858': ['xxx/hd_align_512/n001858/0067_01.jpg', 'xxx/hd_align_512/n001858/0137_01.jpg'],
          'n002036': ['xxx/hd_align_512/n002036/0383_01.jpg'] }
        """
        self.pickle_dict = pickle_dict

        """
        Format, flatten list:
        ['xxx/hd_align_512/n001858/0067_01.jpg',
         'xxx/hd_align_512/n001858/0137_01.jpg',
         'xxx/hd_align_512/n002036/0383_01.jpg']
        """
        self.img_files = [
            item for sublist in list(pickle_dict.values()) for item in sublist
        ]

        self.ids = list(pickle_dict.keys())

        self.image_size = image_size
        self.same_rate = same_rate
        self.transform = transform
        print("setup dataset finished, files count = %d, id count = %d, "
              "full count = %d" % (len(self.img_files), len(self.ids), full_len))

    def __getitem__(self, index):
        l = self.__len__()
        f_img = np.asarray(Image.open(self.img_files[index]).convert("RGB").resize((self.image_size, self.image_size)))
        same = torch.zeros(1).float()
        s_idx = index
        while s_idx == index:
            s_idx = np.random.randint(0, l)
        s_img = np.asarray(Image.open(self.img_files[s_idx]).convert("RGB").resize((self.image_size, self.image_size)))
        f_mask = np.zeros((self.image_size, self.image_size), dtype=np.uint8)

        if self.transform is not None:
            f_img = self.transform(f_img)
            s_img = self.transform(s_img)
            f_mask = transforms.ToTensor()(f_mask)

        return {
            "target_image": f_img,
            "source_image": s_img,
            "target_mask": f_mask,
            "same": same,
            "pair_str": '{:08d}_{:08d}'.format(index, s_idx),
        }

    def __len__(self):
        return len(self.img_files)


import csv
import imageio
class DaGanDataset(Dataset):
    def __init__(self):
        super(DaGanDataset, self).__init__()
        self.crop_root = '/gavin/datasets/celebvhq/crop'
        self.reen_root = '/gavin/datasets/celebvhq/reen'

        self.groups = []
        self.get_groups()
        print('DaGan dataset loaded (len=%d).' % self.__len__())

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def __getitem__(self, index):
        group = self.groups[index]
        t_name, t_frame_idx, s_name, s_frame_idx = group
        group_folder = '%s#%s' % (t_name[:-len('.mp4')], s_name[:-len('.mp4')])

        t_frame = np.array(Image.open(os.path.join(self.reen_root, group_folder, 't_frame.jpg')))
        s_frame = np.array(Image.open(os.path.join(self.reen_root, group_folder, 's_frame.jpg')))

        ts_frames, ts_fps = self._read_video(os.path.join(self.reen_root, group_folder, 'reen_ts.mp4'))
        st_frames, st_fps = self._read_video(os.path.join(self.reen_root, group_folder, 'reen_st.mp4'))
        print('%s ts:%d, t_idx:%d, st:%d, s_idx:%d' % (t_name, len(ts_frames), t_frame_idx, len(st_frames), s_frame_idx))
        ts_reen_image = ts_frames[s_frame_idx]
        st_reen_image = st_frames[t_frame_idx]

        t_frame = self.transform(t_frame)
        s_frame = self.transform(s_frame)
        ts_reen_image = self.transform(ts_reen_image)
        st_reen_image = self.transform(st_reen_image)

        return {
            "target_image": t_frame,
            "source_image": s_frame,
            "reen_ts": ts_reen_image,
            "reen_st": st_reen_image,
            "pair_str": group_folder,
        }

    def __len__(self):
        return len(self.groups)

    def get_groups(self, csv_file: str = '/gavin/code/DaGAN/groups_demo.csv'):
        with open(csv_file, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                self.groups.append((row['t_name'], int(row['t_frame_idx']), row['s_name'], int(row['s_frame_idx'])))

    def _read_video(self, video_path: str):
        reader = imageio.get_reader(video_path)
        fps = reader.get_meta_data()['fps']
        video_frames = []
        for im in reader:
            video_frames.append(im)
        reader.close()
        return video_frames, fps



if __name__ == '__main__':
    train_set = BatchTrainDataset(top_k=1500000)
    train_set.__getitem__(0)