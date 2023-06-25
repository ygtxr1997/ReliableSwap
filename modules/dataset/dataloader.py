import os.path

import torch
from torch.utils.data import Dataset
from torchvision import transforms

import math
import pickle
import random
import numpy as np
from PIL import Image

from modules.dataset.occluder import occlude_with_objects
import RealOcc.msml_occ as msml_occ


def same_or_not(percent):
    return random.randrange(100) < percent


def color_masking(img, r, g, b):
    return np.logical_and(
        np.logical_and(img[:, :, 0] == r, img[:, :, 1] == g), img[:, :, 2] == b
    )


def logical_or_masks(mask_list):
    mask_all = np.zeros_like(mask_list[0], dtype=bool)
    for mask in mask_list:
        mask_all = np.logical_or(mask_all, mask)
    return mask_all


def parsing2mask(paring):
    img_numpy = np.array(paring)

    mask_nose = color_masking(img_numpy, 76, 153, 0)
    mask_left_eye = color_masking(img_numpy, 204, 0, 204)
    mask_right_eye = color_masking(img_numpy, 51, 51, 255)
    mask_skin = color_masking(img_numpy, 204, 0, 0)
    mask_left_eyebrow = color_masking(img_numpy, 255, 204, 204)
    mask_right_eyebrow = color_masking(img_numpy, 0, 255, 255)
    mask_up_lip = color_masking(img_numpy, 255, 255, 0)
    mask_mouth_inside = color_masking(img_numpy, 102, 204, 0)
    mask_down_lip = color_masking(img_numpy, 0, 0, 153)
    mask_left_ear = color_masking(img_numpy, 255, 0, 0)
    mask_right_ear = color_masking(img_numpy, 102, 51, 0)

    mask_face = logical_or_masks(
        [
            mask_nose,
            mask_left_eye,
            mask_right_eye,
            mask_skin,
            mask_left_eyebrow,
            mask_right_eyebrow,
            mask_up_lip,
            mask_mouth_inside,
            mask_down_lip,
            mask_left_ear,
            mask_right_ear,
        ]
    )
    mask_face = 1.0 * mask_face
    mask_face = Image.fromarray(np.array(mask_face))
    return mask_face


#############################################
#
# [BEGIN] New dataset with random pair / occ
#
#############################################


class PickleTrainDataset(Dataset):
    def __init__(
        self,
        # img_root="/apdcephfs/share_1290939/ahbanliang/datasets/original/hd_align_512_more_jaw/data.pickle",
        img_root="/gavin/datasets/original/hd_align_512_more_jaw/data.pickle",
        same_rate=50,
        transform=transforms.Compose(
            [
                # transforms.Resize((256, 256)),
                # transforms.CenterCrop((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        ),
        ffhq_mode=False,
        use_occ=False,
    ):
        super(PickleTrainDataset, self).__init__()
        with open(img_root, "rb") as handle:
            pickle_dict = pickle.load(handle)

        self.pickle_dict = pickle_dict
        self.img_files = [
            item for sublist in list(pickle_dict.values()) for item in sublist
        ]
        self.ids = list(pickle_dict.keys())
        self.same_rate = same_rate
        self.transform = transform
        self.ffhq_mode = ffhq_mode
        self.use_occ = use_occ

        with open(
            "/gavin/datasets/hanbang/hear/hands_voc.pkl",
            "rb"
        ) as handle:
            occluders = pickle.load(handle)
        self.occluders = occluders

        print("setup dataset finished")

    def __getitem__(self, index):
        l = self.__len__()
        if not self.ffhq_mode:
            t_id = self.img_files[index].split("/")[-2]
        else:
            t_id = self.img_files[index]
        f_img = np.array(
            Image.open(self.img_files[index]).convert("RGB").resize((256, 256))
        )
        is_same = same_or_not(self.same_rate)
        if self.use_occ:
            f_img, _ = occlude_with_objects(f_img, self.occluders, count=2)

        if is_same:
            s_id = t_id
        else:
            s_id = random.choice(self.ids)
        if s_id == t_id:
            same = torch.ones(1).float()
            if not self.ffhq_mode:
                s_idx = random.choice(self.pickle_dict[s_id])
            else:
                s_idx = self.img_files[index]
            s_img = np.array(Image.open(s_idx).convert("RGB").resize((256, 256)))
        else:
            same = torch.zeros(1).float()
            s_idx = random.randrange(l)
            s_idx = self.img_files[s_idx]
            s_img = np.array(Image.open(s_idx).convert("RGB").resize((256, 256)))

        f_mask = np.zeros((256, 256), dtype=np.uint8)

        if self.transform is not None:
            f_img = self.transform(f_img)
            s_img = self.transform(s_img)
            f_mask = transforms.ToTensor()(f_mask)

        return {
            "target_image": f_img,
            "source_image": s_img,
            "target_mask": f_mask,
            "same": same,
        }

    def __len__(self):
        return len(self.img_files)


class PickleValDataset(Dataset):
    def __init__(
        self,
        img_root,
        transform=transforms.Compose(
            [
                # transforms.Resize((256, 256)),
                # transforms.CenterCrop((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        ),
        ffhq_mode=False,
    ):
        super(PickleValDataset, self).__init__()
        self.ffhq_mode = ffhq_mode
        with open(img_root, "rb") as handle:
            pickle_dict = pickle.load(handle)

        with open(
            "/gavin/datasets/hanbang/hear/hands_voc.pkl",
            "rb"
        ) as handle:
            occluders = pickle.load(handle)
        self.occluders = occluders

        self.pickle_dict = pickle_dict
        self.img_files = [
            item for sublist in list(pickle_dict.values()) for item in sublist
        ]
        self.img_files = random.sample(self.img_files, 50)

        self.transform = transform

    def __getitem__(self, index):
        l = len(self.img_files)

        t_idx = index // l
        s_idx = index % l

        if t_idx == s_idx:
            same = torch.ones(1)
        else:
            same = torch.zeros(1)

        f_img = Image.open(self.img_files[t_idx])
        s_img = Image.open(self.img_files[s_idx])

        f_img = np.array(f_img.convert("RGB").convert("RGB").resize((256, 256)))
        s_img = np.array(s_img.convert("RGB").convert("RGB").resize((256, 256)))
        f_img, _ = occlude_with_objects(f_img, self.occluders, count=2)
        f_mask = np.zeros((256, 256), dtype=np.uint8)

        if self.transform is not None:
            f_img = self.transform(f_img)
            s_img = self.transform(s_img)
            f_mask = transforms.ToTensor()(f_mask)

        return {
            "target_image": f_img,
            "source_image": s_img,
            "target_mask": f_mask,
            "same": same,
        }

    def __len__(self):
        return len(self.img_files) * len(self.img_files)


############################################
#
#  [END] New dataset with random pair / occ
#
############################################


################################################
#
#  [begin] Batch same and Batch diff
#
################################################


class BatchTrainDataset(Dataset):
    def __init__(
        self,
        img_root="/gavin/datasets/original/image_512_quality.pickle",
        same_rate=20,
        image_size=512,
        transform=transforms.Compose(
            [
                # transforms.Resize((512, 512)),
                # transforms.CenterCrop((512, 512)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        ),
        ffhq_mode=False,
        top_k=3000000,
        use_occ=False,
        simswap_mode = False,
    ):
        super(BatchTrainDataset, self).__init__()
        with open(img_root, "rb") as handle:
            picke_list = pickle.load(handle)
        picke_list = sorted(picke_list, key=lambda x: x[1])[::-1]

        picke_list = picke_list[:top_k]  # 4m out of 5m
        random.shuffle(picke_list)

        ''' occlusions '''
        self.use_occ = use_occ
        if use_occ:
            with open("/gavin/datasets/hanbang/hear/hands_voc.pkl", "rb") as handle:
                occluders = pickle.load(handle)
            self.occluders = occluders
            self.trans_occ = (
                msml_occ.RandomRect(),
                msml_occ.RandomEllipse(),
                msml_occ.RandomConnectedPolygon(),
                msml_occ.RandomGlassesList(['/gavin/code/MSML/datasets/augment/occluder/glasses_crop',
                                            '/gavin/code/MSML/datasets/augment/occluder/eleglasses_crop']),
                msml_occ.RandomScarf('/gavin/code/MSML/datasets/augment/occluder/scarf_crop'),
                msml_occ.RandomRealObject('/gavin/code/MSML/datasets/augment/occluder/object_train'),
                # msml_occ.RealOcc(occ_type='rand'),
                msml_occ.RealOcc(occ_type='hand'),
                msml_occ.RealOcc(occ_type='coco'),
            )

        pickle_dict = {}
        celeb_asia_cnt = 0
        for img, iqa in picke_list:
            id = img.split("/")[-2]  # vggface2
            if simswap_mode and (not 'n' in id):  # celeb-asian
                # simswap_mode will delete some celeb-asian images
                id = img.split("/")[-2] + img.split("/")[-1]
                celeb_asia_cnt += 1
                if celeb_asia_cnt >= 60000:
                    continue
            if id not in pickle_dict:
                pickle_dict[id] = [img]
            else:
                pickle_dict[id] += [img]

        self.image_size = image_size
        self.pickle_dict = pickle_dict
        self.img_files = [
            item for sublist in list(pickle_dict.values()) for item in sublist
        ]
        self.ids = list(pickle_dict.keys())
        self.same_rate = same_rate
        self.transform = transform
        self.ffhq_mode = ffhq_mode
        self.simswap_mode = simswap_mode
        print("setup BatchTrainDataset dataset finished (len=%d, simswap_mode=%s)" % (
            self.__len__(), self.simswap_mode))

    def _getitem_faceshifter_mode(self, index):
        if index >= len(self.img_files):  # '>=' instead of '>'
            index = (np.random.randint(index) * index) % len(self.img_files)
        l = self.__len__()
        t_img = np.array(Image.open(self.img_files[index]).convert("RGB")
                         .resize((self.image_size, self.image_size))).astype(np.uint8)
        is_same = torch.zeros(1).float()

        if np.random.randint(0, 100) > self.same_rate:  # not same
            s_idx = index
            while s_idx == index:
                s_idx = np.random.randint(0, l)
            s_img = np.array(Image.open(self.img_files[s_idx]).convert("RGB").
                             resize((self.image_size, self.image_size))).astype(np.uint8)
            is_same[0] = 0
        else:  # is same
            s_img = t_img
            is_same[0] = 1

        f_mask = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
        return t_img, s_img, is_same, f_mask

    def _getitem_simswap_mode(self, index):
        if index >= len(self.ids):  # '>=' instead of '>'
            index = (np.random.randint(index) * index) % len(self.ids)
        id_name = self.ids[index]
        id_imgs = self.pickle_dict[id_name]

        t_file_path = id_imgs[np.random.randint(0, len(id_imgs))]
        s_file_path = id_imgs[np.random.randint(0, len(id_imgs))]
        t_img = np.array(Image.open(t_file_path).convert('RGB').resize((self.image_size,
                                                                        self.image_size))).astype(np.uint8)
        s_img = np.array(Image.open(s_file_path).convert('RGB').resize((self.image_size,
                                                                        self.image_size))).astype(np.uint8)

        ''' self.same_rate dose not work here '''
        is_same = torch.ones(1).float()  # always same (1), from the same folder
        f_mask = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
        return t_img, s_img, is_same, f_mask

    def __getitem__(self, index):
        if not self.simswap_mode:
            t_img, s_img, is_same, f_mask = self._getitem_faceshifter_mode(index)
        else:
            t_img, s_img, is_same, f_mask = self._getitem_simswap_mode(index)

        if self.use_occ and np.random.randint(0, 100) >= 30:
            # t_img, _ = occlude_with_objects(t_img, self.occluders, count=2)
            t_img, msk = self.trans_occ[np.random.randint(0, len(self.trans_occ))](Image.fromarray(t_img))
            t_img = np.array(t_img)

        if self.transform is not None:
            t_img = self.transform(t_img)
            s_img = self.transform(s_img)
            f_mask = transforms.ToTensor()(f_mask)

        return {
            "target_image": t_img,
            "source_image": s_img,
            "target_mask": f_mask,
            "same": is_same,
        }

    def __len__(self):
        if self.simswap_mode:
            return len(self.ids)
        else:
            return len(self.img_files)


class BatchValDataset(Dataset):
    def __init__(
        self,
        img_root,
        image_size=512,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        ),
        ffhq_mode=False,
        use_occ=False,
        id_cnt: int = 50,
    ):
        super(BatchValDataset, self).__init__()
        self.ffhq_mode = ffhq_mode
        with open(img_root, "rb") as handle:
            picke_list = pickle.load(handle)

        self.use_occ = use_occ
        if use_occ:
            with open(
                    "/gavin/datasets/hanbang/hear/hands_voc.pkl",
                    "rb"
            ) as handle:
                occluders = pickle.load(handle)
            self.occluders = occluders
            self.trans_occ = (
                msml_occ.RandomRect(),
                msml_occ.RandomEllipse(),
                msml_occ.RandomConnectedPolygon(),
                msml_occ.RandomGlassesList(['/gavin/code/MSML/datasets/augment/occluder/glasses_crop',
                                            '/gavin/code/MSML/datasets/augment/occluder/eleglasses_crop']),
                msml_occ.RandomScarf('/gavin/code/MSML/datasets/augment/occluder/scarf_crop'),
                msml_occ.RandomRealObject('/gavin/code/MSML/datasets/augment/occluder/object_train'),
                # msml_occ.RealOcc(occ_type='rand'),
                msml_occ.RealOcc(occ_type='hand'),
                msml_occ.RealOcc(occ_type='coco'),
            )

        picke_list = sorted(picke_list, key=lambda x: x[1])[::-1]
        # picke_list = picke_list[:10000]
        random.shuffle(picke_list)
        picke_list = picke_list[:10000]

        pickle_dict = {}
        for img, iqa in picke_list:
            id = img.split("/")[-2]
            if id not in pickle_dict:
                pickle_dict[id] = [img]
            else:
                # pickle_dict[id] += [img]
                pass

        self.pickle_dict = pickle_dict
        self.img_files = [
            item for sublist in list(pickle_dict.values()) for item in sublist
        ]
        self.img_files = random.sample(self.img_files, id_cnt)
        self.image_size = image_size
        self.transform = transform

    def __getitem__(self, index):
        l = len(self.img_files)

        # t_idx = index // l
        # s_idx = index % l

        t_idx = index
        s_idx = t_idx
        while s_idx == t_idx:
            s_idx = np.random.randint(l)

        if t_idx == s_idx:
            same = torch.ones(1)
        else:
            same = torch.zeros(1)

        f_img = np.array(
            Image.open(self.img_files[t_idx])
            .convert("RGB")
            .resize((self.image_size, self.image_size))
        )
        s_img = np.array(
            Image.open(self.img_files[s_idx])
            .convert("RGB")
            .resize((self.image_size, self.image_size))
        )

        if self.use_occ and np.random.randint(0, 100) >= 30:
            # f_img, _ = occlude_with_objects(f_img, self.occluders, count=2)
            f_img, msk = self.trans_occ[np.random.randint(0, len(self.trans_occ))](Image.fromarray(f_img))
            f_img = np.array(f_img)

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
        }

    def __len__(self):
        # return len(self.img_files) * len(self.img_files)
        return len(self.img_files)


class TripletTrainDataset(Dataset):
    def __init__(self,
                 vanilla_pickle: str = "/gavin/datasets/original/image_512_quality.pickle",
                 triplet_pickle: str = "/gavin/datasets/triplet.pickle",
                 same_rate: int = 20,
                 image_size: int = 256,
                 transform=transforms.Compose([
                     transforms.ToTensor(),
                     transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                          std=(0.5, 0.5, 0.5)),
                 ]),
                 top_k: int = 1500000,
                 triplet_ratio: int = 100,
                 use_real: bool = False,
                 ):
        super(TripletTrainDataset, self).__init__()

        self.use_real = use_real
        self.triplet_count_per_folder = 4 if use_real else 2

        self._load_vanilla(vanilla_pickle, top_k)
        self._load_triplet(triplet_pickle, triplet_ratio)

        """ Other params """
        # self.round_1, self.round_2 = self._get_round()
        self.same_rate = same_rate
        self.image_size = image_size
        self.transform = transform

        print("Setup TripletTrainDataset finished (Vanilla:%d, Triplet:%d)" % (
            self.len_vanilla, self.len_triplet
        ))

    def _load_vanilla(self, vanilla_pickle: str, top_k: int):
        """ (1) Vanilla images sorted by IQA """
        with open(vanilla_pickle, "rb") as handle:
            vanilla_list = pickle.load(handle)
        vanilla_list = sorted(vanilla_list, key=lambda x: x[1])[::-1]

        vanilla_list = vanilla_list[:top_k]  # topk images out of 5m
        random.shuffle(vanilla_list)

        vanilla_dict = {}
        for img, iqa in vanilla_list:
            id = img.split("/")[-2]
            if id not in vanilla_dict:
                vanilla_dict[id] = [img]
            else:
                vanilla_dict[id] += [img]

        ''' Format, key-sublist:
            { 'n001858': ['xxx/hd_align_512/n001858/0067_01.jpg', 'xxx/hd_align_512/n001858/0137_01.jpg'],
              'n002036': ['xxx/hd_align_512/n002036/0383_01.jpg'] }
        '''
        self.vanilla_dict = vanilla_dict

        ''' Format, flatten list:
            ['xxx/hd_align_512/n001858/0067_01.jpg',
             'xxx/hd_align_512/n001858/0137_01.jpg',
             'xxx/hd_align_512/n002036/0383_01.jpg']
        '''
        self.vanilla_files = [item for sublist in list(vanilla_dict.values()) for item in sublist]
        self.ids = list(vanilla_dict.keys())
        self.len_vanilla = len(self.vanilla_files)  # equals to topk
        print('Vanilla dataset loaded from %s' % vanilla_pickle)

    def _load_triplet(self, triplet_pickle: str, triplet_ratio: int = 100):
        """ (2) Triplet Rotational Group images (source, target, st, ts) """
        self.trip_filename = {'s': 'source.jpg', 't': 'target.jpg',
                              'st': 'output_st.jpg', 'ts': 'output_ts.jpg'}
        with open(triplet_pickle, "rb") as handle:
            triplet_list = pickle.load(handle)
        ''' Format, list:
            ['xxx/triplet/00000000_00008210',
             'xxx/triplet/00000044_00008316',]
        '''
        triplet_used = int(len(triplet_list) * triplet_ratio / 100)
        self.triplet_folders = triplet_list[:triplet_used]
        self.len_triplet = len(self.triplet_folders) * self.triplet_count_per_folder
        print('Triplet dataset loaded from %s, folders %d, used %d, images_per_folder: %d' % (
            triplet_pickle, len(triplet_list), len(self.triplet_folders), self.triplet_count_per_folder))

    def _get_round(self,):
        g = math.gcd(self.len_vanilla, self.len_triplet)
        return self.len_vanilla // g, self.len_triplet // g

    def _getitem_vanilla(self, index, type_code: torch.Tensor):
        """ (1) Vanilla """
        # round_cnt = index // (self.round_1 + self.round_2)
        # round_offset = index % (self.round_1 + self.round_2)
        # idx_1 = self.round_1 * round_cnt + min(round_offset, self.round_1 - 1)
        idx_1 = index

        t_img = np.array(Image.open(self.vanilla_files[idx_1]).convert("RGB").
                         resize((self.image_size, self.image_size))).astype(np.uint8)
        if np.random.randint(0, 100) > self.same_rate:  # not same, type 0
            s_idx = idx_1
            while s_idx == idx_1:
                s_idx = np.random.randint(0, self.len_vanilla)
            s_img = np.array(Image.open(self.vanilla_files[s_idx]).convert("RGB").
                             resize((self.image_size, self.image_size))).astype(np.uint8)
            r_img = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
            type_code[0] = 0
        else:  # is same, type 1
            s_img = t_img
            r_img = t_img
            type_code[0] = 1

        return t_img, s_img, r_img, type_code

    def _getitem_triplet(self, index, type_code: torch.Tensor):
        """ (2) Triplet """
        # round_cnt = index // (self.round_1 + self.round_2)
        # round_offset = index % (self.round_1 + self.round_2)
        # idx_2 = self.round_2 * round_cnt + max(round_offset - self.round_1, 0)
        idx_2 = index

        # each 4 or 2 groups share the same folder
        folder = self.triplet_folders[idx_2 // self.triplet_count_per_folder]
        s_r = np.array(Image.open(os.path.join(folder, self.trip_filename['s'])).convert("RGB").
                       resize((self.image_size, self.image_size))).astype(np.uint8)
        t_r = np.array(Image.open(os.path.join(folder, self.trip_filename['t'])).convert("RGB").
                       resize((self.image_size, self.image_size))).astype(np.uint8)
        st_r = np.array(Image.open(os.path.join(folder, self.trip_filename['st'])).convert("RGB").
                        resize((self.image_size, self.image_size))).astype(np.uint8)
        ts_r = np.array(Image.open(os.path.join(folder, self.trip_filename['ts'])).convert("RGB").
                        resize((self.image_size, self.image_size))).astype(np.uint8)

        if self.triplet_count_per_folder == 4:
            ''' (1) 4 cycle triplets per folder (type:2,3,4,5) '''
            if idx_2 % 4 == 0:  # type 2
                s_img, t_img, r_img = s_r, t_r, st_r
            elif idx_2 % 4 == 1:  # type 3
                s_img, t_img, r_img = t_r, s_r, ts_r
            elif idx_2 % 4 == 2:  # type 4
                s_img, t_img, r_img = st_r, ts_r, s_r
            else:  # type 5
                s_img, t_img, r_img = ts_r, st_r, t_r
            type_code[0] = (idx_2 % 4) + 2
        else:
            ''' (2) 2 cycle triplets per folder (type:4,5) '''
            if idx_2 % 2 == 0:  # type 4
                s_img, t_img, r_img = st_r, ts_r, s_r
            else:  # type 5
                s_img, t_img, r_img = ts_r, st_r, t_r
            type_code[0] = (idx_2 % 2) + 4

        ''' same support for triplet '''
        if np.random.randint(0, 100) < self.same_rate:  # is same, type 1
            fake = ts_r if np.random.randint(0, 100) > 50 else st_r
            s_img, t_img, r_img = fake, fake, fake
            type_code[0] = 1

        return t_img, s_img, r_img, type_code

    def __getitem__(self, index):
        """

        :param index:
        :return: dict
        """

        ''' 0: s_v,     t_v     | None
            1: t_v,     t_v     | t_v
            ----------------------------
            2: s_r,     t_r     | st_r
            3: t_r,     s_r     | ts_r
            4: st_r,    ts_r    | s_r
            5: ts_r,    st_r    | t_r
        '''
        type_code = torch.zeros(1).int()

        '''  | v v v t t | v v v t t | ...
        index: 0 1 2 3 4   5 6 7 8 9
        idx_1: 0 1 2       6 7 8
        idx_2:       0 1         2 3
        '''
        # if index % (self.round_1 + self.round_2) < self.round_1:
        #     t_img, s_img, r_img, type_code = self._getitem_vanilla(index, type_code)
        # else:
        #     t_img, s_img, r_img, type_code = self._getitem_triplet(index, type_code)
        if index < self.len_vanilla:
            t_img, s_img, r_img, type_code = self._getitem_vanilla(index, type_code)
        else:
            t_img, s_img, r_img, type_code = self._getitem_triplet(index - self.len_vanilla, type_code)

        """ Normalize """
        f_mask = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
        if self.transform is not None:
            t_img = self.transform(t_img)
            s_img = self.transform(s_img)
            r_img = self.transform(r_img)
            f_mask = transforms.ToTensor()(f_mask)

        is_same = torch.zeros_like(type_code, dtype=torch.uint8)
        is_same[type_code == 1] = 1
        return {
            "target_image": t_img,
            "source_image": s_img,
            "refer_image": r_img,
            "target_mask": f_mask,
            "type_code": type_code,
            "same": is_same,
        }

    def __len__(self):
        return self.len_vanilla + self.len_triplet


class TripletOriginalDataset(Dataset):
    def __init__(self,
                 triplet_pickle: str = "/gavin/datasets/triplet.pickle",
                 image_size: int = 256,
                 transform=transforms.Compose([
                     transforms.ToTensor(),
                     transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                          std=(0.5, 0.5, 0.5)),
                 ]),
                 triplet_ratio: int = 100,
                 use_real: bool = False,
                 ):
        super(TripletOriginalDataset, self).__init__()

        self.use_real = use_real  # real means type_code = 2 or 3
        self.triplet_count_per_folder = 4 if use_real else 2

        self._load_triplet(triplet_pickle, triplet_ratio)

        """ Other params """
        self.image_size = image_size
        self.transform = transform

        print("Setup TripletTrainDataset finished (Triplet:%d)" % (
            self.__len__()
        ))

    def _load_triplet(self, triplet_pickle: str, triplet_ratio: int = 100):
        """ (2) Triplet Rotational Group images (source, target, st, ts) """
        self.trip_filename = {'s': 'source.jpg', 't': 'target.jpg',
                              'st': 'output_st.jpg', 'ts': 'output_ts.jpg'}
        with open(triplet_pickle, "rb") as handle:
            triplet_list = pickle.load(handle)
        ''' Format, list:
            ['xxx/triplet/00000000_00008210',
             'xxx/triplet/00000044_00008316',]
        '''
        triplet_used = int(len(triplet_list) * triplet_ratio / 100)
        self.triplet_folders = triplet_list[:triplet_used]
        self.len_triplet = len(self.triplet_folders) * self.triplet_count_per_folder
        print('Triplet dataset loaded from %s, folders %d, used %d, images_per_folder: %d' % (
            triplet_pickle, len(triplet_list), len(self.triplet_folders), self.triplet_count_per_folder))

    def _getitem_triplet(self, index, type_code: torch.Tensor):
        """ (2) Triplet """
        idx_2 = index

        # each 4 or 2 groups share the same folder
        folder = self.triplet_folders[idx_2 // self.triplet_count_per_folder]
        s_r = np.array(Image.open(os.path.join(folder, self.trip_filename['s'])).convert("RGB").
                       resize((self.image_size, self.image_size))).astype(np.uint8)
        t_r = np.array(Image.open(os.path.join(folder, self.trip_filename['t'])).convert("RGB").
                       resize((self.image_size, self.image_size))).astype(np.uint8)
        st_r = np.array(Image.open(os.path.join(folder, self.trip_filename['st'])).convert("RGB").
                        resize((self.image_size, self.image_size))).astype(np.uint8)
        ts_r = np.array(Image.open(os.path.join(folder, self.trip_filename['ts'])).convert("RGB").
                        resize((self.image_size, self.image_size))).astype(np.uint8)

        if self.triplet_count_per_folder == 4:
            ''' (1) 4 cycle triplets per folder (type:2,3,4,5) '''
            if idx_2 % 4 == 0:  # type 2
                s_img, t_img, r_img = s_r, t_r, st_r
            elif idx_2 % 4 == 1:  # type 3
                s_img, t_img, r_img = t_r, s_r, ts_r
            elif idx_2 % 4 == 2:  # type 4
                s_img, t_img, r_img = st_r, ts_r, s_r
            else:  # type 5
                s_img, t_img, r_img = ts_r, st_r, t_r
            type_code[0] = (idx_2 % 4) + 2
        else:
            ''' (2) 2 cycle triplets per folder (type:4,5) '''
            if idx_2 % 2 == 0:  # type 4
                s_img, t_img, r_img = st_r, ts_r, s_r
            else:  # type 5
                s_img, t_img, r_img = ts_r, st_r, t_r
            type_code[0] = (idx_2 % 2) + 4

        return t_img, s_img, r_img, type_code

    def __getitem__(self, index):
        """

        :param index:
        :return: dict
        """

        ''' 0: s_v,     t_v     | None
            1: t_v,     t_v     | t_v
            ----------------------------
            2: s_r,     t_r     | st_r
            3: t_r,     s_r     | ts_r
            4: st_r,    ts_r    | s_r
            5: ts_r,    st_r    | t_r
        '''
        type_code = torch.zeros(1).int()

        t_img, s_img, r_img, type_code = self._getitem_triplet(index, type_code)

        """ Normalize """
        f_mask = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
        if self.transform is not None:
            t_img = self.transform(t_img)
            s_img = self.transform(s_img)
            r_img = self.transform(r_img)
            f_mask = transforms.ToTensor()(f_mask)

        is_same = torch.zeros_like(type_code, dtype=torch.uint8)
        is_same[type_code == 1] = 1
        return {
            "target_image": t_img,
            "source_image": s_img,
            "refer_image": r_img,
            "target_mask": f_mask,
            "type_code": type_code,
            "same": is_same,
        }

    def __len__(self):
        return self.len_triplet


class VanillaAndTripletDataset(Dataset):
    def __init__(self,
                 vanilla_pickle: str = "/gavin/datasets/original/image_512_quality.pickle",
                 triplet_pickle: str = "/gavin/datasets/triplet.pickle",
                 same_rate: int = 20,
                 image_size: int = 256,
                 transform=transforms.Compose([
                     transforms.ToTensor(),
                     transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                          std=(0.5, 0.5, 0.5)),
                 ]),
                 top_k: int = 1500000,
                 triplet_ratio: int = 100,
                 use_real: bool = False,
                 vanilla_simswap_mode: bool = False,
                 triplet_rate_of_vanilla: float = 0.2,
                 ):
        super(VanillaAndTripletDataset, self).__init__()

        """ simswap_mode: if True, will make source and target from a same folder (id)
        """
        self.vanilla_dataset = BatchTrainDataset(
            img_root=vanilla_pickle,
            same_rate=same_rate,  # not works when simswap_mode=True
            image_size=image_size,
            transform=transform,
            top_k=top_k,
            simswap_mode=vanilla_simswap_mode
        )
        self.len_vanilla = self.vanilla_dataset.__len__()
        self.triplet_dataset = TripletOriginalDataset(
            triplet_pickle=triplet_pickle,
            image_size=image_size,
            transform=transform,
            triplet_ratio=triplet_ratio,
            use_real=use_real,
        )
        self.len_triplet_all = self.triplet_dataset.__len__()
        self.len_triplet_epoch = int(self.len_vanilla * triplet_rate_of_vanilla)

        print("Setup VanillaAndTripletDataset finished (Vanilla:%d, Triplet All:%d, Triplet in Epoch:%d)" % (
            self.len_vanilla, self.len_triplet_all, self.len_triplet_epoch
        ))

    def __getitem__(self, index):
        vanilla_item = self.vanilla_dataset.__getitem__(index)
        vanilla_t_img = vanilla_item["target_image"]
        vanilla_s_img = vanilla_item["source_image"]

        triplet_item = self.triplet_dataset.__getitem__(np.random.randint(self.len_triplet_all))
        triplet_t_img = triplet_item["target_image"]
        triplet_s_img = triplet_item["source_image"]
        triplet_r_img = triplet_item["refer_image"]

        type_code = triplet_item["type_code"]
        is_same = triplet_item["same"]
        f_mask = triplet_item["target_mask"]

        return {
            "target_image": vanilla_t_img,
            "source_image": vanilla_s_img,
            "t_target_image": triplet_t_img,
            "t_source_image": triplet_s_img,
            "t_refer_image": triplet_r_img,
            "target_mask": f_mask,
            "type_code": type_code,
            "same": is_same,
        }

    def __len__(self):
        return self.len_vanilla + self.len_triplet_epoch


if __name__ == '__main__':
    import os
    from torch.utils.data.dataloader import DataLoader
    from PIL import Image

    # train_set = TripletTrainDataset(
    #     triplet_pickle='/gavin/datasets/triplet_lia_0_600000.pickle',
    # )
    # train_set = BatchTrainDataset(
    #     image_size=256,
    #     ffhq_mode=False,
    #     top_k=1500000,
    #     simswap_mode=True
    # )
    # train_set = TripletOriginalDataset(
    #     triplet_pickle='/gavin/datasets/triplet_lia_0_600000.pickle',
    # )
    train_set = VanillaAndTripletDataset(
        triplet_pickle='/gavin/datasets/triplet_lia_0_600000.pickle',
        triplet_ratio=100,
        same_rate=20,
        vanilla_simswap_mode=False,
    )
    train_set.__getitem__(train_set.__len__() - 1)
    batch_size = 16
    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              num_workers=32,
                              drop_last=True,
                              shuffle=False,
                              )

    demo_folder = './demo'
    os.system('rm -rf %s' % demo_folder)
    os.mkdir(demo_folder)

    def tensor_to_arr(tensor):
        arr = ((tensor + 1.) * 127.5).permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
        return arr

    for batch_idx, batch in enumerate(train_loader):
        i_t = batch["target_image"]
        i_s = batch["source_image"]
        tri_i_t = batch["t_target_image"]
        tri_i_s = batch["t_source_image"]
        tri_i_r = batch["t_refer_image"]
        t_code = batch["type_code"]
        # i_r = i_t
        # t_code = torch.zeros(i_s.shape[0])

        if batch_idx == 0 or batch_idx == 2 or batch_idx == 20 or batch_idx == 23:
            arr_t = tensor_to_arr(i_t)
            arr_s = tensor_to_arr(i_s)
            arr_tri_t = tensor_to_arr(tri_i_t)
            arr_tri_s = tensor_to_arr(tri_i_s)
            arr_tri_r = tensor_to_arr(tri_i_r)
            for b in range(batch_size):
                name_t = '{}_{}_{}_t.jpg'.format(batch_idx, b, int(t_code[b]))
                name_s = '{}_{}_{}_s.jpg'.format(batch_idx, b, int(t_code[b]))
                name_tri_t = '{}_{}_{}_tri_t.jpg'.format(batch_idx, b, int(t_code[b]))
                name_tri_s = '{}_{}_{}_tri_s.jpg'.format(batch_idx, b, int(t_code[b]))
                name_tri_r = '{}_{}_{}_tri_r.jpg'.format(batch_idx, b, int(t_code[b]))

                img_t = Image.fromarray(arr_t[b])
                img_s = Image.fromarray(arr_s[b])
                img_tri_t = Image.fromarray(arr_tri_t[b])
                img_tri_s = Image.fromarray(arr_tri_s[b])
                img_tri_r = Image.fromarray(arr_tri_r[b])

                img_t.save(os.path.join(demo_folder, name_t))
                img_s.save(os.path.join(demo_folder, name_s))
                img_tri_t.save(os.path.join(demo_folder, name_tri_t))
                img_tri_s.save(os.path.join(demo_folder, name_tri_s))
                img_tri_r.save(os.path.join(demo_folder, name_tri_r))

        if batch_idx >= 30:
            print('Only test 30 batches.')
            exit(0)
