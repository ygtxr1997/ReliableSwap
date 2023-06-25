import os

from PIL import Image
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms

from inference.alignment import norm_crop, norm_crop_with_M, paste_back
from inference.utils import save, get_5_from_98, get_detector, get_lmk
from inference.PIPNet.lib.tools import get_lmk_model, demo_image

from celeba.augment.rand_occ import NoneOcc
from celeba.augment.rand_occ import RandomRect, RandomEllipse, RandomConnectedPolygon
from celeba.augment.rand_occ import RandomGlasses, RandomScarf, RandomRealObject
from celeba.augment.rand_occ import RealOcc


class CelebADataset(Dataset):
    def __init__(self,
                 celeba_folder: str = '/gavin/datasets/celeba/img_align_celeba/',
                 split_txt: str = '/gavin/datasets/celeba/list_eval_partition.txt',
                 anno_txt: str = '/gavin/datasets/celeba/list_attr_celeba.txt',
                 split_type: str = 'train',
                 ):
        super(CelebADataset, self).__init__()
        self._prepare(celeba_folder, split_txt, anno_txt, split_type)

        """ The other 6 types of occlusion (excluding mask)
        1. No occlusion
            - NoneOcc
        2. Geometric shapes
            - Rectangle, Ellipse, Connected Polygon
        3. Real-life objects
            - Glasses, Scarf, RealObject
        """
        self.trans_occ = (
            # RandomRect(),
            # RandomEllipse(),
            # RandomConnectedPolygon(),
            RandomGlasses('./augment/occluder/glasses_crop'),
            RandomGlasses('./augment/occluder/eleglasses_crop'),
            RandomScarf('./augment/occluder/scarf_crop'),
            RandomRealObject('./augment/occluder/object_train'),
            # RealOcc(occ_type='rand'),
            RealOcc(occ_type='hand'),
            RealOcc(occ_type='coco'),
        )

    def __getitem__(self, index):
        img_name = self.img_list[index]
        img_path = os.path.join(self.celeba_folder, img_name)
        img = Image.open(img_path, 'r')

        anno = self.anno_dict[img_name][self.anno_idx['Eyeglasses']]
        anno = self._anno_str_to_int(anno)

        # add occlusion
        if np.random.randint(0, 10) > 8 and anno == 0:
            add_occ = self.trans_occ[np.random.randint(0, len(self.trans_occ))]
            img, msk = add_occ(img)
            anno = 1

        # to tensor
        img = self.trans(img)
        anno_tensor = torch.LongTensor([anno])
        return img, anno_tensor

    def __len__(self):
        return len(self.img_list)

    def _prepare(self, celeba_folder, split_txt, anno_txt, split_type):
        assert split_type in ('train', 'val', 'test', 'all')
        self.celeba_folder = celeba_folder

        ''' loading attributes '''
        self.anno_idx = {}  # {key:str, value:int}
        self.anno_dict = {}  # {key:str, value:list[]}
        with open(anno_txt, 'r') as f:
            annos = f.readlines()
        idx = 0
        for line in annos:
            if idx == 0:
                print('total len: %s' % line[:-1])
            elif idx == 1:
                line = line.split('\n')[0]
                line = line.split(' ')
                for k, anno in enumerate(line):
                    self.anno_idx[anno] = k  # 'Eyeglasses':15
            else:
                name, anno = self._get_name_anno(line)
                self.anno_dict[name] = anno
            idx += 1

        ''' loading split '''
        self.train_list = []
        self.val_list = []
        self.test_list = []
        with open(split_txt, 'r') as f:
            split = f.readlines()
        idx = 0
        for line in split:
            name, tve = self._get_name_split(line)
            if tve == 0:  # train
                self.train_list.append(name)
            elif tve == 1:  # val
                self.val_list.append(name)
            elif tve == 2:  # eval
                self.test_list.append(name)
            else:
                raise ValueError('Error type of image. %s' % line)
            idx += 1
        print('split finished. train=%d, val=%d, test=%d' % (len(self.train_list),
                                                             len(self.val_list),
                                                             len(self.test_list)))

        ''' choose tve type '''
        if split_type == 'train':
            self.img_list = self.train_list
        elif split_type == 'val':
            self.img_list = self.val_list
        elif split_type == 'test':
            self.img_list = self.test_list
        else:
            self.img_list = self.train_list + self.val_list + self.test_list
        print('dataset type: %s' % split_type)

        ''' transform '''
        self.trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5]),
        ])

    @staticmethod
    def _get_name_anno(line: str):
        line = line.split('\n')[0]
        words = line.split(' ')
        ret_words = []
        for word in words[1:]:
            if word == '':
                continue
            ret_words.append(word)
        return words[0], ret_words

    @staticmethod
    def _get_name_split(line: str):
        line = line.split('\n')[0]
        name, tve = line.split(' ')
        return name, int(tve)

    @staticmethod
    def _anno_str_to_int(anno_str: str):
        if anno_str == '-1':
            return 0
        elif anno_str == '1':
            return 1
        else:
            raise ValueError('Annotation type not supported. anno=%s' % anno_str)


class CelebAAlignCrop(CelebADataset):
    def __init__(self,
                 celeba_folder: str = '/gavin/datasets/celeba/img_align_celeba/',
                 split_txt: str = '/gavin/datasets/celeba/list_eval_partition.txt',
                 anno_txt: str = '/gavin/datasets/celeba/list_attr_celeba.txt',
                 split_type: str = 'all',
                 ):
        super(CelebAAlignCrop, self).__init__(split_type='all')

        ''' face alignment '''
        self.net, self.detector = get_lmk_model()
        self.net.eval()
        print('alignment model loaded')

        ''' check face '''
        from tddfa_v2.FaceBoxes import FaceBoxes
        self.face_boxes = FaceBoxes()
        print('detection box model loaded')

        self.fail_trans = transforms.Compose([
            transforms.CenterCrop(256),
        ])

    def __getitem__(self, index):
        img_name = self.img_list[index]
        img_path = os.path.join(self.celeba_folder, img_name)

        img, has_lmk_box = self._check_lmk_box(index, img_path)

        if not has_lmk_box:
            img = self.fail_trans(Image.fromarray(img))
        img = self.trans(img)

        return {
            "ori_image": img,
            "img_name": img_name,
            "has_lmk_box": has_lmk_box,
        }

    def _check_lmk_box(self, img_idx, img_path):
        full_img = np.array(Image.open(img_path).convert("RGB")).astype(np.uint8)

        ''' face alignment and check landmarks '''
        lmks = demo_image(full_img, self.net, self.detector)
        if len(lmks) > 0:
            lmk = get_5_from_98(lmks[0])
            cropped_img = norm_crop(full_img, lmk, 256, mode='ffhq', borderValue=0.0)
        else:
            print('Landmarks checking failed @ (id=%d)' % img_idx)
            return full_img, False

        ''' check detection boxes '''
        boxes = self.face_boxes(cropped_img)
        if len(boxes) > 0:
            return cropped_img, True
        else:
            print('Detection boxes checking failed @ (id=%d)' % img_idx)
            return full_img, False


def generate_aligned(aligned_folder: str = '/gavin/datasets/celeba/ffhq_aligned'):
    print('Start align and crop')
    torch.multiprocessing.set_start_method('spawn')
    if os.path.exists(aligned_folder):
        os.system('rm -rf %s' % aligned_folder)
    os.mkdir(aligned_folder)

    trainset = CelebAAlignCrop(split_type='all')
    batch_size = 1
    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              num_workers=1,
                              drop_last=False,
                              shuffle=False,
                              )

    def tensor_to_arr(tensor):
        arr = ((tensor + 1.) * 127.5).permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
        return arr

    img_cnt = 0
    for batch in tqdm(train_loader):
        img = batch["ori_image"]
        has_lmk_box = batch["has_lmk_box"]
        img_name = batch["img_name"]

        arr = tensor_to_arr(img)
        for b in range(batch_size):
            img_cnt += 1
            name_t = img_name[b]
            # if not has_lmk_box[b]:
            #     print('skip image %s: %d' % (name_t, img_cnt - 1))
            img_t = Image.fromarray(arr[b])
            img_t.save(os.path.join(aligned_folder, name_t))


if __name__ == '__main__':
    # trainset = CelebADataset(split_type='all')
    # for idx in range(10):
    #     single = trainset.__getitem__(idx)
    #     print(type(single[0]), type(single[1]))
    #     print(single[0].shape, single[1].shape)

    generate_aligned()

