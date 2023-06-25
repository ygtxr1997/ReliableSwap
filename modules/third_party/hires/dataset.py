import torch.utils.data as data
import torch
from PIL import Image
import os
import os.path
import numpy as np
from numpy.random import randint
from torchvision import transforms, utils
import cv2
import torchvision.transforms.functional as tF





class SourceICTargetICLM(data.Dataset):
    def __init__(self, image_path, jpg_tmpl='{}.jpg', png_tmpl='{}.png', npy_tmpl='{}.npy', lmk_tmpl='{}.npy', to_tensor_256=None, to_tensor_1024=None):

        self.image_path = image_path
        self.imgs = self._walkFile(image_path)
        self.jpg_tmpl = jpg_tmpl
        self.png_tmpl = png_tmpl

        self.npy_tmpl = npy_tmpl
        self.lmk_tmpl = lmk_tmpl
        self.to_tensor_256 = to_tensor_256
        self.to_tensor_1024 = to_tensor_1024

    def _load_jpg(self, directory, idx):
        return Image.open(os.path.join(directory, self.jpg_tmpl.format(idx)))

    def _load_png(self, directory, idx):
        return cv2.imread(os.path.join(directory, self.png_tmpl.format(idx)))
    def _load_npy(self, directory, idx):
        return np.load(os.path.join(directory, self.npy_tmpl.format(idx)))

    def _load_lmk(self, directory, idx):
        return np.load(os.path.join(directory, self.lmk_tmpl.format(idx)))

    def _cords_to_map_np(self, cords, img_size=(256,256), sigma=6):
        result = np.zeros(img_size + cords.shape[0:1], dtype='uint8')
        for i, point in enumerate(cords):
            if point[0] == -1 or point[1] == -1:
                continue
            xx, yy = np.meshgrid(np.arange(img_size[1]), np.arange(img_size[0]))
            x = np.exp(-((yy - int(point[0])) ** 2 + (xx - int(point[1])) ** 2) / (2 * sigma ** 2))
            result[..., i] = x
        return result

    def _walkFile(self,file):
        frames = []
        for x in os.listdir(file):
            frames.append(file+"/"+x)
        return frames


    def encode_segmentation_rgb(self, segmentation, no_neck=True):
        parse = segmentation[:,:,0]

        face_part_ids = [1, 6, 7, 4, 5, 3, 2, 11, 12] if no_neck else [1, 6, 7, 4, 5, 3, 2, 11, 12, 17]
        mouth_id = 10
        hair_id = 13


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



    def __getitem__(self, idx):


        s_index = randint(0, len(self.imgs) - 1)
        t_index = randint(0, len(self.imgs) - 1)
        s_img = self._load_jpg(self.image_path,s_index)
        s_code =  self._load_npy(self.image_path.replace('img','latent'),s_index)

        s_lmk = self._load_lmk(self.image_path.replace('img','landmark'),s_index)
        s_map = self._cords_to_map_np(s_lmk)

        t_img = self._load_jpg(self.image_path,t_index)

        t_lmk = self._load_lmk(self.image_path.replace('img','landmark'),t_index)

        t_map = self._cords_to_map_np(t_lmk)


        t_code =  self._load_npy(self.image_path.replace('img','latent'),t_index)


        t_mask = self._load_png(self.image_path.replace('img','mask'),t_index)
        t_mask = self.encode_segmentation_rgb(t_mask)
        t_mask = cv2.resize(t_mask,(1024,1024))
        t_mask = t_mask.transpose((2, 0, 1)).astype(np.float)/255.0
        t_mask = t_mask[0] + t_mask[1]

        t_mask = cv2.dilate(t_mask, np.ones((50,50)), borderType=cv2.BORDER_CONSTANT, borderValue=0)

        s_img = self.to_tensor_256(s_img)
        t_img = self.to_tensor_1024(t_img)

        return s_img,s_code,s_map,s_lmk,t_img,t_code,t_map,t_lmk,t_mask,s_index, t_index

    def __len__(self):
        return len(self.imgs)



