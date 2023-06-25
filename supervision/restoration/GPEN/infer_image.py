import os
import cv2
import numpy as np
from PIL import Image
import glob

import torch
import tqdm
import shutil
import argparse
from supervision.restoration.GPEN.face_enhancement import FaceEnhancement

make_abs_path = lambda fn: os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), fn))


class GPENImageInfer(object):
    def __init__(self):
        super(GPENImageInfer, self).__init__()

        model = {
            "name": "GPEN-BFR-512",
            "in_size": 512,
            "out_size": 512,
            "channel_multiplier": 2,
            "narrow": 1,
        }
        faceenhancer = FaceEnhancement(
            base_dir=make_abs_path('./'),
            use_sr=True,
            in_size=model["in_size"],
            out_size=model["out_size"],
            model=model["name"],
            channel_multiplier=model["channel_multiplier"],
            narrow=model["narrow"],
        )
        self.faceenhancer = faceenhancer

    def image_infer(self, in_img: np.ndarray):
        """

        :param in_img: np.ndarray, (H,W,BGR), in [0,255]
        :return: out_img: np.ndarray, (H,W,BGR), in [0,255]
        """
        h, w, _ = in_img.shape
        out_img, orig_faces, enhanced_faces = self.faceenhancer.process(in_img)
        out_img = cv2.resize(out_img, (w, h))
        return out_img

    def ndarray_infer(self, in_ndarray: np.ndarray,
                      save_folder: str = 'demo_images/out/',
                      save_name: str = 'reen.png',
                      ):
        """

        :param in_ndarray: np.ndarray, (N,H,W,BGR), in [0,255]
        :param save_folder: not used
        :param save_name: not used
        :return: out_ndarray: np.ndarray, (N,H,W,BGR), in [0,255]
        """
        B, H, W, C = in_ndarray.shape

        out_ndarray = np.zeros_like(in_ndarray, dtype=np.uint8)  # (N,H,W,BGR)
        for b_idx in range(B):
            single_img = in_ndarray[b_idx]
            out_img = self.image_infer(single_img)  # (H,W,BGR), in [0,255]
            out_ndarray[b_idx] = out_img
        return out_ndarray

    def batch_infer(self, in_batch: torch.Tensor,
                          save_folder: str = 'demo_images/out/',
                          save_name: str = 'reen.png',
                          save_batch_idx: int = 0,
                          ):
        """

        :param in_batch: (N,RGB,H,W), in [-1,1]
        :return: out_batch: (N,RGB,H,W), in [-1,1]
        """
        B, C, H, W = in_batch.shape

        in_batch = ((in_batch + 1.) * 127.5).permute(0, 2, 3, 1)
        in_batch = in_batch.cpu().numpy().astype(np.uint8)  # (N,H,W,RGB), in [0,255]
        in_batch = in_batch[:, :, :, ::-1]  # (N,H,W,BGR)

        out_batch = np.zeros_like(in_batch, dtype=np.uint8)  # (N,H,W,BGR)
        for b_idx in range(B):
            single_img = in_batch[b_idx]
            out_img = self.image_infer(single_img)  # (H,W,BGR), in [0,255]
            out_batch[b_idx] = out_img[:, :, ::-1]
            if save_batch_idx is not None and b_idx == save_batch_idx:
                cv2.imwrite(os.path.join(save_folder, save_name), out_img)
        out_batch = torch.FloatTensor(out_batch).cuda()
        out_batch = out_batch / 127.5 - 1.  # (N,H,W,RGB)
        out_batch = out_batch.permute(0, 3, 1, 2)  # (N,RGB,H,W)
        out_batch = out_batch.clamp(-1, 1)

        return out_batch


if __name__ == '__main__':
    gpen = GPENImageInfer()

    in_folder = 'examples/imgs/'
    img_list = os.listdir(in_folder)

    for img_name in img_list:
        if 'gpen' in img_name:
            continue

        in_path = os.path.join(in_folder, img_name)
        out_path = in_path.replace('.png', '_gpen.png')
        out_path = in_path.replace('.jpg', '_gpen.jpg')

        im = cv2.imread(in_path, cv2.IMREAD_COLOR)  # BGR
        img = gpen.image_infer(im)
        cv2.imwrite(out_path, img)
