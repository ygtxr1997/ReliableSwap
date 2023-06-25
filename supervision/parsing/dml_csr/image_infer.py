from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import cv2
import numpy as np
import os
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

from inplace_abn import InPlaceABN
from parsing.dml_csr.networks import dml_csr
import parsing.dml_csr.utils

torch.multiprocessing.set_start_method("spawn", force=True)


class DMLImageInfer(object):
    def __init__(self,
                 crop_size: tuple = (256, 256),
                 transform=None,
                 num_classes: int = 19,
                 weight_path: str = 'weights/DML_CSR/dml_csr_celebA.pth',
                 ):
        self.crop_size = crop_size

        self.transform = transform
        if self.transform is None:
            normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                             std=[0.5, 0.5, 0.5])
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
        self.mean = torch.Tensor([0.5, 0.5, 0.5]).cuda()
        self.std = torch.Tensor([0.5, 0.5, 0.5]).cuda()

        self.num_classes = num_classes
        self.weight_path = weight_path

        self.model = None
        self._load_model()

    def _load_model(self, ):
        model = dml_csr.DML_CSR(self.num_classes, InPlaceABN, False)
        restore_from = self.weight_path
        print('[loading DML_CSR model from]:', restore_from)
        state_dict = torch.load(restore_from, map_location='cuda')
        model.load_state_dict(state_dict)
        model.cuda()
        model.eval()
        self.model = model

    def normalize_input(self,
                        input_ndarray: np.ndarray,
                        ) -> torch.Tensor:
        x = torch.from_numpy(input_ndarray).permute(0, 3, 1, 2).cuda()
        x = x / 255.
        x = x - self.mean[None, :, None, None].expand_as(x)
        x = x / self.std[None, :, None, None].expand_as(x)
        return x

    def parsing(self,
                input_tensor: torch.Tensor,
                save_folder: str = 'demo_images/',
                save_name: str = 'parsing.png',
                save_batch_idx: int = 0,
                ) -> np.ndarray:
        """
        Forward function.
        The semantic map when num_classes=19:
        background  - 0
        skin        - 1
        nose        - 2
        eyes        - 5, 4
        eyebrows    - 7, 6
        ears        - 9, 8
        lips        - 11, 12
        tooth       - 10
        hairs       - 13
        neck        - 17
        clothes     - 18

        :param input_tensor: torch.Tensor, (N,C,H,W), in [-1,1]
        :param save_folder:
        :param save_name:
        :return: np.ndarray, np.uint8, (N,H,W), in [0,#seg]
        """
        if self.model is None:
            self._load_model()
        results = self.model(input_tensor)

        interp = torch.nn.Upsample(size=self.crop_size,
                                   mode='bilinear',
                                   align_corners=True)
        parsing = interp(results).data.cpu().numpy()
        parsing = parsing.transpose(0, 2, 3, 1)  # NCHW NHWC

        saved = np.asarray(np.argmax(parsing, axis=3))  # min:0, max:num_classes-1
        if save_folder is not '' and save_batch_idx is not None:
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            img = saved[save_batch_idx] * 10
            cv2.imwrite(os.path.join(save_folder, save_name), img)
        return saved  # NHWC

    def calc_regions(self,
                     seg_s: np.array,
                     seg_ts: np.array,
                     save_folder: str,
                     save_name: str,
                     save_batch_idx: int,
                     ):
        """
        regions = seg_ts - seg_s

        :param seg_s: (N,H,W), in [0,#seg]
        :param seg_ts: (N,H,W), in [0,#seg]
        :param save_folder:
        :param save_name:
        :param save_batch_idx:
        :return: regions, (N,H,W), in {2,3,6,7}
        """
        seg_s[seg_s == 0] = 19
        seg_s[seg_s == 8] = 19
        seg_s[seg_s == 9] = 19
        seg_ts[seg_ts == 0] = 19
        seg_ts[seg_ts == 8] = 19
        seg_ts[seg_ts == 9] = 19

        seg_s[seg_s <= 12] = 1
        seg_s[seg_s != 1] = 2
        seg_ts[seg_ts <= 12] = 4
        seg_ts[seg_ts != 4] = 8

        regions = seg_ts - seg_s
        regions_saved = np.zeros((seg_s.shape[0],
                                  self.crop_size[0],
                                  self.crop_size[1],
                                  3), dtype=np.uint8)
        regions_saved[regions == 2] = (255, 0, 0)  # blue
        regions_saved[regions == 3] = (0, 255, 255)  # yellow
        regions_saved[regions == 6] = (112, 112, 112)  # gray
        regions_saved[regions == 7] = (0, 255, 0)  # green

        if save_folder != '' and save_name != '' and save_batch_idx is not None:
            # np.save(os.path.join(save_folder, save_name), regions)
            cv2.imwrite(os.path.join(save_folder, save_name),
                        regions_saved[save_batch_idx])

        drop = np.ones(seg_s.shape, dtype=np.uint8)
        drop[regions == 2] = 0
        drop[regions == 7] = 0
        drop *= 255  # 0:drop, 255:keep
        return drop, regions


def main():
    # h, w = map(int, args.input_size.split(','))
    # input_size = (h, w)

    cudnn.benchmark = True
    cudnn.enabled = True

    dml_infer = DMLImageInfer(
        crop_size=(256, 256),
    )

    from multi_band import multi_band_blending

    ''' 1. source, target '''
    seg_reen, im_reen = dml_infer.parsing(input_path='infer_images/in/reen_st.png',
                                          save_path='infer_images/out/reen_st_seg.jpg')
    seg_targ, im_targ = dml_infer.parsing(input_path='infer_images/in/target.jpg',
                                          save_path='infer_images/out/target_seg.jpg')
    multi_band_blending(im_targ, im_reen,
                        mask_bbox=seg_reen,
                        mask_bbox_2=seg_reen,
                        save_path='infer_images/out/mb_st.jpg')
    seg_band, im_band = dml_infer.parsing(input_path='infer_images/out/mb_st.jpg',
                                          save_path='infer_images/out/mb_st_seg.jpg')
    drop = dml_infer.calc_regions(seg_reen, seg_targ,
                           save_path='infer_images/out/regions_st.jpg')
    # cv2.imwrite('infer_images/out/drop_st.jpg', drop)

    ''' 2. target, source '''
    seg_reen, im_reen = dml_infer.parsing(input_path='infer_images/in/reen_ts.png',
                                          save_path='infer_images/out/reen_ts_seg.jpg')
    seg_targ, im_targ = dml_infer.parsing(input_path='infer_images/in/source.jpg',
                                          save_path='infer_images/out/source_seg.jpg')
    multi_band_blending(im_targ, im_reen,
                        mask_bbox=seg_reen,
                        mask_bbox_2=seg_reen,
                        save_path='infer_images/out/mb_ts.jpg')
    seg_band, im_band = dml_infer.parsing(input_path='infer_images/out/mb_ts.jpg',
                                          save_path='infer_images/out/mb_ts_seg.jpg')
    drop = dml_infer.calc_regions(seg_reen, seg_targ,
                           save_path='infer_images/out/regions_ts.jpg')
    # cv2.imwrite('infer_images/out/drop_ts.jpg', drop)

if __name__== '__main__':
    main()