import os.path

import cv2
import numpy as np
import torch

from graphics.color_transfer import skin_color_transfer


def Laplacian_Pyramid_Blending_with_mask(A, B, m, num_levels = 6):
    # assume mask is float32 [0,1]

    # generate Gaussian pyramid for A,B and mask
    GA = A.copy()
    GB = B.copy()
    GM = m.copy()
    gpA = [GA]
    gpB = [GB]
    gpM = [GM]
    for i in range(num_levels):
        GA = cv2.pyrDown(GA)
        GB = cv2.pyrDown(GB)
        GM = cv2.pyrDown(GM)
        gpA.append(np.float32(GA))
        gpB.append(np.float32(GB))
        gpM.append(np.float32(GM))

    # generate Laplacian Pyramids for A,B and masks
    lpA  = [gpA[num_levels-1]] # the bottom of the Lap-pyr holds the last (smallest) Gauss level
    lpB  = [gpB[num_levels-1]]
    gpMr = [gpM[num_levels-1]]
    for i in range(num_levels-1,0,-1):
        # Laplacian: subtarct upscaled version of lower level from current level
        # to get the high frequencies
        LA = np.subtract(gpA[i-1], cv2.pyrUp(gpA[i]))
        LB = np.subtract(gpB[i-1], cv2.pyrUp(gpB[i]))
        lpA.append(LA)
        lpB.append(LB)
        gpMr.append(gpM[i-1]) # also reverse the masks

    # Now blend images according to mask in each level
    LS = []
    for la,lb,gm in zip(lpA,lpB,gpMr):
        ls = la * gm + lb * (1.0 - gm)
        LS.append(ls)

    # now reconstruct
    ls_ = LS[0]
    for i in range(1,num_levels):
        ls_ = cv2.pyrUp(ls_)
        ls_ = cv2.add(ls_, LS[i])

    return ls_


def mask_bbox_process(mask_bbox: np.ndarray,):
    """
         dml_csr_19	MGEditing_11
    背景：0		    0
    皮肤：1		    1
    鼻子：2		    6
    眼睛：5，4	    4, 5
    眉毛：7，6	    2, 3
    耳朵：9，8	    0
    嘴唇：11，12	    7, 9
    牙齿：10		    8
    头发：13		    10
    脖子：17
    上衣：18
    :param mask_bbox: 0-bg, 1-face, 13-hair
    :return: mask_bbox: 0-face, 1-bg, 13-hair
    """
    if mask_bbox.ndim == 2:
        mask_bbox = np.repeat(mask_bbox[:, :, None], 3, 2).astype(np.uint8)
    assert mask_bbox.ndim == 3
    ''' op1. whole face '''
    mask_bbox[mask_bbox == 8] = 20  # ear
    mask_bbox[mask_bbox == 9] = 20  # ear
    mask_bbox[mask_bbox == 0] = 20  # bg
    mask_bbox[mask_bbox <= 12] = 1  # step1: get inner face
    mask_bbox[mask_bbox >= 14] = 0  # step2: get bg

    # # smaller face mask?
    # wh = 254
    # diff = 256 - wh
    # mask_bbox = cv2.resize(mask_bbox, dsize=(wh, wh))
    # mask_bbox = cv2.copyMakeBorder(mask_bbox, diff, diff, diff, diff,
    #                                borderType=cv2.BORDER_CONSTANT,
    #                                value=0)

    # reverse
    mask_bbox[mask_bbox == 1] = 2  # tmp
    mask_bbox[mask_bbox == 0] = 1
    mask_bbox[mask_bbox == 2] = 0

    ''' op2. eyes/nose/mouth'''
    # mask_bbox[mask_bbox == 1] = 20
    # mask_bbox[mask_bbox == 2] = 1
    # mask_bbox[mask_bbox == 4] = 1
    # mask_bbox[mask_bbox == 5] = 1
    # mask_bbox[mask_bbox == 6] = 1
    # mask_bbox[mask_bbox == 7] = 1
    # mask_bbox[mask_bbox == 10] = 1
    # mask_bbox[mask_bbox == 11] = 1
    # mask_bbox[mask_bbox == 12] = 1
    # mask_bbox[mask_bbox != 1] = 0
    #
    # import torch.nn as nn
    # import torch
    # x = torch.tensor(mask_bbox, dtype=torch.float32)
    # x = x.transpose(0, 1)
    # x = x.transpose(0, 2)
    # x = x[None, :, :, :]
    # expand = nn.Conv2d(3, 1, kernel_size=5, stride=1, padding=3)
    # expand.weight.data = torch.ones_like(expand.weight.data)
    # with torch.no_grad():
    #     x = expand(x)
    # x = x[0]
    # x = x.transpose(0, 2)
    # x = x.transpose(0, 1)
    # mask_bbox = np.repeat(x.numpy(), 3, 2).astype(np.uint8)
    # mask_bbox[mask_bbox > 0.5] = 1.
    # mask_bbox[mask_bbox != 1] = 0.
    # mask_bbox = 1 - mask_bbox

    return mask_bbox


def multi_band_blending(full_img: np.ndarray,
                        ori_img: np.ndarray,
                        mask_bbox: np.ndarray,
                        mask_bbox_2: np.ndarray,
                        save_path: str
                        ) -> np.ndarray:
    """
    Multi-band

    :param full_img: target attribute, (H,W,C), in [0,255]
    :param ori_img: source identity, (H,W,C), in [0,255]
    :param mask_bbox: segmentation mask, (H,W), in [0,#seg]
    :param mask_bbox_2: another auxiliary segmentation mask, (H,W), in [0,#seg]
    :param save_path: path
    :return: multi_band image: (H,W,C), in [0,255]
    """
    mask_sharp = 1.
    width, height = full_img.shape[0], full_img.shape[1]

    # 1:from full(target), 0:from ori(source)
    mask_bbox = mask_bbox_process(mask_bbox)  # after process: 0-face, 1-bg, 13-hair
    bbox2 = mask_bbox_process(mask_bbox_2)
    mask_bbox[mask_bbox == 13] = 1  # remove hair from source
    mask_bbox[bbox2 == 13] = 1  # remove hair from target

    full_img, ori_img, full_mask = [cv2.resize(x, (512, 512)) for x in
                                    (full_img, ori_img,
                                     np.float32(mask_sharp * mask_bbox))]
    img = Laplacian_Pyramid_Blending_with_mask(full_img, ori_img, full_mask, 10)

    # img in [0, 255]
    img = np.clip(img, 0, 255)
    img = np.uint8(cv2.resize(img, (width, height)))

    # color transfer
    img = skin_color_transfer(img / 255., full_img / 255., ct_mode='lct')
    img = img.astype(np.uint8)

    if save_path != '':
        assert save_path.find('/mb_'), 'multi_band save_path should include \'/mb_\'! '
        bbox_path = save_path.replace('/mb_', '/bbox_')
        cv2.imwrite(bbox_path, mask_bbox * 255)
        cv2.imwrite(save_path, img)

    return img


def multi_band_blending_batch(full_batch: np.ndarray,
                              ori_batch: np.ndarray,
                              mask_batch: np.ndarray,
                              mask_batch_2: np.ndarray,
                              save_folder: str,
                              save_name: str,
                              save_batch_idx: int,
                              ) -> np.ndarray:
    """

    :param full_batch: (N,H,W,BGR), in [0,255]
    :param ori_batch: (N,H,W,BGR), in [0,255]
    :param mask_batch: (N,H,W), in [0,#seg]
    :param mask_batch_2: (N,H,W), in [0,#seg]
    :param save_folder:
    :param save_name:
    :param save_batch_idx:
    :return: np.ndarray, (N,H,W,BGR), in [0,255]
    """
    assert full_batch.ndim == 4
    assert mask_batch.ndim == 3
    B, H, W, C = full_batch.shape

    mb_batch = np.zeros((B, H, W, C), dtype=np.uint8)
    for b_idx in range(B):
        full_img = full_batch[b_idx]
        ori_img = ori_batch[b_idx]
        mask_bbox = mask_batch[b_idx]
        mask_bbox_2 = mask_batch_2[b_idx]

        mb_path = os.path.join(save_folder, save_name) if b_idx == save_batch_idx else ''
        mb_img = multi_band_blending(full_img, ori_img, mask_bbox, mask_bbox_2,
                                     save_path=mb_path)

        mb_batch[b_idx] = mb_img
    return mb_batch


def main():
    full_img = cv2.imread('infer_images/in/source.jpg')
    ori_img = cv2.imread('infer_images/in/reen_ts.png')
    mask_sharp = 1.

    # mask_bbox = cv2.imread('infer_images/out/regions_st.jpg', cv2.IMREAD_GRAYSCALE)
    # mask_bbox = np.repeat(mask_bbox[:, :, None], 3, 2)
    # # mask_bbox = cv2.resize(mask_bbox, dsize=(248, 248))
    # # mask_bbox = cv2.copyMakeBorder(mask_bbox, 8, 8, 8, 8,
    # #                                borderType=cv2.BORDER_CONSTANT,
    # #                                value=0)
    # mask_bbox[mask_bbox == 226] = 1
    # mask_bbox[mask_bbox != 1] = 0
    # mask_bbox = 1 - mask_bbox

    mask_bbox = cv2.imread('infer_images/out/source_seg.jpg')
    mask_bbox[mask_bbox == 0] = 20
    mask_bbox[mask_bbox <= 12] = 1
    mask_bbox[mask_bbox != 1] = 0
    mask_bbox = 1 - mask_bbox
    # 1:full, 0:ori

    full_img, ori_img, full_mask = [cv2.resize(x, (512, 512)) for x in
                                    (full_img, ori_img,
                                     np.float32(mask_sharp * mask_bbox))]
    # full_img = cv2.convertScaleAbs(ori_img*(1-full_mask) + full_img*full_mask)
    img = Laplacian_Pyramid_Blending_with_mask(full_img, ori_img, full_mask, 10)

    ### img in [0, 255]
    width, height = full_img.shape[0], full_img.shape[1]
    img = np.clip(img, 0, 255)
    img = np.uint8(cv2.resize(img, (width, height)))
    cv2.imwrite('multi_band.jpg', img)

if __name__ == '__main__':
    main()