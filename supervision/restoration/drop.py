import os.path

import cv2
import numpy as np


def drop_single(mb_img: np.ndarray,
                regions: np.ndarray,
                save_path: str = ''
                ) -> (np.ndarray, np.ndarray):
    """
    :param mb_img: multi-band result, (H,W,BGR), [0,255]
    :param regions: colored region mask (yellow, green, blue, gray), (H,W,BGR), in {1,2,3,7}
    :param save_path:
    :return np.ndarray, (H,W,BGR), [0,255]; np.ndarray, (H,W,BGR), [0,255]
    """
    H, W = regions.shape[0], regions.shape[1]
    mb_img = cv2.resize(mb_img, (regions.shape[0], regions.shape[1]))
    drop_array = np.zeros((regions.shape[0], regions.shape[1], 1), dtype=np.uint8)
    drop_array[regions == 2] = 255  # blue, to be inpainted with background
    # drop_array[regions == 7] = 255  # green, to be expanded with face

    ''' Expand regions '''
    import torch.nn as nn
    import torch
    x = torch.tensor(drop_array, dtype=torch.float32)

    x = x.transpose(0, 1)
    x = x.transpose(0, 2)
    x = x[None, :, :, :]
    expand = nn.Conv2d(1, 1, kernel_size=5, stride=1, padding=2)
    expand.weight.data = torch.ones_like(expand.weight.data)
    with torch.no_grad():
        x = expand(x)
    x = x[0]
    x = x.transpose(0, 2)
    x = x.transpose(0, 1)

    x[x > 0.5] = 1.
    x[x != 1] = 0.
    drop_array = np.repeat(x.numpy(), 3, 2).astype(np.uint8)
    mb_img[drop_array == 1] = 255

    drop_array = np.repeat(x.numpy(), 3, 2).astype(np.uint8) * 255

    if save_path != '':
        dropped_path = save_path.replace('/drop_mask', '/dropped_')
        cv2.imwrite(dropped_path, mb_img)
        cv2.imwrite(save_path, drop_array)

    return mb_img, drop_array


def drop_batch(mb_batch: np.ndarray,
               regions_batch: np.ndarray,
               save_folder: str,
               save_name: str,
               save_batch_idx: int,
               ) -> (np.ndarray, np.ndarray):
    """

    :param mb_batch:
    :param regions_batch:
    :param save_folder:
    :param save_name:
    :param save_batch_idx:
    :return: np.ndarray, (N,H,W,BGR), in [0,255]
    """
    assert mb_batch.ndim == 4, 'batch input should be (N,H,W,BGR)'
    B, H, W, C = mb_batch.shape

    dropped_imgs = np.zeros_like(mb_batch, dtype=np.uint8)
    drop_masks = np.zeros_like(mb_batch, dtype=np.uint8)
    for b_idx in range(B):
        save_path = os.path.join(save_folder, save_name) if b_idx == save_batch_idx else ''
        im_dropped, mask = drop_single(mb_batch[b_idx],
                                       regions_batch[b_idx],
                                       save_path)
        dropped_imgs[b_idx] = im_dropped
        drop_masks[b_idx] = mask
    return dropped_imgs, drop_masks
