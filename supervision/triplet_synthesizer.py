import os
import os.path
import time

import cv2
import PIL.Image
import numpy as np

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from supervision.dataset.dataloader import BatchTrainDataset
from supervision.parsing import DMLImageInfer
from supervision.reenactment import PIRenderImageInfer  # deprecated
from supervision.reenactment import LIABatchInfer
from supervision.graphics import multi_band_blending_batch
from supervision.restoration.drop import drop_batch
from supervision.restoration.DeepFill_V2.image_infer import DeepFillV2ImageInfer
from supervision.restoration.GPEN.infer_image import GPENImageInfer

make_abs_path = lambda fn: os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), fn))


def set_random_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def bgr_to_rgb(cvimg: np.ndarray) -> np.ndarray:
    pilimg = cvimg[:, :, :, ::-1]
    return pilimg


def rgb_to_bgr(pilimg: np.ndarray) -> np.ndarray:
    return bgr_to_rgb(pilimg)


class RotaTripSynthesizer(pl.LightningModule):
    def __init__(self,
                 batch_size=12,
                 is_demo=False,
                 save_dataset_folder='/gavin/datasets/triplet_tmp',
                 demo_folder='demo_snapshot0',
                 ):
        super(RotaTripSynthesizer, self).__init__()
        self.batch_size = batch_size

        self.parsing = DMLImageInfer(
            crop_size=(256, 256),
            weight_path=make_abs_path('parsing/dml_csr/weights/DML_CSR/dml_csr_celebA.pth')
        )
        # self.reenact = PIRenderImageInfer(
        #     weights_path='reenactment/Deep3DFaceRecon_pytorch/checkpoints/PIRender/result/face/epoch_00190_iteration_000400000_checkpoint.pt',
        #     cfg_path='reenactment/Deep3DFaceRecon_pytorch/checkpoints/PIRender/config/face_demo.yaml',
        #     checkpoints_dir='reenactment/Deep3DFaceRecon_pytorch/checkpoints/PIRender/Deep3DFaceRecon_pytorch/checkpoints'
        # )  # deprecated
        self.reenact = LIABatchInfer()
        self.enhance = GPENImageInfer()
        self.blending = multi_band_blending_batch
        self.dropping = drop_batch
        self.restoring = DeepFillV2ImageInfer(
            checkpoint_dir=make_abs_path('restoration/DeepFill_V2/weights/release_celeba_hq_256_deepfill_v2'),
            config_path=make_abs_path('restoration/DeepFill_V2/inpaint.yml'),
            batch_size=batch_size,
        )

        """ For demo snapshot """
        self.demo_folder = demo_folder
        if os.path.exists(self.demo_folder) and is_demo:
            print('deleting demo folder: %s' % self.demo_folder)
            os.system('rm -r %s' % self.demo_folder)
        os.makedirs(self.demo_folder, exist_ok=True)
        self.snapshot_folder = ''  # will be changed during batch inference

        """ For full dataset """
        self.dataset_root = save_dataset_folder
        if not os.path.exists(self.dataset_root):
            os.mkdir(self.dataset_root)

        self.is_demo = is_demo
        self.save_batch_idx = None
        if is_demo:
            print('demo mode will not impact dataset_root')
            self.dataset_root = None
            self.save_batch_idx = 0

    def forward(self, x):
        pass

    def _save_input(self,
                    source_tensor: torch.Tensor,
                    target_tensor: torch.Tensor,
                    save_idx: int = 0,
                    ) -> None:
        save_t = ((target_tensor[save_idx] + 1.) * 127.5).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        save_s = ((source_tensor[save_idx] + 1.) * 127.5).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        save_t = PIL.Image.fromarray(save_t)
        save_s = PIL.Image.fromarray(save_s)
        save_t.save(os.path.join(self.snapshot_folder, 'target.jpg'))
        save_s.save(os.path.join(self.snapshot_folder, 'source.jpg'))

    def _step1_reenactment(self,
                           source_tensor: torch.Tensor,
                           target_tensor: torch.Tensor,
                           save_name: str,
                           ) -> torch.Tensor:
        reen = self.reenact.infer_batch(
            source_tensor,
            target_tensor,
            save_folder=self.snapshot_folder,
            save_name=save_name,
            save_batch_idx=self.save_batch_idx,
        )
        reen = self.enhance.batch_infer(
            reen,
            save_folder=self.snapshot_folder,
            save_name=save_name.replace('.jpg', '_gpen.jpg'),
            save_batch_idx=self.save_batch_idx,
        )
        return reen

    def _step2_blending(self,
                        reen: torch.Tensor,
                        targ: torch.Tensor,
                        is_st: bool,
                        ):
        st_str = 'st' if is_st else 'ts'
        im_reen: torch.Tensor = (reen + 1.) * 127.5  # [-1,1] to [0,255.]
        im_targ: torch.Tensor = (targ + 1.) * 127.5  # [-1,1] to [0,255.]

        # np.ndarray, (N,H,W,BGR), in [0,255]
        im_reen: np.ndarray = rgb_to_bgr(im_reen.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8))
        im_targ: np.ndarray = rgb_to_bgr(im_targ.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8))

        # np.ndarray, (N,H,W), in [0,#seg]
        seg_reen = self.parsing.parsing(reen,
                                        save_folder=self.snapshot_folder,
                                        save_name='reen_st_seg.png'.replace('st', st_str),
                                        save_batch_idx=self.save_batch_idx)
        seg_targ = self.parsing.parsing(targ,
                                        save_folder=self.snapshot_folder,
                                        save_name='target_st_seg.png'.replace('st', st_str),
                                        save_batch_idx=self.save_batch_idx)
        # np.ndarray
        im_band = self.blending(full_batch=im_targ,
                                ori_batch=im_reen,
                                mask_batch=seg_reen,
                                mask_batch_2=seg_targ,
                                save_folder=self.snapshot_folder,
                                save_name='mb_st.jpg'.replace('st', st_str),
                                save_batch_idx=self.save_batch_idx)
        seg_band = self.parsing.parsing(self.parsing.normalize_input(im_band),
                                        save_folder=self.snapshot_folder,
                                        save_name='mb_st_seg.png'.replace('st', st_str),
                                        save_batch_idx=self.save_batch_idx)
        # _, np.ndarray, (N,H,W), in {2,3,6,7}
        _, im_regions = self.parsing.calc_regions(seg_reen,
                                                  seg_targ,
                                                  save_folder=self.snapshot_folder,
                                                  save_name='regions_st.png'.replace('st', st_str),
                                                  save_batch_idx=self.save_batch_idx)
        return im_band, im_regions

    def _step3_inpaint(self,
                       im_band: np.ndarray,
                       im_regions: np.ndarray,
                       is_st: bool,
                       ) -> np.ndarray:
        """

        :param im_band: (N,H,W,BGR)
        :param im_regions: (N,H,W,BGR)
        :param is_st:
        :return: (N,H,W,BGR)
        """
        st_str = 'st' if is_st else 'ts'
        im_drop, im_masks = self.dropping(im_band,
                                          im_regions,
                                          save_folder=self.snapshot_folder,
                                          save_name='drop_mask_st.png'.replace('st', st_str),
                                          save_batch_idx=self.save_batch_idx)
        im_restore = self.restoring.infer_batch(im_drop,
                                                im_masks,
                                                save_folder=self.snapshot_folder,
                                                save_name='output_st.png'.replace('st', st_str))
        im_restore = self.enhance.ndarray_infer(im_restore,
                                        save_folder=self.snapshot_folder,
                                        save_name='output_st.jpg'.replace('st', st_str).replace('.jpg', '_gpen.jpg'),
                                        )
        return im_restore

    def _step4_save_to_dataset(self,
                               tensor_s: torch.Tensor,
                               tensor_t: torch.Tensor,
                               output_st: np.ndarray,
                               output_ts: np.ndarray,
                               pair_strs: tuple,
                               ) -> None:
        """

        :param tensor_s: (N,RGB,H,W), in [-1,1]
        :param tensor_t: (N,RGB,H,W), in [-1,1]
        :param output_st: (N,H,W,BGR), in [0,255]
        :param output_ts: (N,H,W,BGR), in [0,255]
        :return: None
        """
        N, C, H, W = tensor_s.shape
        array_s = ((tensor_s + 1.) * 127.5).permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
        array_t = ((tensor_t + 1.) * 127.5).permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
        array_s = array_s[:, :, :, ::-1]  # RGB to BGR
        array_t = array_t[:, :, :, ::-1]

        if self.is_demo:
            print('Demo run will not save to datasets folder.')
            cv2.imwrite(os.path.join(self.snapshot_folder, 'output_st.jpg'),
                        output_st[self.save_batch_idx])
            cv2.imwrite(os.path.join(self.snapshot_folder, 'output_ts.jpg'),
                        output_ts[self.save_batch_idx])
            return

        for idx in range(N):
            save_folder = os.path.join(self.dataset_root, pair_strs[idx])
            if not os.path.exists(save_folder):
                os.mkdir(save_folder)
            cv2.imwrite(os.path.join(save_folder, 'source.jpg'), array_s[idx])
            cv2.imwrite(os.path.join(save_folder, 'target.jpg'), array_t[idx])
            cv2.imwrite(os.path.join(save_folder, 'output_st.jpg'), output_st[idx])
            cv2.imwrite(os.path.join(save_folder, 'output_ts.jpg'), output_ts[idx])

    def test_step(self, batch, batch_idx):
        # Tensor, (N,RGB,H,W), in [-1,1]
        tensor_t: torch.Tensor = batch['target_image']
        tensor_s: torch.Tensor = batch['source_image']
        pair_strs: tuple = tuple(batch['pair_str'])

        snapshot_folder = os.path.join(self.demo_folder, 'demo_%05d' % batch_idx)
        if not os.path.exists(snapshot_folder):
            os.mkdir(snapshot_folder)
        self.snapshot_folder = snapshot_folder

        self._save_input(tensor_s, tensor_t, save_idx=0)

        """ 1. reenactment """
        import time
        time_0 = time.time()
        # Tensor, (N,RGB,H,W), in [-1,1]
        reen_st = self._step1_reenactment(tensor_s, tensor_t, 'reen_st.jpg')
        reen_ts = self._step1_reenactment(tensor_t, tensor_s, 'reen_ts.jpg')

        print('step 1: %d ms' % int((time.time() - time_0) * 1000))
        time_0 = time.time()

        """ 2. multi_band blending """
        im_band_st, im_regions_st = self._step2_blending(reen_st, tensor_t, is_st=True)
        im_band_ts, im_regions_ts = self._step2_blending(reen_ts, tensor_s, is_st=False)

        print('step 2: %d ms' % int((time.time() - time_0) * 1000))
        time_0 = time.time()

        """ 3. drop and inpaint """
        im_restore_st = self._step3_inpaint(im_band_st, im_regions_st, is_st=True)
        im_restore_ts = self._step3_inpaint(im_band_ts, im_regions_ts, is_st=False)

        print('step 3: %d ms' % int((time.time() - time_0) * 1000))
        time_0 = time.time()

        """ 4. save as dataset """
        self._step4_save_to_dataset(tensor_s, tensor_t, im_restore_st, im_restore_ts, pair_strs)

    def configure_optimizers(self):
        pass

    def test_dataloader(self):
        top_k = 1500000 if not self.is_demo else 15000
        image512quality_test = BatchTrainDataset(
            same_rate=0,
            image_size=256,
            top_k=top_k,
        )
        return DataLoader(
            image512quality_test,
            batch_size=self.batch_size,
            num_workers=4,
            drop_last=False,
            shuffle=True,
        )

    def training_step(self, batch, batch_idx):
        return self.test_step(batch, batch_idx)

    def train_dataloader(self):
        return self.test_dataloader()


# from inference.alignment import norm_crop, norm_crop_with_M, paste_back
# from inference.utils import save, get_5_from_98, get_detector, get_lmk
# from inference.PIPNet.lib.tools import get_lmk_model, demo_image
# from PIL import Image
# from torchvision.transforms import transforms
# class VideoRotaTripSynthesizer(RotaTripSynthesizer):  # deprecated
#     def __init__(self,
#                  batch_size=16,
#                  is_demo=True,
#                  ):
#         super(VideoRotaTripSynthesizer, self).__init__(batch_size=batch_size,
#                                                        is_demo=is_demo)
#
#         ''' face alignment '''
#         self.align_mode = 'set1'
#         self.net, self.detector = get_lmk_model()
#         self.net.eval()
#         print('alignment model loaded')
#         self.transform = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#         ])
#
#     def _crop_image(self, full_img):
#         B, C, H, W = full_img.shape
#         crop_batch = torch.zeros((B, C, H, W), dtype=torch.float32, device=full_img.device)
#
#         for b in range(B):
#             one_img = ((full_img[b] + 1.) * 127.5).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
#             one_img = Image.fromarray(one_img)
#
#             lmks = demo_image(one_img, self.net, self.detector)
#             if len(lmks) > 0:
#                 lmk = get_5_from_98(lmks[0])
#                 cropped_img = norm_crop(one_img, lmk, 256, mode=self.align_mode, borderValue=0.0)
#             else:
#                 print('Failed.')
#                 cropped_img = one_img
#
#             cropped_img = self.transform(cropped_img)
#             crop_batch[b] = cropped_img
#
#         return crop_batch
#
#     def _step1_reenactment_video(self, tensor_st, save_name):
#         from PIL import Image
#         B, C, H, W = tensor_st.shape
#
#         # (C,H,W) [-1,1] Tensor to (H,W,C) [0,255] np.ndarray to PIL.Image
#         st_img = ((tensor_st[0] + 1.) * 127.5).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
#         st_img = Image.fromarray(st_img)
#
#         st_img.save(os.path.join(self.snapshot_folder, save_name))
#
#         return tensor_st
#
#     def test_step(self, batch, batch_idx):
#         # Tensor, (N,RGB,H,W), in [-1,1]
#         tensor_t: torch.Tensor = batch['target_image']
#         tensor_s: torch.Tensor = batch['source_image']
#         tensor_ts: torch.Tensor = batch['reen_ts']
#         tensor_st: torch.Tensor = batch['reen_st']
#         pair_strs: tuple = tuple(batch['pair_str'])
#
#         snapshot_folder = os.path.join(self.demo_folder, 'demo_' + str(batch_idx))
#         if not os.path.exists(snapshot_folder):
#             os.mkdir(snapshot_folder)
#         self.snapshot_folder = snapshot_folder
#
#         self._save_input(tensor_s, tensor_t, save_idx=0)
#
#         """ 1. reenactment """
#         import time
#         time_0 = time.time()
#         # Tensor, (N,RGB,H,W), in [-1,1]
#         reen_st = self._step1_reenactment_video(tensor_st, 'reen_st.jpg')
#         reen_ts = self._step1_reenactment_video(tensor_ts, 'reen_ts.jpg')
#
#         print('step 1: %d ms' % int((time.time() - time_0) * 1000))
#         time_0 = time.time()
#
#         """ 2. multi_band blending """
#         im_band_st, im_regions_st = self._step2_blending(reen_st, tensor_t, is_st=True)
#         im_band_ts, im_regions_ts = self._step2_blending(reen_ts, tensor_s, is_st=False)
#
#         print('step 2: %d ms' % int((time.time() - time_0) * 1000))
#         time_0 = time.time()
#
#         """ 3. drop and inpaint """
#         im_restore_st = self._step3_inpaint(im_band_st, im_regions_st, is_st=True)
#         im_restore_ts = self._step3_inpaint(im_band_ts, im_regions_ts, is_st=False)
#
#         print('step 3: %d ms' % int((time.time() - time_0) * 1000))
#         time_0 = time.time()
#
#         """ 4. save as dataset """
#         self._step4_save_to_dataset(tensor_s, tensor_t, im_restore_st, im_restore_ts, pair_strs)
#
#     def test_dataloader(self):
#         from dataset.dataloader import DaGanDataset
#         video_dataset = DaGanDataset()
#         return DataLoader(
#             video_dataset,
#             batch_size=self.batch_size,
#             num_workers=32,
#             drop_last=False,
#             shuffle=False,
#         )


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dataset', type=str, default='/gavin/datasets/triplet_lia_tmp', help='saved folder')
    parser.add_argument('--demo_folder', type=str, default='./demo_snapshot_lia', help='saved folder')

    parser.add_argument('--demo', action='store_true', default=False, help='open demo mode')
    args = parser.parse_args()

    if args.demo:
        set_random_seed(41)
    else:
        set_random_seed(int(time.time()))

    s = RotaTripSynthesizer(
        batch_size=1,
        is_demo=args.demo,
        save_dataset_folder=args.save_dataset,
        demo_folder=args.demo_folder,
    )
    # vs = VideoRotaTripSynthesizer(
    #     batch_size=1,
    #     is_demo=True,
    # )

    trainer = pl.Trainer(
        logger=False,
        gpus=1,
        gradient_clip_val=0,
        max_epochs=300,
        num_sanity_val_steps=1,
        limit_val_batches=0.0,
        progress_bar_refresh_rate=1,
        distributed_backend="dp",
        benchmark=True,
    )
    trainer.fit(s)
