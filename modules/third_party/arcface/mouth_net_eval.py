import argparse
import pytorch_lightning as pl
import numpy as np
import torch

from modules.third_party.arcface.mouth_net_pl import MouthNetPL
from modules.third_party.arcface.mouth_net import MouthNet


class MouthTest(object):
    def __init__(self):
        self.dataset_len = 400

        self.fixer_crop_param = (28, 56, 84, 112)
        self.fixer_casia_model = MouthNet(
            bisenet=None,
            feature_dim=128,
            crop_param=self.fixer_crop_param
        ).cuda()
        fixer_path = "/gavin/code/FaceSwapping/modules/third_party/arcface/weights/fixer_net_casia_28_56_84_112.pth"
        self.fixer_casia_model.load_backbone(fixer_path)
        self.fixer_casia_model.eval()
        self.fixer_t = np.zeros((self.dataset_len, 128), dtype=np.float32)
        self.fixer_s = np.zeros_like(self.fixer_t, dtype=np.float32)  # each embedding repeats 10 times in ffplus
        self.fixer_r = np.zeros_like(self.fixer_t, dtype=np.float32)
        print('Fixer model loaded.')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.val_targets = []
    args.rec_folder = "/gavin/datasets/msml/ms1m-retinaface"

    fixer_net = MouthNetPL.load_from_checkpoint(
        "/apdcephfs/share_1290939/gavinyuan/out/fixernet_casia/epoch=22-step=10999-v1.ckpt",
        map_location='cpu', strict=False,
        num_classes=10572,
        batch_size=128,
        dim_feature=128,
        rec_folder=args.rec_folder,
        header_type="AMCosFace",
        crop=(28, 56, 84, 112),
    )

    lower_net_1 = MouthNetPL.load_from_checkpoint(
        "/apdcephfs/share_1290939/gavinyuan/out/mouth_net_1/epoch=24-step=242999.ckpt",
        map_location='cpu', strict=False,
        num_classes=93431,
        batch_size=128,
        dim_feature=128,
        rec_folder=args.rec_folder,
        header_type="AMArcFace",
        crop=(28, 56, 84, 112),
    )

    # test_net = fixer_net
    test_net = lower_net_1
    trainer = pl.Trainer(
        logger=False,
        gpus=1,
        distributed_backend='dp',
        benchmark=True,
    )
    trainer.test(test_net)

    # print('Fixer model loading...')
    # m_test = MouthTest()
