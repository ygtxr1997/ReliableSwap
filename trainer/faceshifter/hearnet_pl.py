import torch
import torchvision
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl

import os
import kornia
import wandb

from modules.networks.faceshifter import FSHearNet
from modules.losses.faceshifter_loss import HearLoss
from modules.dataset.dataloader import BatchTrainDataset, BatchValDataset

make_abs_path = lambda fn: os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), fn))


class HearNetPL(pl.LightningModule):
    def __init__(
        self,
        batch_size=8,
        same_rate=50,
    ):
        super(HearNetPL, self).__init__()

        self.hear = FSHearNet(
            # aei_path="./out/faceshifter_vanilla/epoch=32-step=509999.ckpt"
            aei_path=make_abs_path("./extracted_ckpt/hear_tmp.pth")
        )

        self.hear_loss = HearLoss(
            f_id_checkpoint_path=make_abs_path("../../modules/third_party/arcface/weights/"
                                               "ms1mv3_arcface_r100_fp16/backbone.pth")
        )

        self.automatic_optimization = False
        self.batch_size = batch_size
        self.same_rate = same_rate

        ''' dataset '''
        self.train_dataset_name = "image_512_quality"
        self.val_dataset_name = "image_512_quality"

    def forward(self, source_img, target_img):
        y_st = self.hear(source_img, target_img)[0]
        return y_st

    def postprocess(self, x):
        return x * 0.5 + 0.5

    def training_step(self, batch, batch_idx):
        opt_hear = self.optimizers(use_pl_optimizer=True)
        i_t = batch["target_image"]
        i_s = batch["source_image"]
        same = batch["same"]  # (B,1), in {0,1}

        image_dict = {}

        ''' hear loss '''
        y_st, y_hat_st = self.hear(i_s, i_t)
        hear_loss, hear_loss_dict = self.hear_loss(
            x_s=i_s,
            x_t=i_t,
            y_st=y_st,
            y_hat_st=y_hat_st,
            same=same
        )
        opt_hear.zero_grad()
        self.manual_backward(hear_loss)
        opt_hear.step()

        ''' loss logging '''
        self.logging_dict(hear_loss_dict, prefix="train / ")
        self.logging_dict(hear_loss_dict, prefix="train / ")
        self.logging_lr()

        ''' image logging '''
        if self.global_step % 500 == 0:
            image_dict["I_target"] = self.postprocess(i_t)
            image_dict["I_source"] = self.postprocess(i_s)
            image_dict["I_y_hat_st"] = self.postprocess(y_hat_st)
            image_dict["I_y_st"] = self.postprocess(y_st)
            self.logging_image_dict(image_dict, prefix="train / ")

            # (t,s,y_hat_st,y_st) visualization
            B = i_t.shape[0]
            triplet_images = []
            for b in range(B):
                triplet_images.append(image_dict["I_target"][b])
                triplet_images.append(image_dict["I_source"][b])
                triplet_images.append(image_dict["I_y_hat_st"][b])
                triplet_images.append(image_dict["I_y_st"][b])
            triplet_img = torchvision.utils.make_grid(triplet_images, nrow=8)
            self.logger.experiment.log(
                {"train / triplet_group": wandb.Image(triplet_img.clamp(0, 1))}, commit=False
            )

    def validation_step(self, batch, batch_idx):
        i_t = batch["target_image"]
        i_s = batch["source_image"]
        same = batch["same"]

        M = self.hear_loss.trans_matrix.repeat(i_s.size()[0], 1, 1)

        image_dict = {}

        ''' hear loss '''
        y_st, y_hat_st = self.hear(i_s, i_t)
        hear_loss, hear_loss_dict = self.hear_loss(
            x_s=i_s,
            x_t=i_t,
            y_st=y_st,
            y_hat_st=y_hat_st,
            same=same
        )

        ''' loss and image logging '''
        self.logging_dict(hear_loss_dict, prefix="validation / ")

        i_s = kornia.geometry.transform.warp_affine(i_s, M, (256, 256))
        image_dict["I_target"] = self.postprocess(i_t)
        image_dict["I_source"] = self.postprocess(i_s)
        image_dict["I_y_hat_st"] = self.postprocess(y_hat_st)
        image_dict["I_y_st"] = self.postprocess(y_st)

        return image_dict

    def validation_epoch_end(self, outputs):
        val_images = []

        for idx, output in enumerate(outputs):
            if idx > 30:
                break
            val_images.append(output["I_target"][0])
            val_images.append(output["I_source"][0])
            val_images.append(output["I_y_hat_st"][0])
            val_images.append(output["I_y_st"][0])
        val_image = torchvision.utils.make_grid(val_images, nrow=8)
        self.logger.experiment.log(
            {"validation / val_img": wandb.Image(val_image.clamp(0, 1))}, commit=False
        )

    def logging_dict(self, log_dict, prefix=None):
        for key, val in log_dict.items():
            if prefix is not None:
                key = prefix + key
            self.log(key, val)

    def logging_image_dict(self, image_dict, prefix=None, commit=False):
        for key, val in image_dict.items():
            if prefix is not None:
                key = prefix + key
            self.logger.experiment.log(
                {key: wandb.Image(val.clamp(0, 1))}, commit=commit
            )

    def logging_lr(self):
        opts = self.trainer.optimizers
        for idx, opt in enumerate(opts):
            lr = None
            for param_group in opt.param_groups:
                lr = param_group["lr"]
                break
            self.log(f"lr_{idx}", lr)

    def configure_optimizers(self):
        optimizer_list = []
        optimizer_hear = torch.optim.Adam(
            self.hear.hear.parameters(), lr=0.0004, betas=[0.0, 0.999]
        )
        optimizer_list.append({"optimizer": optimizer_hear})

        return optimizer_list

    def _get_train_dataloader(self, dataset_name: str):
        if dataset_name == 'image_512_quality':
            dataset = BatchTrainDataset(
                img_root="/gavin/datasets/original/image_512_quality.pickle",
                same_rate=self.same_rate,
                image_size=256,
                ffhq_mode=False,
                top_k=1500000,
                use_occ=True,
            )
        else:
            raise ValueError('dataset_name not supported')
        return DataLoader(dataset, self.batch_size, num_workers=32,
                          shuffle=True, drop_last=True)

    def _get_val_dataloader(self, dataset_name: str):
        if dataset_name == 'image_512_quality':
            dataset = BatchValDataset(
                img_root="/gavin/datasets/original/image_512_quality.pickle",
                image_size=256,
                use_occ=True,
            )
        else:
            raise ValueError('dataset_name not supported')
        return DataLoader(dataset, batch_size=1, num_workers=8)

    def train_dataloader(self):
        return self._get_train_dataloader(self.train_dataset_name)

    def val_dataloader(self):
        return self._get_val_dataloader(self.val_dataset_name)
