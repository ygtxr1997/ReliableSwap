import torch
import torchvision
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl

import numpy as np
import kornia
import wandb

from modules.networks.faceshifter import FSGenerator
from modules.layers.discriminator import MultiscaleDiscriminator
from modules.losses.faceshifter_loss import DLoss, GLoss
from modules.dataset.dataloader import PickleTrainDataset, PickleValDataset
from modules.dataset.dataloader import BatchTrainDataset, BatchValDataset
from modules.dataset.dataloader import TripletTrainDataset
from modules.dataset.dataloader import VanillaAndTripletDataset


class FaceshifterPL(pl.LightningModule):
    def __init__(
        self,
        batch_size=10,
        n_layers=3,
        num_D=3,
        same_rate=20,
        config: dict = None,
        finetune: bool = False,
    ):
        super(FaceshifterPL, self).__init__()

        self.config: dict = config
        print(config)
        self.in_size = self.config['model']['in_size']

        self.generator = FSGenerator(
            "/apdcephfs_cq2/share_1290939/gavinyuan/code/FaceShifter/faceswap/faceswap/checkpoints/face_id/ms1mv3_arcface_r100_fp16_backbone.pth",
            mouth_net_param=self.config.get('mouth_net'),
            in_size=self.in_size,
            finetune=finetune,
        )
        self.discriminator = MultiscaleDiscriminator(
            3, n_layers=n_layers, num_D=num_D, getIntermFeat=True
        )
        self.g_loss = GLoss(
            "/apdcephfs_cq2/share_1290939/gavinyuan/code/FaceShifter/faceswap/faceswap/checkpoints/face_id/ms1mv3_arcface_r100_fp16_backbone.pth",
            n_layers=n_layers,
            num_D=num_D,
            loss_config=self.config.get('loss'),
            mouth_net=self.generator.mouth_net,
            mouth_crop_param=self.config.get('mouth_net').get('crop_param'),
        )
        self.d_loss = DLoss()

        if finetune:
            self.discriminator.requires_grad_(False)
            for name, param in self.discriminator.named_parameters():
                if 'scale2' in name:
                    param.requires_grad = True

        self.automatic_optimization = False
        self.batch_size = batch_size
        self.same_rate = same_rate

        # self.bisenet = self.g_loss.realism_loss.bisenet
        # self.get_mask = self.g_loss.realism_loss.get_face_mask

        ''' dataset '''
        self.train_dataset_name = self.config.get('dataset').get('train_dataset')
        self.val_dataset_name = self.config.get('dataset').get('val_dataset')
        self.triplet_ratio = self.config.get('dataset').get('triplet_ratio')
        assert 0 <= self.triplet_ratio <= 100, 'ratio should be in [0,100]'

        same_batch_round, diff_batch_round = self._get_round(same_rate)
        self.same_batch_round = same_batch_round  # 1 if same_rate=20
        self.diff_batch_round = diff_batch_round  # 4 if same_rate=20
        self.full_batch_round = same_batch_round + diff_batch_round  # 5 if same_rate=20
        print('full_round=%d, same_round=%d, diff_round=%d' % (
            self.full_batch_round, self.same_batch_round, self.diff_batch_round
        ))

    def forward(self, source_img, target_img):
        i_r = self.generator(source_img, target_img)[0]
        return i_r

    def postprocess(self, x):
        return x * 0.5 + 0.5

    def create_masks(self, mask, pad=20, out_pad=30):
        # mask B, 1, H, W
        # 创建一个mask，包含边界以及一定宽度的mask
        H, W = mask.size()[2], mask.size()[3]
        mask_inner = F.pad(mask, pad=[pad, pad, pad, pad], mode="constant", value=0)
        mask_inner = F.interpolate(mask_inner, size=(H, W), mode="bilinear", align_corners=True)
        mask_outer = F.interpolate(
            mask_inner[:, :, out_pad:-out_pad, out_pad:-out_pad],
            size=(H, W),
            mode="bilinear",
            align_corners=True,
        )
        mask_border = (mask_outer - mask_inner).abs()
        return mask_inner, mask_outer, mask_border

    @staticmethod
    def _get_round(same_rate):
        import math
        gcd_value = math.gcd(same_rate, 100 - same_rate)
        return same_rate // gcd_value, (100 - same_rate) // gcd_value

    def training_step(self, batch, batch_idx):
        opt_g, opt_d = self.optimizers(use_pl_optimizer=True)

        """
                |x4 diff        |x1 same
                |80%    20%     |80%    20%
        -------------------------------------------
        step:   v_diff  t_diff  v_same  t_same
        batch:  0               1              
        t_code: 0       4/5     1       1
        same:   0       0       1       1
        """

        same: torch.Tensor = batch["same"]  # (B,1), in {0,1}
        batch_same = batch_idx % self.full_batch_round < self.same_batch_round
        batch_triplet = np.random.randint(100) <= 20  # triplet_rate_of_vanilla = 0.2
        use_triplet = 'triplet' in self.train_dataset_name

        ''' 1. vanilla samples '''
        i_target = batch["target_image"]
        i_source = batch["source_image"]

        if not batch_same:
            same = same.fill_(0)
        else:  # same
            i_source = i_target
            same = same.fill_(1)

        i_refer = None
        t_code: torch.Tensor = None
        if use_triplet:
            i_refer = batch["t_refer_image"]
            t_code = batch["type_code"]
            if not batch_triplet:
                i_refer = i_refer.fill_(0)
                if batch_same:
                    t_code = t_code.fill_(1)
                else:
                    t_code = t_code.fill_(0)

        ''' 2. triplet samples '''
        if use_triplet and batch_triplet:
            i_target = batch["t_target_image"]
            i_source = batch["t_source_image"]

            if batch_same:  # same
                i_source = i_target
                i_refer = i_target
                t_code = t_code.fill_(1)

        image_dict = {}

        ''' generator loss '''
        i_result, source_id, _ = self.generator(i_source, i_target)
        result_att = self.generator.get_att(i_result)
        with torch.no_grad():
            target_att = self.generator.get_att(i_target)
        i_cycle, _, _ = self.generator(i_target, i_result.detach())  # .detach()

        d_result = self.discriminator(i_result)
        # d_gt = self.discriminator(i_t)  # deleted

        g_loss, g_loss_dict = self.g_loss(
            i_target=i_target,
            i_source=i_source,
            i_result=i_result,
            i_cycle=i_cycle,
            d_result=d_result,
            target_att=target_att,
            result_att=result_att,
            same=same,
            use_triplet=use_triplet,
            i_refer=i_refer,
            type_code=t_code,
        )

        opt_g.zero_grad()
        self.manual_backward(g_loss)
        opt_g.step()

        ''' discriminator loss '''
        d_gt = self.discriminator(i_target)
        d_fake = self.discriminator(i_result.detach())

        d_loss, d_loss_dict = self.d_loss(d_gt, d_fake, same, None, None)
        opt_d.zero_grad()
        self.manual_backward(d_loss)
        opt_d.step()

        ''' loss logging '''
        self.logging_dict(g_loss_dict, prefix="train / ")
        self.logging_dict(d_loss_dict, prefix="train / ")
        self.logging_lr()

        ''' image logging '''
        if self.global_step % 1000 == 0 or self.global_step % 1000 == 1:
            mask_real = self.g_loss.realism_loss.get_any_mask(i_target, par=[11, 12, 13])
            # mask_real = self.g_loss.realism_loss.get_eye_mouth_mask(i_target)
            # mask_face = self.g_loss.realism_loss.get_face_mask(i_target)
            # mask_real = (self.create_masks(mask_face)[2] + mask_real).clamp(0.0, 1.0)
            image_dict["mask_target"] = mask_real * self.postprocess(i_target)

            # mask_real = self.g_loss.realism_loss.get_eye_mouth_mask(i_r.detach())
            # mask_face = self.g_loss.realism_loss.get_face_mask(i_r.detach())
            # mask_real = (self.create_masks(mask_face)[2] + mask_real).clamp(0.0, 1.0)

            mask_real = self.g_loss.realism_loss.get_any_mask(
                i_result.detach(), par=[11, 12, 13]
            )
            image_dict["mask_r"] = mask_real * self.postprocess(i_result.detach())

            image_dict["I_target"] = self.postprocess(i_target)
            image_dict["I_source"] = self.postprocess(i_source)
            image_dict["I_result"] = self.postprocess(i_result)
            image_dict["I_cycle"] = self.postprocess(i_cycle)
            i_refer = torch.zeros_like(i_source) if i_refer is None else i_refer
            image_dict["I_refer"] = self.postprocess(i_refer)
            self.logging_image_dict(image_dict, prefix="train / ")

            # (st, ts, s, Y) visualization
            B = i_target.shape[0]
            triplet_images = []
            for b in range(B):
                triplet_images.append(image_dict["I_target"][b])
                triplet_images.append(image_dict["I_source"][b])
                triplet_images.append(image_dict["I_result"][b])
                triplet_images.append(image_dict["I_refer"][b])
            triplet_img = torchvision.utils.make_grid(triplet_images, nrow=8)
            self.logger.experiment.log(
                {"train / triplet_group": wandb.Image(triplet_img.clamp(0, 1))}, commit=False
            )

    def validation_step(self, batch, batch_idx):
        i_target = batch["target_image"]
        i_source = batch["source_image"]

        M = self.g_loss.trans_matrix.repeat(i_source.size()[0], 1, 1)

        same = batch["same"]
        image_dict = {}

        # region generator
        # i_result, v_sid, zatt, m = self.generator(i_source, i_target)  ##### experiment
        i_result, v_sid, zatt = self.generator(i_source, i_target)
        ratt = self.generator.get_att(i_result)
        # i_cycle, _, _, _ = self.generator(i_target, i_result)  #### experiment
        i_cycle, _, _ = self.generator(i_target, i_result)
        d_r = self.discriminator(i_result)
        d_gt = self.discriminator(i_target)
        g_loss, g_loss_dict = self.g_loss(
            i_target=i_target,
            i_source=i_source,
            i_result=i_result,
            i_cycle=i_cycle,
            d_result=d_r,
            target_att=zatt,
            result_att=ratt,
            same=same,
        )
        # endregion

        # region discriminator
        d_fake = self.discriminator(i_result.detach())
        d_loss, d_loss_dict = self.d_loss(d_gt, d_fake, same)
        # endregion

        # region logging
        self.logging_dict(g_loss_dict, prefix="validation / ")
        self.logging_dict(d_loss_dict, prefix="validation / ")

        i_source = kornia.geometry.transform.warp_affine(i_source, M, (self.in_size, self.in_size))
        # i_source = kornia.geometry.transform.warp_affine(i_source, M, (512, 512)) # yuange
        image_dict["I_target"] = self.postprocess(i_target)
        image_dict["I_source"] = self.postprocess(i_source)
        image_dict["I_r"] = self.postprocess(i_result)
        image_dict["I_cycle"] = self.postprocess(i_cycle)
        # print(i_target.shape, i_source.shape, i_result.shape, i_cycle.shape)
        # endregion

        return image_dict

    def validation_epoch_end(self, outputs):
        val_images = []

        for idx, output in enumerate(outputs):
            if idx > 30:
                break
            val_images.append(output["I_target"][0])
            val_images.append(output["I_source"][0])
            val_images.append(output["I_r"][0])
            val_images.append(output["I_cycle"][0])
            # yuange
            # print('1', output["I_target"][0].shape)
            # print('2', output["I_source"][0].shape)
            # print('3', output["I_r"][0].shape)
            # print('4', output["I_cycle"][0].shape)
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
        optimizer_g = torch.optim.Adam(
            self.generator.G.parameters(), lr=0.0001, betas=[0.0, 0.999]
        )  # self.generator.G.parameters()
        optimizer_list.append({"optimizer": optimizer_g})
        optimizer_d = torch.optim.Adam(
            self.discriminator.parameters(), lr=0.0001, betas=[0.0, 0.999]
        )
        optimizer_list.append({"optimizer": optimizer_d})

        return optimizer_list

    def _get_train_dataloader(self, dataset_name: str):
        if dataset_name == 'hd':
            dataset = PickleTrainDataset(
                img_root="/apdcephfs_cq2/share_1290939/gavinyuan/datasets/hd.pickle",
                same_rate=self.same_rate,
                ffhq_mode=True,
            )
        elif dataset_name == 'image_512_quality':
            dataset = BatchTrainDataset(
                img_root="/gavin/datasets/original/image_512_quality.pickle",
                same_rate=0,  # self.same_rate,
                image_size=self.in_size,
                ffhq_mode=False,
                top_k=1500000,
            )
        elif 'triplet' in dataset_name:
            # dataset = TripletTrainDataset(
            #     triplet_pickle="/gavin/datasets/{}.pickle".format(dataset_name),
            #     triplet_ratio=self.triplet_ratio,
            #     same_rate=0,
            # )
            dataset = VanillaAndTripletDataset(
                triplet_pickle="/gavin/datasets/{}.pickle".format(dataset_name),
                triplet_ratio=self.triplet_ratio,
                same_rate=0,
                vanilla_simswap_mode=False,
                image_size=self.in_size,
            )
        else:
            raise ValueError('dataset_name not supported')
        return DataLoader(dataset, self.batch_size, num_workers=32,
                          shuffle=True, drop_last=True)

    def _get_val_dataloader(self, dataset_name: str):
        if dataset_name == 'hd':
            dataset = PickleValDataset(
                img_root="/apdcephfs/share_1290939/ahbanliang/datasets/hd.pickle",
                # img_root="/apdcephfs_cq2/share_1290939/gavinyuan/datasets/hd.pickle",
            )
        elif dataset_name == 'image_512_quality':
            dataset = BatchValDataset(
                img_root="/gavin/datasets/original/image_512_quality.pickle",
                image_size=self.in_size,
            )
        else:
            raise ValueError('dataset_name not supported')
        return DataLoader(dataset, batch_size=1, num_workers=8)

    def train_dataloader(self):
        return self._get_train_dataloader(self.train_dataset_name)

    def val_dataloader(self):
        return self._get_val_dataloader(self.val_dataset_name)


class FaceshifterPL512(pl.LightningModule):
    def __init__(
        self,
        batch_size=10,
        n_layers=3,
        num_D=3,
        same_rate=20,
        config: dict = None,
        finetune: bool = False,
        verbose: bool = True,
    ):
        super(FaceshifterPL512, self).__init__()

        self.config: dict = config
        print(config)
        self.in_size = self.config['model']['in_size']

        self.generator = FSGenerator(
            "/apdcephfs_cq2/share_1290939/gavinyuan/code/FaceShifter/faceswap/faceswap/checkpoints/face_id/ms1mv3_arcface_r100_fp16_backbone.pth",
            mouth_net_param=self.config.get('mouth_net'),
            in_size=self.in_size,
            finetune=finetune,
            downup=True,
        )
        self.discriminator = MultiscaleDiscriminator(
            3, n_layers=n_layers, num_D=num_D, getIntermFeat=True,
            finetune=finetune
        )
        self.g_loss = GLoss(
            "/apdcephfs_cq2/share_1290939/gavinyuan/code/FaceShifter/faceswap/faceswap/checkpoints/face_id/ms1mv3_arcface_r100_fp16_backbone.pth",
            n_layers=n_layers,
            num_D=num_D,
            loss_config=self.config.get('loss'),
            mouth_net=self.generator.mouth_net,
            mouth_crop_param=self.config.get('mouth_net').get('crop_param'),
            in_size=self.in_size,
        )
        self.d_loss = DLoss()

        self.finetune = finetune

        self.automatic_optimization = False
        self.batch_size = batch_size
        self.same_rate = same_rate

        # self.bisenet = self.g_loss.realism_loss.bisenet
        # self.get_mask = self.g_loss.realism_loss.get_face_mask

        ''' dataset '''
        self.train_dataset_name = self.config.get('dataset').get('train_dataset')
        self.val_dataset_name = self.config.get('dataset').get('val_dataset')
        self.triplet_ratio = self.config.get('dataset').get('triplet_ratio')
        assert 0 <= self.triplet_ratio <= 100, 'ratio should be in [0,100]'

        same_batch_round, diff_batch_round = self._get_round(same_rate)
        self.same_batch_round = same_batch_round  # 1 if same_rate=20
        self.diff_batch_round = diff_batch_round  # 4 if same_rate=20
        self.full_batch_round = same_batch_round + diff_batch_round  # 5 if same_rate=20
        print('full_round=%d, same_round=%d, diff_round=%d' % (
            self.full_batch_round, self.same_batch_round, self.diff_batch_round
        ))

        ckpt = torch.load('/apdcephfs/share_1290939/gavinyuan/out/triplet10w_38/epoch=13-step=491999.ckpt',
                          map_location='cpu')
        self.load_state_dict(ckpt, strict=False)

        print('[FaceshifterPL] ckpt loaded, finetune:{}, trainable params:'.format(
            self.finetune
        ))
        if verbose:
            for name, param in self.named_parameters():
                if param.requires_grad:
                    print(name)

    def forward(self, source_img, target_img):
        i_r = self.generator(source_img, target_img)[0]
        return i_r

    def postprocess(self, x):
        return x * 0.5 + 0.5

    def create_masks(self, mask, pad=20, out_pad=30):
        # mask B, 1, H, W
        # 创建一个mask，包含边界以及一定宽度的mask
        H, W = mask.size()[2], mask.size()[3]
        mask_inner = F.pad(mask, pad=[pad, pad, pad, pad], mode="constant", value=0)
        mask_inner = F.interpolate(mask_inner, size=(H, W), mode="bilinear", align_corners=True)
        mask_outer = F.interpolate(
            mask_inner[:, :, out_pad:-out_pad, out_pad:-out_pad],
            size=(H, W),
            mode="bilinear",
            align_corners=True,
        )
        mask_border = (mask_outer - mask_inner).abs()
        return mask_inner, mask_outer, mask_border

    @staticmethod
    def _get_round(same_rate):
        import math
        gcd_value = math.gcd(same_rate, 100 - same_rate)
        return same_rate // gcd_value, (100 - same_rate) // gcd_value

    def training_step(self, batch, batch_idx):
        opt_g, opt_d = self.optimizers(use_pl_optimizer=True)

        """
                |x4 diff        |x1 same
                |80%    20%     |80%    20%
        -------------------------------------------
        step:   v_diff  t_diff  v_same  t_same
        batch:  0               1              
        t_code: 0       4/5     1       1
        same:   0       0       1       1
        """

        same: torch.Tensor = batch["same"]  # (B,1), in {0,1}
        batch_same = batch_idx % self.full_batch_round < self.same_batch_round
        batch_triplet = np.random.randint(100) <= 20  # triplet_rate_of_vanilla = 0.2
        use_triplet = 'triplet' in self.train_dataset_name

        ''' 1. vanilla samples '''
        i_target = batch["target_image"]
        i_source = batch["source_image"]

        if not batch_same:
            same = same.fill_(0)
        else:  # same
            i_source = i_target
            same = same.fill_(1)

        i_refer = None
        t_code: torch.Tensor = None
        if use_triplet:
            i_refer = batch["t_refer_image"]
            t_code = batch["type_code"]
            if not batch_triplet:
                i_refer = i_refer.fill_(0)
                if batch_same:
                    t_code = t_code.fill_(1)
                else:
                    t_code = t_code.fill_(0)

        ''' 2. triplet samples '''
        if use_triplet and batch_triplet:
            i_target = batch["t_target_image"]
            i_source = batch["t_source_image"]

            if batch_same:  # same
                i_source = i_target
                i_refer = i_target
                t_code = t_code.fill_(1)

        image_dict = {}

        ''' generator loss '''
        i_result, source_id, _ = self.generator(i_source, i_target)
        result_att = self.generator.get_att(i_result)
        with torch.no_grad():
            target_att = self.generator.get_att(i_target)
        i_cycle, _, _ = self.generator(i_target, i_result.detach())  # .detach()

        d_result = self.discriminator(i_result)
        # d_gt = self.discriminator(i_t)  # deleted

        g_loss, g_loss_dict = self.g_loss(
            i_target=i_target,
            i_source=i_source,
            i_result=i_result,
            i_cycle=i_cycle,
            d_result=d_result,
            target_att=target_att,
            result_att=result_att,
            same=same,
            use_triplet=use_triplet,
            i_refer=i_refer,
            type_code=t_code,
        )

        opt_g.zero_grad()
        self.manual_backward(g_loss)
        opt_g.step()

        ''' discriminator loss '''
        d_gt = self.discriminator(i_target)
        d_fake = self.discriminator(i_result.detach())

        d_loss, d_loss_dict = self.d_loss(d_gt, d_fake, same, None, None)
        opt_d.zero_grad()
        self.manual_backward(d_loss)
        opt_d.step()

        ''' loss logging '''
        self.logging_dict(g_loss_dict, prefix="train / ")
        self.logging_dict(d_loss_dict, prefix="train / ")
        self.logging_lr()

        ''' image logging '''
        if self.global_step % 1000 == 0 or self.global_step % 1000 == 1:
            mask_real = self.g_loss.realism_loss.get_any_mask(i_target, par=[11, 12, 13])
            # mask_real = self.g_loss.realism_loss.get_eye_mouth_mask(i_target)
            # mask_face = self.g_loss.realism_loss.get_face_mask(i_target)
            # mask_real = (self.create_masks(mask_face)[2] + mask_real).clamp(0.0, 1.0)
            image_dict["mask_target"] = mask_real * self.postprocess(i_target)

            # mask_real = self.g_loss.realism_loss.get_eye_mouth_mask(i_r.detach())
            # mask_face = self.g_loss.realism_loss.get_face_mask(i_r.detach())
            # mask_real = (self.create_masks(mask_face)[2] + mask_real).clamp(0.0, 1.0)

            mask_real = self.g_loss.realism_loss.get_any_mask(
                i_result.detach(), par=[11, 12, 13]
            )
            image_dict["mask_r"] = mask_real * self.postprocess(i_result.detach())

            image_dict["I_target"] = self.postprocess(i_target)
            image_dict["I_source"] = self.postprocess(i_source)
            image_dict["I_result"] = self.postprocess(i_result)
            image_dict["I_cycle"] = self.postprocess(i_cycle)
            i_refer = torch.zeros_like(i_source) if i_refer is None else i_refer
            image_dict["I_refer"] = self.postprocess(i_refer)
            self.logging_image_dict(image_dict, prefix="train / ")

            # (st, ts, s, Y) visualization
            B = i_target.shape[0]
            triplet_images = []
            for b in range(B):
                triplet_images.append(image_dict["I_target"][b])
                triplet_images.append(image_dict["I_source"][b])
                triplet_images.append(image_dict["I_result"][b])
                triplet_images.append(image_dict["I_refer"][b])
            triplet_img = torchvision.utils.make_grid(triplet_images, nrow=8)
            self.logger.experiment.log(
                {"train / triplet_group": wandb.Image(triplet_img.clamp(0, 1))}, commit=False
            )

    def validation_step(self, batch, batch_idx):
        i_target = batch["target_image"]
        i_source = batch["source_image"]

        M = self.g_loss.trans_matrix.repeat(i_source.size()[0], 1, 1)

        same = batch["same"]
        image_dict = {}

        # region generator
        # i_result, v_sid, zatt, m = self.generator(i_source, i_target)  ##### experiment
        i_result, v_sid, zatt = self.generator(i_source, i_target)
        ratt = self.generator.get_att(i_result)
        # i_cycle, _, _, _ = self.generator(i_target, i_result)  #### experiment
        i_cycle, _, _ = self.generator(i_target, i_result)
        d_r = self.discriminator(i_result)
        d_gt = self.discriminator(i_target)
        g_loss, g_loss_dict = self.g_loss(
            i_target=i_target,
            i_source=i_source,
            i_result=i_result,
            i_cycle=i_cycle,
            d_result=d_r,
            target_att=zatt,
            result_att=ratt,
            same=same,
        )
        # endregion

        # region discriminator
        d_fake = self.discriminator(i_result.detach())
        d_loss, d_loss_dict = self.d_loss(d_gt, d_fake, same)
        # endregion

        # region logging
        self.logging_dict(g_loss_dict, prefix="validation / ")
        self.logging_dict(d_loss_dict, prefix="validation / ")

        i_source = kornia.geometry.transform.warp_affine(i_source, M, (self.in_size, self.in_size))
        # i_source = kornia.geometry.transform.warp_affine(i_source, M, (512, 512)) # yuange
        image_dict["I_target"] = self.postprocess(i_target)
        image_dict["I_source"] = self.postprocess(i_source)
        image_dict["I_r"] = self.postprocess(i_result)
        image_dict["I_cycle"] = self.postprocess(i_cycle)
        # print(i_target.shape, i_source.shape, i_result.shape, i_cycle.shape)
        # endregion

        return image_dict

    def validation_epoch_end(self, outputs):
        val_images = []

        for idx, output in enumerate(outputs):
            if idx > 30:
                break
            val_images.append(output["I_target"][0])
            val_images.append(output["I_source"][0])
            val_images.append(output["I_r"][0])
            val_images.append(output["I_cycle"][0])
            # yuange
            # print('1', output["I_target"][0].shape)
            # print('2', output["I_source"][0].shape)
            # print('3', output["I_r"][0].shape)
            # print('4', output["I_cycle"][0].shape)
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
        optimizer_g = torch.optim.Adam(
            self.generator.G.trainable_params(), lr=0.0001, betas=[0.0, 0.999]  # only finetune trainable parts
        )  # self.generator.G.parameters()
        optimizer_list.append({"optimizer": optimizer_g})
        optimizer_d = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.discriminator.parameters()),
            lr=0.0001, betas=[0.0, 0.999]
        )
        optimizer_list.append({"optimizer": optimizer_d})

        return optimizer_list

    def _get_train_dataloader(self, dataset_name: str):
        if dataset_name == 'hd':
            dataset = PickleTrainDataset(
                img_root="/apdcephfs_cq2/share_1290939/gavinyuan/datasets/hd.pickle",
                same_rate=self.same_rate,
                ffhq_mode=True,
            )
        elif dataset_name == 'image_512_quality':
            dataset = BatchTrainDataset(
                img_root="/gavin/datasets/original/image_512_quality.pickle",
                same_rate=0,  # self.same_rate,
                image_size=self.in_size,
                ffhq_mode=False,
                top_k=1500000,
            )
        elif 'triplet' in dataset_name:
            # dataset = TripletTrainDataset(
            #     triplet_pickle="/gavin/datasets/{}.pickle".format(dataset_name),
            #     triplet_ratio=self.triplet_ratio,
            #     same_rate=0,
            # )
            dataset = VanillaAndTripletDataset(
                triplet_pickle="/gavin/datasets/{}.pickle".format(dataset_name),
                triplet_ratio=self.triplet_ratio,
                same_rate=0,
                vanilla_simswap_mode=False,
                image_size=self.in_size,
            )
        else:
            raise ValueError('dataset_name not supported')
        return DataLoader(dataset, self.batch_size, num_workers=32,
                          shuffle=True, drop_last=True)

    def _get_val_dataloader(self, dataset_name: str):
        if dataset_name == 'hd':
            dataset = PickleValDataset(
                img_root="/apdcephfs_cq2/share_1290939/gavinyuan/datasets/hd.pickle",
            )
        elif dataset_name == 'image_512_quality':
            dataset = BatchValDataset(
                img_root="/gavin/datasets/original/image_512_quality.pickle",
                image_size=self.in_size,
            )
        else:
            raise ValueError('dataset_name not supported')
        return DataLoader(dataset, batch_size=1, num_workers=8)

    def train_dataloader(self):
        return self._get_train_dataloader(self.train_dataset_name)

    def val_dataloader(self):
        return self._get_val_dataloader(self.val_dataset_name)