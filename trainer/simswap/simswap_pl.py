import torch
import torchvision
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl

import os
import kornia
import wandb
import random

from modules.networks.simswap import fsModel, BaseModel, Generator_Adain_Upsample
from modules.third_party.arcface import iresnet100, MouthNet
from modules.layers.simswap.pg_modules.projected_discriminator import ProjectedDiscriminator
from modules.dataset.dataloader import BatchTrainDataset, BatchValDataset
from modules.dataset.dataloader import TripletTrainDataset
from modules.dataset.dataloader import VanillaAndTripletDataset

from modules.losses.simswap_loss import GLoss, DLoss

make_abs_path = lambda fn: os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), fn))


class SimSwapPL(pl.LightningModule):
    def __init__(
            self,
            batch_size=8,
            same_rate=50,
            config: dict = None,
            use_official_arc: bool = False,
    ):
        super(SimSwapPL, self).__init__()

        self.config: dict = config

        ''' MouthNet '''
        mouth_net_param = config.get('mouth_net')
        self.use_mouth_net = mouth_net_param.get('use')
        self.mouth_feat_dim = 0
        self.mouth_net = None
        self.mouth_crop_param = None
        if self.use_mouth_net:
            self.mouth_feat_dim = mouth_net_param.get('feature_dim')
            self.mouth_crop_param = mouth_net_param.get('crop_param')
            mouth_weight_path = make_abs_path(mouth_net_param.get('weight_path'))
            self.mouth_net = MouthNet(
                bisenet=None,
                feature_dim=self.mouth_feat_dim,
                crop_param=self.mouth_crop_param
            )
            self.mouth_net.load_backbone(mouth_weight_path)
            print("[SimSwapPL] MouthNet loaded from %s" % mouth_weight_path)
            self.mouth_net.eval()
            self.mouth_net.requires_grad_(False)

        ''' Face recognition net '''
        if not use_official_arc:
            netArc_pth = "/apdcephfs_cq2/share_1290939/gavinyuan/code/FaceShifter/faceswap/faceswap/" \
                         "checkpoints/face_id/ms1mv3_arcface_r100_fp16_backbone.pth"  # opt.Arc_path
            self.netArc = iresnet100(pretrained=False, fp16=False)
            self.netArc.load_state_dict(torch.load(netArc_pth, map_location="cpu"))
            self.netArc.eval()
            self.netArc.requires_grad_(False)
        else:
            # from trainer.simswap.models.models import ResNet, IRBlock
            # netArc_pth = make_abs_path("./arcface_model/arcface.pth")
            # self.netarc = ResNet(IRBlock, [3, 13, 30, 3])
            # self.netArc.load_state_dict(torch.load(netArc_pth, map_location="cpu"))
            import sys
            sys.path.insert(0, make_abs_path("./"))
            netArc_ckpt = make_abs_path("./arcface_model/arcface_checkpoint.tar")
            netArc_checkpoint = torch.load(netArc_ckpt, map_location=torch.device("cpu"))
            self.netArc = netArc_checkpoint['model'].module
            self.netArc = self.netArc.cuda()
            self.netArc.eval()
            self.netArc.requires_grad_(False)

        self.netG = Generator_Adain_Upsample(input_nc=3, output_nc=3, latent_size=512, n_blocks=9, deep=False,
                                             mouth_net_param=mouth_net_param)
        self.netD = ProjectedDiscriminator(diffaug=False, interp224=False, **{})
        self.netD.feature_network.requires_grad_(False)

        self.g_loss = GLoss(
            f_id=self.netArc,
            loss_config=self.config.get('loss'),
            mouth_net=self.mouth_net,
            mouth_crop_param=self.mouth_crop_param,
        )
        self.d_loss = DLoss()

        self.automatic_optimization = False
        self.batch_size = batch_size
        self.same_rate = same_rate
        self.randindex = [i for i in range(self.batch_size)]

        ''' dataset '''
        self.train_dataset_name = self.config.get('dataset').get('train_dataset')
        self.val_dataset_name = self.config.get('dataset').get('val_dataset')
        self.triplet_ratio = self.config.get('dataset').get('triplet_ratio')
        assert 0 <= self.triplet_ratio <= 100, 'ratio should be in [0,100]'

    def forward(self, source_img, target_img):
        i_result = self.netG(source=source_img, target=target_img,
                             net_arc=self.netArc,
                             mouth_net=self.mouth_net,
                             )
        return i_result

    def postprocess(self, x):
        img = x * 0.5 + 0.5  # in [0,1]
        return img

    def training_step(self, batch, batch_idx):
        opt_g, opt_d = self.optimizers(use_pl_optimizer=True)

        """
                |x5             |x1
        -------------------------------------------
        step:   v_same  v_diff  t_same  t_diff
        opt:    G   D   G   D   G   D   G   D
        batch:  0   1   2   3   20  21  22  23
        t_code: 1   1   0   0   1   1   4/5 4/5
        same:   1   1   0   0   1   1   0   0
        """

        same: torch.Tensor = batch["same"]  # (B,1), always 1 for vanilla, always 0 for triplet
        batch_same = batch_idx % 4 < 2  # (0,1:same) (2,3:diff)
        batch_triplet = batch_idx % 24 >= 20
        use_triplet = 'triplet' in self.train_dataset_name

        ''' 1. vanilla samples '''
        i_target = batch["target_image"]
        i_source = batch["source_image"]

        if not batch_same:  # diff
            random.shuffle(self.randindex)
            i_source = i_source[self.randindex]
            same = same.fill_(0)
        else:  # same
            same = same.fill_(1)

        i_refer = None
        t_code = None
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
        if use_triplet and batch_triplet:  # 20,21,22,23
            i_target = batch["t_target_image"]
            i_source = batch["t_source_image"]

            if batch_same:  # same
                i_source = i_target
                i_refer = i_target
                t_code = t_code.fill_(1)

        image_dict = {}

        i_result = self.netG(source=i_source, target=i_target,
                             net_arc=self.netArc,
                             mouth_net=self.mouth_net,
                             )

        if batch_idx % 2 == 0:
            ''' generator loss '''
            d_logit_fake, d_feat_fake = self.netD(i_result, None)
            d_feat_real = self.netD.get_feature(i_target)

            g_loss, g_loss_dict = self.g_loss(
                i_source=i_source,
                i_target=i_target,
                i_result=i_result,
                d_logit_fake=d_logit_fake,
                d_feat_fake=d_feat_fake,
                d_feat_real=d_feat_real,
                same=same,
                use_triplet=use_triplet,
                i_refer=i_refer,
                type_code=t_code,
            )

            opt_g.zero_grad()
            self.manual_backward(g_loss)
            opt_g.step()
            self.logging_dict(g_loss_dict, prefix="train / ")
        else:
            ''' discriminator loss '''
            d_logit_fake, _ = self.netD(i_result.detach(), None)
            d_logit_real, _ = self.netD(i_source, None)

            d_loss, d_loss_dict = self.d_loss(
                d_logit_fake=d_logit_fake,
                d_logit_real=d_logit_real
            )
            opt_d.zero_grad()
            self.manual_backward(d_loss)
            opt_d.step()
            self.logging_dict(d_loss_dict, prefix="train / ")

        ''' loss logging '''
        self.logging_lr()

        ''' image logging '''
        if self.global_step % 500 == 0 or self.global_step % 500 == 23:
            image_dict["I_target"] = self.postprocess(i_target)
            image_dict["I_source"] = self.postprocess(i_source)
            image_dict["I_result"] = self.postprocess(i_result)
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

        same = batch["same"]  # (B,1), in {0,1}
        image_dict = {}

        ''' generator loss '''
        # model.netD.requires_grad_(True)
        i_result = self.netG(source=i_source, target=i_target,
                             net_arc=self.netArc,
                             mouth_net=self.mouth_net,
                             )
        d_logit_fake, d_feat_fake = self.netD(i_result, None)

        d_feat_real = self.netD.get_feature(i_target)

        g_loss, g_loss_dict = self.g_loss(
            i_source=i_source,
            i_target=i_target,
            i_result=i_result,
            d_logit_fake=d_logit_fake,
            d_feat_fake=d_feat_fake,
            d_feat_real=d_feat_real,
            same=same,
        )

        ''' discriminator loss '''
        d_logit_fake, _ = self.netD(i_result.detach(), None)
        d_logit_real, _ = self.netD(i_source, None)

        d_loss, d_loss_dict = self.d_loss(
            d_logit_fake=d_logit_fake,
            d_logit_real=d_logit_real
        )

        ''' loss logging '''
        self.logging_dict(g_loss_dict, prefix="validation / ")
        self.logging_dict(d_loss_dict, prefix="validation / ")
        self.logging_lr()

        ''' image logging '''
        # i_source = kornia.geometry.transform.warp_affine(i_source, M, (256, 256))
        image_dict["I_target"] = self.postprocess(i_target)
        image_dict["I_source"] = self.postprocess(i_source)
        image_dict["I_r"] = self.postprocess(i_result)

        return image_dict

    def validation_epoch_end(self, outputs):
        val_images = []

        for idx, output in enumerate(outputs):
            if idx > 30:
                break
            val_images.append(output["I_target"][0])
            val_images.append(output["I_source"][0])
            val_images.append(output["I_r"][0])
        val_image = torchvision.utils.make_grid(val_images, nrow=6)
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
        # optimizer G
        params = list(self.netG.parameters())
        optimizer_G = torch.optim.Adam(params, lr=0.0004, betas=(0.0, 0.99), eps=1e-8)
        optimizer_list.append({"optimizer": optimizer_G})

        # optimizer D
        params = list(self.netD.parameters())
        optimizer_D = torch.optim.Adam(params, lr=0.0004, betas=(0.0, 0.99), eps=1e-8)
        optimizer_list.append({"optimizer": optimizer_D})

        return optimizer_list

    def _get_train_dataloader(self, dataset_name: str):
        if dataset_name == 'image_512_quality':
            dataset = BatchTrainDataset(
                img_root="/gavin/datasets/original/image_512_quality.pickle",
                same_rate=self.same_rate,
                image_size=256,
                ffhq_mode=False,
                top_k=1500000,
                simswap_mode=True,
            )
        elif 'triplet' in dataset_name:
            # dataset = TripletTrainDataset(
            #     triplet_pickle="/gavin/datasets/{}.pickle".format(dataset_name),
            #     triplet_ratio=self.triplet_ratio,
            #     same_rate=self.same_rate,
            # )
            dataset = VanillaAndTripletDataset(
                triplet_pickle="/gavin/datasets/{}.pickle".format(dataset_name),
                triplet_ratio=self.triplet_ratio,
                same_rate=self.same_rate,
                vanilla_simswap_mode=True,
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
            )
        else:
            raise ValueError('dataset_name not supported')
        return DataLoader(dataset, batch_size=1, num_workers=8)

    def train_dataloader(self):
        return self._get_train_dataloader(self.train_dataset_name)

    def val_dataloader(self):
        return self._get_val_dataloader(self.val_dataset_name)


if __name__ == "__main__":
    import thop
    import yaml

    with open(make_abs_path('./config.yaml'), 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config['mouth_net'] = {
        "use": False
    }

    img = torch.randn(1, 3, 112, 112)
    simswap_pl = SimSwapPL(batch_size=1, config=config).requires_grad_(False)
    simswap_pl.eval()
    arc = simswap_pl.netArc
    net = simswap_pl.netG.eval()
    flops, params = thop.profile(arc, inputs=(img,), verbose=False)
    print('#Params=%.2fM, GFLOPS=%.2f' % (params / 1e6, flops / 1e9))
