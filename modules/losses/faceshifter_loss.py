import torch
import torch.nn as nn
import torch.nn.functional as F

import kornia
import wandb
import lpips

from torchvision.models import vgg19

from modules.third_party.arcface import iresnet100
from modules.third_party.tdmm.resnet import ReconNetWrapper
from modules.third_party.tdmm.bfm import ParametricFaceModel
from modules.third_party.bisenet.bisenet import BiSeNet
from modules.third_party.vgg.modules.vgg import VGG_Model
from modules.losses.loss import CXLoss



class MultiScaleGANLoss(nn.Module):
    def __init__(
        self,
        gan_mode="hinge",
        target_real_label=1.0,
        target_fake_label=0.0,
        tensor=torch.FloatTensor,
        opt=None,
    ):
        super(MultiScaleGANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_tensor = None
        self.fake_label_tensor = None
        self.zero_tensor = None
        self.Tensor = tensor
        self.gan_mode = gan_mode
        self.opt = opt
        if gan_mode == "ls":
            pass
        elif gan_mode == "original":
            pass
        elif gan_mode == "w":
            pass
        elif gan_mode == "hinge":
            pass
        else:
            raise ValueError("Unexpected gan_mode {}".format(gan_mode))

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            return torch.ones_like(input).detach()
        else:
            return torch.zeros_like(input).detach()

    def get_zero_tensor(self, input):
        return torch.zeros_like(input).detach()

    def loss(self, input, same, target_is_real, for_discriminator=True):
        if self.gan_mode == "original":  # cross entropy loss
            target_tensor = self.get_target_tensor(input, target_is_real)
            loss = F.binary_cross_entropy_with_logits(input, target_tensor)
            return loss
        elif self.gan_mode == "ls":
            target_tensor = self.get_target_tensor(input, target_is_real)
            return F.mse_loss(input, target_tensor)
        elif self.gan_mode == "hinge":
            if for_discriminator:
                if target_is_real:
                    minval = torch.min(input - 1, self.get_zero_tensor(input))
                    # loss = -torch.mean(minval, dim=(1, 2, 3)) * same
                    # loss = loss / (torch.sum(same) + 1e-6)
                    # loss = torch.mean(loss)
                    loss = -torch.mean(minval)
                else:
                    minval = torch.min(-input - 1, self.get_zero_tensor(input))
                    # loss = -torch.mean(minval, dim=(1, 2, 3)) * same
                    # loss = loss / (torch.sum(same) + 1e-6)
                    # loss = torch.mean(loss)
                    loss = -torch.mean(minval)
            else:
                assert (
                    target_is_real
                ), "The generator's hinge loss must be aiming for real"
                # loss = -torch.mean(input, dim=(1, 2, 3)) * same
                # loss = loss / (torch.sum(same) + 1e-6)
                # loss = torch.mean(loss)
                loss = -torch.mean(input)
            return loss
        else:
            # wgan
            if target_is_real:
                return -input.mean()
            else:
                return input.mean()

    def __call__(self, input, same, target_is_real, for_discriminator=True, mask=None):
        # computing loss is a bit complicated because |input| may not be
        # a tensor, but list of tensors in case of multiscale discriminator
        if isinstance(input, list):
            loss = 0
            for pred_i in input:
                if isinstance(pred_i, list):
                    pred_i = pred_i[-1]
                if not mask is None:
                    h, w = pred_i.size()[2], pred_i.size()[3]
                    mask = F.interpolate(mask, [h, w], )
                    pred_i = pred_i * mask
                loss_tensor = self.loss(pred_i, same, target_is_real, for_discriminator)
                bs = 1 if len(loss_tensor.size()) == 0 else loss_tensor.size(0)
                new_loss = torch.mean(loss_tensor.view(bs, -1), dim=1)
                loss += new_loss
            return loss / len(input)
        else:
            if not mask is None:
                h, w = input.size()[2], input.size()[3]
                mask = F.interpolate(mask, [h, w], )
                input = input * mask
            return self.loss(input, same, target_is_real, for_discriminator)


class SIDLoss(nn.Module):
    def __init__(self,
                 weights_dict={"id": 10, "att": 5},
                 ):
        super(SIDLoss, self).__init__()
        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss(reduction='none')
        self.weights_dict = weights_dict

    def forward(self, v_id_I_s, v_id_I_r, target_att, result_att, same, type_code=None,
                mouth_source = None,
                mouth_result = None):
        if type_code is not None:
            same[type_code == 1] = 0  # also when X_t == X_s
            # same[type_code == 4] = 1  # do not use (st,ts,s)
            # same[type_code == 5] = 1  # do not use (ts,st,t)

        # only if (same == 0)
        id_loss = (1 - F.cosine_similarity(v_id_I_s, v_id_I_r, 1)) * (1 - same)
        id_loss = id_loss.mean()

        # only if (same == 0)
        mouth_loss = 0
        if mouth_source is not None and mouth_result is not None:
            mouth_loss = (1 - F.cosine_similarity(mouth_source, mouth_result, 1)) * (1 - same)
            mouth_loss = mouth_loss.mean()

        # only if (same == 0)
        att_loss = 0.
        for zat, rat in zip(target_att, result_att):
            batch_mse = self.mse(zat, rat).mean(axis=(1, 2, 3)) * (1 - same)
            att_loss += batch_mse.mean()

        sid_loss = (id_loss * self.weights_dict["id"]
                    + mouth_loss * self.weights_dict["mouth"]
                    + att_loss * self.weights_dict["att"])
        return sid_loss, {"id_loss": id_loss,
                          "mouth_loss": mouth_loss,
                          "att_loss": att_loss,
                          }


class RealismLoss(nn.Module):
    def __init__(self,
                 weights_dict_train: dict = None,
                 weights_dict_val: dict = None,
                 ):
        super(RealismLoss, self).__init__()
        self.l1 = nn.L1Loss()
        self.l2 = nn.MSELoss()
        self.vgg_layer = [
            "conv_3_4",
            "conv_5_2",
            "conv_4_2",
            "conv_3_2",
            "conv_2_2",
            "conv_1_2",
            "conv_5_2",
        ]
        self.weights_dict_train = weights_dict_train
        self.weights_dict_val = weights_dict_val
        self.vgg_loss_weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
        vgg = vgg19(pretrained=False)
        vgg.load_state_dict(
            torch.load(
                "/gavin/datasets/hanbang/vgg19-dcbb9e9d.pth",
                map_location="cpu",
            )
        )

        for param in vgg.parameters():
            param.requires_grad = False
        vgg.eval()

        self.bisenet = BiSeNet(19)
        self.bisenet.load_state_dict(
            torch.load(
                "/gavin/datasets/hanbang/79999_iter.pth",
                map_location="cpu",
            )
        )
        self.bisenet.eval()
        self.bisenet.requires_grad_(False)

        self.vgg_model = VGG_Model(vgg, self.vgg_layer)
        for param in self.vgg_model.parameters():
            param.requires_grad = False
        self.vgg_model.eval()
        self.register_buffer(
            name="vgg_mean",
            tensor=torch.tensor([[[0.485]], [[0.456]], [[0.406]]], requires_grad=False),
        )
        self.register_buffer(
            name="vgg_std",
            tensor=torch.tensor([[[0.229]], [[0.224]], [[0.225]]], requires_grad=False),
        )
        self.adv_loss = MultiScaleGANLoss()
        self.cxloss = CXLoss(0.2)

    def forward(self, i_source, i_target, i_result, i_cycle, d_result, same,
                i_refer, type_code,
                ):
        """

        :param i_source: not used
        :param i_target: target image
        :param i_result: swapped face
        :param i_cycle: cycle swapped face
        :param d_result: discriminator output of i_result
        :param same: if i_source == i_target, i_target will be the reference image
        :param i_refer: not used
        :param type_code: not used
        :return:
        """
        assert i_refer is None and type_code is None, 'loss input value error!'
        weights_dict = self.weights_dict_train if self.training else self.weights_dict_val

        # i_target ~= i_result if same is 1
        same = same.unsqueeze(-1).unsqueeze(-1)  # (B,1,1,1), in {0,1}

        # reconstruction loss (only if same)
        reconstruction_loss = self.l1(i_result * same, i_target * same)

        # adversarial loss (only if not same)
        adversarial_loss = self.adv_loss(d_result, (1 - same), True, for_discriminator=False)

        x = (i_target * 0.5 + 0.5).sub(self.vgg_mean.detach()).div(self.vgg_std.detach())
        y = (i_result * 0.5 + 0.5).sub(self.vgg_mean.detach()).div(self.vgg_std.detach())
        z = (i_cycle * 0.5 + 0.5).sub(self.vgg_mean.detach()).div(self.vgg_std.detach())

        # mask loss (not used)

        # segmentation region loss (same and not same)
        with torch.no_grad():
            t_hair_mask = self.get_any_mask(x, par=[17], normalized=True)  # (B,1,H,W), in [0,1], 1:hair
        reconstruction_loss += self.l1(
            t_hair_mask * i_result, (t_hair_mask * i_target).detach()
        )

        # lpips loss (only if same) & cycle loss (only if not same)
        lpips_loss = 0.0
        cycle_loss = self.l1(i_target, i_cycle)
        if weights_dict["rec"] != 0:
            vgg19_features = self.vgg_model(torch.cat([x, y, z], dim=0))
            for ly, loss_weight in zip(
                ["conv_1_2", "conv_2_2", "conv_3_2", "conv_4_2", "conv_5_2"],
                self.vgg_loss_weights,
            ):
                x_feature, y_feature, z_feature = vgg19_features[ly].chunk(3)
                lpips_loss += self.l2(x_feature.detach() * same, y_feature * same) * loss_weight
                cycle_loss += self.l1(x_feature.detach() * (1 - same), z_feature * (1 - same)) * loss_weight

        # contextual loss (only if not same)
        cx_loss = 0.0
        if weights_dict["cx"] != 0:
            vgg19_features = self.vgg_model(torch.cat([x, y], dim=0))
            for ly, loss_weight in zip(
                ["conv_1_2", "conv_2_2", "conv_3_2", "conv_4_2", "conv_5_2"],
                self.vgg_loss_weights,
            ):
                if ly == "conv_1_2" or ly == "conv_2_2":
                    x_resized = F.interpolate(vgg19_features[ly], [32, 32])
                    x_feature, y_feature = x_resized.chunk(2)
                    cx_loss += self.cxloss(x_feature.detach() * (1 - same), y_feature * (1 - same)) * (1.0 / 8.0)
                else:
                    x_resized = vgg19_features[ly]
                    x_feature, y_feature = x_resized.chunk(2)
                    cx_loss += self.cxloss(x_feature.detach() * (1 - same), y_feature * (1 - same)) * loss_weight
            # cx loss may be NAN if same == 1 (there is no same == 0),
            # this error usually occurs when batch size is 1.
            if int(same.sum()) == same.shape[0]:
                cx_loss = 0.0

        # total realism loss
        realism_loss = (
            adversarial_loss
            + reconstruction_loss * weights_dict["rec"]
            + cycle_loss * weights_dict["cycle"]
            + lpips_loss * weights_dict["lpips"]
            + cx_loss * weights_dict["cx"]
        )

        return realism_loss, {
            "reconstruction_loss": reconstruction_loss,
            "cycle_loss": cycle_loss,
            "lpips_loss": lpips_loss,
            "adversarial_loss": adversarial_loss,
            "cx_loss": cx_loss,
            "realism_loss": realism_loss,
        }

    def get_mask(self, img, normalized=False):
        # [0, 'background', 1 'skin', 2 'l_brow', 3 'r_brow', 4 'l_eye', 5 'r_eye',
        # 6 'eye_g', 7 'l_ear', 8 'r_ear', 9 'ear_r',  10 'nose', 11 'mouth', 12 'u_lip',
        # 13 'l_lip', 14 'neck', 15 'neck_l', 16 'cloth', 17 'hair', 18 'hat']
        ori_size = img.size()[-1]
        with torch.no_grad():
            img = F.interpolate(img, size=512, mode="nearest", )
            if not normalized:
                img = img * 0.5 + 0.5
                img = img.sub(self.vgg_mean.detach()).div(self.vgg_std.detach())
            out = self.bisenet(img)[0]
            parsing = out.softmax(1).argmax(1)
        mask = torch.zeros_like(parsing)
        mask = mask + ((parsing == 1).float())
        for i in range(2, 16):
            mask = mask + ((parsing == i).float())
        mask = mask.unsqueeze(1)
        mask = F.interpolate(mask, size=ori_size, mode="bilinear", align_corners=True)
        return mask

    def get_face_mask(self, img, normalized=False):
        # [0, 'background', 1 'skin', 2 'l_brow', 3 'r_brow', 4 'l_eye', 5 'r_eye',
        # 6 'eye_g', 7 'l_ear', 8 'r_ear', 9 'ear_r',  10 'nose', 11 'mouth', 12 'u_lip',
        # 13 'l_lip', 14 'neck', 15 'neck_l', 16 'cloth', 17 'hair', 18 'hat']
        ori_size = img.size()[-1]
        with torch.no_grad():
            img = F.interpolate(img, size=512, mode="nearest", )
            if not normalized:
                img = img * 0.5 + 0.5
                img = img.sub(self.vgg_mean.detach()).div(self.vgg_std.detach())
            out = self.bisenet(img)[0]
            parsing = out.softmax(1).argmax(1)
        mask = torch.zeros_like(parsing)
        mask = mask + ((parsing == 1).float())
        for i in range(2, 14):
            mask = mask + ((parsing == i).float())
        mask = mask.unsqueeze(1)
        mask = F.interpolate(mask, size=ori_size, mode="bilinear", align_corners=True)
        return mask

    def get_any_mask(self, img, par, normalized=False):
        # [0, 'background', 1 'skin', 2 'l_brow', 3 'r_brow', 4 'l_eye', 5 'r_eye',
        # 6 'eye_g', 7 'l_ear', 8 'r_ear', 9 'ear_r',  10 'nose', 11 'mouth', 12 'u_lip',
        # 13 'l_lip', 14 'neck', 15 'neck_l', 16 'cloth', 17 'hair', 18 'hat']
        ori_size = img.size()[-1]
        with torch.no_grad():
            img = F.interpolate(img, size=512, mode="nearest", )
            if not normalized:
                img = img * 0.5 + 0.5
                img = img.sub(self.vgg_mean.detach()).div(self.vgg_std.detach())
            out = self.bisenet(img)[0]
            parsing = out.softmax(1).argmax(1)
        mask = torch.zeros_like(parsing)
        for p in par:
            mask = mask + ((parsing == p).float())
        mask = mask.unsqueeze(1)
        mask = F.interpolate(mask, size=ori_size, mode="bilinear", align_corners=True)
        return mask

    def get_eye_mouth_mask(self, img, normalized=False):
        # [0, 'background', 1 'skin', 2 'l_brow', 3 'r_brow', 4 'l_eye', 5 'r_eye',
        # 6 'eye_g', 7 'l_ear', 8 'r_ear', 9 'ear_r',  10 'nose', 11 'mouth', 12 'u_lip',
        # 13 'l_lip', 14 'neck', 15 'neck_l', 16 'cloth', 17 'hair', 18 'hat']
        ori_size = img.size()[-1]
        with torch.no_grad():
            img = F.interpolate(img, size=512, mode="nearest", )
            if not normalized:
                img = img * 0.5 + 0.5
                img = img.sub(self.vgg_mean.detach()).div(self.vgg_std.detach())
            out = self.bisenet(img)[0]
            parsing = out.softmax(1).argmax(1)
        mask = torch.zeros_like(parsing)
        mask = mask + ((parsing == 4).float())
        mask = mask + ((parsing == 5).float())
        mask = mask + ((parsing == 6).float())
        mask = mask + ((parsing == 11).float())
        mask = mask + ((parsing == 12).float())
        mask = mask + ((parsing == 13).float())

        mask = mask.unsqueeze(1)
        mask = F.interpolate(mask, size=ori_size, mode="bilinear", align_corners=True)
        return mask

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

    def get_eye_brow_mask(self, img, normalized=False):
        # [0, 'background', 1 'skin', 2 'l_brow', 3 'r_brow', 4 'l_eye', 5 'r_eye',
        # 6 'eye_g', 7 'l_ear', 8 'r_ear', 9 'ear_r',  10 'nose', 11 'mouth', 12 'u_lip',
        # 13 'l_lip', 14 'neck', 15 'neck_l', 16 'cloth', 17 'hair', 18 'hat']
        ori_size = img.size()[-1]
        with torch.no_grad():
            img = F.interpolate(img, size=512, mode="nearest", )
            if not normalized:
                img = img * 0.5 + 0.5
                img = img.sub(self.vgg_mean.detach()).div(self.vgg_std.detach())
            out = self.bisenet(img)[0]
            parsing = out.softmax(1).argmax(1)
        mask = torch.zeros_like(parsing)
        mask = mask + ((parsing == 2).float())
        mask = mask + ((parsing == 3).float())
        mask = mask.unsqueeze(1)
        mask = F.interpolate(mask, size=ori_size, mode="bilinear", align_corners=True)
        return mask


class TripletRealismLoss(RealismLoss):
    def forward(self, i_source, i_target, i_result, i_cycle, d_result, same,
                i_refer, type_code,
                ):
        """

        :param i_source: not used
        :param i_target: target image
        :param i_result: swapped face
        :param i_cycle: cycle swapped face
        :param d_result: discriminator output of i_result
        :param same: if i_source == i_target, i_target will be the reference image
        :param i_refer: triplet reference image
        :param type_code: triplet type code (0:no reference; 1,2,3,4,5:exists reference)
        :return:

        TypeCode Definition
        --------
        >>> type_code
        0: s_v,     t_v     | None
        1: t_v,     t_v     | t_v
        ------------------------------------------------------------------------
        2: s_r,     t_r     | st_r
        3: t_r,     s_r     | ts_r
        4: st_r,    ts_r    | s_r
        5: ts_r,    st_r    | t_r
        """
        assert i_refer is not None and type_code is not None, 'triplet loss input cannot be none!'

        # read rec_code, e.g. 010000 or 010011
        weights_dict = self.weights_dict_train
        rec_code = weights_dict.get('rec_code').split(',')
        rec_code = [int(x) for x in rec_code]
        hair_code = [1, 1, 1, 1, 0, 0]
        mouth_code = [0, 1, 1, 1, 1, 1]

        type_code_hair = torch.zeros_like(type_code, dtype=type_code.dtype)
        type_code_mouth = torch.zeros_like(type_code, dtype=type_code.dtype)
        assert len(rec_code) == 6
        for idx in range(6):
            type_code[type_code == idx] = rec_code[idx]
            type_code_hair[type_code == idx] = hair_code[idx]
            type_code_mouth[type_code == idx] = mouth_code[idx]
        type_code = type_code.unsqueeze(-1).unsqueeze(-1)  # is 1 only when type_code in where rec_code = 1
        type_code_hair = type_code_hair.unsqueeze(-1).unsqueeze(-1)  # is 1 only when type_code in [0,1,2,3]
        type_code_mouth = type_code_mouth.unsqueeze(-1).unsqueeze(-1)  # is 1 only when type_code in [1,2,3,4,5]

        # i_target ~= i_result if same is 1
        same = same.unsqueeze(-1).unsqueeze(-1)  # (B,1,1,1), in {0,1}

        # reconstruction loss (only if same)
        reconstruction_loss = self.l1(i_result * same, i_target * same)  # default rec loss
        reconstruction_loss += 0.05 * self.l1(i_result * type_code, i_refer * type_code)  # triplet reconstruction

        # adversarial loss (only if not same)
        adversarial_loss = self.adv_loss(d_result, (1 - same), True, for_discriminator=False)

        x = (i_target * 0.5 + 0.5).sub(self.vgg_mean.detach()).div(self.vgg_std.detach())  # target
        y = (i_result * 0.5 + 0.5).sub(self.vgg_mean.detach()).div(self.vgg_std.detach())  # result
        z = (i_cycle * 0.5 + 0.5).sub(self.vgg_mean.detach()).div(self.vgg_std.detach())  # cycle
        r = (i_refer * 0.5 + 0.5).sub(self.vgg_mean.detach()).div(self.vgg_std.detach())  # reference

        # mask loss (not used)

        # segmentation hair loss (same and not same)
        with torch.no_grad():
            t_hair_mask = self.get_any_mask(x, par=[17], normalized=True)  # (B,1,H,W), in [0,1], 1:hair
        reconstruction_loss += self.l1(
            t_hair_mask * i_result * type_code_hair, (t_hair_mask * i_target * type_code_hair).detach()
        ) * 2

        # segmentation mouth loss (same and not same)
        with torch.no_grad():
            r_mouth_mask = self.get_any_mask(r, par=[11, 12, 13], normalized=True)  # (B,1,H,W), in [0,1], reference
        y_mouth_mask = self.get_any_mask(y, par=[11, 12, 13], normalized=True)  # (B,1,H,W), in [0,1], result
        reconstruction_loss += self.l1(
            r_mouth_mask * i_refer * type_code_mouth, y_mouth_mask * i_result * type_code_mouth
        ) * 0.05

        # lpips loss (only if same) & cycle loss (only if not same)
        lpips_loss = 0.0
        cycle_loss = self.l1(i_target, i_cycle)
        vgg19_features = self.vgg_model(torch.cat([x, y, z, r], dim=0))
        for ly, loss_weight in zip(
            ["conv_1_2", "conv_2_2", "conv_3_2", "conv_4_2", "conv_5_2"],
            self.vgg_loss_weights,
        ):
            x_feature, y_feature, z_feature, r_feature = vgg19_features[ly].chunk(4)
            lpips_loss += self.l2(x_feature.detach() * same, y_feature * same) * loss_weight
            cycle_loss += self.l1(x_feature.detach() * (1 - same), z_feature * (1 - same)) * loss_weight
            reconstruction_loss += 0.5 * self.l1(y_feature.detach() * type_code, r_feature * type_code) * loss_weight

        # contextual loss (only if not same)
        cx_loss = 0.0
        if weights_dict["cx"] != 0:
            vgg19_features = self.vgg_model(torch.cat([x, y], dim=0))
            for ly, loss_weight in zip(
                ["conv_1_2", "conv_2_2", "conv_3_2", "conv_4_2", "conv_5_2"],
                self.vgg_loss_weights,
            ):
                if ly == "conv_1_2" or ly == "conv_2_2":
                    x_resized = F.interpolate(vgg19_features[ly], [32, 32])
                    x_feature, y_feature = x_resized.chunk(2)
                    cx_loss += self.cxloss(x_feature.detach() * (1 - same), y_feature * (1 - same)) * (1.0 / 8.0)
                else:
                    x_resized = vgg19_features[ly]
                    x_feature, y_feature = x_resized.chunk(2)
                    cx_loss += self.cxloss(x_feature.detach() * (1 - same), y_feature * (1 - same)) * loss_weight
            # cx loss may be NAN if same == 1 (there is no same == 0),
            # this error usually occurs when batch size is 1.
            if int(same.sum()) == same.shape[0]:
                cx_loss = 0.0

        # total realism loss
        realism_loss = (
            adversarial_loss
            + reconstruction_loss * weights_dict["rec"]
            + cycle_loss * weights_dict["cycle"]
            + lpips_loss * weights_dict["lpips"]
            + cx_loss * weights_dict["cx"]
        )

        return realism_loss, {
            "reconstruction_loss": reconstruction_loss,
            "cycle_loss": cycle_loss,
            "lpips_loss": lpips_loss,
            "adversarial_loss": adversarial_loss,
            "cx_loss": cx_loss,
            "realism_loss": realism_loss,
        }


class GLoss(nn.Module):
    def __init__(self, f_id_checkpoint_path, n_layers=3, num_D=3,
                 loss_config: dict = None,
                 mouth_net: torch.nn.Module = None,
                 mouth_crop_param: dict = None,
                 in_size: int = 256,
                 ):
        super(GLoss, self).__init__()
        self.face_model = ParametricFaceModel()
        self.in_size = in_size

        ''' MouthNet '''
        self.mouth_net = mouth_net
        self.mouth_crop_param = mouth_crop_param  # (w1,h1,w2,h2) of PIL.Image
        if mouth_net is None:
            self.en_mouth_net = False
            self.mouth_net = lambda x: None  # return None if MouthNet is empty
            self.mouth_crop_param = (28, 56, 84, 112)

        self.f_id = iresnet100(pretrained=False, fp16=False)
        self.f_id.load_state_dict(torch.load(f_id_checkpoint_path, map_location="cpu"))
        self.f_id.eval()
        self.f_id.requires_grad_(False)

        self.register_buffer(
            name="vgg_mean",
            tensor=torch.tensor([[[0.485]], [[0.456]], [[0.406]]], requires_grad=False),
        )
        self.register_buffer(
            name="vgg_std",
            tensor=torch.tensor([[[0.229]], [[0.224]], [[0.225]]], requires_grad=False),
        )

        ''' realism loss '''
        self.realism_weight_train = loss_config.get('realism').get('train')
        self.realism_weight_val = loss_config.get('realism').get('val')
        self.realism_loss = RealismLoss(
            weights_dict_train=self.realism_weight_train,
            weights_dict_val=self.realism_weight_val,
        )

        ''' realism loss for triplet supervision '''
        self.triplet_weight_train = loss_config.get('triplet')
        self.triplet_realism_loss = TripletRealismLoss(
            weights_dict_train=self.triplet_weight_train,
        )

        ''' sid loss '''
        self.sid_loss = SIDLoss(weights_dict=loss_config.get('sid'))
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
        self.num_D = num_D
        self.n_layers = n_layers
        self.register_buffer(
            name="trans_matrix",
            tensor=torch.tensor(
                [
                    [
                        [1.07695457, -0.03625215, -1.56352194 * (in_size / 256)],
                        [0.03625215, 1.07695457, -5.32134629 * (in_size / 256)],
                    ]
                ],
                requires_grad=False,
            ).float(),
        )

    def forward(self, i_target, i_source, i_result, i_cycle, d_result, target_att, result_att,
                same,
                use_triplet=False,
                i_refer=None,
                type_code=None,
                ):
        """

        :param i_target: target
        :param i_source: source
        :param i_result: result
        :param i_cycle: cycle result
        :param d_result: discriminator features of result
        :param target_att: target attribute
        :param result_att: result attribute
        :param same: (B,1), in {0,1}
        :param use_triplet: use triplet realism loss or not
        :param i_refer:
        :param type_code: in {0,1,2,3,4,5}
        :return:
        """
        if use_triplet:
            assert self.training, 'Please use triplet loss only during training'
        else:
            assert i_refer is None and type_code is None, 'Triplet input error'

        ''' arcface & id loss & att loss & mouth loss '''
        w1, h1, w2, h2 = self.mouth_crop_param
        with torch.no_grad():
            M = self.trans_matrix.repeat(i_source.size()[0], 1, 1)
            i_source = kornia.geometry.transform.warp_affine(i_source, M, (self.in_size, self.in_size))
            i_source_resize = F.interpolate(i_source, size=112, mode="bilinear", align_corners=True)  # to 112x112
            v_id_i_source = F.normalize(self.f_id(i_source_resize), dim=-1, p=2)  # id
            mouth_source = self.mouth_net(i_source_resize[:, :, h1:h2, w1:w2])  # mouth

        _i_result = kornia.geometry.transform.warp_affine(i_result, M, (self.in_size, self.in_size))
        _i_result_resize = F.interpolate(_i_result, size=112, mode="bilinear", align_corners=True)  # to 112x112
        v_id_i_result = F.normalize(self.f_id(_i_result_resize), dim=-1, p=2)  # id
        mouth_result = self.mouth_net(_i_result_resize[:, :, h1:h2, w1:w2])  # mouth

        # id & mouth loss for triplet reconstruction
        reconstruct_triplet_id_loss = 0.
        reconstruct_triplet_mouth_loss = 0.
        if i_refer is not None:
            _i_refer = kornia.geometry.transform.warp_affine(i_refer, M, (self.in_size, self.in_size))
            _i_refer_resize = F.interpolate(_i_refer, size=112, mode="bilinear", align_corners=True)  # to 112x112
            v_id_i_refer = F.normalize(self.f_id(_i_refer_resize), dim=-1, p=2)  # id
            mouth_refer = self.mouth_net(_i_refer_resize[:, :, h1:h2, w1:w2])  # mouth

            triplet_weight_train = self.triplet_weight_train
            rec_code = triplet_weight_train.get('rec_code').split(',')
            rec_code = [int(x) for x in rec_code]
            assert len(rec_code) == 6
            binary_type_code = type_code.clone()
            for idx in range(6):
                binary_type_code[type_code == idx] = rec_code[idx]
            binary_type_code = binary_type_code.unsqueeze(-1).unsqueeze(-1)

            reconstruct_triplet_id_loss = (1 - F.cosine_similarity(v_id_i_result, v_id_i_refer, 1)) * binary_type_code
            reconstruct_triplet_id_loss = reconstruct_triplet_id_loss.mean() * 10.

            if mouth_result is not None and mouth_refer is not None:
                reconstruct_triplet_mouth_loss = (1 - F.cosine_similarity(mouth_result, mouth_refer, 1)) * binary_type_code
                reconstruct_triplet_mouth_loss = reconstruct_triplet_mouth_loss.mean() * 1.0

        sid_loss, sid_loss_dict = self.sid_loss(v_id_i_source, v_id_i_result, target_att, result_att, same, type_code,
                                                mouth_source=mouth_source,
                                                mouth_result=mouth_result)

        ''' realism loss (adv, rec, lpips, cycle, cx) '''
        real_loss = self.realism_loss if not use_triplet else self.triplet_realism_loss
        realism_loss, realism_loss_dict = real_loss(
            i_source, i_target, i_result, i_cycle, d_result, same,
            i_refer, type_code,
        )

        ''' total loss '''
        g_loss = sid_loss + \
                 realism_loss + \
                 reconstruct_triplet_id_loss + \
                 reconstruct_triplet_mouth_loss

        return (
            g_loss,
            {
                **sid_loss_dict,
                **realism_loss_dict,
                "rec_triplet_id": reconstruct_triplet_id_loss,
                "rec_triplet_mouth": reconstruct_triplet_mouth_loss,
                "g_loss": g_loss,
            },
        )


class DLoss(nn.Module):
    def __init__(self):
        super(DLoss, self).__init__()
        self.adv_loss = MultiScaleGANLoss()

    def forward(self, d_gt, d_fake, same, mask_fake=None, mask_real=None):
        same = same.unsqueeze(-1).unsqueeze(-1)
        loss_real = self.adv_loss(d_gt, (1 - same), True, mask=mask_real)
        loss_fake = self.adv_loss(d_fake, (1 - same), False, mask=mask_fake)

        d_loss = loss_real + loss_fake

        return d_loss, {
            "loss_real": loss_real,
            "loss_fake": loss_fake,
            "d_loss": d_loss,
        }


class HearLoss(nn.Module):
    def __init__(self, f_id_checkpoint_path):
        super(HearLoss, self).__init__()

        self.f_id = iresnet100(pretrained=False, fp16=False)
        self.f_id.load_state_dict(torch.load(f_id_checkpoint_path, map_location="cpu"))
        self.f_id.eval()

        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()

        self.register_buffer(
            name="vgg_mean",
            tensor=torch.tensor([[[0.485]], [[0.456]], [[0.406]]], requires_grad=False),
        )
        self.register_buffer(
            name="vgg_std",
            tensor=torch.tensor([[[0.229]], [[0.224]], [[0.225]]], requires_grad=False),
        )

        self.register_buffer(
            name="trans_matrix",
            tensor=torch.tensor(
                [
                    [
                        [1.07695457, -0.03625215, -1.56352194],
                        [0.03625215, 1.07695457, -5.32134629],
                    ]
                ],
                requires_grad=False,
            ).float(),
        )

    def forward(self, x_s, x_t, y_st, y_hat_st, same):
        """

        :param x_s:
        :param x_t:
        :param y_st:
        :param y_hat_st:
        :param same:
        :return:
        """

        ''' 1. id loss '''
        with torch.no_grad():
            M = self.trans_matrix.repeat(x_s.size()[0], 1, 1)
            x_s = kornia.geometry.transform.warp_affine(x_s, M, (256, 256))
            v_id_x_s = F.normalize(
                self.f_id(F.interpolate(x_s, size=112, mode="bilinear", align_corners=True)),
                dim=-1,
                p=2,
            )

        _y_st = kornia.geometry.transform.warp_affine(y_st, M, (256, 256))
        v_id_y_st = F.normalize(
            self.f_id(F.interpolate(_y_st, size=112, mode="bilinear", align_corners=True)),
            dim=-1,
            p=2,
        )

        # only if (same == 0)
        id_loss = (1 - F.cosine_similarity(v_id_y_st, v_id_x_s, 1))  #* (1 - same)
        id_loss = id_loss.mean()

        ''' 2. change loss '''
        # only if (same == 0)
        same = same.unsqueeze(-1).unsqueeze(-1)  # (B,1,1,1), in {0,1}
        chg_loss = self.l1(y_hat_st, y_st)

        ''' 3. reconstruction loss '''
        # only if (same == 1)
        reconstruction_loss = self.mse(y_st * same, x_t * same)

        hear_loss = id_loss + chg_loss + reconstruction_loss

        return hear_loss, {
            "loss_id": id_loss,
            "loss_chg": chg_loss,
            "loss_rec": reconstruction_loss,
            "loss_hear": hear_loss
        }


if __name__ == '__main__':
    from PIL import Image
    import numpy as np
    from torchvision.transforms import transforms
    in_size = 512
    real_loss = RealismLoss(
        weights_dict_train={"rec": 10, "cycle": 0, "lpips": 0, "cx": 0},
        weights_dict_val={"rec": 10, "cycle": 0, "lpips": 0, "cx": 0}
    )
    trans = transforms.Compose([
        transforms.Resize((in_size, in_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ])

    def save_tensor_to_img(tensor: torch.Tensor, path: str):
        tensor = tensor.permute(0, 2, 3, 1)[0]  # in [0,1]
        print(path, type(tensor), tensor.shape, tensor.sum())
        tensor = tensor * 255
        tensor_np = tensor.cpu().numpy().astype(np.uint8)
        if tensor_np.shape[-1] == 1:  # channel dim
            tensor_np = tensor_np.repeat(3, axis=-1)
        tensor_img = Image.fromarray(tensor_np)
        tensor_img.save(path)

    img_path = "./00_target.png"
    i_t = Image.open(img_path).convert('RGB')
    i_t.save('target.jpg')
    i_t = trans(i_t)
    i_t = i_t.unsqueeze(0)
    x = (i_t * 0.5 + 0.5).sub(real_loss.vgg_mean.detach()).div(real_loss.vgg_std.detach())

    # contextual loss
    cx_loss = 0.0
    mask_zero_one = torch.cat(
        [torch.ones(1, 1, 128, in_size), torch.zeros(1, 1, 128, in_size)], dim=2
    ).to(i_t.device)
    with torch.no_grad():
        mask_t = real_loss.get_mask(x, True)
        eye_brow_out_mask_t = real_loss.get_eye_brow_mask(x, True)
        # eye_brow_out_mask_r = real_loss.get_eye_brow_mask(y, True)
        eye_brow_out_mask_t = 1 - eye_brow_out_mask_t
        # eye_brow_out_mask_r = 1 - eye_brow_out_mask_r

        face_mask_t = real_loss.create_masks(real_loss.get_face_mask(x, True), out_pad=40)[0]

    save_tensor_to_img(eye_brow_out_mask_t, 'eye_brow_out_mask_t.jpg')
    save_tensor_to_img(face_mask_t, 'face_mask_t.jpg')
    save_tensor_to_img(real_loss.get_any_mask(x, par=[17], normalized=True), 'get_any_mask_17.jpg')

    mouth_mask = real_loss.get_any_mask(x, par=[11, 12, 13], normalized=True)
    save_tensor_to_img(mouth_mask, 'get_any_mask_11_12_13.jpg')
    save_tensor_to_img((i_t * 0.5 + 0.5) * mouth_mask, 'x_mouth.jpg')
