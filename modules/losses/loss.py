import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import vgg19

from modules.third_party.arcface.iresnet import iresnet100
from modules.third_party.tdmm.resnet import ReconNetWrapper
from modules.third_party.tdmm.bfm import ParametricFaceModel
from modules.third_party.vgg.modules.vgg import VGG_Model
from modules.third_party.bisenet.bisenet import BiSeNet


# def color_transfer(source, target):
#     source = kornia.color.rgb_to_lab(source)
#     target = kornia.color.rgb_to_lab(target)

#     mean_tar = torch.mean(target, [2, 3], keepdim=True)
#     std_tar = torch.std(target, [2, 3], keepdim=True)

#     mean_src = torch.mean(source, [2, 3], keepdim=True)
#     std_src = torch.std(source, [2, 3], keepdim=True)

#     target = torch.clamp(
#         (std_src / std_tar) * (target - mean_tar) + mean_src, 0.0, 255.0
#     )
#     transfer = kornia.color.lab_to_rgb(target)

#     return transfer


class CXLoss(nn.Module):
    def __init__(self, sigma=0.1, b=1.0, similarity="consine"):
        super(CXLoss, self).__init__()
        self.similarity = similarity
        self.sigma = sigma
        self.b = b

    def center_by_T(self, featureI, featureT):
        # Calculate mean channel vector for feature map.
        meanT = (
            featureT.mean(0, keepdim=True).mean(2, keepdim=True).mean(3, keepdim=True)
        )
        return featureI - meanT, featureT - meanT

    def l2_normalize_channelwise(self, features):
        # Normalize on channel dimension (axis=1)
        norms = features.norm(p=2, dim=1, keepdim=True)
        features = features.div(norms)
        return features

    def patch_decomposition(self, features):
        N, C, H, W = features.shape
        assert N == 1
        P = H * W
        # NCHW --> 1x1xCxHW --> HWxCx1x1
        patches = features.view(1, 1, C, P).permute((3, 2, 0, 1))
        return patches

    def calc_relative_distances(self, raw_dist, axis=1):
        epsilon = 1e-5
        div = torch.min(raw_dist, dim=axis, keepdim=True)[0]
        relative_dist = raw_dist / (div + epsilon)
        return relative_dist

    def calc_CX(self, dist, axis=1):
        W = torch.exp((self.b - dist) / self.sigma)
        W_sum = W.sum(dim=axis, keepdim=True)
        return W.div(W_sum)

    def forward(self, featureT, featureI):
        """
        :param featureT: target
        :param featureI: inference
        :return:
        """
        # NCHW
        featureI, featureT = self.center_by_T(featureI, featureT)

        featureI = self.l2_normalize_channelwise(featureI)
        featureT = self.l2_normalize_channelwise(featureT)

        dist = []
        N = featureT.size()[0]
        for i in range(N):
            # NCHW
            featureT_i = featureT[i, :, :, :].unsqueeze(0)
            # NCHW
            featureI_i = featureI[i, :, :, :].unsqueeze(0)
            featureT_patch = self.patch_decomposition(featureT_i)
            # Calculate cosine similarity
            # See the torch document for functional.conv2d
            dist_i = F.conv2d(featureI_i, featureT_patch)
            dist.append(dist_i)

        # NCHW
        dist = torch.cat(dist, dim=0)
        raw_dist = (1.0 - dist) / 2.0
        relative_dist = self.calc_relative_distances(raw_dist)
        CX = self.calc_CX(relative_dist)
        CX = CX.max(dim=3)[0].max(dim=2)[0]
        CX = CX.mean(1)
        CX = -torch.log(CX)
        CX = torch.mean(CX)
        return CX


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

    def loss(self, input, target_is_real, for_discriminator=True):
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
                    loss = -torch.mean(minval)
                else:
                    minval = torch.min(-input - 1, self.get_zero_tensor(input))
                    loss = -torch.mean(minval)
            else:
                assert (
                    target_is_real
                ), "The generator's hinge loss must be aiming for real"
                loss = -torch.mean(input)
            return loss
        else:
            # wgan
            if target_is_real:
                return -input.mean()
            else:
                return input.mean()

    def __call__(self, input, target_is_real, for_discriminator=True):
        # computing loss is a bit complicated because |input| may not be
        # a tensor, but list of tensors in case of multiscale discriminator
        if isinstance(input, list):
            loss = 0
            for pred_i in input:
                if isinstance(pred_i, list):
                    pred_i = pred_i[-1]
                loss_tensor = self.loss(pred_i, target_is_real, for_discriminator)
                bs = 1 if len(loss_tensor.size()) == 0 else loss_tensor.size(0)
                new_loss = torch.mean(loss_tensor.view(bs, -1), dim=1)
                loss += new_loss
            return loss / len(input)
        else:
            return self.loss(input, target_is_real, for_discriminator)


class SIDLoss(nn.Module):
    def __init__(self):
        super(SIDLoss, self).__init__()
        self.l1 = nn.L1Loss()

    def forward(
        self, q_fuse, q_r, q_low, v_id_I_s, v_id_I_r, v_id_I_low, v_mask_id_i_s
    ):
        shape_loss = (
            torch.mean(torch.abs(q_fuse - q_r), dim=[1, 2]).clamp(0.0, 10.0).mean()
            + torch.mean(torch.abs(q_fuse - q_low), dim=[1, 2]).clamp(0.0, 10.0).mean()
        )
        id_loss = -(
            F.cosine_similarity(v_id_I_s, v_id_I_r, 1)
            + F.cosine_similarity(v_id_I_s, v_id_I_low, 1)
            + F.cosine_similarity(v_id_I_s, v_mask_id_i_s, 1)
        ).mean()

        sid_loss = 15 * id_loss + 0.5 * shape_loss

        return sid_loss, {
            "shape_loss": shape_loss,
            "id_loss": id_loss,
            "sid_loss": sid_loss,
        }


class RealismLoss(nn.Module):
    def __init__(self, bisenet):
        super(RealismLoss, self).__init__()
        # self.loss_fn_vgg = lpips.LPIPS(net="vgg")
        self.l1 = nn.L1Loss()
        self.vgg_layer = [
            "conv_3_4",
            "conv_5_2",
            "conv_4_2",
            "conv_3_2",
            "conv_2_2",
            "conv_1_2",
            "conv_5_2",
        ]
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
        self.bisenet = bisenet
        self.cxloss_style = CXLoss(sigma=0.1)

        self.adv_loss = MultiScaleGANLoss()

    def get_parsing(self, img):
        with torch.no_grad():
            out = self.bisenet(img)[0]
            parsing = torch.argmax(out, 1)

        skin_mask = torch.zeros_like(parsing)
        skin_mask = skin_mask + ((parsing == 1).float())
        for i in range(6, 11):
            skin_mask = skin_mask + ((parsing == i).float())
        skin_mask = skin_mask.unsqueeze(1)

        return skin_mask

    def get_mask(self, img):
        # [0, 'background', 1 'skin', 2 'l_brow', 3 'r_brow', 4 'l_eye', 5 'r_eye', 6 'eye_g', 7 'l_ear', 8 'r_ear', 9 'ear_r',  10 'nose', 11 'mouth', 12 'u_lip', 13 'l_lip', 14 'neck', 15 'neck_l', 16 'cloth', 17 'hair', 18 'hat']
        with torch.no_grad():
            out = self.bisenet(img)[0]
            parsing = torch.argmax(out, 1)

        mask = torch.zeros_like(parsing)
        mask = mask + ((parsing == 1).float())
        for i in range(2, 14):
            if i == 7 or i == 8 or i == 9:
                continue
            mask = mask + ((parsing == i).float())
        mask = mask.unsqueeze(1)
        return mask

    def get_whole_mask(self, img):
        # [0, 'background', 1 'skin', 2 'l_brow', 3 'r_brow', 4 'l_eye', 5 'r_eye', 6 'eye_g', 7 'l_ear', 8 'r_ear', 9 'ear_r',  10 'nose', 11 'mouth', 12 'u_lip', 13 'l_lip', 14 'neck', 15 'neck_l', 16 'cloth', 17 'hair', 18 'hat']
        with torch.no_grad():
            out = self.bisenet(img)[0]
            parsing = torch.argmax(out, 1)

        mask = torch.zeros_like(parsing)
        mask = mask + ((parsing == 1).float())
        for i in range(2, 16):
            mask = mask + ((parsing == i).float())
        mask = mask.unsqueeze(1)
        return mask

    def forward(self, m_tar, m_r, i_t, i_s, i_r, i_low, i_cycle, d_r, same):
        same = same.unsqueeze(-1).unsqueeze(-1)
        m_tar = self.get_mask(
            i_t.sub(self.vgg_mean.detach()).div(self.vgg_std.detach())
        )
        segmentation_loss = self.l1(m_tar, m_r)
        reconstruction_loss = self.l1(i_r * same, i_t * same) + self.l1(
            i_low * same, F.interpolate(i_t, scale_factor=0.25, mode="bilinear", align_corners=True) * same
        )
        cycle_loss = self.l1(i_t, i_cycle)
        # lpips_loss = (
        #     self.loss_fn_vgg(i_t * same, i_r * same).mean()
        #     + self.loss_fn_vgg(i_t, i_cycle).mean()
        # )
        adversarial_loss = self.adv_loss(d_r, True, for_discriminator=False)

        # x = color_transfer(i_t, i_s)
        x = i_t.sub(self.vgg_mean.detach()).div(self.vgg_std.detach())
        y = i_r.sub(self.vgg_mean.detach()).div(self.vgg_std.detach())
        z = i_cycle.sub(self.vgg_mean.detach()).div(self.vgg_std.detach())
        x_mask = x * self.get_whole_mask(x)
        y_mask = y * self.get_whole_mask(y)

        cxloss = 0.0
        x_feature = self.vgg_model(x_mask)
        y_feature = self.vgg_model(y_mask)

        # contextual loss, face inpainter
        # x_ = x_feature["conv_4_2"]
        # y_ = y_feature["conv_4_2"]
        # cxloss += self.cxloss_style(y_, x_)
        # x_ = x_feature["conv_3_2"]
        # y_ = y_feature["conv_3_2"]
        # cxloss += self.cxloss_style(y_, x_)
        # contextual loss, superresolution
        # x_ = x_feature["conv_3_4"]
        # y_ = y_feature["conv_3_4"]
        # cxloss += self.cxloss_style(y_, x_)

        # vgg19 loss
        lpips_loss = 0.0
        vgg19_features = self.vgg_model(torch.cat([x * same, y * same], dim=0))
        for ly, loss_weight in zip(
            ["conv_1_2", "conv_2_2", "conv_3_2", "conv_4_2", "conv_5_2"],
            self.vgg_loss_weights,
        ):
            x_feature, y_feature = vgg19_features[ly].chunk(2)
            lpips_loss += self.l1(x_feature, y_feature) * loss_weight

        vgg19_features = self.vgg_model(torch.cat([x, z], dim=0))
        for ly, loss_weight in zip(
            ["conv_1_2", "conv_2_2", "conv_3_2", "conv_4_2", "conv_5_2"],
            self.vgg_loss_weights,
        ):
            x_feature, y_feature = vgg19_features[ly].chunk(2)
            lpips_loss += self.l1(x_feature, y_feature) * loss_weight

        realism_loss = (
            adversarial_loss
            + 100 * segmentation_loss
            + 20 * reconstruction_loss
            + 5 * cycle_loss
            + 5 * lpips_loss
        )

        return realism_loss, {
            "segmentation_loss": segmentation_loss,
            "reconstruction_loss": reconstruction_loss,
            "cycle_loss": cycle_loss,
            "lpips_loss": lpips_loss,
            "adversarial_loss": adversarial_loss,
            "realism_loss": realism_loss,
        }


class GLoss(nn.Module):
    def __init__(
        self, f_3d_checkpoint_path, f_id_checkpoint_path, realism_config, sid_config
    ):
        super(GLoss, self).__init__()
        self.f_3d = ReconNetWrapper(net_recon="resnet50", use_last_fc=False)
        self.f_3d.load_state_dict(
            torch.load(f_3d_checkpoint_path, map_location="cpu")["net_recon"]
        )
        self.f_3d.eval()
        self.face_model = ParametricFaceModel()

        self.f_id = iresnet100(pretrained=False, fp16=False)
        self.f_id.load_state_dict(torch.load(f_id_checkpoint_path, map_location="cpu"))
        self.f_id.eval()

        self.bisenet = BiSeNet(n_classes=19)
        self.bisenet.load_state_dict(
            torch.load(
                "/gavin/datasets/hanbang/79999_iter.pth",
                map_location="cpu",
            )
        )
        self.bisenet.eval()

        self.register_buffer(
            name="vgg_mean",
            tensor=torch.tensor([[[0.485]], [[0.456]], [[0.406]]], requires_grad=False),
        )
        self.register_buffer(
            name="vgg_std",
            tensor=torch.tensor([[[0.229]], [[0.224]], [[0.225]]], requires_grad=False),
        )

        self.realism_loss = RealismLoss(bisenet=self.bisenet, **realism_config)
        self.sid_loss = SIDLoss(**sid_config)

    def get_mask(self, img):
        with torch.no_grad():
            out = self.bisenet(img)[0]
            parsing = torch.argmax(out, 1)

        mask = torch.zeros_like(parsing)
        mask = mask + ((parsing == 1).float())
        for i in range(2, 16):
            mask = mask + ((parsing == i).float())
        mask = mask.unsqueeze(1)
        return mask

    def forward(self, i_s, i_t, i_r, i_low, i_cycle, m_tar, m_r, d_r, same):
        # region 3DMM
        with torch.no_grad():
            c_s = self.f_3d(F.interpolate(i_s, size=224, mode="bilinear", align_corners=True))
            c_t = self.f_3d(F.interpolate(i_t, size=224, mode="bilinear", align_corners=True))
        c_r = self.f_3d(F.interpolate(i_r, size=224, mode="bilinear", align_corners=True))
        c_low = self.f_3d(F.interpolate(i_low, size=224, mode="bilinear", align_corners=True))

        """
        (B, 257)
        80 # id layer
        64 # exp layer
        80 # tex layer
        3  # angle layer
        27 # gamma layer
        2  # tx, ty
        1  # tz
        """
        with torch.no_grad():
            c_fuse = torch.cat((c_s[:, :80], c_t[:, 80:]), dim=1)
            _, _, _, q_fuse = self.face_model.compute_for_render(c_fuse)
            q_fuse = q_fuse  # [:, :17]

        _, _, _, q_r = self.face_model.compute_for_render(c_r)
        _, _, _, q_low = self.face_model.compute_for_render(c_low)
        q_low = q_low  # [:, :17]
        q_r = q_r  # [:, :17]
        # endregion

        mask_source = self.get_mask(
            i_s.sub(self.vgg_mean.detach()).div(self.vgg_std.detach())
        )
        masked_source = mask_source * i_s

        # region arcface
        with torch.no_grad():
            v_id_i_s = (
                F.normalize(
                    self.f_id(
                        F.interpolate((i_s - 0.5) / 0.5, size=112, mode="bilinear", align_corners=True)
                    ),
                    dim=-1,
                    p=2,
                )
                * (1.0 - same).view(-1, 1)
            )
            v_mask_id_i_s = (
                F.normalize(
                    self.f_id(
                        F.interpolate(
                            (masked_source - 0.5) / 0.5, size=112, mode="bilinear", align_corners=True
                        )
                    ),
                    dim=-1,
                    p=2,
                )
                * (1.0 - same).view(-1, 1)
            )

        v_id_i_r = (
            F.normalize(
                self.f_id(F.interpolate((i_r - 0.5) / 0.5, size=112, mode="bilinear", align_corners=True)),
                dim=-1,
                p=2,
            )
            * (1.0 - same).view(-1, 1)
        )
        v_id_i_low = (
            F.normalize(
                self.f_id(
                    F.interpolate((i_low - 0.5) / 0.5, size=112, mode="bilinear", align_corners=True)
                ),
                dim=-1,
                p=2,
            )
            * (1.0 - same).view(-1, 1)
        )
        # endregion

        sid_loss, sid_loss_dict = self.sid_loss(
            q_fuse, q_r, q_low, v_id_i_s, v_id_i_r, v_id_i_low, v_mask_id_i_s
        )
        realism_loss, realism_loss_dict = self.realism_loss(
            m_tar, m_r, i_t, i_s, i_r, i_low, i_cycle, d_r, same
        )

        g_loss = sid_loss + realism_loss  # + torch.abs(c_r - c_fuse).mean()

        return (
            g_loss,
            {
                **sid_loss_dict,
                **realism_loss_dict,
                "g_loss": g_loss,
            },
            {
                "m_tar": m_tar,
                "m_r": m_r,
            },
        )


class DLoss(nn.Module):
    def __init__(self):
        super(DLoss, self).__init__()
        self.adv_loss = MultiScaleGANLoss()

    def forward(self, d_gt, d_fake):
        loss_real = self.adv_loss(d_gt, True)
        loss_fake = self.adv_loss(d_fake, False)

        d_loss = loss_real + loss_fake

        return d_loss, {
            "loss_real": loss_real,
            "loss_fake": loss_fake,
            "d_loss": d_loss,
        }
