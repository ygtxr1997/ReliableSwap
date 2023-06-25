import torch
import torch.nn as nn
import torch.nn.functional as F

import kornia

from torchvision.models import vgg19

from modules.third_party.arcface import iresnet100
from modules.third_party.vgg.modules.vgg import VGG_Model


class SIDLoss(nn.Module):
    def __init__(self,
                 weights_dict={"id": 10},
                 ):
        super(SIDLoss, self).__init__()
        self.weights_dict = weights_dict

    def forward(self, source_id, result_id, same, type_code=None,
                source_mouth=None,
                result_mouth=None,
                ):
        if type_code is not None:
            same[type_code == 1] = 0  # also when X_t == X_s

        # only if (same == 0)
        id_loss = (1 - F.cosine_similarity(source_id, result_id, 1)) * (1 - same)
        id_loss = id_loss.mean()

        # only if (same == 0)
        mouth_loss = 0.
        if source_mouth is not None and result_mouth is not None:
            mouth_loss = (1 - F.cosine_similarity(source_mouth, result_mouth, 1)) * (1 - same)
            mouth_loss = mouth_loss.mean()

        sid_loss = id_loss * self.weights_dict["id"] \
                   + mouth_loss * self.weights_dict["mouth"]
        return sid_loss, {"id_loss": id_loss,
                          "mouth_loss": mouth_loss,
                          }


class WeakFeatMatchLoss(nn.Module):
    def __init__(self):
        super(WeakFeatMatchLoss, self).__init__()
        self.l1 = nn.L1Loss()

    def forward(self, d_feat_fake, d_feat_real):
        wfm_loss = self.l1(d_feat_fake["3"], d_feat_real["3"])
        return wfm_loss


class RealismLoss(nn.Module):
    def __init__(self,
                 weights_dict_train: dict = None,
                 weights_dict_val: dict = None,
                 ):
        super(RealismLoss, self).__init__()

        self.l1 = nn.L1Loss()
        self.l2 = nn.MSELoss()

        self.weights_dict_train = weights_dict_train
        self.weights_dict_val = weights_dict_val

        ''' vgg related '''
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
        self.vgg_model.requires_grad_(False)
        self.register_buffer(
            name="vgg_mean",
            tensor=torch.tensor([[[0.485]], [[0.456]], [[0.406]]], requires_grad=False),
        )
        self.register_buffer(
            name="vgg_std",
            tensor=torch.tensor([[[0.229]], [[0.224]], [[0.225]]], requires_grad=False),
        )

        self.wfm = WeakFeatMatchLoss()

    def forward(self,
                i_target,
                i_result,
                d_logit_fake,
                d_feat_fake,
                d_feat_real,
                same,
                i_refer,
                refer_id,
                result_id,
                refer_mouth,
                result_mouth,
                type_code,
                ):
        assert i_refer is None and type_code is None, 'loss input value error!'
        weights_dict = self.weights_dict_train if self.training else self.weights_dict_val

        # i_target ~= i_result if same is 1
        same = same.unsqueeze(-1).unsqueeze(-1)  # (B,1,1,1), in {0,1}

        # reconstruction loss (only if same)
        reconstruction_loss = self.l1(i_result * same, i_target * same)

        # adversarial loss
        adversarial_loss = (-d_logit_fake).mean()

        # weak feature matching loss
        wfm_loss = self.wfm(d_feat_fake, d_feat_real)

        # total realism loss
        realism_loss = (
            adversarial_loss
            + reconstruction_loss * weights_dict["rec"]
            + wfm_loss * weights_dict["wfm"]
        )

        return realism_loss, {
            "reconstruction_loss": reconstruction_loss,
            "adversarial_loss": adversarial_loss,
            "wfm_loss": wfm_loss,
            "realism_loss": realism_loss,
        }


class TripletRealismLoss(RealismLoss):
    def forward(self,
                i_target,
                i_result,
                d_logit_fake,
                d_feat_fake,
                d_feat_real,
                same,
                i_refer,
                refer_id,
                result_id,
                refer_mouth,
                result_mouth,
                type_code,
                ):
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
        reconstruction_loss = self.l1(i_result * same, i_target * same)
        reconstruction_loss += 0.05 * self.l1(i_result * type_code, i_refer * type_code)  # triplet reconstruction

        # adversarial loss
        adversarial_loss = (-d_logit_fake).mean()

        # weak feature matching loss
        wfm_loss = self.wfm(d_feat_fake, d_feat_real)

        # triplet vgg loss
        x = (i_target * 0.5 + 0.5).sub(self.vgg_mean.detach()).div(self.vgg_std.detach())  # target
        y = (i_result * 0.5 + 0.5).sub(self.vgg_mean.detach()).div(self.vgg_std.detach())  # result
        r = (i_refer * 0.5 + 0.5).sub(self.vgg_mean.detach()).div(self.vgg_std.detach())  # reference

        vgg19_features = self.vgg_model(torch.cat([x, y, r], dim=0))
        for ly, loss_weight in zip(
                ["conv_1_2", "conv_2_2", "conv_3_2", "conv_4_2", "conv_5_2"],
                self.vgg_loss_weights,
        ):
            x_feature, y_feature, r_feature = vgg19_features[ly].chunk(3)
            reconstruction_loss += 0.5 * self.l1(y_feature.detach() * type_code,
                                                 r_feature * type_code) * loss_weight  # triplet vgg

        # triplet id loss
        triplet_id_loss = (1 - F.cosine_similarity(result_id, refer_id, 1)) * type_code
        reconstruction_loss += triplet_id_loss.mean()

        # triplet mouth loss
        triplet_mouth_loss = 0.
        if result_mouth is not None and refer_mouth is not None:
            triplet_mouth_loss = (1 - F.cosine_similarity(result_mouth, refer_mouth, 1)) * type_code
            reconstruction_loss += 0.05 * triplet_mouth_loss.mean()

        # total realism loss
        realism_loss = (
                adversarial_loss
                + reconstruction_loss * weights_dict["rec"]
                + wfm_loss * weights_dict["wfm"]
        )

        return realism_loss, {
            "reconstruction_loss": reconstruction_loss,
            "adversarial_loss": adversarial_loss,
            "wfm_loss": wfm_loss,
            "realism_loss": realism_loss,
        }


class GLoss(nn.Module):
    def __init__(self,
                 f_id: nn.Module,
                 loss_config: dict = None,
                 mouth_net: torch.nn.Module = None,
                 mouth_crop_param: dict = None,
                 ):
        super(GLoss, self).__init__()

        ''' MouthNet '''
        self.mouth_net = mouth_net  # maybe None and return None if MouthNet is empty
        self.mouth_crop_param = mouth_crop_param  # (w1,h1,w2,h2) of PIL.Image
        if mouth_net is None:
            self.mouth_net = lambda x: None  # return None if MouthNet is empty
            self.mouth_crop_param = (28, 56, 84, 112)

        self.f_id = f_id

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

    def forward(self,
                i_source,
                i_target,
                i_result,
                d_logit_fake,
                d_feat_fake,
                d_feat_real,
                same,
                use_triplet=False,
                i_refer=None,
                type_code=None,
                ):
        if use_triplet:
            assert self.training, 'Please use triplet loss only during training'
        else:
            assert i_refer is None and type_code is None, 'Triplet input error'

        w1, h1, w2, h2 = self.mouth_crop_param
        with torch.no_grad():
            M = self.trans_matrix.repeat(i_source.size()[0], 1, 1)
            # i_source = kornia.geometry.transform.warp_affine(i_source, M, (256, 256))
            i_source_resize = F.interpolate(i_source, size=112, mode="bilinear", align_corners=True)  # to 112x112
            v_id_i_source = F.normalize(self.f_id(i_source_resize), dim=-1, p=2)  # id
            source_mouth = self.mouth_net(i_source_resize[:, :, h1:h2, w1:w2])  # mouth

        # _i_result = kornia.geometry.transform.warp_affine(i_result, M, (256, 256))
        _i_result = i_result
        _i_result_resize = F.interpolate(_i_result, size=112, mode="bilinear", align_corners=True)  # to 112x112
        v_id_i_result = F.normalize(self.f_id(_i_result_resize), dim=-1, p=2)  # id
        result_mouth = self.mouth_net(_i_result_resize[:, :, h1:h2, w1:w2])  # mouth

        v_id_i_refer = None
        refer_mouth = None
        if i_refer is not None:
            # _i_refer = kornia.geometry.transform.warp_affine(i_refer, M, (256, 256))
            _i_refer = i_refer
            _i_refer_resize = F.interpolate(_i_refer, size=112, mode="bilinear", align_corners=True)  # to 112x112
            v_id_i_refer = F.normalize(self.f_id(_i_refer_resize), dim=-1, p=2)  # id
            refer_mouth = self.mouth_net(_i_refer_resize[:, :, h1:h2, w1:w2])  # mouth

        ''' id loss '''
        sid_loss, sid_loss_dict = self.sid_loss(
            source_id=v_id_i_source,
            result_id=v_id_i_result,
            same=same,
            type_code=type_code,
            source_mouth=source_mouth,
            result_mouth=result_mouth,
        )

        ''' realism loss (adv, rec, wfm, triplet) '''
        real_loss = self.realism_loss if not use_triplet else self.triplet_realism_loss
        realism_loss, realism_loss_dict = real_loss(
            i_target=i_target,
            i_result=i_result,
            d_logit_fake=d_logit_fake,
            d_feat_fake=d_feat_fake,
            d_feat_real=d_feat_real,
            same=same,
            i_refer=i_refer,
            refer_id=v_id_i_refer,
            result_id=v_id_i_result,
            refer_mouth=refer_mouth,
            result_mouth=result_mouth,
            type_code=type_code,
        )

        ''' total loss '''
        g_loss = sid_loss + realism_loss

        return (
            g_loss,
            {
                **sid_loss_dict,
                **realism_loss_dict,
                "g_loss": g_loss,
            }
        )


class DLoss(nn.Module):
    def __init__(self):
        super(DLoss, self).__init__()

    def forward(self, d_logit_fake, d_logit_real):
        loss_Dfake = (F.relu(torch.ones_like(d_logit_fake) + d_logit_fake)).mean()
        loss_Dreal = (F.relu(torch.ones_like(d_logit_real) - d_logit_real)).mean()
        loss_D = loss_Dreal + loss_Dfake
        return loss_D, {
            "loss_real": loss_Dreal,
            "loss_fake": loss_Dfake,
            "d_loss": loss_D,
        }


