import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import kornia
import warnings

from modules.layers.faceshifter.layers import AEI_Net
from modules.layers.faceshifter.hear_layers import Hear_Net
from modules.third_party.arcface import iresnet100, MouthNet

make_abs_path = lambda fn: os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), fn))


class FSGenerator(nn.Module):
    def __init__(self,
                 id_ckpt: str = None,
                 id_dim: int = 512,
                 mouth_net_param: dict = None,
                 in_size: int = 256,
                 finetune: bool = False,
                 downup: bool = False,
                 ):
        super(FSGenerator, self).__init__()

        ''' MouthNet '''
        self.use_mouth_net = mouth_net_param.get('use')
        self.mouth_feat_dim = 0
        self.mouth_net = None
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
            print("[FaceShifter Generator] MouthNet loaded from %s" % mouth_weight_path)
            self.mouth_net.eval()
            self.mouth_net.requires_grad_(False)

        self.G = AEI_Net(c_id=id_dim + self.mouth_feat_dim, finetune=finetune, downup=downup)
        self.iresnet = iresnet100()
        if not id_ckpt is None:
            self.iresnet.load_state_dict(torch.load(id_ckpt, "cpu"))
        else:
            warnings.warn("Face ID backbone [%s] not found!" % id_ckpt)
            raise FileNotFoundError("Face ID backbone [%s] not found!" % id_ckpt)
        self.iresnet.eval()
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
        self.in_size = in_size

        self.iresnet.requires_grad_(False)

    def forward(self, source, target, infer=False):
        with torch.no_grad():
            ''' 1. get id '''
            if infer:
                resize_input = F.interpolate(source, size=112, mode="bilinear", align_corners=True)
                id_vector = F.normalize(self.iresnet(resize_input), dim=-1, p=2)
            else:
                M = self.trans_matrix.repeat(source.size()[0], 1, 1)
                source = kornia.geometry.transform.warp_affine(source, M, (self.in_size, self.in_size))

                # import cv2
                # from tricks import Trick
                # cv2.imwrite('warpped_source.png', Trick.tensor_to_arr(source)[0, :, :, ::-1])

                resize_input = F.interpolate(source, size=112, mode="bilinear", align_corners=True)
                id_vector = F.normalize(self.iresnet(resize_input), dim=-1, p=2)

            ''' 2. get mouth feature '''
            if self.use_mouth_net:
                w1, h1, w2, h2 = self.mouth_crop_param
                mouth_input = resize_input[:, :, h1:h2, w1:w2]  # 112->mouth
                mouth_feat = self.mouth_net(mouth_input)
                id_vector = torch.cat([id_vector, mouth_feat], dim=-1)  # (B,dim_id+dim_mouth)

        x, att = self.G(target, id_vector)
        return x, id_vector, att

    def get_recon(self):
        return self.G.get_recon_tensor()

    def get_att(self, x):
        return self.G.get_attr(x)


class FSHearNet(nn.Module):
    def __init__(self, aei_path: str):
        super(FSHearNet, self).__init__()
        ''' Stage I. AEI_Net '''
        self.aei = FSGenerator(
            id_ckpt=make_abs_path("../../modules/third_party/arcface/weights/ms1mv3_arcface_r100_fp16/backbone.pth")
        ).requires_grad_(False)
        print('Loading pre-trained AEI-Net from %s...' % aei_path)
        self._load_pretrained_aei(aei_path)
        print('Loaded.')

        ''' Stage II. HEAR_Net '''
        self.hear = Hear_Net()

    def _load_pretrained_aei(self, path: str):
        if '.ckpt' in path:
            from trainer.faceshifter.extract_ckpt import extract_generator
            pth_folder = make_abs_path('../../trainer/faceshifter/extracted_ckpt')
            pth_name = 'hear_tmp.pth'
            assert '.pth' in pth_name
            state_dict = extract_generator(load_path=path, path=os.path.join(pth_folder, pth_name))
            self.aei.load_state_dict(state_dict, strict=False)
            self.aei.eval()
        elif '.pth' in path:
            self.aei.load_state_dict(torch.load(path, "cpu"), strict=False)
            self.aei.eval()
        else:
            raise FileNotFoundError('%s (.ckpt or .pth) not found.' % path)

    def forward(self, source, target):
        with torch.no_grad():
            y_hat_st, _, _ = self.aei(source, target, infer=True)
            y_hat_tt, _, _ = self.aei(target, target, infer=True)
            delta_y_t = target - y_hat_tt
            y_cat = torch.cat([y_hat_st, delta_y_t], dim=1)  # (B,6,256,256)

        y_st = self.hear(y_cat)

        return y_st, y_hat_st  # both (B,3,256,256)


if __name__ == '__main__':

    source = torch.randn(8, 3, 512, 512)
    target = torch.randn(8, 3, 512, 512)
    net = FSGenerator(
        id_ckpt="/apdcephfs_cq2/share_1290939/gavinyuan/code/FaceShifter/faceswap/faceswap/checkpoints/"
                "face_id/ms1mv3_arcface_r100_fp16_backbone.pth",
        mouth_net_param={
            'use': False
        }
    )
    result, _, _ = net(source, target)
    print('result:', result.shape)

    # stage2 = FSHearNet(
    #     aei_path=make_abs_path("../../trainer/faceshifter/out/faceshifter_vanilla/epoch=32-step=509999.ckpt")
    # )
    # final_out, _ = stage2(source, target)
    # print('final out:', final_out.shape)
