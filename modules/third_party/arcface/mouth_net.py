import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from modules.third_party.arcface.iresnet import iresnet50, iresnet100

class MouthNet(nn.Module):
    def __init__(self,
                 bisenet: nn.Module,
                 feature_dim: int = 64,
                 crop_param: tuple = (0, 0, 112, 112),
                 iresnet_pretrained: bool = False,
                 ):
        super(MouthNet, self).__init__()

        crop_size = (crop_param[3] - crop_param[1], crop_param[2] - crop_param[0])  # (H,W)
        fc_scale = int(math.ceil(crop_size[0] / 112 * 7) * math.ceil(crop_size[1] / 112 * 7))

        self.bisenet = bisenet
        self.backbone = iresnet50(
            pretrained=iresnet_pretrained,
            num_features=feature_dim,
            fp16=False,
            fc_scale=fc_scale,
        )

        self.register_buffer(
            name="vgg_mean",
            tensor=torch.tensor([[[0.485]], [[0.456]], [[0.406]]], requires_grad=False),
        )
        self.register_buffer(
            name="vgg_std",
            tensor=torch.tensor([[[0.229]], [[0.224]], [[0.225]]], requires_grad=False),
        )

    def forward(self, x):
        # with torch.no_grad():
        #     x_mouth_mask = self.get_any_mask(x, par=[11, 12, 13], normalized=True)  # (B,1,H,W), in [0,1], 1:chosed
        x_mouth_mask = 1
        x_mouth = x * x_mouth_mask  # (B,3,112,112)
        mouth_feature = self.backbone(x_mouth)
        return mouth_feature

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

    def save_backbone(self, path: str):
        torch.save(self.backbone.state_dict(), path)

    def load_backbone(self, path: str):
        self.backbone.load_state_dict(torch.load(path))


if __name__ == "__main__":
    from modules.third_party.bisenet.bisenet import BiSeNet

    bisenet = BiSeNet(19)
    bisenet.load_state_dict(
        torch.load(
            "/gavin/datasets/hanbang/79999_iter.pth",
            map_location="cpu",
        )
    )
    bisenet.eval()
    bisenet.requires_grad_(False)

    crop_param = (28, 56, 84, 112)

    import numpy as np
    img = np.random.randn(112, 112, 3) * 225
    from PIL import Image
    img = Image.fromarray(img.astype(np.uint8))
    img = img.crop(crop_param)

    from torchvision import transforms
    trans = transforms.ToTensor()
    img = trans(img).unsqueeze(0)
    img = img.repeat(3, 1, 1, 1)
    print(img.shape)

    net = MouthNet(
        bisenet=bisenet,
        feature_dim=64,
        crop_param=crop_param
    )
    mouth_feat = net(img)
    print(mouth_feat.shape)

    import thop

    crop_size = (crop_param[3] - crop_param[1], crop_param[2] - crop_param[0])  # (H,W)
    fc_scale = int(math.ceil(crop_size[0] / 112 * 7) * math.ceil(crop_size[1] / 112 * 7))
    backbone = iresnet100(
        pretrained=False,
        num_features=64,
        fp16=False,
        # fc_scale=fc_scale,
    )
    flops, params = thop.profile(backbone, inputs=(torch.randn(1, 3, 112, 112),), verbose=False)
    print('#Params=%.2fM, GFLOPS=%.2f' % (params / 1e6, flops / 1e9))
