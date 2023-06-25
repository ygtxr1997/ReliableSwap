"""
This file only for testing mask regularzation.
If it works, it will be merged with `layers.py`.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AADLayer(nn.Module):
    def __init__(self, c_x, attr_c, c_id=256):
        super(AADLayer, self).__init__()
        self.attr_c = attr_c
        self.c_id = c_id
        self.c_x = c_x

        self.conv1 = nn.Conv2d(
            attr_c, c_x, kernel_size=1, stride=1, padding=0, bias=True
        )
        self.conv2 = nn.Conv2d(
            attr_c, c_x, kernel_size=1, stride=1, padding=0, bias=True
        )
        self.fc1 = nn.Linear(c_id, c_x)
        self.fc2 = nn.Linear(c_id, c_x)
        self.norm = nn.InstanceNorm2d(c_x, affine=False)

        self.conv_h = nn.Conv2d(c_x, 1, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, h_in, z_attr, z_id):
        # h_in cxnxn
        # zid 256x1x1
        # zattr cxnxn
        h = self.norm(h_in)
        gamma_attr = self.conv1(z_attr)
        beta_attr = self.conv2(z_attr)

        gamma_id = self.fc1(z_id)
        beta_id = self.fc2(z_id)
        A = gamma_attr * h + beta_attr
        gamma_id = gamma_id.reshape(h.shape[0], self.c_x, 1, 1).expand_as(h)
        beta_id = beta_id.reshape(h.shape[0], self.c_x, 1, 1).expand_as(h)
        I = gamma_id * h + beta_id

        M = torch.sigmoid(self.conv_h(h))

        out = (torch.ones_like(M).to(M.device) - M) * A + M * I
        return out, torch.mean(torch.ones_like(M).to(M.device) - M, dim=[1, 2, 3])


class AAD_ResBlk(nn.Module):
    def __init__(self, cin, cout, c_attr, c_id=256):
        super(AAD_ResBlk, self).__init__()
        self.cin = cin
        self.cout = cout

        self.AAD1 = AADLayer(cin, c_attr, c_id)
        self.conv1 = nn.Conv2d(cin, cin, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu1 = nn.ReLU(inplace=True)

        self.AAD2 = AADLayer(cin, c_attr, c_id)
        self.conv2 = nn.Conv2d(
            cin, cout, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.relu2 = nn.ReLU(inplace=True)

        if cin != cout:
            self.AAD3 = AADLayer(cin, c_attr, c_id)
            self.conv3 = nn.Conv2d(
                cin, cout, kernel_size=3, stride=1, padding=1, bias=False
            )
            self.relu3 = nn.ReLU(inplace=True)

    def forward(self, h, z_attr, z_id):
        x, m1_ = self.AAD1(h, z_attr, z_id)
        x = self.relu1(x)
        x = self.conv1(x)

        x, m2_ = self.AAD2(x, z_attr, z_id)
        x = self.relu2(x)
        x = self.conv2(x)

        m = m1_ + m2_

        if self.cin != self.cout:
            h, m3_ = self.AAD3(h, z_attr, z_id)
            h = self.relu3(h)
            h = self.conv3(h)
            m += m3_
        x = x + h

        return x, m


def weight_init(m):
    if isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.001)
        m.bias.data.zero_()
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data)

    if isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_normal_(m.weight.data)


def conv4x4(in_c, out_c, norm=nn.BatchNorm2d):
    return nn.Sequential(
        nn.Conv2d(
            in_channels=in_c,
            out_channels=out_c,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False,
        ),
        norm(out_c),
        nn.LeakyReLU(0.1, inplace=True),
    )


class deconv4x4(nn.Module):
    def __init__(self, in_c, out_c, norm=nn.BatchNorm2d):
        super(deconv4x4, self).__init__()
        self.deconv = nn.ConvTranspose2d(
            in_channels=in_c,
            out_channels=out_c,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False,
        )
        self.bn = norm(out_c)
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, input, skip):
        x = self.deconv(input)
        x = self.bn(x)
        x = self.lrelu(x)
        return torch.cat((x, skip), dim=1)


class MLAttrEncoder(nn.Module):
    def __init__(self, finetune=False, downup=False):
        super(MLAttrEncoder, self).__init__()

        self.downup = downup
        if self.downup:
            self.conv00 = conv4x4(3, 16)
            self.conv01 = conv4x4(16, 32)
            self.deconv7 = deconv4x4(64, 16)

        self.conv1 = conv4x4(3, 32)
        self.conv2 = conv4x4(32, 64)
        self.conv3 = conv4x4(64, 128)
        self.conv4 = conv4x4(128, 256)
        self.conv5 = conv4x4(256, 512)
        self.conv6 = conv4x4(512, 1024)
        self.conv7 = conv4x4(1024, 1024)

        self.deconv1 = deconv4x4(1024, 1024)
        self.deconv2 = deconv4x4(2048, 512)
        self.deconv3 = deconv4x4(1024, 256)
        self.deconv4 = deconv4x4(512, 128)
        self.deconv5 = deconv4x4(256, 64)
        self.deconv6 = deconv4x4(128, 32)

        self.apply(weight_init)

        self.finetune = finetune
        if finetune:
            for name, param in self.named_parameters():
                param.requires_grad = False
            if self.downup:
                self.conv00.requires_grad_(True)
                self.conv01.requires_grad_(True)
                self.deconv7.requires_grad_(True)

    def forward(self, Xt):
        if self.downup:
            feat0 = self.conv00(Xt)  # (16,256,256)
            feat1 = self.conv01(feat0)  # (32,128,128)
        else:
            feat0 = None
            feat1 = self.conv1(Xt)
            # 32x128x128

        feat2 = self.conv2(feat1)
        # 64x64x64
        feat3 = self.conv3(feat2)
        # 128x32x32
        feat4 = self.conv4(feat3)
        # 256x16xx16
        feat5 = self.conv5(feat4)
        # 512x8x8
        feat6 = self.conv6(feat5)
        # 1024x4x4

        if self.downup:
            z_attr1 = self.conv7(feat6)
            # 1024x2x2
            z_attr2 = self.deconv1(z_attr1, feat6)
            z_attr3 = self.deconv2(z_attr2, feat5)
            z_attr4 = self.deconv3(z_attr3, feat4)
            z_attr5 = self.deconv4(z_attr4, feat3)
            z_attr6 = self.deconv5(z_attr5, feat2)
            z_attr7 = self.deconv6(z_attr6, feat1)  # (128,64,64)+(32,128,128)->(64,128,128)
            z_attr8 = self.deconv7(z_attr7, feat0)  # (64,128,128)+(16,256,256)->(32,256,256)
            z_attr9 = F.interpolate(
                z_attr8, scale_factor=2, mode="bilinear", align_corners=True
            )  # (32,512,512)
            return (
                z_attr1,
                z_attr2,
                z_attr3,
                z_attr4,
                z_attr5,
                z_attr6,
                z_attr7,
                z_attr8,
                z_attr9
            )
        else:
            z_attr1 = self.conv7(feat6)
            # 1024x2x2
            z_attr2 = self.deconv1(z_attr1, feat6)
            z_attr3 = self.deconv2(z_attr2, feat5)
            z_attr4 = self.deconv3(z_attr3, feat4)
            z_attr5 = self.deconv4(z_attr4, feat3)
            z_attr6 = self.deconv5(z_attr5, feat2)
            z_attr7 = self.deconv6(z_attr6, feat1)
            z_attr8 = F.interpolate(
                z_attr7, scale_factor=2, mode="bilinear", align_corners=True
            )
            return (
                z_attr1,
                z_attr2,
                z_attr3,
                z_attr4,
                z_attr5,
                z_attr6,
                z_attr7,
                z_attr8,
            )


class AADGenerator(nn.Module):
    def __init__(self, c_id=256, finetune=False, downup=False):
        super(AADGenerator, self).__init__()
        self.up1 = nn.ConvTranspose2d(c_id, 1024, kernel_size=2, stride=1, padding=0)
        self.AADBlk1 = AAD_ResBlk(1024, 1024, 1024, c_id)
        self.AADBlk2 = AAD_ResBlk(1024, 1024, 2048, c_id)
        self.AADBlk3 = AAD_ResBlk(1024, 1024, 1024, c_id)
        self.AADBlk4 = AAD_ResBlk(1024, 512, 512, c_id)
        self.AADBlk5 = AAD_ResBlk(512, 256, 256, c_id)
        self.AADBlk6 = AAD_ResBlk(256, 128, 128, c_id)
        self.AADBlk7 = AAD_ResBlk(128, 64, 64, c_id)
        self.AADBlk8 = AAD_ResBlk(64, 3, 64, c_id)

        self.downup = downup
        if downup:
            self.AADBlk8_0 = AAD_ResBlk(64, 32, 32, c_id)
            self.AADBlk8_1 = AAD_ResBlk(32, 3, 32, c_id)

        self.apply(weight_init)

        if finetune:
            for name, param in self.named_parameters():
                param.requires_grad = False
            self.AADBlk8_0.requires_grad_(True)
            self.AADBlk8_1.requires_grad_(True)

    def forward(self, z_attr, z_id):
        m = self.up1(z_id.reshape(z_id.shape[0], -1, 1, 1))
        scale= z_attr[0].shape[2] // 2  # adaptive support for 512x512, 1024x1024
        m = F.interpolate(m, scale_factor=scale, mode='bilinear', align_corners=True)
        m2, m2_ = self.AADBlk1(m, z_attr[0], z_id)
        m2 = F.interpolate(
            m2,
            scale_factor=2,
            mode="bilinear",
            align_corners=True,
        )
        m3, m3_ = self.AADBlk2(m2, z_attr[1], z_id)
        m3 = F.interpolate(
            m3,
            scale_factor=2,
            mode="bilinear",
            align_corners=True,
        )
        m4, m4_ = self.AADBlk3(m3, z_attr[2], z_id)
        m4 = F.interpolate(
            m4,
            scale_factor=2,
            mode="bilinear",
            align_corners=True,
        )
        m5, m5_ = self.AADBlk4(m4, z_attr[3], z_id)
        m5 = F.interpolate(
            m5,
            scale_factor=2,
            mode="bilinear",
            align_corners=True,
        )
        m6, m6_ = self.AADBlk5(m5, z_attr[4], z_id)
        m6 = F.interpolate(
            m6,
            scale_factor=2,
            mode="bilinear",
            align_corners=True,
        )
        m7, m7_ = self.AADBlk6(m6, z_attr[5], z_id)
        m7 = F.interpolate(
            m7,
            scale_factor=2,
            mode="bilinear",
            align_corners=True,
        )
        m8, m8_ = self.AADBlk7(m7, z_attr[6], z_id)
        m8 = F.interpolate(
            m8,
            scale_factor=2,
            mode="bilinear",
            align_corners=True,
        )

        if self.downup:
            y0, m9_ = self.AADBlk8_0(m8, z_attr[7], z_id)
            y0 = F.interpolate(y0, scale_factor=2, mode='bilinear', align_corners=True)
            y1, m10_ = self.AADBlk8_1(y0, z_attr[8], z_id)
            y = torch.tanh(y1)
        else:
            y, m9_ = self.AADBlk8(m8, z_attr[7], z_id)
            y = torch.tanh(y)
        return y  # , m  # yuange


class AEI_Net(nn.Module):
    def __init__(self, c_id=512, finetune=False, downup=False):
        super(AEI_Net, self).__init__()
        self.encoder = MLAttrEncoder(finetune=finetune, downup=downup)
        self.generator = AADGenerator(c_id, finetune=finetune, downup=downup)

    def forward(self, Xt, z_id):
        attr = self.encoder(Xt)
        Y = self.generator(attr, z_id)  # yuange
        return Y, attr

    def get_attr(self, X):
        return self.encoder(X)

    def trainable_params(self):
        train_params = []
        for param in self.parameters():
            if param.requires_grad:
                train_params.append(param)
        return train_params


if __name__ == "__main__":
    aie = AEI_Net(512).eval()
    x = aie(torch.randn(1, 3, 512, 512), torch.randn(1, 512))


    # def numel(m: torch.nn.Module, only_trainable: bool = False):
    #     """
    #     returns the total number of parameters used by `m` (only counting
    #     shared parameters once); if `only_trainable` is True, then only
    #     includes parameters with `requires_grad = True`
    #     """
    #     parameters = list(m.parameters())
    #     if only_trainable:
    #         parameters = [p for p in parameters if p.requires_grad]
    #     unique = {p.data_ptr(): p for p in parameters}.values()
    #     return sum(p.numel() for p in unique)
    #
    #
    # print(numel(aie, True))
    # print(x[0].size())
    # print(len(x[-1]))


    import thop

    img = torch.randn(1, 3, 256, 256)
    latent = torch.randn(1, 512)
    net = aie
    flops, params = thop.profile(net, inputs=(img, latent), verbose=False)
    print('#Params=%.2fM, GFLOPS=%.2f' % (params / 1e6, flops / 1e9))
