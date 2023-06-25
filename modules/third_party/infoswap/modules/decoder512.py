import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential


def weight_init(m):
    if isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.001)
        m.bias.data.zero_()
    if isinstance(m, nn.Conv2d):
        # nn.init.xavier_normal_(m.weight.data)
        nn.init.kaiming_normal_(m.weight.data)

    if isinstance(m, nn.ConvTranspose2d):
        # nn.init.xavier_normal_(m.weight.data)
        nn.init.kaiming_normal_(m.weight.data)


def conv4x4(in_c, out_c, norm=nn.BatchNorm2d):
    return Sequential(
        nn.Conv2d(in_channels=in_c, out_channels=out_c,
                  kernel_size=4, stride=2, padding=1, bias=False),
        norm(out_c),
        nn.LeakyReLU(0.1, inplace=True)
    )


class deconv4x4(nn.Module):
    def __init__(self, in_c, out_c, norm=nn.BatchNorm2d):
        super(deconv4x4, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels=in_c, out_channels=out_c,
                                         kernel_size=4, stride=2, padding=1, bias=False)
        self.bn = norm(out_c)
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, input, skip):
        x = self.deconv(input)
        x = self.bn(x)
        x = self.lrelu(x)
        return torch.cat((x, skip), dim=1)


class UnetDecoder512(nn.Module):
    def __init__(self, first_use_img=True):
        super(UnetDecoder512, self).__init__()
        self.conv_0 = nn.Conv2d(3, 32, 4, 2, 1, bias=False)
        self.first_use_img = first_use_img

        self.conv_1 = nn.Sequential(nn.PixelShuffle(upscale_factor=2),
                                    nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False))
        self.conv_2 = nn.Conv2d(64, 64, 3, 1, 1, bias=False)
        self.conv_3 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.conv_4 = nn.Conv2d(256, 256, 3, 1, 1, bias=False)
        self.conv_5 = nn.Conv2d(256, 512, 4, 2, 1, bias=False)  # 512
        self.conv_6 = nn.Conv2d(512, 1024, 4, 2, 1, bias=False)  # 512

        self.conv7 = conv4x4(1024, 1024)
        self.deconv1 = deconv4x4(1024, 1024)
        self.deconv2 = deconv4x4(2048, 512)
        self.deconv3 = deconv4x4(1024, 256)
        self.deconv4 = deconv4x4(512, 128)
        self.deconv5 = deconv4x4(256, 64)
        self.deconv6 = deconv4x4(128, 32)

        self.apply(weight_init)

    def forward(self, feats, lambs, use_lambda=True):
        """
        :param feats: Arcface intermediate features
        :return: decoded features
        """
        if self.first_use_img:
            feat1 = self.conv_0(feats[0])
            # 3x512x512 -> 32x256x256
        else:
            feat1 = self.conv_1(feats[0])
            # 64x128x128 -> 32x256x256

        feat2 = self.conv_2(F.interpolate(feats[1], scale_factor=2, mode='bilinear', align_corners=True))
        # 64x128x128

        feat3 = torch.cat((feats[2], feats[3]), dim=1)
        feat3 = self.conv_3(feat3)
        # 128x64x64

        feat4 = torch.cat((feats[4], feats[5]), dim=1)
        feat4 = self.conv_4(feat4)
        # 256x32x32

        feat5 = self.conv_5(torch.cat((feats[6], feats[7]), dim=1))
        # 512x16x16

        feat6 = self.conv_6(torch.cat((feats[8], feats[9]), dim=1))
        # 1024x8x8

        z_attr1 = self.conv7(feat6)
        # 1024x4x4

        z_attr2 = self.deconv1(z_attr1, feat6)
        # 2048x8x8
        z_attr3 = self.deconv2(z_attr2, feat5)
        # 1024x16x16
        z_attr4 = self.deconv3(z_attr3, feat4)
        # 512x32x32
        z_attr5 = self.deconv4(z_attr4, feat3)
        # 256x64x64
        z_attr6 = self.deconv5(z_attr5, feat2)
        # print(z_attr6.shape)
        # 128x128x128
        z_attr7 = self.deconv6(z_attr6, feat1)  # z_attr6 --> 32x256x256, then || feat1
        # 64x256x256
        z_attr8 = F.interpolate(z_attr7, scale_factor=2, mode='bilinear', align_corners=True)
        # 64x512x512

        if not use_lambda:
            return z_attr1, z_attr2, z_attr3, z_attr4, z_attr5, z_attr6, z_attr7, z_attr8
        else:
            lamb1 = lambs[0]
            lamb1 = self.conv_1(lamb1)  # 32x256x256
            lamb0 = F.interpolate(lamb1, scale_factor=2, mode='bilinear', align_corners=True)  # 32x512x512
            lamb2 = self.conv_2(F.interpolate(lambs[1], scale_factor=2, mode='bilinear', align_corners=True))  # 64x128x128

            lamb3 = torch.cat((lambs[2], lambs[3]), dim=1)
            lamb3 = self.conv_3(lamb3)  # 128x64x64

            lamb4 = torch.cat((lambs[4], lambs[5]), dim=1)
            lamb4 = self.conv_4(lamb4)  # 256x32x32

            lamb5 = self.conv_5(torch.cat((lambs[6], lambs[7]), dim=1))  # 512x16x16
            lamb6 = self.conv_6(torch.cat((lambs[8], lambs[9]), dim=1))  # 1024x8x8
            lamb7 = self.conv7(lamb6)  # 1024x4x4
            return [[z_attr1, z_attr2, z_attr3, z_attr4, z_attr5, z_attr6, z_attr7, z_attr8],
                    [lamb7, lamb6, lamb5, lamb4, lamb3, lamb2, lamb1, lamb0]]