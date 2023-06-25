import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential

from infoswap.utils import ConvexUpsample


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


class AIILayerLambda(nn.Module):
    def __init__(self, c_h, c_attr, c_id, c_lamb):
        super(AIILayerLambda, self).__init__()
        self.attr_c = c_attr
        self.c_id = c_id
        self.c_h = c_h

        self.conv1 = nn.Conv2d(c_attr, c_h, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv2 = nn.Conv2d(c_attr, c_h, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc1 = nn.Linear(c_id, c_h)
        self.fc2 = nn.Linear(c_id, c_h)
        self.norm = nn.InstanceNorm2d(c_h, affine=False)

        self.conv_2h = nn.Conv2d(c_h + c_lamb, 1, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, h_in, z_attr, z_id, lamb):
        h = self.norm(h_in)

        # together calculate:
        if lamb is None:
            M = torch.sigmoid(self.conv_2h(h))
        else:
            M = torch.sigmoid(self.conv_2h(torch.cat((h, lamb), dim=1)))

        gamma_attr = self.conv1(z_attr)
        beta_attr = self.conv2(z_attr)
        A = gamma_attr * h + beta_attr

        gamma_id = self.fc1(z_id)
        beta_id = self.fc2(z_id)
        gamma_id = gamma_id.reshape(h.shape[0], self.c_h, 1, 1).expand_as(h)  # broadcast
        beta_id = beta_id.reshape(h.shape[0], self.c_h, 1, 1).expand_as(h)
        I = gamma_id * h + beta_id

        out = (torch.ones_like(M).to(M.device) - M) * A + M * I
        return out


class AIIResBlkLambda(nn.Module):
    def __init__(self, cin, cout, c_attr, c_id, c_lamb):
        super(AIIResBlkLambda, self).__init__()
        self.cin = cin
        self.cout = cout

        self.AAD1 = AIILayerLambda(cin, c_attr, c_id, c_lamb)  # out channel == cin
        self.conv1 = nn.Conv2d(cin, cin, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu1 = nn.LeakyReLU(negative_slope=1e-2, inplace=True)

        self.AAD2 = AIILayerLambda(cin, c_attr, c_id, c_lamb)
        self.conv2 = nn.Conv2d(cin, cout, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu2 = nn.LeakyReLU(negative_slope=1e-2, inplace=True)

        if cin != cout:
            self.AAD3 = AIILayerLambda(cin, c_attr, c_id, c_lamb)
            self.conv3 = nn.Conv2d(cin, cout, kernel_size=3, stride=1, padding=1, bias=False)
            self.relu3 = nn.LeakyReLU(negative_slope=1e-2, inplace=True)

    def forward(self, h, z_attr, z_id, lamb):
        x = self.AAD1(h, z_attr, z_id, lamb)
        x = self.relu1(x)
        x = self.conv1(x)

        x = self.AAD2(x, z_attr, z_id, lamb)
        x = self.relu2(x)
        x = self.conv2(x)

        if self.cin != self.cout:
            h = self.AAD3(h, z_attr, z_id, lamb)
            h = self.relu3(h)
            h = self.conv3(h)

        x = x + h

        return x


class AII512(nn.Module):
    def __init__(self, c_id=512):
        super(AII512, self).__init__()
        self.up1 = nn.ConvTranspose2d(c_id, 1024, kernel_size=2, stride=1, padding=0)
        self.up2 = nn.ConvTranspose2d(1024, 1024, kernel_size=2, stride=2, padding=0)

        self.AADBlk1 = AIIResBlkLambda(1024, 1024, c_attr=1024, c_id=c_id, c_lamb=1024)
        self.AADBlk2 = AIIResBlkLambda(1024, 1024, 2048, c_id, c_lamb=1024)
        self.AADBlk3 = AIIResBlkLambda(1024, 1024, 1024, c_id, c_lamb=512)
        self.AADBlk4 = AIIResBlkLambda(1024, 512, 512, c_id, c_lamb=256)
        self.AADBlk5 = AIIResBlkLambda(512, 256, 256, c_id, c_lamb=128)
        self.AADBlk6 = AIIResBlkLambda(256, 128, 128, c_id, c_lamb=64)
        self.AADBlk7 = AIIResBlkLambda(128, 64, 64, c_id, c_lamb=32)
        # self.last_no_lamb = last_no_lamb
        self.AADBlk8 = AIIResBlkLambda(64, 3, 64, c_id, c_lamb=32)

        self.deconv = nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=4, stride=2, padding=1, bias=False)

        self.apply(weight_init)

    def forward(self, z_id, z_attr, lamb):
        m = self.up1(z_id.reshape(z_id.shape[0], -1, 1, 1))  # [B, 1024, 2, 2]  for 256 generation
        m = self.up2(m)  # [B, 1024, 4, 4] for 512 generation
        # print(m.shape)
        # if m.shape[-1] != z_attr[0].shape[-1]:
        #     m = self.up3(m)  # [B, 1024, 8, 8] for 1024 generation

        m = self.AADBlk1(m, z_attr[0], z_id, lamb[0])
        m = F.interpolate(m, scale_factor=2, mode='bilinear', align_corners=True)
        # print(m.shape)

        m = self.AADBlk2(m, z_attr[1], z_id, lamb[1])
        m = F.interpolate(m, scale_factor=2, mode='bilinear', align_corners=True)
        # print(m.shape)

        m = self.AADBlk3(m, z_attr[2], z_id, lamb[2])
        m = F.interpolate(m, scale_factor=2, mode='bilinear', align_corners=True)
        # print(m.shape)

        m = self.AADBlk4(m, z_attr[3], z_id, lamb[3])
        m = F.interpolate(m, scale_factor=2, mode='bilinear', align_corners=True)
        # print(m.shape)

        m = self.AADBlk5(m, z_attr[4], z_id, lamb[4])
        m = F.interpolate(m, scale_factor=2, mode='bilinear', align_corners=True)
        # print(m.shape)

        m = self.AADBlk6(m, z_attr[5], z_id, lamb[5])
        m = F.interpolate(m, scale_factor=2, mode='bilinear', align_corners=True)
        # print(m.shape)

        m = self.AADBlk7(m, z_attr[6], z_id, lamb[6])
        m = F.interpolate(m, scale_factor=2, mode='bilinear', align_corners=True)
        # print(m.shape)

        y = self.AADBlk8(m, z_attr[7], z_id, lamb[7])
        # print(y.shape)

        return torch.tanh(y)


class AII256(nn.Module):
    def __init__(self, c_id=512, last_no_lamb=False, use_lamb=False):
        super(AII256, self).__init__()
        self.up1 = nn.ConvTranspose2d(c_id, 1024, kernel_size=2, stride=1, padding=0)
        # self.up2 = nn.ConvTranspose2d(1024, 1024, kernel_size=2, stride=2, padding=0)

        self.use_lamb = use_lamb
        if use_lamb:
            self.AADBlk1 = AIIResBlkLambda(1024, 1024, c_attr=1024, c_id=c_id, c_lamb=1024)
            self.AADBlk2 = AIIResBlkLambda(1024, 1024, 2048, c_id, c_lamb=1024)
            self.AADBlk3 = AIIResBlkLambda(1024, 1024, 1024, c_id, c_lamb=512)
            self.AADBlk4 = AIIResBlkLambda(1024, 512, 512, c_id, c_lamb=256)
            self.AADBlk5 = AIIResBlkLambda(512, 256, 256, c_id, c_lamb=128)
            self.AADBlk6 = AIIResBlkLambda(256, 128, 128, c_id, c_lamb=64)
            self.AADBlk7 = AIIResBlkLambda(128, 64, 64, c_id, c_lamb=64)
            self.AADBlk8 = AIIResBlkLambda(64, 3, 64, c_id, c_lamb=64)
        else:
            self.AADBlk1 = AIIResBlkLambda(1024, 1024, c_attr=1024, c_id=c_id, c_lamb=0)
            self.AADBlk2 = AIIResBlkLambda(1024, 1024, 2048, c_id, 0)
            self.AADBlk3 = AIIResBlkLambda(1024, 1024, 1024, c_id, 0)
            self.AADBlk4 = AIIResBlkLambda(1024, 512, 512, c_id, 0)
            self.AADBlk5 = AIIResBlkLambda(512, 256, 256, c_id, 0)
            self.AADBlk6 = AIIResBlkLambda(256, 128, 128, c_id, 0)
            self.AADBlk7 = AIIResBlkLambda(128, 64, 64, c_id, 0)
            self.AADBlk8 = AIIResBlkLambda(64, 3, 64, c_id, 0)

        self.deconv = nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=4, stride=2, padding=1, bias=False)

        self.apply(weight_init)

    def forward(self, z_id, z_attr, lamb):
        m = self.up1(z_id.reshape(z_id.shape[0], -1, 1, 1))  # [B, 1024, 2, 2]  for 256 generation
        if not self.use_lamb:
            lamb = [None for _ in range(8)]

        m = self.AADBlk1(m, z_attr[0], z_id, lamb[0])
        m = F.interpolate(m, scale_factor=2, mode='bilinear', align_corners=True)
        # print(m.shape)

        m = self.AADBlk2(m, z_attr[1], z_id, lamb[1])
        m = F.interpolate(m, scale_factor=2, mode='bilinear', align_corners=True)
        # print(m.shape)

        m = self.AADBlk3(m, z_attr[2], z_id, lamb[2])
        m = F.interpolate(m, scale_factor=2, mode='bilinear', align_corners=True)
        # print(m.shape)

        m = self.AADBlk4(m, z_attr[3], z_id, lamb[3])
        m = F.interpolate(m, scale_factor=2, mode='bilinear', align_corners=True)
        # print(m.shape)

        m = self.AADBlk5(m, z_attr[4], z_id, lamb[4])
        m = F.interpolate(m, scale_factor=2, mode='bilinear', align_corners=True)
        # print(m.shape)

        m = self.AADBlk6(m, z_attr[5], z_id, lamb[5])
        m = F.interpolate(m, scale_factor=2, mode='bilinear', align_corners=True)
        # print(m.shape)

        m = self.AADBlk7(m, z_attr[6], z_id, lamb[6])
        m = F.interpolate(m, scale_factor=2, mode='bilinear', align_corners=True)
        # print(m.shape)

        y = self.AADBlk8(m, z_attr[7], z_id, lamb[7])

        return torch.tanh(y)


class AII1024(nn.Module):
    def __init__(self, c_id=512, last_no_lamb=False):
        super(AII1024, self).__init__()
        self.up1 = nn.ConvTranspose2d(c_id, 1024, kernel_size=2, stride=1, padding=0)
        self.up2 = nn.ConvTranspose2d(1024, 1024, kernel_size=2, stride=2, padding=0)
        self.up3 = nn.ConvTranspose2d(1024, 1024, kernel_size=2, stride=2, padding=0)

        self.AADBlk1 = AIIResBlkLambda(1024, 1024, c_attr=1024, c_id=c_id, c_lamb=1024)
        self.AADBlk2 = AIIResBlkLambda(1024, 1024, 2048, c_id, c_lamb=1024)
        self.AADBlk3 = AIIResBlkLambda(1024, 1024, 1024, c_id, c_lamb=512)
        self.AADBlk4 = AIIResBlkLambda(1024, 512, 512, c_id, c_lamb=256)
        self.AADBlk5 = AIIResBlkLambda(512, 256, 256, c_id, c_lamb=128)
        self.AADBlk6 = AIIResBlkLambda(256, 128, 128, c_id, c_lamb=64)
        self.AADBlk7 = AIIResBlkLambda(128, 64, 64, c_id, c_lamb=32)
        self.AADBlk_8 = AIIResBlkLambda(64, 64, 64, c_id, c_lamb=32)
        self.AADBlk9 = AIIResBlkLambda(64, 3, 64, c_id, c_lamb=32)

        self.apply(weight_init)

    def forward(self, z_id, z_attr, lamb):
        m = self.up1(z_id.reshape(z_id.shape[0], -1, 1, 1))  # B, 1024, 2, 2
        # print(m.shape)
        m = self.up2(m)  # [B, 1024, 4, 4]
        # print(m.shape)
        if m.shape[-1] != z_attr[0].shape[-1]:
            m = self.up3(m)  # [B, 1024, 8, 8] for 1024 generation

        m = self.AADBlk1(m, z_attr[0], z_id, lamb[0])
        m = F.interpolate(m, scale_factor=2, mode='bilinear', align_corners=True)
        # print(m.shape)

        m = self.AADBlk2(m, z_attr[1], z_id, lamb[1])
        m = F.interpolate(m, scale_factor=2, mode='bilinear', align_corners=True)
        # print(m.shape)

        m = self.AADBlk3(m, z_attr[2], z_id, lamb[2])
        m = F.interpolate(m, scale_factor=2, mode='bilinear', align_corners=True)
        # print(m.shape)

        m = self.AADBlk4(m, z_attr[3], z_id, lamb[3])
        m = F.interpolate(m, scale_factor=2, mode='bilinear', align_corners=True)
        # print(m.shape)

        m = self.AADBlk5(m, z_attr[4], z_id, lamb[4])
        m = F.interpolate(m, scale_factor=2, mode='bilinear', align_corners=True)
        # print(m.shape)

        m = self.AADBlk6(m, z_attr[5], z_id, lamb[5])
        m = F.interpolate(m, scale_factor=2, mode='bilinear', align_corners=True)
        # print(m.shape)

        m = self.AADBlk7(m, z_attr[6], z_id, lamb[6])
        m = F.interpolate(m, scale_factor=2, mode='bilinear', align_corners=True)
        # print(m.shape)

        m = self.AADBlk_8(m, z_attr[7], z_id, lamb[7])  # 64x512x512
        m = F.interpolate(m, scale_factor=2, mode='bilinear', align_corners=True)  #

        y = self.AADBlk9(m, z_attr[8], z_id, lamb[8])

        return torch.tanh(y)
