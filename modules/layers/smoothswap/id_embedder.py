import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.layers.smoothswap.resnet import resnet50


class IdentityHead(nn.Module):
    def __init__(self):
        super(IdentityHead, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(512 * 4, 1024),
            nn.BatchNorm1d(num_features=1024),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(num_features=512)
        )

        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d,)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = F.normalize(x)
        return x


class IdentityEmbedder(nn.Module):
    def __init__(self):
        super(IdentityEmbedder, self).__init__()

        self.backbone = resnet50(pretrained=False)
        self.head = IdentityHead()

    def forward(self, x_src):
        x_src = self.backbone(x_src)
        x_src = self.head(x_src)
        return x_src


if __name__ == '__main__':
    img = torch.randn((11, 3, 256, 256)).cuda()
    net = IdentityEmbedder().cuda()
    out = net(img)
    print(out.shape)
