import torch
from torchvision import models
import numpy as np
import torch.nn as nn

from inplace_abn import InPlaceABN


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class SimpleModel(nn.Module):
    def __init__(self,
                 abn=InPlaceABN,
                 ):
        super().__init__()
        self.inplanes = 128

        self.conv1 = conv3x3(3, 64, stride=2)
        self.bn1 = abn(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(64, 64)
        self.bn2 = abn(64)
        self.relu2 = nn.ReLU(inplace=False)
        self.conv3 = conv3x3(64, 128)
        self.bn3 = abn(128)
        self.relu3 = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.maxpool(x)

        return x

if __name__ == '__main__':
    ''' 1. Check for torch.cuda '''
    print(torch.cuda.is_available())

    image = np.random.random(size=[2, 3, 224, 224])
    image.dtype = 'float32'

    image_tensor = torch.from_numpy(image).cuda()

    model = models.resnet50(pretrained=False)
    model = model.cuda()

    out = model(image_tensor)
    print(out)

    ''' 2. Check for inplace-abn '''
    net = SimpleModel().cuda()
    net = net.eval()
    x = torch.randn((2, 3, 256, 256)).cuda()
    y = net(x)
    print(y.shape)

    print('[GPU environment OK!]')