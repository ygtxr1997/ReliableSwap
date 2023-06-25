from collections import namedtuple

import torch.nn as nn
import torch.nn.functional as F


import torch
from torchvision.models import vgg19
from collections import OrderedDict


vgg_layer = {
    "conv_1_1": 0,
    "conv_1_2": 2,
    "pool_1": 4,
    "conv_2_1": 5,
    "conv_2_2": 7,
    "pool_2": 9,
    "conv_3_1": 10,
    "conv_3_2": 12,
    "conv_3_3": 14,
    "conv_3_4": 16,
    "pool_3": 18,
    "conv_4_1": 19,
    "conv_4_2": 21,
    "conv_4_3": 23,
    "conv_4_4": 25,
    "pool_4": 27,
    "conv_5_1": 28,
    "conv_5_2": 30,
    "conv_5_3": 32,
    "conv_5_4": 34,
    "pool_5": 36,
}

vgg_layer_inv = {
    0: "conv_1_1",
    2: "conv_1_2",
    4: "pool_1",
    5: "conv_2_1",
    7: "conv_2_2",
    9: "pool_2",
    10: "conv_3_1",
    12: "conv_3_2",
    14: "conv_3_3",
    16: "conv_3_4",
    18: "pool_3",
    19: "conv_4_1",
    21: "conv_4_2",
    23: "conv_4_3",
    25: "conv_4_4",
    27: "pool_4",
    28: "conv_5_1",
    30: "conv_5_2",
    32: "conv_5_3",
    34: "conv_5_4",
    36: "pool_5",
}


class VGG_Model(nn.Module):
    def __init__(self, vgg, listen_list=None, use_224=True):
        super(VGG_Model, self).__init__()
        self.vgg_model = vgg.features
        vgg_dict = vgg.state_dict()
        vgg_f_dict = self.vgg_model.state_dict()
        vgg_dict = {k: v for k, v in vgg_dict.items() if k in vgg_f_dict}
        vgg_f_dict.update(vgg_dict)
        # no grad
        for p in self.vgg_model.parameters():
            p.requires_grad = False
        if listen_list == []:
            self.listen = []
        else:
            self.listen = set()
            for layer in listen_list:
                self.listen.add(vgg_layer[layer])
        self.features = OrderedDict()
        self.use_224 = use_224

    def forward(self, x):
        if self.use_224:
            x = F.interpolate(x, size=(224, 224), )
        for index, layer in enumerate(self.vgg_model):
            x = layer(x)
            if index in self.listen:
                self.features[vgg_layer_inv[index]] = x
        return self.features


class VGG19(nn.Module):
    def __init__(self, requires_grad=False):
        super(VGG19, self).__init__()
        vgg_pretrained_features = vgg.vgg19(pretrained=True).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 18):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(18, 27):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(27, 36):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        X = F.upsample(X, (224, 224))
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_4 = h
        h = self.slice4(h)
        h_relu4_4 = h
        h = self.slice5(h)
        h_relu5_4 = h

        vgg_outputs = namedtuple(
            "VggOutputs", ["relu1_2", "relu2_2", "relu3_4", "relu4_4", "relu5_4"]
        )
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_4, h_relu4_4, h_relu5_4)

        return out
