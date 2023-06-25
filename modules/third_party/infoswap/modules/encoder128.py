from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, InstanceNorm2d ,PReLU, ReLU, Sigmoid, Dropout2d, Dropout, AvgPool2d, MaxPool2d, AdaptiveAvgPool2d, Module, Parameter
import torch.nn.functional as F
import torch
from collections import namedtuple
import math
import pdb
from torch.nn import Sequential


class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output


class SEModule(Module):
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = AdaptiveAvgPool2d(1)
        self.fc1 = Conv2d(
            channels, channels // reduction, kernel_size=1, padding=0, bias=False)
        self.relu = ReLU(inplace=True)
        self.fc2 = Conv2d(
            channels // reduction, channels, kernel_size=1, padding=0, bias=False)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class bottleneck_IR_SE(Module):
    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR_SE, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                BatchNorm2d(depth))
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
            PReLU(depth),
            Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            BatchNorm2d(depth),
            SEModule(depth, 16)
        )

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut


class Bottleneck(namedtuple('Block', ['in_channel', 'depth', 'stride'])):
    '''A named tuple describing a ResNet block.'''


def get_block(in_channel, depth, num_units, stride=2):
    block = [bottleneck_IR_SE(in_channel, depth, stride)] + [bottleneck_IR_SE(depth, depth, 1) for i in range(num_units - 1)]
    # return Sequential(*block)
    return block


def get_blocks(num_layers):
    blocks = []
    if num_layers == 50:
        blocks += get_block(in_channel=64, depth=64, num_units=3)  # MaxPool2d
        blocks += get_block(in_channel=64, depth=128, num_units=4)
        blocks += get_block(in_channel=128, depth=256, num_units=14)
        blocks += get_block(in_channel=256, depth=512, num_units=3)
    elif num_layers == 100:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),  # MaxPool2d
            get_block(in_channel=64, depth=128, num_units=13),
            get_block(in_channel=128, depth=256, num_units=30),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 152:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),  # MaxPool2d
            get_block(in_channel=64, depth=128, num_units=8),
            get_block(in_channel=128, depth=256, num_units=36),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    return Sequential(*blocks)


class Backbone(Module):
    def __init__(self, num_layers, drop_ratio, mode='ir'):
        super(Backbone, self).__init__()
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)  # list
        self.features = []
        self.input_layer = Sequential(Conv2d(3, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))

        self.body = blocks

        self.output_layer = Sequential(BatchNorm2d(512),
                             Dropout(drop_ratio),
                             Flatten(),
                             Linear(512 * 7 * 7, 512),
                             BatchNorm1d(512))

    def forward(self, x, cache_feats=False, train_header=False):
        self.features = []

        if x.dim() == 3:  # need to be 4-dimensions
            x = x.unsqueeze(0)

        x = self.input_layer(x)
        if cache_feats:
            self.features.append(x)
        # print(x.shape)

        for i, m in enumerate(self.body.children()):
            # print(m)
            x = m(x)
            # print(x.shape)
            if cache_feats:
                self.features.append(x)

        if train_header:
            return x
        else:
            x = self.output_layer(x)
            return l2_norm(x)


class Backbone128(Module):  # 50, 0.6, 'ir_se'
    def __init__(self, num_layers, drop_ratio, mode='ir_se'):
        super(Backbone128, self).__init__()
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)  # list
        self.features = []
        self.input_layer = Sequential(Conv2d(3, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))

        self.body = blocks

        self.output_layer128 = Sequential(BatchNorm2d(512),
                                          Dropout(drop_ratio),
                                          Flatten(),
                                          Linear(512 * 8 * 8, 512),
                                          BatchNorm1d(512))

    def forward(self, x, cache_feats=False, train_header=False):
        if x.dim() == 3:  # need to be 4-dimensions
            x = x.unsqueeze(0)
        self.features = []

        x = self.input_layer(x)
        if cache_feats:
            self.features.append(x)
        # print(x.shape)

        for i, m in enumerate(self.body.children()):
            x = m(x)
            # print(x.shape)
            if cache_feats:
                self.features.append(x)

        if train_header:
            return x
        else:
            x = self.output_layer128(x)
            return l2_norm(x)

    def restrict_forward(self, z, index):
        """
        Execute the Information Bottleneck
        :param z: the feature with unwanted information being filtered out
        :param index: which inter-feature to be replaced
        :return: new id vector
        """
        if index == 0:  # replace the output of input layer
            for i, m in enumerate(self.body.children()):
                z = m(z)
        else:  # index > 0:
            for i, m in enumerate(self.body.children()):
                if i + 1 > index:
                    z = m(z)
        z = self.output_layer128(z)
        return l2_norm(z)


class Header128(Module):
    def __init__(self, drop_ratio):
        super(Header128, self).__init__()
        self.output_layer128 = Sequential(BatchNorm2d(512),
                                          Dropout(drop_ratio),
                                          Flatten(),
                                          Linear(512 * 8 * 8, 512),
                                          BatchNorm1d(512))

    def forward(self, x):
        x = self.output_layer128(x)
        return l2_norm(x)
