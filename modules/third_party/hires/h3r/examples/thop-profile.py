import os
import argparse
import torch
import torch.nn as nn

from thop import profile, clever_format
from torchalign import FacialLandmarkDetector

parser = argparse.ArgumentParser(description='Model Profiling.')
parser.add_argument('--model', type=str, default='output/wflw/hrnet18_256x256_p1/',
                    help='path/to/model/')
args = parser.parse_args()

model = FacialLandmarkDetector(args.model, pretrained=False)
model.eval()
print(model)

macs, params = profile(
    nn.Sequential(*[model.backbone, model.heatmap_head]),
    inputs=(torch.randn(1, 3, *model.config.INPUT.SIZE),)
)

macs, params = clever_format([macs, params], "%.2f")
print(macs, params)
print('Done!')
