import os
import argparse

import torch
import torch.nn as nn
import torch.utils.data as data

from os import listdir
from os.path import isfile, join

import torchfile 

from PIL import Image

from torchalign import FacialLandmarkDetector

parser = argparse.ArgumentParser(description='Facial Landmark Detection.')
parser.add_argument('--model', type=str, default='models/300wlp/hrnet18_256x256_p2/',
                    help='path/to/model')
args = parser.parse_args()
print(args)

model = FacialLandmarkDetector(args.model)
model.eval()
print(model)

if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

model.to(device)

class LS3DW(data.Dataset):
    """LS3DW Dataset. 
    """
    def __init__(self, root='data/LS3DW/', split='test'):
        self.data_source = []
        for i in range(7200):
            fname = '{}.jpg'.format(i+1)
            if isfile(os.path.join(root, 'new_dataset', fname)) and fname.endswith('.jpg'):
                landmark = torch.Tensor(torchfile.load(os.path.join(root, 'new_dataset', '{}.t7'.format(i+1))))
                bbox = torch.cat([landmark.min(0)[0], landmark.max(0)[0]])
                self.data_source.append({
                    'landmark': landmark,
                    'filename': os.path.join(root, 'new_dataset', fname),
                    'bbox': bbox,
                    'distance': (bbox[2:]-bbox[:2]).prod().sqrt(),
                })

    def __getitem__(self, index):
        anno = self.data_source[index]
        img = Image.open(anno['filename'])
        return img.convert('RGB'), anno

    def __len__(self):
        return len(self.data_source)

test_dataset = LS3DW(root='data/LS3DW/', split='test')

test_nme = 0
for i, (img, anno) in enumerate(test_dataset):
    if i % 10 == 0:
        print('Test: {}/{}'.format(i, len(test_dataset)))
    if model.config.INPUT.BBOX == 'P1':
        bbox = torch.cat([
            anno['landmark'].min(0)[0],
            anno['landmark'].max(0)[0]
        ], 0).unsqueeze(0)
    else:
        bbox = anno['bbox'].unsqueeze(0)
    landmark = model(img, bbox, device=device)
    # evaluate nme
    diff = (landmark-anno['landmark'].type_as(landmark)).norm(dim=2)
    diff = diff.mean(dim=1) / anno['distance'].type_as(landmark)
    test_nme += diff.sum().item()

print('NME: {:.6f}'.format(test_nme/len(test_dataset)))

