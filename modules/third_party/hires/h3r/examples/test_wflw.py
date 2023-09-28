import os
import argparse

import torch
import torch.nn as nn
import torch.utils.data as data

from PIL import Image

from torchalign import FacialLandmarkDetector

parser = argparse.ArgumentParser(description='Facial Landmark Detection.')
parser.add_argument('--model', type=str, default='models/wflw/hrnet18_256x256_p1/',
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

class WFLW(data.Dataset):
    """WFLW Dataset. 
    """
    def __init__(self, root='data/WFLW', split='test'):
        self.data_source = []
        with open(os.path.join(root, '{}.txt'.format(split)), 'r') as f:
            for line in f.readlines():
                line = line.strip().split()
                filename = os.path.join(root, 'images', str(line[-1]))
                landmark = torch.Tensor([float(x) for x in line[:196]]).view(-1,2)
                bbox = torch.Tensor([float(x) for x in line[196:200]])
                self.data_source.append({
                    'filename': filename, 'bbox': bbox, 'landmark': landmark,
                    'distance': (landmark[60]-landmark[72]).norm(),
                })
        print('=> {} images.'.format(len(self.data_source)))

    def __getitem__(self, index):
        anno = self.data_source[index]
        return Image.open(anno['filename']), anno

    def __len__(self):
        return len(self.data_source)

test_dataset = WFLW(root='data/WFLW', split='test')

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

print('NME: {:.5f}'.format(test_nme/len(test_dataset)))

