import os
import argparse

import torch
import torch.nn as nn
import torch.utils.data as data

from PIL import Image

from torchalign import FacialLandmarkDetector

parser = argparse.ArgumentParser(description='Facial Landmark Detection.')
parser.add_argument('--model', type=str, default='models/300w/hrnet18_256x256_p2/',
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

class F300W(data.Dataset):
    """300W Dataset. 
    """
    def __init__(self, root='data/F300W/', split='test'):
        self.data_source = []
        with open(os.path.join(root, '{}.txt'.format(split)), 'r') as f:
            for fname in [line.strip() for line in f.readlines()]:
                with open(os.path.join(root, 'images', '{}.pts'.format(fname[:-4])), 'r') as ff:
                    landmark = [line.strip() for line in ff.readlines()[3:-1]]
                landmark = torch.Tensor([[float(x) for x in pt.split()] for pt in landmark])
                self.data_source.append({
                    'landmark': landmark-1,
                    'filename': os.path.join(root, 'images', fname),
                    'bbox': torch.cat([landmark.min(0)[0], landmark.max(0)[0]]),
                    'distance': (landmark[36]-landmark[45]).norm(),
                    #'interpupil': (landmark[36:42].mean(0)-landmark[42:48].mean(0)).norm(),
                })

    def __getitem__(self, index):
        anno = self.data_source[index]
        img = Image.open(anno['filename'])
        return img.convert('RGB'), anno

    def __len__(self):
        return len(self.data_source)

test_dataset = F300W(root='data/F300W', split='test')

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

