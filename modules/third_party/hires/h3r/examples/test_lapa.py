import os
import argparse

import torch
import torch.nn as nn
import torch.utils.data as data

from PIL import Image

from torchalign import FacialLandmarkDetector

parser = argparse.ArgumentParser(description='Facial Landmark Detection.')
parser.add_argument('--model', type=str, default='models/lapa/hrnet18_256x256_p2/',
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

class LAPA(data.Dataset):
    """LAPA Dataset. 
    """
    def __init__(self, root='data/LAPA/', split='test'):
        self.data_source = []
        for fname in os.listdir(os.path.join(root, split, 'images')):
            if not os.path.isfile(os.path.join(root, split, 'images', fname)):
                contine
            bbox = self.load_from_txt(os.path.join(root, split, 'bbox',
                '{}.txt'.format(fname.split('.')[0])), 0)
            bbox = torch.Tensor(bbox[0])
            landmark = self.load_from_txt(os.path.join(root, split, 'landmarks',
                '{}.txt'.format(fname.split('.')[0])), 1)
            landmark = torch.Tensor(landmark)
            self.data_source.append({
                'filename': os.path.join(root, split, 'images', fname),
                'landmark': landmark, 'bbox': bbox,
                'distance': (landmark[66]-landmark[79]).norm(), # interocular
                #'distance': (landmark[104]-landmark[105]).norm(), # interpupil
            })
    
    def load_from_txt(self, fname, start=1):
        anno_list = []
        with open(fname, 'r') as f:
            for line in f.readlines()[start:]:
                anno_list.append([float(x) for x in line.strip().split()])
        return anno_list    

    def __getitem__(self, index):
        anno = self.data_source[index]
        return Image.open(anno['filename']), anno

    def __len__(self):
        return len(self.data_source)

test_dataset = LAPA(root='data/LAPA', split='test')

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

