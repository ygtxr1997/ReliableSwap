import os
import argparse
import torch
import torch.nn as nn

from PIL import Image, ImageDraw

from torchalign import FacialLandmarkDetector

parser = argparse.ArgumentParser(description='Facial Landmark Detection Demo.')
parser.add_argument('-i', '--image', type=str, default='data/demo.jpg',
                    help='path/to/demo/img/')
parser.add_argument('-w', '--model', type=str, default='models/wflw/hrnet18_256x256_p2/',
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

print('=> use device: {}'.format(device))
model.to(device)

print('=> load image: {}'.format(args.image))
img = Image.open(args.image)

def load_bbox_from_txt(filename):
    with open(filename, 'r') as f:
        lines = [x.strip().split() for x in f.readlines()]
    return torch.Tensor([[float(x) for x in line] for line in lines])

print('=> load bbox: {}.bbox'.format(args.image))
bbox = load_bbox_from_txt('{}.bbox'.format(args.image))

print('=> detect landmark: #bbox={}'.format(len(bbox)))

# with torch.no_grad():
#     landmark = model(img, bbox[:,:4], device=device).cpu()

with torch.no_grad():
    landmark = [model(img, x[:,:4], device=device) 
        for x in bbox.chunk(max(1,len(bbox)//256), 0)]
    landmark = torch.cat(landmark, 0).cpu()

print('=> save result: {}.landmark'.format(args.image))
draw = ImageDraw.Draw(img)
with open('{}.landmark'.format(args.image),'w') as f:
    for i, box in enumerate(bbox):
        draw.rectangle(box[:4].tolist(), outline='blue')
        for x,y in landmark[i]:
            draw.point([x,y], fill='yellow')
            f.write('{:.1f} {:.1f} '.format(x,y))
        f.write('\n')

print('=> save result: {}.png'.format(args.image))
img.save('{}.png'.format(args.image))

img.show()

print('Done!')
