import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F

import os
import cv2
from PIL import Image
import numpy as np

from simswap.models.models import create_model

make_abs_path = lambda fn: os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), fn))


class SimSwapConfig(object):
    def __init__(self):
        self.name = 'people'
        self.gpu_ids = '0'
        self.checkpoints_dir = make_abs_path('./checkpoints')
        self.model = 'pix2pixHD'
        self.norm = 'batch'
        self.use_dropout = False
        self.data_type = 32
        self.verbose = False
        self.fp16 = False
        self.local_rank = 0
        self.isTrain = False

        self.batchSize = 8
        self.loadSize = 1024
        self.fineSize = 512
        self.label_nc = 0
        self.input_nc = 3
        self.output_nc = 3

        self.dataroot = './datasets/cityscapes/'
        self.resize_or_crop = 'scale_width'
        self.serial_batches = False
        self.no_flip = False
        self.nThreads = 2
        self.max_dataset_size = "inf"

        self.netG = 'global'
        self.latent_size = 512
        self.ngf = 64
        self.n_downsample_global = 3
        self.n_blocks_global = 6
        self.n_blocks_local = 3
        self.n_local_enhancers = 1
        self.niter_fix_global = 0

        self.ntest=float("inf")
        self.results_dir='./results/'
        self.aspect_ratio=1.0
        self.phase='test'
        self.which_epoch='latest'
        self.how_man=50,
        self.cluster_path='features_clustered_010.npy',
        self.use_encoded_image = False
        self.export_onnx = ''
        self.engine = ''
        self.onnx = ''
        self.Arc_path = make_abs_path('arcface_model/arcface_fixed.tar')
        self.id_thres=0.03,
        self.no_simswaplogo=False
        self.use_mask=False
        self.crop_size=224


class SimSwapOfficialImageInfer(object):
    def __init__(self):
        torch.nn.Module.dump_patches = True
        self.opt = SimSwapConfig()
        self.model = create_model(self.opt).eval()

        self.transformer_Arcface = transforms.Compose([
            # transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        print('[SimSwap Official] model loaded.')

    @torch.no_grad()
    def image_infer(self, source_tensor, target_tensor):
        source_tensor = source_tensor / 2 + 0.5
        target_tensor = target_tensor / 2 + 0.5

        source_tensor = self.transformer_Arcface(source_tensor)
        source_id = self.model.netArc(F.interpolate(source_tensor, size=(112,112), mode='bilinear', align_corners=True))
        source_id = F.normalize(source_id, dim=1)

        result = self.model(source_tensor, target_tensor, source_id, source_id, True)  # in [0,1]

        return result * 2 - 1  # to [-1,1]

    @staticmethod
    def save_tensor_to_img(tensor: torch.Tensor, path: str, scale=256):
        tensor = tensor.permute(0, 2, 3, 1)[0]  # in [0,1]
        tensor = tensor * scale
        tensor_np = tensor.cpu().numpy().astype(np.uint8)
        if tensor_np.shape[-1] == 1:  # channel dim
            tensor_np = tensor_np.repeat(3, axis=-1)
        tensor_img = Image.fromarray(tensor_np)
        tensor_img.save(path)

    def save_arcface(self):
        arcface = self.model.netArc
        torch.save({'model': arcface}, './arcface_model/arcface_fixed.tar')


if __name__ == '__main__':
    source_pil = Image.open('./infer_images/source.jpg')
    target_pil = Image.open('./infer_images/target.jpg')

    trans = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    source_tensor = trans(source_pil).unsqueeze(0).cuda()
    target_tensor = trans(target_pil).unsqueeze(0).cuda()

    simswap_image_infer = SimSwapOfficialImageInfer()
    # result = simswap_image_infer.image_infer(source_tensor, target_tensor)
    # result = (result.clamp(-1, 1) + 1) / 2
    # simswap_image_infer.save_tensor_to_img(result, path='./infer_images/result.jpg')
    # simswap_image_infer.save_arcface()

