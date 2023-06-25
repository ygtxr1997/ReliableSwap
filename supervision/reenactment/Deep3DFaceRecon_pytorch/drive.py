import os
import glob
import numpy as np
from os import makedirs, name
from PIL import Image
from tqdm import tqdm
import argparse
import random

import torch
import torch.nn as nn
from torch.nn import init
from torchvision import transforms
import torchvision.transforms.functional as F

import sys
sys.path.append('reenactment/')
from .options.inference_options import InferenceOptions
from .models import create_model
from .util.preprocess import align_img
from .util.load_mats import load_lm3d
from .extract_kp_videos import KeypointExtractor

from .generators.face_model import FaceGenerator
from .config import Config
# from util.trainer import get_model_optimizer_and_scheduler, set_random_seed, get_trainer


class CoeffDetector(nn.Module):
    def __init__(self, opt):
        super().__init__()

        self.model = create_model(opt)
        self.model.setup(opt)
        self.model.device = 'cuda'
        self.model.parallelize()
        self.model.eval()

        self.lm3d_std = load_lm3d(opt.bfm_folder) 

    def forward(self, img, lm):
        
        img, trans_params = self.image_transform(img, lm)

        data_input = {                
                'imgs': img[None],
                }        
        self.model.set_input(data_input)  
        self.model.test()
        pred_coeff = {key:self.model.pred_coeffs_dict[key].cpu().numpy() for key in self.model.pred_coeffs_dict}
        pred_coeff = np.concatenate([
            pred_coeff['id'], 
            pred_coeff['exp'], 
            pred_coeff['tex'], 
            pred_coeff['angle'],
            pred_coeff['gamma'],
            pred_coeff['trans'],
            trans_params[None],
            ], 1)
    
        return {'coeff_3dmm':pred_coeff, 
                'crop_img': Image.fromarray((img.cpu().permute(1, 2, 0).numpy()*255).astype(np.uint8))}

    def image_transform(self, images, lm):
        """
        param:
            images:          -- PIL image 
            lm:              -- numpy array
        """
        W,H = images.size
        if np.mean(lm) == -1:
            lm = (self.lm3d_std[:, :2]+1)/2.
            lm = np.concatenate(
                [lm[:, :1]*W, lm[:, 1:2]*H], 1
            )
        else:
            lm[:, -1] = H - 1 - lm[:, -1]

        trans_params, img, lm, _ = align_img(images, lm, self.lm3d_std)  
        img = torch.tensor(np.array(img)/255., dtype=torch.float32).permute(2, 0, 1)
        trans_params = np.array([float(item) for item in np.hsplit(trans_params, 5)])
        trans_params = torch.tensor(trans_params.astype(np.float32))
        return img, trans_params        


def find_crop_norm_ratio(source_coeff, target_coeffs):
    alpha = 0.3
    exp_diff = np.mean(np.abs(target_coeffs[:,80:144] - source_coeff[:,80:144]), 1)
    angle_diff = np.mean(np.abs(target_coeffs[:,224:227] - source_coeff[:,224:227]), 1)
    index = np.argmin(alpha*exp_diff + (1-alpha)*angle_diff)
    crop_norm_ratio = source_coeff[:,-3] / target_coeffs[index:index+1, -3]
    return crop_norm_ratio

def transform_semantic(source_semantic, target_semantic, crop_norm_ratio, semantic_radius=13):
    source_semantic = source_semantic[None].repeat(semantic_radius*2+1, 0)
    target_semantic = target_semantic[None].repeat(semantic_radius*2+1, 0)
    ex_coeff = target_semantic[:, 80:144] #expression, choose source or target
    angles = target_semantic[:, 224:227] #euler angles for pose
    translation = target_semantic[:, 254:257] #translation
    crop = target_semantic[:, 259:262] #crop param

    # crop[:, -3] = crop[:, -3] * crop_norm_ratio

    coeff_3dmm = np.concatenate([ex_coeff, angles, translation, crop], 1).astype(np.float32)
    return torch.Tensor(coeff_3dmm).permute(1, 0) 


def trans_image(image, resolution=256):
    image = F.resize(image, size=resolution, interpolation=Image.BICUBIC)
    image = F.to_tensor(image)
    image = F.normalize(image, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    return image

def tensor2pil(tensor: torch.Tensor) -> Image.Image:
    x = tensor.squeeze(0).permute(1, 2, 0).add(1).mul(255).div(2).squeeze()
    x = x.detach().cpu().numpy()
    x = np.rint(x).clip(0, 255).astype(np.uint8)
    return Image.fromarray(x)


def set_random_seed(seed):
    r"""Set random seeds for everything.

    Args:
        seed (int): Random seed.
        by_rank (bool):
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def drive_source_demo(source_img, target_img):
    # checkpoint_path = '/apdcephfs_cq2/share_1290939/branchwang/py_projects/PIRender/result/face/epoch_00190_iteration_000400000_checkpoint.pt'
    # cfg_pth = '/apdcephfs_cq2/share_1290939/branchwang/py_projects/PIRender/config/face_demo.yaml'
    checkpoint_path = 'Deep3DFaceRecon_pytorch/checkpoints/PIRender/result/face/epoch_00190_iteration_000400000_checkpoint.pt'
    cfg_pth = 'Deep3DFaceRecon_pytorch/checkpoints/PIRender/config/face_demo.yaml'

    set_random_seed(0)
    opt = Config(cfg_pth, is_train=False)
    opt.distributed = False
    opt.device = torch.cuda.current_device()
    # opt.logdir = '/apdcephfs_cq2/share_1290939/branchwang/py_projects/PIRender/result/face/'
    
    # create a model
    # net_G, net_G_ema, opt_G, sch_G = get_model_optimizer_and_scheduler(opt)
    # trainer = get_trainer(opt, net_G, net_G_ema, opt_G, sch_G, None)
    # current_epoch, current_iteration = trainer.load_checkpoint(opt)                          
    # net_G = trainer.net_G_ema.eval()
    net_G = FaceGenerator(**opt.gen.param).to(opt.device)
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    # net_G.load_state_dict(checkpoint['net_G'], strict=False)
    net_G.load_state_dict(checkpoint['net_G_ema'], strict=False)
    net_G.eval()

    opt = InferenceOptions().parse() 
    # opt.checkpoints_dir = '/apdcephfs_cq2/share_1290939/branchwang/py_projects/PIRender/Deep3DFaceRecon_pytorch/checkpoints'
    opt.checkpoints_dir = 'Deep3DFaceRecon_pytorch/checkpoints/PIRender/Deep3DFaceRecon_pytorch/checkpoints'
    coeff_detector = CoeffDetector(opt)
    kp_extractor = KeypointExtractor()

    pred = []
    for image in [source_img.resize((256, 256)), target_img.resize((256, 256))]:
        lm = kp_extractor.extract_keypoint(image)
        predicted = coeff_detector(image, lm)
        pred.append(predicted)

    # pred[0]['crop_img'].save('s_crop.png')
    # pred[1]['crop_img'].save('t_crop.png')

    crop_norm_ratio = find_crop_norm_ratio(pred[0]['coeff_3dmm'], pred[1]['coeff_3dmm'])
    target_coeffs = transform_semantic(pred[0]['coeff_3dmm'].reshape(-1), pred[1]['coeff_3dmm'].reshape(-1), crop_norm_ratio=crop_norm_ratio)

    with torch.no_grad():
        # input_source = trans_image(pred[0]['crop_img'])[None].cuda()
        input_source = trans_image(source_img)[None].cuda()
        target_coeffs = target_coeffs[None].cuda()
        output_dict = net_G(input_source, target_coeffs)

        warp_image = output_dict['warp_image'].cpu().clamp_(-1, 1)
        fake_image = output_dict['fake_image'].cpu().clamp_(-1, 1)

        # tensor2pil(warp_image).save('warp_image.png')
        # tensor2pil(fake_image).save('output_image.png')
    
    return tensor2pil(fake_image)


class PIRenderImageInfer(object):
    def __init__(self,
                 weights_path='Deep3DFaceRecon_pytorch/checkpoints/PIRender/result/face/epoch_00190_iteration_000400000_checkpoint.pt',
                 cfg_path='Deep3DFaceRecon_pytorch/checkpoints/PIRender/config/face_demo.yaml',
                 checkpoints_dir='Deep3DFaceRecon_pytorch/checkpoints/PIRender/Deep3DFaceRecon_pytorch/checkpoints',
                 ):
        self.cfg_path = cfg_path
        self.weights_path = weights_path
        self.checkpoints_dir = checkpoints_dir

        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                         std=[0.5, 0.5, 0.5])
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

        self._load_model()

    def _load_model(self,):
        opt = Config(self.cfg_path, is_train=False)
        opt.distributed = False
        opt.device = torch.cuda.current_device()
        net_G = FaceGenerator(**opt.gen.param).to(opt.device)
        checkpoint = torch.load(self.weights_path, map_location=lambda storage, loc: storage)
        net_G.load_state_dict(checkpoint['net_G_ema'], strict=False)
        net_G.eval()

        opt = InferenceOptions().parse()
        opt.checkpoints_dir = self.checkpoints_dir
        opt.bfm_folder = 'reenactment/Deep3DFaceRecon_pytorch/checkpoints/PIRender/Deep3DFaceRecon_pytorch/BFM'
        coeff_detector = CoeffDetector(opt)
        kp_extractor = KeypointExtractor()

        self.net_G = net_G
        self.coeff_detector = coeff_detector
        self.kp_extractor = kp_extractor

    def infer_single(self,
                     source_img: Image,
                     target_img: Image,
                     ) -> Image:
        """

        :param source_img: PIL.Image
        :param target_img: PIL.Image
        :return: PIL.Image
        """
        pred = []
        for image in [source_img.resize((256, 256)), target_img.resize((256, 256))]:
            lm = self.kp_extractor.extract_keypoint(image)
            predicted = self.coeff_detector(image, lm)
            pred.append(predicted)

        crop_norm_ratio = find_crop_norm_ratio(pred[0]['coeff_3dmm'], pred[1]['coeff_3dmm'])
        target_coeffs = transform_semantic(pred[0]['coeff_3dmm'].reshape(-1), pred[1]['coeff_3dmm'].reshape(-1),
                                           crop_norm_ratio=crop_norm_ratio)

        with torch.no_grad():
            input_source = trans_image(source_img)[None].cuda()
            target_coeffs = target_coeffs[None].cuda()
            output_dict = self.net_G(input_source, target_coeffs)

            warp_image = output_dict['warp_image'].cpu().clamp_(-1, 1)
            fake_image = output_dict['fake_image'].cpu().clamp_(-1, 1)

        return tensor2pil(fake_image)

    def infer_batch(self,
                    source_batch: torch.Tensor,
                    target_batch: torch.Tensor,
                    save_folder: str = 'demo_images/out/',
                    save_name: str = 'reen.png',
                    ) -> torch.Tensor:
        """

        :param source_batch: (N,RGB,H,W), in [-1,1]
        :param target_batch: (N,RGB,H,W), in [-1,1]
        :param save_folder:
        :param save_name:
        :return: reen_batch, (N,RGB,H,W), in [-1,1]
        """
        assert source_batch.ndim == 4, 'batch input should be (N,C,H,W)'
        B, C, H, W = source_batch.shape

        reen_batch = torch.zeros((B, C, H, W), dtype=torch.float32, device=source_batch.device)
        for b_idx in range(B):
            source_img = source_batch[b_idx]
            target_img = target_batch[b_idx]

            # (C,H,W) [-1,1] Tensor to (H,W,C) [0,255] np.ndarray to PIL.Image
            source_img = ((source_img + 1.) * 127.5).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            target_img = ((target_img + 1.) * 127.5).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            source_img = Image.fromarray(source_img)
            target_img = Image.fromarray(target_img)

            reen_img = self.infer_single(source_img, target_img)

            if b_idx == 0 and save_folder is not '':
                reen_img.save(os.path.join(save_folder, save_name))

            reen_img = self.transform(reen_img)
            reen_batch[b_idx] = reen_img

        return reen_batch
    
