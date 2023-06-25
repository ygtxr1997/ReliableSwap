import torch
import torch.nn as nn
import argparse
import numpy as np
import torchvision
import os
from PIL import Image
from pathlib import Path
from tqdm import tqdm

from reenactment.LIA.networks.generator import Generator

make_abs_path = lambda fn: os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), fn))


def load_image(filename, size):
    img = Image.open(filename).convert('RGB')
    img = img.resize((size, size))
    img = np.asarray(img)
    img = np.transpose(img, (2, 0, 1))  # 3 x 256 x 256

    return img / 255.0


def img_preprocessing(img_path, size):
    img = load_image(img_path, size)  # [0, 1]
    img = torch.from_numpy(img).unsqueeze(0).float()  # [0, 1]
    imgs_norm = (img - 0.5) * 2.0  # [-1, 1]

    return imgs_norm


def vid_preprocessing(vid_path):
    vid_dict = torchvision.io.read_video(vid_path, pts_unit='sec')
    vid = vid_dict[0].permute(0, 3, 1, 2).unsqueeze(0)
    fps = vid_dict[2]['video_fps']
    vid_norm = (vid / 255.0 - 0.5) * 2.0  # [-1, 1]

    return vid_norm, fps


def save_video(vid_target_recon, save_path, fps):
    vid = vid_target_recon.permute(0, 2, 3, 4, 1)  # (N,C,T,H,W) -> (N,T,H,W,C)
    vid = vid.clamp(-1, 1).cpu()
    vid = ((vid - vid.min()) / (vid.max() - vid.min()) * 255).type('torch.ByteTensor')

    torchvision.io.write_video(save_path, vid[0], fps=fps)


def save_image(img_target_recon: torch.Tensor, batch_idx: int, save_path: str):
    assert img_target_recon.ndim == 4, 'image should be (N,C,H,W)'
    img = img_target_recon.permute(0, 2, 3, 1)  # (N,C,H,W) -> (N,H,W,C)
    img = img.clamp(-1, 1).cpu()
    img = ((img - img.min()) / (img.max() - img.min()) * 255).numpy().astype(np.uint8)
    img = Image.fromarray(img[batch_idx])

    img.save(save_path)


class LIAImageInfer(nn.Module):
    def __init__(self, args):
        super(LIAImageInfer, self).__init__()

        self.args = args

        if args.model == 'vox':
            model_path = 'checkpoints/vox.pt'
        elif args.model == 'taichi':
            model_path = 'checkpoints/taichi.pt'
        elif args.model == 'ted':
            model_path = 'checkpoints/ted.pt'
        else:
            raise NotImplementedError

        print('==> loading model')
        self.gen = Generator(args.size, args.latent_dim_style, args.latent_dim_motion, args.channel_multiplier).cuda()
        weight = torch.load(model_path, map_location=lambda storage, loc: storage)['gen']
        self.gen.load_state_dict(weight)
        self.gen.eval()

        print('==> loading data')
        self.save_path = args.save_folder
        os.makedirs(self.save_path, exist_ok=True)
        self.save_path = os.path.join(self.save_path, args.model + '_' + Path(args.source_path).stem + '_' + Path(args.driving_path).stem + '.png')
        self.img_source = img_preprocessing(args.source_path, args.size).cuda()
        self.img_drive = img_preprocessing(args.driving_path, args.size).cuda()

        # self.vid_target, self.fps = vid_preprocessing(args.driving_path)
        self.vid_target = self.img_drive.unsqueeze(dim=1)

    def run(self):
        print('==> running')
        with torch.no_grad():
            if self.args.model == 'ted':
                h_start = None
            else:
                h_start = self.gen.enc.enc_motion(self.vid_target[:, 0, :, :, :])  # (1,T,C,H,W)
                h_start = None
                # h_start = self.gen.enc.enc_motion(self.img_source)  # (1,T,C,H,W)

            img_target = self.vid_target[:, 0, :, :, :]
            img_recon = self.gen(self.img_source, img_target, h_start)
            save_image(img_recon, self.save_path)


class LIABatchInfer(nn.Module):
    def __init__(self,
                 model_name: str = 'vox',
                 size: int = 256,
                 latent_dim_style: int = 512,
                 latent_dim_motion: int = 20,
                 channel_multiplier: int = 1,
                 ):
        super(LIABatchInfer, self).__init__()

        self.root_folder = make_abs_path('./')
        self.model_name = model_name
        if model_name == 'vox':
            model_path = 'checkpoints/vox.pt'
        elif model_name == 'taichi':
            model_path = 'checkpoints/taichi.pt'
        elif model_name == 'ted':
            model_path = 'checkpoints/ted.pt'
        else:
            raise NotImplementedError

        print('==> loading LIA model')
        model_path = os.path.join(self.root_folder, model_path)
        self.gen = Generator(size, latent_dim_style, latent_dim_motion, channel_multiplier).cuda()
        weight = torch.load(model_path, map_location=lambda storage, loc: storage)['gen']
        self.gen.load_state_dict(weight)
        self.gen.eval()

    def forward(self,
                source_batch: torch.Tensor,
                target_batch: torch.Tensor,
                save_folder: str = 'demo_images/out/',
                save_name: str = 'reen.png',
                save_batch_idx: int = 0,
                ) -> torch.Tensor:
        """

        :param source_batch: (N,RGB,H,W), in [-1,1]
        :param target_batch: (N,RGB,H,W), in [-1,1]
        :param save_folder:
        :param save_name:
        :param save_batch_idx:
        :return: reen_batch, (N,RGB,H,W), in [-1,1]
        """
        with torch.no_grad():
            if self.model_name == 'ted':
                h_start = None
            else:
                # h_start = self.gen.enc.enc_motion(self.vid_target[:, 0, :, :, :])  # (1,T,C,H,W)
                h_start = None

            reen_batch = self.gen(source_batch, target_batch, h_start)
            if save_batch_idx is not None:
                save_image(reen_batch, batch_idx=save_batch_idx,
                           save_path=os.path.join(save_folder, save_name))
            reen_batch = reen_batch.clamp(-1, 1)

        return reen_batch

    def infer_batch(self,
                source_batch: torch.Tensor,
                target_batch: torch.Tensor,
                save_folder: str = 'demo_images/out/',
                save_name: str = 'reen.png',
                save_batch_idx: int = 0,
                ) -> torch.Tensor:
        """

        :param source_batch: (N,RGB,H,W), in [-1,1]
        :param target_batch: (N,RGB,H,W), in [-1,1]
        :param save_folder:
        :param save_name:
        :return: reen_batch, (N,RGB,H,W), in [-1,1]
        """
        return self(source_batch, target_batch, save_folder, save_name, save_batch_idx)


if __name__ == '__main__':
    # training params
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--channel_multiplier", type=int, default=1)
    parser.add_argument("--model", type=str, choices=['vox', 'taichi', 'ted'], default='')
    parser.add_argument("--latent_dim_style", type=int, default=512)
    parser.add_argument("--latent_dim_motion", type=int, default=20)
    parser.add_argument("--source_path", type=str, default='')
    parser.add_argument("--driving_path", type=str, default='')
    parser.add_argument("--save_folder", type=str, default='./res')
    args = parser.parse_args()

    # demo
    demo = LIAImageInfer(args)
    demo.run()
