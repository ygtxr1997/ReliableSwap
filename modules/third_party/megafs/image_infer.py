import argparse
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F

import os
import cv2
from PIL import Image
import numpy as np

from megafs.inference.megafs import resnet50, HieRFE, Generator, FaceTransferModule

make_abs_path = lambda fn: os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), fn))


class SoftErosion(nn.Module):
    def __init__(self, kernel_size=15, threshold=0.6, iterations=1):
        super(SoftErosion, self).__init__()
        r = kernel_size // 2
        self.padding = r
        self.iterations = iterations
        self.threshold = threshold

        # Create kernel
        y_indices, x_indices = torch.meshgrid(torch.arange(0., kernel_size), torch.arange(0., kernel_size))
        dist = torch.sqrt((x_indices - r) ** 2 + (y_indices - r) ** 2)
        kernel = dist.max() - dist
        kernel /= kernel.sum()
        kernel = kernel.view(1, 1, *kernel.shape)
        self.register_buffer('weight', kernel)

    def forward(self, x):
        x = x.float()
        for i in range(self.iterations - 1):
            x = torch.min(x, F.conv2d(x, weight=self.weight, groups=x.shape[1], padding=self.padding))
        x = F.conv2d(x, weight=self.weight, groups=x.shape[1], padding=self.padding)

        mask = x >= self.threshold
        x[mask] = 1.0
        x[~mask] /= x[~mask].max()

        return x, mask


class MegaFSImageInfer(torch.nn.Module):
    def __init__(self, swap_type: str = 'ftm'):
        super(MegaFSImageInfer, self).__init__()
        self.size = 1024
        self.swap_type = swap_type

        self.vgg_mean = torch.tensor([[[0.485]], [[0.456]], [[0.406]]],
                                     requires_grad=False, device=torch.device(0))
        self.vgg_std = torch.tensor([[[0.229]], [[0.224]], [[0.225]]],
                                    requires_grad=False, device=torch.device(0))

        # Model
        # "ftm"    "injection"     "lcr"
        num_blocks = 3 if self.swap_type == "ftm" else 1
        latent_split = [4, 6, 8]
        num_latents = 18
        swap_indice = 4
        self.encoder = HieRFE(resnet50(False), num_latents=latent_split, depth=50).cuda()
        self.swapper = FaceTransferModule(num_blocks=num_blocks, swap_indice=swap_indice, num_latents=num_latents,
                                          typ=self.swap_type).cuda()
        ckpt_e = make_abs_path("./inference/checkpoint/{}_final.pth".format(self.swap_type))
        if ckpt_e is not None:
            print("load encoder & swapper:", ckpt_e)
            ckpts = torch.load(ckpt_e, map_location=torch.device("cpu"))
            self.encoder.load_state_dict(ckpts["e"])
            self.swapper.load_state_dict(ckpts["s"])
            del ckpts

        self.generator = Generator(self.size, 512, 8, channel_multiplier=2).cuda()
        ckpt_f = make_abs_path("./inference/checkpoint/stylegan2-ffhq-config-f.pth")
        if ckpt_f is not None:
            print("load generator:", ckpt_f)
            ckpts = torch.load(ckpt_f, map_location=torch.device("cpu"))
            self.generator.load_state_dict(ckpts["g_ema"], strict=False)
            del ckpts

        self.smooth_mask = SoftErosion(kernel_size=17, threshold=0.9, iterations=7).cuda()

        self.encoder.eval()
        self.swapper.eval()
        self.generator.eval()
        self.load_bisenet()
        print('[MegaFS] model loaded.')

    def load_bisenet(self):
        from bisenet.bisenet import BiSeNet
        bisenet_model = BiSeNet(n_classes=19)
        bisenet_model.load_state_dict(
            torch.load("/gavin/datasets/hanbang/79999_iter.pth", map_location="cpu")
        )
        bisenet_model.eval()
        self.bisenet_model = bisenet_model.cuda(0)

    def _get_any_mask(self, img, par=None, normalized=False):
        # [0, 'background', 1 'skin', 2 'l_brow', 3 'r_brow', 4 'l_eye', 5 'r_eye',
        # 6 'eye_g', 7 'l_ear', 8 'r_ear', 9 'ear_r',  10 'nose', 11 'mouth', 12 'u_lip',
        # 13 'l_lip', 14 'neck', 15 'neck_l', 16 'cloth', 17 'hair', 18 'hat']
        ori_size = img.size()[-1]
        with torch.no_grad():
            img = F.interpolate(img, size=512, mode="nearest", )
            if not normalized:
                img = img * 0.5 + 0.5
                img = img.sub(self.vgg_mean.detach()).div(self.vgg_std.detach())
            out = self.bisenet_model(img)[0]
            parsing = out.softmax(1).argmax(1)
        # mask = torch.zeros_like(parsing)
        # for p in par:
        #     mask = mask + ((parsing == p).float())
        # mask = mask.unsqueeze(1)
        # mask = F.interpolate(mask, size=ori_size, mode="bilinear", align_corners=True)
        # return mask
        mask = parsing.float()
        mask = mask.unsqueeze(1)
        mask = F.interpolate(mask, size=ori_size, mode="bilinear", align_corners=True)
        return mask

    @staticmethod
    def _encode_segmentation_rgb(segmentation, no_neck=True):
        parse = segmentation[:, :, 0]

        face_part_ids = [1, 2, 3, 4, 5, 6, 10, 12, 13] if no_neck else [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 13, 14]
        mouth_id = 11
        hair_id = 17

        face_map = np.zeros([parse.shape[0], parse.shape[1]])
        mouth_map = np.zeros([parse.shape[0], parse.shape[1]])
        hair_map = np.zeros([parse.shape[0], parse.shape[1]])

        for valid_id in face_part_ids:
            valid_index = np.where(parse == valid_id)
            face_map[valid_index] = 255
        valid_index = np.where(parse == mouth_id)
        mouth_map[valid_index] = 255
        valid_index = np.where(parse == hair_id)
        hair_map[valid_index] = 255

        return np.stack([face_map, mouth_map, hair_map], axis=2)

    @staticmethod
    def save_tensor_to_img(tensor: torch.Tensor, path: str, scale=256):
        tensor = tensor.permute(0, 2, 3, 1)[0]  # in [0,1]
        tensor = tensor * scale
        tensor_np = tensor.cpu().numpy().astype(np.uint8)
        if tensor_np.shape[-1] == 1:  # channel dim
            tensor_np = tensor_np.repeat(3, axis=-1)
        tensor_img = Image.fromarray(tensor_np)
        tensor_img.save(path)

    @torch.no_grad()
    def image_infer(self, source_tensor: torch.Tensor, target_tensor: torch.Tensor):
        target_mask = self._get_any_mask(target_tensor)
        target_mask = target_mask[0].permute(1, 2, 0).cpu().numpy()  # (H,W,C)
        target_mask = self._encode_segmentation_rgb(target_mask)  # (H,W,C)

        ts = torch.cat([target_tensor, source_tensor], dim=0).cuda()
        lats, struct = self.encoder(ts)

        idd_lats = lats[1:]
        att_lats = lats[0].unsqueeze_(0)
        att_struct = struct[0].unsqueeze_(0)

        swapped_lats = self.swapper(idd_lats, att_lats)
        fake_swap, _ = self.generator(att_struct, [swapped_lats, None], randomize_noise=False)
        fake_swap = F.interpolate(fake_swap, size=(256, 256), mode='bilinear', align_corners=True)

        fake_swap_max = torch.max(fake_swap)
        fake_swap_min = torch.min(fake_swap)
        fake_swap = (fake_swap - fake_swap_min) / (fake_swap_max - fake_swap_min)  # in [0,1]
        fake_swap = fake_swap * 2 - 1  # in [-1,1]
        # denormed_fake_swap = (fake_swap - fake_swap_min) / (fake_swap_max - fake_swap_min) * 255.0

        ''' postprocess '''
        target_mask = torch.from_numpy(target_mask.transpose((2, 0, 1))).float().mul_(1/255.0).cuda()  # (C,H,W)
        face_mask_tensor = target_mask[0] + target_mask[1]  # face + mouth, (H,W)

        soft_face_mask_tensor, _ = self.smooth_mask(face_mask_tensor.unsqueeze_(0).unsqueeze_(0))
        fake_swap = fake_swap * soft_face_mask_tensor + target_tensor * (1 - soft_face_mask_tensor)

        return fake_swap
    
    def forward(self, source, target):
        return self.image_infer(source, target)


if __name__ == "__main__":
    source_pil = Image.open('./inference/imgs/source.jpg')
    target_pil = Image.open('./inference/imgs/target.jpg')

    trans = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    source_tensor = trans(source_pil).unsqueeze(0).cuda()
    target_tensor = trans(target_pil).unsqueeze(0).cuda()

    megafs_image_infer = MegaFSImageInfer().cuda().eval()
    result = megafs_image_infer.image_infer(source_tensor, target_tensor)
    result = (result.clamp(-1, 1) + 1) / 2
    megafs_image_infer.save_tensor_to_img(result, path='./inference/imgs/result.jpg')

    import thop

    net = megafs_image_infer
    flops, params = thop.profile(net, inputs=(source_tensor, target_tensor), verbose=False)
    print('#Params=%.2fM, GFLOPS=%.2f' % (params / 1e6, flops / 1e9))

