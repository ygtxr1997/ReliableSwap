import argparse
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F

import os
import cv2
from PIL import Image
import numpy as np

from hires.models.encoders.psp_encoders import GradualLandmarkEncoder
from hires.models.stylegan2.model import GPENEncoder
from hires.models.stylegan2.model import Generator, Decoder
from hires.models.nets import F_mapping

make_abs_path = lambda fn: os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), fn))


class EmptyArgs(object):
    def __init__(self):
        pass


class HiResImageInfer(torch.nn.Module):
    def __init__(self):
        super(HiResImageInfer, self).__init__()
        args = EmptyArgs()
        args.batch = 1
        args.size = 1024
        args.latent = 512
        args.n_mlp = 8
        args.channel_multiplier = 2
        args.coarse = 7
        args.least_size = 8
        args.largest_size = 512
        args.mapping_layers = 18
        args.mapping_fmaps = 512
        args.mapping_lrmul = 1
        args.mapping_nonlinearity = 'linear'
        self.args = args

        self.vgg_mean = torch.tensor([[[0.485]], [[0.456]], [[0.406]]],
                                requires_grad=False, device=torch.device(0))
        self.vgg_std = torch.tensor([[[0.229]], [[0.224]], [[0.225]]],
                               requires_grad=False, device=torch.device(0))

        self.trans = transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.to_tensor_1024 = transforms.Compose([
            transforms.Resize(1024),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.load_psp()
        self.load_h3r()
        self.load_bisenet()
        self._load_hires(args=self.args)
        print('[HiRes] model loaded.')

    def _load_hires(self, args):
        device = 0
        encoder_lmk = GradualLandmarkEncoder(106 * 2).to(device)
        encoder_target = GPENEncoder(args.largest_size).to(device)
        generator = Generator(args.size, args.latent, args.n_mlp).to(device)
        decoder = Decoder(args.least_size, args.size).to(device)
        bald_model = F_mapping(mapping_lrmul=args.mapping_lrmul, mapping_layers=args.mapping_layers,
                               mapping_fmaps=args.mapping_fmaps, mapping_nonlinearity=args.mapping_nonlinearity).to(device)
        bald_model.eval()

        encoder_lmk = torch.nn.parallel.DataParallel(encoder_lmk)
        encoder_target = torch.nn.parallel.DataParallel(encoder_target)
        decoder = torch.nn.parallel.DataParallel(decoder)
        generator = torch.nn.parallel.DataParallel(generator)
        bald_model = torch.nn.parallel.DataParallel(bald_model)

        hires_ckpt = make_abs_path('./weights/CELEBA-HQ-1024.pt')
        hires_weights = torch.load(hires_ckpt,  map_location=torch.device('cpu'))

        encoder_lmk.load_state_dict(hires_weights["encoder_lmk"])
        encoder_target.load_state_dict(hires_weights["encoder_target"])
        decoder.load_state_dict(hires_weights["decoder"])
        generator.load_state_dict(hires_weights["generator"])
        bald_model.load_state_dict(hires_weights["bald_model"])

        self.encoder_lmk = encoder_lmk
        self.encoder_target = encoder_target
        self.decoder = decoder
        self.generator = generator
        self.bald_model = bald_model

    def load_psp(self):
        from psp.models.psp import pSp
        psp_opts = argparse.ArgumentParser
        psp_opts.encoder_type = 'GradualStyleEncoder'
        psp_opts.checkpoint_path = make_abs_path('./weights/psp_ffhq_encode.pt')
        psp_opts.start_from_latent_avg = True
        psp_opts.stylegan_size = 1024
        psp_opts.input_nc = 3
        psp_opts.device = 0
        psp_model = pSp(psp_opts)
        psp_model.eval()
        self.psp_model = psp_model.cuda(0)

    def load_h3r(self):
        from h3r.torchalign import FacialLandmarkDetector
        h3r_folder = make_abs_path('/gavin/code/FaceSwapping/modules/third_party/h3r/models/lapa/hrnet18_256x256_p2')
        h3r_model = FacialLandmarkDetector(root=h3r_folder)
        h3r_model.eval()
        self.h3r_model = h3r_model.cuda(0)

    def load_bisenet(self):
        from bisenet.bisenet import BiSeNet
        bisenet_model = BiSeNet(n_classes=19)
        bisenet_model.load_state_dict(
            torch.load(
                "/gavin/datasets/hanbang/79999_iter.pth",
                map_location="cpu",
            )
        )
        bisenet_model.eval()
        self.bisenet_model = bisenet_model.cuda(0)

    @staticmethod
    def _cords_to_map_np(cords, img_size=(256, 256), sigma=6):
        result = np.zeros(img_size + cords.shape[0:1], dtype='uint8')
        for i, point in enumerate(cords):
            if point[0] == -1 or point[1] == -1:
                continue
            xx, yy = np.meshgrid(np.arange(img_size[1]), np.arange(img_size[0]))
            x = np.exp(-((yy - int(point[0])) ** 2 + (xx - int(point[1])) ** 2) / (2 * sigma ** 2))
            result[..., i] = x
        return result

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

        # face_part_ids = [1, 6, 7, 4, 5, 3, 2, 11, 12] if no_neck else [1, 6, 7, 4, 5, 3, 2, 11, 12, 17]
        # mouth_id = 10
        # hair_id = 13
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
    def demo_infer(self,
                   s_img, s_code, s_map,
                   t_img, t_code, t_map, t_mask,
                   ):
        """

        :return:
        """

        ''' latent: (18,512), in [-32,32]
            lmk: (98,2), in [0,H/W]
            seg: (19, H, W), in [-3,6]
        '''
        args = self.args
        encoder_lmk = self.encoder_lmk
        encoder_target = self.encoder_target
        decoder = self.decoder
        generator = self.generator
        bald_model = self.bald_model

        s_frame_code = s_code
        t_frame_code = t_code

        input_map = torch.cat([s_map, t_map], dim=1)
        t_mask = t_mask.unsqueeze_(1).float()

        t_lmk_code = encoder_lmk(input_map)

        zero_latent = torch.zeros((args.batch, 18 - args.coarse, 512)).to(0).detach()
        t_lmk_code = torch.cat([t_lmk_code, zero_latent], dim=1)
        fusion_code = s_frame_code + t_lmk_code

        fusion_code = torch.cat([fusion_code[:, :18 - args.coarse], t_frame_code[:, 18 - args.coarse:]], dim=1)
        fusion_code = bald_model(fusion_code.view(fusion_code.size(0), -1), 2)
        fusion_code = fusion_code.view(t_frame_code.size())

        source_feas = generator([fusion_code], input_is_latent=True, randomize_noise=False)
        target_feas = encoder_target(t_img)

        blend_img = decoder(source_feas, target_feas, t_mask)

        return blend_img

    @torch.no_grad()
    def image_infer(self, source_pil, target_pil):
        """

        :param source_pil: PIL.Image, in [0,255]
        :param target_pil: PIL.Image, in [0,255]
        :return:
        """
        # source_pil = Image.open('infer_images/00030.jpg')
        # target_pil = Image.open('infer_images/00020.jpg')

        source_tensor = self.trans(source_pil).cuda(0)
        target_tensor = self.trans(target_pil).cuda(0)
        source_tensor = source_tensor.unsqueeze(0)
        target_tensor = target_tensor.unsqueeze(0)

        ''' 1. psp '''
        _, s_code = self.psp_model(source_tensor, return_latents=True)  # (1,18,512)
        _, t_code = self.psp_model(target_tensor, return_latents=True)
        # print('s_code:', s_code.min(), s_code.max())

        ''' 2. h3r '''
        s_lmk = self.h3r_model(source_pil, device=0)
        t_lmk = self.h3r_model(target_pil, device=0)
        s_lmk = s_lmk[0].cpu().numpy()  # (106,2)
        t_lmk = t_lmk[0].cpu().numpy()
        s_map = self._cords_to_map_np(s_lmk)  # (256,256,106)
        t_map = self._cords_to_map_np(t_lmk)
        s_map = torch.FloatTensor(s_map).cuda(0).unsqueeze(0).transpose(1, 3)  # (1,106,256,256)
        t_map = torch.FloatTensor(t_map).cuda(0).unsqueeze(0).transpose(1, 3)
        # print('s_map:', s_map.min(), s_map.max())

        ''' 3. bisenet '''
        seg = self._get_any_mask(target_tensor)
        # self.save_tensor_to_img(seg, './infer_images/target_seg.jpg', scale=1)
        seg = seg.permute(0, 2, 3, 1)[0].cpu().numpy()
        t_mask = self._encode_segmentation_rgb(seg)
        t_mask = cv2.resize(t_mask, (1024, 1024))
        t_mask = t_mask.transpose((2, 0, 1)).astype(np.float) / 255.0
        t_mask = t_mask[0] + t_mask[1]
        t_mask = cv2.dilate(t_mask, np.ones((50, 50)), borderType=cv2.BORDER_CONSTANT, borderValue=0)
        t_mask = torch.FloatTensor(t_mask).cuda(0).unsqueeze(0)  # (1,1024,1024)
        # print('t_mask:', t_mask.min(), t_mask.max())

        ''' 4. post process '''
        s_img = source_tensor
        t_img = self.to_tensor_1024(target_pil)
        t_img = t_img.unsqueeze(0)

        result = self.demo_infer(s_img, s_code, s_map,
                                 t_img, t_code, t_map, t_mask,
                                 )  # in [-1,1]
        # result = (result.clamp(-1, 1) + 1) / 2
        # self.save_tensor_to_img(result, './infer_images/result.jpg', scale=256)
        return result


if __name__ == "__main__":
    import thop

    with torch.no_grad():
        img_pil = Image.fromarray(np.ones((256, 256, 3), dtype=np.uint8))
        img_tensor = torch.randn(1, 3, 256, 256).cuda()
        hires_image_infer = HiResImageInfer().cuda()

        _, s_code = hires_image_infer.psp_model(img_tensor, return_latents=True)  # (1,18,512)
        flops, params = thop.profile(hires_image_infer.psp_model, inputs=(img_tensor, True), verbose=False)
        print('#Params=%.2fM, GFLOPS=%.2f' % (params / 1e6, flops / 1e9))

        s_lmk = hires_image_infer.h3r_model(img_pil, device=0)
        s_lmk = s_lmk[0].cpu().numpy()  # (106,2)
        s_map = hires_image_infer._cords_to_map_np(s_lmk)  # (256,256,106)
        s_map = torch.FloatTensor(s_map).cuda(0).unsqueeze(0).transpose(1, 3)  # (1,106,256,256)

        s_img = img_tensor
        t_img = hires_image_infer.to_tensor_1024(img_pil).cuda()
        t_img = t_img.unsqueeze(0)

        args = hires_image_infer.args
        encoder_lmk = hires_image_infer.encoder_lmk
        encoder_target = hires_image_infer.encoder_target
        decoder = hires_image_infer.decoder
        generator = hires_image_infer.generator
        bald_model = hires_image_infer.bald_model

        s_frame_code = s_code
        t_frame_code = s_code

        input_map = torch.cat([s_map, s_map], dim=1)
        t_mask = torch.randn(1, 1, 256, 256).cuda()
        # t_mask = t_mask.unsqueeze_(1).float()

        t_lmk_code = encoder_lmk(input_map)

        zero_latent = torch.zeros((args.batch, 18 - args.coarse, 512)).cuda()
        t_lmk_code = torch.cat([t_lmk_code, zero_latent], dim=1)
        fusion_code = s_frame_code + t_lmk_code

        fusion_code = torch.cat([fusion_code[:, :18 - args.coarse], t_frame_code[:, 18 - args.coarse:]], dim=1)
        fusion_code = bald_model(fusion_code.view(fusion_code.size(0), -1), 2)
        fusion_code = fusion_code.view(t_frame_code.size())
        print(fusion_code.shape)

        source_feas = generator([fusion_code], input_is_latent=True, randomize_noise=False)
        print(len(source_feas))
        flops, params = thop.profile(generator, inputs=([fusion_code], True, False), verbose=False)
        print('#Params=%.2fM, GFLOPS=%.2f' % (params / 1e6, flops / 1e9))

        target_feas = encoder_target(t_img)
        flops, params = thop.profile(encoder_target, inputs=(t_img,), verbose=True, report_missing=True)
        print('#Params=%.2fM, GFLOPS=%.2f' % (params / 1e6, flops / 1e9))

        blend_img = decoder(source_feas, target_feas, t_mask)
        print(blend_img.shape)
        flops, params = thop.profile(decoder, inputs=(source_feas, target_feas, t_mask,), verbose=True)
        print('#Params=%.2fM, GFLOPS=%.2f' % (params / 1e6, flops / 1e9))

