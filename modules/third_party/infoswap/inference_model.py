import os
import cv2
import time
import argparse
import logging
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms as transforms
from datetime import datetime, timedelta

from infoswap.utils import laplacian_blending, make_image
from infoswap.modules.encoder128 import Backbone128
from infoswap.modules.iib import IIB
from infoswap.modules.aii_generator import AII512
from infoswap.modules.decoder512 import UnetDecoder512

from infoswap.preprocess.mtcnn import MTCNN


make_abs_path = lambda fn: os.path.join(os.path.dirname(os.path.realpath(__file__)), fn)


def to_np(t: torch.Tensor):
    t = t.detach()
    if t.is_cuda:
        t = t.cpu()
    return t.numpy()


class InfoSwapInference(torch.nn.Module):
    def __init__(self):
        super(InfoSwapInference, self).__init__()
        self.ib_mode = 'smooth'
        self.N = 10
        self.G = None
        self.iib = None
        self.encoder = None
        self.decoder = None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.mtcnn = MTCNN()
        self.TRANSFORMS = transforms.Compose([
            transforms.Resize((512, 512), interpolation=2),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def load_model(self):
        ROOT = {
            'smooth': {'root': make_abs_path('weights/checkpoints_512/w_kernel_smooth'),
                       'path': 'ckpt_ks_*.pth'},
            'no_smooth': {'root': make_abs_path('weights/checkpoints_512/wo_kernel_smooth'),
                          'path': 'ckpt_*.pth'}
        }

        ib_mode = self.ib_mode
        device = self.device

        root = ROOT[ib_mode]['root']
        path = ROOT[ib_mode]['path']

        pathG = path.replace('*', 'G')
        pathE = path.replace('*', 'E')
        pathI = path.replace('*', 'I')

        encoder = Backbone128(50, 0.6, 'ir_se').eval().to(device)
        state_dict = torch.load(make_abs_path('weights/model_128_ir_se50.pth'),
                                map_location=device)
        encoder.load_state_dict(state_dict, strict=True)

        G = AII512().eval().to(device)
        decoder = UnetDecoder512().eval().to(device)

        # Define Information Bottlenecks:
        N = self.N
        _ = encoder(torch.rand(1, 3, 128, 128).to(device), cache_feats=True)
        _readout_feats = encoder.features[:(N + 1)]  # one layer deeper than the z_attrs needed
        in_c = sum(map(lambda f: f.shape[-3], _readout_feats))
        out_c_list = [_readout_feats[i].shape[-3] for i in range(N)]

        iib = IIB(in_c, out_c_list, device, smooth=self.ib_mode == 'smooth', kernel_size=1)
        iib = iib.eval()

        G.load_state_dict(torch.load(os.path.join(root, pathG), map_location=device), strict=True)
        print("Successfully load G!")
        decoder.load_state_dict(torch.load(os.path.join(root, pathE), map_location=device), strict=True)
        print("Successfully load Decoder!")
        # 3) load IIB:
        iib.load_state_dict(torch.load(os.path.join(root, pathI), map_location=device),
                            strict=ib_mode == 'smooth')
        print("Successfully load IIB!")

        self.G = G
        self.iib = iib
        self.encoder = encoder
        self.decoder = decoder
        return

    def infer_batch(self, source_batch, target_batch):
        """

        :param source_batch: (N,RGB,H,W), in [-1,1]
        :param target_batch: (N,RGB,H,W), in [-1,1]
        :return: result_batch: (N,RGB,H,W), in [-1,1]
        """
        device = self.device
        in_size = source_batch.shape[-1]
        """ load pre-calculated mean and std: """
        param_dict = []
        for i in range(self.N + 1):
            state = torch.load(make_abs_path(f'modules/weights128/readout_layer{i}.pth'), map_location=device)
            n_samples = state['n_samples'].float()
            std = torch.sqrt(state['s'] / (n_samples - 1)).to(device)
            neuron_nonzero = state['neuron_nonzero'].float()
            active_neurons = (neuron_nonzero / n_samples) > 0.01
            param_dict.append([state['m'].to(device), std, active_neurons])

        """ inference: """
        Xs = F.interpolate(source_batch, size=512, mode="bilinear", align_corners=True)
        Xs = (Xs.permute(0, 2, 3, 1) + 1.) * 127.5
        assert Xs.shape[0] == 1
        Xs = Image.fromarray(Xs[0].cpu().numpy()[:, :, ::-1].astype(np.uint8))
        face_s = self.mtcnn.align_multi(Xs, min_face_size=64., thresholds=[0.6, 0.7, 0.8], factor=0.707,
                                        crop_size=(512, 512))
        if face_s is not None:
            Xs = face_s[0]
        else:
            print('[Warning] no face found here!')

        Xs = self.TRANSFORMS(Xs).unsqueeze(0)
        Xs = Xs.to(device)

        N = self.N
        for idx in range(1):
            with torch.no_grad():
                '''(1) load Xt: '''
                xt = F.interpolate(target_batch, size=512, mode="bilinear", align_corners=True)
                xt = (xt.permute(0, 2, 3, 1) + 1.) * 127.5
                # print(xt.shape)

                Xt = Image.fromarray(xt[0].cpu().numpy()[:, :, ::-1].astype(np.uint8))
                out = self.mtcnn.align_multi(Xt, min_face_size=64., thresholds=[0.6, 0.7, 0.7],
                                        crop_size=(512, 512), reverse=True)
                if out is not None:
                    faces, tfm_invs, boxes = out
                    if faces is not None:
                        ss = 0
                        fi = 0
                        for j in range(len(boxes)):
                            box = boxes[j]
                            w = box[2] - box[0] + 1.0
                            h = box[3] - box[1] + 1.0
                            s = w * h
                            if s > ss:
                                ss = s
                                fi = j
                        Xt = faces[fi]
                        tfm_inv = tfm_invs[fi]
                else:
                    try:
                        mini = 20.
                        th1, th2, th3 = 0.6, 0.6, 0.6
                        while out is None:
                            out = self.mtcnn.align_multi(Xt, min_face_size=mini, thresholds=[th1, th2, th3],
                                                    crop_size=(512, 512), reverse=True)
                            if out is not None:
                                faces, tfm_invs, boxes = out
                                ss = 0
                                fi = 0
                                for j in range(len(boxes)):
                                    box = boxes[j]
                                    w = box[2] - box[0] + 1.0
                                    h = box[3] - box[1] + 1.0
                                    s = w * h
                                    if s > ss:
                                        ss = s
                                        fi = j
                                Xt = faces[fi]
                                tfm_inv = tfm_invs[fi]
                            else:
                                th1 *= 0.8
                                th2 *= 0.8
                                th2 *= 0.8
                                mini *= 0.8
                    except Exception as e:
                        print(e)
                        continue

                '''(2) generate Y: '''
                B = 1
                Xt = self.TRANSFORMS(Xt).unsqueeze(0).to(device)
                X_id = self.encoder(
                    F.interpolate(torch.cat((Xs, Xt), dim=0)[:, :, 37:475, 37:475], size=[128, 128],
                                  mode='bilinear', align_corners=True),
                    cache_feats=True
                )
                # 01 Get Inter-features After One Feed-Forward:
                # batch size is 2 * B, [:B] for Xs and [B:] for Xt
                min_std = torch.tensor(0.01, device=device)
                readout_feats = [(self.encoder.features[i] - param_dict[i][0]) / torch.max(param_dict[i][1], min_std)
                                 for i in range(N + 1)]

                # 02 information restriction:
                X_id_restrict = torch.zeros_like(X_id).to(device)  # [2*B, 512]
                Xt_feats, X_lambda = [], []
                Xt_lambda = []
                Rs_params, Rt_params = [], []
                for i in range(N):
                    R = self.encoder.features[i]  # [2*B, Cr, Hr, Wr]
                    Z, lambda_, _ = getattr(self.iib, f'iba_{i}')(
                        R, readout_feats,
                        m_r=param_dict[i][0], std_r=param_dict[i][1],
                        active_neurons=param_dict[i][2],
                    )
                    X_id_restrict += self.encoder.restrict_forward(Z, i)

                    Rs, Rt = R[:B], R[B:]
                    lambda_s, lambda_t = lambda_[:B], lambda_[B:]

                    m_s = torch.mean(Rs, dim=0)  # [C, H, W]
                    std_s = torch.mean(Rs, dim=0)
                    Rs_params.append([m_s, std_s])

                    eps_s = torch.randn(size=Rt.shape).to(Rt.device) * std_s + m_s
                    feat_t = Rt * (1. - lambda_t) + lambda_t * eps_s

                    Xt_feats.append(feat_t)  # only related with lambda
                    Xt_lambda.append(lambda_t)

                X_id_restrict /= float(N)
                Xs_id = X_id_restrict[:B]
                Xt_feats[0] = Xt
                Xt_attr, Xt_attr_lamb = self.decoder(Xt_feats, lambs=Xt_lambda, use_lambda=True)

                Y = self.G(Xs_id, Xt_attr, Xt_attr_lamb)

            '''(3) save Y: '''
            img_Y = (Y[0].cpu().numpy().transpose([1, 2, 0]) * 0.5 + 0.5) * 255
            img_Y = img_Y.astype(np.uint8)

            _, H, W, _ = xt.shape
            frame = cv2.warpAffine(img_Y.astype(np.float32), tfm_inv.astype(np.float32),
                                   dsize=(int(W), int(H)), borderValue=0)  # (BGR)

            mask = np.zeros(img_Y.shape, img_Y.dtype)
            mask[37:475, 90:422, :] = 1  # 90:422
            mask = cv2.warpAffine(mask,
                                  tfm_inv.astype(np.float32), dsize=(int(W), int(H)),
                                  borderValue=0)  # can not set cv2.BORDER_TRANSPARENT !

            # plt.imsave('infer_images/frame.jpg', cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_BGR2RGB))
            xt = xt[0].cpu().numpy()[:, :, ::-1]  # (RGB) to (BGR)
            # plt.imsave('infer_images/xt.jpg', cv2.cvtColor(xt.astype(np.uint8), cv2.COLOR_BGR2RGB))
            res = laplacian_blending(A=frame, B=xt, m=mask)  # (BGR)

            res = res[:, :, ::-1]  # (H,W,RGB)
            res_batch = torch.from_numpy(res.copy()).unsqueeze(0)  # (N,H,W,RGB)
            res_batch = res_batch.permute(0, 3, 1, 2)  # (N,RGB,H,W), in [0,255]
            res_batch = (res_batch - 127.5) / 127.5
            res_batch = F.interpolate(res_batch, size=in_size, mode="bilinear", align_corners=True)

            # res_plt = Image.fromarray(((res_batch[0] + 1.) * 127.5).permute(1, 2, 0).numpy().astype(np.uint8))
            # res_plt.save('infer_images/res.jpg')
            return res_batch  # (N,RGB,H,W), in [-1,1]

    def forward(self, source, target, infer=True):
        return self.infer_batch(source_batch=source, target_batch=target)


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    s_img = Image.open('infer_images/source/source.jpg')
    t_img = Image.open('infer_images/target/target.jpg')
    TRANSFORMS = transforms.Compose([
        transforms.Resize((256, 256), interpolation=2),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    s_tensor = TRANSFORMS(s_img)[None, :, :, :]
    t_tensor = TRANSFORMS(t_img)[None, :, :, :]

    infoswap_model = InfoSwapInference()
    infoswap_model.load_model()
    infoswap_model.infer_batch(s_tensor, t_tensor)

    import thop

    flops, params = thop.profile(infoswap_model, inputs=(s_tensor, t_tensor,), verbose=False)
    print('#Params=%.2fM, GFLOPS=%.2f' % (params / 1e6, flops / 1e9))
