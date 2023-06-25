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

from preprocess.mtcnn import MTCNN

mtcnn = MTCNN()

TRANSFORMS = transforms.Compose([
    transforms.Resize((512, 512), interpolation=2),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


def to_np(t: torch.Tensor):
    t = t.detach()
    if t.is_cuda:
        t = t.cpu()
    return t.numpy()


def inference(src_img_path, tar_dir, save_dir):
    """
    :param src_img_path: path to a source image
    :param tar_dir: path to the dir of target images
    :return: no return
    """
    os.makedirs(save_dir, exist_ok=True)
    test_date = str(datetime.strptime(time.strftime(
        "%a, %d %b %Y %H:%M:%S", time.localtime()), "%a, %d %b %Y %H:%M:%S") + timedelta(hours=12)).split(' ')[
        0]
    save_dir = os.path.join(save_dir, test_date)
    os.makedirs(save_dir, exist_ok=True)

    logger = logging.getLogger('inference')
    logger.setLevel(logging.DEBUG)
    logger.propagate = True
    train_handler = logging.FileHandler(filename=os.path.join(save_dir, f'similarity_{test_date}.log'))
    train_formatter = logging.Formatter('%(message)s')
    train_handler.setFormatter(train_formatter)
    logger.addHandler(train_handler)

    if tar_dir.endswith('.png') or tar_dir.endswith('.jpg'):
        tar_list = [tar_dir, ]
    else:
        tmp_list = [f for f in os.listdir(tar_dir) if f.endswith('jpg') or f.endswith('png')]
        tar_list = sorted(tmp_list)
    M = len(tar_list)

    """ load pre-calculated mean and std: """
    param_dict = []
    for i in range(N + 1):
        state = torch.load(f'./modules/weights128/readout_layer{i}.pth', map_location=device)
        n_samples = state['n_samples'].float()
        std = torch.sqrt(state['s'] / (n_samples - 1)).to(device)
        neuron_nonzero = state['neuron_nonzero'].float()
        active_neurons = (neuron_nonzero / n_samples) > 0.01
        param_dict.append([state['m'].to(device), std, active_neurons])

    """ inference: """
    Xs = cv2.imread(src_img_path)
    Xs = Image.fromarray(Xs)
    face_s = mtcnn.align_multi(Xs, min_face_size=64., thresholds=[0.6, 0.7, 0.8], factor=0.707, crop_size=(512, 512))
    if face_s is not None:
        Xs = face_s[0]
    else:
        print('s')
        Xs = None
    Xs = TRANSFORMS(Xs).unsqueeze(0)
    Xs = Xs.to(device)

    for idx in range(M):
        tar_img_path = os.path.join(tar_dir, tar_list[idx])
        prefix = tar_list[idx].split('.')[0]
        suffix = tar_img_path.split('.')[-1]
        save_path = os.path.join(save_dir, prefix + '_gen.' + suffix)
        if os.path.exists(save_path):
            continue

        with torch.no_grad():
            '''(1) load Xt: '''
            print(tar_img_path, end=', ')
            xt = cv2.imread(tar_img_path)
            print(xt.shape)

            Xt = Image.fromarray(xt)
            out = mtcnn.align_multi(Xt, min_face_size=64., thresholds=[0.6, 0.7, 0.7],
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
                        out = mtcnn.align_multi(Xt, min_face_size=mini, thresholds=[th1, th2, th3],
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
                    plt.imsave(save_path, cv2.cvtColor(xt.astype(np.uint8), cv2.COLOR_RGB2BGR))
                    plt.close()
                    continue

            '''(2) generate Y: '''
            B = 1
            Xt = TRANSFORMS(Xt).unsqueeze(0).to(device)
            X_id = encoder(
                F.interpolate(torch.cat((Xs, Xt), dim=0)[:, :, 37:475, 37:475], size=[128, 128],
                              mode='bilinear', align_corners=True),
                cache_feats=True
            )
            # 01 Get Inter-features After One Feed-Forward:
            # batch size is 2 * B, [:B] for Xs and [B:] for Xt
            min_std = torch.tensor(0.01, device=device)
            readout_feats = [(encoder.features[i] - param_dict[i][0]) / torch.max(param_dict[i][1], min_std)
                             for i in range(N + 1)]

            # 02 information restriction:
            X_id_restrict = torch.zeros_like(X_id).to(device)  # [2*B, 512]
            Xt_feats, X_lambda = [], []
            Xt_lambda = []
            Rs_params, Rt_params = [], []
            for i in range(N):
                R = encoder.features[i]  # [2*B, Cr, Hr, Wr]
                Z, lambda_, _ = getattr(iib, f'iba_{i}')(
                    R, readout_feats,
                    m_r=param_dict[i][0], std_r=param_dict[i][1],
                    active_neurons=param_dict[i][2],
                )
                X_id_restrict += encoder.restrict_forward(Z, i)

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
            Xt_attr, Xt_attr_lamb = decoder(Xt_feats, lambs=Xt_lambda, use_lambda=True)

            Y = G(Xs_id, Xt_attr, Xt_attr_lamb)
            save_path_Y = os.path.join(save_dir, prefix + 'result.' + suffix)
            print(Y.shape)
            save_Y: np.ndarray = (Y.cpu().numpy()[0] + 1) * 127.5
            cv2.imwrite(save_path_Y, save_Y.transpose([1, 2, 0]),
                        # [int(cv2.IMWRITE_PNG_COMPRESSION), 0]
                        )
            # exit(0)

            encoder.features = []

            # log identity similarities:
            Y_id_gt = encoder(
                F.interpolate(Y[:, :, 37:475, 37:475], size=[128, 128], mode='bilinear', align_corners=True),
                cache_feats=False
            )
            Xs_id_gt, Xt_id_gt = X_id[:B], X_id[B:]
            msg = ''
            msg += "cos<Xs, Xt>=%.3f | " % torch.cosine_similarity(Xs_id_gt, Xt_id_gt,
                                                                   dim=1).mean().detach().cpu().numpy()
            msg += "cos<Y, Xt>=%.3f | " % torch.cosine_similarity(Xt_id_gt, Y_id_gt,
                                                                  dim=1).mean().detach().cpu().numpy()
            msg += "cos<Y, Xs>=%.3f | " % torch.cosine_similarity(Xs_id_gt, Y_id_gt,
                                                                  dim=1).mean().detach().cpu().numpy()
            logger.info(msg)

        '''(3) save Y: '''
        I = [Xs, Xt, Y]
        image = make_image(I, 1)
        save_path_Y = os.path.join(save_dir, prefix + '_xs_st_y.' + suffix)
        print("save path Y: ", save_path_Y)
        cv2.imwrite(save_path_Y, image.transpose([1, 2, 0]),
                    # [int(cv2.IMWRITE_PNG_COMPRESSION), 0]
                    )

        img_Y = (Y[0].cpu().numpy().transpose([1, 2, 0]) * 0.5 + 0.5) * 255
        img_Y = img_Y.astype(np.uint8)
        H, W, _ = xt.shape
        frame = cv2.warpAffine(img_Y.astype(np.float32), tfm_inv.astype(np.float32),
                               dsize=(int(W), int(H)), borderValue=0)

        mask = np.zeros(img_Y.shape, img_Y.dtype)
        mask[37:475, 90:422, :] = 1  # 90:422
        mask = cv2.warpAffine(mask,
                              tfm_inv.astype(np.float32), dsize=(int(W), int(H)),
                              borderValue=0)  # can not set cv2.BORDER_TRANSPARENT !
        cv2.imwrite(os.path.join(save_dir, 'mask.jpg'), mask)
        print(save_path)
        # try:
        #     src = np.array([255., 255., 1.]).reshape(3, 1)
        #     x, y = np.matmul(tfm_inv, src)
        #     print(x, y)
        #
        #     m = np.zeros(img_Y.shape, img_Y.dtype)
        #     m[40:472, 80:432, :] = 1  # 90:432
        #     m = cv2.warpAffine(
        #         m, tfm_inv.astype(np.float32),
        #         dsize=(int(W), int(H)), borderValue=0)
        #     print(m.shape)
        #     res_possion = cv2.seamlessClone(frame.astype(np.uint8), xt.astype(np.uint8), m.astype(np.uint8)*255,
        #                                     p=(x, y), flags=cv2.NORMAL_CLONE)
        #     # plt.imshow(cv2.cvtColor(res_possion.astype(np.uint8), cv2.COLOR_RGB2BGR))
        #     plt.imsave(save_path, cv2.cvtColor(res_possion.astype(np.uint8), cv2.COLOR_RGB2BGR))
        #     # plt.show()
        #     # plt.close()
        # except Exception as e:
        #     print(e)
        #     res = laplacian_blending(A=frame, B=xt, m=mask)
        #     # plt.imshow(cv2.cvtColor(res.astype(np.uint8), cv2.COLOR_RGB2BGR))
        #     plt.imsave(save_path, cv2.cvtColor(res.astype(np.uint8), cv2.COLOR_RGB2BGR))
        #     # plt.show()
        #     # plt.close()

        res = laplacian_blending(A=frame, B=xt, m=mask)
        # plt.imshow(cv2.cvtColor(res.astype(np.uint8), cv2.COLOR_RGB2BGR))
        plt.imsave(save_path, cv2.cvtColor(res.astype(np.uint8), cv2.COLOR_RGB2BGR))
        # plt.show()
        # plt.close()


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    ROOT = {
        'smooth': {'root': '/gavin/code/FaceSwapping/modules/third_party/infoswap/weights/'
                           'checkpoints_512/w_kernel_smooth',
                   'path': 'ckpt_ks_*.pth'},
        'no_smooth': {'root': '/gavin/code/FaceSwapping/modules/third_party/infoswap/weights/'
                              'checkpoints_512/wo_kernel_smooth',
                      'path': 'ckpt_*.pth'}
    }

    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument('-ib', '--ib_mode', type=str, default='smooth', choices=list(ROOT.keys()))
    p.add_argument('-src', '--src_path', type=str, default='infer_images/source/source.jpg')
    p.add_argument('-tar', '--tar_dir', type=str, default='infer_images/target')
    p.add_argument('-save', '--save_dir', type=str, default='infer_images')
    args = p.parse_args()

    """ Prepare Models: """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = ROOT[args.ib_mode]['root']
    path = ROOT[args.ib_mode]['path']

    pathG = path.replace('*', 'G')
    pathE = path.replace('*', 'E')
    pathI = path.replace('*', 'I')

    encoder = Backbone128(50, 0.6, 'ir_se').eval().to(device)
    state_dict = torch.load('/gavin/code/FaceSwapping/modules/third_party/infoswap/weights/model_128_ir_se50.pth',
                            map_location=device)
    encoder.load_state_dict(state_dict, strict=True)

    G = AII512().eval().to(device)
    decoder = UnetDecoder512().eval().to(device)

    # Define Information Bottlenecks:
    N = 10
    _ = encoder(torch.rand(1, 3, 128, 128).to(device), cache_feats=True)
    _readout_feats = encoder.features[:(N + 1)]  # one layer deeper than the z_attrs needed
    in_c = sum(map(lambda f: f.shape[-3], _readout_feats))
    out_c_list = [_readout_feats[i].shape[-3] for i in range(N)]

    iib = IIB(in_c, out_c_list, device, smooth=args.ib_mode == 'smooth', kernel_size=1)
    iib = iib.eval()

    G.load_state_dict(torch.load(os.path.join(root, pathG), map_location=device), strict=True)
    print("Successfully load G!")
    decoder.load_state_dict(torch.load(os.path.join(root, pathE), map_location=device), strict=True)
    print("Successfully load Decoder!")
    # 3) load IIB:
    iib.load_state_dict(torch.load(os.path.join(root, pathI), map_location=device),
                        strict=args.ib_mode == 'smooth')
    print("Successfully load IIB!")

    with torch.no_grad():
        inference(args.src_path, args.tar_dir, args.save_dir)