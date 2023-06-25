import os.path
import random
from abc import ABC, abstractmethod
import numpy as np
import pickle
import cv2
import glob
from tqdm import tqdm
from multiprocessing.pool import Pool

from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import mean_squared_error
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision.transforms import transforms
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from inference.ffplus.dataloader import FFPlusEvalDataset
from inference.celebahq.dataloader import CelebaHQEvalDataset, CelebaHQEvalDatasetHanbang
from inference.ffhq.dataloader import FFHQEvalDataset
from inference.web.dataloader import WebEvalDataset

from inference.PIPNet.lib.tools import get_lmk_model, demo_image
from inference.landmark_smooth import kalman_filter_landmark, savgol_filter_landmark
from inference.alignment import norm_crop, norm_crop_with_M, paste_back
from inference.utils import save, get_5_from_98, get_detector, get_lmk

from supervision.restoration.GPEN.infer_image import GPENImageInfer


make_abs_path = lambda fn: os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), fn))


class PilArrayDataset(torch.utils.data.Dataset):
    def __init__(self,
                 pil_arr: list,
                 transform = transforms.Compose([
                     transforms.ToTensor(),
                     transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                          std=(0.5, 0.5, 0.5)),
                 ]),
                 ):
        super(PilArrayDataset, self).__init__()
        self.pil_arr = pil_arr
        self.trans = transform

    def __getitem__(self, index):
        return self.trans(self.pil_arr[index])

    def __len__(self):
        return len(self.pil_arr)


class VideoInferDataset(torch.utils.data.Dataset):
    def __init__(self,
                 target_video_frames: list,
                 source_images: list,
                 ):
        target_video_frames = [frame for clip in target_video_frames for frame in clip]
        target_video_frames = [Image.fromarray(x) for x in target_video_frames]
        source_images = [Image.fromarray(x) for x in source_images]
        self.target_dataset = PilArrayDataset(target_video_frames)
        self.source_dataset = PilArrayDataset(source_images)

        self.len_t = len(self.target_dataset)
        self.len_s = len(self.source_dataset)
        self.len = len(self.target_dataset) * len(self.source_dataset)

    def __getitem__(self, index):
        t_idx = index % self.len_t  # ts: [00,10,20,...,90; 01,11,21,...,91; ...]
        s_idx = index // self.len_t

        t_frame = self.target_dataset[t_idx]
        s_image = self.source_dataset[s_idx]

        return {
            "t_frame": t_frame,
            "s_image": s_image,
            "t_idx": np.array([t_idx]),
            "s_idx": np.array([s_idx]),
        }

    def __len__(self):
        return self.len_t * self.len_s


vgg_mean = torch.tensor([[[0.485]], [[0.456]], [[0.406]]],
                             requires_grad=False, device=torch.device(0))
vgg_std = torch.tensor([[[0.229]], [[0.224]], [[0.225]]],
                            requires_grad=False, device=torch.device(0))
def load_bisenet():
    from bisenet.bisenet import BiSeNet
    bisenet_model = BiSeNet(n_classes=19)
    bisenet_model.load_state_dict(
        torch.load("/gavin/datasets/hanbang/79999_iter.pth", map_location="cpu")
    )
    bisenet_model.eval()
    bisenet_model = bisenet_model.cuda(0)

    from modules.third_party.megafs.image_infer import SoftErosion
    smooth_mask = SoftErosion(kernel_size=17, threshold=0.9, iterations=7).cuda()
    print('[Global] bisenet loaded.')
    return bisenet_model, smooth_mask

global_bisenet, global_smooth_mask = load_bisenet()


class VideoInferBase(pl.LightningModule, ABC):
    def __init__(self,
                 target_video_path_list: str,
                 source_image_folder: str,
                 result_folder: str,
                 image_size: int = 256,
                 batch_size: int = 1,
                 landmark_smooth = 'kalman',
                 align_target: str = 'ffhq',
                 align_source: str = 'ffhq',
                 pool_process: int = 170,
                 en_audio: bool = True,
                 en_concat: bool = False,
                 en_save_frames: bool = False,
                 en_vis_lmks: bool = True,
                 ):
        super(VideoInferBase, self).__init__()

        ''' FaceSwap model '''
        self.faceswap_model = None
        self._callback_load_faceswap_model()

        ''' Video task '''
        if target_video_path_list is None:
            return
        self.target_video_path_list = target_video_path_list
        self.target_video_name_list = [os.path.basename(x).split('.')[0] for x in target_video_path_list]
        self.source_image_folder = source_image_folder
        self.result_folder = result_folder
        os.makedirs(result_folder, exist_ok=True)
        for t_idx, target_video_path in enumerate(target_video_path_list):
            os.system('cp %s %s' % (target_video_path,
                os.path.join(result_folder, 'target%s.mp4' % self.target_video_name_list[t_idx])))

        # if not args.demo:
        #     demo_folder = None
        # self.demo_folder = demo_folder
        # if demo_folder is not None:
        #     if os.path.exists(self.demo_folder):
        #         print('deleting demo_folder: %s...' % self.demo_folder)
        #         os.system('rm -r %s' % self.demo_folder)
        #     os.mkdir(self.demo_folder)
        #     os.makedirs(os.path.join(self.demo_folder, 'target'))
        #     os.makedirs(os.path.join(self.demo_folder, 'source'))
        #     os.makedirs(os.path.join(self.demo_folder, 'result'))

        self.demo_t_imgs = []
        self.demo_s_imgs = []
        self.demo_r_imgs = []
        self.result_tmp_paths = []
        self.t_clip_idx = []
        self.t_frame_in_clip_idx = []

        ''' Alignment Model '''
        self.net, self.detector = get_lmk_model()
        self.landmark_smooth = landmark_smooth
        self.align_target = align_target
        self.align_source = align_source
        self.en_vis_lmks = en_vis_lmks

        ''' Preprocess '''
        self.image_size = image_size
        self.t_facial_masks = []
        t_ori_frames, t_vis_frames, t_crop_frames, M_list, fps = self._preprocess_target_video()
        self.t_ori_frames = t_ori_frames
        self.t_vis_frames = t_vis_frames
        self.t_crop_frames = t_crop_frames
        self.m_list = M_list
        self.fps = fps
        assert len(t_ori_frames) == len(M_list) and len(t_crop_frames) == len(fps)
        self.s_ori_images, self.s_crop_images = self._preprocess_source_images()

        ''' Evaluation Dataset '''
        self.batch_size = batch_size
        self.test_dataset = self._get_dataset()
        self.dataset_len = self.test_dataset.__len__()
        self.target_len = self.test_dataset.len_t
        self.source_len = self.test_dataset.len_s

        ''' Postprocess '''
        self.pool_process = pool_process
        self.en_audio = en_audio
        self.en_concat = en_concat
        self.gpen = None

    @abstractmethod
    def _callback_load_faceswap_model(self):
        pass

    @abstractmethod
    def callback_infer_batch(self, i_s, i_t):
        pass

    def _preprocess_target_video(self):
        for folder in os.listdir(self.result_folder):
            if "tmp" not in folder:
                continue
            tmp_folder = os.path.join(self.result_folder, folder)
            print("deleting tmp_folder: %s" % tmp_folder)
            os.system("rm -r %s" % tmp_folder)

        all_target_ori_frames = []
        all_target_vis_frames = []
        all_target_crop_frames = []
        all_M_list = []
        all_fps_list = []
        for t_idx, target_video_path in enumerate(self.target_video_path_list):
            ''' Get fps '''
            video_capture = cv2.VideoCapture(target_video_path)
            fps = video_capture.get(cv2.CAP_PROP_FPS)
            all_fps_list.append(fps)

            ''' Convert video to frames '''
            in_tmp_folder = os.path.join(self.result_folder, "in_tmp_%d" % t_idx)
            os.mkdir(in_tmp_folder)
            os.system(
                f"ffmpeg -loglevel quiet -i {target_video_path} "
                f"-qscale:v 1 -qmin 1 -qmax 1 -vsync 0 "
                f"{in_tmp_folder}/frame_%05d.jpg"
            )

            target_frame_paths = sorted(glob.glob(os.path.join(in_tmp_folder, "*.jpg")))

            ''' Get target landmarks '''
            print('[Extracting target landmarks...]')
            target_lmks = []
            for frame_path in tqdm(target_frame_paths):
                target = np.array(Image.open(frame_path).convert("RGB"))
                lmk = demo_image(target, self.net, self.detector)
                lmk = lmk[0]
                target_lmks.append(lmk)

            ''' Landmark smoothing '''
            target_lmks = np.array(target_lmks, np.float32)  # (#frames, 98, 2)
            if self.landmark_smooth == 'kalman':
                target_lmks = kalman_filter_landmark(target_lmks,
                                                     process_noise=0.01,
                                                     measure_noise=0.01).astype(np.int)
            elif self.landmark_smooth == 'savgol':
                target_lmks = savgol_filter_landmark(target_lmks).astype(np.int)
            elif self.landmark_smooth == 'cancel':
                target_lmks = target_lmks.astype(np.int)
            else:
                raise KeyError('Not supported landmark_smooth choice')

            ''' Crop target frames according to the landmarks '''
            frame_idx = 0
            trans = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=0.5, std=0.5)
            ])
            one_ori_frames = []
            one_vis_frames = []
            one_crop_frames = []
            one_M_list = []
            for frame_path in tqdm(target_frame_paths):
                target = np.array(Image.open(frame_path).convert("RGB"))
                one_ori_frames.append(target)
                self.t_clip_idx.append(t_idx)
                self.t_frame_in_clip_idx.append(frame_idx)

                num_keypoints = 98 if self.en_vis_lmks else 0
                target_vis = target.copy()
                for point in range(num_keypoints):
                    cv2.circle(target_vis, target_lmks[frame_idx][point], 1, (255, 0, 0), 1, )
                one_vis_frames.append(target_vis)
                lmk = get_5_from_98(target_lmks[frame_idx])

                target_crop, M = norm_crop_with_M(target, lmk, self.image_size,
                                                  mode=self.align_target, borderValue=0.0)
                one_crop_frames.append(target_crop)
                one_M_list.append(M)

                target_tensor = trans(target_crop).cuda().unsqueeze(0)  # in [-1,1]
                target_facial_mask = self._get_any_mask(target_tensor,
                                                        par=[1,2,3,4,5,6,10,11,12,13]).squeeze()  # in [0,1]
                target_facial_mask = target_facial_mask.cpu().numpy().astype(np.float)
                target_facial_mask = self._finetune_mask(target_facial_mask, target_lmks)  # in [0,1]
                self.t_facial_masks.append(target_facial_mask)

                frame_idx += 1

            if self.en_vis_lmks:
                self._save_frames_to_video(one_vis_frames, fps,
                                           os.path.join(self.result_folder,
                                                        'target_vis_%s.mp4' % self.target_video_name_list[t_idx]))

            all_target_ori_frames.append(one_ori_frames)
            all_target_vis_frames.append(one_vis_frames)
            all_target_crop_frames.append(one_crop_frames)
            all_M_list.append(one_M_list)

        return all_target_ori_frames, all_target_vis_frames, all_target_crop_frames, all_M_list, all_fps_list

    def _preprocess_source_images(self):
        source_image_paths = os.listdir(self.source_image_folder)
        source_image_paths = [os.path.join(self.source_image_folder, img_path) for img_path in source_image_paths]

        source_ori_images = []
        source_crop_images = []
        print('[Cropping source images]')
        for source_path in tqdm(source_image_paths):
            source_img = np.array(Image.open(source_path).convert("RGB"))
            source_ori_images.append(source_img)

            lmk = demo_image(source_img, self.net, self.detector)
            lmk = lmk[0]
            lmk = get_5_from_98(lmk)

            source_crop = norm_crop(source_img, lmk, self.image_size,
                                    mode=self.align_source, borderValue=0.0)
            source_crop_images.append(source_crop)

        return source_ori_images, source_crop_images

    @staticmethod
    def _save_frames_to_video(frames: list, fps: int, path: str):
        h, w = frames[0].shape[0], frames[0].shape[1]
        vid = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        for frame in frames:
            vid.write(frame[:, :, ::-1])  # RGB to BGR

    def forward(self, source_img, target_img):
        return self.callback_infer_batch(source_img, target_img)

    def test_step(self, batch, batch_idx):
        i_t = batch["t_frame"]
        i_s = batch["s_image"]
        t_idx = batch["t_idx"]
        s_idx = batch["s_idx"]
        i_r = self.forward(i_s, i_t)  # forward, (B,C,H,W), in [-1,1]

        self._snapshot(i_t, i_s, i_r, t_idx, s_idx, batch_idx)

    def _snapshot(self, i_t, i_s, i_r, t_idx, s_idx, batch_idx):
        img_t = self._tensor_to_arr(i_t)[0]
        img_s = self._tensor_to_arr(i_s)[0]
        img_r = self._tensor_to_arr(i_r)[0]

        if self.gpen is not None:
            img_r = self.gpen.image_infer(img_r)

        self.demo_t_imgs.append(img_t)
        self.demo_s_imgs.append(img_s)
        self.demo_r_imgs.append(img_r)

        ''' result saving path '''
        result_tmp_folder = "out_tmp_%d_%d" % (self.t_clip_idx[int(t_idx)], int(s_idx))
        result_tmp_folder = os.path.join(self.result_folder, result_tmp_folder)
        os.makedirs(result_tmp_folder, exist_ok=True)
        result_tmp_name = "frame_%05d.jpg" % (self.t_frame_in_clip_idx[int(t_idx)])
        self.result_tmp_paths.append(os.path.join(result_tmp_folder, result_tmp_name))

    @staticmethod
    def _get_any_mask(img, par=None, normalized=False):
        # [0, 'background', 1 'skin', 2 'l_brow', 3 'r_brow', 4 'l_eye', 5 'r_eye',
        # 6 'eye_g', 7 'l_ear', 8 'r_ear', 9 'ear_r',  10 'nose', 11 'mouth', 12 'u_lip',
        # 13 'l_lip', 14 'neck', 15 'neck_l', 16 'cloth', 17 'hair', 18 'hat']
        ori_h, ori_w = img.shape[2], img.shape[3]
        with torch.no_grad():
            img = F.interpolate(img, size=512, mode="nearest", )
            if not normalized:
                img = img * 0.5 + 0.5
                img = img.sub(vgg_mean.detach()).div(vgg_std.detach())
            out = global_bisenet(img)[0]
            parsing = out.softmax(1).argmax(1)
        mask = torch.zeros_like(parsing)
        for p in par:
            mask = mask + ((parsing == p).float())
        mask = mask.unsqueeze(1)
        mask = F.interpolate(mask, size=(ori_h, ori_w), mode="bilinear", align_corners=True)
        return mask

    @staticmethod
    def _finetune_mask(facial_mask: np.ndarray, lmk_98: np.ndarray = None):
        assert facial_mask.shape[1] == 256
        facial_mask = (facial_mask * 255).astype(np.uint8)
        # h_min = lmk_98[33:41, 0].min() + 20
        h_min = 80

        facial_mask = cv2.dilate(facial_mask, (40, 40), iterations=1)
        facial_mask[:h_min] = 0  # black
        facial_mask[255 - 20:] = 0

        kernel_size = (20, 20)
        blur_size = tuple(2 * j + 1 for j in kernel_size)
        facial_mask = cv2.GaussianBlur(facial_mask, blur_size, 0)

        return facial_mask.astype(np.float) / 255

    def test_epoch_end(self, outputs):
        t_crop_list = self.t_crop_frames
        t_ori_list = self.t_ori_frames
        result_tmp_paths = self.result_tmp_paths
        t_m_list = self.m_list
        fps_list = self.fps

        audio_cmd = " -map 0:v:0 -map 1:a:0? -c:a copy " if self.en_audio else ""

        for s_idx, source_path in enumerate(os.listdir(self.source_image_folder)):
            t_frame_left, t_frame_right = 0, 0
            for t_idx in tqdm(range(len(t_ori_list)),
                              desc='[%d/%d] saving as videos' % (s_idx, len(os.listdir(self.source_image_folder)))):
                target_video_path = self.target_video_path_list[t_idx]
                one_t_ori_list = t_ori_list[t_idx]
                t_frame_right += len(one_t_ori_list)

                one_m_list = t_m_list[t_idx]
                one_fps = fps_list[t_idx]
                h, w = one_t_ori_list[0].shape[0], one_t_ori_list[0].shape[1]

                # print('left=', t_frame_left, 'right=', t_frame_right)
                # print('range is:', s_idx * self.target_len + t_frame_left,
                #       s_idx * self.target_len + t_frame_right,
                #       'total len is:', len(self.demo_r_imgs), len(result_tmp_paths), len(self.t_facial_masks))
                one_demo_r_imgs = self.demo_r_imgs[s_idx * self.target_len + t_frame_left
                                                   : s_idx * self.target_len + t_frame_right]
                one_result_tmp_paths = result_tmp_paths[s_idx * self.target_len + t_frame_left
                                                   : s_idx * self.target_len + t_frame_right]
                one_t_facial_masks = self.t_facial_masks[t_frame_left: t_frame_right]

                t_frame_left = t_frame_right

                with Pool(self.pool_process) as pool:
                    pool.map(save, zip(one_demo_r_imgs,
                                       one_m_list,
                                       one_t_ori_list,
                                       one_result_tmp_paths,
                                       one_t_facial_masks
                                       ))

                source_name = os.path.basename(source_path).split('.')[0]
                out_frames_folder = os.path.join(self.result_folder, "out_tmp_%d_%d" % (t_idx, s_idx))
                out_video_path = os.path.join(
                    self.result_folder, "result_%s_%s.mp4" % (self.target_video_name_list[t_idx], source_name))
                ffmpeg_cmd = f"ffmpeg -y -loglevel quiet -r {one_fps} " \
                             f"-i {out_frames_folder}/frame_%05d.jpg " \
                             f"-i {target_video_path} " \
                             f"{audio_cmd} -c:v libx264 -crf 10 -pix_fmt yuv420p {out_video_path}"
                # print('output result:')
                # print(ffmpeg_cmd)
                os.system(ffmpeg_cmd)

                if self.en_concat:
                    source_abs_path = os.path.join(self.source_image_folder, source_path)
                    w_s, h_s = Image.open(source_abs_path).size
                    scale_s = h / h_s
                    w_s, h_s = int(w_s * scale_s), int(h_s * scale_s)

                    concat_video_save_path = os.path.join(
                        self.result_folder, "cat_target%s_%s.mp4" % (self.target_video_name_list[t_idx], source_name))
                    filter_complex = f" \"[1:v]scale={w_s}:{h_s},setsar=sar=1[video];[0:v][video][2:v]hstack=3\" "
                    cat_cmd = f"ffmpeg -y -loglevel quiet " \
                              f"-i {target_video_path} " \
                              f"-i {source_abs_path} " \
                              f"-i {out_video_path} " \
                              f"-filter_complex {filter_complex} " \
                              f"{concat_video_save_path}"
                    # print('concat target, source, and result:')
                    # print(cat_cmd)
                    os.system(cat_cmd)

        for folder in os.listdir(self.result_folder):
            if "tmp" not in folder:
                continue
            tmp_folder = os.path.join(self.result_folder, folder)
            os.system("rm -r %s" % tmp_folder)


    @staticmethod
    def _tensor_to_arr(tensor):
        return ((tensor + 1.) * 127.5).permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)

    @staticmethod
    def _arr_to_tensor(arr, norm: bool = True):
        tensor = torch.tensor(arr, dtype=torch.float).cuda() / 255  # in [0,1]
        tensor = (tensor - 0.5) / 0.5 if norm else tensor  # in [-1,1]
        tensor = tensor.permute(0, 3, 1, 2)
        return tensor

    def _get_dataset(self):
        test_dataset = VideoInferDataset(
            target_video_frames=self.t_crop_frames,
            source_images=self.s_crop_images,
        )
        return test_dataset

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=8,
            shuffle=False,
            drop_last=False,
        )


class VideoInferFaceShifter(VideoInferBase):
    def __init__(self,
                 load_path: str,
                 pt_path: str,
                 mouth_helper: torch.nn.Module = None,
                 gpen: torch.nn.Module = None,
                 **kwargs
                 ):
        self.load_path = load_path
        self.pt_path = pt_path

        ''' MouthNet params '''
        if 'mouth1' in pt_path:
            mouth_net_param = {
                "use": True,
                "feature_dim": 128,
                "crop_param": (28, 56, 84, 112),
                "weight_path": "../../modules/third_party/arcface/weights/mouth_net_28_56_84_112.pth",
            }
        elif 'mouth2' in pt_path:
            mouth_net_param = {
                "use": True,
                "feature_dim": 128,
                "crop_param": (19, 74, 93, 112),
                "weight_path": "../../modules/third_party/arcface/weights/mouth_net_19_74_93_112.pth",
            }
        else:
            mouth_net_param = {
                "use": False
            }
        self.mouth_net_param = mouth_net_param

        ''' Post process '''
        self.bisenet_model = None
        self.smooth_mask = None

        super(VideoInferFaceShifter, self).__init__(**kwargs)

        self.mouth_helper = mouth_helper
        self.gpen = gpen

    def _callback_load_faceswap_model(self):
        load_path = self.load_path
        pt_path = self.pt_path

        self._extract_generator(load_path=load_path, path=pt_path, mouth_net_param=self.mouth_net_param)
        G = self._load_extracted(path=pt_path, mouth_net_param=self.mouth_net_param)
        G = G.cuda()
        self.faceswap_model = G
        if 'post' in pt_path:
            self.bisenet_model = global_bisenet
            self.smooth_mask = global_smooth_mask
            self.vgg_mean = vgg_mean
            self.vgg_std = vgg_std
        print('FaceShifter model loaded from %s.' % load_path)

    def callback_infer_batch(self, i_s, i_t):
        i_r = self.faceswap_model(i_s, i_t)[0]  # x, id_vector, att

        if self.bisenet_model is not None:
            target_hair_mask = self._get_any_mask(i_t, par=[0, 17])
            target_hair_mask, _ = self.smooth_mask(target_hair_mask)
            i_r = target_hair_mask * i_t + (target_hair_mask * (-1) + 1) * i_r

        i_r = self._finetune_mouth(i_s, i_t, i_r)

        return i_r

    def _finetune_mouth(self, i_s, i_t, i_r):
        if self.mouth_helper is None:
            return i_r
        helper_face = self.mouth_helper(i_s, i_t)[0]
        helper_mouth_mask = self._get_any_mask(helper_face, par=[11, 12, 13])  # (B,1,H,W)
        i_r_mouth_mask = self._get_any_mask(helper_face, par=[11, 12, 13])  # (B,1,H,W)
        i_r_mouth_mask[helper_mouth_mask == 1] = 1

        ''' dilate and blur by cv2 '''
        i_r_mouth_mask = self._tensor_to_arr(i_r_mouth_mask)[0]  # (H,W,C)
        i_r_mouth_mask = cv2.dilate(i_r_mouth_mask, (1, 1), iterations=1)

        kernel_size = (15, 15)
        blur_size = tuple(2 * j + 1 for j in kernel_size)
        i_r_mouth_mask = cv2.GaussianBlur(i_r_mouth_mask, blur_size, 0)  # (H,W,C)
        i_r_mouth_mask = i_r_mouth_mask.squeeze()[None, :, :, None]  # (1,H,W,1)
        i_r_mouth_mask = self._arr_to_tensor(i_r_mouth_mask, norm=False)  # in [0,1]

        return helper_face * i_r_mouth_mask + i_r * (1 - i_r_mouth_mask)

    @staticmethod
    def _load_extracted(path="./extracted_ckpt/G_tmp.pth", mouth_net_param: dict = None):
        from modules.networks.faceshifter import FSGenerator
        G = FSGenerator(
            make_abs_path("../../modules/third_party/arcface/weights/ms1mv3_arcface_r100_fp16/backbone.pth"),
            mouth_net_param=mouth_net_param,
        )
        G.load_state_dict(torch.load(path, "cpu"), strict=False)
        G.eval()
        return G

    @staticmethod
    def _extract_generator(
            load_path="/gavin/code/FaceSwapping/trainer/faceshifter/out/hello/epoch=7-step=110999.ckpt",
            path="./extracted_ckpt/G_tmp.pth",
            mouth_net_param: dict = None,
            n_layers=3,
            num_D=3,
        ):
        if 'hanbang' in load_path:
            print('Use cached model (hanbang).')
            return
        from trainer.faceshifter.faceshifter_pl import FaceshifterPL
        import yaml
        with open(make_abs_path('../../trainer/faceshifter/config.yaml'), 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        config['mouth_net'] = mouth_net_param

        net = FaceshifterPL(n_layers=n_layers, num_D=num_D, config=config)
        checkpoint = torch.load(
            load_path,
            map_location="cpu",
        )
        net.load_state_dict(checkpoint["state_dict"], strict=False)
        net.eval()

        G = net.generator
        torch.save(G.state_dict(), path)


class VideoInferFaceShifterHanBang(VideoInferFaceShifter):
    def __init__(self, **kwargs):
        kwargs["load_path"] = 'hanbang'
        kwargs["pt_path"] = make_abs_path('../../trainer/faceshifter/extracted_ckpt/G_step14999_v2.pth')
        kwargs["align_target"] = 'set1'
        kwargs["align_source"] = 'arcface'
        super(VideoInferFaceShifterHanBang, self).__init__(**kwargs)


class VideoInferSimSwapOfficial(VideoInferBase):
    def _callback_load_faceswap_model(self):
        from simswap.image_infer import SimSwapOfficialImageInfer
        self.faceswap_model = SimSwapOfficialImageInfer()

    def callback_infer_batch(self, i_s, i_t):
        i_r = self.faceswap_model.image_infer(source_tensor=i_s,
                                              target_tensor=i_t)
        i_r = i_r.clamp(-1, 1)
        return i_r


class VideoInferInfoSwap(VideoInferBase):
    def _callback_load_faceswap_model(self):
        from infoswap.inference_model import InfoSwapInference
        self.faceswap_model = InfoSwapInference()
        self.faceswap_model.load_model()

    def callback_infer_batch(self, i_s, i_t):
        i_r = self.faceswap_model.infer_batch(i_s, i_t)
        i_r = i_r.cuda()
        i_r = F.interpolate(i_r, size=(256, 256), mode='bilinear', align_corners=True)
        return i_r


class VideoInferHiRes(VideoInferBase):
    def _callback_load_faceswap_model(self):
        from hires.image_infer import HiResImageInfer
        self.faceswap_model = HiResImageInfer()

    def callback_infer_batch(self, i_s, i_t):
        from PIL import Image
        i_s = (i_s + 1) * 127.5
        i_t = (i_t + 1) * 127.5
        source_np = i_s.permute(0, 2, 3, 1)[0].cpu().numpy().astype(np.uint8)
        target_np = i_t.permute(0, 2, 3, 1)[0].cpu().numpy().astype(np.uint8)
        source_pil = Image.fromarray(source_np)
        target_pil = Image.fromarray(target_np)
        i_r = self.faceswap_model.image_infer(source_pil=source_pil,
                                              target_pil=target_pil)
        i_r = F.interpolate(i_r, size=256, mode="bilinear", align_corners=True)
        i_r = i_r.clamp(-1, 1)
        return i_r


if __name__ == "__main__":

    # video_path = "/apdcephfs/share_1290939/gavinyuan/datasets/ff+/original_sequences/youtube/c23/videos/004.mp4"

    video_path = [
        # "/apdcephfs/share_1290939/gavinyuan/datasets/celebvhq/35666/0s1UUn9aSSw_1.mp4",
        # "/apdcephfs/share_1290939/gavinyuan/datasets/celebvhq/35666/0O1-u4vyKQc_2.mp4",
        "/apdcephfs/share_1290939/gavinyuan/datasets/celebvhq/35666/_NKXqc5vAN8_0.mp4",
        # "/apdcephfs/share_1290939/gavinyuan/datasets/celebvhq/35666/_h3NFrBsJAM_0.mp4",
        # "/apdcephfs/share_1290939/gavinyuan/datasets/celebvhq/35666/_TYRa6vxxb0_10.mp4",
        # "/apdcephfs/share_1290939/gavinyuan/datasets/celebvhq/35666/_W4Em_fHubY_8.mp4",
        # "/apdcephfs/share_1290939/gavinyuan/datasets/celebvhq/35666/0KbxR3VjAg8_6.mp4",

        # "/apdcephfs/share_1290939/gavinyuan/datasets/celebvhq/35666/0tx4o3yXM64_9.mp4",
        # "/apdcephfs/share_1290939/gavinyuan/datasets/celebvhq/35666/0vYayoxls7M_0.mp4",
        # "/apdcephfs/share_1290939/gavinyuan/datasets/celebvhq/35666/0xjr1vVzFKY_3.mp4",
        # "/apdcephfs/share_1290939/gavinyuan/datasets/celebvhq/35666/1CpmDoCudEY_3.mp4",
        # "/apdcephfs/share_1290939/gavinyuan/datasets/celebvhq/35666/1kxUOnNuteY_1.mp4",

        # "/apdcephfs/share_1290939/gavinyuan/datasets/celebvhq/35666/1R_wzGrOLFs_6.mp4",
        # "/apdcephfs/share_1290939/gavinyuan/datasets/celebvhq/35666/1RRRM17-4Ok_2.mp4",
        # "/apdcephfs/share_1290939/gavinyuan/datasets/celebvhq/35666/1sMrtt7sAR8_0.mp4",
        # "/apdcephfs/share_1290939/gavinyuan/datasets/celebvhq/35666/1tmjWKC3jlY_0.mp4",
        # "/apdcephfs/share_1290939/gavinyuan/datasets/celebvhq/35666/1To7zCTHAv4_1.mp4",
        # "/apdcephfs/share_1290939/gavinyuan/datasets/celebvhq/35666/1UzpiYrUfxE_2.mp4",
        # "/apdcephfs/share_1290939/gavinyuan/datasets/celebvhq/35666/1YklLet-e8s_4.mp4",
    ]

    image_folder = "/gavin/code/FaceSwapping/inference/infer_images/appendix/source_batch2"
    root_folder = "/gavin/code/FaceSwapping/inference/infer_images/appendix/"

    task_list = [
        # ("hanbang", "hanbang"),  # fixed
        # ("hires", "hires"),  # fixed
        # ("infoswap", "infoswap"),  # fixed
        # ("../../trainer/faceshifter/out/faceshifter_vanilla_5/epoch=11-step=548999.ckpt",
        #  "./extracted_ckpt/G_tmp_v5.pth"),
        ("../../trainer/faceshifter/out/triplet10w_33/epoch=10-step=596999.ckpt",
         "./extracted_ckpt/G_mouth1_t33_post.pth"),
    ]

    video_mouth_helper_pl = VideoInferFaceShifter(
        load_path="../../weights/reliableswap_weights/ckpt/triplet10w_34/epoch=13-step=737999.ckpt",
        pt_path="../../weights/reliableswap_weights/extracted_pth/G_t34_helper_post.pth",
        target_video_path_list=None,
        source_image_folder=None,
        result_folder=None,
    )
    print("[Global] Mouth helper loaded.")

    global_gpen = GPENImageInfer()
    print("[Global] GPEN loaded.")

    # video_test_pl = VideoInferSimSwapOfficial(
    #     target_video_path=video_path,
    #     source_image_folder=image_folder,
    #     result_folder=os.path.join(root_folder, 'result_simswap_official'),
    #     en_concat=True,
    # )

    for idx in range(len(task_list)):
        ckpt_path, pt_path = task_list[idx]
        if '../' in ckpt_path:
            task_list[idx] = (make_abs_path(ckpt_path), pt_path)

        ckpt_path, pt_path = task_list[idx]
        if not os.path.exists(ckpt_path) and ckpt_path[0] == '/':
            if 'faceshifter' in ckpt_path:
                task_list[idx] = (ckpt_path.replace(
                    '/apdcephfs_cq2/share_1290939/gavinyuan/code/FaceSwapping/trainer/faceshifter/out/',
                    '/apdcephfs/share_1290939/gavinyuan/out/'), pt_path)
            elif 'simswap' in ckpt_path:
                task_list[idx] = (ckpt_path.replace(
                    '/apdcephfs_cq2/share_1290939/gavinyuan/code/FaceSwapping/trainer/simswap/out/',
                    '/apdcephfs/share_1290939/gavinyuan/out/'), pt_path)

    pl_list = []
    for idx, task in enumerate(task_list):
        ckpt_path, pt_path = task
        demo_folder = 'result_%s' % (os.path.split(os.path.dirname(ckpt_path))[-1])

        if os.path.exists(ckpt_path) and ('faceshifter' in ckpt_path or 'triplet10w' in ckpt_path):
            mouth_helper = None
            used_gpen = None
            if 'triplet' in ckpt_path:
                mouth_helper = video_mouth_helper_pl.faceswap_model
                used_gpen = global_gpen

            video_test_pl = VideoInferFaceShifter(
                load_path=ckpt_path,
                pt_path=pt_path,
                mouth_helper=mouth_helper,
                gpen=used_gpen,
                target_video_path_list=video_path,
                source_image_folder=image_folder,
                result_folder=os.path.join(root_folder, demo_folder),
                en_concat=False,
            )
        elif 'hanbang' in ckpt_path:
            video_test_pl = VideoInferFaceShifterHanBang(
                mouth_helper=None,
                gpen=global_gpen,
                target_video_path_list=video_path,
                source_image_folder=image_folder,
                result_folder=os.path.join(root_folder, 'result_hanbang'),
                en_concat=False,
            )
        elif 'hires' in ckpt_path:
            video_test_pl = VideoInferHiRes(
                target_video_path_list=video_path,
                source_image_folder=image_folder,
                result_folder=os.path.join(root_folder, 'result_hires'),
                en_concat=False,
            )
        elif 'infoswap' in ckpt_path:
            video_test_pl = VideoInferInfoSwap(
                target_video_path_list=video_path,
                source_image_folder=image_folder,
                result_folder=os.path.join(root_folder, 'result_infoswap'),
                en_concat=False,
            )
        else:
            raise ValueError('[%d] ckpt_path not supported: %s' % (idx, ckpt_path))

        pl_list.append(video_test_pl)

    trainer = pl.Trainer(
        logger=False,
        gpus=1,
        distributed_backend='dp',
        benchmark=True,
    )
    for test_pl in pl_list:
        trainer.test(test_pl)
