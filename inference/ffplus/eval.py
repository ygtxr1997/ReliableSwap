import os.path
import random
from abc import ABC, abstractmethod
import numpy as np
import pickle
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import mean_squared_error
from PIL import Image
import cv2

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from inference.ffplus.dataloader import FFPlusEvalDataset
from inference.celebahq.dataloader import CelebaHQEvalDataset, CelebaHQEvalDatasetHanbang
from inference.ffhq.dataloader import FFHQEvalDataset
from inference.web.dataloader import WebEvalDataset

from modules.third_party.arcface import iresnet100, MouthNet
from modules.third_party import cosface
# from modules.third_party.tddfa_v2 import TDDFAExtractor

from supervision.restoration.GPEN.infer_image import GPENImageInfer

make_abs_path = lambda fn: os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), fn))

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# set_random_seed(41)  # mostly used
set_random_seed(38)

repeat_cnt = 1
celebahq_t_list = np.arange(29893).repeat(repeat_cnt)
np.random.shuffle(celebahq_t_list)
celebahq_s_list = np.array([np.random.randint(29893) for _ in range(29893 * repeat_cnt)])
celebahq_ts_list = np.ones((29893 * repeat_cnt, 2), dtype=np.uint)
celebahq_ts_list[:, 0] = celebahq_t_list
celebahq_ts_list[:, 1] = celebahq_s_list
# celebahq_ts_list = np.arange(25000)
# np.random.shuffle(celebahq_ts_list)

ffhq_ts_list = np.arange(70000)
np.random.shuffle(ffhq_ts_list)

# ffplus_ts_list = '/gavin/datasets/ff+/original_sequences/youtube/c23/ts_pairs.pickle'
# def _load_ts_pairs(self, ts_pairs_pickle: str):
#     with open(ts_pairs_pickle, "rb") as handle:
#         ts_pairs_dict: list = pickle.load(handle)
#     ''' Format, list[[target, source], ...]:
#         {0: [0, 3],
#          999: [999, 960]}
#     '''
#
#     self.ts_list = [0] * len(ts_pairs_dict)
#     for target_id, source_id in ts_pairs_dict:
#         self.ts_list[target_id] = source_id


class Evaluator(pl.LightningModule,
                ABC):
    def __init__(self,
                 demo_folder: str,
                 benchmark: str,
                 batch_size: int = 1,
                 en_id_eval: bool = False,
                 en_fixer_eval: bool = False,
                 en_pse_eval: bool = False,
                 en_hopenet_eval: bool = False,
                 en_deep3d_eval: bool = False,
                 ):
        super(Evaluator, self).__init__()

        ''' FaceSwap model '''
        self.faceswap_model = None
        self._callback_load_faceswap_model()

        if benchmark is None:
            return
        self.benchmark = benchmark

        if not args.demo:
            demo_folder = None
        self.demo_folder = demo_folder
        if demo_folder is not None:
            if os.path.exists(self.demo_folder):
                print('deleting demo_folder: %s...' % self.demo_folder)
                os.system('rm -r %s' % self.demo_folder)
            os.mkdir(self.demo_folder)
            os.makedirs(os.path.join(self.demo_folder, 'target'))
            os.makedirs(os.path.join(self.demo_folder, 'source'))
            os.makedirs(os.path.join(self.demo_folder, 'result'))

        self.demo_t_imgs = []
        self.demo_s_imgs = []
        self.demo_r_imgs = []

        self.batch_size = batch_size
        self.en_id_eval = en_id_eval
        self.en_fixer_eval = en_fixer_eval
        self.en_pse_eval = en_pse_eval
        self.en_hopenet_eval = en_hopenet_eval
        self.en_deep3d_eval = en_deep3d_eval

        self.test_dataset = self._get_dataset()
        self.dataset_len = self.test_dataset.__len__()
        # self.ts_list = self.test_dataset.ts_list

        ''' ID retrieval '''
        if en_id_eval:
            self.id_model = iresnet100().cuda()
            # id_path = '/gavin/code/FaceSwapping/modules/third_party/arcface/weights/' \
            #           'glint360k_cosface_r100_fp16_0.1/backbone.pth'
            self.id_model = cosface.net.sphere().cuda()
            id_path = '/gavin/code/FaceSwapping/modules/third_party/cosface/net_sphere20_data_vggface2_acc_9955.pth'
            weights = torch.load(id_path)
            self.id_model.load_state_dict(weights)
            self.id_model.eval()
            self.id_t = np.zeros((self.dataset_len, 512), dtype=np.float32)
            self.id_s = np.zeros_like(self.id_t, dtype=np.float32)  # each embedding repeats 10 times in ffplus
            self.id_r = np.zeros_like(self.id_t, dtype=np.float32)
            print('ID model loaded.')

        if en_fixer_eval:
            self.fixer_crop_param = (28, 56, 84, 112)
            self.fixer_casia_model = MouthNet(
                bisenet=None,
                feature_dim=128,
                crop_param=self.fixer_crop_param
            ).cuda()
            fixer_path = "/gavin/code/FaceSwapping/modules/third_party/arcface/weights/fixer_net_casia_28_56_84_112.pth"
            self.fixer_casia_model.load_backbone(fixer_path)
            self.fixer_casia_model.eval()
            self.fixer_t = np.zeros((self.dataset_len, 128), dtype=np.float32)
            self.fixer_s = np.zeros_like(self.fixer_t, dtype=np.float32)  # each embedding repeats 10 times in ffplus
            self.fixer_r = np.zeros_like(self.fixer_t, dtype=np.float32)
            print('Fixer model loaded.')

        ''' pose & shape & expression '''
        if en_pse_eval:
            # load model, default output is dimension with length 62 = 12(pose) + 40(shape) +10(expression)
            self.pse_model = TDDFAExtractor()
            self.pse_t = np.zeros((self.dataset_len, 62), dtype=np.float32)
            self.pse_s = np.zeros_like(self.pse_t, dtype=np.float32)
            self.pse_r = np.zeros_like(self.pse_t, dtype=np.float32)
            print('Pose/Shape/Expression model loaded.')

        ''' Hopenet Pose '''
        if en_hopenet_eval:
            from hopenet.hopenet import Hopenet
            import torchvision
            model = Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)
            snapshot_path = '../../modules/third_party/hopenet/hopenet_robust_alpha1.pkl'
            saved_state_dict = torch.load(snapshot_path)
            model.load_state_dict(saved_state_dict)
            model.eval()
            norm = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

            self.hopenet_model = model.cuda()
            self.hopenet_norm = norm
            self.hope_t = np.zeros((self.dataset_len, 66 * 3), dtype=np.float32)  # 198=66*3
            self.hope_s = np.zeros_like(self.hope_t, dtype=np.float32)
            self.hope_r = np.zeros_like(self.hope_t, dtype=np.float32)

        ''' Deep3d Expression '''
        if en_deep3d_eval:
            from deep3d.image_infer import Deep3DImageInfer
            self.deep3d_model = Deep3DImageInfer()
            self.deep3d_t = np.zeros((self.dataset_len, 64), dtype=np.float32)
            self.deep3d_s = np.zeros_like(self.deep3d_t, dtype=np.float32)
            self.deep3d_r = np.zeros_like(self.deep3d_t, dtype=np.float32)

        ''' Post Process '''
        self.gpen = None

    @abstractmethod
    def _callback_load_faceswap_model(self):
        pass

    @abstractmethod
    def callback_infer_batch(self, i_s, i_t):
        pass

    def forward(self, source_img, target_img):
        return self.callback_infer_batch(source_img, target_img)

    def test_step(self, batch, batch_idx):
        # if batch_idx % 10 != 0 or batch_idx >= 1000:
        #     return
        i_t = batch["target_image"]
        i_s = batch["source_image"]

        i_r = self.forward(i_s, i_t)  # forward, (B,C,H,W)

        self._record_id_batch(i_t, i_s, i_r, batch_idx)
        self._record_fixer_batch(i_t, i_s, i_r, batch_idx)
        self._record_pse_batch(i_t, i_s, i_r, batch_idx)
        self._record_hopenet_batch(i_t, i_s, i_r, batch_idx)
        self._record_deep3d_batch(i_t, i_s, i_r, batch_idx)
        self._snapshot(i_t, i_s, i_r, batch_idx)

    def _record_id_batch(self, i_t, i_s, i_r, batch_idx):
        if not self.en_id_eval:
            return
        total = torch.cat((i_t, i_s, i_r), dim=0)
        total = F.interpolate(total, size=112, mode="bilinear", align_corners=True)
        embeddings: torch.Tensor = self.id_model(total).cpu()
        id_t, id_s, id_r = torch.chunk(embeddings, chunks=3, dim=0)

        left, right = batch_idx * self.batch_size, (batch_idx + 1) * self.batch_size
        self.id_t[left:right] = id_t.numpy()
        self.id_s[left:right] = id_s.numpy()
        self.id_r[left:right] = id_r.numpy()

    def _record_fixer_batch(self, i_t, i_s, i_r, batch_idx):
        if not self.en_fixer_eval:
            return
        total = torch.cat((i_t, i_s, i_r), dim=0)
        total = F.interpolate(total, size=112, mode="bilinear", align_corners=True)
        w1, h1, w2, h2 = self.fixer_crop_param
        total = total[:, :, h1:h2, w1:w2]  # crop
        embeddings: torch.Tensor = self.fixer_casia_model(total).cpu()
        fix_t, fix_s, fix_r = torch.chunk(embeddings, chunks=3, dim=0)

        left, right = batch_idx * self.batch_size, (batch_idx + 1) * self.batch_size
        self.fixer_t[left:right] = fix_t.numpy()
        self.fixer_s[left:right] = fix_s.numpy()
        self.fixer_r[left:right] = fix_r.numpy()

    def _record_pse_batch(self, i_t, i_s, i_r, batch_idx):
        if not self.en_pse_eval:
            return
        left, right = batch_idx * self.batch_size, (batch_idx + 1) * self.batch_size
        self.pse_t[left:right] = self._get_pse_for_one_type(i_t)
        self.pse_s[left:right] = self._get_pse_for_one_type(i_s)
        self.pse_r[left:right] = self._get_pse_for_one_type(i_r)

    def _get_pse_for_one_type(self, i_x):
        array_x: np.ndarray = ((i_x + 1.) * 127.5).permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
        array_x = array_x[:, :, :, ::-1]  # RGB to BGR
        param = self.pse_model.get_pose_exp_batch(array_x)
        return param

    def _record_hopenet_batch(self, i_t, i_s, i_r, batch_idx):
        if not self.en_hopenet_eval:
            return
        total = torch.cat((i_t, i_s, i_r), dim=0)  # in [-1,1]
        total = self.hopenet_norm(total)  # (B*3,C,H,W), in [vgg_min,vgg_max]
        yaw, pitch, roll = self.hopenet_model(total)  # each is (B*3,66)

        # param = (yaw + pitch + roll).cpu()  # to (B*3,66)
        # hope_t, hope_s, hope_r = torch.chunk(param, chunks=3, dim=0)  # each is (B,66)

        param = torch.cat((yaw, pitch, roll), dim=-1).cpu()  # to (B*3,198)
        hope_t, hope_s, hope_r = torch.chunk(param, chunks=3, dim=0)  # each is (B,198)

        left, right = batch_idx * self.batch_size, (batch_idx + 1) * self.batch_size
        self.hope_t[left:right] = hope_t.numpy()
        self.hope_s[left:right] = hope_s.numpy()
        self.hope_r[left:right] = hope_r.numpy()

    def _record_deep3d_batch(self, i_t, i_s, i_r, batch_idx):
        if not self.en_deep3d_eval:
            return
        left, right = batch_idx * self.batch_size, (batch_idx + 1) * self.batch_size
        self.deep3d_t[left:right] = self._get_deep3d_exp_for_one_type(i_t)
        self.deep3d_s[left:right] = self._get_deep3d_exp_for_one_type(i_s)
        self.deep3d_r[left:right] = self._get_deep3d_exp_for_one_type(i_r)

    def _get_deep3d_exp_for_one_type(self, i_x):
        array_x: np.ndarray = ((i_x + 1.) * 127.5).permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
        pil_x = Image.fromarray(array_x[0])
        param = self.deep3d_model.infer_image(img_pil=pil_x)
        return param['exp'].cpu().numpy()

    def _snapshot(self, i_t, i_s, i_r, batch_idx):
        if self.benchmark == 'ffplus' and (batch_idx % 10 != 0 or batch_idx >= 2000000):
            return
        img_t = self._tensor_to_arr(i_t)[0]
        img_s = self._tensor_to_arr(i_s)[0]
        img_r = self._tensor_to_arr(i_r)[0]

        if self.gpen is not None:
            img_r = self.gpen.image_infer(img_r)

        img_t = Image.fromarray(img_t)
        img_s = Image.fromarray(img_s)
        img_r = Image.fromarray(img_r)

        if self.demo_folder is not None:
            save_target_folder = os.path.join(self.demo_folder, 'target')
            save_source_folder = os.path.join(self.demo_folder, 'source')
            save_result_folder = os.path.join(self.demo_folder, 'result')
            save_name = '%05d.jpg' % batch_idx

            img_t.save(os.path.join(save_target_folder, save_name))
            img_s.save(os.path.join(save_source_folder, save_name))
            img_r.save(os.path.join(save_result_folder, save_name))

        self.demo_t_imgs.append(img_t)
        self.demo_s_imgs.append(img_s)
        self.demo_r_imgs.append(img_r)

    def test_epoch_end(self, outputs):
        # if self.en_id_eval:
        #     np.save('id_t', self.id_t)
        #     np.save('id_s', self.id_s)
        #     np.save('id_r', self.id_r)
        # if self.en_pse_eval:
        #     np.save('pse_t', self.pse_t)
        #     np.save('pse_s', self.pse_s)
        #     np.save('pse_r', self.pse_r)

        self._eval_id()
        self._eval_fixer()
        self._eval_pose_shape_expression()
        self._eval_hopenet()
        self._eval_deep3d()
        print('[%s] Evaluation finished on: %s' % (self.faceswap_model.__class__, self.benchmark))

    def _eval_id(self):
        if not self.en_id_eval:
            return
        # self.id_t = np.load('id_t.npy')
        # self.id_s = np.load('id_s.npy')
        # self.id_r = np.load('id_r.npy')

        embedding_target = self.id_t  # (dataset_len,512)
        embedding_source = self.id_s
        embedding_result = self.id_r
        embedding_source_no_repeat = self.id_s[::10]  # (dataset_len//10,512)

        ''' find dis<source, result> is small but looks not same '''
        # n = self.dataset_len
        # cos_dis_larger_64 = 0
        # cos_dis_larger_67 = 0
        # for index in range(n):
        #     vec1 = embedding_source[index]
        #     vec2 = embedding_result[index]
        #     cos_dis = vec1.dot(vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        #     if cos_dis > 0.64:
        #         cos_dis_larger_64 += 1
        #     if cos_dis > 0.67:
        #         cos_dis_larger_67 += 1
        #     print('[%d] cosine similarity = %.2f' % (index, cos_dis))
        # print('cosing similarity <0.64 = %.2f %%, <0.67=%.2f %%' % (
        #         1 - cos_dis_larger_64 / n, 1 - cos_dis_larger_67 / n))

        ''' calculate id retrieval '''
        idx_source_gt = np.arange(self.dataset_len)  # (dataset_len,), e.g. [0,1,2,3,...,dataset_len-1]
        # dists = pairwise_distances(embedding_result, embedding_source_no_repeat, metric='cosine')  # (N,N//10)
        dists = pairwise_distances(embedding_result, embedding_source, metric='cosine')  # (dataset_len,dataset_len)
        idx_source_pred = dists.argmin(axis=1)  # (dataset_len,)
        cos_sims = (1 - dists)

        if self.benchmark == 'ffplus':
            # (len(ts_list)*10,), e.g. [0,0,...,0,1,1,...,1,2,2,...,2,...]
            idx_source_gt = np.arange(self.dataset_len // 10).repeat(10)
            # (len(ts_list)*10,), e.g. [0,0,...,0,10,10,...,10,20,20,...,20,...]
            idx_source_pred = idx_source_pred // 10

        diff = np.zeros_like(idx_source_pred)
        ones = np.ones_like(idx_source_pred)
        diff[idx_source_pred != idx_source_gt] = ones[idx_source_pred != idx_source_gt]
        acc = 1. - diff.sum() / diff.shape[0]
        print('id retrieval acc = %.2f %%, cosine_sim = %.4f' % (
            acc * 100.,
            cos_sims[idx_source_gt, idx_source_gt].mean()))

    def _eval_fixer(self):
        if not self.en_fixer_eval:
            return

        embedding_target = self.fixer_t  # (dataset_len,128)
        embedding_source = self.fixer_s
        embedding_result = self.fixer_r
        embedding_source_no_repeat = self.fixer_s[::10]  # (dataset_len//10,128)

        ''' calculate id retrieval '''
        idx_source_gt = np.arange(self.dataset_len)  # (dataset_len,), e.g. [0,1,2,3,...,dataset_len-1]
        # dists = pairwise_distances(embedding_result, embedding_source_no_repeat, metric='cosine')  # (N,N//10)
        dists = pairwise_distances(embedding_result, embedding_source, metric='cosine')  # (dataset_len,dataset_len)
        idx_source_pred = dists.argmin(axis=1)  # (dataset_len,)
        cos_sims = (1 - dists)

        if self.benchmark == 'ffplus':
            # (len(ts_list)*10,), e.g. [0,0,...,0,1,1,...,1,2,2,...,2,...]
            idx_source_gt = np.arange(self.dataset_len // 10).repeat(10)
            # (len(ts_list)*10,), e.g. [0,0,...,0,10,10,...,10,20,20,...,20,...]
            idx_source_pred = idx_source_pred // 10

        diff = np.zeros_like(idx_source_pred)
        ones = np.ones_like(idx_source_pred)
        diff[idx_source_pred != idx_source_gt] = ones[idx_source_pred != idx_source_gt]
        acc = 1. - diff.sum() / diff.shape[0]
        print('fixer retrieval acc = %.2f %%, cosine_sim = %.4f' % (
            acc * 100.,
            cos_sims[idx_source_gt, idx_source_gt].mean()))

    def _eval_pose_shape_expression(self):
        if not self.en_pse_eval:
            return
        self.pse_t = np.load('pse_t.npy')
        self.pse_s = np.load('pse_s.npy')
        self.pse_r = np.load('pse_r.npy')

        pose_t, shape_t, exp_t = self._extract_pose_shape_expression(self.pse_t)
        pose_s, shape_s, exp_s = self._extract_pose_shape_expression(self.pse_s)
        pose_r, shape_r, exp_r = self._extract_pose_shape_expression(self.pse_r)

        dist_pose = mean_squared_error(pose_t, pose_r)
        dist_shape = mean_squared_error(shape_s, shape_r)
        dist_exp = mean_squared_error(exp_t, exp_r)
        print('pose MSE = %.3f, exp MSE = %.3f, shape MSE = %.3f' % (dist_pose, dist_exp, dist_shape))

        ''' L2 = sqrt(MSE * dim) '''
        dim_pose, dim_shape, dim_exp = pose_t.shape[-1], shape_t.shape[-1], exp_t.shape[-1]
        print('param dims:', dim_pose, dim_shape, dim_exp)
        print('pose L2 = %.3f, exp L2 = %.3f, shape L2 = %.3f' % (np.sqrt(dist_pose * dim_pose),
                                                                  np.sqrt(dist_exp * dim_exp),
                                                                  np.sqrt(dist_shape * dim_shape)))

    def _eval_hopenet(self):
        if not self.en_hopenet_eval:
            return

        # pose_t = self.hope_t
        # pose_s = self.hope_s
        # pose_r = self.hope_r
        #
        # dist_pose = mean_squared_error(pose_t, pose_r)
        # print('pose MSE = %.3f' % (dist_pose))
        #
        # ''' L2 = sqrt(MSE * dim) '''
        # dim_pose = pose_t.shape[-1]
        # print('param dims:', dim_pose)
        # print('pose L2 = %.3f' % (np.sqrt(dist_pose * dim_pose) / 2))

        yaw_t, pitch_t, roll_t = self._extract_hopenet_yaw_pitch_roll(self.hope_t)  # each is (B,66)
        yaw_s, pitch_s, roll_s = self._extract_hopenet_yaw_pitch_roll(self.hope_s)  # each is (B,66)
        yaw_r, pitch_r, roll_r = self._extract_hopenet_yaw_pitch_roll(self.hope_r)  # each is (B,66)

        yaw_mse, yaw_l2, yaw_dim = self._calc_mse_l2(yaw_r, yaw_t)
        pitch_mse, pitch_l2, pitch_dim = self._calc_mse_l2(pitch_r, pitch_t)
        roll_mse, roll_l2, roll_dim = self._calc_mse_l2(roll_r, roll_t)
        print('hopenet average MSE=%.3f, L2=%.3f, dim=%d' % (
            (yaw_mse + pitch_mse + roll_mse) / 3,
            (yaw_l2 + pitch_l2 + roll_l2) / 3,
            yaw_dim,
        ))

    def _eval_deep3d(self):
        if not self.en_deep3d_eval:
            return
        exp_t = torch.tensor(self.deep3d_t)
        exp_s = torch.tensor(self.deep3d_s)
        exp_r = torch.tensor(self.deep3d_r)

        exp_dim = exp_r.shape[-1]
        mse_sum = torch.nn.functional.mse_loss(exp_t, exp_r, reduction='mean')
        l2 = torch.sqrt(mse_sum * exp_dim)
        print('deep3d MSE_SUM=%.3f, L2=%.3f, dim=%d' % (mse_sum.data, l2.data, exp_dim))

    @staticmethod
    def _extract_pose_shape_expression(param: np.ndarray):
        pose, shape, expression = param[:, :12], param[:, 12:-10], param[:, -10:]
        return pose, shape, expression  # (N,12), (N,40), (N,10)

    @staticmethod
    def _extract_hopenet_yaw_pitch_roll(param: np.ndarray):
        yaw, pitch, roll = param[:, :66], param[:, 66:-66], param[:, -66:]
        return yaw, pitch, roll  # (N,66), (N,66), (N,66)

    @staticmethod
    def _calc_mse_l2(vec1: np.ndarray, vec2: np.ndarray):
        mse_dist = mean_squared_error(vec1, vec2)
        ''' L2 = sqrt(MSE * dim) '''
        dim = vec1.shape[-1]
        l2_dist = np.sqrt(mse_dist * dim) / 2
        return mse_dist, l2_dist, dim

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
        test_dataset = None
        if self.benchmark == 'ffplus':
            test_dataset = FFPlusEvalDataset(
                ffplus_root='/apdcephfs/share_1290939/gavinyuan/datasets/ff+/',
                image_size=256,
                frames_per_id=10,
            )
        elif self.benchmark == 'celebahq':
            test_dataset = CelebaHQEvalDataset(
                ts_list=celebahq_ts_list,
                image_size=256,
                dataset_len=400
            )
        elif self.benchmark == 'ffhq':
            test_dataset = FFHQEvalDataset(
                ts_list=ffhq_ts_list,
                ffhq_pickle='/gavin/datasets/ffhq/1024x1024.pickle',
                image_size=256,
            )
        elif self.benchmark == 'web':
            test_dataset = WebEvalDataset(
                image_size=256,
                source_cnt=14,
                target_cnt=14,
            )
        elif self.benchmark == 'vggface2':
            from modules.dataset.dataloader import BatchValDataset
            test_dataset = BatchValDataset(
                img_root="/gavin/datasets/original/image_512_quality.pickle",
                image_size=256,
                id_cnt=5000,
            )
        else:
            raise ValueError('Benchmark not supported.')
        return test_dataset

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=8,
            shuffle=False,
            drop_last=False,
        )


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


class EvaluatorFaceShifter(Evaluator):
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

        super(EvaluatorFaceShifter, self).__init__(**kwargs)

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
        i_r_mouth_mask = self._get_any_mask(i_r, par=[11, 12, 13])  # (B,1,H,W)

        ''' dilate and blur by cv2 '''
        i_r_mouth_mask = self._tensor_to_arr(i_r_mouth_mask)[0]  # (H,W,C)
        i_r_mouth_mask = cv2.dilate(i_r_mouth_mask, (20, 20), iterations=1)

        kernel_size = (5, 5)
        blur_size = tuple(2 * j + 1 for j in kernel_size)
        i_r_mouth_mask = cv2.GaussianBlur(i_r_mouth_mask, blur_size, 0)  # (H,W,C)
        i_r_mouth_mask = i_r_mouth_mask.squeeze()[None, :, :, None]  # (1,H,W,1)
        i_r_mouth_mask = self._arr_to_tensor(i_r_mouth_mask, norm=False)  # in [0,1]

        return helper_face * i_r_mouth_mask + i_r * (1 - i_r_mouth_mask)

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
        mask = torch.zeros_like(parsing)
        for p in par:
            mask = mask + ((parsing == p).float())
        mask = mask.unsqueeze(1)
        mask = F.interpolate(mask, size=ori_size, mode="bilinear", align_corners=True)
        return mask

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
        if load_path == '' or load_path is None:
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


class EvaluatorFaceShifterHanbang(EvaluatorFaceShifter):
    def __init__(self, **kwargs):
        self.load_path = ''
        self.pt_path = make_abs_path('../../trainer/faceshifter/extracted_ckpt/G_step14999_v2.pth')
        super(EvaluatorFaceShifterHanbang, self).__init__(**kwargs)

    def test_dataloader(self):
        # test_dataset = FFPlusEvalDataset(
        #     image_size=256,
        # )
        test_dataset = CelebaHQEvalDatasetHanbang(
            ts_list=celebahq_ts_list,
            image_size=256,
            dataset_len=400,
        )
        # test_dataset = FFHQEvalDataset(
        #     ts_list=ffhq_ts_list,
        #     ffhq_pickle='/gavin/datasets/ffhq/1024x1024.pickle',
        #     image_size=256,
        # )
        # test_dataset = WebEvalDataset(
        #     image_size=256,
        #     source_cnt=20,
        #     target_cnt=20,
        # )

        return DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            num_workers=8,
            shuffle=False,
            drop_last=False,
        )


class EvaluatorInfoSwap(Evaluator):
    def _callback_load_faceswap_model(self,
                             ):
        from infoswap.inference_model import InfoSwapInference
        self.faceswap_model = InfoSwapInference()
        self.faceswap_model.load_model()

    def callback_infer_batch(self, i_s, i_t):
        i_r = self.faceswap_model.infer_batch(i_s, i_t)
        i_r = i_r.cuda()
        i_r = F.interpolate(i_r, size=(256, 256), mode='bilinear', align_corners=True)
        return i_r


class EvaluatorHiRes(Evaluator):
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


class EvaluatorMegaFS(Evaluator):
    def _callback_load_faceswap_model(self):
        from megafs.image_infer import MegaFSImageInfer
        self.faceswap_model = MegaFSImageInfer()

    def callback_infer_batch(self, i_s, i_t):
        i_r = self.faceswap_model.image_infer(source_tensor=i_s,
                                              target_tensor=i_t)
        if torch.isnan(i_r).any():
            print('NAN in i_r, will be set to 0.')
            i_r = torch.where(torch.isnan(i_r), torch.full_like(i_r, 0.), i_r)
        i_r = i_r.clamp(-1, 1)
        return i_r


class EvaluatorHearNet(Evaluator):
    def __init__(self,
                 load_path: str,
                 pt_path: str,
                 **kwargs
                 ):
        self.load_path = load_path
        self.pt_path = pt_path
        super(EvaluatorHearNet, self).__init__(**kwargs)

    def _callback_load_faceswap_model(self,):
        self.set_weight_path()
        ckpt_path = self.load_path
        pt_path = self.pt_path

        self._extract_hear_net(load_path=ckpt_path, path=pt_path)
        hear = self._load_extracted(path=pt_path)
        hear = hear.cuda()
        self.faceswap_model = hear
        print('Hearnet model loaded from %s.' % ckpt_path)

    def callback_infer_batch(self, i_s, i_t):
        y_st, y_hat_st = self.faceswap_model(i_s, i_t)
        return y_st

    @staticmethod
    def _load_extracted(path=".pth"):
        from modules.networks.faceshifter import FSHearNet
        hear = FSHearNet(
            aei_path=make_abs_path("../../trainer/faceshifter/extracted_ckpt/hear_tmp.pth")
        )
        hear.load_state_dict(torch.load(path, "cpu"), strict=False)
        hear.eval()
        return hear

    @staticmethod
    def _extract_hear_net(
            load_path=".ckpt",
            path=".pth",
        ):
        from trainer.faceshifter.hearnet_pl import HearNetPL
        net = HearNetPL()
        checkpoint = torch.load(
            load_path,
            map_location="cpu",
        )
        net.load_state_dict(checkpoint["state_dict"], strict=False)
        net.eval()

        torch.save(net.hear.state_dict(), path)


class EvaluatorSimSwapOfficial(Evaluator):
    def _callback_load_faceswap_model(self):
        from simswap.image_infer import SimSwapOfficialImageInfer
        self.faceswap_model = SimSwapOfficialImageInfer()

    def callback_infer_batch(self, i_s, i_t):
        i_r = self.faceswap_model.image_infer(source_tensor=i_s,
                                              target_tensor=i_t)
        i_r = i_r.clamp(-1, 1)
        return i_r


class EvaluatorSimSwap(Evaluator):
    def __init__(self,
                 load_path: str,
                 pt_path: str,
                 **kwargs
                 ):
        self.load_path = load_path
        self.pt_path = pt_path
        self.use_official = 'off' in pt_path
        self.trick = None

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

        super(EvaluatorSimSwap, self).__init__(**kwargs)

        ''' Face Recognition Network '''
        # from modules.third_party.arcface import iresnet100
        # netArc_pth = "/apdcephfs_cq2/share_1290939/gavinyuan/code/FaceShifter/faceswap/faceswap/" \
        #              "checkpoints/face_id/ms1mv3_arcface_r100_fp16_backbone.pth"  # opt.Arc_path
        # self.netArc = iresnet100(pretrained=False, fp16=False)
        # self.netArc.load_state_dict(torch.load(netArc_pth, map_location="cpu"))
        # self.netArc.eval()

    def _callback_load_faceswap_model(self):
        load_path = self.load_path
        pt_path = self.pt_path

        self._extract_generator(load_path=load_path, pth_path=pt_path)
        G = self._load_extracted(pth_path=pt_path)
        G = G.cuda()
        self.faceswap_model = G
        print('Simswap model (%s) loaded from %s.' % (type(G), load_path))

    def callback_infer_batch(self, i_s, i_t):
        i_r = self.faceswap_model(source=i_s, target=i_t,
                                  net_arc=self.netArc,
                                  mouth_net=self.mouth_net,
                                  )
        if 'post' in self.pt_path:
            from inference.gradio_demo.tricks import Trick
            if self.trick is None:
                self.trick = Trick()
            target_hair_mask = self.trick.get_any_mask(i_t, par=[0, 17])
            target_hair_mask = self.trick.smooth_mask(target_hair_mask)
            i_r = target_hair_mask * i_t + (target_hair_mask * (-1) + 1) * i_r
        # lo, hi = i_r.min(), i_r.max()
        # i_r = (i_r - lo) / (hi - lo)  # in [0,1]
        i_r = i_r.clamp(-1, 1)
        return i_r

    def _load_extracted(self, pth_path: str):
        from modules.networks.simswap import Generator_Adain_Upsample
        G = Generator_Adain_Upsample(
            input_nc=3, output_nc=3, latent_size=512, n_blocks=9, deep=False,
            mouth_net_param=self.mouth_net_param
        )
        G.load_state_dict(torch.load(pth_path, "cpu"), strict=False)
        G.eval()
        return G

    def _extract_generator(self, load_path: str, pth_path: str):
        from trainer.simswap.simswap_pl import SimSwapPL
        import yaml
        with open(make_abs_path('../../trainer/simswap/config.yaml'), 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        config['mouth_net'] = self.mouth_net_param

        net = SimSwapPL(config=config, use_official_arc=self.use_official)
        checkpoint = torch.load(load_path, map_location="cpu")
        net.load_state_dict(checkpoint["state_dict"], strict=False)
        net.eval()
        self.mouth_net = net.mouth_net  # maybe None
        self.netArc = net.netArc

        G = net.netG  # generator
        torch.save(G.state_dict(), pth_path)


if __name__ == '__main__':

    ''' Modify the ckpt_path_list as you need '''
    task_list = [
        # ("hanbang", "hanbang"),  # fixed

        # ("hires", "hires"),  # fixed
        # ("megafs", "megafs"),  # fixed
        # ("../../trainer/simswap/out/simswap_vanilla_3/epoch=705-step=1511999.ckpt",
        #   "./extracted_ckpt/G_tmp_sv3.pth"),  # hififace
        # ("infoswap", "infoswap"),  # fixed
        # ("simswap_official", "simswap_official"),  # fixed
        ("../../trainer/simswap/out/simswap_vanilla_4/epoch=694-step=1487999.ckpt",
         "./extracted_ckpt/G_tmp_sv4_off.pth"),  # worse than sv2 and sv3
        # ("../../trainer/simswap/out/simswap_triplet_5/epoch=12-step=782999.ckpt",
        #  "./extracted_ckpt/G_mouth1_st5.pth"),
        # ("../../trainer/faceshifter/out/faceshifter_vanilla_5/epoch=11-step=548999.ckpt",
        #  "./extracted_ckpt/G_tmp_v5.pth"),  # v5
        # ("../../trainer/faceshifter/out/faceshifter_vanilla_6/epoch=20-step=968999.ckpt",
        #  "./extracted_ckpt/G_tmp_v6.pth"),

        #
        # # ("../../trainer/simswap/out/simswap_vanilla_1/epoch=2-step=98999.ckpt",
        # #  "./extracted_ckpt/G_tmp_sv1.pth"),
        # ("../../trainer/simswap/out/simswap_vanilla_2/epoch=516-step=1106999.ckpt",
        #  "./extracted_ckpt/G_tmp_sv2.pth"),
        # ("../../trainer/simswap/out/simswap_vanilla_3/epoch=705-step=1511999.ckpt",
        #  "./extracted_ckpt/G_tmp_sv3.pth"),
        # ("../../trainer/simswap/out/simswap_vanilla_4/epoch=694-step=1487999.ckpt",
        #  "./extracted_ckpt/G_tmp_sv4_off.pth"),  # worse than sv2 and sv3
        # # ("../../trainer/simswap/out/simswap_triplet_1/epoch=24-step=1205999.ckpt",
        # #  "./extracted_ckpt/G_tmp_st1.pth"),
        # # ("../../trainer/simswap/out/simswap_triplet_2/epoch=30-step=1967999.ckpt",
        # #  "./extracted_ckpt/G_tmp_st2.pth"),
        # # ("../../trainer/simswap/out/simswap_triplet_3/epoch=11-step=725999.ckpt",
        # #  "./extracted_ckpt/G_mouth1_st3.pth"),
        # # ("../../trainer/simswap/out/simswap_triplet_4/epoch=11-step=728999.ckpt",
        # #  "./extracted_ckpt/G_mouth2_st4.pth"),
        # ("../../trainer/simswap/out/simswap_triplet_5/epoch=12-step=782999.ckpt",
        #  "./extracted_ckpt/G_mouth1_st5.pth"),
        # # ("../../trainer/simswap/out/simswap_triplet_6/epoch=10-step=590999.ckpt",
        # #  "./extracted_ckpt/G_mouth2_st6.pth"),
        # ("../../trainer/simswap/out/simswap_triplet_7/epoch=570-step=1466999.ckpt",
        #  "./extracted_ckpt/G_mouth1_st7.pth"),
        ("../../trainer/simswap/out/simswap_fixernet_1/epoch=193-step=413999.ckpt",
         "./extracted_ckpt/G_mouth1_st_f1.pth"),
        # ("../../trainer/faceshifter/out/faceshifter_vanilla_4/epoch=11-step=536999.ckpt",
        #  "./extracted_ckpt/G_tmp_v4.pth"),  # fixed

        # ("../../trainer/faceshifter/out/faceshifter_vanilla_6/epoch=20-step=968999.ckpt",
        #  "./extracted_ckpt/G_tmp_v6.pth"),
        # ("../../trainer/faceshifter/out/faceshifter_vanilla_7/epoch=12-step=527999.ckpt",
        #  "./extracted_ckpt/G_mouth1_v7.pth"),

        # ("../../trainer/faceshifter/out/triplet10w_9/epoch=13-step=626999.ckpt",
        #  "./extracted_ckpt/G_tmp_t9.pth"),
        # ("../../trainer/faceshifter/out/triplet10w_10/epoch=28-step=1280999.ckpt",
        #  "./extracted_ckpt/G_tmp_t10.pth"),
        # ("../../trainer/faceshifter/out/triplet10w_11/epoch=25-step=1277999.ckpt",
        #  "./extracted_ckpt/G_tmp_t11.pth"),
        # ("../../trainer/faceshifter/out/triplet10w_12/epoch=19-step=1124999.ckpt",
        #  "./extracted_ckpt/G_tmp_t12.pth"),
        # ("../../trainer/faceshifter/out/triplet10w_13/epoch=24-step=1172999.ckpt",
        #  "./extracted_ckpt/G_tmp_t13.pth"),
        # ("../../trainer/faceshifter/out/triplet10w_14/epoch=9-step=458999.ckpt",
        #  "./extracted_ckpt/G_tmp_t14.pth"),
        # ("../../trainer/faceshifter/out/triplet10w_15/epoch=14-step=743999.ckpt",
        #  "./extracted_ckpt/G_tmp_t15.pth"),
        # ("../../trainer/faceshifter/out/triplet10w_16/epoch=5-step=281999.ckpt",
        #  "./extracted_ckpt/G_tmp_t16.pth"),
        # ("../../trainer/faceshifter/out/triplet10w_17/epoch=11-step=743999.ckpt",
        #  "./extracted_ckpt/G_tmp_t17.pth"),
        # ("../../trainer/faceshifter/out/triplet10w_18/epoch=10-step=677999.ckpt",
        #  "./extracted_ckpt/G_tmp_t18.pth"),
        # ("../../trainer/faceshifter/out/triplet10w_19/epoch=9-step=530999.ckpt",
        #  "./extracted_ckpt/G_mouth1_t19.pth"),
        # ("../../trainer/faceshifter/out/triplet10w_20/epoch=9-step=539999.ckpt",
        #  "./extracted_ckpt/G_mouth2_t20.pth"),
        # ("../../trainer/faceshifter/out/triplet10w_21/epoch=5-step=359999.ckpt",
        #  "./extracted_ckpt/G_mouth1_t21.pth"),
        # ("../../trainer/faceshifter/out/triplet10w_22/epoch=9-step=617999.ckpt",
        #  "./extracted_ckpt/G_mouth1_t22_post.pth"),
        # ("../../trainer/faceshifter/out/triplet10w_23/epoch=6-step=344999.ckpt",
        #  "./extracted_ckpt/G_mouth2_t23.pth"),
        # ("../../trainer/faceshifter/out/triplet10w_24/epoch=9-step=602999.ckpt",
        #  "./extracted_ckpt/G_mouth1_t24_post.pth"),
        # ("../../trainer/faceshifter/out/triplet10w_25/epoch=8-step=548999.ckpt",
        #  "./extracted_ckpt/G_mouth1_t25_post.pth"),
        # ("../../trainer/faceshifter/out/triplet10w_26/epoch=9-step=605999.ckpt",
        #  "./extracted_ckpt/G_mouth1_t26_post.pth"),
        # ("../../trainer/faceshifter/out/triplet10w_27/epoch=11-step=665999.ckpt",
        #  "./extracted_ckpt/G_mouth1_t27_post.pth"),  # t27

        # # ("../../trainer/faceshifter/out/triplet10w_28/epoch=18-step=695999.ckpt",
        # #  "./extracted_ckpt/G_mouth1_t28_post.pth"),  # bad
        # # ("../../trainer/faceshifter/out/triplet10w_29/epoch=12-step=668999.ckpt",
        # #  "./extracted_ckpt/G_mouth1_t29_post.pth"),  # bad
        #
        # # ("../../trainer/faceshifter/out/triplet10w_30/epoch=14-step=803999.ckpt",
        # #  "./extracted_ckpt/G_mouth1_t30_post.pth"),
        # # ("../../trainer/faceshifter/out/triplet10w_31/epoch=11-step=638999.ckpt",
        # #  "./extracted_ckpt/G_mouth1_t31_post.pth"),  # bad
        # ("../../trainer/faceshifter/out/triplet10w_32/epoch=9-step=509999.ckpt",
        #  "./extracted_ckpt/G_mouth1_t32_post.pth"),

        # ("../../trainer/faceshifter/out/faceshifter_vanilla_5/epoch=11-step=548999.ckpt",
        #  "./extracted_ckpt/G_tmp_v5.pth"),  # v5
        # ("../../trainer/faceshifter/out/faceshifter_vanilla_8/epoch=12-step=581999.ckpt",
        #  "./extracted_ckpt/G_mouth1_v8.pth"),  # v8
        #
        # ("../../trainer/faceshifter/out/triplet10w_35/epoch=7-step=446999.ckpt",
        #  "./extracted_ckpt/G_t35_post.pth"),  # 200k
        # ("../../trainer/faceshifter/out/triplet10w_34/epoch=13-step=737999.ckpt",
        #  "./extracted_ckpt/G_t34_post.pth"),  # 600k
        #
        # ("../../trainer/faceshifter/out/triplet10w_33/epoch=10-step=596999.ckpt",
        #  "./extracted_ckpt/G_mouth1_t33.pth"),  # t33 little better than t27

        # ("../../trainer/faceshifter/out/triplet10w_36/epoch=10-step=578999.ckpt",
        #  "./extracted_ckpt/G_mouth1_t36.pth"),  #
        # ("../../trainer/faceshifter/out/triplet10w_37/epoch=10-step=584999.ckpt",
        #  "./extracted_ckpt/G_mouth1_t37.pth"),  #
        # ("../../trainer/faceshifter/out/triplet10w_38/epoch=11-step=440999.ckpt",
        #  "./extracted_ckpt/G_mouth1_t38_post.pth"),  #
        # ("../../trainer/faceshifter/out/triplet10w_38/epoch=13-step=491999.ckpt",
        #  "./extracted_ckpt/G_mouth1_t40.pth"),  #
        # ("../../trainer/faceshifter/out/triplet10w_38/epoch=16-step=629999.ckpt",
        #  "./extracted_ckpt/G_mouth1_t41.pth"),  #
        # ("../../trainer/faceshifter/out/triplet10w_39/epoch=12-step=470999.ckpt",
        #  "./extracted_ckpt/G_mouth1_t39.pth"),  #
    ]
    for idx in range(len(task_list)):
        ckpt_path, pt_path = task_list[idx]
        if '../' in ckpt_path:
            task_list[idx] = (make_abs_path(ckpt_path), pt_path)

        ckpt_path, pt_path = task_list[idx]
        if not os.path.exists(ckpt_path) and ckpt_path[0] == '/':
            if 'faceshifter' in ckpt_path:
                task_list[idx] = (ckpt_path.replace('/apdcephfs_cq2/share_1290939/gavinyuan/code/FaceSwapping/trainer/faceshifter/out/',
                                                    '/apdcephfs/share_1290939/gavinyuan/out/'), pt_path)
            elif 'simswap' in ckpt_path:
                task_list[idx] = (ckpt_path.replace('/apdcephfs_cq2/share_1290939/gavinyuan/code/FaceSwapping/trainer/simswap/out/',
                                  '/apdcephfs/share_1290939/gavinyuan/out/'), pt_path)

    import argparse
    parser = argparse.ArgumentParser(description="benchmark evaluation")
    parser.add_argument("-b", "--benchmark", type=str, choices=['celebahq', 'ffhq', 'ffplus', 'web', 'vggface2'],
                        default='celebahq', help='benchmark name')
    parser.add_argument("-o", "--out_name", type=str, default='celebahq1.jpg', help='save figure name')
    parser.add_argument("--demo", dest="demo", action="store_true", default=False, help='save demo folder or not')

    parser.add_argument("--id", dest="id_eval", action="store_true", help='evaluate id or not')
    parser.add_argument("--no-id", dest="id_eval", action="store_false", help='evaluate id or not')
    parser.add_argument("--fixer", dest="fixer_eval", action="store_true", help='evaluate fixer or not')
    parser.add_argument("--no-fixer", dest="fixer_eval", action="store_false", help='evaluate fixer or not')
    parser.add_argument("--pse", dest="pse_eval", action="store_true", help='evaluate pose,shape,expression or not')
    parser.add_argument("--no-pse", dest="pse_eval", action="store_false", help='evaluate pose,shape,expression or not')
    parser.add_argument("--hope", dest="hope_eval", action="store_true", help='evaluate hopenet pose or not')
    parser.add_argument("--no-hope", dest="hope_eval", action="store_false", help='evaluate hopenet pose or not')
    parser.add_argument("--deep3d", dest="deep3d_eval", action="store_true", help='evaluate deep3d exp or not')
    parser.add_argument("--no-deep3d", dest="deep3d_eval", action="store_false", help='evaluate deep3d exp or not')
    parser.set_defaults(id_eval=False)
    parser.set_defaults(pse_eval=False)
    parser.set_defaults(hope_eval=False)
    parser.set_defaults(deep3d_eval=False)

    args = parser.parse_args()

    mouth_helper_pl = EvaluatorFaceShifter(
        load_path="../../weights/reliableswap_weights/ckpt/triplet10w_34/epoch=13-step=737999.ckpt",
        pt_path="../../weights/reliableswap_weights/extracted_pth/G_t34_helper_post.pth",
        benchmark=None,
        demo_folder=None,
    )
    print("[Mouth helper] loaded.")

    ''' FaceShifter evaluation '''
    evaluators = []
    for idx, task in enumerate(task_list):
        ckpt_path, pt_path = task
        demo_folder = 'demo_%s' % (os.path.split(os.path.dirname(ckpt_path))[-1])

        if os.path.exists(ckpt_path) and ('faceshifter' in ckpt_path or 'triplet10w' in ckpt_path):
            mouth_helper = None
            gpen = None
            if 'triplet' in ckpt_path:
                mouth_helper = mouth_helper_pl.faceswap_model
                # gpen = GPENImageInfer()

            evaluator = EvaluatorFaceShifter(
                load_path=ckpt_path,
                pt_path=pt_path,
                mouth_helper=mouth_helper,
                gpen=gpen,
                benchmark=args.benchmark,
                demo_folder=demo_folder,
                batch_size=1,
                en_id_eval=args.id_eval,
                en_fixer_eval=args.fixer_eval,
                en_pse_eval=args.pse_eval,
                en_hopenet_eval=args.hope_eval,
                en_deep3d_eval=args.deep3d_eval,
            )
        elif 'hanbang' in ckpt_path:
            evaluator = EvaluatorFaceShifterHanbang(
                benchmark=args.benchmark,
                demo_folder='demo_faceshifter_hanbang',
                batch_size=1,
                en_id_eval=args.id_eval,
                en_fixer_eval=args.fixer_eval,
                en_pse_eval=args.pse_eval,
                en_hopenet_eval=args.hope_eval,
                en_deep3d_eval=args.deep3d_eval,
            )
        elif 'infoswap' in ckpt_path:
            evaluator = EvaluatorInfoSwap(
                benchmark=args.benchmark,
                demo_folder='demo_infoswap',
                batch_size=1,
                en_id_eval=args.id_eval,
                en_fixer_eval=args.fixer_eval,
                en_pse_eval=args.pse_eval,
                en_hopenet_eval=args.hope_eval,
                en_deep3d_eval=args.deep3d_eval,
            )
        elif 'hires' in ckpt_path:
            evaluator = EvaluatorHiRes(
                benchmark=args.benchmark,
                demo_folder='demo_hires',
                batch_size=1,
                en_id_eval=args.id_eval,
                en_fixer_eval=args.fixer_eval,
                en_pse_eval=args.pse_eval,
                en_hopenet_eval=args.hope_eval,
                en_deep3d_eval=args.deep3d_eval,
            )
        elif 'megafs' in ckpt_path:
            evaluator = EvaluatorMegaFS(
                benchmark=args.benchmark,
                demo_folder='demo_megafs',
                batch_size=1,
                en_id_eval=args.id_eval,
                en_fixer_eval=args.fixer_eval,
                en_pse_eval=args.pse_eval,
                en_hopenet_eval=args.hope_eval,
                en_deep3d_eval=args.deep3d_eval,
            )
        elif 'simswap_official' in ckpt_path:
            evaluator = EvaluatorSimSwapOfficial(
                benchmark=args.benchmark,
                demo_folder='demo_simswap_official',
                batch_size=1,
                en_id_eval=args.id_eval,
                en_fixer_eval=args.fixer_eval,
                en_pse_eval=args.pse_eval,
                en_hopenet_eval=args.hope_eval,
                en_deep3d_eval=args.deep3d_eval,
            )
        elif 'simswap' in ckpt_path:
            evaluator = EvaluatorSimSwap(
                load_path=ckpt_path,
                pt_path=pt_path,
                benchmark=args.benchmark,
                demo_folder=demo_folder,
                batch_size=1,
                en_id_eval=args.id_eval,
                en_fixer_eval=args.fixer_eval,
                en_pse_eval=args.pse_eval,
                en_hopenet_eval=args.hope_eval,
                en_deep3d_eval=args.deep3d_eval,
            )
        else:
            raise ValueError('[%d] ckpt_path not supported: %s' % (idx, ckpt_path))

        evaluators.append(evaluator)

    trainer = pl.Trainer(
        logger=False,
        gpus=1,
        distributed_backend='dp',
        benchmark=True,
    )
    for evaluator in evaluators:
        trainer.test(evaluator)

    vis_lists = [evaluators[0].demo_t_imgs, evaluators[0].demo_s_imgs]  # [target, source]
    for evaluator in evaluators:
        vis_lists.append(evaluator.demo_r_imgs)  # result

    if args.benchmark == 'web' and False:
        shuffle_idx = list(np.arange(len(vis_lists[0])).astype(np.uint))
        random.shuffle(shuffle_idx)
        for i, col in enumerate(vis_lists):
            shuffled = []
            for j in range(len(col)):
                shuffled.append(col[shuffle_idx[j]])
            vis_lists[i] = shuffled

    if args.demo or args.id_eval or args.fixer_eval or args.hope_eval or args.deep3d_eval:
        print('Quantitative evaluation does not save images. Existing.')
        exit(0)
    from inference.ffplus.image_cat import cat_cols_and_save, save_each_row
    cat_cols_and_save(vis_lists, save_name=args.out_name)
    save_each_row(vis_lists, os.path.splitext(os.path.basename(args.out_name))[0])
