import os
import pickle

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
from pytorch_fid import fid_score

from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import mean_squared_error


def save_intermediate_triplet_to_folder(
        demo_folder: str = "/gavin/code/FaceSwapping/supervision/demo_snapshot_lia/",
        max_cnt: int = 100,
        triplet_step: int = 0,
        to_folder: str = "/gavin/datasets/triplet_lia_intermediate_demo",
        ):
    to_folder = os.path.join(to_folder, 'step_%d' % triplet_step)
    inter_name1 = {'s': 'source.jpg', 't': 'target.jpg',
                   'st': 'reen_st.jpg', 'ts': 'reen_ts.jpg'}
    inter_name2 = {'s': 'source.jpg', 't': 'target.jpg',
                   'st': 'mb_st.jpg', 'ts': 'mb_ts.jpg'}
    inter_name3 = {'s': 'source.jpg', 't': 'target.jpg',
                   'st': 'output_st.jpg', 'ts': 'output_ts.jpg'}
    inter_names = [inter_name1, inter_name2, inter_name3]
    trip_filename = inter_names[triplet_step]

    triplet_folders = os.listdir(demo_folder)
    triplet_folders.sort()
    triplet_folders = [os.path.join(demo_folder, x) for x in triplet_folders]

    if os.path.exists(to_folder):
        os.system('rm -r %s' % to_folder)
    for k in trip_filename.keys():
        os.makedirs(os.path.join(to_folder, k), exist_ok=True)

    cnt = 0
    for folder in tqdm(triplet_folders, desc='copying'):
        to_name = '%05d.jpg' % cnt
        for k, v in trip_filename.items():
            cmd = "cp %s %s" % (os.path.join(folder, v),
                                os.path.join(to_folder, k, to_name))
            os.system(cmd)
        cnt += 1
        if cnt >= max_cnt:
            return


def save_triplet_to_folder(
        triplet_pickle: str = "/gavin/datasets/triplet_lia_0_600000.pickle",
        triplet_ratio: int = 10,
        to_folder: str = "/gavin/datasets/triplet_lia_demo"
        ):
    trip_filename = {'s': 'source.jpg', 't': 'target.jpg',
                     'st': 'output_st.jpg', 'ts': 'output_ts.jpg'}
    with open(triplet_pickle, "rb") as handle:
        triplet_list = pickle.load(handle)
    ''' Format, list:
        ['xxx/triplet/00000000_00008210',
         'xxx/triplet/00000044_00008316',]
    '''
    triplet_used = int(len(triplet_list) * triplet_ratio / 100)
    triplet_folders = triplet_list[:triplet_used]
    print('Triplet dataset loaded from %s, folders %d, used %d' % (
        triplet_pickle, len(triplet_list), len(triplet_folders)))

    if os.path.exists(to_folder):
        os.system('rm -r %s' % to_folder)
    for k in trip_filename.keys():
        os.makedirs(os.path.join(to_folder, k), exist_ok=True)

    cnt = 0
    for folder in tqdm(triplet_folders, desc='copying'):
        to_name = '%05d.jpg' % cnt
        for k, v in trip_filename.items():
            cmd = "cp %s %s" % (os.path.join(folder, v),
                                os.path.join(to_folder, k, to_name))
            os.system(cmd)
        cnt += 1


def calc_fid(stage: int = 1, step: int = -1):
    if stage == 1:
        folder1 = '/gavin/code/FaceSwapping/inference/ffplus/demo_triplet10w_38/source'
        folder2 = '/gavin/code/FaceSwapping/inference/ffplus/demo_triplet10w_38/source'
    else:
        folder1 = '/gavin/datasets/triplet_lia_demo/t'
        folder2 = '/gavin/datasets/triplet_lia_demo/s'
        if step >= 0:
            folder1 = '/gavin/datasets/triplet_lia_intermediate_demo/step_%d/t' % step
            folder2 = '/gavin/datasets/triplet_lia_intermediate_demo/step_%d/ts' % step

    val = fid_score.calculate_fid_given_paths([folder1, folder2], batch_size=16, device=0, dims=2048, num_workers=4)
    print('FID = %.2f' % val)


class CalcIdPoseExp(object):
    def __init__(self, demo_folder: str, batch_size: int = 40):
        trip_filename = {'s': 'source.jpg', 't': 'target.jpg',
                         'st': 'output_st.jpg', 'ts': 'output_ts.jpg'}

        class FolderDataset(Dataset):
            def __init__(self, root_folder: str):
                super(FolderDataset, self).__init__()
                self.root_folder = root_folder
                self.img_names = os.listdir(os.path.join(root_folder, 's'))
                self.transform = transforms.Compose(
                    [
                        transforms.Resize((256, 256)),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    ]
                )
                print('FolderDataset loaded from %s, len = %d' % (root_folder, self.__len__()))

            def __getitem__(self, index):
                i_s = Image.open(os.path.join(self.root_folder,
                                              's',
                                              self.img_names[index])).convert('RGB')
                i_t = Image.open(os.path.join(self.root_folder,
                                              't',
                                              self.img_names[index])).convert('RGB')
                i_st = Image.open(os.path.join(self.root_folder,
                                               'st',
                                               self.img_names[index])).convert('RGB')
                i_ts = Image.open(os.path.join(self.root_folder,
                                               'ts',
                                               self.img_names[index])).convert('RGB')
                i_s = self.transform(i_s)
                i_t = self.transform(i_t)
                i_st = self.transform(i_st)
                i_ts = self.transform(i_ts)
                return {
                    "s": i_s,
                    "t": i_t,
                    "st": i_st,
                    "ts": i_ts
                }

            def __len__(self):
                return len(self.img_names)

        self.batch_size = batch_size
        self.demo_folder = demo_folder
        self.dataset = FolderDataset(demo_folder)
        self.dataset_len = len(self.dataset)
        self.dataloader = DataLoader(self.dataset,
                                     batch_size=batch_size, num_workers=12, shuffle=False)

        ''' ID retrieval '''
        from modules.third_party import cosface
        # self.id_model = iresnet100().cuda()
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

        ''' Hopenet Pose '''
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
        from deep3d.image_infer import Deep3DImageInfer
        self.deep3d_model = Deep3DImageInfer()
        self.deep3d_t = np.zeros((self.dataset_len, 64), dtype=np.float32)
        self.deep3d_s = np.zeros_like(self.deep3d_t, dtype=np.float32)
        self.deep3d_r = np.zeros_like(self.deep3d_t, dtype=np.float32)

    def _record_id_batch(self, i_t, i_s, i_r, batch_idx):
        total = torch.cat((i_t, i_s, i_r), dim=0)
        total = F.interpolate(total, size=112, mode="bilinear", align_corners=True)
        embeddings: torch.Tensor = self.id_model(total).cpu()
        id_t, id_s, id_r = torch.chunk(embeddings, chunks=3, dim=0)

        left, right = batch_idx * self.batch_size, (batch_idx + 1) * self.batch_size
        right = min(self.dataset_len, right)
        self.id_t[left:right] = id_t.numpy()
        self.id_s[left:right] = id_s.numpy()
        self.id_r[left:right] = id_r.numpy()

    def _record_hopenet_batch(self, i_t, i_s, i_r, batch_idx):
        total = torch.cat((i_t, i_s, i_r), dim=0)  # in [-1,1]
        total = self.hopenet_norm(total)  # (B*3,C,H,W), in [vgg_min,vgg_max]
        yaw, pitch, roll = self.hopenet_model(total)  # each is (B*3,66)

        # param = (yaw + pitch + roll).cpu()  # to (B*3,66)
        # hope_t, hope_s, hope_r = torch.chunk(param, chunks=3, dim=0)  # each is (B,66)

        param = torch.cat((yaw, pitch, roll), dim=-1).cpu()  # to (B*3,198)
        hope_t, hope_s, hope_r = torch.chunk(param, chunks=3, dim=0)  # each is (B,198)

        left, right = batch_idx * self.batch_size, (batch_idx + 1) * self.batch_size
        right = min(self.dataset_len, right)
        self.hope_t[left:right] = hope_t.numpy()
        self.hope_s[left:right] = hope_s.numpy()
        self.hope_r[left:right] = hope_r.numpy()

    def _record_deep3d_batch(self, i_t, i_s, i_r, batch_idx):
        left, right = batch_idx * self.batch_size, (batch_idx + 1) * self.batch_size
        right = min(self.dataset_len, right)
        self.deep3d_t[left:right] = self._get_deep3d_exp_for_one_type(i_t)
        self.deep3d_s[left:right] = self._get_deep3d_exp_for_one_type(i_s)
        self.deep3d_r[left:right] = self._get_deep3d_exp_for_one_type(i_r)

    def _get_deep3d_exp_for_one_type(self, i_x):
        array_x: np.ndarray = ((i_x + 1.) * 127.5).permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
        pil_x = Image.fromarray(array_x[0])
        param = self.deep3d_model.infer_image(img_pil=pil_x)
        return param['exp'].cpu().numpy()

    @torch.no_grad()
    def inference(self):
        step = 0
        for batch in tqdm(self.dataloader, desc='inferring'):
            i_t = batch["s"].cuda()  # pose, exp
            i_s = batch["t"].cuda()  # id
            i_r = batch["ts"].cuda()

            self._record_id_batch(i_t, i_s, i_r, step)
            self._record_hopenet_batch(i_t, i_s, i_r, step)
            self._record_deep3d_batch(i_t, i_s, i_r, step)
            step += 1

    def calc(self):
        self._eval_id()
        self._eval_hopenet()
        self._eval_deep3d()

    def _eval_id(self):
        # if not self.en_id_eval:
        #     return

        embedding_target = self.id_t  # (dataset_len,512)
        embedding_source = self.id_s
        embedding_result = self.id_r
        embedding_source_no_repeat = self.id_s[::10]  # (dataset_len//10,512)

        ''' calculate id retrieval '''
        idx_source_gt = np.arange(self.dataset_len)  # (dataset_len,), e.g. [0,1,2,3,...,dataset_len-1]
        # dists = pairwise_distances(embedding_result, embedding_source_no_repeat, metric='cosine')  # (N,N//10)
        dists = pairwise_distances(embedding_result, embedding_source, metric='cosine')  # (dataset_len,dataset_len)
        idx_source_pred = dists.argmin(axis=1)  # (dataset_len,)
        cos_sims = (1 - dists)

        diff = np.zeros_like(idx_source_pred)
        ones = np.ones_like(idx_source_pred)
        diff[idx_source_pred != idx_source_gt] = ones[idx_source_pred != idx_source_gt]
        acc = 1. - diff.sum() / diff.shape[0]
        print('id retrieval acc = %.2f %%, cosine_sim = %.4f' % (
            acc * 100.,
            cos_sims[idx_source_gt, idx_source_gt].mean()))

    def _eval_hopenet(self):
        # if not self.en_hopenet_eval:
        #     return

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
        exp_t = torch.tensor(self.deep3d_t)
        exp_s = torch.tensor(self.deep3d_s)
        exp_r = torch.tensor(self.deep3d_r)

        exp_dim = exp_r.shape[-1]
        mse_sum = torch.nn.functional.mse_loss(exp_t, exp_r, reduction='mean')
        l2 = torch.sqrt(mse_sum * exp_dim)
        print('deep3d MSE_SUM=%.3f, L2=%.3f, dim=%d' % (mse_sum.data, l2.data, exp_dim))

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


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=int, default=0, help='0:synthesizing; 1:training')
    parser.add_argument('--step', type=int, default=-1, help='0:reenact; 1:mult-band; 2:reshaping')
    parser.add_argument('--copy', action='store_true', default=False, help='copy or not')
    args = parser.parse_args()

    ''' (1) save to folder '''
    if args.copy and args.step >= 0:
        save_intermediate_triplet_to_folder(max_cnt=2600, triplet_step=args.step)
    #     save_triplet_to_folder()

    ''' (2) calc fid '''
    # calc_fid(stage=args.stage, step=args.step)

    ''' (3) calc id, pose, exp '''
    if args.stage == 0:
        folder = '/gavin/datasets/triplet_lia_demo'
        if args.step >= 0:
            folder = '/gavin/datasets/triplet_lia_intermediate_demo'
            folder = os.path.join(folder, 'step_%d' % args.step)
    else:
        folder = '/gavin/code/FaceSwapping/inference/ffplus/demo_triplet10w_38/'
    calculator = CalcIdPoseExp(demo_folder=folder)
    calculator.inference()
    calculator.calc()

