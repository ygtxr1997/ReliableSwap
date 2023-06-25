# Author: Jan Niklas Kolf, 2020
import os

import torch.cuda
from torchvision.transforms import transforms
import torch.nn.functional as F

from face_image_quality import SER_FIQ
from modules.third_party.arcface import iresnet100
from modules.third_party.celeba.model import GlassesDetectorPL

import cv2
import numpy as np
from tqdm import tqdm
import pickle
import kornia


class TripletWalker(object):
    def __init__(self,
                 source_folder: str,
                 source_pickle: str = None,
                 save_good: str = None,
                 lo: int = 0,
                 hi: int = 100000,
                 gpu: int = 0,
                 use_iqa: bool = True,
                 use_fid: bool = True,
                 use_glasses: bool = True,
                 threshold_iqa: float = 0.2,
                 threshold_fid: float = 0.6,
                 threshold_glasses: float = 0.9999,
                 use_snapshot: bool = False,
                 ):
        super(TripletWalker, self).__init__()

        ''' source_folder: /gavin/datasets/triplet
            source_pickle: /gavin/datasets/triplet_x_x.pickle
        '''
        self.root_path = source_folder
        self.cached_pickle = source_pickle
        self.save_good = save_good

        self.bad_folder = './bad'
        self.good_folder = './good'
        self.use_snapshot = use_snapshot
        if self.use_snapshot:
            if os.path.exists(self.bad_folder):
                os.system('rm -r %s' % self.bad_folder)
            os.mkdir(self.bad_folder)
            if os.path.exists(self.good_folder):
                os.system('rm -r %s' % self.good_folder)
            os.mkdir(self.good_folder)

        self.lo = lo
        self.hi = hi
        self.gpu = gpu
        self.use_iqa = use_iqa
        self.use_fid = use_fid
        self.use_glasses = use_glasses
        self.threshold_iqa = threshold_iqa
        self.threshold_fid = threshold_fid
        self.threshold_glasses = threshold_glasses

        ''' Create the SER-FIQ Model
            Choose the GPU, default is 0. '''
        self.ser_fiq = SER_FIQ(gpu=gpu)
        print('FIQ model loaded. (used: %s)' % use_iqa)

        ''' Face Recognition Model '''
        self.f_id = iresnet100(pretrained=False, fp16=False).cuda(device=gpu)
        f_id_checkpoint_path = "/gavin/code/FaceShifter/faceswap/faceswap/" \
                               "checkpoints/face_id/ms1mv3_arcface_r100_fp16_backbone.pth"
        self.f_id.load_state_dict(torch.load(f_id_checkpoint_path, map_location="cpu"))
        self.f_id.eval()
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        trans_matrix = torch.tensor([[[1.07695457, -0.03625215, -1.56352194],
                                      [0.03625215, 1.07695457, -5.32134629]]],
                                    requires_grad=False).float()  # shape:(1,2,3)
        self.M = trans_matrix.repeat(1, 1, 1).cuda(device=gpu)
        print('Face recognition model loaded. (used: %s)' % use_fid)

        ''' Glasses Occlusion Detector '''
        self.glasses_detector = GlassesDetectorPL.load_from_checkpoint(
            checkpoint_path='../celeba/out/tmp/epoch=25-step=16499.ckpt',
        ).cuda(device=gpu).eval()
        print('Glasses detector model loaded. (used: %s)' % use_glasses)

        self.score_dict = {}
        self.delete_list = []
        self.keep_list = []

    def iterate_folder(self, ):
        """ """
        ''' 1. get dataset folders to be filtered '''
        group_folders = self._step1_get_group_folders()

        ''' 2. iterate folders and calculate scores '''
        self._step2_iterate_folders(group_folders)

        ''' 3. filter according scores '''
        self._step3_filter_folders()

        ''' 4. save to pickle '''
        self._step4_save_good_to_pickle()

    def _step1_get_group_folders(self):
        if self.cached_pickle is None:  # from '/gavin/datasets/triplet'
            print('Getting group folder list from %s...' % self.root_path)
            group_folders = os.listdir(self.root_path)

            pair_list = []

            print('Getting st_mtime of dataset folders...')
            for target_source_folder in tqdm(group_folders,
                                             desc='[gpu %d] Getting st_mtime' % self.gpu,
                                             position=self.gpu):
                st_mtime = os.path.getmtime(os.path.join(self.root_path, target_source_folder))
                pair = (target_source_folder, st_mtime)
                pair_list.append(pair)

            print('Sorting...')
            pair_list.sort(key=lambda x: x[1])
            pair_list = pair_list[self.lo: self.hi]

            group_folders = []
            for pair in pair_list:
                target_source_folder, st_mtime = pair
                group_folders.append(target_source_folder)

        else:  # from '/gavin/datasets/triplet_x_x.pickle'
            print('Use cached pickle %s' % self.cached_pickle)
            with open(self.cached_pickle, 'rb') as handle:
                dataset_list = pickle.load(handle)
            ''' Format, list:
                ['xxx/triplet/00000000_00008210',
                 'xxx/triplet/00000044_00008316',]
            '''
            group_folders = []
            for line in dataset_list:
                target_source_folder = os.path.join(line, 'nothing').split('/')[-2]  # '/xxx/xxx/yy_zz' to 'yy_zz'
                group_folders.append(target_source_folder)
            group_folders = group_folders[self.lo: self.hi]

        return group_folders

    def _step2_iterate_folders(self, group_folders: list):
        print('Iterating folders:')
        idx = 0
        for pair_strs in tqdm(group_folders,
                              desc='gpu %d' % self.gpu,
                              position=self.gpu):
            path_s = os.path.join(self.root_path, pair_strs, 'source.jpg')
            path_t = os.path.join(self.root_path, pair_strs, 'target.jpg')
            path_st = os.path.join(self.root_path, pair_strs, 'output_st.jpg')
            path_ts = os.path.join(self.root_path, pair_strs, 'output_ts.jpg')
            img_s = cv2.imread(path_s)
            img_t = cv2.imread(path_t)
            img_st = cv2.imread(path_st)
            img_ts = cv2.imread(path_ts)

            folder_score_dict = {
                "iqa": [1., 1., 1., 1.],
                "id_dis": [0., 0.],
                "glasses_score": [0., 0.],
            }

            if self.use_iqa:
                score_s = self._calc_fiq(img_s)
                score_t = self._calc_fiq(img_t)
                score_st = self._calc_fiq(img_st)
                score_ts = self._calc_fiq(img_ts)
                folder_score_dict["iqa"] = [score_s, score_t, score_st, score_ts]

            if self.use_fid:
                id_dis_s = self._calc_id_dis(img_s, img_st)
                id_dis_t = self._calc_id_dis(img_t, img_ts)
                folder_score_dict["id_dis"] = [id_dis_s, id_dis_t]

            if self.use_glasses:
                glasses_score_s = self._calc_glasses_score(img_s)
                glasses_score_t = self._calc_glasses_score(img_t)
                folder_score_dict["glasses_score"] = [glasses_score_s, glasses_score_t]

            self.score_dict[pair_strs] = folder_score_dict
            idx += 1

    def _calc_fiq(self, cv2_image: np.ndarray):
        # aligned_img = self.ser_fiq.apply_mtcnn(img)  # (3,112,112)
        # print(aligned_img.shape)
        aligned_img = cv2.resize(cv2_image, (112, 112))

        # Calculate the quality score of the image
        # T=100 (default) is a good choice
        # Alpha and r parameters can be used to scale your
        # score distribution.
        score = self.ser_fiq.get_score(aligned_img, T=100)
        return float(score)

    def _calc_id_dis(self, cv2_image1: np.ndarray, cv2_image2: np.ndarray):
        rgb1 = cv2.cvtColor(cv2_image1, cv2.COLOR_BGR2RGB)
        rgb2 = cv2.cvtColor(cv2_image2, cv2.COLOR_BGR2RGB)
        tensor1 = self.norm(rgb1).unsqueeze(0).cuda(device=self.gpu)  # (1,RGB,H,W)
        tensor2 = self.norm(rgb2).unsqueeze(0).cuda(device=self.gpu)  # (1,RGB,H,W)
        tensor1 = kornia.geometry.transform.warp_affine(tensor1, self.M, (256, 256), align_corners=True)
        tensor2 = kornia.geometry.transform.warp_affine(tensor2, self.M, (256, 256), align_corners=True)
        embedding1 = F.normalize(
            self.f_id(F.interpolate(tensor1, size=112, mode="bilinear", align_corners=True)), dim=-1, p=2
        )
        embedding2 = F.normalize(
            self.f_id(F.interpolate(tensor2, size=112, mode="bilinear", align_corners=True)), dim=-1, p=2
        )
        id_dis = (1 - F.cosine_similarity(embedding1, embedding2, 1)).mean()
        return float(id_dis)

    def _calc_glasses_score(self, cv2_image: np.ndarray):
        rgb = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        x_in = self.norm(rgb).unsqueeze(0).cuda(device=self.gpu)  # (1,RGB,H,W)
        y_out = self.glasses_detector(x_in)
        glasses_score = y_out[0][1]  # (B,2)
        return float(glasses_score)

    def load_cached_score_dict(self, cached_score_dicts: list):
        for cached_pickle in cached_score_dicts:
            with open(cached_pickle, 'rb') as handle:
                cached_score_dict: dict = pickle.load(handle)
            for pair_strs, scores in cached_score_dict.items():
                self.score_dict[pair_strs] = scores
            print('Cached pickle %s loaded, cnt=%d.' % (cached_pickle, len(cached_score_dict)))
        self._step3_filter_folders()
        self._step4_save_good_to_pickle()

    def _step3_filter_folders(self, ):
        num = len(self.score_dict)
        print('Filtering folders: (total: %d)' % num)
        arr_iqa = np.zeros(num * 2, dtype=np.float)
        arr_fid = np.zeros(num * 2, dtype=np.float)
        arr_glasses = np.zeros(num * 2, dtype=np.float)

        idx = 0
        for pair_strs, scores in self.score_dict.items():
            iqa_s, iqa_t, iqa_st, iqa_ts = scores["iqa"]
            id_dis_s, id_dis_t = scores["id_dis"]
            glasses_s, glasses_t = scores["glasses_score"]

            iqa_diff_s = iqa_s - iqa_ts
            iqa_diff_t = iqa_t - iqa_st

            arr_iqa[idx * 2] = iqa_diff_s
            arr_iqa[idx * 2 + 1] = iqa_diff_t
            arr_fid[idx * 2] = id_dis_s
            arr_fid[idx * 2 + 1] = id_dis_t
            arr_glasses[idx * 2] = glasses_s
            arr_glasses[idx * 2 + 1] = glasses_t

            if iqa_diff_s >= self.threshold_iqa or iqa_diff_t >= self.threshold_iqa:  # iqa
                reason = 'iqa_%.4f_%.4f_or_%.4f_%.4f' % (
                    iqa_diff_s, self.threshold_iqa, iqa_diff_t, self.threshold_iqa
                )
                self._delete_folder(os.path.join(self.root_path, pair_strs), reason)
            elif id_dis_s >= self.threshold_fid or id_dis_t >= self.threshold_fid:  # fid
                reason = 'fid_%.4f_%.4f_or_%.4f_%.4f' % (
                    id_dis_s, self.threshold_fid, id_dis_t, self.threshold_fid
                )
                self._delete_folder(os.path.join(self.root_path, pair_strs), reason)
            elif glasses_s >= self.threshold_glasses or glasses_t >= self.threshold_glasses:  # glasses
                reason = 'glasses_%.4f_%.4f_or_%.4f_%.4f' % (
                    glasses_s, self.threshold_glasses, glasses_t, self.threshold_glasses
                )
                self._delete_folder(os.path.join(self.root_path, pair_strs), reason)
            else:  # good folders passing all filtering rules
                reason = 'good_%s' % pair_strs
                self._keep_folder(os.path.join(self.root_path, pair_strs), reason)

            idx += 1

        np.save('arr_iqa_gpu%d.npy' % self.gpu, arr_iqa)
        np.save('arr_fid_gpu%d.npy' % self.gpu, arr_fid)
        np.save('arr_glasses_gpu%d.npy' % self.gpu, arr_glasses)

        self._stat_interval(arr_iqa, title='iqa')
        self._stat_interval(arr_fid, title='fid', left=0.)
        self._stat_interval(arr_glasses, title='glasses', left=0.)

        pickle_name = 'score_dict_%d_%d.pickle' % (self.lo, self.hi)
        pickle.dump(self.score_dict, open(pickle_name, 'wb+'))
        keep_name = 'keep_%d_%d.pickle' % (self.lo, self.hi)
        pickle.dump(self.keep_list, open(keep_name, 'wb+'))
        delete_name = 'delete_%d_%d.pickle' % (self.lo, self.hi)
        pickle.dump(self.delete_list, open(delete_name, 'wb+'))
        print('[gpu:%d] filter finished, keep=%d, delete=%d, all=%d' % (
            self.gpu, len(self.keep_list), len(self.delete_list), len(self.score_dict)))

    def _step4_save_good_to_pickle(self):
        if self.save_good is not None:
            abs_good_list = []
            for good_folder in self.keep_list:
                abs_good_list.append(os.path.join(self.root_path, good_folder))
            pickle.dump(abs_good_list, open(self.save_good, 'ab+'))

    @staticmethod
    def _stat_interval(arr_record: np.ndarray, title: str,
                       left: float = -1.0, right: float = 1.0,
                       interval_num: int = 20,
                       ):  # (num,)
        print(('*' * 20) + title + ('*' * 20))
        step = (right - left) / interval_num

        stat = np.zeros(interval_num, dtype=np.uint)
        stat_str = []
        for interval_idx in range(interval_num):
            lo = left + step * interval_idx
            hi = lo + step
            for val in arr_record:
                if lo <= val < hi - 1e-6:
                    stat[interval_idx] += 1
            stat_str.append('[%.2f ~ %.2f]' % (lo, hi))

        for interval_idx in range(interval_num):
            print('%s : %.3f%%' % (stat_str[interval_idx], stat[interval_idx] / arr_record.shape[0] * 100.))

    def _delete_folder(self, abs_folder: str, reason: str):
        self.delete_list.append(abs_folder)
        if self.use_snapshot:
            print('delete folder %s. (%s)' % (abs_folder, reason))
            os.system('cp -r %s %s' % (abs_folder, os.path.join(self.bad_folder, reason)))

    def _keep_folder(self, abs_folder: str, reason: str):
        self.keep_list.append(abs_folder)
        if self.use_snapshot:
            os.system('cp -r %s %s' % (abs_folder, os.path.join(self.good_folder, reason)))


def one_task(args, lo, hi, gpu):
    print(os.getpid(), '[%d,%d] on gpu %d' % (lo, hi, gpu))
    walker = TripletWalker(
        source_folder=args.target_folder,
        source_pickle=args.target_pickle,
        save_good=args.save_good,
        lo=lo,
        hi=hi,
        gpu=gpu,
        use_iqa=args.iqa,
        use_fid=args.fid,
        use_glasses=args.fid,
        use_snapshot=False,
    )
    walker.iterate_folder()


def triplet_walker(args):
    """ """
    ''' python3 triplet_walker.py --target_folder /gavin/datasets/triplet_lia \
        --save-good /gavin/datasets/triplet_lia_0_100000.pickle
    '''
    import multiprocessing
    multiprocessing.set_start_method('spawn')
    gpu_cnt = torch.cuda.device_count()
    step = (args.hi - args.lo) // gpu_cnt

    input('Press Enter to delete old %s' % args.save_good)
    os.system('rm %s' % args.save_good)

    process_list = []
    for gpu_id in range(gpu_cnt):
        low = gpu_id * step
        high = (gpu_id + 1) * step
        process = multiprocessing.Process(target=one_task, args=(args, low, high, gpu_id,))
        process_list.append(process)
    for process in process_list:
        process.start()


def demo_run(args):
    """ """
    ''' python3 triplet_walker.py --target_folder /gavin/datasets/triplet_lia --demo yes \
        --save-good /gavin/datasets/triplet_lia_tmp.pickle
    '''
    print('---------- demo run ----------')
    walker = TripletWalker(
        source_folder=args.target_folder,
        source_pickle=args.target_pickle,
        save_good=args.save_good,
        lo=0,
        hi=100,
        gpu=0,
        use_iqa=True,
        use_fid=True,
        use_glasses=True,
        use_snapshot=True,
    )
    walker.iterate_folder()
    # score_dicts = [
    #     'score_dict_0_25000.pickle',
    #     'score_dict_25000_50000.pickle',
    #     'score_dict_50000_75000.pickle',
    #     'score_dict_75000_100000.pickle',
    # ]
    # walker.load_cached_score_dict(score_dicts)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_folder', type=str, help='/gavin/datasets')
    parser.add_argument('--target_pickle', type=str, default=None, help='/gavin/datasets/triplet_0_100000.pickle')
    parser.add_argument('--save-good', type=str, help='/gavin/datasets/triplet_0_100000_fid_glasses.pickle')

    parser.add_argument('--lo', type=int, default=0, help='left range')
    parser.add_argument('--hi', type=int, default=-1, help='right range')

    parser.add_argument('--iqa', type=str, default='yes', help='iqa')
    parser.add_argument('--fid', type=str, default='yes', help='fid')
    parser.add_argument('--glasses', type=str, default='yes', help='glasses')

    parser.add_argument('--demo', type=str, default='no', help='demo run or real task')
    parser.add_argument('--check', type=str, default='no', help='check generated pickle')
    args = parser.parse_args()

    # directly use argparse bool type may raise bug
    args.iqa = True if args.iqa == 'yes' else False
    args.fid = True if args.fid == 'yes' else False
    args.glasses = True if args.glasses == 'yes' else False
    args.demo = True if args.demo == 'yes' else False
    args.check = True if args.check == 'yes' else False

    if args.check:
        print('Concat pickle records in %s' % args.save_good)
        check_triplet_list = []
        with open(args.save_good, "rb") as handle:
            while True:
                try:
                    l_node = pickle.load(handle)
                    check_triplet_list.extend(l_node)
                except EOFError:
                    break
        pickle.dump(check_triplet_list, open(args.save_good, 'wb+'))

        print('Concat finished. Checking saved pickle %s' % args.save_good)
        with open(args.save_good, "rb") as handle:
            check_triplet_list = pickle.load(handle)
        print('abs_good_list[0]=%s' % check_triplet_list[0] if len(check_triplet_list) > 0 else None,
              'total=%d' % len(check_triplet_list))
        exit(0)

    if args.demo:
        demo_run(args)
        exit(0)
    else:
        triplet_walker(args)
