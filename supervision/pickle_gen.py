import os
import time
from tqdm import tqdm
import pickle
import numpy as np


def stat_interval(diff: np.ndarray, interval_num: int = 50):
    left, right = np.min(diff), np.max(diff)
    step = (right - left) / interval_num

    stat = np.zeros(interval_num, dtype=np.uint)
    stat_str = []
    for interval_idx in range(interval_num):
        lo = left + step * interval_idx
        hi = lo + step
        for val in diff:
            if lo <= val < hi:
                stat[interval_idx] += 1
        stat_str.append('[%.2f ~ %.2f]' % (lo, hi))

    print('*' * 50)
    for interval_idx in range(interval_num):
        print('%s : %.3f%%' % (stat_str[interval_idx], stat[interval_idx] / diff.shape[0] * 100.))


def folder_str_to_idx(target_source_folder: str):
    str_t, str_s = target_source_folder.split('_')
    idx_t = int(str_t)
    idx_s = int(str_s)
    return idx_t, idx_s


def check_pickle(pickle_file: str):
    print('Checking pickle list [%s]...' % pickle_file)
    with open(pickle_file, 'rb') as f_from:
        p_from = pickle.load(f_from)
    print('pickle type:', type(p_from), ', total len:', len(p_from))
    print('pickle[0]:', p_from[0])
    print('pickle[1]:', p_from[1])
    print('pickle[2]:', p_from[2])
    print('pickle[-1]:', p_from[-1])


def insert_prefix(pickle_file: str, prefix: str = '/gavin/datasets/triplet'):
    with open(pickle_file, "rb") as handle:
        folder_list = pickle.load(handle)
    list_to = []
    for folder in folder_list:
        list_to.append(os.path.join(prefix, folder))
    pickle.dump(list_to, open(pickle_file, 'wb+'))


def gen_triplet(in_root: str = '/gavin/datasets/triplet',
                out_folder: str = '/gavin/datasets/',
                lo: int = 0,
                hi: int = 100000,
                split: int = 8,
                ):
    assert (hi - lo) % split == 0 and hi > lo
    dataset_list = os.listdir(in_root)
    pair_list = []

    print('Getting st_mtime of dataset folders...(about 0.002s/it)')
    for target_source_folder in tqdm(dataset_list):
        st_mtime = os.path.getmtime(os.path.join(in_root, target_source_folder))
        pair = (target_source_folder, st_mtime)
        pair_list.append(pair)

    print('Sorting...')
    pair_list.sort(key=lambda x: x[1])
    pair_list = pair_list[lo:hi]

    print('Generating pickle list...')
    list_all = []  # sorted folder list
    dict_all = {}  # not used
    list_target = []  # for stat
    for pair in pair_list:
        target_source_folder, st_mtime = pair
        list_all.append(os.path.join(in_root, target_source_folder))
        idx_t, idx_s = folder_str_to_idx(target_source_folder)
        if dict_all.get(idx_t) is not None:
            dict_all[idx_t].append(target_source_folder)
        else:
            dict_all[idx_t] = [target_source_folder]
        list_target.append(idx_t)
    stat_interval(np.array(list_target, dtype=np.uint32))

    part_cnt = (hi - lo) // split
    right = part_cnt
    for _ in range(split):
        out_name = 'triplet_%d_%d.pickle' % (0, right)
        out_path = os.path.join(out_folder, out_name)

        print('Dumping pickle...(out to [%s])' % out_path)
        list_to = list_all[0: right]
        pickle.dump(list_to, open(out_path, 'wb+'))
        right += part_cnt

        check_pickle(out_path)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--lo", type=int, default=0, help="Lower bound of dataset range.")
    parser.add_argument("--hi", type=int, default=-1, help="Higher bound of dataset range.")
    parser.add_argument("--split", type=int, default=8, help="Split into k parts.")
    parser.add_argument("-o", "--out", type=str, required=True, help="Saved folder.")
    args = parser.parse_args()

    gen_triplet(
        out_folder=args.out,
        lo=args.lo,
        hi=args.hi,
        split=args.split,
    )
