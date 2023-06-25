import os
import time
from tqdm import tqdm
import pickle


def gen_celebahq(in_root: str = '/gavin/datasets/celeba_hq/data256x256/',
                 out_path: str = '/gavin/datasets/celeba_hq/256x256.pickle',
                 ):
    dataset_list = os.listdir(in_root)
    dataset_list.sort()

    print('Generating CelebA-HQ pickle list...(saved to %s)' % out_path)
    list_to = []
    for img_name in tqdm(dataset_list):
        list_to.append(os.path.join(in_root, img_name))
    pickle.dump(list_to, open(out_path, 'wb+'))

    print('Checking pickle list...')
    with open(out_path, 'rb') as f_from:
        p_from = pickle.load(f_from)
    print('pickle type:', type(p_from), ', total len:', len(p_from))
    print('pickle[0]:', p_from[0])
    print('pickle[1]:', p_from[1])
    print('pickle[-2]:', p_from[-2])
    print('pickle[-1]:', p_from[-1])
    print('-' * 50)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str,
                        default='/gavin/datasets/celeba_hq/data256x256/',
                        help="Input path.")
    parser.add_argument("-o", "--out", type=str,
                        default='/gavin/datasets/celeba_hq/256x256.pickle',
                        help="Saved pickle file name.")
    args = parser.parse_args()

    ''' generate ffplus full pickle '''
    gen_celebahq(
        in_root=args.input,
        out_path=args.out,
    )
