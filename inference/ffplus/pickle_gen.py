import os
import time
from tqdm import tqdm
import pickle


def gen_ts_pairs(in_root: str = '/gavin/datasets/ff+/manipulated_sequences/FaceShifter/c23/videos',
                 out_path: str = '/gavin/datasets/ff+/original_sequences/youtube/c23/images.pickle',
                 ):
    video_list = os.listdir(in_root)
    video_list.sort()
    out_path = out_path.replace('images.pickle', 'ts_pairs.pickle')

    print('Generating ts_pairs pickle list...(saved to %s)' % out_path)
    list_to = []
    for video in tqdm(video_list):
        video = video[:-len('.mp4')]
        ts_pair = (int(video[:3]), int(video[-3:]))
        list_to.append(ts_pair)
    pickle.dump(list_to, open(out_path, 'wb+'))

    print('Checking pickle list...')
    with open(out_path, 'rb') as f_from:
        p_from = pickle.load(f_from)
    print('pickle type:', type(p_from), ', total len:', len(p_from))
    print('pickle[0][0]:', p_from[0][0])
    print('pickle[0][-1]:', p_from[0][-1])
    print('pickle[999][0]:', p_from[999][0])
    print('pickle[999][-1]:', p_from[999][-1])
    print('-' * 50)


def gen_ffplus(in_root: str = '/gavin/datasets/ff+/original_sequences/youtube/c23/images',
               out_path: str = '/gavin/datasets/ff+/original_sequences/youtube/c23/images.pickle',
               ):
    dataset_list = os.listdir(in_root)
    dataset_list.sort()

    print('Generating ffplus pickle list...(saved to %s)' % out_path)
    dict_to = {}
    for folder_id in tqdm(dataset_list):
        frame_list = os.listdir(os.path.join(in_root, folder_id))
        frame_list.sort()

        list_to = []
        for frame in frame_list:
            list_to.append(os.path.join(in_root, folder_id, frame))

        dict_to[int(folder_id)] = list_to

    pickle.dump(dict_to, open(out_path, 'wb+'))

    print('Checking pickle list...')
    with open(out_path, 'rb') as f_from:
        p_from = pickle.load(f_from)
    print('pickle type:', type(p_from), ', total len:', len(p_from))
    print('pickle[0][0]:', p_from[0][0])
    print('pickle[0][-1]:', p_from[0][-1])
    print('pickle[999][0]:', p_from[999][0])
    print('pickle[999][-1]:', p_from[999][-1])
    print('-' * 50)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str,
                        default='/gavin/datasets/ff+/original_sequences/youtube/c23/images',
                        help="Input path.")
    parser.add_argument("-o", "--out", type=str,
                        default='/gavin/datasets/ff+/original_sequences/youtube/c23/images.pickle',
                        help="Saved pickle file name.")
    args = parser.parse_args()

    ''' generate ffplus full pickle '''
    gen_ffplus(
        in_root=args.input,
        out_path=args.out,
    )

    # gen_ts_pairs(
    #     out_path=args.out,
    # )

    ''' generate aligned pickle '''
    # gen_ffplus(
    #     in_root='/gavin/datasets/ff+/aligned',
    #     out_path='/gavin/datasets/ff+/aligned.pickle',
    # )
