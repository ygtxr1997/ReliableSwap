import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

def iterate_folder(in_folder: str,
                   out_folder: str,
                   col_names: list,
                   save_idxs: list,
                   shuffle: bool = True,
                   ):
    input_list = os.listdir(in_folder)
    input_list.sort()
    if os.path.exists(out_folder):
        os.system('rm -r %s' % out_folder)
    os.mkdir(out_folder)
    question_folder = out_folder.replace('split', 'question')
    if os.path.exists(question_folder):
        os.system('rm -r %s' % question_folder)
    os.mkdir(question_folder)

    split_cnt = len(col_names)

    for idx in range(len(input_list)):
        input_name = input_list[idx]
        if save_idxs is not None and idx not in save_idxs:
            continue
        print(idx, input_name)
        input_path = os.path.join(in_folder, input_name)
        one_row = Image.open(input_path)

        w, h = one_row.size
        w_split = w // split_cnt
        h_split = h

        shuffle_idx = np.arange(split_cnt)
        if shuffle:
            np.random.shuffle(shuffle_idx[2:])  # target and source keep in order

        w1, h1, w2, h2 = 0, 0, w_split, h_split
        img_split_list = [0] * split_cnt
        for col in range(split_cnt):
            img_split = one_row.crop((w1, h1, w2, h2))

            img_name = input_name.split('.')[0] + ('_%d_%s.jpg' % (shuffle_idx[col], col_names[col]))
            img_split.save(os.path.join(out_folder, img_name))

            img_split_list[shuffle_idx[col]] = img_split

            w1 += w_split
            w2 += w_split

        text_h = 40
        row_shuffle = Image.new("RGB", (w_split * 3, h_split * 3 + text_h * 3), color=(255, 255, 255))
        draw = ImageDraw.Draw(row_shuffle)
        fontStyle = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 24, encoding="utf-8")
        for col in range(split_cnt):
            if col <= 1:  # target, source
                w_pos = w_split * col
                h_pos = 0
            else:
                w_pos = w_split * ((col - 2) % 3)
                h_pos = (h_split + text_h) * (1 + (col - 2) // 3)
            img_split = img_split_list[col]
            row_shuffle.paste(img_split, (w_pos, h_pos))
        draw.text((256 * 0 + 80, 256 * 1 + 5), 'target', (0, 0, 0), font=fontStyle)
        draw.text((256 * 1 + 80, 256 * 1 + 5), 'source', (0, 0, 0), font=fontStyle)
        draw.text((256 * 0 + 112, 256 * 2 + 5 + text_h * 1), 'A', (0, 0, 0), font=fontStyle)
        draw.text((256 * 1 + 112, 256 * 2 + 5 + text_h * 1), 'B', (0, 0, 0), font=fontStyle)
        draw.text((256 * 2 + 112, 256 * 2 + 5 + text_h * 1), 'C', (0, 0, 0), font=fontStyle)
        draw.text((256 * 0 + 112, 256 * 3 + 5 + text_h * 2), 'D', (0, 0, 0), font=fontStyle)
        draw.text((256 * 1 + 112, 256 * 3 + 5 + text_h * 2), 'E', (0, 0, 0), font=fontStyle)
        question_name = input_name
        row_shuffle.save(os.path.join(question_folder, question_name))

if __name__ == '__main__':
    ori_folder = 'celebahq1'
    split_folder = 'celebahq1_split'
    good_list = [
        0, 100, 200, 300,
        3, 10, 37, 44, 57, 114, 139, 162, 173, 192,
        207, 224, 235, 241, 252, 256, 308, 312,
        320, 325, 336, 344, 377, 383, 391,
        403, 416, 453, 469, 483, 527,
        532, 538, 542, 590,
        639, 688, 718, 739, 740, 747, 765, 782,
        806, 818, 819, 830, 860, 913, 965,
        1006, 1008, 1033, 1047, 1048, 1057, 1058, 1067, 1069, 1070, 1079, 1084, 1086, 1092,
        1122, 1123, 1126, 1154, 1178, 1185,
        1227, 1232, 1250, 1257, 1262, 1266, 1290, 1297, 1316, 1334, 1338, 1317, 1356, 1357,
        1933, 1973, 2011, 2013, 2040,
        2043, 2070, 2086, 2088, 2121, 2127, 2143, 2193, 2198, 2211,
        2221, 2222, 2224, 2248, 2250, 2261, 2275, 2287, 2291, 2309,
        2327, 2332, 2346, 2369, 2371, 2387, 2408, 2425, 2427, 2428,
        2439, 2450, 2463, 2472, 2475, 2483, 2506, 2512, 2526, 2558,
        2559, 2563, 2566, 2567, 2575, 2590, 2594, 2596, 2610, 2617,
        2621, 2632, 2639, 2645, 2650, 2663, 2666, 2684, 2693, 2718,
        2724, 2733, 2734, 2735, 2748, 2753, 2755, 2781, 2787, 2791,
    ]
    good_list = None  # None means all
    iterate_folder(ori_folder, split_folder,
                   col_names=['target', 'source', 'hires', 'megafs', 'infoswap', 'simswap', 'faceshifter', 'ours'],
                   save_idxs=good_list
                   )
