"""
name        RS?     aligned?    Used  | folder                    | pickle  IQA
------------------------------------------------------------------------------------
VGGFace2    no      no          yes     original/ft_local/cele
celebasian  no      no          yes     original/ft_local/cele
------------------------------------------------------------------------------------
VGGFace2    yes     now         yes     hd                          1
celebasian  yes     now         yes     hd                          1
------------------------------------------------------------------------------------
VGGFace2    yes     ffhq        yes     original/hd_align_512       2       yes
celebasian  yes     ffhq        yes     original/hd_align_512       2       yes
------------------------------------------------------------------------------------
FFHQ        no                  ?       original/hd_align_512       2       yes
celebahq    no                  no


====================================================================================
1. ~/datasets/hd.pickle
folder: ~/datasets/hd
including:
    - VGGFace2 (RS, align?)
    - celebasian (RS, align?)

2. ~/datasets/original/image_512_quality.pickle
folder: ~/datasets/original/hd_align_512
including:
    - VGGFace2 (RS, ffhq align)
    - celebasian (RS, ffhq align)

"""

import pickle
import os

hd_file = 'hd.pickle'  # file1
sr512_file = 'original/image_512_quality.pickle'  # file2

sr512_as_hd_file = 'original/image_512_quality_as_hd.pickle'

hanbang_root = '/apdcephfs/share_1290939/ahbanliang/datasets'  # root path used in pickle
hanbang_hd = os.path.join('/gavin/datasets/hanbang', hd_file)
hanbang_512 = os.path.join('/gavin/datasets/hanbang', sr512_file)

yuange_root = '/gavin/datasets'  # root path used in pickle
yuange_hd = os.path.join(yuange_root, hd_file)
yuange_512 = os.path.join(yuange_root, sr512_file)


def copy_hd():
    print('Running copy_hd()')
    with open(hanbang_hd, 'rb') as f_from:
        p_from = pickle.load(f_from)
    print('from type:', type(p_from))  # dict
    print('from key:', list(p_from.keys())[0])
    print('from value', list(p_from.values())[0][1])

    dict_to = {}
    for key, val in p_from.items():  # key: str, val: list
        tmp_val = []
        for img in val:
            img = yuange_root + '/hd_xvf' + img[len(hanbang_root):]
            tmp_val.append(img)
        dict_to[key] = tmp_val

    print('to key:', list(dict_to.keys())[0])
    print('to value:', list(dict_to.values())[0][1])
    print('file exists? -', os.path.exists(list(dict_to.values())[0][1]))

    print('pickle dumping...')
    pickle.dump(dict_to, open(yuange_hd, 'wb+'))
    print('copy hd finished.')
    print('---------------------------------------------')


def copy_512():
    print('Running copy_512()')

    with open(hanbang_512, 'rb') as f_from:
        p_from = pickle.load(f_from)

    print('from type:', type(p_from))  # list
    print('from item:', p_from[0])  # list

    list_to = []
    for item in p_from:
        img, iqa = item[0], item[1]
        img = yuange_root + img[len(hanbang_root):]
        list_to.append([img, iqa])

    print('to type:', type(list_to))
    print('to item:', list_to[0])
    print('file exists? -', os.path.exists(list_to[0][0]))

    print('pickle dumping...')
    pickle.dump(list_to, open(yuange_512, 'wb+'))
    print('copy 512 finished.')
    print('---------------------------------------------')


if __name__ == '__main__':

    copy_hd()
    copy_512()