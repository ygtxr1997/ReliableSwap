import logging
import os


def shuffle_file(filename):
    """Shuffle lines in file.

    """
    sp = filename.split('/')
    shuffled_filename = '/'.join(sp[:-1] + ['shuffled_{}'.format(sp[-1])])
    print(shuffled_filename)
    os.system('shuf {} > {}'.format(filename, shuffled_filename))
    return shuffled_filename


def split_file(prefix, filename, split_num):
    """Split one file to split_num parts, return splited filenames.
    It could be viewed as a method of shuffling.

    """
    import random
    import string
    # each file will contain about file_lines lines
    file_lines = int(sum(1 for line in open(filename)) / split_num)
    dst_dir_hash = ''.join(random.choice(
        string.ascii_uppercase + string.digits) for _ in range(10))
    dst_dir = os.path.join(prefix, dst_dir_hash)
    os.system('rm {}/*'.format(dst_dir))
    os.system('mkdir -p {}'.format(dst_dir))
    os.system('split -l {} {} {}/'.format(file_lines, filename, dst_dir))
    return [os.path.join(dst_dir, f) for f in os.listdir(dst_dir)]


def compute_mean(sess, images, steps):
    """Compute channel-wise mean of dataset.

    """
    import numpy as np
    import tensorflow._api.v2.compat.v1 as tf
    from neuralgym.utils.logger import ProgressBar
    bar = ProgressBar()
    mean_list = []
    for i in range(steps):
        mean = sess.run(tf.reduce_mean(images, [0, 1, 2]))
        mean_list.append(mean)
        bar.progress(i/steps, 'Computing image means...')
    mean = np.mean(mean_list, 0)
    print('Image Mean: %s', mean)
    return mean
