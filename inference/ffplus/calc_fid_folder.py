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


if __name__ == "__main__":
    folder1 = '/gavin/code/FaceSwapping/inference/ffplus/demo_triplet10w_38/source'
    # folder2 = '/gavin/datasets/stylegan/stylegan3-r-ffhq-1024x1024'
    folder2 = os.path.join('/gavin/code/TextualInversion/exp_eval/db',
                           'all')

    val = fid_score.calculate_fid_given_paths([folder1, folder2], batch_size=16, device=0, dims=2048, num_workers=4)
    print('FID = %.2f' % val)
