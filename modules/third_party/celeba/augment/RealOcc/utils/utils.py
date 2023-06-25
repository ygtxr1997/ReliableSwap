import os
# Inspiration: https://github.com/honnibal/spacy-ray/pull/
# 1/files#diff-7ede881ddc3e8456b320afb958362b2aR12-R45
from asyncio import Event
from typing import Tuple
from time import sleep

import albumentations as A
import math
import numpy as np
import cv2
from .random_shape_generator import *
import random
import glob
import skimage
from tqdm import tqdm

# import ray
# # For typing purposes
# from ray.actor import ActorHandle
# from tqdm import tqdm

def validate_path(name,dir):
    if not os.path.exists(dir) or not os.path.isdir(dir):
        raise ValueError(f'The path for the directory "{dir}"" does not exist or is not a folder')
    files_name=os.listdir(dir)
    print(f'Total number of files in the path of {name}: {len(files_name)}')
    return files_name


def validate_img_mask_pair(images_name,maskDir):
    validated_images_name=[]
    for img in images_name:
        img_name=img.split(".")[0]
        if os.path.exists(maskDir+f"{img_name}.png"):
            validated_images_name.append(img)
        else:
            print(f'skipping img {img_name} ...')
            continue 
    return validated_images_name


def get_src_augmentor():
    """
    Face augmentor
    """
    aug=A.Compose([
            A.AdvancedBlur(),
            A.HorizontalFlip(p=0.5),              
            # A.OneOf([
            #     # A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
            #     A.RandomSunFlare (num_flare_circles_lower=3, num_flare_circles_upper=5,src_radius=200),
            #     A.RandomShadow(shadow_dimension=4,),
            #     A.RandomSnow(snow_point_upper=0.2,brightness_coeff=1.5),
            #     A.RandomRain( blur_value=3)
            #     # A.OpticalDistortion(distort_limit=2, shift_limit=0.5, p=1)   ,
            #     # A.ImageCompression (quality_lower=70,p=0.5),               
            #     ], p=0.5),
            # A.Affine  (
            #     scale=(0.8,1.2),
            #     rotate=(-50,50),
            #     shear=(-8,8),
            #     fit_output=True,
            #     p=0.7
            # ),
            # A.CLAHE(p=0.5),
            A.RandomBrightnessContrast(p=0.5),],
        additional_targets={
            'mask1': 'mask'
        }
          )  
        # A.RandomGamma(p=0.5)])
    return aug


#https://stackoverflow.com/questions/62195081/calculate-specific-angle-between-two-lines
def angle3pt(a, b, c):
    """Counterclockwise angle in degrees by turning from a to c around b
        Returns a float between 0.0 and 360.0"""
    ang = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
    return ang + 360 if ang < 0 else ang


def get_occluder_augmentor():
    """
    Occludor augmentor
    """
    aug=A.Compose([
        A.AdvancedBlur(),
        # A.OneOf([
        #     A.GaussNoise(),
        #     A.GlassBlur (),
        #     A.MotionBlur (),   
        # ], p=0.5),    
        # A.VerticalFlip(p=0.5),              
        # A.RandomRotate90(p=0.5),
        A.OneOf([
            # A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
            # A.RandomSunFlare (num_flare_circles_lower=3, num_flare_circles_upper=5,src_radius=400),
            # A.OpticalDistortion(distort_limit=2, shift_limit=0.5, p=1)   ,
            A.ImageCompression (quality_lower=70,p=0.5),               
            ], p=0.5),
        A.Affine  (
            scale=(0.8,1.2),
            rotate=(-15,15),
            shear=(-8,8),
            fit_output=True,
            p=0.7
        ),
        # A.CLAHE(p=0.5),
        A.RandomBrightnessContrast(p=0.5,brightness_limit=0.1, contrast_limit=0.1, brightness_by_max=False),    
        # A.RandomGamma(p=0.5)
        ])
    return aug

# https://github.com/isarandi/synthetic-occlusion
def resize_by_factor(im, factor):
    """Returns a copy of `im` resized by `factor`, using bilinear interp for up and area interp
    for downscaling.
    """
    new_size = tuple(np.round(np.array([im.shape[1], im.shape[0]]) * factor).astype(int))
    # interp = cv2.INTER_LANCZOS4 if factor > 1.0 else cv2.INTER_AREA
    return cv2.resize(im, new_size, fx=factor, fy=factor, interpolation= cv2.INTER_LANCZOS4)


def augment_occluder(aug,occluder_img,occluder_mask,src_rect):
    occluder_rect = cv2.boundingRect(occluder_mask)
    #random resize
    try:
        scale_factor = (((src_rect[2]*src_rect[3]))/(occluder_rect[2]*occluder_rect[3]) )*np.random.uniform(0.5, 1)
        scale_factor=np.sqrt(scale_factor)
    except Exception as e:
        print(e)
        scale_factor=1
    occluder_img = resize_by_factor(occluder_img, scale_factor)
    occluder_mask= resize_by_factor(occluder_mask,scale_factor)
    #perform augmentation
    transformed  = aug(image=occluder_img, mask=occluder_mask)
    occluder_img, occluder_mask = transformed["image"],transformed["mask"]

    # convert rgb to rgba to convert black to transparent
    occluder_img = cv2.cvtColor(occluder_img, cv2.COLOR_RGB2RGBA)
    occluder_img[:, :, 3] = occluder_mask

    return occluder_img,occluder_mask


def get_randomOccluderNmask(dtd_folder: str = "./dataset/DTD/images/"):
    #get random shape mask
    rad = np.random.rand()
    edgy = np.random.rand()
    mask_shape=512
    no_of_points=random.randint(3, 15)
    a = get_random_points(n=no_of_points, scale=mask_shape) 
    x,y, _ = get_bezier_curve(a,rad=rad, edgy=edgy)
    occluder_mask=skimage.draw.polygon2mask((mask_shape,mask_shape),list(zip(x,y))).astype(np.uint8)*255

    # get random texture
    texture_list= os.listdir(dtd_folder)
    # texture_list.remove('freckled')
    texture_choice=random.sample(texture_list,1)[0]
    texture_img = random.sample(glob.glob(f"{dtd_folder}/{texture_choice}/*.jpg"),1)[0]
    ori_occluder_img= cv2.imread(texture_img,-1)
    ori_occluder_img=cv2.resize(ori_occluder_img,(mask_shape,mask_shape))
    try:
        ori_occluder_img = cv2.cvtColor(ori_occluder_img, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(e)
    #cropped out the hand img
    try:
        occluder_img=cv2.bitwise_and(ori_occluder_img,ori_occluder_img,mask=occluder_mask)
    except Exception as e:
        print(e)
        return
    occluder_rect = cv2.boundingRect(occluder_mask)
    cropped_occluder_mask = occluder_mask[ occluder_rect[1]:(occluder_rect[1]+occluder_rect[3]),occluder_rect[0]:(occluder_rect[0]+occluder_rect[2])]
    cropped_occluder_img = occluder_img[ occluder_rect[1]:(occluder_rect[1]+occluder_rect[3]),occluder_rect[0]:(occluder_rect[0]+occluder_rect[2])] 
    return cropped_occluder_img, cropped_occluder_mask


class RandomOccluderNmask(object):
    def __init__(self,
                 dtd_folder: str = "./dataset/DTD/images/",
                 mask_shape: int = 512,
                 ):
        self.mask_shape = mask_shape

        # get all texture
        self.ori_occluder_imgs = []
        texture_list = os.listdir(dtd_folder)
        print('loading dtd images...')
        for texture_choice in tqdm(texture_list):
            for texture_img in os.listdir(os.path.join(dtd_folder, texture_choice)):
                img_path = os.path.join(dtd_folder, texture_choice, texture_img)
                if img_path[-4:] != '.jpg':
                    print('skip %s' % img_path)
                    continue
                ori_occluder_img = cv2.imread(img_path, -1)
                ori_occluder_img = cv2.resize(ori_occluder_img, (mask_shape, mask_shape))
                try:
                    ori_occluder_img = cv2.cvtColor(ori_occluder_img, cv2.COLOR_BGR2RGB)
                except Exception as e:
                    print(e)
                self.ori_occluder_imgs.append(ori_occluder_img)

    def get_img_mask(self):
        # get random shape mask
        rad = np.random.rand()
        edgy = np.random.rand()
        mask_shape = self.mask_shape
        no_of_points = random.randint(3, 7)
        a = get_random_points(n=no_of_points, scale=mask_shape)

        x, y, _ = get_bezier_curve(a, rad=rad, edgy=edgy)
        keep_cnt = np.random.randint(7, 13)  # only keep 20 points to save time
        step = x.shape[0] // keep_cnt
        x = x[::step]
        y = y[::step]
        occluder_mask = skimage.draw.polygon2mask((mask_shape, mask_shape),
                                                  list(zip(x, y))).astype(np.uint8) * 255  # very slow

        # cropped out the hand img
        ori_occluder_img = self.ori_occluder_imgs[np.random.randint(0, len(self.ori_occluder_imgs))]
        try:
            occluder_img = cv2.bitwise_and(ori_occluder_img, ori_occluder_img, mask=occluder_mask)
        except Exception as e:
            print(e)
            return
        occluder_rect = cv2.boundingRect(occluder_mask)
        cropped_occluder_mask = occluder_mask[ occluder_rect[1]:(occluder_rect[1]+occluder_rect[3]),occluder_rect[0]:(occluder_rect[0]+occluder_rect[2])]
        cropped_occluder_img = occluder_img[ occluder_rect[1]:(occluder_rect[1]+occluder_rect[3]),occluder_rect[0]:(occluder_rect[0]+occluder_rect[2])]
        return cropped_occluder_img, cropped_occluder_mask



def get_srcNmask(image_file,img_path,mask_path):
    """
    Get the face image and mask
    """
    img_name=image_file.split(".")[0]
    src_img= cv2.imread(os.path.abspath(os.path.join(img_path,image_file)),-1)
    src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)

    src_mask= cv2.imread(mask_path+f"{img_name}.png")
    src_mask=cv2.resize(src_mask,(1024,1024),interpolation= cv2.INTER_LANCZOS4)
    src_mask=cv2.cvtColor(src_mask,cv2.COLOR_RGB2GRAY)

    return src_img, src_mask


def get_occluderNmask(occluder_file,img_path,mask_path):
    occluder_name=occluder_file.split(".")[0]
    ori_occluder_img= cv2.imread(os.path.abspath(os.path.join(img_path,occluder_file)),-1)
    try:
        ori_occluder_img = cv2.cvtColor(ori_occluder_img, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(e)
        exit()    
    occluder_mask= cv2.imread(os.path.abspath(os.path.join(mask_path,occluder_name+".png")))
    occluder_mask = cv2.cvtColor(occluder_mask, cv2.COLOR_BGR2GRAY)

    occluder_mask=cv2.resize(occluder_mask,(ori_occluder_img.shape[1],ori_occluder_img.shape[0]),interpolation= cv2.INTER_LANCZOS4)
    
    #cropped out the hand img
    try:
        occluder_img=cv2.bitwise_and(ori_occluder_img,ori_occluder_img,mask=occluder_mask)
    except Exception as e:
        print(e)
        return

    occluder_rect = cv2.boundingRect(occluder_mask)
    cropped_occluder_mask = occluder_mask[ occluder_rect[1]:(occluder_rect[1]+occluder_rect[3]),occluder_rect[0]:(occluder_rect[0]+occluder_rect[2])]
    cropped_occluder_img = occluder_img[ occluder_rect[1]:(occluder_rect[1]+occluder_rect[3]),occluder_rect[0]:(occluder_rect[0]+occluder_rect[2])]
    return cropped_occluder_img, cropped_occluder_mask


class OccluderNmask(object):
    def __init__(self,
                 occluders_list: list = None,
                 img_path: str = None,
                 mask_path: str = None,
                 ):
        self.occluders_list = occluders_list
        self.img_path = img_path
        self.mask_path = mask_path

        self.occluder_imgs = []
        self.occluder_masks = []
        print('loading %s images...' % img_path)
        for occluder_file in tqdm(occluders_list):
            occluder_name = occluder_file.split(".")[0]
            ori_occluder_img = cv2.imread(os.path.abspath(os.path.join(img_path, occluder_file)), -1)
            try:
                ori_occluder_img = cv2.cvtColor(ori_occluder_img, cv2.COLOR_BGR2RGB)
            except Exception as e:
                print(e)
                exit()

            occluder_mask = cv2.imread(os.path.abspath(os.path.join(mask_path, occluder_name + ".png")))
            occluder_mask = cv2.cvtColor(occluder_mask, cv2.COLOR_BGR2GRAY)

            occluder_mask = cv2.resize(occluder_mask, (ori_occluder_img.shape[1], ori_occluder_img.shape[0]),
                                       interpolation=cv2.INTER_LANCZOS4)
            self.occluder_imgs.append(ori_occluder_img)
            self.occluder_masks.append(occluder_mask)

    def get_img_mask(self):
        idx = np.random.randint(0, len(self.occluder_imgs))
        ori_occluder_img = self.occluder_imgs[idx]
        occluder_mask = self.occluder_masks[idx]

        # cropped out the hand img
        try:
            occluder_img = cv2.bitwise_and(ori_occluder_img, ori_occluder_img, mask=occluder_mask)
        except Exception as e:
            print(e)
            return

        occluder_rect = cv2.boundingRect(occluder_mask)
        cropped_occluder_mask = occluder_mask[occluder_rect[1]:(occluder_rect[1] + occluder_rect[3]),
                                occluder_rect[0]:(occluder_rect[0] + occluder_rect[2])]
        cropped_occluder_img = occluder_img[occluder_rect[1]:(occluder_rect[1] + occluder_rect[3]),
                               occluder_rect[0]:(occluder_rect[0] + occluder_rect[2])]
        return cropped_occluder_img, cropped_occluder_mask


# @ray.remote
# class ProgressBarActor:
#     counter: int
#     delta: int
#     event: Event
#
#     def __init__(self) -> None:
#         self.counter = 0
#         self.delta = 0
#         self.event = Event()
#
#     def update(self, num_items_completed: int) -> None:
#         """Updates the ProgressBar with the incremental
#         number of items that were just completed.
#         """
#         self.counter += num_items_completed
#         self.delta += num_items_completed
#         self.event.set()
#
#     async def wait_for_update(self) -> Tuple[int, int]:
#         """Blocking call.
#
#         Waits until somebody calls `update`, then returns a tuple of
#         the number of updates since the last call to
#         `wait_for_update`, and the total number of completed items.
#         """
#         await self.event.wait()
#         self.event.clear()
#         saved_delta = self.delta
#         self.delta = 0
#         return saved_delta, self.counter
#
#     def get_counter(self) -> int:
#         """
#         Returns the total number of complete items.
#         """
#         return self.counter
#
#
# class ProgressBar:
#     progress_actor: ActorHandle
#     total: int
#     description: str
#     pbar: tqdm
#
#     def __init__(self, total: int, description: str = ""):
#         # Ray actors don't seem to play nice with mypy, generating
#         # a spurious warning for the following line,
#         # which we need to suppress. The code is fine.
#         self.progress_actor = ProgressBarActor.remote()  # type: ignore
#         self.total = total
#         self.description = description
#
#     @property
#     def actor(self) -> ActorHandle:
#         """Returns a reference to the remote `ProgressBarActor`.
#
#         When you complete tasks, call `update` on the actor.
#         """
#         return self.progress_actor
#
#     def print_until_done(self) -> None:
#         """Blocking call.
#
#         Do this after starting a series of remote Ray tasks, to which you've
#         passed the actor handle. Each of them calls `update` on the actor.
#         When the progress meter reaches 100%, this method returns.
#         """
#         pbar = tqdm(desc=self.description, total=self.total)
#         while True:
#             delta, counter = ray.get(self.actor.wait_for_update.remote())
#             pbar.update(delta)
#             if counter >= self.total:
#                 pbar.close()
#                 return