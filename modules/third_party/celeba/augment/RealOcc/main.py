import argparse
import os 
import cv2
import random
import numpy as np
import imutils
import albumentations as A
import torch
import cupy as cp    
import ray
import imgaug
import time
import os
from utils.utils import *
from utils.colour_transfer import *
from utils.paste_over import *
from utils.random_shape_generator import *
from configs.config import cfg

#https://github.com/open-mmlab/mmediting/blob/23213c839ff2d1907a80d6ea29f13c32a24bb8ef/mmedit/apis/train.py
def set_random_seed(seed, deterministic=False):
    """Set random seed.
    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    # ia.seed(seed)
    cp.random.seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    imgaug.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)



@ray.remote(num_cpus=1,num_gpus=torch.cuda.device_count()/int(os.environ.get("NUM_WORKERS")))
class Occlusion_Generator:
    def __init__(self,args, images_list,occluders_list,seeds):
        self.args=args
        self.image_augmentor= get_src_augmentor()
        self.occluder_augmentor= get_occluder_augmentor()
        self.images_list=images_list
        self.occluders_list=occluders_list
        self.seeds=seeds
        
    def occlude_images(self,index):
        try:
            image=self.images_list[index]
            occluder=self.occluders_list[index]
            seed=self.seeds[index]
            
             #set seed
            set_random_seed(seed)

            # get source img and mask
            src_img, src_mask = get_srcNmask(image,self.args["srcImageDir"],self.args["srcMaskDir"])

            #get occluder img and mask
            if self.args["randomOcclusion"]:
                occluder_img , occluder_mask= get_randomOccluderNmask()
            else:
                occluder_img , occluder_mask= get_occluderNmask(occluder,self.args["occluderDir"],self.args["occluderMaskDir"])
           
            src_rect = cv2.boundingRect(src_mask)

            #colour transfer
            if self.args["colour_transfer_sot"]:
                try:
                    occluder_img=self.colour_transfer(src_img,src_mask,occluder_img,src_rect)
                except Exception as e:
                    print(e)
            #augment occluders
            occluder_img, occluder_mask =augment_occluder(self.occluder_augmentor,occluder_img,occluder_mask,src_rect)
            #random location around src
            occluder_coord = np.random.uniform([src_rect[0],src_rect[1]], [src_rect[0]+src_rect[2],src_rect[1]+src_rect[3]])

            if self.args["rotate_around_center"]:
                src_center=(src_rect[0]+(src_rect[2]/2),(src_rect[1]+src_rect[3]/2))
                rotation= angle3pt((src_center[0],occluder_coord[1]),src_center,occluder_coord)
                if occluder_coord[1]>src_center[1]:
                    rotation=rotation+180
                occluder_img= imutils.rotate_bound(occluder_img,rotation)
                occluder_mask=imutils.rotate_bound(occluder_mask,rotation)

            #overlay occluder to src images
            try:
                occlusion_mask=np.zeros(src_mask.shape, np.uint8)
                occlusion_mask[(occlusion_mask>0) & (occlusion_mask<255)]=255
                #paste occluder to src image
                result_img,result_mask,occlusion_mask=paste_over(occluder_img,occluder_mask,src_img,src_mask,occluder_coord,occlusion_mask,self.args["randomOcclusion"])
            except Exception as e:
                print(e)
                print(f'Failed: {image} , {occluder}')
                return

            #blur edges of occluder
            kernel = np.ones((5,5),np.uint8)
            occlusion_mask_edges=cv2.dilate(occlusion_mask,kernel,iterations = 2)-cv2.erode(occlusion_mask,kernel,iterations = 2)
            ret, filtered_occlusion_mask_edges = cv2.threshold(occlusion_mask_edges, 240, 255, cv2.THRESH_BINARY)
            blurred_image = cv2.GaussianBlur(result_img,(5,5),0)
            result_img = np.where(np.dstack((np.invert(filtered_occlusion_mask_edges==255),)*3), result_img, blurred_image)


            # augment occluded image
            transformed  = self.image_augmentor(image=result_img, mask=result_mask,mask1= occlusion_mask)
            result_img, result_mask,occlusion_mask = transformed["image"],transformed["mask"],transformed["mask1"]
            result_img = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)

            #save images
            self.save_images(image.split(".")[0],result_img,result_mask,occlusion_mask)
        except Exception as e: 
            print(e)
            print(image)
    def save_images(self,img_name,image,mask,occlusion_mask):

        cv2.imwrite(self.args["outputImgDir"]+f"{img_name}.jpg",image)
        cv2.imwrite(self.args["outputMaskDir"]+f"{img_name}.png",mask/255   )
        if self.args["maskForOcclusion"]:
            cv2.imwrite(self.args["occlusionMaskDir"]+f"{img_name}.png",occlusion_mask)

    def colour_transfer(self,src_img,src_mask,occluder_img,src_rect):
        ##change the colour of the occluder 
        #crop the src image 
        temp_src=cv2.bitwise_or(src_img,src_img,mask=src_mask)
        cropped_src = temp_src[ src_rect[1]:(src_rect[1]+src_rect[3]),src_rect[0]:(src_rect[0]+src_rect[2])] 
        #crop the mask 
        cropped_src_mask = src_mask[ src_rect[1]:(src_rect[1]+src_rect[3]),src_rect[0]:(src_rect[0]+src_rect[2])] 
        cropped_src=cv2.resize(cropped_src,(occluder_img.shape[1],occluder_img.shape[0]),interpolation= cv2.INTER_LANCZOS4)
        #resize to the size of src image
        cropped_src_mask=cv2.resize(cropped_src_mask,(occluder_img.shape[1],occluder_img.shape[0]),interpolation= cv2.INTER_LANCZOS4)

        ##solve black imbalance
        #get the mean and std in each channel 
        r=np.mean(cropped_src[:,:,0][cropped_src[:,:,0] != 0])
        g=np.mean(cropped_src[:,:,1][cropped_src[:,:,1] != 0])
        b=np.mean(cropped_src[:,:,2][cropped_src[:,:,2] != 0])
        r_std=np.std(cropped_src[:,:,0][cropped_src[:,:,0] != 0])
        g_std=np.std(cropped_src[:,:,1][cropped_src[:,:,1] != 0])
        b_std=np.std(cropped_src[:,:,2][cropped_src[:,:,2] != 0])

        # calculate the black ratio. src/occluder  
        # current lower threshold is set to half the mean in each channel
        black_ratio=np.round((np.sum(cropped_src < (r/2,g/2,b/2))/np.sum(occluder_img == (0,0,0)))-1,2)
        
        if black_ratio>1:
            black_ratio=1

        if (black_ratio) >0:
            cropped_src_mask[cropped_src_mask==0]=np.random.binomial(n=1, p=1-black_ratio, size=[cropped_src_mask[cropped_src_mask==0].size])
            cropped_src[:,:,:3][np.invert(cropped_src_mask.astype(bool))] = [r, g, b]
        # handle pixels that is too bright
        # current upper threshold set to mean + 1 std
        r2, g2, b2 = r+r_std,g+g_std,b+b_std 
        red, green, blue = cropped_src[:,:,0], cropped_src[:,:,1], cropped_src[:,:,2]
        mask = (red > r2) | (green > g2) | (blue > b2)
        cropped_src[:,:,:3][mask] = [min(255,r+r_std),min(255,g+g_std), min(255,b+b_std) ]

        occluder_img=color_transfer_sot(occluder_img/255,cropped_src/255)
        occluder_img = (np.clip( occluder_img, 0.0, 1.0)*255).astype("uint8")
        return occluder_img


if __name__ == "__main__":
    parser= argparse.ArgumentParser(description="Occlusion Augmentation on image dataset.")
    parser.add_argument("--config",required=True,default=".", help="path to config file")
    parser.add_argument("-s","--seed",default=2, help="seed for reproducible")
    parser.add_argument("--opts",help="Modify config options using the command-line 'KEY VALUE' pairs",default=[],nargs=argparse.REMAINDER)
    args=parser.parse_args()
    start_time = time.time()

    cfg.merge_from_file(args.config)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    print(cfg)
    
    #set seed and env
    set_random_seed(int(args.seed))

    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    
    gpu_counts= torch.cuda.device_count()
    print("Number of GPU's:",gpu_counts)
    
    if (gpu_counts==0):
        print("At least 1 gpu is required")
        exit()


    #validate if path exists
    images_name = validate_path("Source Image",cfg.SOURCE_DATASET.IMG_DIR)
    _ = validate_path("Source mask",cfg.SOURCE_DATASET.MASK_DIR)


    if not cfg.MODE.RANDOCC:
        occluders_name = validate_path("Occluder Image", cfg.OCCLUDER_DATASET.IMG_DIR)
        _ = validate_path("Occluder mask",cfg.OCCLUDER_DATASET.MASK_DIR)
    
    #check if pairs exists, only return img name with valid mask
    images_list = validate_img_mask_pair(images_name,cfg.SOURCE_DATASET.MASK_DIR)
    
    if not cfg.MODE.RANDOCC:
        occluders_list = validate_img_mask_pair(occluders_name,cfg.OCCLUDER_DATASET.MASK_DIR)

    arguments={
            "srcImageDir":cfg.SOURCE_DATASET.IMG_DIR,
            "srcMaskDir":cfg.SOURCE_DATASET.MASK_DIR,
            "occluderDir": cfg.OCCLUDER_DATASET.IMG_DIR,
            "occluderMaskDir":cfg.OCCLUDER_DATASET.MASK_DIR,
            "outputImgDir": cfg.OUTPUT_PATH +"img/",
            "outputMaskDir": cfg.OUTPUT_PATH +"mask/",
            "colour_transfer_sot": cfg.AUGMENTATION.SOT,
            "rotate_around_center": cfg.AUGMENTATION.ROTATE_AROUND_CENTER,
            "maskForOcclusion":cfg.OCCLUSION_MASK,
            "occlusionMaskDir": cfg.OUTPUT_PATH +"occlusion_mask/",
            "randomOcclusion":cfg.MODE.RANDOCC
    }


    # create output folder if not exists
    if not os.path.exists(arguments["outputImgDir"]):
        os.makedirs(arguments["outputImgDir"],exist_ok=True)
    if not os.path.exists(arguments["outputMaskDir"]):
        os.makedirs(arguments["outputMaskDir"], exist_ok=True)
    if not os.path.exists(arguments["occlusionMaskDir"]):
        os.makedirs(arguments["occlusionMaskDir"], exist_ok=True)


    #https://gist.github.com/suchow/3cd1fa50234a1d5cf31fa2f242487039 reproducible multi processing
    #set seed explicity for each image to ensure reproducibility

    seeds = [random.getrandbits(32) for _ in range(len(images_list))]

    if not cfg.MODE.RANDOCC:
        occluders= random.choices(occluders_list,k=len(images_list))
    else:
        occluders=[0]*len(images_list)
    
    ray.init(num_cpus=int(os.environ["NUM_WORKERS"]), num_gpus=gpu_counts) 

    pool = ray.util.ActorPool([Occlusion_Generator.remote(arguments,images_list,occluders,seeds) for _ in range(int(os.environ["NUM_WORKERS"]))])
    import tqdm
    for i in tqdm.tqdm(pool.map_unordered(lambda actor, i: actor.occlude_images.remote(i),list(range(len(images_list)))),total=len(images_list)):
        pass

    print("--- %s seconds ---" % (time.time() - start_time))
