import argparse
import math
import random
import os,time
import numpy as np
import torch
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
import torch.distributed as dist
import torchvision
from torchvision import transforms, utils
from tqdm import tqdm

from dataset import *
from torch.autograd import Variable
import matplotlib as mlb


import itertools
from tensorboardX import SummaryWriter
from utils import *


from models.encoders.psp_encoders import *
from models.stylegan2.model import *
from models.nets import *


from distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)

def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)



def train(args, train_loader, encoder_lmk, encoder_target, generator, decoder, bald_model, smooth_mask, device):

    train_loader = sample_data(train_loader)
    generator.eval()
    encoder_lmk.eval()
    encoder_target.eval()

    zero_latent = torch.zeros((args.batch,18-args.coarse,512)).to(device).detach()

    trans_256 = transforms.Resize(256)
    trans_1024 = transforms.Resize(1024)

    pbar = range(args.sample_number)
    pbar = tqdm(pbar, initial=args.start_iter, dynamic_ncols=True, smoothing=0.01)

    for idx in pbar:
        i = idx + args.start_iter

        if i > args.sample_number:
            print("Done!")
            break

        time0 = time.time()

        s_img,s_code,s_map,s_lmk,t_img,t_code,t_map,t_lmk,t_mask,s_index, t_index = next(train_loader) #256;1024;...
        time1 = time.time()
        s_img = s_img.to(device)
        s_map = s_map.to(device).transpose(1,3).float()#[:,33:]
        t_img = t_img.to(device)
        t_map = t_map.to(device).transpose(1,3).float()#[:,33:]
        t_lmk = t_lmk.to(device)
        t_mask = t_mask.to(device)

        s_frame_code = s_code.to(device)
        t_frame_code = t_code.to(device)



        input_map = torch.cat([s_map,t_map],dim=1)
        t_mask = t_mask.unsqueeze_(1).float()

        t_lmk_code = encoder_lmk(input_map) 


        t_lmk_code = torch.cat([t_lmk_code,zero_latent],dim=1)
        fusion_code = s_frame_code + t_lmk_code


        fusion_code = torch.cat([fusion_code[:,:18-args.coarse],t_frame_code[:,18-args.coarse:]],dim=1)
        fusion_code = bald_model(fusion_code.view(fusion_code.size(0), -1), 2)
        fusion_code = fusion_code.view(t_frame_code.size())


        source_feas = generator([fusion_code], input_is_latent=True, randomize_noise=False)
        target_feas = encoder_target(t_img)

        blend_img = decoder(source_feas,target_feas,t_mask)



        name = str(int(s_index[0]))+'_'+str(int(t_index[0]))


        with torch.no_grad():
            sample = torch.cat([s_img.detach(), t_img.detach()])
            sample = torch.cat([sample, blend_img.detach()])
            t_mask = torch.stack([t_mask,t_mask,t_mask],dim=1).squeeze(2)
            sample = torch.cat([sample, t_mask.detach()])

            utils.save_image(
                blend_img,
                _dirs[0]+"/"+name+".png",
                nrow=int(args.batch),
                normalize=True,
                range=(-1, 1),
            )

            utils.save_image(
                sample,
                _dirs[1]+"/"+name+".png",
                nrow=int(args.batch),
                normalize=True,
                range=(-1, 1),
            )






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--e_ckpt", type=str, default='./CELEBA-HQ-1024.pt')


    parser.add_argument("--image_path", type=str, default="../CelebAMask-HQ/CelebA-HQ-img/")
    
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--sample_number", type=int, default=1000)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.00005)
    parser.add_argument("--local_rank", type=int, default=0)

    parser.add_argument("--lpips", type=float, default=5.0)
    parser.add_argument("--hm", type=float, default=0.5)
    parser.add_argument("--l2", type=float, default=10.0)

    parser.add_argument("--fd", type=float, default=10.0)



    parser.add_argument("--id", type=float, default=10.0)
    parser.add_argument("--lmk", type=float, default=0.1)
    parser.add_argument("--adv", type=float, default=5.0)  
    parser.add_argument("--tv", type=float, default=10.0)  

    parser.add_argument("--mask", type=float, default=5.0)  


    parser.add_argument("--r1", type=float, default=10)
    parser.add_argument("--d_reg_every", type=int, default=16)
    parser.add_argument("--tensorboard", action="store_true",default=True)
    



    
    args = parser.parse_args()

    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend="nccl", init_method="env://")
    synchronize()
    device = args.device

    args.path = ".//"    
    args.start_iter = 0

    args.size = 1024
    args.latent = 512
    args.n_mlp = 8
    args.channel_multiplier = 2
    args.coarse = 7
    args.least_size = 8
    args.largest_size = 512
    args.mapping_layers = 18
    args.mapping_fmaps = 512
    args.mapping_lrmul = 1
    args.mapping_nonlinearity = 'linear'


    encoder_lmk = GradualLandmarkEncoder(106*2).to(device)
    encoder_target = GPENEncoder(args.largest_size).to(device)
    generator = Generator(args.size,args.latent,args.n_mlp).to(device)
    
    decoder = Decoder(args.least_size,args.size).to(device)

    bald_model = F_mapping(mapping_lrmul= args.mapping_lrmul, mapping_layers=args.mapping_layers, mapping_fmaps=args.mapping_fmaps, mapping_nonlinearity = args.mapping_nonlinearity).to(device)
    bald_model.eval()



    _dirs = ['./test/swapped/','./test/fuse/']
    
    for x in _dirs:
        try:
            os.makedirs(x)
        except:
            pass

    to_tensor = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    to_tensor2 = transforms.Compose([
        transforms.Resize(args.size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])


    dataset = SourceICTargetICLM(args.image_path,to_tensor_256 = to_tensor2, to_tensor_1024=to_tensor2)
    train_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch,
            sampler=data_sampler(dataset, shuffle=True, distributed=True),
            drop_last=True,
        )



    encoder_lmk = nn.parallel.DistributedDataParallel(encoder_lmk,device_ids=[args.local_rank],output_device=args.local_rank,broadcast_buffers=False,find_unused_parameters=True)
    encoder_target = nn.parallel.DistributedDataParallel(encoder_target,device_ids=[args.local_rank],output_device=args.local_rank,broadcast_buffers=False,find_unused_parameters=True)
    generator = nn.parallel.DistributedDataParallel(generator,device_ids=[args.local_rank],output_device=args.local_rank,broadcast_buffers=False,find_unused_parameters=True)
    decoder = nn.parallel.DistributedDataParallel(decoder,device_ids=[args.local_rank],output_device=args.local_rank,broadcast_buffers=False,find_unused_parameters=True)
    bald_model = nn.parallel.DistributedDataParallel(bald_model,device_ids=[args.local_rank],output_device=args.local_rank,broadcast_buffers=False,find_unused_parameters=True)


    e_ckpt = torch.load(args.e_ckpt,  map_location=torch.device('cpu'))

    encoder_lmk.load_state_dict(e_ckpt["encoder_lmk"])
    encoder_target.load_state_dict(e_ckpt["encoder_target"])
    decoder.load_state_dict(e_ckpt["decoder"])
    generator.load_state_dict(e_ckpt["generator"])
    bald_model.load_state_dict(e_ckpt["bald_model"])






    train(args, train_loader, encoder_lmk, encoder_target, generator, decoder, bald_model, smooth_mask, device)
