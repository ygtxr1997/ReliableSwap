#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: fs_model_fix_idnorm_donggp_saveoptim copy.py
# Created Date: Wednesday January 12th 2022
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Thursday, 21st April 2022 8:13:37 pm
# Modified By: Chen Xuanhong
# Copyright (c) 2022 Shanghai Jiao Tong University
#############################################################


import torch
import torch.nn as nn

from modules.layers.simswap.base_model import BaseModel
from modules.layers.simswap.fs_networks_fix import Generator_Adain_Upsample

from modules.layers.simswap.pg_modules.projected_discriminator import ProjectedDiscriminator


def compute_grad2(d_out, x_in):
    batch_size = x_in.size(0)
    grad_dout = torch.autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grad_dout2 = grad_dout.pow(2)
    assert(grad_dout2.size() == x_in.size())
    reg = grad_dout2.view(batch_size, -1).sum(1)
    return reg


class fsModel(BaseModel):
    def name(self):
        return 'fsModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        # if opt.resize_or_crop != 'none' or not opt.isTrain:  # when training at full res this causes OOM
        self.isTrain = opt.isTrain

        # Generator network
        self.netG = Generator_Adain_Upsample(input_nc=3, output_nc=3, latent_size=512, n_blocks=9, deep=opt.Gdeep)
        self.netG.cuda()

        # Id network
        from modules.third_party.arcface import iresnet100
        netArc_pth = "/apdcephfs_cq2/share_1290939/gavinyuan/code/FaceShifter/faceswap/faceswap/" \
                     "checkpoints/face_id/ms1mv3_arcface_r100_fp16_backbone.pth"  #opt.Arc_path
        self.netArc = iresnet100(pretrained=False, fp16=False)
        self.netArc.load_state_dict(torch.load(netArc_pth, map_location="cpu"))
        # netArc_checkpoint = opt.Arc_path
        # netArc_checkpoint = torch.load(netArc_checkpoint, map_location=torch.device("cpu"))
        # self.netArc = netArc_checkpoint['model'].module
        self.netArc = self.netArc.cuda()
        self.netArc.eval()
        self.netArc.requires_grad_(False)
        if not self.isTrain:
            pretrained_path =  opt.checkpoints_dir
            self.load_network(self.netG, 'G', opt.which_epoch, pretrained_path)
            return
        self.netD = ProjectedDiscriminator(diffaug=False, interp224=False, **{})
        # self.netD.feature_network.requires_grad_(False)
        self.netD.cuda()


        if self.isTrain:
            # define loss functions
            self.criterionFeat  = nn.L1Loss()
            self.criterionRec   = nn.L1Loss()

            # initialize optimizers
            # optimizer G
            params = list(self.netG.parameters())
            self.optimizer_G = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.99),eps=1e-8)

            # optimizer D
            params = list(self.netD.parameters())
            self.optimizer_D = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.99),eps=1e-8)

        # load networks
        if opt.continue_train:
            pretrained_path = '' if not self.isTrain else opt.load_pretrain
            # print (pretrained_path)
            self.load_network(self.netG, 'G', opt.which_epoch, pretrained_path)
            self.load_network(self.netD, 'D', opt.which_epoch, pretrained_path)
            self.load_optim(self.optimizer_G, 'G', opt.which_epoch, pretrained_path)
            self.load_optim(self.optimizer_D, 'D', opt.which_epoch, pretrained_path)
        torch.cuda.empty_cache()

    def cosin_metric(self, x1, x2):
        #return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))
        return torch.sum(x1 * x2, dim=1) / (torch.norm(x1, dim=1) * torch.norm(x2, dim=1))

    def save(self, which_epoch):
        self.save_network(self.netG, 'G', which_epoch)
        self.save_network(self.netD, 'D', which_epoch)
        self.save_optim(self.optimizer_G, 'G', which_epoch)
        self.save_optim(self.optimizer_D, 'D', which_epoch)
        '''if self.gen_features:
            self.save_network(self.netE, 'E', which_epoch, self.gpu_ids)'''

    def update_fixed_params(self):
        raise ValueError('Not used')
        # after fixing the global generator for a number of iterations, also start finetuning it
        params = list(self.netG.parameters())
        if self.gen_features:
            params += list(self.netE.parameters())
        self.optimizer_G = torch.optim.Adam(params, lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        if self.opt.verbose:
            print('------------ Now also finetuning global generator -----------')

    def update_learning_rate(self):
        raise ValueError('Not used')
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        if self.opt.verbose:
            print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr


if __name__ == "__main__":
    import os
    import argparse

    def str2bool(v):
        return v.lower() in ('true')


    class TrainOptions:
        def __init__(self):
            self.parser = argparse.ArgumentParser()
            self.initialized = False

        def initialize(self):
            self.parser.add_argument('--name', type=str, default='simswap',
                                     help='name of the experiment. It decides where to store samples and models')
            self.parser.add_argument('--gpu_ids', default='0')
            self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints',
                                     help='models are saved here')
            self.parser.add_argument('--isTrain', type=str2bool, default='True')

            # input/output sizes
            self.parser.add_argument('--batchSize', type=int, default=8, help='input batch size')

            # for displays
            self.parser.add_argument('--use_tensorboard', type=str2bool, default='False')

            # for training
            self.parser.add_argument('--dataset', type=str, default="/path/to/VGGFace2",
                                     help='path to the face swapping dataset')
            self.parser.add_argument('--continue_train', type=str2bool, default='False',
                                     help='continue training: load the latest model')
            self.parser.add_argument('--load_pretrain', type=str, default='./checkpoints/simswap224_test',
                                     help='load the pretrained model from the specified location')
            self.parser.add_argument('--which_epoch', type=str, default='10000',
                                     help='which epoch to load? set to latest to use latest cached model')
            self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
            self.parser.add_argument('--niter', type=int, default=10000, help='# of iter at starting learning rate')
            self.parser.add_argument('--niter_decay', type=int, default=10000,
                                     help='# of iter to linearly decay learning rate to zero')
            self.parser.add_argument('--beta1', type=float, default=0.0, help='momentum term of adam')
            self.parser.add_argument('--lr', type=float, default=0.0004, help='initial learning rate for adam')
            self.parser.add_argument('--Gdeep', type=str2bool, default='False')

            # for discriminators
            self.parser.add_argument('--lambda_feat', type=float, default=10.0, help='weight for feature matching loss')
            self.parser.add_argument('--lambda_id', type=float, default=30.0, help='weight for id loss')
            self.parser.add_argument('--lambda_rec', type=float, default=10.0, help='weight for reconstruction loss')

            self.parser.add_argument("--Arc_path", type=str, default='arcface_model/arcface_checkpoint.tar',
                                     help="run ONNX model via TRT")
            self.parser.add_argument("--total_step", type=int, default=1000000, help='total training step')
            self.parser.add_argument("--log_frep", type=int, default=200, help='frequence for printing log information')
            self.parser.add_argument("--sample_freq", type=int, default=1000, help='frequence for sampling')
            self.parser.add_argument("--model_freq", type=int, default=10000, help='frequence for saving the model')

            self.isTrain = True

        def parse(self, save=True):
            if not self.initialized:
                self.initialize()
            self.opt = self.parser.parse_args()
            self.opt.isTrain = self.isTrain  # train or test

            args = vars(self.opt)

            print('------------ Options -------------')
            for k, v in sorted(args.items()):
                print('%s: %s' % (str(k), str(v)))
            print('-------------- End ----------------')

            # save to the disk
            # if self.opt.isTrain:
            #     expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
            #     util.mkdirs(expr_dir)
            #     if save and not self.opt.continue_train:
            #         file_name = os.path.join(expr_dir, 'opt.txt')
            #         with open(file_name, 'wt') as opt_file:
            #             opt_file.write('------------ Options -------------\n')
            #             for k, v in sorted(args.items()):
            #                 opt_file.write('%s: %s\n' % (str(k), str(v)))
            #             opt_file.write('-------------- End ----------------\n')
            return self.opt

    source = torch.randn(8, 3, 256, 256).cuda()
    target = torch.randn(8, 3, 256, 256).cuda()

    opt = TrainOptions().parse()
    model = fsModel()
    model.initialize(opt)

    import torch.nn.functional as F
    img_id_112 = F.interpolate(source, size=(112, 112), mode='bicubic')
    latent_id = model.netArc(img_id_112)
    latent_id = F.normalize(latent_id, p=2, dim=1)

    img_fake = model.netG(target, latent_id)
    gen_logits, _ = model.netD(img_fake.detach(), None)
    loss_Dgen = (F.relu(torch.ones_like(gen_logits) + gen_logits)).mean()

    real_logits, _ = model.netD(source, None)

    print('img_fake:', img_fake.shape, 'real_logits:', real_logits.shape)
