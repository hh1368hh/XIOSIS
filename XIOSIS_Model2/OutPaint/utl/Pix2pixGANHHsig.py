import torch
from torch import nn
import torch.nn.functional as F
import math
import torch

import sys
# adding Path to the system path
sys.path.insert(0, '../')

from GANmodels.base_model import BaseModel
from GANmodels import networkssig as networks



class defaults:
    def __init__(self,input_nc=1,output_nc=1,is_train=True,norm='batch',netG='unet_512',netD='basic',pool_size=50, gan_mode='vanilla',gpu_ids=[0]):
        self.input_nc=input_nc
        self.output_nc=output_nc
        self.gpu_ids=gpu_ids # 'gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU'
        self.isTrain=is_train
        self.norm=norm # 'instance normalization or batch normalization [instance | batch | none]
        self.netG=netG
        self.netD=netD
        self.pool_size=pool_size # 'the size of image buffer that stores previously generated images'
        self.gan_mode=gan_mode #the type of GAN objective. [vanilla| lsgan | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.
        self.checkpoints_dir='./checkpoints' #'models are saved here'
        self.input_nc=input_nc
        self.output_nc=output_nc
        self.ngf=64 #'# of gen filters in the last conv layer'
        self.ndf=64 # # of discrim filters in the first conv layer'
        self.n_layers_D=3 #'instance normalization or batch normalization [instance | batch | none]
        self.init_type='normal' #'network initialization [normal | xavier | kaiming | orthogonal]'
        self.init_gain=0.02 #scaling factor for normal, xavier and orthogonal.
        self.name='HHPix2pix'
        self.no_dropout=False #'no dropout for the generator'
        self.lr=0.0002 # 'initial learning rate for adam'
        self.beta1=0.5 # 'momentum term of adam'
        self.n_epochs=100 # 'number of epochs with the initial learning rate'
        self.n_epochs_decay=100 # 'number of epochs to linearly decay learning rate to zero'
        # self.gan_mode='vanilla' #the type of GAN objective. [vanilla| lsgan | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.
        self.lr_policy='linear' #'learning rate policy. [linear | step | plateau | cosine]'
        self.lr_decay_iters=50 #'multiply by a gamma every lr_decay_iters iterations'
        self.direction='AtoB'
        self.lambda_L1=100.0 #'weight for L1 loss'


class Pix2PixModel(BaseModel):
    def __init__(self,input_nc=6,output_nc=3,is_train=True,norm='batch',netG='unet_512',netD='basic',pool_size=0, gan_mode='vanilla',gpu_ids=[0]):
        opt=defaults()
        opt.input_nc=input_nc #'# of input image channels: 3 for RGB and 1 for grayscale'
        opt.output_nc=output_nc #'# of output image channels: 3 for RGB and 1 for grayscale'
        opt.gpu_ids=gpu_ids #'gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU'
        opt.isTrain=is_train
        opt.norm=norm
        opt.netG=netG #specify generator architecture [resnet_9blocks | resnet_6blocks | unet_256 | unet_128]
        opt.netD=netD #specify discriminator architecture [basic | n_layers | pixel]. The basic model is a 70x70 PatchGAN. n_layers allows you to specify the layers in the discriminator
        opt.pool_size=pool_size
        opt.gan_mode=gan_mode
        self.Options=opt
        # print(len(gpu_ids))
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        # print('aaaa')
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        # AtoB = self.opt.direction == 'AtoB'
        # self.real_A = input['A' if AtoB else 'B'].to(self.device)
        # self.real_B = input['B' if AtoB else 'A'].to(self.device)
        # self.image_paths = input['A_paths' if AtoB else 'B_paths']
        self.real_A = input[0].to(self.device, non_blocking=True)
        self.real_B = input[1].to(self.device, non_blocking=True)

    def forward(self):
        # print(self.real_A.device)
        
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG(self.real_A)  # G(A)
        # print(self.fake_B.device)
    def Hforward_G(self,x):
        out=self.netG(x)
        return out
    def Hforward_D(self,x):
        out=self.netD(x)
        return out

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        # print(self.real_A.device)
        # print(self.fake_B.device)
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad(set_to_none=True)     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad(set_to_none=True)        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights

