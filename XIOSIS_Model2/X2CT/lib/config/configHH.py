# ------------------------------------------------------------------------------
# Copyright (c) Tencent
# Licensed under the GPLv3 License.
# Created by Kai Ma (makai0324@gmail.com)
# ------------------------------------------------------------------------------

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

from easydict import EasyDict
import os
import numpy as np
import torch

def initial_opt():

  opt = EasyDict()
  opt.model_name='MultiViewCTGAN'
  # opt.gpu_ids=range(torch.cuda.device_count())
  opt.gpu_ids=[]
  for i in range(torch.cuda.device_count()):
    opt.gpu_ids.append(i)


  ISIZE=128

  # Model Path
  # opt.MODEL_SAVE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'save_models'))
  opt.MODEL_SAVE_PATH=[]
  opt.CT_MIN_MAX = [0, 3000]
  opt.XRAY1_MIN_MAX = [0, 1]
  opt.XRAY2_MIN_MAX = [0, 1]
  opt.XRAY3_MIN_MAX = [0, 1]
  opt.CT_MEAN_STD = [0., 1.0]
  # opt.XRAY1_MEAN_STD = [0., 1.0]
  # opt.XRAY2_MEAN_STD = [0., 1.0]

  '''
  Network
    Generator
  '''
  opt.NETWORK = EasyDict()
  # of input image channels
  opt.NETWORK.input_nc_G = 512
  # of output image channels
  opt.NETWORK.output_nc_G = 1
  # of gen filters in first conv layer
  opt.NETWORK.ngf = 64
  # selects model to use for netG
  opt.NETWORK.which_model_netG = 'multiview_network_denseUNetFuse_transposed'
  # instance normalization or batch normalization
  opt.NETWORK.norm_G = 'batch3d'
  # no dropout for the generator
  opt.NETWORK.no_dropout = True
  # network initialization [normal|xavier|kaiming|orthogonal]
  opt.NETWORK.init_type = 'normal'
  # gan, lsgan, wgan, wgan_gp
  opt.NETWORK.ganloss = 'lsgan'
  # down sampling
  opt.NETWORK.n_downsampling = 4
  opt.NETWORK.n_blocks = 1
  # activation
  opt.NETWORK.activation_type = 'relu'

  '''
  Network
    Discriminator
  '''
  # of input image channels
  opt.NETWORK.input_nc_D = 1
  # of output image channels
  # opt.NETWORK.output_nc_D = 1
  # of discrim filters in first conv layer
  opt.NETWORK.ndf = 64
  # selects model to use for netD
  opt.NETWORK.which_model_netD = 'basic3d'
  # only used if which_model_netD==n_layers, dtype = int
  opt.NETWORK.n_layers_D = 3
  # instance normalization or batch normalization, dtype = str
  opt.NETWORK.norm_D = 'batch3d'
  # output channels of discriminator network, dtype = int
  opt.NETWORK.n_out_ChannelsD = 1
  opt.NETWORK.pool_size = 50
  opt.NETWORK.if_pool = False
  opt.NETWORK.num_D = 3
  # add condition to discriminator network
  opt.NETWORK.conditional_D = True

  # of input image channels
  opt.NETWORK.map_input_nc_D = 1
  # of discrim filters in first conv layer
  opt.NETWORK.map_ndf = 64
  # selects model to use for netD
  opt.NETWORK.map_which_model_netD = 'multi2d'
  # only used if which_model_netD==n_layers
  opt.NETWORK.map_n_layers_D = 3
  # instance normalization or batch normalization, dtype = str
  opt.NETWORK.map_norm_D = 'batch'
  # output channels of discriminator network, dtype = int
  opt.NETWORK.map_n_out_ChannelsD = 1
  opt.NETWORK.map_pool_size = 50
  opt.NETWORK.map_num_D = 3

  '''
  Train
  '''
  opt.TRAIN = EasyDict()
  # initial learning rate for adam
  opt.TRAIN.lr = 0.0002
  # momentum term of adam
  opt.TRAIN.beta1 = 0.5
  opt.TRAIN.beta2 = 0.9
  # if true, takes images in order to make batches, otherwise takes them randomly
  opt.TRAIN.serial_batches = False
  opt.TRAIN.batch_size = 1
  # threads for loading data
  opt.TRAIN.nThreads = 5
  # opt.TRAIN.max_epoch = 10
  # learning rate policy: lambda|step|plateau
  opt.TRAIN.lr_policy = 'lambda'
  # of iter at starting learning rate
  opt.TRAIN.niter = 100
  # of iter to linearly decay learning rate to zero
  opt.TRAIN.niter_decay = 100
  # multiply by a gamma every lr_decay_iters iterations
  opt.TRAIN.lr_decay_iters = 50
  # frequency of showing training results on console
  opt.TRAIN.print_freq = 10
  # frequency of showing training results on console
  opt.TRAIN.print_img_freq = 200
  # save model
  opt.TRAIN.save_latest_freq = 3000
  # save model frequent
  opt.TRAIN.save_epoch_freq = 5
  opt.TRAIN.begin_save_epoch = 0

  opt.TRAIN.weight_decay_if = False

  opt.TRAIN.use_lsgan=True

  opt.TRAIN.critic_times=1
  '''
  TEST
  '''
  opt.TEST = EasyDict()
  opt.TEST.howmany_in_train = 10

  '''
  Data
  Augmentation
  '''
  opt.DATA_AUG = EasyDict()
  opt.DATA_AUG.select_slice_num = 0
  opt.DATA_AUG.fine_size = 256
  opt.DATA_AUG.ct_channel = 256
  opt.DATA_AUG.xray_channel = 1
  opt.DATA_AUG.resize_size = 289

  '''
  2D GAN define loss
  '''
  opt.TD_GAN = EasyDict()
  # identity loss
  opt.TD_GAN.idt_lambda = 10.
  opt.TD_GAN.idt_reduction = 'elementwise_mean'
  opt.TD_GAN.idt_weight = 0.
  opt.TD_GAN.idt_weight_range = [0., 1.]
  opt.TD_GAN.restruction_loss = 'l1'
  # perceptual loss
  opt.TD_GAN.fea_m_lambda = 10.
  # output of discriminator
  opt.TD_GAN.discriminator_feature = False
  # wgan-gp
  opt.TD_GAN.wgan_gp_lambda = 10.
  # identity loss of map
  opt.TD_GAN.map_m_lambda = 0.
  # 'l1' or 'mse'
  opt.TD_GAN.map_m_type = 'l1'
  opt.TD_GAN.fea_m_map_lambda = 10.
  # Discriminator train times
  opt.TD_GAN.critic_times = 1

  '''
  3D GD-GAN define structure
  '''
  opt.D3_GAN = EasyDict()
  opt.D3_GAN.noise_len = 1000
  opt.D3_GAN.input_shape = [4,4,4]
  # opt.D3_GAN.input_shape_nc = 512
  # opt.D3_GAN.output_shape = [512,512,512]
  opt.D3_GAN.output_shape = [ISIZE,ISIZE,ISIZE]

  # opt.D3_GAN.output_shape_nc = 1
  # opt.D3_GAN.encoder_input_shape = [512, 512]
  opt.D3_GAN.encoder_input_shape = [ISIZE, ISIZE]

  opt.D3_GAN.encoder_input_nc = 1
  opt.D3_GAN.encoder_norm = 'instance'
  opt.D3_GAN.encoder_blocks = 3
  opt.D3_GAN.multi_view = [1,2,3]
  opt.D3_GAN.min_max_norm = False
  opt.D3_GAN.skip_number = 1
  # DoubleBlockLinearUnit Activation [low high k]
  opt.D3_GAN.dblu = [0., 1.0, 1.0]

  '''
  CT GAN
  '''
  opt.CTGAN = EasyDict()
  # input x-ray direction, 'H'-FrontBehind 'D'-UpDown 'W'-LeftRight
  # 'HDW' Means that deepness is 'H' and projection in plane of 'DW'
  #  relative to CT.

  # opt.CTGAN.Xray1_Direction = 'HDW'
  # opt.CTGAN.Xray2_Direction = 'WDH'

  opt.CTGAN.Xray1_Direction = '330'
  opt.CTGAN.Xray2_Direction = '000'
  opt.CTGAN.Xray3_Direction = '030'

  # dimension order of input CT is 'DHW'(should add 'NC'-01 to front when training)
  opt.CTGAN.CTOrder = [0, 1, 2, 3, 4]
  # NCHDW to xray1 and NCWDH to xray2


# #@@@@@@@@ Newer X-Ray Orientations 
# The 212 orientation
  opt.CTGAN.CTOrder_Xray1 = [0, 1, 4, 2, 3] 
  opt.CTGAN.CTOrder_Xray2 = [0, 1, 3, 2, 4]
  opt.CTGAN.CTOrder_Xray3 = [0, 1, 4, 2, 3]



# #@@@@@@@@ New X-Ray Orientations for Result 10 and 16 and 20, 21 and 22

# The 111 orientation

  # opt.CTGAN.CTOrder_Xray1 = [0, 1, 3, 2, 4]
  # opt.CTGAN.CTOrder_Xray2 = [0, 1, 3, 2, 4]
  # opt.CTGAN.CTOrder_Xray3 = [0, 1, 3, 2, 4]

#@@@@@@@@ Old X-Ray Orientations for Result 9 and below
  # opt.CTGAN.CTOrder_Xray1 = [0, 1, 3, 2, 4]
  # opt.CTGAN.CTOrder_Xray2 = [0, 1, 4, 2, 3]
  # opt.CTGAN.CTOrder_Xray3 = [0, 1, 3, 2, 4]


  # Spacer identity loss'weight
  opt.CTGAN.spc_idt_lambda = 30.
  opt.CTGAN.spc_idt_loss = 'l1'

  # Spacer Contrast loss'weight
  opt.CTGAN.spc_con_lambda = 30.
  opt.CTGAN.spc_con_loss = 'l1'

  # Spacer DICE loss'weight
  opt.CTGAN.spc_dsc_lambda = 0.
  opt.CTGAN.spc_dsc_loss = 'l1'

  # Constrast Fidelity loss'weight
  opt.CTGAN.con_fid_lambda = 10.
  opt.CTGAN.con_fid_loss = 'l1'

  # identity loss'weight
  opt.CTGAN.idt_lambda = 10.
  opt.CTGAN.idt_reduction = 'mean'
  opt.CTGAN.idt_weight = 0.
  opt.CTGAN.idt_weight_range = [0., 1.]
  # 'l1' or 'mse'
  opt.CTGAN.idt_loss = 'l1'
  # feature metrics loss
  opt.CTGAN.feature_D_lambda = 0. ###############################################
  # projection loss'weight
  opt.CTGAN.map_projection_lambda = 10.
  # 'l1' or 'mse'
  opt.CTGAN.map_projection_loss = 'l1'
  # gan loss'weight
  opt.CTGAN.gan_lambda = 0.1
  # multiView GAN auxiliary loss
  opt.CTGAN.auxiliary_lambda = 0.
  # 'l1' or 'mse'
  opt.CTGAN.auxiliary_loss = 'mse'
  # map discriminator
  opt.CTGAN.feature_D_map_lambda = 10.
  opt.CTGAN.map_gan_lambda = 1.0
  return opt






