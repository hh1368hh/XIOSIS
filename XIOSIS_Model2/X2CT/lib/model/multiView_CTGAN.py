# ------------------------------------------------------------------------------
# Copyright (c) Tencent
# Licensed under the GPLv3 License.
# Created by Kai Ma (makai0324@gmail.com)
# ------------------------------------------------------------------------------

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import re



import torch
import numpy as np
from .base_model import Base_Model
from .nets import factory as factory
from torch import tensor
from .loss.multi_gan_loss import GANLoss, RestructionLoss, FocalTverskyLoss
from ..utils.image_pool import ImagePool
from ..utils import metrics as Metrics
from ....utl.dset import projector
from torchvision.transforms.functional import rotate as TensorRotate
from ....OutPaint.utl.Training_OUTPAINT import UnetTrainingApp as outpaintnetclass
# import SimpleITK as sitk

import matplotlib.pyplot as plt
from ipywidgets import interact, fixed

def display_images_with_alpha_numpy(image_z, alpha, fixed, moving):
    img = (1.0 - alpha)*fixed[image_z,:,:] + alpha*moving[image_z,:,:] 
    plt.figure(figsize=(16,9))
    plt.imshow(img,cmap=plt.cm.Greys_r);
    plt.axis('off')
    plt.show()



class CTGAN(Base_Model):
  def __init__(self):
    super(CTGAN, self).__init__()

  @property
  def name(self):
    return 'multiView_CTGAN'

  '''
  Init network architecture
  '''
  def init_network(self, opt):
    Base_Model.init_network(self, opt)

    self.if_pool = opt.NETWORK.if_pool
    self.multi_view = opt.D3_GAN.multi_view
    self.conditional_D = opt.NETWORK.conditional_D
    self.auxiliary_loss = False
    assert len(self.multi_view) > 0

    self.loss_names = ['D', 'G']
    self.metrics_names = ['Mse', 'CosineSimilarity', 'PSNR']
    self.visual_names = ['G_real', 'G_fake', 'G_input1', 'G_input2', 'G_Map_fake_F', 'G_Map_real_F', 'G_Map_fake_S', 'G_Map_real_S']

    # identity loss
    if self.opt.CTGAN.idt_lambda > 0:
      self.loss_names += ['idt']

    # spacer identity loss
    if self.opt.CTGAN.spc_idt_lambda > 0:
      self.loss_names += ['spc_idt']

    if self.opt.CTGAN.spc_dsc_lambda > 0:
      self.loss_names += ['spc_dsc']

    if self.opt.CTGAN.spc_con_lambda > 0:
      self.loss_names += ['spc_con']

    # Contrant fidelity loss
    if self.opt.CTGAN.con_fid_lambda > 0:
      self.loss_names += ['con_fid']

    # # Contrant fidelity loss Lung
    # if self.opt.CTGAN.con_fid_lambda > 0:
    #   self.loss_names += ['con_fid_Lung']

    # feature metric loss
    if self.opt.CTGAN.feature_D_lambda > 0:
      self.loss_names += ['fea_m']

    # map loss
    if self.opt.CTGAN.map_projection_lambda > 0:
      self.loss_names += ['map_m']

    # auxiliary loss
    if self.opt.CTGAN.auxiliary_lambda > 0:
      self.loss_names += ['auxiliary']
      self.auxiliary_loss = True
    print(self.loss_names)
    if self.training:
      self.model_names = ['G', 'D']
    else:  # during test time, only load Gs
      self.model_names = ['G']
    # print(self.training)
    self.netG = factory.define_3DG(opt.D3_GAN.noise_len, opt.D3_GAN.input_shape, opt.D3_GAN.output_shape,
                                   opt.NETWORK.input_nc_G, opt.NETWORK.output_nc_G, opt.NETWORK.ngf, opt.NETWORK.which_model_netG,
                                   opt.NETWORK.n_downsampling, opt.NETWORK.norm_G, not opt.NETWORK.no_dropout,
                                   opt.NETWORK.init_type, self.gpu_ids, opt.NETWORK.n_blocks,
                                   opt.D3_GAN.encoder_input_shape, opt.D3_GAN.encoder_input_nc, opt.D3_GAN.encoder_norm,
                                   opt.D3_GAN.encoder_blocks, opt.D3_GAN.skip_number, opt.NETWORK.activation_type, opt=opt)

    if self.training:
      # out of discriminator is not probability when
      # GAN loss is LSGAN
      use_sigmoid = False

      # conditional Discriminator
      if self.conditional_D:
        print('Conditional Training is On')
        d_input_channels = opt.NETWORK.input_nc_D + 3
      else:
        d_input_channels = opt.NETWORK.input_nc_D
      self.netD = factory.define_D(d_input_channels, opt.NETWORK.ndf,
                                   opt.NETWORK.which_model_netD,
                                   opt.NETWORK.n_layers_D, opt.NETWORK.norm_D,
                                   use_sigmoid, opt.NETWORK.init_type, self.gpu_ids,
                                   opt.TD_GAN.discriminator_feature, num_D=opt.NETWORK.num_D, n_out_channels=opt.NETWORK.n_out_ChannelsD)
      if self.if_pool:
        self.fake_pool = ImagePool(opt.pool_size)


    OutPaint=outpaintnetclass(valid_split = 0.1, batch_size = 3, n_jobs = 0, n_epochs = 30, caching=False, train_dir='Data')
    self.netP=OutPaint.model.netG
    # print(self.Netoutpaint)

    


  
  # correspond to visual_names
  def get_normalization_list(self):
    return [
      [self.opt.CT_MEAN_STD[0], self.opt.CT_MEAN_STD[1]],
      [self.opt.CT_MEAN_STD[0], self.opt.CT_MEAN_STD[1]],
      [self.opt.XRAY1_MEAN_STD[0], self.opt.XRAY1_MEAN_STD[1]],
      [self.opt.XRAY2_MEAN_STD[0], self.opt.XRAY2_MEAN_STD[1]],
      [self.opt.CT_MEAN_STD[0], self.opt.CT_MEAN_STD[1]],
      [self.opt.CT_MEAN_STD[0], self.opt.CT_MEAN_STD[1]],
      [self.opt.CT_MEAN_STD[0], self.opt.CT_MEAN_STD[1]],
      [self.opt.CT_MEAN_STD[0], self.opt.CT_MEAN_STD[1]]
    ]

  def init_loss(self, opt):
    Base_Model.init_loss(self, opt)

    # #####################
    # define loss functions
    # #####################
    # GAN loss
    self.criterionGAN = GANLoss(use_lsgan=opt.TRAIN.use_lsgan).to(self.device)

    # identity loss
    self.criterionIdt = RestructionLoss(opt.CTGAN.idt_loss, opt.CTGAN.idt_reduction).to(self.device)

    # Spacer Dice Loss
    self.criterionSdl = torch.nn.BCELoss().to(self.device)

    # Spacer FocalTverskyLoss
    self.criterionSpcFTL=FocalTverskyLoss().to(self.device)

    # Spacer identity loss
    self.criterionSpcIdt = RestructionLoss(opt.CTGAN.spc_idt_loss, 'mean').to(self.device)

    # Spacer Contrast loss
    self.criterionSpcCon = RestructionLoss(opt.CTGAN.spc_con_loss, opt.CTGAN.idt_reduction).to(self.device)

    # Contrast fidelity loss
    self.criterionConFid = RestructionLoss(opt.CTGAN.con_fid_loss, opt.CTGAN.idt_reduction).to(self.device)

    # feature metric loss
    self.criterionFea = torch.nn.L1Loss().to(self.device)

    # map loss
    self.criterionMap = RestructionLoss(opt.CTGAN.map_projection_loss).to(self.device)

    # auxiliary loss
    self.criterionAuxiliary = RestructionLoss(opt.CTGAN.auxiliary_loss).to(self.device)

    # #####################
    # initialize optimizers
    #####################
    # self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
    #                                     lr=opt.TRAIN.lr, betas=(opt.TRAIN.beta1, opt.TRAIN.beta2))
    # self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
    #                                     lr=opt.TRAIN.lr, betas=(opt.TRAIN.beta1, opt.TRAIN.beta2))
    # opt.lr=0.0001
    opt.lr=0.001
    
    opt.beta1=0.9
    epsilon1=10**-8
    self.optimizer_G = torch.optim.AdamW(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.99),eps=epsilon1)
    self.optimizer_D = torch.optim.AdamW(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.99),eps=epsilon1)
    self.optimizer_P = torch.optim.AdamW(self.netP.parameters(), lr=opt.lr, betas=(opt.beta1, 0.99),eps=epsilon1)

    self.optimizers = []
    self.optimizers.append(self.optimizer_G)
    self.optimizers.append(self.optimizer_D)

  '''
    Train -Forward and Backward
  '''
  def set_input(self, input):
    # s,l,t,spc=input
    # print(s.shape)

    self.P_input = input[0].to(self.device, non_blocking=True)
    self.L_input = input[1].to(self.device)
    # print(input[1].shape)
    # self.G_input1 = s[:,0,:,:].unsqueeze(1).to(self.device)
    # self.G_input2 = s[:,1,:,:].unsqueeze(1).to(self.device)
    # self.G_input3 = s[:,2,:,:].unsqueeze(1).to(self.device)
    # print(self.G_input3.shape)
    self.G_real = input[2].to(self.device)
    self.SP_mask= input[3].to(self.device)
    # print(self.G_real.shape)
    # self.image_paths = input[2:]


  



  def ConLoss(self, fake, real, dim=(2,3,4), epsilon=1):
    # print(fake.isnan().any())
    # print(real_label.isnan().any())
    # def dice(A,B,dim=(2,3,4),eps=1e-6):
    #   intersection = torch.sum(A * B, dim)
    #   cardinality = torch.sum(A + B, dim)
    #   dice_score = 2. * intersection / (cardinality + eps)
    #   return dice_score

    # m = torch.nn.Threshold(2500., 0.)
    # fake_label=torch.where(m(fake)>0.,1.,0.)
    # fake_label=fake/2500
    # m = torch.nn.Threshold(2500, 0)
    # r_label=torch.where(m(real)>0,1.,0.)
    # real_label=real
    # print(fake_label.dtype)
    # fake_label=torch.sigmoid((fake-2300)/25)*3000
    # real_label=torch.sigmoid((real-2300)/25)*3000
    
    # import SimpleITK as sitk
    # sitk.Show(sitk.GetImageFromArray(real_label[0,0,:].detach().cpu()))
    # sitk.Show(sitk.GetImageFromArray(fake_label[0,0,:].detach().cpu()))

    # import SimpleITK as sitk
    # sitk.Show(sitk.GetImageFromArray(torch.cat((real_label[0,0,:].detach().cpu(),fake_label[0,0,:].detach().cpu()),dim=2)))
    
    # dice = (2 * (fake_label*real_label).sum(dim=[2,3,4])+epsilon)/(fake_label.sum(dim=[2,3,4]) + real_label.sum(dim=[2,3,4])+epsilon)
    # dice = (2 * (fake_label*real_label).sum(dim=[2,3,4])+epsilon)/(fake_label.sum(dim=[2,3,4]) + real_label.sum(dim=[2,3,4])+epsilon)
    # print(dice)

    # print(torch.sum(real_label,dim)==0.)d
    # print(fake_label.shape)
    # # r_label=real_label
    
    # intersection = torch.sum(fake_label * real_label, dim)
    # cardinality = torch.sum(fake_label, dim) + torch.sum(real_label, dim)
    # dice = 2. * intersection / (cardinality + 1e-6)

    # self.criterionConFid

    # print(torch.sum(real_label,dim))
    # print(torch.sum(real_label,dim)==0.)
    # dice[torch.sum(real_label,dim)==0.]=1.

    # print(dice)



    # print((1 - dice)*30)
    # return (1 - dice).mean() 
    return self.criterionSpcCon(torch.sigmoid((fake-2300)/25)*3000,torch.sigmoid((real-2300)/25)*3000)



  def spacer_enhance(self, img):
    # out=(1/(1 + torch.exp(-(img-2000)/20)))*3000
    # out=(1/(1 + torch.exp(-(img-2200)/100)))*3000
    out=(1/(1 + torch.exp(-(img-1500)/25)))*3000
    
    
    # out=(1/(1 + torch.exp(-(img-1050)/100)))*3000
    # out=(1/(1 + torch.exp(-(img-2000)/20)))+(1/(1 + torch.exp(-(-img+3000)/20)))
    return out

  def constrast_enhance_L(self, img):
    out=(1/(1 + torch.exp(-(img-500)/500)))*3000
    return out

  # Constrast Enhancement Function
  def constrast_enhance(self, img):
    out=(1/(1 + torch.exp(-(img-1050)/100)))*3000
    return out

  # map function
  def output_map(self, v, dim):
    '''
    :param v: tensor
    :param dim:  dimension be reduced
    :return:
      N1HW
    '''
    ori_dim = v.dim()
    # tensor [NDHW]
    if ori_dim == 4:
      map = torch.mean(torch.abs(v), dim=dim)
      # [NHW] => [NCHW]
      return map.unsqueeze(1)
    # tensor [NCDHW] and c==1
    elif ori_dim == 5:
      # [NCHW]
      map = torch.mean(torch.abs(v), dim=dim)
      return map
    else:
      raise NotImplementedError()

  def output_map_add(self, v, dim):
    ''' 
    :param v: tensor
    :param dim:  dimension be reduced
    :return:
      N1HW
    '''
    ori_dim = v.dim()
    # tensor [NDHW]
    if ori_dim == 4:
      map = torch.sum(torch.abs((v-1000)/1000*0.2 + 0.2), dim=dim)
      # [NHW] => [NCHW]
      return map.unsqueeze(1)
    # tensor [NCDHW] and c==1
    elif ori_dim == 5:
      # [NCHW]
      map = torch.sum(torch.abs((v-1000)/1000*0.2 + 0.2), dim=dim)
      return map
    else:
      raise NotImplementedError()

  def output_map_max(self, v, dim):
    ''' 
    :param v: tensor
    :param dim:  dimension be reduced
    :return:
      N1HW
    '''
    ori_dim = v.dim()
    # tensor [NDHW]
    if ori_dim == 4:
      map,indx = torch.max(v, dim=dim)
      # [NHW] => [NCHW]
      return map.unsqueeze(1)
    # tensor [NCDHW] and c==1
    elif ori_dim == 5:
      # [NCHW]
      map,indx = torch.max(v, dim=dim)
      return map
    else:
      raise NotImplementedError()



      # map function
  def output_map_angle(self, v, dim, angle):
    '''
    :param v: tensor
    :param dim:  dimension be reduced
    :return:
      N1HW
    '''
    # vp=v-1000
    # vp[vp<-1000]=-1000
    # vp=(vp/1000)*0.2+0.2
    # torch.ro
    # vp=TensorRotate(v.squeeze(0),angle=angle).unsqueeze(0)
    ori_dim = v.dim()
    # tensor [NDHW]
    if ori_dim == 4:
      map = torch.mean(torch.abs(TensorRotate(v[j,:],angle=angle)), dim=dim)
      # [NHW] => [NCHW]
      return map.unsqueeze(1)
    # tensor [NCDHW] and c==1
    elif ori_dim == 5:
      # [NCHW]
      print(v.shape)

      map = torch.mean(torch.abs(TensorRotate(v.squeeze(1),angle=angle)), dim=dim)
      
      return map
    else:
      raise NotImplementedError()

  def transition(self, predict):
    p_max, p_min = predict.max(), predict.min()
    new_predict = (predict - p_min) / (p_max - p_min)
    return new_predict

  def ct_unGaussian(self, value):
    return value * self.opt.CT_MEAN_STD[1] + self.opt.CT_MEAN_STD[0]

  def ct_Gaussian(self, value):
    return (value - self.opt.CT_MEAN_STD[0]) / self.opt.CT_MEAN_STD[1]

  def post_process(self, attributes_name):
    if not self.training:
      if self.opt.CT_MEAN_STD[0] == 0 and self.opt.CT_MEAN_STD[0] == 0:
        for name in attributes_name:
          setattr(self, name, torch.clamp(getattr(self, name), 0, 1))
      elif self.opt.CT_MEAN_STD[0] == 0.5 and self.opt.CT_MEAN_STD[0] == 0.5:
        for name in attributes_name:
          setattr(self, name, torch.clamp(getattr(self, name), -1, 1))
      else:
        raise NotImplementedError()

  def projection_visual(self):
    # map F is projected in dimension of H
    self.G_Map_real_F = self.transition(self.output_map(self.ct_unGaussian(self.G_real), 2))
    self.G_Map_fake_F = self.transition(self.output_map(self.ct_unGaussian(self.G_fake), 2))
    # map S is projected in dimension of W
    self.G_Map_real_S = self.transition(self.output_map(self.ct_unGaussian(self.G_real), 3))
    self.G_Map_fake_S = self.transition(self.output_map(self.ct_unGaussian(self.G_fake), 3))

  def metrics_evaluation(self):
    # 3D metrics including mse, cs and psnr
    g_fake_unNorm = self.ct_unGaussian(self.G_fake)
    g_real_unNorm = self.ct_unGaussian(self.G_real)

    self.metrics_Mse = Metrics.Mean_Squared_Error(g_fake_unNorm, g_real_unNorm)
    self.metrics_CosineSimilarity = Metrics.Cosine_Similarity(g_fake_unNorm, g_real_unNorm)
    self.metrics_PSNR = Metrics.Peak_Signal_to_Noise_Rate(g_fake_unNorm, g_real_unNorm, PIXEL_MAX=1.0)

  def dimension_order_std(self, value, order):
    # standard CT dimension
    return value.permute(*tuple(np.argsort(order)))

  def forward(self):
    '''
    self.G_fake is generated object
    self.G_real is GT object
    '''
    # G_fake_D is [B 1 D H W]
    # a=self.netG([self.G_input1, self.G_input2, self.G_input3])
    # print(len(a))
    # asd
    P_out=self.netP(self.P_input)
    # print(P_out.shape)
    self.G_input1 = P_out[:,0,:,:].unsqueeze(1).to(self.device)
    self.G_input2 = P_out[:,1,:,:].unsqueeze(1).to(self.device)
    self.G_input3 = P_out[:,2,:,:].unsqueeze(1).to(self.device)
    
    self.G_fake_D1, self.G_fake_D2, self.G_fake_D3, self.G_fake_D = self.netG([self.G_input1, self.G_input2, self.G_input3])
    # visual object should be [B D H W]
   
    self.G_fake = torch.squeeze(self.G_fake_D, 1)
    # print('hey hey', self.G_fake.shape)
    self.G_real_D = self.G_real


    # input of Discriminator is [B 1 D H W]
    # self.G_real_D = torch.unsqueeze(self.G_real, 1)
    # print('hey hey', self.G_real_D.shape)

    # print(self.auxiliary_loss)
    # if add auxiliary loss to generator
    if self.auxiliary_loss and self.training:
      self.G_real_D1 = self.G_real_D.permute(*self.opt.CTOrder_Xray1).detach()
      self.G_real_D2 = self.G_real_D.permute(*self.opt.CTOrder_Xray2).detach()
      self.G_real_D3 = self.G_real_D.permute(*self.opt.CTOrder_Xray3).detach()
    # if add condition to discriminator, expanding x-ray
    # as the same shape and dimension order as CT

    # if self.conditional_D and self.training:
    #   self.G_condition_D = torch.cat((
    #     self.dimension_order_std(self.G_input1.unsqueeze(1).expand_as(self.G_real_D), self.opt.CTGAN.CTOrder_Xray1),
    #     self.dimension_order_std(self.G_input2.unsqueeze(1).expand_as(self.G_real_D), self.opt.CTGAN.CTOrder_Xray2),
    #     self.dimension_order_std(self.G_input3.unsqueeze(1).expand_as(self.G_real_D), self.opt.CTGAN.CTOrder_Xray3)
    #   ), dim=1).detach()

    # print(self.training)
    # if self.conditional_D and self.training:
    #   self.G_condition_D = torch.cat((
    #     self.dimension_order_std(self.L_input[0,0,:].unsqueeze(0).unsqueeze(0).unsqueeze(1).expand_as(self.G_real_D), self.opt.CTGAN.CTOrder_Xray1),
    #     self.dimension_order_std(self.L_input[0,1,:].unsqueeze(0).unsqueeze(0).unsqueeze(1).expand_as(self.G_real_D), self.opt.CTGAN.CTOrder_Xray2),
    #     self.dimension_order_std(self.L_input[0,2,:].unsqueeze(0).unsqueeze(0).unsqueeze(1).expand_as(self.G_real_D), self.opt.CTGAN.CTOrder_Xray3)
    #   ), dim=1).detach()

    if self.conditional_D and self.training:
      self.G_condition_D = torch.cat((
        self.dimension_order_std(self.P_input[0,0,:].unsqueeze(0).unsqueeze(0).unsqueeze(1).expand_as(self.G_real_D), self.opt.CTGAN.CTOrder_Xray1),
        self.dimension_order_std(self.P_input[0,2,:].unsqueeze(0).unsqueeze(0).unsqueeze(1).expand_as(self.G_real_D), self.opt.CTGAN.CTOrder_Xray2),
        self.dimension_order_std(self.P_input[0,4,:].unsqueeze(0).unsqueeze(0).unsqueeze(1).expand_as(self.G_real_D), self.opt.CTGAN.CTOrder_Xray3)
      ), dim=1).detach()


    # print(self.G_input1.shape)
    # print(self.L_input[0,0,:].unsqueeze(0).unsqueeze(0).shape)

    # # print(self.G_condition_D.shape)
    # import SimpleITK as sitk
    # sitk.Show(sitk.GetImageFromArray(np.moveaxis(self.G_real_D[0,0,:].cpu().detach().numpy(),2,0)))
    # ss=np.moveaxis(self.G_condition_D[0,0,:].cpu().detach().numpy(),2,0)
    # sitk.Show(sitk.GetImageFromArray(ss))
    # asd
      # self.G_condition_D = torch.cat((
      #   self.dimension_order_std(self.G_input1.unsqueeze(1).expand_as(self.G_real_D), self.opt.CTGAN.CTOrder_Xray1),
      #   self.dimension_order_std(self.G_input2.unsqueeze(1).expand_as(self.G_real_D), self.opt.CTGAN.CTOrder_Xray2),
      #   self.dimension_order_std(self.G_input3.unsqueeze(1).expand_as(self.G_real_D), self.opt.CTGAN.CTOrder_Xray3)
      # ), dim=1).detach()
    # print(self.G_condition_D.shape)
    # gcd=np.moveaxis(self.G_condition_D.cpu().numpy(),2,3)
    # gct=np.moveaxis(self.G_real_D.cpu().numpy(),2,3)
    # interact(display_images_with_alpha_numpy, image_z=(0,gcd[0,0,:].shape[0] - 1), alpha=(0.0,1.0,0.05), fixed = fixed(gcd[0,0,:]), moving=fixed(gct[0,0,:]));
    # interact(display_images_with_alpha_numpy, image_z=(0,gcd[0,0,:].shape[0] - 1), alpha=(0.0,1.0,0.05), fixed = fixed(gcd[0,1,:]), moving=fixed(gcd[0,2,:]));
    # asdsd
    # post processing, used only in testing
    self.post_process(['G_fake'])
    if not self.training:
      # visualization of x-ray projection
      self.projection_visual()
      # metrics
      self.metrics_evaluation()
    # multi-view projection maps for training
    # Note: self.G_real_D and self.G_fake_D are in dimension order of 'NCDHW'
    if self.training:
      # out_p=torch.from_numpy(projector(self.G_real_D.squeeze(0).squeeze(0).cpu().numpy())).cuda()
      # print(out_p.shape)
      angles=[330,0,30]
      for i in self.multi_view:
        # out_map=out_p[i-1,:].unsqueeze(0).unsqueeze(0)
        # print(self.G_real_D.shape)

        #@@@@ New Rotation Version
        # out_map=self.output_map_max(TensorRotate(self.G_real_D.squeeze(1),angle=angles[i-1]).unsqueeze(1), 3)
        
        
        # print(out_map.shape)
        # if i==2:
        #   sitk.Show(sitk.GetImageFromArray(out_map.cpu().numpy().squeeze(0)))

        #@@@@ No change in mean and STD version
        out_map=self.output_map_max(self.G_real_D, i + 1)

        #@@@@ Original Version, Change in mean and std
        # out_map = self.output_map(self.ct_unGaussian(self.G_real_D), i + 1)
        # out_map = self.ct_Gaussian(out_map)

        setattr(self, 'G_Map_{}_real'.format(i), out_map)

        #@@@@ New Rotation Version
        # out_map=self.output_map_max(TensorRotate(self.G_fake_D.squeeze(1),angle=angles[i-1]).unsqueeze(1), 3)
        
        
        # if i==2:
        #   sitk.Show(sitk.GetImageFromArray(out_map.cpu().detach().numpy().squeeze(0)))

        #@@@@ No change in mean and STD version
        out_map=self.output_map_max(self.G_fake_D, i + 1)
        
        #@@@@ Original Version, Change in mean and std
        # out_map = self.output_map(self.ct_unGaussian(self.G_fake_D), i + 1)
        # out_map = self.ct_Gaussian(out_map)

        setattr(self, 'G_Map_{}_fake'.format(i), out_map)

  # feature metrics loss
  def feature_metric_loss(self, D_fake_pred, D_real_pred, loss_weight, num_D, feat_weights, criterionFea):
    fea_m_lambda = loss_weight
    loss_G_fea = 0
    feat_weights = feat_weights
    D_weights = 1.0 / num_D
    weight = feat_weights * D_weights

    # multi-scale discriminator
    if isinstance(D_fake_pred[0], list):
      for di in range(num_D):
        for i in range(len(D_fake_pred[di]) - 1):
          loss_G_fea += weight * criterionFea(D_fake_pred[di][i], D_real_pred[di][i].detach()) * fea_m_lambda
    # single discriminator
    else:
      for i in range(len(D_fake_pred) - 1):
        loss_G_fea += feat_weights * criterionFea(D_fake_pred[i], D_real_pred[i].detach()) * fea_m_lambda

    return loss_G_fea

  def backward_D_basic(self, D_network, input_real, input_fake, fake_pool, criterionGAN, loss_weight):
    D_real_pred = D_network(input_real)
    gan_loss_real = criterionGAN(D_real_pred, True)

    if self.if_pool:
      g_fake_pool = fake_pool.query(input_fake)
    else:
      g_fake_pool = input_fake
    D_fake_pool_pred = D_network(g_fake_pool.detach())
    gan_loss_fake = criterionGAN(D_fake_pool_pred, False)
    gan_loss = (gan_loss_fake + gan_loss_real) * loss_weight
    gan_loss.backward()
    return gan_loss

  def backward_D(self):
    if self.conditional_D:
      fake_input = torch.cat([self.G_condition_D, self.G_fake_D], 1)
      real_input = torch.cat([self.G_condition_D, self.G_real_D], 1)
    else:
      fake_input = self.G_fake_D
      real_input = self.G_real_D
    self.loss_D = self.backward_D_basic(self.netD, real_input, fake_input,self.fake_pool if self.if_pool else None, self.criterionGAN, self.opt.CTGAN.gan_lambda)

  def backward_G_basic(self, D_network, input_fake, criterionGAN, loss_weight):
    D_fake_pred = D_network(input_fake) 
    loss_G = criterionGAN(D_fake_pred, True) * loss_weight
    return loss_G, D_fake_pred

  def backward_G(self,retain_graph=False):
    idt_lambda = self.opt.CTGAN.idt_lambda
    spc_idt_lambda = self.opt.CTGAN.spc_idt_lambda
    spc_dsc_lambda = self.opt.CTGAN.spc_dsc_lambda
    spc_con_lambda = self.opt.CTGAN.spc_con_lambda
    con_fid_lambda = self.opt.CTGAN.con_fid_lambda
    fea_m_lambda = self.opt.CTGAN.feature_D_lambda
    map_m_lambda = self.opt.CTGAN.map_projection_lambda

    ############################################
    # BackBone GAN
    ############################################
    # GAN loss
    if self.conditional_D:
      fake_input = torch.cat([self.G_condition_D, self.G_fake_D], 1)
      real_input = torch.cat([self.G_condition_D, self.G_real_D], 1)
    else:
      fake_input = self.G_fake_D
      real_input = self.G_real_D
    
    # print(self.G_fake_D.shape)

    # print(fake_input.shape)

    self.loss_G, D_fake_pred = self.backward_G_basic(self.netD, fake_input, self.criterionGAN, self.opt.CTGAN.gan_lambda)

    # identity loss
    if idt_lambda > 0:
      # focus area weight assignment
      if self.opt.CTGAN.idt_reduction == 'none' and self.opt.CTGAN.idt_weight > 0:
        idt_low, idt_high = self.opt.CTGAN.idt_weight_range
        idt_weight = self.opt.CTGAN.idt_weight
        loss_idt = self.criterionIdt(self.G_fake_D, self.G_real_D)
        mask = (self.G_real_D > idt_low) & (self.G_real_D < idt_high)
        loss_idt[mask] = loss_idt[mask] * idt_weight
        self.loss_idt = loss_idt.mean() * idt_lambda
      else:
        self.loss_idt = self.criterionIdt(self.G_fake_D, self.G_real_D) * idt_lambda

    # print(self.criterionIdt(self.G_fake_D, self.G_real_D))
    # Contrast Fidelity loss
    if con_fid_lambda>0:

      self.loss_con_fid = self.criterionConFid(self.constrast_enhance(self.G_fake_D), self.constrast_enhance(self.G_real_D)) * con_fid_lambda
      self.loss_con_fid +=  self.criterionConFid(self.constrast_enhance_L(self.G_fake_D), self.constrast_enhance_L(self.G_real_D)) * con_fid_lambda

    if spc_con_lambda >0:
      # self.loss_spc_dsc = self.ConLoss(self.G_fake_D,self.G_real_D) * spc_con_lambda
      self.loss_spc_con = self.criterionSpcCon(self.spacer_enhance(self.G_fake_D), self.spacer_enhance(self.G_real_D))*100 * spc_con_lambda

    




    # Spacer DICE loss
    if spc_dsc_lambda > 0:
      # spacer area weight assignment
      # self.loss_spc_dsc = self.diceLoss(self.spacer_enhance(self.G_fake_D),self.SP_mask).mean() * spc_dsc_lambda

      # self.loss_spc_dsc = self.diceLoss(self.G_fake_D,self.SP_mask).mean() * spc_dsc_lambda


      ### Used this one for Old_Dicts, first it was 30 gain and then from 7 I increased to 50

      # self.loss_spc_dsc = self.diceLoss(self.G_fake_D,self.G_real_D) * spc_dsc_lambda
      
      
      # print(self.G_fake_D.shape)
      # print(self.SP_mask.shape)
      # self.loss_spc_dsc = self.diceLoss(self.G_fake_D,self.SP_mask) * spc_dsc_lambda
      self.loss_spc_dsc = self.criterionSpcFTL(self.spacer_enhance(self.G_fake_D),self.SP_mask) * spc_dsc_lambda
      # print(self.criterionSpcFTL(self.spacer_enhance(self.G_real_D),self.SP_mask)*50)

      # print(self.loss_spc_dsc)
      # self.loss_spc_dsc = self.criterionConFid(self.spacer_enhance(self.G_fake_D), self.spacer_enhance(self.G_real_D)) * spc_dsc_lambda
      # self.loss_spc_dsc=self.criterionConFid(self.spacer_enhance(self.G_fake_D), self.spacer_enhance(self.SP_mask)) * spc_dsc_lambda
      # loss_spc_idt = self.criterionSpcIdt(self.G_fake_D *self.SP_mask, self.G_real_D * self.SP_mask)
      # import SimpleITK
      # SimpleITK.Show(SimpleITK.GetImageFromArray((self.G_real_D *self.SP_mask).squeeze(0).squeeze(0).cpu().numpy()))
      # SimpleITK.Show(SimpleITK.GetImageFromArray((self.G_fake_D *self.SP_mask).squeeze(0).squeeze(0).detach().cpu().numpy()))
      # mask = (self.SP_mask>0.5)
      # loss_spc_idt[self.SP_mask>0.5] = loss_spc_idt[self.SP_mask>0.5] #* idt_weight
      # self.loss_spc_idt = loss_spc_idt.mean() * spc_idt_lambda
      # self.loss_spc_dsc = loss_spc_dsc.mean() * spc_dsc_lambda







    # Spacer identity loss
    if spc_idt_lambda > 0:
      # spacer area weight assignment
      # loss_spc_idt = self.criterionSpcIdt(self.G_fake_D[self.SP_mask>0.5], self.G_real_D[self.SP_mask>0.5])
      loss_spc_idt = self.criterionSpcIdt(self.G_fake_D * self.SP_mask, self.G_real_D * self.SP_mask)
      
      # print(loss_spc_idt)
      # loss_spc_idt = self.criterionSpcIdt(self.G_fake_D *self.SP_mask, self.G_real_D * self.SP_mask)
      # import SimpleITK
      # SimpleITK.Show(SimpleITK.GetImageFromArray((self.G_real_D *self.SP_mask).squeeze(0).squeeze(0).cpu().numpy()))
      # SimpleITK.Show(SimpleITK.GetImageFromArray((self.G_fake_D *self.SP_mask).squeeze(0).squeeze(0).detach().cpu().numpy()))
      # mask = (self.SP_mask>0.5)
      # loss_spc_idt[self.SP_mask>0.5] = loss_spc_idt[self.SP_mask>0.5] #* idt_weight
      # self.loss_spc_idt = loss_spc_idt.mean() * spc_idt_lambda
      self.loss_spc_idt = loss_spc_idt * spc_idt_lambda

    D_real_pred = self.netD(real_input)

    # feature metric loss
    if fea_m_lambda > 0:
      self.loss_fea_m = self.feature_metric_loss(D_fake_pred, D_real_pred, loss_weight=fea_m_lambda, num_D=self.opt.NETWORK.num_D, feat_weights=4.0 / (self.opt.NETWORK.n_layers_D + 1), criterionFea=self.criterionFea)

    # map loss
    if map_m_lambda > 0:
      self.loss_map_m = 0.
      for direction in self.multi_view:
        self.loss_map_m += self.criterionMap(
          getattr(self, 'G_Map_{}_fake'.format(direction)),
          getattr(self, 'G_Map_{}_real'.format(direction))) * map_m_lambda
      self.loss_map_m = self.loss_map_m / len(self.multi_view)

    # auxiliary loss
    if self.auxiliary_loss:
      self.loss_auxiliary = (self.criterionAuxiliary(self.G_fake_D1, self.G_real_D1) +
                             self.criterionAuxiliary(self.G_fake_D2, self.G_real_D2) +
                             self.criterionAuxiliary(self.G_fake_D3, self.G_real_D3)) * \
                            self.opt.auxiliary_lambda

    # 0.0 must be add to loss_total, otherwise loss_G will
    # be changed when loss_total_G change

    self.loss_total_G = self.loss_G + 0.0
    
    if idt_lambda > 0:
      # print(self.loss_total_G.shape)
      # print(self.loss_idt.shape)
      self.loss_total_G += self.loss_idt
      #print('Identity Loss is On')

    if spc_con_lambda>0:
      self.loss_total_G += self.loss_spc_con

    if spc_dsc_lambda>0:
      self.loss_total_G += self.loss_spc_dsc

    if con_fid_lambda>0:

      self.loss_total_G += self.loss_con_fid

      

    if spc_idt_lambda > 0:
      # print(self.loss_total_G.shape)
      # print(self.loss_idt.shape)
      self.loss_total_G += self.loss_spc_idt
      #print('Identity Loss is On')
    if fea_m_lambda > 0:
      self.loss_total_G += self.loss_fea_m
      #print('Feature Loss is On')
    if map_m_lambda > 0:
      self.loss_total_G += self.loss_map_m
      #print('Map Projection Loss is On')
    if self.auxiliary_loss:
      self.loss_total_G += self.loss_auxiliary
      #print('Auxiliary Loss is On')
    self.loss_total_G.backward(retain_graph=retain_graph)

  def optimize_parameters(self):
    # forward
    self.forward()


    # self.set_requires_grad([self.netP], True)
    # self.set_requires_grad([self.netG], False)
    # self.set_requires_grad([self.netD], False)
    # self.optimizer_P.zero_grad()
    # self.backward_G(retain_graph=True)
    # self.optimizer_P.step()

    self.set_requires_grad([self.netP], True)
    self.set_requires_grad([self.netG], True)
    self.set_requires_grad([self.netD], False)
    self.optimizer_P.zero_grad()
    self.optimizer_G.zero_grad()
    self.backward_G()
    self.optimizer_G.step()
    self.optimizer_P.step()

    self.set_requires_grad([self.netP], True)
    self.set_requires_grad([self.netG], True)
    self.set_requires_grad([self.netD], True)
    self.optimizer_D.zero_grad()
    self.backward_D()
    self.optimizer_D.step()

  def optimize_D(self):
    # forward
    self()
    self.set_requires_grad([self.netD], True)
    self.optimizer_D.zero_grad()
    self.backward_D()
    self.optimizer_D.step()


