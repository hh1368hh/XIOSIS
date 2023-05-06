from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler
import torch.nn as nn
from utl.Load_CT import LoadCT
import dicom2nifti
import os
from pathlib import Path
import os
import numpy as np
import matplotlib.pyplot as plt
import imageio
import glob
import nibabel as nib
from utl.dset import CNNDataset, RandomCrop3D, Permute, Gains, RandomZoom, ToTensor
import torch
from utl.resunetHH import UNet
from torchvision import transforms
import math
from utl.prepcache import HHPrepCacheApp
from ipywidgets import interact, fixed
from utl.Pix2pixGANHHsig import Pix2PixModel
import os 


class UnetTrainingApp:
    def __init__(self, valid_split = 0.1, batch_size = 1, n_jobs = 0, n_epochs = 200, caching=True, train_dir=None ):

        self.valid_split=valid_split
        self.batch_size=batch_size
        self.n_jobs=n_jobs
        self.n_epochs=n_epochs
        self.caching=caching
        self.train_dir=train_dir
        device = (torch.device('cuda') if torch.cuda.is_available()
            else torch.device('cpu'))
        print(f"Training on {device}.")

        assert torch.cuda.is_available()
        self.device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True
        self.use_cuda=torch.cuda.is_available()
        torch.cuda.empty_cache()

        self.model=self.initModel()
        # self.optimizer = self.initOptimizer()

        

    def initModel(self):
        # model=UNet(in_channels=1, n_classes=3, batch_norm=True, up_mode='upsample',depth=3, padding=1,wf=4)
        model = Pix2PixModel()

        
        # if torch.cuda.device_count() > 1:
        #   print("Let's use", torch.cuda.device_count(), "GPUs!")
        #   # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        #   model = nn.DataParallel(model)
        # if self.use_cuda:
        #     model = model.to(self.device)

        return model

    def initOptimizer(self):
        return torch.optim.AdamW(self.model.parameters(), weight_decay=1e-6)
        # return torch.optim.Adagrad(self.model.parameters())


    def initTrainDataset(self,tfms=None):
        # self.train_dir = 'Data_proj'
        # tfms = transforms.Compose([Permute(0,2) ,RandomCrop3D((24,120,120),3),ToTensor()])
        # tfms = transforms.Compose([RandomCrop3D((128,128,128),4) ,ToTensor()])
        
        train_ds = CNNDataset(self.train_dir,tfms)
        s,t=train_ds[0]
        print(s.shape)
        print(t.shape)
        self.train_ds=train_ds
        # print(train_ds.datainfolist)
        if (self.caching):
            self.doCaching(tfms)

       # if self.use_cuda:
        #    self.batch_size *= torch.cuda.device_count()

        num_train = len(train_ds)
        indices = list(range(num_train))
        split = int(math.ceil(self.valid_split * num_train))
        valid_idx = np.random.choice(indices, size=split, replace=False)
        train_idx = list(set(indices) - set(valid_idx))
        train_sampler = SubsetRandomSampler(train_idx)
        # valid_sampler = SubsetRandomSampler(valid_idx)
        train_loader = DataLoader(train_ds, sampler=train_sampler, batch_size=self.batch_size,
                                num_workers=self.n_jobs, pin_memory=self.use_cuda)
        # valid_loader = DataLoader(train_ds, sampler=valid_sampler, batch_size=batch_size,
        #                         num_workers=n_jobs, pin_memory=use_cuda)
        self.valid_idx=valid_idx
        self.train_idx=train_idx
        return train_loader


    def doCaching(self,tfms):
        # import shutil
 
        # def cleanCache():
        #     shutil.rmtree('data-unversioned/cache')
        #     os.mkdir('data-unversioned/cache')

        CachePrep=HHPrepCacheApp(batch_size=self.batch_size,num_workers=self.n_jobs,data_dir=self.train_dir, tfm=tfms)

        CachePrep.main()


    def computeBatchLoss(self, batch_ndx, batch_tup):
        criterion = nn.SmoothL1Loss()


        src, tgt ,thr =batch_tup
        src_g = src.to(self.device, non_blocking=True)
        tgt_g = tgt.to(self.device, non_blocking=True)
        thr_g = thr.to(self.device, non_blocking=True)
        
        out_g=self.model(src_g)
        
        print("Outside: input size", src_g.size(),
          "output_size", out_g.size())
        
        # ww=torch.Tensor.repeat(src_g>thr_g,[1,3,1,1,1])
        # ww=src_g>thr_g
        # tgt_g = tgt_g*torch.Tensor.repeat(src_g>torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(thr_g,-1),-1),-1),-1),[1,3,1,1,1])
        loss = criterion(
        tgt_g[torch.Tensor.repeat(src_g>torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(thr_g,-1),-1),-1),-1),[1,3,1,1,1])],
        out_g[torch.Tensor.repeat(src_g>torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(thr_g,-1),-1),-1),-1),[1,3,1,1,1])])


        # loss = criterion(out_g, tgt_g)
        
        return loss



    def doTraining(self, train_loader):
        self.model.netG.train(True)
        self.model.netD.train(True)
        for batch_ndx, batch_tup in enumerate(train_loader):
            # print(len(batch_tup))
            self.model.set_input(batch_tup)         # unpack data from dataset and apply preprocessing
            self.model.optimize_parameters()   # calculate loss functions, get gradients, update network weights





    def main(self,tfms,NNFileName):

        train_loader=self.initTrainDataset(tfms)
        torch.backends.cudnn.benchmark = True

        for t in range(1, self.n_epochs + 1):
            self.doTraining(train_loader)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            log = f'Epoch: {t}' + f'  Training Loss: {self.model.loss_G.item()}'
            print(log)
        
        self.saveModel(NNFileName)


    def saveModel(self,NNFileName):
 
        savefolder="./ResultCNN"
        savefolderbase=savefolder
        folder_exist=True
        i=0
        while folder_exist:
            try:
                os.makedirs(savefolder)
                folder_exist=False
            except:
                i=i+1
                savefolder=savefolderbase + '_' + str(i)
 
        # os.mkdir(savefolder,mode = 0o666) 
        torch.save(self.model.netG.state_dict(), savefolder + '/' + NNFileName[0])
        torch.save(self.model.netD.state_dict(), savefolder + '/' + NNFileName[1])
        print("Training is Done!!!!!!!!!!!!!!!")




    def dataVisualize(self,tfms,idx):
        def display_images_with_alpha_numpy(image_z, alpha, fixed, moving):
            img = (1.0 - alpha)*fixed[image_z,:,:] + alpha*moving[image_z,:,:] 
            plt.figure(figsize=(16,9))
            plt.imshow(img,cmap=plt.cm.Greys_r);
            plt.axis('off')
            plt.show()
        train_dir  = 'Data'
        train_ds = CNNDataset(train_dir,tfms)
        s,t,thr=train_ds[idx]
        print(s.shape)
        print(s.shape)
        
        interact(display_images_with_alpha_numpy, image_z=(0,s[0,:,:,:].numpy().shape[0] - 1), alpha=(0.0,1.0,0.001), fixed = fixed(s[0,:,:,:].numpy()), moving=fixed(t[0,:,:,:].numpy()));

    def dataVisualize_proj(self,tfms,idx):
        train_dir  = self.train_dir
        train_ds = CNNDataset(train_dir,tfms)
        s,t=train_ds[idx]
        print(s.shape)
        print(t.shape)
        plt.figure(figsize=(20, 60))
        plt.subplot(3,3,1)
        plt.imshow(s[0,:,:],cmap='gray_r')
        plt.subplot(3,3,2)
        plt.imshow(s[1,:,:],cmap='gray_r')
        plt.subplot(3,3,3)
        plt.imshow(s[2,:,:],cmap='gray_r')
        
        plt.subplot(3,3,4)
        plt.imshow(s[3,:,:],cmap='gray_r')
        plt.subplot(3,3,5)
        plt.imshow(s[4,:,:],cmap='gray_r')
        plt.subplot(3,3,6)
        plt.imshow(s[5,:,:],cmap='gray_r')

        plt.subplot(3,3,7)
        plt.imshow(t[0,:,:],cmap='gray_r')
        plt.subplot(3,3,8)
        plt.imshow(t[1,:,:],cmap='gray_r')
        plt.subplot(3,3,9)
        plt.imshow(t[2,:,:],cmap='gray_r')

        # plt.subplot(3,3,8)
        # plt.imshow(np.abs(t[0,:,:]),cmap='gray')




