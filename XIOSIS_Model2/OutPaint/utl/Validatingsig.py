from numpy.core.numeric import outer
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
from utl.dset import CNNDataset
from utl.dset import RandomCrop3D
from utl.dset import ToTensor
from utl.dset import Permute
import torch
# from utl.model import HH_Unet3
from utl.resunetHH import UNet
from torchvision import transforms


import dicom2nifti
import os
from pathlib import Path
import os
import numpy as np
import matplotlib.pyplot as plt
import imageio
import glob
import nibabel as nib
from itertools import chain
from util.imgsitk import *
import SimpleITK as sitk
import imagej
from ipywidgets import interact, fixed
import itk
from IPython.display import clear_output
from util.spineSeg import *
from util.spineSegC import *
from lungmask import mask
from ipywidgets import interact, fixed
# from util.valdset import *
from utl.dset import *
from util.imageprocessing import rescale
import torch.nn.functional as F
from utl.Pix2pixGANHHsig import Pix2PixModel
import os 
import astra
import torch
import torchvision.transforms as transforms

class UnetValidatingApp:

    def __init__(self,val_dir='Data'):

        device = (torch.device('cuda') if torch.cuda.is_available()
            else torch.device('cpu'))
        print(f"Testing on {device}.")

        self.val_dir_proj=val_dir
        assert torch.cuda.is_available()
        self.device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True
        self.use_cuda=torch.cuda.is_available()
        torch.cuda.empty_cache()
        self.model=self.initModel()
        self.initValDataset()
        self.smoothing=False
        self.patching=False

    def initModel(self):
        # model=UNet(in_channels=1, n_classes=3, batch_norm=True, up_mode='upsample',depth=3, padding=1,wf=4)
        model = Pix2PixModel()

        
        # if torch.cuda.device_count() > 0:
        #   print("Let's use", torch.cuda.device_count(), "GPUs!")
        #   # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        #   model = nn.DataParallel(model)
        # if self.use_cuda:
        #     model = model.to(self.device)
        return model

    def display_images_with_alpha_numpy(self,image_z, alpha, fixed, moving):
        img = (1.0 - alpha)*fixed[image_z,:,:] + alpha*moving[image_z,:,:] 
        plt.figure(figsize=(16,9))
        plt.imshow(img,cmap=plt.cm.Greys_r)
        plt.axis('off')
        plt.show()

    def initValDataset(self,tfms=None):
        # self.val_dir_proj = 'Data'
        self.datainfolist=getdatainfo(self.val_dir_proj)
        # self.val_dir_CT = 'DataNii'
        # self.datainfolist_CT=getdatainfo_CT(self.val_dir_CT)



    def getCase_CT(self,idx,angle):
        self.Case=self.datainfolist_CT[idx]
        #print(self.Case[0])
        # s=str(i).find('\P')
        # e=str(i).find('_CT')
        s=(self.Case[0]).find('\P')
        e=str(self.Case[0]).find('_CT')
        #print([s,e])
        casenum=int(str(self.Case[0])[s+3:e])
        print('the case number is ' + str(casenum))
        self.CT=get_full_CT(self.Case)
        projc, projp=getvaldata_CT(self.Case,angle)
        self.projc_col=projc
        self.projp_col=projp
    
    
    def getCase(self,idx):
        self.Case=self.datainfolist[idx]
        # print(self.Case)
        #print(self.Case[0])
        # s=str(i).find('\P')
        # e=str(i).find('_CT')
        # s=(self.Case[0]).find('\P')
        # e=str(self.Case[0]).find('_CT')
        #print([s,e])
        # casenum=int(str(self.Case[0])[s+3:e])
        print('the file name are \n')
        print('\n Small FOV \n') 
        print(self.Case.SFOVaddress)
        print('\n CTD \n')
        print(self.Case.CTDaddress)
        print('\n Large FOV \n')
        print(self.Case.LFOVaddress)
        proji, projo=getvaldata(self.Case)
        self.proji=proji
        self.projo=projo
    
    
    def doPatching(self,Crop_size):
        x = torch.from_numpy(self.image).unsqueeze(0)
        kc, kh, kw = Crop_size  # kernel size
        dc, dh, dw = Crop_size  # stride
        # Pad to multiples of 32
        # x = F.pad(x, (x.size(3)%kw // 2, x.size(3)%kw // 2,
        #               x.size(2)%kh // 2, x.size(2)%kh // 2,
        #               x.size(1)%kc // 2, x.size(1)%kc // 2),value=-1000)

        sp3l=((x.size(3)//kw +1)*kw -x.size(3)) // 2
        sp3r=((x.size(3)//kw +1)*kw -x.size(3)) // 2 + ((x.size(3)//kw +1)*kw -x.size(3))%2

        sp2l=((x.size(2)//kh +1)*kh -x.size(2)) // 2
        sp2r=((x.size(2)//kh +1)*kh -x.size(2)) // 2 + ((x.size(2)//kh +1)*kh -x.size(2))%2

        sp1l=((x.size(1)//kc +1)*kc -x.size(1)) // 2
        sp1r=((x.size(1)//kc +1)*kc -x.size(1)) // 2 + ((x.size(1)//kc +1)*kc -x.size(1))%2

        x = F.pad(x, (sp3l, sp3r,
                    sp2l, sp2r,
                    sp1l, sp1r),value=-1000)

        patches = x.unfold(1, kc, dc).unfold(2, kh, dh).unfold(3, kw, dw)
        self.unfold_shape = patches.size()
        self.patches = patches.contiguous().view(-1, kc, kh, kw)
        self.x=x
        self.patching=True

        # return patches


    def doValidation(self,dictaddress):

        torch.cuda.empty_cache()
        

        # model=HH_Unet3(in_channel=1, conv_channel=24, out_channel=3, kernel_size=3, up_mode='upsample').float()
        # model=UNet(in_channels=1, n_classes=3, batch_norm=True, up_mode='upsample',depth=3, padding=1,wf=4)

        # model.load_state_dict(torch.load( 'trained1_e200_p02_C50250250_Gs1t100.pth'))

        self.model.netG.load_state_dict(torch.load(dictaddress))

        self.model.netG.eval()
        # self.x=torch.cat(self.projc)
        self.x=torch.from_numpy(self.proji)
        with torch.no_grad():
            out = self.model.netG.forward(self.x.unsqueeze(0).to(self.device)).cpu().detach().squeeze(0)
        
        self.out=out.numpy()
        # self.out=out.numpy()
        # self.x_croped=self.x.squeeze(0).numpy()
        # self.out=out
        # return out
        



    def doValidation_CT(self,dictaddress):

        torch.cuda.empty_cache()
        

        # model=HH_Unet3(in_channel=1, conv_channel=24, out_channel=3, kernel_size=3, up_mode='upsample').float()
        # model=UNet(in_channels=1, n_classes=3, batch_norm=True, up_mode='upsample',depth=3, padding=1,wf=4)

        # model.load_state_dict(torch.load( 'trained1_e200_p02_C50250250_Gs1t100.pth'))

        self.model.netG.load_state_dict(torch.load(dictaddress))

        self.model.netG.eval()
        # self.x=torch.cat(self.projc)
        col=torch.from_numpy(self.projc_col).squeeze(0)
        outcol=torch.zeros_like(col)
        for i in range(col.shape[0]):
            with torch.no_grad():
                outcol[i,:,:] = self.model.netG.forward(col[i,:,:].unsqueeze(0).unsqueeze(0).to(self.device)).cpu().detach().squeeze(0).squeeze(0)
        
        self.outcol=outcol.numpy()
        # self.out=out.numpy()
        # self.x_croped=self.x.squeeze(0).numpy()
        # self.out=out
        # return out

    def reshapePatch(self):
        field=torch.zeros(3,self.x.shape[1],self.x.shape[2],self.x.shape[3])
        for hh in range(0,3):
            temp=self.out[:,hh,:,:,:]
            patches_orig = temp.view(self.unfold_shape)
            output_c = self.unfold_shape[1] * self.unfold_shape[4]
            output_h = self.unfold_shape[2] * self.unfold_shape[5]
            output_w = self.unfold_shape[3] * self.unfold_shape[6]
            patches_orig = patches_orig.permute(0, 1, 4, 2, 5, 3, 6).contiguous()
            patches_orig = patches_orig.view(1, output_c, output_h, output_w)
            field[hh,:,:,:]=patches_orig
            self.field=field
        # return field
    

    def fieldSmooting(self):
        import scipy
        field_smoothed=np.zeros_like(self.field.numpy())
        field_smoothed[0,:,:,:]=scipy.ndimage.gaussian_filter(self.field[0,:,:,:].numpy(), 5)
        field_smoothed[1,:,:,:]=scipy.ndimage.gaussian_filter(self.field[1,:,:,:].numpy(), 5)
        field_smoothed[2,:,:,:]=scipy.ndimage.gaussian_filter(self.field[2,:,:,:].numpy(), 5)
        self.field_smoothed=torch.from_numpy(field_smoothed)
        self.smoothing=True



    def crop_center(self,img,cropz,cropy,cropx,pad_val):
        z,y,x = img.shape

        
        # img= np.pad(img, ((0, 0), (y-cropy,0),(0,0)), 'constant', constant_values=(pad_val))
        startx = x//2 - cropx//2
        endx = x//2 + cropx//2 + cropx%2
        starty = y//2 - cropy//2
        endy = y//2 + cropy//2 + cropy%2
        startz = z//2 - cropz//2
        endz = z//2 + cropz//2 + cropz%2
        out=img[startz:endz, starty:endy, startx:endx]

        return out
    

    def cropCenter(self):
        x1=self.x.squeeze(0).numpy()

        s0,s1,s2 =self.image.shape
        
        self.x_croped=self.crop_center(x1,s0,s1,s2,-1000)

        if self.smoothing:
            field1=self.field_smoothed
        else:
            field1=self.field

        field1_croped=np.zeros([3,self.image.shape[0],self.image.shape[1],self.image.shape[2]],np.float32)
        field1_croped[0,:,:,:]=self.crop_center(field1[0,:,:,:].numpy(),s0,s1,s2,0)
        field1_croped[1,:,:,:]=self.crop_center(field1[1,:,:,:].numpy(),s0,s1,s2,0)
        field1_croped[2,:,:,:]=self.crop_center(field1[2,:,:,:].numpy(),s0,s1,s2,0)

        self.field_croped=field1_croped



    def applyField(self,smoothing=False):
        if smoothing:
            # self.field_smoothed=torch.from_numpy(field_smoothed)
            field_smoothed=np.zeros_like(self.field_croped)
            field_smoothed[0,:,:,:]=scipy.ndimage.gaussian_filter(self.field[0,:,:,:], 5)
            field_smoothed[1,:,:,:]=scipy.ndimage.gaussian_filter(self.field[1,:,:,:], 5)
            field_smoothed[2,:,:,:]=scipy.ndimage.gaussian_filter(self.field[2,:,:,:], 5)
            # self.field_smoothed=torch.from_numpy(field_smoothed)
            self.field_croped=field_smoothed

        field1_croped2=np.moveaxis(self.field_croped,0,3)
        image_itk=sitk.ReadImage(self.Case[0])
        FF=sitk.GetImageFromArray(field1_croped2)
        FF.CopyInformation(image_itk)
        XX=sitk.GetImageFromArray(self.x_croped)
        XX.CopyInformation(image_itk)

        warpFilter = sitk.WarpImageFilter()
        warpFilter.SetInterpolator(sitk.sitkBSpline)
        warpFilter.SetEdgePaddingValue(-1000)
        warpFilter.SetOutputParameteresFromImage(image_itk)
        out_itk=warpFilter.Execute(XX,FF)
        self.out_itk=out_itk
        self.deformed_image=sitk.GetArrayFromImage(out_itk)

    def save(self):
        self.deformed_address=(self.Case[0]).replace("image","deformed")
        # print(address)
        sitk.WriteImage(self.out_itk,self.deformed_address)
    
    def saveField(self):
        image_itk=sitk.ReadImage(self.Case[0])
        field1_croped2=np.moveaxis(self.field_croped,0,3)
        FF=sitk.GetImageFromArray(field1_croped2)
        FF.CopyInformation(image_itk)
        temp=(self.Case[0]).replace("image","netfield")
        # print(address)
        sitk.WriteImage(FF,temp)



    def getCTreconview(self):
        
        self.CTrecon=rescale(reconCTview(get_full_CT(self.Case)),-0.01,0.01,-0.01,0.01)