from torch._C import dtype
from torch.utils.data.dataset import Dataset
from .Load_CT import LoadCT
import dicom2nifti
import os
from pathlib import Path
import os
import numpy as np
import matplotlib.pyplot as plt
import imageio
from glob import glob
import nibabel as nib
import torch
from .disk import getCache
import functools
import torch.nn.functional as F
from collections import namedtuple
import scipy.ndimage as ndimage
from skimage import filters
import astra
import torch
import torchvision.transforms as transforms
import astra
import torch
import torchvision.transforms as transforms

DataInfoTuple = namedtuple('DataInfoTuple','sourceaddress')

# raw_cache = getCache('T1_T2')




# @functools.lru_cache(1)
def getdatainfo(data_dir):
    # data_dir = 'Data/small'
    CT_dir = os.path.join(data_dir, 'CT')
    # STR_dir = os.path.join(data_dir, 'STR')
    CT_fns = glob(os.path.join(CT_dir, '*.nii*'))
    # STR_fns = glob(os.path.join(STR_dir, '*.nii*'))
    # assert len(CT_fns) == len(STR_fns) and len(CT_fns) != 0
    # if len(CT_fns) != len(STR_fns) or len(CT_fns) == 0:
        # raise ValueError(f'Number of source and target images must be equal and non-zero')
    datainfolist=[]
    for i,c in enumerate(CT_fns):

        # STR_add=CT_fns[i].replace('CT','STR')
        # field_add=field_add.replace('image','field')
        #print(field_add)
        #print(image_fns[i])

        datainfolist.append(DataInfoTuple(CT_fns[i]))

    return datainfolist








class CNNDataset(Dataset):
    def __init__(self, source_dir, transform=None, preload=True):
        self.transform = transform
        self.datainfolist = getdatainfo(source_dir)

    def __len__(self):
        # fill this in
        return len(self.datainfolist)

    def __getitem__(self, idx):
        dataInfo_tup=self.datainfolist[idx]
        data_tup=(dataInfo_tup,self.transform)
        sample = getsample(data_tup)
        return sample



# @raw_cache.memoize(typed=True)
def getsample(data_tup):
    filenames,transform=data_tup

    # print(data_tup)    
    rawimg=getData(filenames)
    # raw_tup=(rawimg,transform)
    if transform is not None:
        sample=transform(rawimg)
    else:
        sample=rawimg
        trnone=True
        totensor=ToTensor()
    # sample = applytransfrom(raw_tup)
    ## Converitng HU to attenuation Coeficient

    # img=sample
    # print((img.dtype))
    # img[np.where(img<0)]=0
    sample=np.where(sample < -1000, -1000, sample)
    # img=img-1000
    sample=(sample/1000)*0.2+0.2
    projc, projp = projector(sample)
    # print(projc.shape)

    # projc=sample[50,:,:]
    # projp=sample[100,:,:]
    # projc = projc[np.newaxis, ...]  # add channel axis
    # projp = projp[np.newaxis, ...]
    # # print(projc.shape)

    # a=torch.from_numpy(projp)
    # a=a.unsqueeze(0).unsqueeze(0)
    # # print(a.shape)
    # b=torch.nn.functional.interpolate(a,(projp.shape[0],512,512),mode='trilinear')
    # projp=b.squeeze(0).squeeze(0).numpy().squeeze(0)

    # a=torch.from_numpy(projc)
    # a=a.unsqueeze(0).unsqueeze(0)
    # # print(a.shape)
    # b=torch.nn.functional.interpolate(a,(projc.shape[0],512,512),mode='trilinear')
    # projc=b.squeeze(0).squeeze(0).numpy().squeeze(0)
    # # print(projc.shape)


    projc = projc[np.newaxis, ...]  # add channel axis
    projp = projp[np.newaxis, ...]
    # thr=filters.threshold_otsu(img.numpy())
    # print(type(thr))
    # thr=torch.from_numpy(np.array(thr)).float()
    sample=(projc,projp)
    if trnone:
        sample=totensor(sample)
    # print(thr)
    return sample






# @functools.lru_cache()
# def applytransfrom(raw_tup):
#     # print(sample.shape)
#     rawimg,tfm=raw_tup
#     sample=tfm(rawimg)
#     return sample




# @functools.lru_cache(1, typed=True)
def getData(filenames):
    # print(filenames)
    CT_fns=filenames.sourceaddress
    sample=(np.flip(np.moveaxis(np.moveaxis(nib.load(CT_fns).get_fdata().astype(np.float32),2,0),1,2),1))
    return sample


# class RandomCrop3D:
#     def __init__(self, args):
#         # fill this in
#
#     def __call__(self, sample):
#         # fill this in

class CropBase:
    """ base class for crop transform """

    def __init__(self, out_dim, output_size, threshold = None):
        """ provide the common functionality for RandomCrop2D and RandomCrop3D """
        assert isinstance(output_size, (int, tuple, list))
        if isinstance(output_size, int):
            self.output_size = (output_size,)
            for _ in range(out_dim - 1):
                self.output_size += (output_size,)
        else:
            assert len(output_size) == out_dim
            self.output_size = output_size
        self.out_dim = out_dim
        self.thresh = threshold

    def _get_sample_idxs(self, img):
        """ get the set of indices from which to sample (foreground) """
        # A three row list of point x,y,z
        mask = np.where(img >= (img.mean() if self.thresh is None else self.thresh))  # returns a tuple of length 3
        c = np.random.randint(0, len(mask[0]))  # choose the set of idxs to use
        h, w, d = [m[c] for m in mask]  # pull out the chosen idxs
        return h, w, d


class RandomCrop3D(CropBase):
    """
    Randomly crop a 3d patch from a (pair of) 3d image

    Args:
        output_size (tuple or int): Desired output size.
            If int, cube crop is made.
    """

    def __init__(self, output_size,NetDepth, threshold=None):
        super().__init__(3, output_size, threshold)
        self.Netdepth=NetDepth

    def __call__(self, sample):
        src, tgt = sample
        *cs, h, w, d = src.shape
        *ct, _, _, _ = tgt.shape
        hh, ww, dd = self.output_size
        
        if hh==-2:
            hh=h
        if ww==-2:
            ww=w
        if dd==-2:
            dd=d
        # print((dd,ww,hh))

        max_idxs = (h - hh // 2, w - ww // 2, d - dd // 2)
        min_idxs = (hh // 2, ww // 2, dd // 2)
        s = src[0] if len(cs) > 0 else src  # use the first image to determine sampling if multimodal
        s_idxs = super()._get_sample_idxs(s)
        # print(s_idxs)
        i, j, k = [i if min_i <= i <= max_i else max_i if i > max_i else min_i
                   for max_i, min_i, i in zip(max_idxs, min_idxs, s_idxs)]
        oh = 0 if hh % 2 == 0 else 1
        ow = 0 if ww % 2 == 0 else 1
        od = 0 if dd % 2 == 0 else 1
        # print(i)
        s = src[..., i - hh // 2:i + hh // 2 + oh, j - ww // 2:j + ww // 2 + ow, k - dd // 2:k + dd // 2 + od]
        t = tgt[..., i - hh // 2:i + hh // 2 + oh, j - ww // 2:j + ww // 2 + ow, k - dd // 2:k + dd // 2 + od]
        
        dnum=2**self.Netdepth
        # s,t=padcompatible(s,t,dnum)
        
        if len(cs) == 0: s = s[np.newaxis, ...]  # add channel axis if empty
        if len(ct) == 0: t = t[np.newaxis, ...]
        return s, t



class CenterCropBase:
    """ base class for crop transform """

    def __init__(self, out_dim, output_size, threshold = None):
        """ provide the common functionality for RandomCrop2D and RandomCrop3D """
        assert isinstance(output_size, (int, tuple, list))
        if isinstance(output_size, int):
            self.output_size = (output_size,)
            for _ in range(out_dim - 1):
                self.output_size += (output_size,)
        else:
            assert len(output_size) == out_dim
            self.output_size = output_size
        self.out_dim = out_dim
        self.thresh = threshold

    def _get_sample_idxs(self, img):
        """ get the set of indices from which to sample (foreground) """
        # A three row list of point x,y,z
        self.thresh=-500
        mask = np.where(img >= (img.mean() if self.thresh is None else self.thresh))  # returns a tuple of length 3
        # c = np.random.randint(0, len(mask[0]))  # choose the set of idxs to use
        ch= len(mask[0])//2
        tm=np.sort(mask[1][mask[0]==mask[0][ch]])
        cw=len(tm)//2
        tm=np.sort(mask[2][tm]==tm[cw])
        cd=len(tm)//2
        print(ch)
        h=mask[0][ch]
        w=mask[1][cw]
        d=mask[2][cd]


        # h, w, d = [m[ch] for m in mask]  # pull out the chosen idxs
        return h, w, d

class CenterCrop3D(CenterCropBase):
    """
    Randomly crop a 3d patch from a (pair of) 3d image

    Args:
        output_size (tuple or int): Desired output size.
            If int, cube crop is made.
    """

    def __init__(self, output_size,NetDepth, threshold=None):
        super().__init__(3, output_size, threshold)
        self.Netdepth=NetDepth

    def __call__(self, sample):
        src, tgt = sample
        *cs, h, w, d = src.shape
        *ct, _, _, _ = tgt.shape
        hh, ww, dd = self.output_size
        
        if hh==-2:
            hh=h
        if ww==-2:
            ww=w
        if dd==-2:
            dd=d
        # print((dd,ww,hh))

        max_idxs = (h - hh // 2, w - ww // 2, d - dd // 2)
        min_idxs = (hh // 2, ww // 2, dd // 2)
        s = src[0] if len(cs) > 0 else src  # use the first image to determine sampling if multimodal
        s_idxs = super()._get_sample_idxs(s)
        # s_idxs= (s.shape[0] // 2, s.shape[1] // 2, s.shape[2] // 2)

        # print(s_idxs)

        i, j, k = [i if min_i <= i <= max_i else max_i if i > max_i else min_i
                   for max_i, min_i, i in zip(max_idxs, min_idxs, s_idxs)]
        oh = 0 if hh % 2 == 0 else 1
        ow = 0 if ww % 2 == 0 else 1
        od = 0 if dd % 2 == 0 else 1
        # print(i)
        s = src[..., i - hh // 2:i + hh // 2 + oh, j - ww // 2:j + ww // 2 + ow, k - dd // 2:k + dd // 2 + od]
        t = tgt[..., i - hh // 2:i + hh // 2 + oh, j - ww // 2:j + ww // 2 + ow, k - dd // 2:k + dd // 2 + od]
        
        dnum=2**self.Netdepth
        # s,t=padcompatible(s,t,dnum)
        
        if len(cs) == 0: s = s[np.newaxis, ...]  # add channel axis if empty
        if len(ct) == 0: t = t[np.newaxis, ...]
        return s, t



class Permute():
    def __init__(self,ax1,ax2):
        self.ax1=ax1
        self.ax2=ax2

    def __call__(self, sample):
       src, tgt = sample
    #    print(src.shape)
    #    print(tgt.shape)
       s=np.swapaxes(src,self.ax1,self.ax2)
       t=np.swapaxes(tgt,self.ax1+1,self.ax2+1)
       return s,t

class Gains():
    def __init__(self,Gs=1,Gt=1):
        self.Gs=Gs
        self.Gt=Gt

    def __call__(self, sample):
       src, tgt = sample
    #    print(src.shape)
    #    print(tgt.shape)
       s=src * self.Gs
       t=tgt * self.Gt
       return s,t

class RandomZoom():
    # def __init__(self):
    #     # self.ZF=ZF
    
    def __call__(self, sample):
        src, tgt = sample
        ZF=np.round(np.random.uniform(0.7,1),1)
        
        s = ndimage.zoom(src,ZF,order=1)
        t=np.zeros((3,s.shape[0],s.shape[1],s.shape[2]))

        for i in range(0,tgt.shape[0]):
           t[i,:,:,:] = ndimage.zoom(tgt[i,:,:,:],ZF,order=1) 
        t = t * ZF
        return s,t

class ToTensor():
    """ Convert images in sample to Tensors """
    def __call__(self, sample):
        src, tgt = sample
        # print(src.shape)
        src = torch.from_numpy(src).float()
        tgt = torch.from_numpy(tgt).float()
        sample=(src, tgt)
        return sample

# @functools.lru_cache()
def iseven(num):
    if (num % 2) == 0:
        even=True
    else:
        even=False
    return even

    
# @functools.lru_cache()
def padcompatible(A,B,dnum):
    s=torch.from_numpy(A)
    t=torch.from_numpy(B)
    r=s.shape[2] % dnum
    padsize=dnum-r
    if r != 0:
        if iseven(r):
            padsizebefore=padsize/2
            padsizeafter=padsize/2
            
        else:
            padsizebefore=round(padsize/2)
            padsizeafter=round(padsize/2)+1
        
        padsizebefore=int(padsizebefore)
        padsizeafter=int(padsizeafter)

        s=F.pad(s,(padsizebefore,padsizeafter,0,0,0,0))
        t=F.pad(t,(padsizebefore,padsizeafter,0,0,0,0))

    r=s.shape[1] % dnum
    padsize=dnum-r
    if r != 0:
        if iseven(r):
            padsizebefore=padsize/2
            padsizeafter=padsize/2
        
        else:
            padsizebefore=round(padsize/2)
            padsizeafter=round(padsize/2)+1

        padsizebefore=int(padsizebefore)
        padsizeafter=int(padsizeafter)
        s=F.pad(s,(0,0,padsizebefore,padsizeafter,0,0))
        t=F.pad(t,(padsizebefore,padsizeafter,0,0,0,0))

    r=s.shape[0] % dnum
    padsize=dnum-r
    if r != 0:
        if iseven(r):
            padsizebefore=padsize/2
            padsizeafter=padsize/2
        
        else:
            padsizebefore=round(padsize/2)
            padsizeafter=round(padsize/2)+1

        padsizebefore=int(padsizebefore)
        padsizeafter=int(padsizeafter)
        s=F.pad(s,(0,0,0,0,padsizebefore,padsizeafter))
        t=F.pad(t,(padsizebefore,padsizeafter,0,0,0,0))

    s=s.numpy()
    t=t.numpy()

    return s,t

def projector(img):
    distance_source_origin = 1000  # [mm]
    distance_origin_detector = 500  # [mm]
    detector_pixel_size = 1  # [mm]
    detector_rows = 512  # Vertical size of detector [pixels].
    detector_cols = 512  # Horizontal size of detector [pixels].

    angles=np.random.uniform(0,2 * np.pi,1)
    sz=img.shape

    ## Cone Beam Projection
    # num_of_projections = 180
    # angles = np.linspace(0, 2 * np.pi, num=num_of_projections, endpoint=False)
    proj_geom = \
    astra.create_proj_geom('cone', 1, 1, 512, 512, angles,distance_source_origin,distance_origin_detector)
                        #  (distance_source_origin + distance_origin_detector) /
                        #  detector_pixel_size, 0)

    vol_geom = astra.creators.create_vol_geom(sz[1], sz[2],
                                            sz[0])

    ## Image Projection
    phantom_id = astra.data3d.create('-vol', vol_geom, data=img)

    projections_id, projections = \
    astra.creators.create_sino3d_gpu(phantom_id, proj_geom, vol_geom)
    # projections /= np.max(projections)

    projections = (projections-projections.mean()) / projections.std()
    projections=np.clip(projections,-3,3)/3


    projc=np.moveaxis(projections,1,0)

    # astra.data3d.delete(proj_geom)
    astra.data3d.delete(projections_id)

    ## Parallel Beam Projection
    proj_geomp = \
        astra.create_proj_geom('parallel3d', 1, 1, 341, 341, angles)
                        #  (distance_source_origin + distance_origin_detector) /
                        #  detector_pixel_size, 0)

    # vol_geom = astra.creators.create_vol_geom(sz[1], sz[2],
    #                                         sz[0])

    ## Image Projection
    # phantom_idp = astra.data3d.create('-vol', vol_geom, data=img)

    projections_idp, projectionsp = \
        astra.creators.create_sino3d_gpu(phantom_id, proj_geomp, vol_geom)

    # projectionsp /= np.max(projectionsp)
    projectionsp = (projectionsp-projectionsp.mean()) / projectionsp.std()
    projectionsp=np.clip(projectionsp,-3,3)/3


    projp=np.moveaxis(projectionsp,1,0)
    a=torch.from_numpy(projp)
    a=a.unsqueeze(0).unsqueeze(0)
    # print(a.shape)
    b=torch.nn.functional.interpolate(a,(projp.shape[0],512,512),mode='trilinear')
    projp=b.squeeze(0).squeeze(0).numpy()

    # astra.data3d.delete(vol_geom)
    astra.data3d.delete(phantom_id)
    # astra.data3d.delete(proj_geom)
    astra.data3d.delete(projections_idp)

    return projc.squeeze(0),projp.squeeze(0)