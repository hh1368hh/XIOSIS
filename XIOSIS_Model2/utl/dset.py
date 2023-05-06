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
from skimage.transform import resize

DataInfoTuple = namedtuple('DataInfoTuple','SFOVaddress, CTDaddress, LFOVaddress, CTFaddress, SPFaddress')

# raw_cache = getCache('T1_T2')




# @functools.lru_cache(1)
def getdatainfo(data_dir):
    # data_dir = 'Data/small'
    # CTD_dir = os.path.join(data_dir, 'CTD')
    SFOV_dir = os.path.join(data_dir, 'SFOV')
    CTD_dir = os.path.join(data_dir, 'CTD')
    LFOV_dir = os.path.join(data_dir, 'LFOV')
    CTF_dir = os.path.join(data_dir, 'CTF')
    SPF_dir = os.path.join(data_dir, 'SPF')

    SFOV_fns = glob(os.path.join(SFOV_dir, '*.nii'))
    CTD_fns = glob(os.path.join(CTD_dir, '*.nii'))
    LFOV_fns = glob(os.path.join(LFOV_dir, '*.nii'))
    CTF_fns = glob(os.path.join(CTF_dir, '*.nii.gz'))
    SPF_fns = glob(os.path.join(SPF_dir, '*.nii.gz'))
    
    # CTF_fns=list(set(CTF_fns) - set(glob(os.path.join(CTF_dir,"*LFOV.nii.gz"))))
    # SPF_fns=list(set(SPF_fns) - set(glob(os.path.join(SPF_dir,"*LFOV.nii.gz"))))


    # print(len(CTF_fns))
    # print(CTF_fns)
    
    # print(len(LFOV_fns))
    # assert len(CTD_fns)/3 == 14.
    assert len(SFOV_fns) == len(CTF_fns) * 3 and len(CTF_fns) != 0
    assert len(LFOV_fns) == len(CTF_fns) * 3 and len(CTF_fns) != 0
    assert len(LFOV_fns) == len(CTF_fns) * 3 and len(CTF_fns) != 0
    assert len(SPF_fns) == len(CTF_fns)
    # if len(CT_fns) != len(STR_fns) or len(CT_fns) == 0:
        # raise ValueError(f'Number of source and target images must be equal and non-zero')
    datainfolist=[]
    for i,c in enumerate(CTF_fns):

        CasecolCTF=CTF_fns[i][:CTF_fns[i].find('F_S')+6]
        CaseCTD=CTF_fns[i][:CTF_fns[i].find('_CTF')-3]

        CTF_add=CasecolCTF+'.nii.gz'
        
        FOV_add=(CasecolCTF + '_0.nii',CasecolCTF + '_1.nii', CasecolCTF + '_2.nii')
        CaseCTD=CaseCTD.replace('CTF','CTD')
        CTD_add=(CaseCTD + 'CTD_0.nii',CaseCTD + 'CTD_1.nii', CaseCTD + 'CTD_2.nii')

   
        SPF_add=CTF_add.replace('CTF','SPF')
        # CTDNum=SFOV_fns[i][:SFOV_fns[i].find('\\P')+4]
        # CTDNum=CTDNum.replace('SFOV','CTD')
        # CTD_add=(CTDNum + '_CTD_0.nii',CTDNum + '_CTD_1.nii', CTDNum + '_CTD_2.nii')
        LFOV_add=(FOV_add[0].replace('CTF','LFOV'),FOV_add[1].replace('CTF','LFOV'),FOV_add[2].replace('CTF','LFOV'))
        SFOV_add=(FOV_add[0].replace('CTF','SFOV'),FOV_add[1].replace('CTF','SFOV'),FOV_add[2].replace('CTF','SFOV'))

        # LFOV_add=LFOV_add.replace('projc','projp')
        # field_add=field_add.replace('image','field')
        #print(field_add)
        #print(image_fns[i])

        datainfolist.append(DataInfoTuple(SFOV_add,CTD_add,LFOV_add,CTF_add,SPF_add))


    # R=np.random.randint(low=0, high=len(datainfolist), size=50)
    # x_arr = np.asarray(datainfolist, dtype=object)
    # print(R)
    # datainfolist=list(x_arr[R])
    # print(datainfolist)
    
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
        try: 
            sample = getsample(data_tup)
        except:
            print('file is corrupted, skipped')
            print(dataInfo_tup.CTFaddress)
            print(dataInfo_tup.SPFaddress)
            print(dataInfo_tup.LFOVaddress)
            print(dataInfo_tup.SFOVaddress)
            print(dataInfo_tup.CTDaddress)
            pass
            sample=None
        return sample



# @raw_cache.memoize(typed=True)
def getsample(data_tup):
    filenames,transform=data_tup

    # print(data_tup)
     
    rawimg=getData(filenames)

    # raw_tup=(rawimg,transform)
    if transform is not None:
        rawimg1=transform(rawimg)
        proji,projL,cto,spM=rawimg1
    else:
        proji,projL,cto,spM=rawimg
        trnone=True
        totensor=ToTensor()

    # projc = projc[np.newaxis, ...]  # add channel axis
    cto = np.flip(cto,0)[np.newaxis, ...].copy()
    spM = np.flip(spM,0)[np.newaxis, ...].copy()

    sample=(proji,projL,cto,spM)
    if trnone:
        sample=totensor(sample)
    # print(proji.shape)
    # print(projo.shape)
    # asd
    if sample[0].isnan().any():
      print ('PROJIs are nan')
      print(filenames)
      raise NameError('PROJs are nan')
    if sample[1].isnan().any():
      print ('PROJLs are nan')
      print(filenames)
      raise NameError('PROJs are nan')
    if sample[2].isnan().any():
      print ('CTF is nan')
      print(filenames)
      raise NameError('CTF is nan')
    if sample[3].isnan().any():
      print ('spM is nan')
      print(filenames)
      raise NameError('SPF is nan')
    return sample

def getvaldata(data_tup):
    transform=None
    filenames=data_tup

    # print(data_tup)    
    rawimg=getData(filenames)
    # raw_tup=(rawimg,transform)
    if transform is not None:
        rawimg1=transform(rawimg)
        proji,projL,cto,spM=rawimg1
    else:
        proji,projL,cto,spM=rawimg
        trnone=True
        # totensor=ToTensor()

    # projc = projc[np.newaxis, ...]  # add channel axis
    cto = np.flip(cto,0)[np.newaxis, ...].copy()
    spM = np.flip(spM,0)[np.newaxis, ...].copy()
    
    sample=(proji,projL,cto,spM)
    # if trnone:
    #     sample=totensor(sample)
    # print(proji.shape)
    # print(projo.shape)
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
    
    SFOV_fns=filenames.SFOVaddress
    CTD_fns=filenames.CTDaddress
    LFOV_fns=filenames.LFOVaddress
    CTF_fns=filenames.CTFaddress
    SPF_fns=filenames.SPFaddress
    # SFOV=nib.load(SFOV_fns).get_fdata().astype(np.float32)
    # LFOV=nib.load(LFOV_fns).get_fdata().astype(np.float32)

    # inP=np.stack((SFOV, nib.load(CTD_fns[0]).get_fdata().astype(np.float32),\
    #     nib.load(CTD_fns[1]).get_fdata().astype(np.float32),\
    #         nib.load(CTD_fns[2]).get_fdata().astype(np.float32)))

    
    ### NEW DATA ORder

    inP=np.stack((\
        nib.load(SFOV_fns[0]).get_fdata().astype(np.float32),\
        nib.load(CTD_fns[0]).get_fdata().astype(np.float32),\

        nib.load(SFOV_fns[1]).get_fdata().astype(np.float32),\
        nib.load(CTD_fns[1]).get_fdata().astype(np.float32),\

        nib.load(SFOV_fns[2]).get_fdata().astype(np.float32),\
        nib.load(CTD_fns[2]).get_fdata().astype(np.float32)))

    
    ### OLD DATA ORDER plus CTD between 0 and 1

    # inP=np.stack((\
    #     nib.load(SFOV_fns[0]).get_fdata().astype(np.float32),\
    #     nib.load(SFOV_fns[1]).get_fdata().astype(np.float32),\
    #     nib.load(SFOV_fns[2]).get_fdata().astype(np.float32),\
    #     nib.load(CTD_fns[0]).get_fdata().astype(np.float32),\
    #     nib.load(CTD_fns[1]).get_fdata().astype(np.float32),\
    #     nib.load(CTD_fns[2]).get_fdata().astype(np.float32)))

    # inP[3,:]=inP[3,:]/inP[3,:].max()
    # inP[4,:]=inP[4,:]/inP[4,:].max()
    # inP[5,:]=inP[5,:]/inP[5,:].max()
    
    # LFOV = LFOV[np.newaxis, ...]
    inL=np.stack((\
        nib.load(LFOV_fns[0]).get_fdata().astype(np.float32),\
        nib.load(LFOV_fns[1]).get_fdata().astype(np.float32),\
        nib.load(LFOV_fns[2]).get_fdata().astype(np.float32)))
    # print(inP.shape)
    # from skimage import exposure
    # # inP = exposure.equalize_hist(inP).astype(np.float32)

    # inP=np.stack((\
    #     exposure.equalize_hist(inP[0]).astype(np.float32),\
    #     exposure.equalize_hist(inP[1]).astype(np.float32),\
    #     exposure.equalize_hist(inP[2]).astype(np.float32)))
    
    # print(inP.shape)

    ISIZE=128

    inP=resize(inP,(6,ISIZE,ISIZE))
    inL=resize(inL,(3,ISIZE,ISIZE))


    # inP[1,:,:]=np.fliplr(inP[5,:,:])
    # inP[3,:,:]=np.fliplr(inP[3,:,:])
    # inP[5,:,:]=np.fliplr(inP[1,:,:])



        
    outP=nib.load(CTF_fns).get_fdata().astype(np.float32)



    # temp=inP[0,:,:].copy()
    # inP[0,:,:]=np.fliplr(inP[4,:,:])

    # inP[2,:,:]=np.fliplr(inP[2,:,:])
    # inP[4,:,:]=np.fliplr(temp)
    

    
    # # For Cadaver


    # s0=outP.shape[0]
    # s2=outP.shape[2]
    # sn=np.int(s2*ISIZE/s0)
    # outP=outP + 1000
    # outP=resize(outP,(ISIZE,ISIZE,sn))

    ## For DRRs
    
    outP=resize(outP,(ISIZE,ISIZE,ISIZE))


    outM=nib.load(SPF_fns).get_fdata().astype(np.float32)
    outM=resize(outM,(ISIZE,ISIZE,ISIZE))
    outM[np.where(outM>0.5)]=1
    outM[np.where(outM<0.5)]=0





    # outP=np.fliplr(np.swapaxes(outP,0,1))
    # outM=np.fliplr(np.swapaxes(outM,0,1))




    # import SimpleITK as sitk
    # sitk.Show(sitk.GetImageFromArray(np.fliplr(np.swapaxes(outP,0,1))))
    # sitk.Show(sitk.GetImageFromArray(inP))



    if outM.sum()==0:
        # print ('hello')
        outM=1-outM

    
    sample=(inP,inL,outP,outM)

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
        src, long, tgt, spc = sample
        # print(src.shape)
        src = torch.from_numpy(src).float()
        long = torch.from_numpy(long).float()
        tgt = torch.from_numpy(tgt).float()
        spc = torch.from_numpy(spc).float()
        sample=(src, long, tgt, spc)
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
    # distance_source_origin = 1000  # [mm]
    # distance_origin_detector = 500  # [mm]
    # detector_pixel_size = 1  # [mm]
    # detector_rows = 512  # Vertical size of detector [pixels].
    # detector_cols = 512  # Horizontal size of detector [pixels].
    # # matct=img_data1.astype('float32')
    matct=img-1000
    matct=np.where(img < -1000, -1000, img)
    matct=np.where(img > 2000, 2000, img)
    
    matct=(matct/1000)*0.2+0.2
    matct_rs=matct
    sz=matct_rs.shape

    num_of_projections = 360
    angles = np.linspace(0, 2 * np.pi, num=num_of_projections, endpoint=False)
    angles = angles[[330,0,30]]

    vol_geom = astra.creators.create_vol_geom(sz[1], sz[2],
                                            sz[0])

    ## Image Projection
    phantom_id = astra.data3d.create('-vol', vol_geom, data=img)

    ## Parallel Beam Projection
    proj_geomp = \
        astra.create_proj_geom('parallel3d', 1, 1, sz[1], sz[2], angles)

    projections_idp, projectionsp = \
        astra.creators.create_sino3d_gpu(phantom_id, proj_geomp, vol_geom)

    # projectionsp /= np.max(projectionsp)
    projectionsp = (projectionsp-projectionsp.mean()) / projectionsp.std()
    projectionsp = (projectionsp-projectionsp.min())/(projectionsp.max()-projectionsp.min())


    projp=np.moveaxis(projectionsp,1,0)
    # a=torch.from_numpy(projp)
    # a=a.unsqueeze(0).unsqueeze(0)
    # # print(a.shape)
    # b=torch.nn.functional.interpolate(a,(projp.shape[0],512,512),mode='trilinear')
    # projp=b.squeeze(0).squeeze(0).numpy()

            # for i in range(projcL.shape[0]):
            # projcL[i,:,:]=np.flipud(projcL[i,:,:])


    # astra.data3d.delete(vol_geom)
    astra.data3d.delete(phantom_id)
    # astra.data3d.delete(proj_geom)
    astra.data3d.delete(projections_idp)
    # print(projectionsp.shape)
    return projp