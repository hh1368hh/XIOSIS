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




raw_cache = getCache('T1_T2')




@functools.lru_cache()
def getdatainfo(data_dir):
    # data_dir = 'Data/small'
    t1_dir = os.path.join(data_dir, 't1')
    t2_dir = os.path.join(data_dir, 't2')
    t1_fns = glob(os.path.join(t1_dir, '*.nii*'))
    t2_fns = glob(os.path.join(t2_dir, '*.nii*'))
    assert len(t1_fns) == len(t2_fns) and len(t1_fns) != 0

    return t1_fns, t2_fns








class NiftiDataset(Dataset):
    def __init__(self, source_dir, transform=None, preload=True):
        self.transform = transform
        self.t1_fns, self.t2_fns = getdatainfo(source_dir)
        if len(self.t1_fns) != len(self.t2_fns) or len(self.t1_fns) == 0:
            raise ValueError(f'Number of source and target images must be equal and non-zero')
        self.preload = preload
        if preload:

            # self.imgs = [getmr([s,t]) for s, t in zip(self.t1_fns, self.t2_fns)]
            self.imgs=[]
            for s, t in zip(self.t1_fns, self.t2_fns):
                # print([s,t])
                self.imgs.append(getmr((s,t)))

    def __len__(self):
        # fill this in
        return len(self.t1_fns)

    def __getitem__(self, idx):
        # fill this in
        if not self.preload:
            imageT1 = nib.load(self.t1_fns[idx])
            imageT1_data = imageT1.get_fdata()
            imageT2 = nib.load(self.t2_fns[idx])
            imageT2_data = imageT2.get_fdata()
            sample = (imageT1_data, imageT2_data)
        else:
            sample = self.imgs[idx]

        
        

        if self.transform is not None:
            sample = self.transform(sample)

        return sample



@raw_cache.memoize(typed=True)
def getmr(filenames):
    t1_fns,t2_fns=filenames
    sample=(nib.load(t1_fns).get_data().astype(np.float32),
            nib.load(t2_fns).get_data().astype(np.float32))
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
        s,t=padcompatible(s,t,dnum)
        
        
        print(s.shape)
        if len(cs) == 0: s = s[np.newaxis, ...]  # add channel axis if empty
        if len(ct) == 0: t = t[np.newaxis, ...]
        return s, t



class ToTensor():
    """ Convert images in sample to Tensors """
    def __call__(self, sample):
        src, tgt = sample
        # print(src.shape)
        src = torch.from_numpy(src).float()
        tgt = torch.from_numpy(tgt).float()
        return src, tgt

@functools.lru_cache()
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