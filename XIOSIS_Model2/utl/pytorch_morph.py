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
import SimpleITK as sitk
import imagej
import nrrd
from sklearn.mixture import GaussianMixture
from ipywidgets import interact, fixed
from IPython.display import display

import SimpleITK as sitk
import imagej
# from PIL import Image
import pydicom as dicom
import os
import numpy
from matplotlib import pyplot, cm
from rt_utils import RTStructBuilder
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
from rt_utils import RTStructBuilder
import matplotlib.pyplot as plt

import numpy as np
import torch
from torch.nn import functional as f
from scipy.ndimage import grey_dilation as dilation_scipy
import matplotlib.pyplot as plt





# Definition of the dilation using PyTorch
def dilate_pytorch(image, strel, origin=(0, 0), border_value=0, additive=False):
    # first pad the image to have correct unfolding; here is where the origins is used
    
    image.unsqueeze_(0)
    image_pad = f.pad(image, (origin[0], strel.shape[0] - origin[0] - 1, origin[1], strel.shape[1] - origin[1] - 1), mode='replicate')
    image.squeeze_(0)
    image_pad.squeeze_(0)

    # Unfold the image to be able to perform operation on neighborhoods
    image_pad = f.unfold(image_pad.unsqueeze(0), kernel_size=strel.shape)
    # Flatten the structural element since its two dimensions have been flatten when unfolding
    image_pad=image_pad.view((image.shape[0],strel.shape[0]*strel.shape[1],-1))
    strel_flatten = torch.flatten(strel).unsqueeze(0).unsqueeze(-1)
    # Perform the greyscale operation; sum would be replaced by rest if you want erosion

    if additive:
        image_pad = image_pad + strel_flatten
    # print(sums.shape)
    else:
        image_pad = image_pad[:,strel_flatten.squeeze(0).squeeze(1),:]
    # Take maximum over the neighborhood
    image_pad, _ = image_pad.max(dim=1)
    # Reshape the image to recover initial shape
    return torch.reshape(image_pad, image.shape)

# Definition of the dilation using PyTorch
def erode_pytorch(image, strel, origin=(0, 0), border_value=0, additive=False):
    # first pad the image to have correct unfolding; here is where the origins is used
    image.unsqueeze_(0)
    image_pad = f.pad(image, (origin[0], strel.shape[0] - origin[0] - 1, origin[1], strel.shape[1] - origin[1] - 1), mode='replicate')
    image.squeeze_(0)
    image_pad.squeeze_(0)

    # Unfold the image to be able to perform operation on neighborhoods
    image_pad = f.unfold(image_pad.unsqueeze(0), kernel_size=strel.shape)
    # Flatten the structural element since its two dimensions have been flatten when unfolding
    image_pad=image_pad.view((image.shape[0],strel.shape[0]*strel.shape[1],-1))
    strel_flatten = torch.flatten(strel).unsqueeze(0).unsqueeze(-1)
    # Perform the greyscale operation; sum would be replaced by rest if you want erosion

    if additive:
        strel_flatten=strel_flatten.float()
        image_pad = image_pad - strel_flatten
    else:
        image_pad = image_pad[:,strel_flatten.squeeze(0).squeeze(1),:]
    # Take maximum over the neighborhood
    image_pad, _ = image_pad.min(dim=1)
    # Reshape the image to recover initial shape
    return torch.reshape(image_pad, image.shape)



def open_pytorch(image, strel, origin=(0, 0), border_value=0, additive=False):
    out= dilate_pytorch(erode_pytorch(image=image, strel=strel, origin=origin, border_value=border_value, additive=additive),strel=strel, origin=origin, border_value=border_value,additive=additive)
    return out
def close_pytorch(image, strel, origin=(0, 0), border_value=0, additive=False):
    out= erode_pytorch(dilate_pytorch(image=image, strel=strel, origin=origin, border_value=border_value,additive=additive),strel=strel, origin=origin, border_value=border_value,additive=additive)
    return out

def tophat_pytorch(image, strel, origin=(0, 0), border_value=0, additive=False):
    out =image - open_pytorch(image=image,strel=strel, origin=origin, border_value=border_value, additive=additive)
    return out


# Pytorch conditional dialation
def cond_dilate_pytorch(marker, image, strel, origin=(0, 0), border_value=0,k=1, additive=False):
    out=marker
    for i in range(k):
        out=torch.minimum(dilate_pytorch(out,strel,origin,border_value, additive=additive),image)
    return out

# Pytorch Reconstrution
def recon_pytorch(marker, image, strel, origin=(0, 0), border_value=0,k=1, additive=False):
    sz=image.shape
    temp=torch.zeros(k,sz[0],sz[1],sz[2])
    for i in range(k):
        temp[i,:,:]=cond_dilate_pytorch(marker, image, strel, origin, border_value,k=i, additive=additive)
    out,_=temp.max(dim=0)
    return out

# Pytorch tophat by reconstruction
def tophat_recon_pytorch(image, strel, origin=(0, 0), border_value=0,k=1 , additive=False):
    out = image - recon_pytorch(open_pytorch(image, strel, origin, border_value, additive=additive), image, strel, origin, border_value,k, additive=additive)
    return out

# def otsu_pytorch(image):

