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
import functools
import torch.nn.functional as F
from collections import namedtuple
import astra
import torch
import torchvision.transforms as transforms


DataInfoTuple = namedtuple('DataInfoTuple','sourceaddress')



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



def getvaldata(data_tup, angle):
    filenames=data_tup


    # print(data_tup)    
    rawimg=getData(filenames)
    # raw_tup=(rawimg,transform)
    sample=rawimg
    sample=np.where(sample < -1000, -1000, sample)
    # img=img-1000
    sample=(sample/1000)*0.2+0.2
    projc, projp = projector(sample,angle)
    # sample = applytransfrom(raw_tup)

    projc = projc[np.newaxis, ...]  # add channel axis
    projp = projp[np.newaxis, ...]
    # thr=filters.threshold_otsu(img.numpy())
    # print(type(thr))
    # thr=torch.from_numpy(np.array(thr)).float()
    sample=(projc,projp)
    # print(thr)
    return sample

def getData(filenames):
    # print(filenames)
    CT_fns=filenames.sourceaddress
    sample=(np.flip(np.moveaxis(np.moveaxis(nib.load(CT_fns).get_fdata().astype(np.float32),2,0),1,2),1))
    return sample
    


def projector(img,angle):
    distance_source_origin = 1000  # [mm]
    distance_origin_detector = 500  # [mm]
    detector_pixel_size = 1  # [mm]
    detector_rows = 512  # Vertical size of detector [pixels].
    detector_cols = 512  # Horizontal size of detector [pixels].

    # angles=np.random.uniform(0,2 * np.pi,1)
    angles=np.deg2rad(angle)

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
    projc=np.moveaxis(projections,1,0)
    
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


    projp=np.moveaxis(projectionsp,1,0)
    a=torch.from_numpy(projp)
    a=a.unsqueeze(0).unsqueeze(0)
    # print(a.shape)
    b=torch.nn.functional.interpolate(a,(projp.shape[0],512,512),mode='trilinear')
    projp=b.squeeze(0).squeeze(0).numpy()
    astra.data3d.delete(projections_id)

    return projc.squeeze(0),projp.squeeze(0)