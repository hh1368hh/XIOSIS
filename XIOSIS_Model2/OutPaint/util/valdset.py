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


DataInfoTuple = namedtuple('DataInfoTuple','SFOVaddress, CTDaddress, LFOVaddress')


def getdatainfo(data_dir):
    # data_dir = 'Data/small'
    CTD_dir = os.path.join(data_dir, 'CTD')
    LFOV_dir = os.path.join(data_dir, 'LFOV')
    SFOV_dir = os.path.join(data_dir, 'SFOV')

    LFOV_fns = glob(os.path.join(LFOV_dir, '*.nii'))
    SFOV_fns = glob(os.path.join(SFOV_dir, '*.nii'))
    assert len(LFOV_fns) == len(SFOV_fns) and len(SFOV_fns) != 0
    
    # if len(CT_fns) != len(STR_fns) or len(CT_fns) == 0:
        # raise ValueError(f'Number of source and target images must be equal and non-zero')
    datainfolist=[]
    for i,c in enumerate(SFOV_fns):
        CTDNum=SFOV_fns[i][:SFOV_fns[i].find('\\P')+4]
        CTDNum=CTDNum.replace('SFOV','CTD')
        CTD_add=(CTDNum + '_CTD_0.nii',CTDNum + '_CTD_1.nii', CTDNum + '_CTD_2.nii')
        LFOV_add=SFOV_fns[i].replace('SFOV','LFOV')
        # LFOV_add=LFOV_add.replace('projc','projp')
        # field_add=field_add.replace('image','field')
        #print(field_add)
        #print(image_fns[i])

        datainfolist.append(DataInfoTuple(SFOV_fns[i],CTD_add,LFOV_add))


    return datainfolist

DataInfoTuple_CT_STR = namedtuple('DataInfoTuple','CTaddress , STRaddress')


def getdatainfo_CT(data_dir):
    # data_dir = 'Data/small'
    CT_dir = os.path.join(data_dir)
    # STR_dir = os.path.join(data_dir, 'STR')
    CT_fns = glob(os.path.join(CT_dir, '*IMG*'))
    STR_fns = glob(os.path.join(CT_dir, '*STR*'))
    # assert len(CT_fns) == len(STR_fns) and len(CT_fns) != 0
    # if len(CT_fns) != len(STR_fns) or len(CT_fns) == 0:
        # raise ValueError(f'Number of source and target images must be equal and non-zero')
    datainfolist_CT_STR=[]
    for i,c in enumerate(CT_fns):

        STR_add=CT_fns[i].replace('IMG','STR')
        # field_add=field_add.replace('image','field')
        #print(field_add)
        #print(image_fns[i])
        if STR_add in STR_fns:
            datainfolist_CT_STR.append(DataInfoTuple_CT_STR(CT_fns[i],STR_add))
        else:
            STR_add=[]
            datainfolist_CT_STR.append(DataInfoTuple_CT_STR(CT_fns[i],STR_add))

    return datainfolist_CT_STR



def get_CT_STR(data_tup):
    filenames=data_tup
    # print(data_tup)    
    rawimg=getData_CT_STR(filenames)
    # raw_tup=(rawimg,transform)
    sample=rawimg
    sample=np.where(sample < -1000, -1000, sample)
    # img=img-1000
    sample=(sample/1000)*0.2+0.2
    return sample



def getvaldata_CT(data_tup, angle):
    filenames=data_tup


    # print(data_tup)    
    rawimg=getData_CT(filenames)
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

def getData_CT_STR(filenames):
    # print(filenames)
    CT_fns=filenames.CTaddress
    SRT_fns=filenames.STRaddress
    img_data = nib.load(CT_fns).get_fdata()

    img_data1=np.moveaxis(img_data,2,0)
    img_data1=np.moveaxis(img_data1,1,2)
    img_data1=np.flip(img_data1,1)

    if SRT_fns is not None: 
        STRM = np.flip(np.moveaxis(np.moveaxis(nib.load(SRT_fns).get_fdata().astype(np.float32),2,0),1,1),1)
        for g in range(STRM.shape[0]):
            STRM[g,:,:]=np.flipud(STRM[g,:,:])
    else:
        STRM=np.zeros_like(img_data1)
    sample=(img_data1, STRM)
    
    return sample
    

def getvaldata(data_tup):
    filenames =data_tup

    # print(data_tup)    
    proji,projo =getData(filenames)

    # projc = projc[np.newaxis, ...]  # add channel axis
    # projp = projp[np.newaxis, ...]

    sample=(proji,projo)
    return sample

def getData(filenames):
    # print(filenames)
    SFOV_fns=filenames.SFOVaddress
    CTD_fns=filenames.CTDaddress
    LFOV_fns=filenames.LFOVaddress

    SFOV=nib.load(SFOV_fns).get_fdata().astype(np.float32)
    LFOV=nib.load(LFOV_fns).get_fdata().astype(np.float32)

    inP=np.stack((SFOV, nib.load(CTD_fns[0]).get_fdata().astype(np.float32),\
        nib.load(CTD_fns[1]).get_fdata().astype(np.float32),\
            nib.load(CTD_fns[2]).get_fdata().astype(np.float32)))

    LFOV = LFOV[np.newaxis, ...]

    sample=(inP,LFOV)

    return sample







def projector(img,angle):
    SOD = 1000  # [mm]
    ODD = 500  # [mm]
    DS = 1  # [mm]
    ## then each pixel on detector corresponds to SOD/(SOD+ODD) mm of the object
    detector_rows = 512  # Vertical size of detector [pixels].
    detector_cols = 512  # Horizontal size of detector [pixels].
    # num_of_projections = 360
    # angles = np.linspace(0, 2 * np.pi, num=angle, endpoint=False)
    angles=angle
    sz=img.shape

    ## Cone Beam Projection
    # num_of_projections = 180
    # angles = np.linspace(0, 2 * np.pi, num=num_of_projections, endpoint=False)

    proj_geom = \
    astra.create_proj_geom('cone', DS, DS, 512, 512, angles,SOD,ODD)
                            #  (distance_source_origin + distance_origin_detector) /
                            #  detector_pixel_size, 0)

    vol_geom = astra.creators.create_vol_geom(sz[1], sz[2],
                                            sz[0])

    ## Image Projection
    phantom_id = astra.data3d.create('-vol', vol_geom, data=img)

    projections_id, projections = \
    astra.creators.create_sino3d_gpu(phantom_id, proj_geom, vol_geom)

    projections = (projections-projections.mean()) / projections.std()
    projections=np.clip(projections,-3,3)/3


    projc=np.moveaxis(projections,1,0)
        
    astra.data3d.delete(projections_id)
    astra.data3d.delete(phantom_id)
    ## Parallel Beam Projection


    
    proj_geom = \
    astra.create_proj_geom('parallel3d', DS, DS, round(512 * SOD/(SOD+ODD)), round(512 * SOD/(SOD+ODD)), angles)
    # proj_geom = \
    #   astra.create_proj_geom('cone', DS, DS, 512, 512, angles,SOD,ODD)
                            #  (distance_source_origin + distance_origin_detector) /
                            #  detector_pixel_size, 0)

    vol_geom = astra.creators.create_vol_geom(sz[1], sz[2],
                                            sz[0])

    ## Image Projection
    phantom_id = astra.data3d.create('-vol', vol_geom, data=img)

    projections_id, projections = \
    astra.creators.create_sino3d_gpu(phantom_id, proj_geom, vol_geom)




    

    projections = (projections-projections.mean()) / projections.std()
    projections=np.clip(projections,-3,3)/3

    projp=np.moveaxis(projections,1,0)


    a=torch.from_numpy(projp)
    a=a.unsqueeze(0).unsqueeze(0)
    # print(a.shape)
    b=torch.nn.functional.interpolate(a,(projp.shape[0],512,512),mode='trilinear')
    projp=b.squeeze(0).squeeze(0).numpy()
    astra.data3d.delete(projections_id)
    astra.data3d.delete(phantom_id)


    return projc,projp




def reconCTview(CT):
    
    SOD = 1000  # [mm]
    ODD = 500  # [mm]
    DS = 1  # [mm]
    ## then each pixel on detector corresponds to SOD/(SOD+ODD) mm of the object
    detector_rows = 512  # Vertical size of detector [pixels].
    detector_cols = 512  # Horizontal size of detector [pixels].
    num_of_projections = 360
    angles = np.linspace(0, 2 * np.pi, num=num_of_projections, endpoint=False)

    proj_geom = \
    astra.create_proj_geom('cone', DS, DS, 512, 512, angles,SOD,ODD)


    projc, _ = projector(CT,angles)
    projections_id = astra.data3d.create('-sino', proj_geom, np.moveaxis(projc,0,1))
    vol_geom = astra.creators.create_vol_geom(round(512 * SOD/(SOD+ODD)), round(512 * SOD/(SOD+ODD)),
                                            round(512 * SOD/(SOD+ODD)))
    # Create a data object for the reconstruction

    rec_idp = astra.data3d.create('-vol', vol_geom)

    cfgp = astra.astra_dict('CGLS3D_CUDA')
    cfgp = astra.astra_dict('SIRT3D_CUDA')
    cfgp = astra.astra_dict('FDK_CUDA')


    cfgp['ReconstructionDataId'] = rec_idp
    cfgp['ProjectionDataId'] = projections_id

    # Create the algorithm object from the configuration structure
    alg_idp = astra.algorithm.create(cfgp)
    # Run 150 iterations of the algorithm
    astra.algorithm.run(alg_idp, 100)
    out = astra.data3d.get(rec_idp)

    # Clean up. Note that GPU memory is tied up in the algorithm object,
    # and main RAM in the data objects.
    astra.algorithm.delete(alg_idp)
    astra.data3d.delete(rec_idp)
    astra.data3d.delete(projections_id)

    return out
