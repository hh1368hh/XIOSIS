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




def spine_extract(fixed_NC):
    

    t, fixed_skin=couch_remove(fixed_NC)

    # boneseg=bones_by_threshold(sitk.GetArrayFromImage(fixed_NC))

    lung_segmentation = mask.apply(fixed_NC)  # default model is U-net(R231)
    # lung_segmentation[lung_segmentation>0]=1
    lung_segmentation=lung_segmentation.astype(float)
    # sitk.Show(sitk.GetImageFromArray(lung_segmentation))
    lungROI=sitk.GetImageFromArray(lung_segmentation)


    l1=np.zeros(lung_segmentation.size)
    l1=np.where(lung_segmentation==1,1,0)
    l2=np.where(lung_segmentation==2,1,0)

    sss = sitk.LabelShapeStatisticsImageFilter()
    temp=sitk.GetArrayFromImage(fixed_skin)
    sss.Execute(sitk.GetImageFromArray(temp))
    fixed_skiny=np.array(sss.GetCentroid(1)[1]).astype(int)

    # sitk.Cast(skin,sitk.sitkFloat64)
    sss = sitk.LabelShapeStatisticsImageFilter()
    sss.Execute(sitk.GetImageFromArray(l1))
    l1Cx=np.array(sss.GetCentroid(1)[0]).astype(int)

    sss = sitk.LabelShapeStatisticsImageFilter()
    sss.Execute(sitk.GetImageFromArray(l2))
    l2Cx=np.array(sss.GetCentroid(1)[0]).astype(int)

    l1xyz=np.argwhere(l1)
    l1z=np.min(l1xyz,0)[0]
    l2xyz=np.argwhere(l2)
    l2z=np.min(l2xyz,0)[0]

    spineRegion=np.zeros(l1.shape)
    spineRegion[0:np.min([l1z,l2z]),0:fixed_skiny+10,np.min([l1Cx,l2Cx]):np.max([l1Cx,l2Cx])]=1
    # spineRegion[0:np.min([l1z,l2z]),0:fixed_skiny+10,np.min([l1Cx,l2Cx]):np.max([l1Cx,l2Cx])]=1

    fixed_NC_th_abd=hu_normalize(fixed_NC,max=215,min=-135)
    # sitk.Show(fixed_NC_th_abd)

    boneseg=bones_by_threshold(sitk.GetArrayFromImage(fixed_NC))
    bsitk=sitk.GetImageFromArray(boneseg)
    bsitk=sitk.Cast(bsitk,sitk.sitkInt8)
    CCfilter=sitk.ConnectedComponentImageFilter()
    ccf=CCfilter.Execute(bsitk)

    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(sitk.ConnectedComponent(ccf))

    label_sizes = [ stats.GetNumberOfPixels(l) for l in stats.GetLabels()]

    cci=label_sizes.index(max(label_sizes))

    bsitk=ccf==cci+1
    boneseg=sitk.GetArrayFromImage(bsitk)
    nn=np.argwhere(boneseg)
    nn1=np.zeros(nn.shape,int)
    nn1[:,[0,1,2]]=nn[:,[2,1,0]]
    nnlist=nn1.tolist()
    # aa=[]
    # for ii in range(0,len(nnlist)):

    #     aa.append(fixed_NC_th_abd.GetPixel(nnlist[ii][0],nnlist[ii][1],nnlist[ii][2]))


    filt=sitk.ConnectedThresholdImageFilter()
    filt.SetSeedList(nnlist)
    filt.SetLower(150)
    filt.SetUpper(215)
    oo=filt.Execute(fixed_NC_th_abd)

    CCfilter=sitk.ConnectedComponentImageFilter()
    ccf=CCfilter.Execute(oo)
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(sitk.ConnectedComponent(ccf))
    label_sizes = [ stats.GetNumberOfPixels(l) for l in stats.GetLabels()]
    cci=label_sizes.index(max(label_sizes))
    oo=ccf==cci+1

    boneseg=sitk.GetArrayFromImage(oo)
    nn=np.argwhere(boneseg)
    nn1=np.zeros(nn.shape,int)
    nn1[:,[0,1,2]]=nn[:,[2,1,0]]
    nnlist=nn1.tolist()


    filt=sitk.ConfidenceConnectedImageFilter()
    filt.SetInitialNeighborhoodRadius(3)
    filt.SetMultiplier(0.9)
    # filt.SetNumberOfIterations(2)
    filt.SetSeedList(nnlist)

    oo=filt.Execute(fixed_NC_th_abd)
    # print(filt.GetMean())
    # print(filt.GetVariance())
    s1=oo*1000
    # sitk.Show(s1)
    filt=sitk.ConnectedThresholdImageFilter()
    filt.AddSeed([0,0,0])
    # filt.SetLower=0
    # filt.SetUpper=0.5
    skin_out = filt.Execute(s1)
    filt=sitk.NotImageFilter()
    skin=filt.Execute(skin_out)


    CCfilter=sitk.ConnectedComponentImageFilter()
    ccf=CCfilter.Execute(skin)

    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(sitk.ConnectedComponent(ccf))

    label_sizes = [ stats.GetNumberOfPixels(l) for l in stats.GetLabels()]

    cci=label_sizes.index(max(label_sizes))

    skin=ccf==cci+1

    filt=sitk.BinaryFillholeImageFilter()
    boneROI=filt.Execute(skin)
    boneROIarray=sitk.GetArrayFromImage(boneROI)
    SpineROI=sitk.GetImageFromArray(spineRegion)

    l1Cz=np.array(sss.GetCentroid(1)[2]).astype(int)
    l2Cz=np.array(sss.GetCentroid(1)[2]).astype(int)

    abdRegion=np.zeros(l1.shape)
    abdRegion[0:round((np.min([l1Cz,l2Cz])+np.min([l1z,l2z]))/2),fixed_skiny+10:,:]=1

    abdskinRegion=np.zeros(l1.shape)
    abdskinRegion[0:round(np.min([l1z,l2z])),:,:]=1
    abdskinRegion1=sitk.GetImageFromArray(abdskinRegion)

    # abdskinRegion=temp
    # abdskinRegion[round(np.min([l1z,l2z])):,:,:]=0
    # abdskinRegion1=sitk.GetImageFromArray(abdskinRegion)    

    # sitk.Show(sitk.GetImageFromArray(abdRegion))    
    abdROI=sitk.GetImageFromArray(abdRegion)


    lungROI=sitk.Cast(lungROI,sitk.sitkFloat64)
    lungROI.CopyInformation(fixed_NC)

    boneROI=sitk.Cast(boneROI,sitk.sitkFloat64)
    boneROI.CopyInformation(fixed_NC)

    SpineROI=sitk.Cast(SpineROI,sitk.sitkFloat64)
    SpineROI.CopyInformation(fixed_NC)

    abdROI=sitk.Cast(abdROI,sitk.sitkFloat64)
    abdROI.CopyInformation(fixed_NC)

    skinROI=sitk.Cast(fixed_skin,sitk.sitkFloat64)
    skinROI.CopyInformation(fixed_NC)

    abdskinROI=sitk.Cast(abdskinRegion1,sitk.sitkFloat64)
    abdskinROI.CopyInformation(fixed_NC)

    return lungROI, boneROI, SpineROI, abdROI, skinROI, abdskinROI