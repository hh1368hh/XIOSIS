import dicom2nifti
import os
from pathlib import Path
import os
import numpy as np
import matplotlib.pyplot as plt
import imageio
import glob
import nibabel as nib




class LoadCT:

    # the output is set to be numpy
    # Image has the header and info
    # image_data is the numpy array of image
    def __init__(self, input_path, output_path=None):

        if output_path is None:
            output_path = input_path

        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.nii_list = Path.rglob(self.input_path, '*.nii.gz')
        DataList = []
        for i in self.nii_list:
            DataList.append(i)
        self.DataList = DataList
        print(f"{len(DataList)} data is available (index from zero)")

    def dcm2nii(self):

        dicom2nifti.convert_directory(self.input_path, self.output_path, compression=True, reorient=True)

    def loadnii(self, ndx = 0):
            image = nib.load(self.DataList[ndx])
            image_data = image.get_fdata()

            return image, image_data

    def loaddcm(self):

        image = imageio.volread(self.input_path, 'DICOM')
        image_data = (np.array(image))

        return image, image_data