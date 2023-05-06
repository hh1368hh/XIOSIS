import argparse
import sys

import numpy as np

import torch.nn as nn
from torch.autograd import Variable
from torch.optim import SGD
from torch.utils.data import DataLoader

from .util import enumerateWithEstimate
from .dset import CNNDataset
from .logconf import logging
from .model import HH_Unet3

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
log.setLevel(logging.INFO)
# log.setLevel(logging.DEBUG)


class HHPrepCacheApp:
    @classmethod
    def __init__(self, batch_size, num_workers,data_dir, tfm=None):
        self.batch_size=batch_size
        self.num_workers=num_workers
        self.data_dir=data_dir
        self.tfm=tfm
    def main(self):

        self.prep_dl = DataLoader(
            CNNDataset(self.data_dir,self.tfm),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

        batch_iter = enumerateWithEstimate(
            self.prep_dl,
            "Stuffing cache",
            start_ndx=self.prep_dl.num_workers,
        )
        for _ in batch_iter:
            pass

