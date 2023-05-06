import torch
from torch import nn
import torch.nn.functional as F
import math




class HH_Net(nn.Module):
    def __init__(self, in_channel=1,  out_channel=1, kernel_size=3):
        super().__init__()



