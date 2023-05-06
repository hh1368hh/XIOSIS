# Adapted from https://discuss.pytorch.org/t/unet-implementation/426

import torch
from torch import nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self, in_channels=1, n_classes=3, depth=5, wf=6, padding=False,
                 batch_norm=False, up_mode='upconv', residual=False):
        """
        Implementation of
        U-Net: Convolutional Networks for Biomedical Image Segmentation
        (Ronneberger et al., 2015)
        https://arxiv.org/abs/1505.04597
        Using the default arguments will yield the exact version used
        in the original paper
        Args:
            in_channels (int): number of input channels
            n_classes (int): number of output channels
            depth (int): depth of the network
            wf (int): number of filters in the first layer is 2**wf
            padding (bool): if True, apply padding such that the input shape
                            is the same as the output.
                            This may introduce artifacts
            batch_norm (bool): Use BatchNorm after layers with an
                               activation function
            up_mode (str): one of 'upconv' or 'upsample'.
                           'upconv' will use transposed convolutions for
                           learned upsampling.
                           'upsample' will use bilinear upsampling.
            residual: if True, residual connections will be added
        """
        super(UNet, self).__init__()
        assert up_mode in ('upconv', 'upsample')
        self.padding = padding
        self.depth = depth
        prev_channels = in_channels
        self.down_path = nn.ModuleList()
        for i in range(depth):
            if i == 0 and residual:
                self.down_path.append(UNetConvBlock(prev_channels, 2 ** (wf + i),
                                                    padding, batch_norm, residual, first=True))
            else:
                self.down_path.append(UNetConvBlock(prev_channels, 2 ** (wf + i),
                                                    padding, batch_norm, residual))
            prev_channels = 2 ** (wf + i)

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(UNetUpBlock(prev_channels, 2 ** (wf + i), up_mode,
                                            padding, batch_norm, residual))
            prev_channels = 2 ** (wf + i)

        self.last = nn.Conv3d(prev_channels, n_classes, kernel_size=1)
        # self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # print("\tIn Model: input size", x.size())
        blocks = []
        for i, down in enumerate(self.down_path):
            # print(x.shape)
            x = down(x)
            if i != len(self.down_path) - 1:
                blocks.append(x)
                x = F.avg_pool3d(x, 2)

        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i - 1])

        res = self.last(x)
        # print("\tIn Model: output size", res.size())
        return res


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, padding, batch_norm, residual=False, first=False):
        super(UNetConvBlock, self).__init__()
        self.residual = residual
        self.out_size = out_size
        self.in_size = in_size
        self.batch_norm = batch_norm
        self.first = first
        self.residual_input_conv = nn.Conv3d(self.in_size, self.out_size, kernel_size=1)
        self.residual_batchnorm = nn.BatchNorm3d(self.out_size)

        if residual:
            padding = 1
        block = []

        if residual and not first:
            block.append(nn.LeakyReLU())
            if batch_norm:
                block.append(nn.BatchNorm3d(in_size))

        block.append(nn.Conv3d(in_size, out_size, kernel_size=3,
                               padding=int(padding)))
        block.append(nn.LeakyReLU())
        if batch_norm:
            block.append(nn.BatchNorm3d(out_size))

        block.append(nn.Conv3d(out_size, out_size, kernel_size=3,
                               padding=int(padding)))

        if not residual:
            block.append(nn.LeakyReLU())
            if batch_norm:
                block.append(nn.BatchNorm3d(out_size))
        self.block = nn.Sequential(*block)

    def forward(self, x):
        
        out = self.block(x)
        if self.residual:
            if self.in_size != self.out_size:
                x = self.residual_input_conv(x)
                x = self.residual_batchnorm(x)
            out = out + x
        
        
        
        return out


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, up_mode, padding, batch_norm, residual=False):
        super(UNetUpBlock, self).__init__()
        self.residual = residual
        self.in_size = in_size
        self.out_size = out_size
        self.residual_input_conv = nn.Conv3d(self.in_size, self.out_size, kernel_size=1)
        self.residual_batchnorm = nn.BatchNorm3d(self.out_size)

        if up_mode == 'upconv':
            self.up = nn.ConvTranspose3d(in_size, out_size, kernel_size=2,
                                         stride=2)
        elif up_mode == 'upsample':
            self.up = nn.Sequential(nn.Upsample(mode='nearest', scale_factor=2),
                                    nn.Conv3d(in_size, out_size, kernel_size=1))

        self.conv_block = UNetConvBlock(in_size, out_size, padding, batch_norm)

    @staticmethod
    def center_crop(upsampled, bypass):
        # _, _, layer_height, layer_width = layer.size()
        # diff_y = (layer_height - target_size[0]) // 2
        # diff_x = (layer_width - target_size[1]) // 2
        # return layer[:, :, diff_y:(diff_y + target_size[0]), diff_x:(diff_x + target_size[1])]
        if bypass.size()[3] != bypass.size()[4]:
            b=torch.argmax(torch.tensor(bypass.shape[3:5]))
            if b==1: 
                mm2 = torch.nn.ReplicationPad3d((0, 0, 0, 1, 0, 0))
            else:
                mm2 = torch.nn.ReplicationPad3d((0, 1, 0, 0, 0, 0))
            bypass=mm2(bypass)
        
        c = (bypass.size()[3] - upsampled.size()[3]) // 2 + (bypass.size()[3] - upsampled.size()[3]) % 2
        # print(upsampled.shape)
        # print(bypass.shape)
        # print(c)
        mm = nn.ReplicationPad3d((c//2, c//2 + c%2, c//2, c//2 + c%2, 0, 0))
        upsampled = mm(upsampled)
        # print(upsampled.shape)
        # print(bypass.shape)
        
        if bypass.shape[2] != upsampled.shape[2]:
            mm1 = nn.ReplicationPad3d((0, 0, 0, 0, 0, 1))
            upsampled = mm1(upsampled)

        return torch.cat([upsampled, bypass], 1)

    def forward(self, x, bridge):
        up = self.up(x)
        # print(x.shape)
        ## crop1 = self.center_crop(up, bridge)
        out_orig = self.center_crop(up, bridge)
        # print(crop1.shape)
        # print(bridge.shape)

        ## out_orig = torch.cat([crop1, bridge], 1)
        out = self.conv_block(out_orig)
        if self.residual:
            if self.in_size != self.out_size:
                out_orig = self.residual_input_conv(out_orig)
                out_orig = self.residual_batchnorm(out_orig)
            out = out + out_orig

        return out
