import torch
from torch import nn
import torch.nn.functional as F
import math




# 3D U-Net implementation in PyTorch

class HH_Unet3(nn.Module):
    def __init__(self, in_channel=1, conv_channel=8, out_channel=1, kernel_size=3, up_mode='upconv'):
        super().__init__()
        
        ### Tail
        self.tail_BN = nn.BatchNorm3d(in_channel)
        self.tail_block = BridgeBlock(in_channel, conv_channel, conv_channel, kernel_size=kernel_size)
        
        ### Body
        
        ## Encoding Path
        
        # 1st Layer
        self.en_block1 = EncodeBlock(conv_channel, conv_channel, conv_channel, kernel_size=kernel_size)
        
        # 2nd Layer
        self.en_block2 = EncodeBlock(conv_channel, conv_channel*2, conv_channel*2, kernel_size=kernel_size)
        
        # 3rd Layer
        self.en_block3 = EncodeBlock(conv_channel*2, conv_channel*4, conv_channel*4, kernel_size=kernel_size)
        
        
        
        ## Bridge
        self.br_block3 = BridgeBlock(conv_channel*4, conv_channel*8, conv_channel*4, kernel_size=kernel_size)
        
        

        ## Decoding Path
        
        # 1st Layer
        self.de_block3 = DecodeBlock(conv_channel*8, conv_channel*4, conv_channel*2, kernel_size= kernel_size, up_mode=up_mode)
        
        # 2nd Layer
        self.de_block2 = DecodeBlock(conv_channel*4, conv_channel*2, conv_channel, kernel_size=kernel_size, up_mode= up_mode)
    
        # 3rd Layer
        self.de_block1 = OutBlock(conv_channel*2, conv_channel, conv_channel, kernel_size=kernel_size, up_mode= up_mode)

        # Out Layer
        self.head_block = BridgeBlockOut(conv_channel*2, conv_channel, out_channel, kernel_size=kernel_size)
        
        self._init_weights()
    

    def forward(self, input_batch):
        # print(input_batch.shape)
        ## Tail
        block_out = self.tail_BN(input_batch)
        # print(block_out.size())
        tail_block_out=self.tail_block(block_out)
        # print(tail_block_out.size())
        
        ## Body
        # Encoding Path
        
        en1_block_out = self.en_block1(tail_block_out)
        # print(en1_block_out.size())
        en2_block_out = self.en_block2(en1_block_out)
        # print(en2_block_out.size())
        en3_block_out = self.en_block3(en2_block_out)
        # print(en3_block_out.size())
        # Bridge
        br_block_out = self.br_block3(en3_block_out)
        # print(br_block_out.size())
        
        # Decoding Path
        de3_block_out=self.de_block3(self.crop_and_concat(br_block_out, en3_block_out, crop=True))
        # print(de3_block_out.size())
        de2_block_out=self.de_block2(self.crop_and_concat(de3_block_out, en2_block_out, crop=True))
        # print(de2_block_out.size())
        
        # Head
        de1_block_out=self.de_block1(self.crop_and_concat(de2_block_out, en1_block_out, crop=True))
        # print(de3_block_out.size())
        block_out=self.head_block(self.crop_and_concat(de1_block_out, tail_block_out, crop=True))
        # print(block_out.size())

        return block_out
        
        
    def _init_weights(self):
        for m in self.modules():
            if type(m) in {
                nn.Linear,
                nn.Conv3d,
                nn.Conv2d,
                nn.ConvTranspose2d,
                nn.ConvTranspose3d,
            }:
                nn.init.kaiming_normal_(
                    m.weight.data, a=0, mode='fan_out', nonlinearity='leaky_relu',
                )
                if m.bias is not None:
                    fan_in, fan_out = \
                        nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)
        

    def crop_and_concat(self, upsampled, bypass, crop=False):
        if crop:
            c = (bypass.size()[3] - upsampled.size()[3]) // 2 + (bypass.size()[3] - upsampled.size()[3]) % 2
            # print(bypass.size())
            # print(upsampled.size())
            # print(c)
            # upsampled = F.pad(upsampled, (c//2, c//2 + c%2, c//2, c//2 + c%2),mode= 'replicate' )
            mm = nn.ReplicationPad3d((c//2, c//2 + c%2, c//2, c//2 + c%2, 0, 0))
            upsampled = mm(upsampled)

            if bypass.shape[2] != upsampled.shape[2]:
                mm1 = nn.ReplicationPad3d((0, 0, 0, 0, 0, 1))
                upsampled = mm1(upsampled)
            # print((c//2, c//2 + c%2, c//2, c//2 + c%2))
            # print(padvalue)
            # print(bypass.size())
        return torch.cat((upsampled, bypass), 1)
        
        
        
        
        
        
################ BLOCK DEFINITION ########################        
        
class EncodeBlock(nn.Module):
    def __init__(self, in_channels, mid_channel, out_channels, kernel_size=3):
        super().__init__()

        # 1st Convolution Layer
        self.conv1=nn.Conv3d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel, padding=1, bias=True)
        self.BN1=nn.BatchNorm3d(mid_channel)

        # 2nd Convolution Layer
        self.conv2=torch.nn.Conv3d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=out_channels, padding=1)
        self.BN2=torch.nn.BatchNorm3d(out_channels)

        # General Use
        self.maxpool = nn.MaxPool3d(2, 2)
        self.LeakyReLU = nn.LeakyReLU(inplace=True)

    def forward(self, input_batch):

        # 1st Convolution Layer
        block_out = self.conv1(input_batch)
        block_out = self.BN1(block_out)
        block_out = self.LeakyReLU(block_out)

        # 2nd Convolution Layer
        block_out = self.conv2(block_out)
        block_out = self.BN2(block_out)
        block_out = self.LeakyReLU(block_out)
        
        # Maxpooling
        block_out = self.maxpool(block_out)
        
        return block_out
    
    
class DecodeBlock(nn.Module):
        def __init__(self, in_channels, mid_channel, out_channels, kernel_size=3, up_mode='upconv'):
            super().__init__()

            # Up Convolution Layer
            if up_mode == 'upconv':
                self.upconv = nn.ConvTranspose3d(in_channels=in_channels, out_channels=in_channels, kernel_size=3,
                                                 stride=2, padding=1, output_padding=1)

            elif up_mode == 'upsample':
                self.upconv = nn.Sequential(nn.Upsample(mode='nearest', scale_factor=2),
                                            nn.Conv3d(in_channels=in_channels, out_channels=in_channels,
                                                      kernel_size=1))
                # self.upconv = nn.Sequential(F.interpolate(mode='bilinear', scale_factor=2),
                #                             nn.Conv3d(in_channels=in_channels, out_channels=in_channels,
                #                                       kernel_size=1))



            # 1st Convolution Layer
            self.conv1=nn.Conv3d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel, padding=1, bias=True)
            self.BN1=nn.BatchNorm3d(mid_channel)

            # 2nd Convolution Layer
            self.conv2=torch.nn.Conv3d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=out_channels, padding=1)
            self.BN2=torch.nn.BatchNorm3d(out_channels)
            

        
        

            # General Use
            self.maxpool = nn.MaxPool3d(2, 2)
            self.LeakyReLU = nn.LeakyReLU(inplace=True)
            
            
            
        def forward(self, input_batch):

            # Up Convolution/Sampling
            block_out = self.upconv(input_batch)


            # 1st Convolution Layer
            block_out = self.conv1(block_out)
            block_out = self.BN1(block_out)
            block_out = self.LeakyReLU(block_out)


            # 2nd Convolution Layer
            block_out = self.conv2(block_out)
            block_out = self.BN2(block_out)
            block_out = self.LeakyReLU(block_out)



            return block_out

            
                
                
                
class BridgeBlock(nn.Module):
    
    def __init__(self, in_channels, mid_channel, out_channels, kernel_size=3):
        super().__init__()

        self.conv1 = torch.nn.Conv3d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel, padding=1)
        self.BN1 = torch.nn.BatchNorm3d(mid_channel)

        self.conv2 = torch.nn.Conv3d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=out_channels, padding=1)
        self.BN2 = torch.nn.BatchNorm3d(out_channels)
        
        # # Up Convolution Layer
        # if up_mode == 'upconv':
        #     self.upconv = nn.ConvTranspose3d(in_channels=mid_channel, out_channels=out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        #
        # elif up_mode == 'upsample':
        #     self.upconv = nn.Sequential(nn.Upsample(mode='bilinear', scale_factor=2),
        #                         nn.Conv3d(in_channels=mid_channel, out_channels=out_channels, kernel_size=1))
        # General Use
        self.maxpool = nn.MaxPool3d(2, 2)
        self.LeakyReLU = nn.LeakyReLU(inplace=True)
        
        
        
    def forward(self,input_batch):
         
            
        # 1st Convolution Layer
        block_out = self.conv1(input_batch)
        block_out = self.BN1(block_out)
        block_out = self.LeakyReLU(block_out) 


        # 2nd Convolution Layer
        block_out = self.conv2(block_out)
        block_out = self.BN2(block_out)
        block_out = self.LeakyReLU(block_out)

        # # Up Convolution/Sampling
        # block_out = self.upconv(block_out)

        return block_out       



class BridgeBlockOut(nn.Module):
    
    def __init__(self, in_channels, mid_channel, out_channels, kernel_size=3):
        super().__init__()

        self.conv1 = torch.nn.Conv3d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel, padding=1)
        self.BN1 = torch.nn.BatchNorm3d(mid_channel)

        self.conv2 = torch.nn.Conv3d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=out_channels, padding=1)
        self.BN2 = torch.nn.BatchNorm3d(out_channels)

        self.conv3 = torch.nn.Conv3d(kernel_size=1, in_channels=out_channels, out_channels=out_channels, padding=0)
        self.BN3 = torch.nn.BatchNorm3d(out_channels)
        
        # # Up Convolution Layer
        # if up_mode == 'upconv':
        #     self.upconv = nn.ConvTranspose3d(in_channels=mid_channel, out_channels=out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        #
        # elif up_mode == 'upsample':
        #     self.upconv = nn.Sequential(nn.Upsample(mode='bilinear', scale_factor=2),
        #                         nn.Conv3d(in_channels=mid_channel, out_channels=out_channels, kernel_size=1))
        # General Use
        self.maxpool = nn.MaxPool3d(2, 2)
        # self.linear = nn.Linear()
        
        
        
    def forward(self,input_batch):
         
            
        # 1st Convolution Layer
        block_out = self.conv1(input_batch)
        block_out = self.BN1(block_out)
        # block_out = self.linear(block_out) 


        # 2nd Convolution Layer
        block_out = self.conv2(block_out)
        block_out = self.BN2(block_out)
        # block_out = self.linear(block_out)


        # 2nd Convolution Layer
        block_out = self.conv3(block_out)
        block_out = self.BN3(block_out)
        # block_out = self.linear(block_out)
        # # Up Convolution/Sampling
        # block_out = self.upconv(block_out)

        return block_out       
        
    
class OutBlock(nn.Module):

    def __init__(self, in_channels, mid_channel, out_channels, kernel_size=3, up_mode = 'upconv'):
        super().__init__()

        # Up Convolution Layer
        if up_mode == 'upconv':
            self.upconv = nn.ConvTranspose3d(in_channels=in_channels, out_channels=in_channels, kernel_size=3,
                                             stride=2, padding=1, output_padding=1)

        elif up_mode == 'upsample':
            self.upconv = nn.Sequential(nn.Upsample(mode='nearest', scale_factor=2),
                                        nn.Conv3d(in_channels=in_channels, out_channels=in_channels,
                                                  kernel_size=1))
            # self.upconv = nn.Sequential(F.interpolate(mode='bilinear', scale_factor=2),
            #                                 nn.Conv3d(in_channels=in_channels, out_channels=in_channels,
            #                                           kernel_size=1))


        # 1st Convolution Layer
        self.conv1 = nn.Conv3d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel, padding=1, bias=True)
        self.BN1 = nn.BatchNorm3d(mid_channel)

        # 2nd Convolution Layer
        self.conv2 = torch.nn.Conv3d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=mid_channel, padding=1)
        self.BN2 = torch.nn.BatchNorm3d(mid_channel)

        # 3rd Convolution Layer
        self.conv3 = torch.nn.Conv3d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=out_channels, padding=1)
        # self.BN3=torch.nn.BatchNorm3d(mid_channel)

        # General Use
        self.maxpool = nn.MaxPool3d(2, 2)
        self.LeakyReLU = nn.LeakyReLU(inplace=True)
            
            
            
    def forward(self,input_batch):

        # Up Convolution/Sampling
        block_out = self.upconv(input_batch)
            
        # 1st Convolution Layer
        block_out = self.conv1(block_out)
        block_out = self.BN1(block_out)
        block_out = self.LeakyReLU(block_out) 


        # 2nd Convolution Layer
        block_out = self.conv2(block_out)
        block_out = self.BN2(block_out)
        block_out = self.LeakyReLU(block_out)
        
        # 3rd Convolution Layer
        block_out = self.conv3(block_out)
        # block_out = self.BN2(block_out)
        block_out = self.LeakyReLU(block_out)
        
        return block_out  

class Interpolate(nn.Module):
    def __init__(self, scale=2, mode='bilinear'):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.scale = scale
        self.mode = mode
        
    def forward(self, x):
        x = self.interp(x, mode='bilinear', scale_factor=2)
        return x

class HeadBlock(nn.Module):
    
    def __init__(self, in_channels, mid_channel, out_channels, kernel_size=3):
        super().__init__()

        self.conv1 = torch.nn.Conv3d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel, padding=1)
        self.BN1 = torch.nn.BatchNorm3d(mid_channel)

        self.conv2 = torch.nn.Conv3d(kernel_size=1, in_channels=mid_channel, out_channels=out_channels, padding=0)
        self.BN2 = torch.nn.BatchNorm3d(out_channels)
        
        # # Up Convolution Layer
        # if up_mode == 'upconv':
        #     self.upconv = nn.ConvTranspose3d(in_channels=mid_channel, out_channels=out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        #
        # elif up_mode == 'upsample':
        #     self.upconv = nn.Sequential(nn.Upsample(mode='bilinear', scale_factor=2),
        #                         nn.Conv3d(in_channels=mid_channel, out_channels=out_channels, kernel_size=1))
        # General Use
        self.maxpool = nn.MaxPool3d(2, 2)
        self.LeakyReLU = nn.LeakyReLU(inplace=True)
        
        
        
    def forward(self,input_batch):
         
            
        # 1st Convolution Layer
        block_out = self.conv1(input_batch)
        block_out = self.BN1(block_out)
        block_out = self.LeakyReLU(block_out) 


        # 2nd Convolution Layer
        block_out = self.conv2(block_out)
        # block_out = self.BN2(block_out)
        # block_out = self.LeakyReLU(block_out)

        # # Up Convolution/Sampling
        # block_out = self.upconv(block_out)

        return block_out