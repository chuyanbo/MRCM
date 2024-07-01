# -*- coding: utf-8 -*-
# @Time    : 2021/7/8 8:59 上午
# @File    : UCTransNet.py
# @Software: PyCharm
import torch.nn as nn
import torch
import torch.nn.functional as F
from  .CTrans import ChannelTransformer
# from .DCUnet import 

def Conv2dSame(in_channels, out_channels, kernel_size, use_bias=True, padding_layer=torch.nn.ReflectionPad2d):
    ka = kernel_size // 2
    kb = ka - 1 if kernel_size % 2 == 0 else ka
    return [
        padding_layer((ka, kb, ka, kb)),
        torch.nn.Conv2d(in_channels, out_channels, kernel_size, bias=use_bias)
    ]


def conv2d_bn(in_channels, filters, kernel_size, padding='same', activation='relu'):
    assert padding == 'same'
    affine = False if activation == 'relu' or activation == 'sigmoid' else True
    sequence = []
    sequence += Conv2dSame(in_channels, filters, kernel_size, use_bias=False)
    sequence += [torch.nn.BatchNorm2d(filters, affine=affine)]
    if activation == "relu":
        sequence += [torch.nn.ReLU()]
    elif activation == "sigmoid":
        sequence += [torch.nn.Sigmoid()]
    elif activation == 'tanh':
        sequence += [torch.nn.Tanh()]
    return torch.nn.Sequential(*sequence)


class DCBlock(torch.nn.Module):
    def __init__(self, in_channels, u, alpha=1.67, use_dropout=False):
        super().__init__()
        w = alpha * u
        self.out_channel = int(w * 0.167) + int(w * 0.333) + int(w * 0.5)
        self.conv2d_bn = conv2d_bn(in_channels, self.out_channel, 1, activation=None)
        self.conv3x3 = conv2d_bn(in_channels, int(w * 0.167), 3, activation='relu')
        self.conv5x5 = conv2d_bn(int(w * 0.167), int(w * 0.333), 3, activation='relu')
        self.conv7x7 = conv2d_bn(int(w * 0.333), int(w * 0.5), 3, activation='relu')
        
        self.conv3x3_2 = conv2d_bn(in_channels, int(w * 0.167), 3, activation='relu')
        self.conv5x5_2 = conv2d_bn(int(w * 0.167), int(w * 0.333), 3, activation='relu')
        self.conv7x7_2 = conv2d_bn(int(w * 0.333), int(w * 0.5), 3, activation='relu')
        
        self.bn_1 = torch.nn.BatchNorm2d(self.out_channel)
        self.bn_1_2 = torch.nn.BatchNorm2d(self.out_channel)
        self.relu = torch.nn.ReLU()
        self.bn_2 = torch.nn.BatchNorm2d(self.out_channel)
        self.use_dropout = use_dropout
        if use_dropout:
            self.dropout = torch.nn.Dropout(0.5)

    def forward(self, inp):
        if self.use_dropout:
            x = self.dropout(inp)
        else:
            x = inp
        shortcut = self.conv2d_bn(x)
        conv3x3 = self.conv3x3(x)
        conv5x5 = self.conv5x5(conv3x3)
        conv7x7 = self.conv7x7(conv5x5)
        out = torch.cat([conv3x3, conv5x5, conv7x7], dim=1)
        out = self.bn_1(out)
        
        conv3x3_2 = self.conv3x3_2(x)
        conv5x5_2 = self.conv5x5_2(conv3x3_2)
        conv7x7_2 = self.conv7x7_2(conv5x5_2)
        out_2 = torch.cat([conv3x3_2, conv5x5_2, conv7x7_2], dim=1)
        out_2 = self.bn_1_2(out_2)
        
        out_f = shortcut + out + out_2
        out_f = self.relu(out_f)
        out_f = self.bn_2(out_f)
        return out_f


class ResPathBlock(torch.nn.Module):
    def __init__(self, in_channels, filters):
        super(ResPathBlock, self).__init__()
        self.conv2d_bn1 = conv2d_bn(in_channels, filters, 1, activation=None)
        self.conv2d_bn2 = conv2d_bn(in_channels, filters, 3, activation='relu')
        self.relu = torch.nn.ReLU()
        self.bn = torch.nn.BatchNorm2d(filters)

    def forward(self, inp):
        shortcut = self.conv2d_bn1(inp)
        out = self.conv2d_bn2(inp)
        out = torch.add(shortcut, out)
        out = self.relu(out)
        out = self.bn(out)
        return out

def get_activation(activation_type):
    activation_type = activation_type.lower()
    if hasattr(nn, activation_type):
        return getattr(nn, activation_type)()
    else:
        return nn.ReLU()

def _make_nConv(in_channels, out_channels, nb_Conv, activation='ReLU',k_size=3):
    layers = []
    layers.append(ConvBatchNorm(in_channels, out_channels, activation,k_size))

    for _ in range(nb_Conv - 1):
        layers.append(ConvBatchNorm(out_channels, out_channels, activation,k_size))
    return nn.Sequential(*layers)

class ConvBatchNorm(nn.Module):
    """(convolution => [BN] => ReLU)"""

    def __init__(self, in_channels, out_channels, activation='ReLU',k_size=3):
        super(ConvBatchNorm, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=k_size, padding=int(k_size//2))
        # self.conv = nn.Conv2d(in_channels, out_channels,
        #                       kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(activation)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out)

class DownBlock_old(nn.Module):
    """Downscaling with maxpool convolution"""
    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super(DownBlock_old, self).__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.nConvs33 = _make_nConv(in_channels, out_channels, nb_Conv, activation,k_size=3)
        self.nConvs33_1 = _make_nConv(in_channels, out_channels, nb_Conv-1, activation,k_size=3)
        self.nConvs33_2 = _make_nConv(in_channels, out_channels, nb_Conv-2, activation,k_size=3)
        
        # self.nConvs55 = _make_nConv(in_channels, out_channels, nb_Conv, activation,k_size=5)
        # self.nConvs55_1 = _make_nConv(in_channels, out_channels, nb_Conv-1, activation,k_size=5)
        # self.nConvs55_2 = _make_nConv(in_channels, out_channels, nb_Conv-2, activation,k_size=5)

    def forward(self, x):
        out = self.maxpool(x)
        conv33 = self.nConvs33(out) + self.nConvs33_1(out) + self.nConvs33_2(out)
        # conv55 = self.nConvs55(out) + self.nConvs55_1(out) + self.nConvs55_2(out)
        return conv33

class DownBlock(nn.Module):
    """Downscaling with maxpool convolution"""
    def __init__(self, in_channels, out_channels , nb_Conv, activation='ReLU'):
        super(DownBlock, self).__init__()
        self.mres_block = DCBlock(in_channels, u=out_channels)
        self.maxpool = nn.MaxPool2d(2)
        self.out_channel = self.mres_block.out_channel
        self.nConvs33 = _make_nConv(self.mres_block.out_channel, out_channels, 1 , activation,k_size=3)
        # self.nConvs55 = _make_nConv(in_channels, out_channels, nb_Conv, activation,k_size=5)
        # self.nConvs55_1 = _make_nConv(in_channels, out_channels, nb_Conv-1, activation,k_size=5)
        # self.nConvs55_2 = _make_nConv(in_channels, out_channels, nb_Conv-2, activation,k_size=5)

    def forward(self, x):
        x = self.mres_block(x)
        x = self.maxpool(x)
        out = self.nConvs33(x)
        return out

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class CCA(nn.Module):
    """
    CCA Block
    """
    def __init__(self, F_g, F_x):
        super().__init__()
        self.mlp_x = nn.Sequential(
            Flatten(),
            nn.Linear(F_x, F_x))
        self.mlp_g = nn.Sequential(
            Flatten(),
            nn.Linear(F_g, F_x))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # channel-wise attention
        avg_pool_x = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        channel_att_x = self.mlp_x(avg_pool_x)
        avg_pool_g = F.avg_pool2d( g, (g.size(2), g.size(3)), stride=(g.size(2), g.size(3)))
        channel_att_g = self.mlp_g(avg_pool_g)
        channel_att_sum = (channel_att_x + channel_att_g)/2.0
        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        x_after_channel = x * scale
        out = self.relu(x_after_channel)
        return out

class UpBlock_attention(nn.Module):
    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.coatt = CCA(F_g=in_channels//2, F_x=in_channels//2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x, skip_x):
        up = self.up(x)
        skip_x_att = self.coatt(g=up, x=skip_x)
        x = torch.cat([skip_x_att, up], dim=1)  # dim 1 is the channel dimension
        return self.nConvs(x)

class ResUCTransNet(nn.Module):
    def __init__(self, config,n_channels=3, n_classes=1,img_size=224,vis=False):
        super().__init__()
        self.vis = vis
        self.n_channels = n_channels
        self.n_classes = n_classes
        in_channels = config.base_channel
        self.inc = ConvBatchNorm(n_channels, in_channels)
        self.down1 = DownBlock(in_channels, in_channels*2, nb_Conv=3)
        self.down2 = DownBlock(in_channels*2, in_channels*4, nb_Conv=3)
        self.down3 = DownBlock(in_channels*4, in_channels*8, nb_Conv=3)
        self.down4 = DownBlock(in_channels*8, in_channels*8, nb_Conv=3)
        self.mtc = ChannelTransformer(config, vis, img_size,
                                     channel_num=[in_channels, in_channels*2, in_channels*4, in_channels*8],
                                     patchSize=config.patch_sizes)
        self.up4 = UpBlock_attention(in_channels*16, in_channels*4, nb_Conv=3)
        self.up3 = UpBlock_attention(in_channels*8, in_channels*2, nb_Conv=3)
        self.up2 = UpBlock_attention(in_channels*4, in_channels, nb_Conv=3)
        self.up1 = UpBlock_attention(in_channels*2, in_channels, nb_Conv=3)
        self.outc = nn.Conv2d(in_channels, n_classes, kernel_size=(1,1), stride=(1,1))
        self.last_activation = nn.Sigmoid() # if using BCELoss
        self.last_activation2 = nn.ReLU() # if using BCELoss

    def forward(self, x):
        x = x.float()
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x1,x2,x3,x4,att_weights = self.mtc(x1,x2,x3,x4)
        x = self.up4(x5, x4)
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        if self.n_classes ==1:
            logits = self.last_activation(self.outc(x))
            #logits = self.outc(x)
        else:
            logits = self.outc(x) # if nusing BCEWithLogitsLoss or class>1
        if self.vis: # visualize the attention maps
            return logits, att_weights
        else:
            return logits
        





