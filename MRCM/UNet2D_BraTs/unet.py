import torch
import torch.nn as nn
import torch.nn.functional as F


class Downsample_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Downsample_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.pooling = nn.MaxPool2d(2,2)
        self.relu = nn.ReLU()

    def forward(self, x ):
        x = self.relu(self.bn1(self.conv1(x)))
        y = self.relu(self.bn2(self.conv2(x)))
        x = self.pooling(y)

        return x, y


class Upsample_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample_block, self).__init__()
        self.transconv = nn.ConvTranspose2d(in_channels, out_channels, 4, padding=1, stride=2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        
    def forward(self, x, y):
        x = self.transconv(x)
        x = torch.cat((x, y), dim=1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))

        return x


class UnetB(nn.Module):
    def __init__(self):
        in_chan = 1
        out_chan = 1
        h_chan = 128
        super(UnetB, self).__init__()
        self.down1 = Downsample_block(in_chan, h_chan)
        self.down2 = Downsample_block(h_chan, h_chan*2)
        self.down3 = Downsample_block(h_chan*2, h_chan*4)
        self.down4 = Downsample_block(h_chan*4, h_chan*8)
        self.conv1 = nn.Conv2d(h_chan*8, h_chan*16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(h_chan*16)
        self.conv2 = nn.Conv2d(h_chan*16, h_chan*16, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(h_chan*16)
        self.up4 = Upsample_block(h_chan*16, h_chan*8)
        self.up3 = Upsample_block(h_chan*8, h_chan*4)
        self.up2 = Upsample_block(h_chan*4, h_chan*2)
        self.up1 = Upsample_block(h_chan*2, h_chan)
        self.outconv = nn.Conv2d(h_chan, out_chan, 1)
        self.outconvp1 = nn.Conv2d(h_chan, out_chan, 1)
        self.outconvm1 = nn.Conv2d(h_chan, out_chan, 1)
        self.dropout2d = nn.Dropout2d()
        self.relu = nn.ReLU()

    def forward(self, x):
        x, y1 = self.down1(x)
        x, y2 = self.down2(x)
        x, y3 = self.down3(x)
        x, y4 = self.down4(x)
        x = self.dropout2d(self.relu(self.bn1(self.conv1(x))))
        x = self.dropout2d(self.relu(self.bn2(self.conv2(x))))
        x = self.up4(x, y4)
        x = self.up3(x, y3)
        x = self.up2(x, y2)
        x = self.up1(x, y1)
        x1 = self.outconv(x)

        return x1