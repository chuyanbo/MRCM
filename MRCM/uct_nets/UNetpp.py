import torch.nn as nn
import torch

def get_activation(activation_type):
    activation_type = activation_type.lower()
    if hasattr(nn, activation_type):
        return getattr(nn, activation_type)()
    else:
        return nn.ReLU()

def _make_nConv(in_channels, out_channels, nb_Conv, activation='ReLU'):
    layers = []
    layers.append(ConvBatchNorm(in_channels, out_channels, activation))

    for _ in range(nb_Conv - 1):
        layers.append(ConvBatchNorm(out_channels, out_channels, activation))
    return nn.Sequential(*layers)

class ConvBatchNorm(nn.Module):
    """(convolution => [BN] => ReLU)"""

    def __init__(self, in_channels, out_channels, activation='ReLU'):
        super(ConvBatchNorm, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(activation)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out)

class DownBlock(nn.Module):
    """Downscaling with maxpool convolution"""

    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super(DownBlock, self).__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x):
        out = self.maxpool(x)
        return self.nConvs(out)

class UpBlock2(nn.Module):
    """Upscaling then conv"""

    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super(UpBlock2, self).__init__()

        # self.up = nn.Upsample(scale_factor=2)
        self.up = nn.ConvTranspose2d(in_channels,in_channels//2,(2,2),2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x, skip_x):
        out = self.up(x)
        x = torch.cat([out, skip_x], dim=1)  # dim 1 is the channel dimension
        x = self.nConvs(x)
        return x

class UpBlock3(nn.Module):
    """Upscaling then conv"""

    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super(UpBlock3, self).__init__()

        # self.up = nn.Upsample(scale_factor=2)
        self.up = nn.ConvTranspose2d(in_channels,in_channels//2,(2,2),2)
        self.nConvs = _make_nConv(in_channels//2*3, out_channels, nb_Conv, activation)

    def forward(self, x, skip_x0,skip_x1):
        out = self.up(x)
        x = torch.cat([out, skip_x0, skip_x1], dim=1)  # dim 1 is the channel dimension
        x = self.nConvs(x)
        return x
    
class UpBlock4(nn.Module):
    """Upscaling then conv"""

    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super(UpBlock4, self).__init__()

        # self.up = nn.Upsample(scale_factor=2)
        self.up = nn.ConvTranspose2d(in_channels,in_channels//2,(2,2),2)
        self.nConvs = _make_nConv(in_channels//2*4, out_channels, nb_Conv, activation)

    def forward(self, x, skip_x0, skip_x1, skip_x2):
        out = self.up(x)
        x = torch.cat([out, skip_x0, skip_x1, skip_x2], dim=1)  # dim 1 is the channel dimension
        x = self.nConvs(x)
        return x

class UpBlock5(nn.Module):
    """Upscaling then conv"""

    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super(UpBlock5, self).__init__()

        # self.up = nn.Upsample(scale_factor=2)
        self.up = nn.ConvTranspose2d(in_channels,in_channels//2,(2,2),2)
        self.nConvs = _make_nConv(in_channels//2*5, out_channels, nb_Conv, activation)

    def forward(self, x, skip_x0, skip_x1, skip_x2, skip_x3):
        out = self.up(x)
        x = torch.cat([out, skip_x0, skip_x1, skip_x2, skip_x3], dim=1)  # dim 1 is the channel dimension
        x = self.nConvs(x)
        return x

class UNetpp(nn.Module):
    def __init__(self, n_channels=1, n_classes=1):
        '''
        n_channels : number of channels of the input.
                        By default 3, because we have RGB images
        n_labels : number of channels of the ouput.
                      By default 3 (2 labels + 1 for the background)
        '''
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        # Question here
        in_channels = 64
        self.inc = ConvBatchNorm(n_channels, in_channels)
        self.down1 = DownBlock(in_channels, in_channels*2, nb_Conv=2)
        self.down2 = DownBlock(in_channels*2, in_channels*4, nb_Conv=2)
        self.down3 = DownBlock(in_channels*4, in_channels*8, nb_Conv=2)
        self.down4 = DownBlock(in_channels*8, in_channels*16, nb_Conv=2)
        self.up31 = UpBlock2(in_channels*16, in_channels*8, nb_Conv=2)
        self.up21 = UpBlock2(in_channels*8, in_channels*4, nb_Conv=2)
        self.up22 = UpBlock3(in_channels*8, in_channels*4, nb_Conv=2)
        self.up11 = UpBlock2(in_channels*4, in_channels*2, nb_Conv=2)
        self.up12 = UpBlock3(in_channels*4, in_channels*2, nb_Conv=2)
        self.up13 = UpBlock4(in_channels*4, in_channels*2, nb_Conv=2)
        self.up01 = UpBlock2(in_channels*2, in_channels, nb_Conv=2)
        self.up02 = UpBlock3(in_channels*2, in_channels, nb_Conv=2)
        self.up03 = UpBlock4(in_channels*2, in_channels, nb_Conv=2)
        self.up04 = UpBlock5(in_channels*2, in_channels, nb_Conv=2)
        self.outc = nn.Conv2d(in_channels, n_classes, kernel_size=(1,1))
        if n_classes == 1:
            self.last_activation = nn.Sigmoid()
        else:
            self.last_activation = None

    def forward(self, x):
        # Question here
        x = x.float()
        x00 = self.inc(x)
        x10 = self.down1(x00)
        x20 = self.down2(x10)
        x30 = self.down3(x20)
        x40 = self.down4(x30)
        # print(x00.shape,x10.shape,x20.shape,x30.shape,x40.shape)
        
        x31 = self.up31(x40, x30)
        # print(x31.shape)
        x21 = self.up21(x30, x20) #x30和x20维度不一样？
        x22 = self.up22(x31, x20, x21) # 问题，x21维度和x20并不一样
        # print(x21.shape,x22.shape)
        x11 = self.up11(x20, x10)
        x12 = self.up12(x21, x10, x11)
        x13 = self.up13(x22, x10, x11, x12)
        # print(x11.shape,x12.shape,x13.shape)
        x01 = self.up01(x10, x00)
        x02 = self.up02(x11, x00, x01)
        x03 = self.up03(x12, x00, x01, x02)
        x04 = self.up04(x13, x00, x01, x02, x03)
        # print(x01.shape,x02.shape,x03.shape,x04.shape)
        
        x = x04 # 问题，目前还没有搞清楚loss函数
        if self.last_activation is not None:
            logits = self.last_activation(self.outc(x))
            # print("111")
        else:
            logits = self.outc(x)
            # print("222")
        # logits = self.outc(x) # if using BCEWithLogitsLoss
        # print(logits.size())
        return logits

if __name__=="__main__":
    batch=2
    channel_coler=1
    pixel_length=256
    x=torch.randn(batch,channel_coler,pixel_length,pixel_length)
    num_class=1
    net=UNetpp()
    # print(net)
    y=net(x)
    print(x.shape,y.shape,"  Unet++ tested  ")