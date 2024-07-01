import torch
from torch import nn
from torch.nn import functional as F
from uct_nets.UCTransNet import UCTransNet

#Unet 3 channel 256*256 ==> num_class channel 256*256
class Conv_Block(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(Conv_Block, self).__init__()
        self.layer=nn.Sequential(
            nn.Conv2d(in_channel,out_channel,3,1,1,padding_mode='reflect',bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Dropout2d(0.3),
            nn.LeakyReLU(),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Dropout2d(0.3),
            nn.LeakyReLU()
        )
    def forward(self,x):
        return self.layer(x)


class DownSample(nn.Module):
    def __init__(self,channel):
        super(DownSample, self).__init__()
        self.layer=nn.Sequential(
            nn.Conv2d(channel,channel,3,2,1,padding_mode='reflect',bias=False),
            nn.BatchNorm2d(channel),
            nn.LeakyReLU()
        )
    def forward(self,x):
        return self.layer(x)


class UpSample(nn.Module):
    def __init__(self,channel):
        super(UpSample, self).__init__()
        self.layer=nn.Conv2d(channel,channel//2,1,1)
    def forward(self,x,feature_map):
        up=F.interpolate(x,scale_factor=2,mode='nearest')
        out=self.layer(up)
        return torch.cat((out,feature_map),dim=1)


class UNet(nn.Module):
    def __init__(self,ch_in,num_classes):
        super(UNet, self).__init__()
        self.c1=Conv_Block(ch_in,64)
        self.d1=DownSample(64)
        self.c2=Conv_Block(64,128)
        self.d2=DownSample(128)
        self.c3=Conv_Block(128,256)
        self.d3=DownSample(256)
        self.c4=Conv_Block(256,512)
        self.d4=DownSample(512)
        self.c5=Conv_Block(512,1024)
        self.u1=UpSample(1024)
        self.c6=Conv_Block(1024,512)
        self.u2 = UpSample(512)
        self.c7 = Conv_Block(512, 256)
        self.u3 = UpSample(256)
        self.c8 = Conv_Block(256, 128)
        self.u4 = UpSample(128)
        self.c9 = Conv_Block(128, 64)
        self.out=nn.Conv2d(64,num_classes,3,1,1)

    def forward(self,x):
        R1=self.c1(x)
        R2=self.c2(self.d1(R1))
        R3 = self.c3(self.d2(R2))
        R4 = self.c4(self.d3(R3))
        R5 = self.c5(self.d4(R4))
        O1=self.c6(self.u1(R5,R4))
        O2 = self.c7(self.u2(O1, R3))
        O3 = self.c8(self.u3(O2, R2))
        O4 = self.c9(self.u4(O3, R1))

        return self.out(O4)

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten,self).__init__()
    def forward(self,input):
        return input.view(input.size(0),-1)

class ResBlk(nn.Module):
    def __init__(self,ch_in,ch_out,stride_setting=1):
        super(ResBlk,self).__init__()
        self.conv1=nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=stride_setting,padding=1)
        self.bn1=nn.BatchNorm2d(ch_out)
        self.conv2=nn.Conv2d(ch_out,ch_out,kernel_size=3,stride=1,padding=1)
        self.bn2=nn.BatchNorm2d(ch_out)
        
        self.extra=nn.Sequential()
        if ch_out != ch_in:
            self.extra=nn.Sequential(
                nn.Conv2d(ch_in,ch_out,kernel_size=1,stride=stride_setting,padding=0),
                nn.BatchNorm2d(ch_out)
            )
    def forward(self,x):
        out=F.relu(self.bn1(self.conv1(x)))
        out=self.bn2(self.conv2(out))
        out=self.extra(x)+out             
        return out
    
class Resnet18(nn.Module):
    def __init__(self,num_class):
        super(Resnet18,self).__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(1,64,kernel_size=5,stride=4,padding=2),
            nn.BatchNorm2d(64)
        )
        self.blk1=ResBlk(64,128,stride_setting=2)
        self.blk2=ResBlk(128,256,stride_setting=2)
        self.blk3=ResBlk(256,512,stride_setting=2)
        self.blk4=ResBlk(512,1024,stride_setting=2)
        self.flatten=Flatten()
        self.outlayer=nn.Linear(512,num_class)
        
    def forward(self,x):
        x=F.relu(self.conv1(x))
        x=self.blk1(x)
        x=F.relu(x)
        x=self.blk2(x)
        x=self.blk3(x)
        #x=self.blk4(x)
        x=F.adaptive_avg_pool2d(x,[1,1])
        x=self.flatten(x)
        logist=self.outlayer(x)
        return logist

def debug_Unet():
    batch=2
    channel_coler=1
    pixel_length=640
    x=torch.randn(batch,channel_coler,pixel_length,pixel_length)
    num_class=1
    net=UNet(channel_coler,num_class)
    y=net(x)
    print(x.shape,y.shape)

def debug_Unet2(config):
    batch=2
    channel_coler=1
    pixel_length=256
    x=torch.randn(batch,channel_coler,pixel_length,pixel_length)
    num_class=1
    net=UCTransNet(config,n_channels=channel_coler,n_classes=num_class,img_size=256)
    y=net(x)
    print(x.shape,y.shape,"  Unet2 tested  ")

class config():
    def __init__(self,ratio = 1,trans_ratio = 1):
        super(config, self).__init__()
        self.ratio = ratio
        self.trans_ratio = trans_ratio
        self.KV_size = int(960 * self.ratio)
        self.transformer_num_heads  = int(4 * self.trans_ratio)
        self.transformer_num_layers = int(4 * self.trans_ratio)
        self.expand_ratio           = int(4 * 2 * self.ratio) # MLP channel dimension expand ratio
        self.transformer_embeddings_dropout_rate = 0.1
        self.transformer_attention_dropout_rate = 0.1
        self.transformer_dropout_rate = 0
        self.patch_sizes = [16,8,4,2]
        self.base_channel = int(64 * self.ratio) # base channel of U-Net
        self.n_classes = 7
        
def debug_Resnet():
    batch=32
    channel_coler=1
    pixel_length=256
    x=torch.randn(batch,channel_coler,pixel_length,pixel_length)
    num_class=2
    net=Resnet18(num_class)
    y=net(x)
    print(x.shape,y.shape)

if __name__ == '__main__':
    config1=config()
    debug_Unet2(config1)
