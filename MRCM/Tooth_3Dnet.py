import torch
import torch.nn as nn
import torch.nn.functional as F

######################_______Unet_______################################
class UNet(nn.Module):
    def __init__(self, in_channel=1, out_channel=2, training=True,out_scale=(1,1,1)):
        super(UNet, self).__init__()
        self.training = training
        self.encoder1 = nn.Conv3d(in_channel, 32, 3, stride=1, padding=1)  # b, 16, 10, 10
        self.encoder2=   nn.Conv3d(32, 64, 3, stride=1, padding=1)  # b, 8, 3, 3
        self.encoder3=   nn.Conv3d(64, 128, 3, stride=1, padding=1)
        self.encoder4=   nn.Conv3d(128, 256, 3, stride=1, padding=1)
        # self.encoder5=   nn.Conv3d(256, 512, 3, stride=1, padding=1)
        
        # self.decoder1 = nn.Conv3d(512, 256, 3, stride=1,padding=1)  # b, 16, 5, 5
        self.decoder2 =   nn.Conv3d(256, 128, 3, stride=1, padding=1)  # b, 8, 15, 15
        self.decoder3 =   nn.Conv3d(128, 64, 3, stride=1, padding=1)  # b, 1, 28, 28
        self.decoder4 =   nn.Conv3d(64, 32, 3, stride=1, padding=1)
        self.decoder5 =   nn.Conv3d(32, 2, 3, stride=1, padding=1)
        
        self.map4 = nn.Sequential(
            nn.Conv3d(2, out_channel, 1, 1),
            nn.Upsample(scale_factor=out_scale, mode='trilinear'),
            nn.Softmax(dim =1)
        )

        # 128*128 尺度下的映射
        self.map3 = nn.Sequential(
            nn.Conv3d(64, out_channel, 1, 1),
            nn.Upsample(scale_factor=(out_scale[0]*4,out_scale[1]*4,out_scale[2]*4), mode='trilinear'),
            nn.Softmax(dim =1)
        )

        # 64*64 尺度下的映射
        self.map2 = nn.Sequential(
            nn.Conv3d(128, out_channel, 1, 1),
            nn.Upsample(scale_factor=(out_scale[0]*8,out_scale[1]*8,out_scale[2]*8), mode='trilinear'),
            nn.Softmax(dim =1)
        )

        # 32*32 尺度下的映射
        self.map1 = nn.Sequential(
            nn.Conv3d(256, out_channel, 1, 1),
            nn.Upsample(scale_factor=(out_scale[0]*16,out_scale[1]*16,out_scale[2]*16), mode='trilinear'),
            nn.Softmax(dim =1)
        )

    def forward(self, x):

        out = F.relu(F.max_pool3d(self.encoder1(x),2,2))
        t1 = out
        out = F.relu(F.max_pool3d(self.encoder2(out),2,2))
        t2 = out
        out = F.relu(F.max_pool3d(self.encoder3(out),2,2))
        t3 = out
        out = F.relu(F.max_pool3d(self.encoder4(out),2,2))
        # t4 = out
        # out = F.relu(F.max_pool3d(self.encoder5(out),2,2))
        # t2 = out
        # out = F.relu(F.interpolate(self.decoder1(out),scale_factor=(2,2,2),mode ='trilinear'))
        # print(out.shape,t4.shape)
        output1 = self.map1(out)
        out = F.relu(F.interpolate(self.decoder2(out),scale_factor=(2,2,2),mode ='trilinear'))
        out = torch.add(out,t3)
        output2 = self.map2(out)
        out = F.relu(F.interpolate(self.decoder3(out),scale_factor=(2,2,2),mode ='trilinear'))
        out = torch.add(out,t2)
        output3 = self.map3(out)
        out = F.relu(F.interpolate(self.decoder4(out),scale_factor=(2,2,2),mode ='trilinear'))
        out = torch.add(out,t1) 
        out = F.relu(F.interpolate(self.decoder5(out),scale_factor=(2,2,2),mode ='trilinear'))
        output4 = self.map4(out)
        # print(out.shape)
        # print(output1.shape,output2.shape,output3.shape,output4.shape)
        return output4
        """ if self.training is True:
            return output1, output2, output3, output4
        else:
            return output4"""
        
def debug_Unet():
    batch=1
    channel_coler=3
    pixel_length=128
    x=torch.randn(batch,channel_coler,pixel_length,pixel_length,pixel_length)
    num_class=3
    net=UNet(channel_coler,num_class)
    y=net(x)
    print("Debug_Unet:")
    print(x.shape,y.shape)

######################_______Vnet_______################################
# 输入头
class InputBlock(nn.Module):
    def __init__(self, in_channel,out_channel):
        super(InputBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channel, out_channel, kernel_size=5, padding=2)
        self.activation1 = nn.PReLU()
        self.conv2 = nn.Conv3d(out_channel, out_channel, kernel_size=2, stride=2)
        self.activation2 = nn.PReLU()
    def forward(self, x):
        #x[b,c,128,128]
        c1 = self.conv1(x)
        #c1[b,64,...]
        c2 = self.activation1(c1)
        c3 = c1 + x
        c4 = self.conv2(c3)
        out = self.activation2(c2)
        return out

# 压缩模块
class CompressionBlock(nn.Module):
    def __init__(self, in_channel, out_channel, layer_num):
        super(CompressionBlock, self).__init__()
        self.layer_num = layer_num
        self.conv1 = nn.Conv3d(in_channel, out_channel, kernel_size=5, padding=2)
        self.activation1 = nn.PReLU()
        self.conv2 = nn.Conv3d(out_channel, out_channel, kernel_size=5, padding=2)
        self.activation2 = nn.PReLU()
        if self.layer_num == 3:
            self.conv3 = nn.Conv3d(out_channel, out_channel, kernel_size=5, padding=2)
            self.activation3 = nn.PReLU()
        self.conv4 = nn.Conv3d(out_channel, out_channel, kernel_size=2, stride=2)
        self.activation4 = nn.PReLU()
        
    def forward(self, x):
        c1 = self.conv1(x)
        c1 = self.activation1(c1)
        out = self.conv2(c1)
        out = self.activation2(out)
        if self.layer_num == 3:
            out = self.conv3(out)
            out = self.activation4(out)
        out = out + c1
        out = self.conv4(out)
        out = self.activation4(out)
        return out

# 解压缩模块
class DeCompressionBlock(nn.Module):
    def __init__(self, in_channel, out_channel, com_block_channel, layer_num):
        super(DeCompressionBlock, self).__init__()
        self.layer_num = layer_num
        self.deconv1 = nn.ConvTranspose3d(in_channel, out_channel, kernel_size=2, stride=2)
        self.activation1 = nn.PReLU()
        self.conv2 = nn.Conv3d(out_channel + com_block_channel, out_channel, kernel_size=5, padding=2)
        self.activation2 = nn.PReLU()
        self.conv3 = nn.Conv3d(out_channel, out_channel, kernel_size=5, padding=2)
        self.activation3 = nn.PReLU()
        if self.layer_num == 3:
            self.conv4 = nn.Conv3d(out_channel, out_channel, kernel_size=5, padding=2)
            self.activation4 = nn.PReLU()
            
        
    def forward(self, x1, x2):
        dc1 = self.deconv1(x1)
        a1 = self.activation1(dc1)
        concat = torch.cat((a1, x2), axis=1)
        out = self.conv2(concat)
        out = self.activation2(out)
        out = self.conv3(out)
        out = self.activation3(out)
        if self.layer_num == 3:
            out = self.conv4(out)
            out = self.activation4(out)
        out = out + a1
        return out

# 输出头
class OutputBlock(nn.Module):
    def __init__(self, in_channel, out_channel, com_block_channel, classes):
        super(OutputBlock, self).__init__()
        self.deconv1 = nn.ConvTranspose3d(in_channel, out_channel, kernel_size=2, stride=2)
        self.activation1 = nn.PReLU()
        self.conv2 = nn.Conv3d(out_channel + com_block_channel, out_channel, kernel_size=5, padding=2)
        self.activation2 = nn.PReLU()
        self.conv3 = nn.Conv3d(out_channel, classes, kernel_size=1, padding=0)
        self.activation3 = nn.Softmax(1)
            
        
    def forward(self, x1, x2):
        dc1 = self.deconv1(x1)
        a1 = self.activation1(dc1)
        concat = torch.cat((a1, x2), axis=1)
        out = self.conv2(concat)
        out = self.activation2(out)
        out = out + a1
        out = self.conv3(out)
        out = self.activation3(out)
        return out
#Vnet
class VNet(nn.Module):
    def __init__(self,coler,classes):
        super(VNet, self).__init__()
        self.input_block = InputBlock(coler,out_channel = 16)
        self.cb1 = CompressionBlock(in_channel = 16, out_channel = 32, layer_num = 2)
        self.cb2 = CompressionBlock(in_channel = 32, out_channel = 64, layer_num = 3)
        self.cb3 = CompressionBlock(in_channel = 64, out_channel = 128, layer_num = 3)
        self.cb4 = CompressionBlock(in_channel = 128, out_channel = 256, layer_num = 3)
        self.dcb1 = DeCompressionBlock(in_channel = 256, out_channel = 256, com_block_channel = 128, layer_num = 3)
        self.dcb2 = DeCompressionBlock(in_channel = 256, out_channel = 128, com_block_channel = 64, layer_num = 3)
        self.dcb3 = DeCompressionBlock(in_channel = 128, out_channel = 64, com_block_channel = 32, layer_num = 2)
        self.output_block = OutputBlock(in_channel = 64, out_channel = 32, com_block_channel = 16, classes=classes)
        
    def forward(self, x):
        i = self.input_block(x)
        #print(i.shape)#[1, 16, 128, 128, 128]
        c1 = self.cb1(i)
        #print(c1.shape)#[1, 32, 64, 64, 64]
        c2 = self.cb2(c1)
        #print(c2.shape)#[1, 64, 32, 32, 32]
        c3 = self.cb3(c2)
        #print(c3.shape)#[1, 128, 16, 16, 16]
        c4 = self.cb4(c3)
        #print(c4.shape)#[1, 256, 8, 8, 8]
        dc1 = self.dcb1(c4, c3)
        #print(dc1.shape)#[1, 256, 16, 16, 16]
        dc2 = self.dcb2(dc1, c2)
        #print(dc2.shape)#[1, 128, 32, 32, 32]
        dc3 = self.dcb3(dc2, c1)
        #print(dc3.shape)#[1, 64, 64, 64, 64]
        out = self.output_block(dc3, i)
        #print(out.shape)#[1, 2, 128, 128, 128]
        return out

#dice_loss
def dice_loss(target,predictive,ep=1e-8):
    intersection = 2 * torch.sum(predictive * target) + ep
    union = torch.sum(predictive) + torch.sum(target) + ep
    loss = 1 - intersection / union
    return

def debug_Vnet():
    batch=1
    channel_coler=1 # coler is nessarily to be 1
    pixel_length=128
    num_class=3
    x=torch.randn(batch,channel_coler,pixel_length,pixel_length,pixel_length)
    net=VNet(channel_coler,num_class)
    y=net(x)
    print("Debug_Vnet:")
    print(x.shape,y.shape)

######################_______Main_______################################

if __name__ == '__main__':
    debug_Unet()
    debug_Vnet()