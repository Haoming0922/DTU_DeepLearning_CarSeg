import torch
from torch import nn
from torch.nn import functional as F

class Conv_Block(nn.Module):    # 仅仅增加了通道数，也就是输入的x多大的尺寸，输出就是多大的尺寸
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
            nn.Conv2d(channel,channel,3,2,1,padding_mode='reflect',bias=False), # kernal size = 3, stride = 2, padding = 1, 
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
        # print("interpolate before:", x.shape)  # torch.Size([2, 256, 64, 64])
        up=F.interpolate(x,scale_factor=2,mode='nearest')  # 可以理解为将x的尺寸放大两倍，以便后面和feature_map相加
        # print("interpolate after:", up.shape)   # torch.Size([2, 256, 128, 128])
        out=self.layer(up)                                 # 对齐通道数
        # print("adjust channels after:", out.shape)    # torch.Size([2, 128, 128, 128])
        # print('----------------------')
        return torch.cat((out,feature_map),dim=1)


class UNet(nn.Module):
    def __init__(self, inch, outch):
        super(UNet, self).__init__()
        self.inch = inch
        self.outch = outch
        self.c1=Conv_Block(3,64)  # 卷积Block
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
        self.out = nn.Conv2d(64, self.outch, 3, 1, 1)  # inc=64, outc=3 , kernal_size=3, stride=1, padding=1
        self.Th = nn.Sigmoid()

    def forward(self,x):
        R1=self.c1(x)
        # print('R1.shape:', R1.shape)  # 2*64*256*256
        R2=self.c2(self.d1(R1))
        # print('R2.shape:', R2.shape) # 2*128*128*128
        R3 = self.c3(self.d2(R2))
        # print('R3.shape:', R3.shape)  # 2*256*64*64
        R4 = self.c4(self.d3(R3))     
        # print('R4.shape:', R4.shape)  # 2*512*32*32
        R5 = self.c5(self.d4(R4))
        # print('R5.shape:', R5.shape)  # 2*1024*16*16
        O1 = self.c6(self.u1(R5,R4))  # 2*1024*16*16  （变化） cat 2*512*32*32   -> 2*512*32*32
        O2 = self.c7(self.u2(O1, R3)) # 2*512*32*32   （变化） cat 2*256*64*64   -> 2*256*64*64
        O3 = self.c8(self.u3(O2, R2)) # 2*256*64*64   （变化） cat 2*128*128*128 -> 2*128*128*128
        O4 = self.c9(self.u4(O3, R1)) # 2*128*128*128 （变化） cat 2*64*256*256  -> 2*64*256*256

        # return self.Th(self.out(O4))  # 2*64*256*256 -> 2*3*256*256  -> sigmoid() 求了一个概率值
        return (self.out(O4),)

if __name__ == '__main__':
    x=torch.randn(2,3,512,512)
    net=UNet()
    net(x)