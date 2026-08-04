from torch import nn
import torch

#ImageNet所有网络结构的第一部分
class Conv1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv=nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        self.bn=nn.BatchNorm2d(64)
        self.relu=nn.ReLU(inplace=True)
    def forward(self,x):
        return self.relu(self.bn(self.conv(x)))


#用于convn_x,in_channels=out_channels
class BasicBlock(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        assert in_channels==out_channels
        self.conv1=nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
            )
        self.bn1=nn.BatchNorm2d(in_channels)
        self.relu=nn.ReLU(inplace=True)
        self.conv2=nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.bn2=nn.BatchNorm2d(in_channels)
        

    def forward(self,x):
        out=self.relu(self.bn1(self.conv1(x)))
        out=self.bn2(self.conv2(x))
        return out

#用于convn_x的第一个block,用stride=2进行下采样,out_channels=2*in_channels=2*n_channels
class BasicDownsampleBlock(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        assert out_channels==2*in_channels
        self.conv1=nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False
            )
        self.bn1=nn.BatchNorm2d(out_channels)
        self.relu=nn.ReLU(inplace=True)
        self.conv2=nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.bn2=nn.BatchNorm2d(out_channels)

    def forward(self,x):
        out=self.relu(self.bn1(self.conv1(x)))
        out=self.bn2(self.conv2(out))
        return out

#############################

#以下Block供PlainNet使用

#############################

class PlainNetBlock(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.block=BasicBlock(in_channels,out_channels)
        self.relu=nn.ReLU(inplace=True)
    def forward(self,x):
        return self.relu(self.block(x))

class PlainNetDownsampleBlock(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.block=BasicDownsampleBlock(in_channels,out_channels)
        self.relu=nn.ReLU(inplace=True)
    def forward(self,x):
        return self.relu(self.block(x))

#############################

#以下Block供ResNet使用

#############################

#Identity shortcuts
class IdentityResBlock(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.block=BasicBlock(in_channels,out_channels)
        self.relu=nn.ReLU(inplace=True)
    def forward(self,x):
        residual=x
        out=self.block(x)
        out+=residual
        out=self.relu(out)
        return out

#zero-padding shortcuts
class ZeroPadResBlock(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        assert out_channels == 2*in_channels
        self.block=BasicDownsampleBlock(in_channels,out_channels)
        self.relu=nn.ReLU(inplace=True)
    def forward(self,x):
        residual=x[:,:,::2,::2]                     #经过DownsampleBlock,feature map减小
        zero=torch.zeros_like(residual)
        residual=torch.cat([residual,zero],dim=1)   #沿channel叠加
        out=self.block(x)
        out+=residual
        out=self.relu(out)
        return out

#projection shortcuts
class ProjectionResBlock(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        assert (out_channels==2*in_channels) or (out_channels==in_channels)
        self.downsample=True if out_channels==2*in_channels else False
        self.block=BasicDownsampleBlock(in_channels,out_channels) if self.downsample else BasicBlock(in_channels,out_channels)
        self.stride=2 if self.downsample else 1
        self.conv1x1=nn.Sequential(
            nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=self.stride,
            bias=False
            ),
            nn.BatchNorm2d(out_channels)
        )
        self.relu=nn.ReLU(inplace=True)
    def forward(self,x):
        residual=self.conv1x1(x)
        out=self.block(x)
        out+=residual
        out=self.relu(out)
        return out