from torch import nn


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
    def __init__(self,n_channels):
        super().__init__()
        self.conv1=nn.Conv2d(
            in_channels=n_channels,
            out_channels=n_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
            )
        self.bn1=nn.BatchNorm2d(n_channels)
        self.relu=nn.ReLU(inplace=True)
        self.conv2=nn.Conv2d(
            in_channels=n_channels,
            out_channels=n_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.bn2=nn.BatchNorm2d(n_channels)
        

    def forward(self,x):
        out=self.relu(self.bn1(self.conv1(x)))
        out=self.bn2(self.conv2(x))
        return out

#用于convn_x的第一个block,用stride=2进行下采样,out_channels=2*in_channels=2*n_channels
class BasicDownsampleBlock(nn.Module):
    def __init__(self,n_channels):
        super().__init__()
        self.conv1=nn.Conv2d(
            in_channels=n_channels,
            out_channels=2*n_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False
            )
        self.bn1=nn.BatchNorm2d(2*n_channels)
        self.relu=nn.ReLU(inplace=True)
        self.conv2=nn.Conv2d(
            in_channels=2*n_channels,
            out_channels=2*n_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.bn2=nn.BatchNorm2d(2*n_channels)

    def forward(self,x):
        out=self.relu(self.bn1(self.conv1(x)))
        out=self.bn2(self.conv2(x))
        return out

#############################

#以下Block供PlainNet使用

#############################

class PlainNetBlock(nn.Module):
    def __init__(self,n_channels):
        super().__init__()
        self.block=BasicBlock(n_channels)
        self.relu=nn.ReLU(inplace=True)
    def forward(self,x):
        return self.relu(self.block(x))

class PlainNetDownsampleBlock(nn.Module):
    def __init__(self,n_channels):
        super().__init__()
        self.block=BasicDownsampleBlock(n_channels)
        self.relu=nn.ReLU(inplace=True)
    def forward(self,x):
        return self.relu(self.block(x))
