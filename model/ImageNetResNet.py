from model.ImageNetBlock import IdentityResBlock,ZeroPadResBlock,ProjectionResBlock,Conv1
from torch import nn
import torch

#input size [64,112,112]
#output size [64,56,56]
class Conv2(nn.Module):
    def __init__(self,nums_of_blocks,option='A'):
        super().__init__()
        assert option in ['A','B','C']
        self.pool=nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=1
        )
        
        self.conv=nn.Sequential(
            *[
                (
                ProjectionResBlock(64,64)
                if option=='C' 
                else IdentityResBlock(64,64)
                )
                for i in range(nums_of_blocks)
            ]
        )

    def forward(self,x):
        return self.conv(self.pool(x))

#conv3、4、5共用此模板
class ConvN(nn.Module):
    def __init__(self,nums_of_blocks,in_channels:int,out_channels:int,option='A'):
        super().__init__()
        assert option in ['A','B','C']
        self.first_block=ZeroPadResBlock(in_channels,out_channels) if option=='A' else ProjectionResBlock(in_channels,out_channels)

        self.conv=nn.Sequential(
            *[
                (
                ProjectionResBlock(out_channels,out_channels)
                if option=='C' 
                else IdentityResBlock(out_channels,out_channels)
                ) 
                for i in range(nums_of_blocks-1)
            ]
        )
    def forward(self,x):
        return self.conv(self.first_block(x))



class ResNet18(nn.Module):
    def __init__(self,option='A'):
        super().__init__()
        assert option in ['A','B','C']
        self.conv1=Conv1()
        self.conv2=Conv2(nums_of_blocks=2,option=option)
        self.conv3=ConvN(nums_of_blocks=2,in_channels=64,out_channels=128,option=option)
        self.conv4=ConvN(nums_of_blocks=2,in_channels=128,out_channels=256,option=option)  
        self.conv5=ConvN(nums_of_blocks=2,in_channels=256,out_channels=512,option=option)
        self.avgpool=nn.AdaptiveAvgPool2d((1,1))
        self.fc=nn.Linear(512,1000)
    def forward(self,x):
        out=self.conv1(x)
        out=self.conv2(out)
        out=self.conv3(out)
        out=self.conv4(out)
        out=self.conv5(out)
        out=self.avgpool(out)
        out=torch.flatten(out,1)
        out=self.fc(out)
        return out

class ResNet34(nn.Module):
    def __init__(self,option='A'):
        super().__init__()
        assert option in ['A','B','C']
        self.conv1=Conv1()
        self.conv2=Conv2(nums_of_blocks=3,option=option)
        self.conv3=ConvN(nums_of_blocks=4,in_channels=64,out_channels=128,option=option)
        self.conv4=ConvN(nums_of_blocks=6,in_channels=128,out_channels=256,option=option)  
        self.conv5=ConvN(nums_of_blocks=3,in_channels=256,out_channels=512,option=option)
        self.avgpool=nn.AdaptiveAvgPool2d((1,1))
        self.fc=nn.Linear(512,1000)
    def forward(self,x):
        out=self.conv1(x)
        out=self.conv2(out)
        out=self.conv3(out)
        out=self.conv4(out)
        out=self.conv5(out)
        out=self.avgpool(out)
        out=torch.flatten(out,1)
        out=self.fc(out)
        return out

if __name__ == "__main__":
    models = {
        "ResNet18": ResNet18(option='A'),
        "ResNet34A": ResNet34(option='A'),
        "ResNet34B": ResNet34(option='B'),
        "ResNet34C": ResNet34(option='C')
    }

    x = torch.randn(2, 3, 224, 224)

    for name, model in models.items():
        print("=" * 60)
        print(name)
        print("=" * 60)

        model.eval()

        with torch.no_grad():
            out = x
            print("Input:      ", out.shape)

            out = model.conv1(out)
            print("conv1:      ", out.shape)

            out = model.conv2(out)
            print("conv2_x:    ", out.shape)

            out = model.conv3(out)
            print("conv3_x:    ", out.shape)

            out = model.conv4(out)
            print("conv4_x:    ", out.shape)

            out = model.conv5(out)
            print("conv5_x:    ", out.shape)

            out = model.avgpool(out)
            print("avgpool:    ", out.shape)

            out = torch.flatten(out, 1)
            print("flatten:    ", out.shape)

            out = model.fc(out)
            print("fc output:  ", out.shape)

        # 参数量
        params = sum(
            p.numel() for p in model.parameters()
            if p.requires_grad
        )

        print(f"Trainable parameters: {params/1e6:.2f} M")

        # 统计卷积层数量
        conv_layers = sum(
            1 for m in model.modules()
            if isinstance(m, nn.Conv2d)
        )

        print(f"Conv2d layers: {conv_layers}")

        # 统计BN数量
        bn_layers = sum(
            1 for m in model.modules()
            if isinstance(m, nn.BatchNorm2d)
        )

        print(f"BatchNorm layers: {bn_layers}")
        
        print()