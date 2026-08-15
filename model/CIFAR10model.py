from torch import nn
import torch

class Block(nn.Module):
    def __init__(self,in_channels,out_channels,downsample=False,Residual_learning=True):
        super().__init__()

        self.Residual_learning=Residual_learning
        self.downsample=downsample

        self.conv1=nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1,
                             stride=2 if downsample else 1,
                             bias=False)
        self.bn1=nn.BatchNorm2d(out_channels)
        self.relu=nn.ReLU(inplace=True)
        self.conv2=nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1,stride=1,bias=False)
        self.bn2=nn.BatchNorm2d(out_channels)
    def forward(self,x):
        out=  self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x)))))
        #PlainNet通过relu后直接输出
        if not self.Residual_learning:
            return self.relu(out)
        
        #ResNet在relu前加上shortcut,一律采用option A:
        if not self.downsample:
            return self.relu(out+x) #identity shortcut
        else:
            residual=x[:,:,::2,::2] #zero-padding shortcut
            zero_pad = torch.zeros(
                residual.size(0),
                out.size(1) - residual.size(1),
                residual.size(2),
                residual.size(3),
                device=x.device,
                dtype=x.dtype
            )
            residual=torch.cat((residual,zero_pad),dim=1)
            return self.relu(residual+out)

def make_layer(nums_blocks,layer_index=1,Residual_learning=True):
    assert layer_index in [1,2,3]

    in_channels=[16,16,32]
    out_channels=[16,32,64]

    blocks=[]
    first_block=Block(in_channels[layer_index-1],out_channels[layer_index-1],
                      downsample=False if layer_index==1 else True,
                      Residual_learning=Residual_learning)
    blocks.append(first_block)
    for _ in range(nums_blocks-1):
        blocks.append(Block(out_channels[layer_index-1],
                            out_channels[layer_index-1],
                            downsample=False,
                            Residual_learning=Residual_learning))
    return nn.Sequential(*blocks)

class PlainNet(nn.Module):
    def __init__(self,nums_blocks):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3,stride=1, padding=1, bias=False)
        self.bn=nn.BatchNorm2d(16)
        self.relu=nn.ReLU(inplace=True)
        self.layer1=make_layer(nums_blocks,layer_index=1,Residual_learning=False)
        self.layer2=make_layer(nums_blocks,layer_index=2,Residual_learning=False)
        self.layer3=make_layer(nums_blocks,layer_index=3,Residual_learning=False)
        self.avgpool = nn.AvgPool2d(kernel_size=8)
        self.fc=nn.Linear(64,10)

    def forward(self,x):
        out=self.relu(self.bn(self.conv1(x)))
        out=self.layer1(out)
        out=self.layer2(out)
        out=self.layer3(out)
        out=self.avgpool(out)
        out=torch.flatten(out,1)
        out=self.fc(out)
        return out

class ResNet(nn.Module):
    def __init__(self,nums_blocks):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3,stride=1, padding=1, bias=False)
        self.bn=nn.BatchNorm2d(16)
        self.relu=nn.ReLU(inplace=True)
        self.layer1=make_layer(nums_blocks,layer_index=1,Residual_learning=True)
        self.layer2=make_layer(nums_blocks,layer_index=2,Residual_learning=True)
        self.layer3=make_layer(nums_blocks,layer_index=3,Residual_learning=True)
        self.avgpool = nn.AvgPool2d(kernel_size=8)
        self.fc=nn.Linear(64,10)

    def forward(self,x):
        out=self.relu(self.bn(self.conv1(x)))
        out=self.layer1(out)
        out=self.layer2(out)
        out=self.layer3(out)
        out=self.avgpool(out)
        out=torch.flatten(out,1)
        out=self.fc(out)
        return out
    

def ResNet20():
    return ResNet(nums_blocks=3)
def ResNet32():
    return ResNet(nums_blocks=5)
def ResNet44():
    return ResNet(nums_blocks=7)
def ResNet56():
    return ResNet(nums_blocks=9)
def ResNet110():
    return ResNet(nums_blocks=18)
def ResNet1202():
    return ResNet(nums_blocks=200)

def PlainNet20():
    return PlainNet(nums_blocks=3)
def PlainNet32():
    return PlainNet(nums_blocks=5)
def PlainNet44():
    return PlainNet(nums_blocks=7)
def PlainNet56():
    return PlainNet(nums_blocks=9)
def PlainNet110():
    return PlainNet(nums_blocks=18)
def PlainNet1202():
    return PlainNet(nums_blocks=200)


if __name__=="__main__":


    models = {
        "ResNet20": ResNet20(),
        "ResNet32": ResNet32(),
        "ResNet44": ResNet44(),
        "ResNet56": ResNet56(),
        "ResNet110": ResNet110(),
        "ResNet1202": ResNet1202(),
        "PlainNet20": PlainNet20(),
        "PlainNet56": PlainNet56(),
    }

    # CIFAR-10 图像大小为 32x32，Batch Size 设为 2
    x = torch.randn(2, 3, 32, 32)

    for name, model in models.items():
        print("=" * 60)
        print(f"Testing Model: {name}")
        print("=" * 60)

        model.eval()

        with torch.no_grad():
            out = x
            print("Input:        ", out.shape)

            # stem 阶段：conv1 + bn + relu
            out = model.relu(model.bn(model.conv1(out)))
            print("conv1:        ", out.shape)

            # 三个主要的 layer 阶段 (CIFAR-10 架构)
            out = model.layer1(out)
            print("layer1:       ", out.shape)

            out = model.layer2(out)
            print("layer2:       ", out.shape)

            out = model.layer3(out)
            print("layer3:       ", out.shape)

            # 全局池化与分类器
            out = model.avgpool(out)
            print("avgpool:      ", out.shape)

            out = torch.flatten(out, 1)
            print("flatten:      ", out.shape)

            out = model.fc(out)
            print("fc output:    ", out.shape)

        # 1. 可训练参数量统计
        params = sum(
            p.numel() for p in model.parameters()
            if p.requires_grad
        )
        print("-" * 60)
        print(f"Trainable parameters: {params/1e6:.4f} M ({params} params)")

        # 2. 统计 Conv2d 层数
        conv_layers = sum(
            1 for m in model.modules()
            if isinstance(m, nn.Conv2d)
        )
        print(f"Conv2d layers:        {conv_layers}")

        # 3. 统计 BatchNorm2d 层数
        bn_layers = sum(
            1 for m in model.modules()
            if isinstance(m, nn.BatchNorm2d)
        )
        print(f"BatchNorm layers:     {bn_layers}")
        
        print("\n")


