import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

# 训练集的数据增强
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4,padding_mode='reflect'),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (1, 1, 1))#仅减去均值
])
#验证集与测试集的数据增强
val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (1, 1, 1))
])

#下载训练集，准备做45k/5k分割
full_train_dataset = torchvision.datasets.CIFAR10(
    root='./data', 
    train=True, 
    download=True
)

#划分45,000张训练集和5,000张验证集
train_dataset, val_dataset = random_split(full_train_dataset, [45000, 5000])

# 为划分后的Dataset绑定不同的Transform
class DatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)

train_dataset = DatasetWrapper(train_dataset, transform=train_transform)
val_dataset = DatasetWrapper(val_dataset, transform=val_transform)

# 10,000张测试集
test_dataset = torchvision.datasets.CIFAR10(
    root='./data', 
    train=False, 
    download=True, 
    transform=val_transform
)

def CIFAR10_get_dataloader(batch_size=128,num_workers=16):
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True
        )
    val_loader   = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True
        )
    test_loader  = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True
        )
    return train_loader,val_loader,test_loader

if __name__=='__main__':
    train_loader,val_loader,test_loader=CIFAR10_get_dataloader()
    print(f"训练集数量: {len(train_dataset)}")  # 45000
    print(f"验证集数量: {len(val_dataset)}")    # 5000
    print(f"测试集数量: {len(test_dataset)}")   # 10000