from torchvision import transforms,datasets
from torch.utils.data import DataLoader
import random
import torch

seed=45

imagenet_root='./data/imagenet'
imagenet_train_trans=transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

imagenet_val_trans=transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

#ResNet的原数据增强方法其一，用于替换RandomResizedCrop
#scale augmentation引用自论文[41]    
#resize图像，使短边随机到[256,480]
class ResizeShorter:
    def __init__(self,min_size=256,max_size=480):
        self.min_size=min_size
        self.max_size=max_size

    def __call__(self,img):
        size=random.randint(self.min_size,self.max_size)
        return transforms.Resize(size)(img)

#return train_loader and val_loader
def get_dataloader(data_root:str,train_trans,val_trans,batch_size=256,num_workers=8):
    train_path=data_root+'/train'
    val_path=data_root+'/val'
    
    train_dataset=datasets.ImageFolder(train_path,train_trans)
    val_dataset=datasets.ImageFolder(val_path,val_trans)

    train_loader=DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader=DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader,val_loader

#set seed
def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)

if __name__ == "__main__":
    seed_everything(seed)
    train_loader, val_loader = get_dataloader(
        imagenet_root,
        imagenet_train_trans,
        imagenet_val_trans,
        batch_size=4,
        num_workers=2
    )

    images,labels = next(iter(train_loader))

    print("image shape:", images.shape)
    print("label shape:", labels.shape)

    print("image min:", images.min())
    print("image max:", images.max())

    print("classes:", len(train_loader.dataset.classes))

