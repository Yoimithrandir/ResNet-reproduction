import argparse
import numpy as np
import os
import random

from model.ImageNetPlainNet import PlainNet18,PlainNet34
from model.ImageNetResNet import ResNet18,ResNet34
from datasets.dataloader import get_dataloader,imagenet_train_trans,imagenet_root

import torch
from torchvision import transforms


val_trans_10crop = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.TenCrop(224),  # 输出 10 个 224x224 的 PIL Image
        # 对 10 个 crop 统一进行 Tensor 转换与归一化
        transforms.Lambda(
            lambda crops: torch.stack(
                [
                    transforms.Compose(
                        [
                            transforms.ToTensor(),
                            transforms.Normalize(
                                mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225],
                            ),
                        ]
                    )(crop)
                    for crop in crops
                ]
            )
        ),
    ]
)



def create_model(args):
    match args.model:
        case "PlainNet18":
            return PlainNet18()
        case "PlainNet34":
            return PlainNet34()
        case "ResNet18":
            return ResNet18(option=args.option)
        case "ResNet34":
            return ResNet34(option=args.option)
        case _:
            raise ValueError("Unknown model")

def validate(val_loader,model,device,args,dataset_size):
    model.eval()

    with torch.no_grad():
        acc_top1=0
        acc_top5=0
        for imgs,labels in val_loader:
            imgs=imgs.to(device,non_blocking=True)
            labels=labels.to(device,non_blocking=True)
            bs,ncrops,c,h,w=imgs.size()
            #将[batch_size, 10, C, H, W]展平为[batch_size * 10, C, H, W]送入模型
            imgs_flat = imgs.view(-1, c, h, w)
            output = model(imgs_flat)  #预测结果shape:[batch_size * 10, num_classes]

            # 将预测结果重新 reshape 回 [batch_size, 10, num_classes]
            output = output.view(bs, ncrops, -1)

            # 对 10 个 crop 的预测结果取平均,output shape: [batch_size, num_classes]
            output_avg = output.mean(dim=1)  

            # 计算分类概率
            _,pred_top5 = torch.topk(output_avg,k=5,dim=-1)
            pred_top1=pred_top5[:,0]

            acc_top1+=(pred_top1==labels).sum().item()
            acc_top5+=(pred_top5==labels.view(-1,1)).sum().item()

        acc_top1/=dataset_size
        acc_top5/=dataset_size
    return acc_top1,acc_top5


#加载模型权重
def load_checkpoint(path,model):
    checkpoint=torch.load(path)
    model.load_state_dict(checkpoint["model_state_dict"])

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def main(args):
    #选择模型
    seed_everything(seed=args.seed)
    device='cuda' if torch.cuda.is_available() else "cpu"
    model=create_model(args)
    model.to(device)
    #准备数据
    _,val_loader=get_dataloader(
        data_root=imagenet_root,
        train_trans=imagenet_train_trans,
        val_trans=val_trans_10crop,
        batch_size=args.batch_size,
        num_workers=args.num_workers      
        )
   
    #保存路径
    save_dir=os.path.join("checkpoints","ImageNet",args.model,f"option_{args.option}")
    checkpoint_path=os.path.join(save_dir,"best_model.pth")
    load_checkpoint(checkpoint_path,model)

    ###################

    #验证流程

    ###################

    print('='*60)
    print('start validating')        
    acc_top1,acc_top5=validate(val_loader,model,device,args,50000)
    print(f'top1_error:{1-acc_top1:.4f} top5_error:{1-acc_top5:.4f}')
    print("="*60)
    print()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    #模型选择
    parser.add_argument("--model",type=str,
                        choices=['PlainNet18','PlainNet34','ResNet18','ResNet34'],
                        default='PlainNet18',
                        help='choose which model to train')
    parser.add_argument('--option',type=str,
                        choices=['A','B','C'],
                        default='A',
                        help='decide which shortcut to use')
    
    #训练参数
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=16)


    args=parser.parse_args()
    main(args)