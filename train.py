import argparse
from model.ImageNetPlainNet import PlainNet18,PlainNet34
from model.ImageNetResNet import ResNet18,ResNet34
from datasets.dataloader import *
import torch
from torch import nn
import os
import random
from torch.utils.tensorboard import SummaryWriter



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

def validate(val_loader,criterion,model,device):
    model.eval()
    acc=0
    total_loss=0
    with torch.no_grad():

        for imgs,labels in val_loader:
            imgs=imgs.to(device)
            labels=labels.to(device)
            output=model(imgs)
            loss=criterion(output,labels)     #此处loss为一个batch的平均

            total_loss+=loss.item()*imgs.size[0]

            #统计准确率
            _,pred=torch.max(output,dim=1)
            acc+=(pred==labels).sum().item()

    acc/=len(val_loader.dataset)
    total_loss/=len(val_loader.dataset)

    return acc,total_loss

def save_checkpoint(model,optimizer,scheduler,args,iteration,path):
    checkpoint = {
        "iteration": iteration,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict":scheduler.state_dict(),
        "args": vars(args)
    }

    torch.save(checkpoint,path)

def load_checkpoint(path,model,optimizer,scheduler):
    checkpoint=torch.load(path)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    iteration=checkpoint["iteration"]

    return iteration

def init_weights(model):

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')

        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight,1)

            nn.init.constant_(m.bias,0)

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)

def train(args):
    #选择模型
    seed_everything(seed=args.seed)
    device='cuda' if torch.cuda.is_available() else "cpu"
    model=create_model(args)
    model.apply(init_weights)
    model.to(device)
    #准备数据
    train_loader,val_loader=get_dataloader(
        data_root=imagenet_root,
        train_trans=imagenet_train_trans,
        val_trans=imagenet_val_trans,
        batch_size=args.batch_size,
        num_workers=args.num_workers      
        )
    #准备优化器
    criterion=nn.CrossEntropyLoss().to(device)
    optimizer=torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
    #动态调整学习率
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.factor,
        patience=5
        )
    
    #保存路径
    save_dir=os.path.join("checkpoints","ImageNet",args.model)
    os.makedirs(save_dir,exist_ok=True)
    log_dir=os.path.join('runs',"ImageNet",args.model,f"option_{args.option}")
    os.makedirs(log_dir,exist_ok=True)
    writer=SummaryWriter(log_dir)
    #继续训练情况下，加载权重
    iteration=0
    if args.continue_train:
        checkpoint_path=os.path.join(save_dir,f'iter_{args.which_iters}.pth')
        iteration=load_checkpoint(checkpoint_path,model,optimizer,scheduler)

    ###################

    #训练流程

    ###################
    while iteration<args.iters:
        
        for imgs,labels in train_loader:
            model.train()
            imgs=imgs.to(device)
            labels=labels.to(device)
            output=model(imgs)
            loss=criterion(output,labels)     #此处loss为一个batch的平均
        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            
            iteration+=1

            #打印信息
            if iteration%args.print_freq==0:
                #统计准确率
                _,pred=torch.max(output,dim=1)
                acc=(pred==labels).sum().item()/labels.size[0]
                writer.add_scalar("train/loss",loss.item(),iteration)
                writer.add_scalar("train/acc",acc,iteration)

                lr=optimizer.param_groups[0]["lr"]

                print(f'iters:{iteration}   train_loss:{loss.item()}    acc:{acc}    lr:{lr}')

            #验证
            if iteration%args.val_freq==0:
                print('*'*60)
                print('start validating')
                print('*'*60)
                val_acc,val_loss=validate(val_loader,criterion,model,device)
                writer.add_scalar('val/loss',val_loss,iteration)
                writer.add_scalar('val/acc',val_acc,iteration)

                print(f'iters:{iteration}   validate_loss:{val_loss}    acc:{val_acc}')
                scheduler.step(1-val_acc)   #论文写当error不下降时改学习率

            #保存模型
            if iteration%args.save_freq==0:
                path=os.path.join(save_dir,f'iter_{iteration}.pth')
                save_checkpoint(model,optimizer,scheduler,args,iteration,path)
                print('*'*60)
                print(f'iter_{iteration} saved!!!')
                print('*'*60)

            if iteration>=args.iters:
                break

    writer.close()



        
          



    



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
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--iters", type=int, default=600000,help='nums of iterations')
    parser.add_argument("--save_freq", type=int, default=50000,help='frequency of saving model')
    parser.add_argument("--val_freq", type=int, default=10000,help='frequency of validating model')
    parser.add_argument("--print_freq", type=int, default=1000,help='frequency of printing information')
    parser.add_argument("--continue_train",action="store_true",help="continue training from checkpoint")
    parser.add_argument("--which_iters", type=int,default=None,help='which model to load when continuing training')

    #优化器参数
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--factor", type=float, default=0.1)


    args=parser.parse_args()
    train(args)
