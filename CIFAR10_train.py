import argparse
import numpy as np
import time
import os
import random

from model.CIFAR10model import (
    ResNet20, ResNet32, ResNet44, ResNet56, ResNet110, ResNet1202,
<<<<<<< HEAD
    PlainNet20, PlainNet32, PlainNet44, PlainNet56, PlainNet110, PlainNet1202
)
from datasets.CIFAR10dataloader import CIFAR10_get_dataloader



import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast,grad_scaler


=======
    PlainNet20, PlainNet32, PlainNet44, PlainNet56
)
from datasets.CIFAR10dataloader import CIFAR10_get_dataloader

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler

CIFAR10_root = "./data"
>>>>>>> train
MODEL_DICT = {
    # ResNet 系列
    "ResNet20": ResNet20,
    "ResNet32": ResNet32,
    "ResNet44": ResNet44,
    "ResNet56": ResNet56,
    "ResNet110": ResNet110,
    "ResNet1202": ResNet1202,
    
    # PlainNet 系列
    "PlainNet20": PlainNet20,
    "PlainNet32": PlainNet32,
    "PlainNet44": PlainNet44,
<<<<<<< HEAD
    "PlainNet56": PlainNet56,
    "PlainNet110": PlainNet110,
    "PlainNet1202": PlainNet1202,
}
def create_model(args):

=======
    "PlainNet56": PlainNet56
}

def create_model(args):
>>>>>>> train
    if args.model not in MODEL_DICT:
        raise ValueError(f"Unknown model: {args.model}. Available models: {list(MODEL_DICT.keys())}")
    
    # 动态获取对应的模型构建函数并实例化
    model_fn = MODEL_DICT[args.model]
    return model_fn()

<<<<<<< HEAD
def validate(val_loader,criterion,model,device,args):
=======
def get_parameter_groups(model, weight_decay):
    decay = []
    no_decay = []
    
    for param in model.parameters():
        if not param.requires_grad:
            continue
        # 维度 >= 2 的参数是 Conv/FC 的权重，应用 weight decay
        if param.ndim >= 2:
            decay.append(param)
        # 维度 < 2 的参数（1D 的 Bias 和 BN 参数），不应用 weight decay
        else:
            no_decay.append(param)
            
    return [
        {'params': decay, 'weight_decay': weight_decay},
        {'params': no_decay, 'weight_decay': 0.0}
    ]

def get_resnet110_scheduler(optimizer, warmup_iters=400):
    def lr_lambda(current_iter):
        if current_iter < warmup_iters:
            return 0.1
        if current_iter < 32000:
            return 1.0        # 学习率为 0.1 * 1
        elif current_iter < 48000:
            return 0.1        # 学习率为 0.1 * 0.1 = 0.01
        else:
            return 0.01       # 学习率为 0.1 * 0.01 = 0.001

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

def validate(val_loader, criterion, model, device, args):
>>>>>>> train
    model.eval()
    acc=0
    total_loss=0
    dataset_size=len(val_loader.dataset)
<<<<<<< HEAD
    with torch.no_grad():

        for imgs,labels in val_loader:
            imgs=imgs.to(device,non_blocking=True)
            labels=labels.to(device,non_blocking=True)
=======
    
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
>>>>>>> train
            if args.amp:
                with autocast("cuda"):
                    output=model(imgs)
                    loss=criterion(output,labels)
            else:
                output=model(imgs)
                loss=criterion(output,labels)

            total_loss+=loss.item()*imgs.shape[0]

            #统计准确率
            _,pred=torch.max(output,dim=1)
            acc+=(pred==labels).sum().item()

<<<<<<< HEAD
    acc/=dataset_size
    total_loss/=dataset_size

    return acc,total_loss

def save_checkpoint(path,model,optimizer,scheduler,args,iteration):
    checkpoint = {
        "iteration": iteration,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict":scheduler.state_dict(),
        "args": vars(args)
    }

    torch.save(checkpoint,path)

#加载权重时会返回上次训练到的iteration数
def load_checkpoint(path,model,optimizer,scheduler)->int:
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
=======
    acc /= dataset_size
    total_loss /= dataset_size
    return acc, total_loss

def save_checkpoint(path, model, optimizer, scheduler, scaler, args, epoch, iteration):
    checkpoint = {
        "epoch": epoch,
        "iteration": iteration,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "args": vars(args)
    }
    torch.save(checkpoint, path)

def load_checkpoint(path, model, optimizer, scheduler, scaler):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if scaler is not None and checkpoint.get("scaler_state_dict") is not None:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    
    start_epoch = checkpoint.get("epoch", 0)
    iteration = checkpoint.get("iteration", 0)
    return start_epoch, iteration

def init_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
>>>>>>> train

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def train(args):
    #选择模型
    seed_everything(seed=args.seed)
    device='cuda' if torch.cuda.is_available() else "cpu"
    model=create_model(args)
    model.apply(init_weights)
    model.to(device)
    #准备数据
    train_loader,val_loader,_=CIFAR10_get_dataloader(
        batch_size=args.batch_size,
        num_workers=args.num_workers
<<<<<<< HEAD
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
# 动态调整学习率：在 32000 和 48000 次 iteration 时衰减为原来的 0.1
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[32000, 48000],
        gamma=0.1
        )

    if args.amp:
        scaler=grad_scaler.GradScaler("cuda")
    
    #保存路径
    save_dir=os.path.join("checkpoints","CIFAR10",args.model)
    os.makedirs(save_dir,exist_ok=True)
    log_dir=os.path.join('runs',"CIFAR10",args.model)
    os.makedirs(log_dir,exist_ok=True)
    writer=SummaryWriter(log_dir)
    #继续训练情况下，加载权重
    iteration=0
    if args.continue_train:
        checkpoint_path=os.path.join(save_dir,f'iter_{args.which_iters}.pth')
        iteration=load_checkpoint(checkpoint_path,model,optimizer,scheduler)

    best_acc=0
    ###################

    #训练流程

    ###################
    start_time=time.time()
    while iteration<args.iters:
        
        for imgs,labels in train_loader:
            model.train()
            imgs=imgs.to(device,non_blocking=True)
            labels=labels.to(device,non_blocking=True)
=======
    )
    
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(
        get_parameter_groups(model, args.weight_decay),
        lr=args.lr,
        momentum=args.momentum
    )

    if args.deep_Network:
        scheduler = get_resnet110_scheduler(optimizer, warmup_iters=400)
    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[32000, 48000],
            gamma=0.1
        )

    scaler = GradScaler("cuda") if args.amp else None
    
    # 日志与 Checkpoint 存储配置
    save_dir = os.path.join("checkpoints", "CIFAR10", args.model)
    os.makedirs(save_dir, exist_ok=True)
    log_dir = os.path.join('runs', "CIFAR10", args.model)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    start_epoch = 0
    iteration = 0
    best_acc = 0.0

    # 加载断点权重
    if args.continue_train:
        checkpoint_path = os.path.join(save_dir, f'epoch_{args.which_epoch}.pth')
        if os.path.exists(checkpoint_path):
            start_epoch, iteration = load_checkpoint(checkpoint_path, model, optimizer, scheduler, scaler)
            print(f"Resumed training from Epoch {start_epoch} (Iter {iteration})")
        else:
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    start_time = time.time()

    ###########################################################################
    # 主训练循环
    ###########################################################################
    for epoch in range(start_epoch + 1, args.n_epochs + 1):
        model.train()
        
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        
        for imgs, labels in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
>>>>>>> train

            optimizer.zero_grad()

            if args.amp:
                with autocast('cuda'):
                    output=model(imgs)
                    loss=criterion(output,labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                output=model(imgs)
                loss=criterion(output,labels)     #此处loss为一个batch的平均
                loss.backward()
                optimizer.step()
            
<<<<<<< HEAD
            iteration+=1

            #打印信息
            if iteration%args.print_freq==0:
                #统计准确率
                _,pred=torch.max(output,dim=1)
                acc=(pred==labels).sum().item()/labels.shape[0]
                writer.add_scalar("train/loss",loss.item(),iteration)
                writer.add_scalar("train/acc",acc,iteration)

                lr=optimizer.param_groups[0]["lr"]

                print(f'iters:{iteration}   train_loss:{loss.item():.2f}    acc:{acc:.4f}    lr:{lr}')

            #验证
            if iteration%args.val_freq==0:
                print('='*60)
                print('start validating')
                
                val_acc,val_loss=validate(val_loader,criterion,model,device,args)
                writer.add_scalar('val/loss',val_loss,iteration)
                writer.add_scalar('val/acc',val_acc,iteration)

                print(f'iters:{iteration}   validate_loss:{val_loss:.2f}    acc:{val_acc:.4f}')
                print("="*60)
                print()

                if val_acc>best_acc:
                    best_acc=val_acc
                    best_path=os.path.join(save_dir,'best_model.pth')
                    save_checkpoint(best_path,model,optimizer,scheduler,args,iteration)
                    print('best model saved!!!')
                

            #保存模型
            if iteration%args.save_freq==0:
                path=os.path.join(save_dir,f'iter_{iteration}.pth')
                save_checkpoint(path,model,optimizer,scheduler,args,iteration)
                print('='*60)
                print(f'iter_{iteration} saved!!!')
                print('='*60)
                print()
  
            scheduler.step()
            if iteration>=args.iters:
                break

    total_time=time.time()-start_time
    print("="*50)
    print(f"batch_size: {args.batch_size}")
    print(f"num_workers: {args.num_workers}")
    print(f"iters: {args.iters}")
    print(f"time: {total_time:.3f}s")
    print(f"iter/s: {args.iters/total_time:.3f}")
    print(f"img/s: {args.iters*args.batch_size/total_time:.1f}")
=======
            # 学习率调整器按 iter 步进
            scheduler.step()
            iteration += 1

            # 统计整个 Epoch 的指标
            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            _, pred = torch.max(output, dim=1)
            running_corrects += (pred == labels).sum().item()
            total_samples += batch_size

        #整轮Epoch完成，计算本Epoch的平均训练Loss和Acc
        epoch_train_loss = running_loss / total_samples
        epoch_train_acc = running_corrects / total_samples
        current_lr = optimizer.param_groups[0]["lr"]

        # 写入TensorBoard，横轴统一使用累加的iteration
        writer.add_scalar(f"{args.model}/train_loss", epoch_train_loss, iteration)
        writer.add_scalar(f"{args.model}/train_acc", epoch_train_acc, iteration)
        writer.add_scalar(f"{args.model}/train_error", 1-epoch_train_acc, iteration)
        writer.add_scalar(f"{args.model}/lr", current_lr, iteration)

        print(f"Epoch:[{epoch}/{args.n_epochs}]  Iter:{iteration}  Train_Loss:{epoch_train_loss:.4f}  Train_Acc:{epoch_train_acc:.4f}  LR:{current_lr:.6f}")

        #验证
        if epoch % args.val_epochs == 0:
            print('='*60)
            print(f'Start validating at Epoch {epoch} (Iter {iteration})...')
            
            val_acc, val_loss = validate(val_loader, criterion, model, device, args)
            
            # 验证结果写入 TensorBoard
            writer.add_scalar(f'{args.model}/val_loss', val_loss, iteration)
            writer.add_scalar(f'{args.model}/val_acc', val_acc, iteration)
            writer.add_scalar(f'{args.model}/val_error', 1-val_acc, iteration)

            print(f'Epoch:{epoch}  Iter:{iteration}  Val_Loss:{val_loss:.4f}  Val_Acc:{val_acc:.4f}')
            print("="*60)

            # 保存 Best Model
            if val_acc > best_acc:
                best_acc = val_acc
                best_path = os.path.join(save_dir, 'best_model.pth')
                save_checkpoint(best_path, model, optimizer, scheduler, scaler, args, epoch, iteration)
                print(f'New best model saved with Acc: {best_acc:.4f}!')
            print()

        # 按照save_epochs频率保存常规权重
        if epoch % args.save_epochs == 0:
            save_path = os.path.join(save_dir, f'epoch_{epoch}.pth')
            save_checkpoint(save_path, model, optimizer, scheduler, scaler, args, epoch, iteration)
            print(f'Epoch {epoch} checkpoint saved to {save_path}!\n')

    total_time=time.time()-start_time
    print("="*50)
    print(f"Model: {args.model}")
    print(f"Total Epochs: {args.n_epochs}")
    print(f"Total Iters: {iteration}")
    print(f"Time Elapsed: {total_time:.3f}s")
    print(f"Throughput: {iteration * args.batch_size / total_time:.1f} img/s")
>>>>>>> train
    print("="*50)
    writer.close()




if __name__=='__main__':
    parser=argparse.ArgumentParser()
    #模型选择
    parser.add_argument("--model",type=str,
                        choices=list(MODEL_DICT.keys()),
                        default='PlainNet20',
                        help='choose which model to train')
    
<<<<<<< HEAD
    #训练参数
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--amp",action="store_true")

    parser.add_argument("--iters", type=int, default=64000,help='nums of iterations')
    parser.add_argument("--save_freq", type=int, default=2500,help='frequency of saving model')
    parser.add_argument("--val_freq", type=int, default=500,help='frequency of validating model')
    parser.add_argument("--print_freq", type=int, default=100,help='frequency of printing information')
    
    
    
    parser.add_argument("--continue_train",action="store_true",help="continue training from checkpoint")
    parser.add_argument("--which_iters", type=int,default=None,help='which model to load when continuing training')

    #优化器参数
=======
    # 基础训练参数
    parser.add_argument("--seed", type=int, default=127)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--amp", action="store_true",help='use amp in training')
    parser.add_argument("--deep_Network", action="store_true",help='use learning rate warm up for deep Network')

    # Epoch 与频次控制
    parser.add_argument("--n_epochs", type=int, default=182, help='total nums of epochs')
    parser.add_argument("--save_epochs", type=int, default=10, help='frequency of saving model (in epochs)')
    parser.add_argument("--val_epochs", type=int, default=2, help='frequency of validating model (in epochs)')
    
    # 断点恢复参数
    parser.add_argument("--continue_train", action="store_true", help="continue training from checkpoint")
    parser.add_argument("--which_epoch", type=int, default=None, help='which epoch checkpoint to load when continuing training')

    # 优化器超参数
>>>>>>> train
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--momentum", type=float, default=0.9)

    args=parser.parse_args()
    train(args)