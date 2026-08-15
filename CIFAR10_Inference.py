import argparse
import numpy as np
import os
import random

# 导入 CIFAR10 模型与数据加载器
from model.CIFAR10model import (
    ResNet20, ResNet32, ResNet44, ResNet56, ResNet110, ResNet1202,
    PlainNet20, PlainNet32, PlainNet44, PlainNet56
)
from datasets.CIFAR10dataloader import CIFAR10_get_dataloader

import torch

CIFAR10_root = "./data"

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
    "PlainNet56": PlainNet56
}

def create_model(args):
    if args.model not in MODEL_DICT:
        raise ValueError(f"Unknown model: {args.model}")
    return MODEL_DICT[args.model]()

def validate(val_loader, model, device, args):
    model.eval()
    correct_top1 = 0
    total_samples = 0

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            output = model(imgs)

            # 计算 Top-1 匹配数
            _, pred = torch.max(output, dim=1)
            correct_top1 += (pred == labels).sum().item()
            total_samples += labels.size(0)

    acc_top1 = correct_top1 / total_samples
    error_top1 = 1.0 - acc_top1
    return acc_top1, error_top1

def load_checkpoint(path, model, device):
    if not os.path.exists(path):
        raise FileNotFoundError(f"checkpoint not found: {path}")

    # 加载权重
    checkpoint = torch.load(path, map_location=device)
    print(checkpoint['args'])

    model.load_state_dict(checkpoint["model_state_dict"])
    epoch = checkpoint.get("epoch", "未知")
    iteration = checkpoint.get("iteration", "未知")
    print(f"成功加载 Checkpoint, (训练记录于: Epoch {epoch}, Iter {iteration})")


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def main(args):
    seed_everything(seed=args.seed)
    device = 'cuda' if torch.cuda.is_available() else "cpu"

    model = create_model(args)
    model.to(device)

    # 获取测试集数据加载器
    _, _, test_loader = CIFAR10_get_dataloader(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    #根据参数选择加载的模型
    if args.which_epoch is None:
        filename = "best_model.pth"
    else:
        filename = f"epoch_{args.which_epoch}.pth"

    checkpoint_path = os.path.join("checkpoints", "CIFAR10", args.model, filename)

    load_checkpoint(checkpoint_path, model, device)
    print(f"已加载Checkpoint: {checkpoint_path}")

    #开始测试
    print('=' * 60)
    print(f'Start evaluating [{args.model}] on CIFAR-10 Test Set...')
    acc_top1, error_top1 = validate(test_loader, model, device, args)
    
    print('=' * 60)
    print(f'Model          : {args.model}')
    print(f'Test Accuracy (Top-1) : {acc_top1 * 100:.2f}%')
    print(f'Test Error    (Top-1) : {error_top1 * 100:.2f}%')
    print('=' * 60)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CIFAR-10 Model Evaluation")
    
    # 模型设置
    parser.add_argument("--model", type=str, choices=list(MODEL_DICT.keys()), default='PlainNet20')
    
    # 测试参数
    parser.add_argument("--seed", type=int, default=127)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=16)
    
    #指定Epoch测量的参数（默认 None 即加载best_model.pth）
    parser.add_argument("--which_epoch", type=int, default=None, help="指定测试的 epoch 编号，默认测试 best_model.pth")

    args = parser.parse_args()
    main(args)