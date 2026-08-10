# ResNet Reproduction

PyTorch 实现的 ResNet 网络，基于论文 [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)（He et al., CVPR 2016），在 ImageNet-1K 数据集上训练和评估。

## 已实现的模型

| 模型 | 层数 | 说明 |
|------|------|------|
| PlainNet18 | 18 | 无残差连接的基础网络（对照组） |
| PlainNet34 | 34 | 无残差连接的基础网络（对照组） |
| ResNet18 | 18 | 带残差连接，支持 A/B/C 三种 shortcut |
| ResNet34 | 34 | 带残差连接，支持 A/B/C 三种 shortcut |

## Shortcut 选项

对于维度（通道数/特征图大小）发生变化的层，提供三种 shortcut 策略：

| 选项 | 维度不变时 | 维度变化时 |
|------|-----------|-----------|
| A | Identity | Zero-padding（补零填充通道，间隔采样缩小特征图） |
| B | Identity | 1×1 卷积投影 |
| C | 1×1 卷积投影 | 1×1 卷积投影 |

## 项目结构

```
ResNet-reproduction/
├── model/
│   ├── ImageNetBlock.py       # 基础模块：BasicBlock, 残差Block, PlainNet Block
│   ├── ImageNetPlainNet.py    # PlainNet18 / PlainNet34 网络定义
│   └── ImageNetResNet.py      # ResNet18 / ResNet34 网络定义
├── datasets/
│   ├── base.py                # DALI DataLoader 封装
│   ├── dataloader.py          # PyTorch DataLoader + 数据增强
│   └── imagenet.py            # NVIDIA DALI 数据管线
├── scripts/
│   └── move_valimg.py         # 将 ImageNet 验证集图片按类别整理到子文件夹
├── train.py                   # 训练脚本
├── Inference.py               # 推理/验证脚本（10-crop 评估）
├── requirements.txt           # 依赖
└── README.md
```

## 环境要求

- Python 3.12+
- PyTorch 2.12+（CUDA 13.0）
- NVIDIA DALI（可选，用于加速数据加载）
- TensorBoard（日志记录）

安装依赖：

```bash
pip install -r requirements.txt
```

核心依赖：
- `torch` / `torchvision`：模型训练与数据增强
- `nvidia-dali-cuda130`：GPU 加速数据管线（可选）
- `tensorboard`：训练指标可视化

## 数据准备

### 1. 下载 ImageNet-1K

从 [image-net.org](https://image-net.org/) 下载 ILSVRC2012 数据集，按如下结构放置：

```
data/imagenet/
├── train/
│   ├── n01440764/
│   ├── n01443537/
│   └── ...
├── val/
│   ├── ILSVRC2012_val_00000001.JPEG
│   └── ...
└── ILSVRC2012_devkit_t12/
    └── data/
        ├── meta.mat
        └── ILSVRC2012_validation_ground_truth.txt
```

### 2. 整理验证集

下载 [ILSVRC2012_devkit_t12](https://image-net.org/challenges/LSVRC/2012/index) 并解压后，运行：

```bash
python scripts/move_valimg.py
```

该脚本将 `val/` 中的图片按类别移动到对应的子文件夹，使其格式与训练集一致。

## 训练

### 基本用法

```bash
# 训练 ResNet18（Option A shortcut）
python train.py --model ResNet18 --option A

# 训练 ResNet34（Option B shortcut）
python train.py --model ResNet34 --option B

# 训练 PlainNet18（无残差连接对照）
python train.py --model PlainNet18
```

### 完整参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--model` | str | `PlainNet18` | 模型选择：PlainNet18 / PlainNet34 / ResNet18 / ResNet34 |
| `--option` | str | `A` | Shortcut 类型：A / B / C（仅 ResNet） |
| `--seed` | int | `42` | 随机种子 |
| `--batch_size` | int | `256` | 批次大小 |
| `--num_workers` | int | `16` | DataLoader 工作线程数 |
| `--iters` | int | `600000` | 训练迭代次数 |
| `--lr` | float | `0.1` | 初始学习率 |
| `--momentum` | float | `0.9` | SGD 动量 |
| `--weight_decay` | float | `0.0001` | 权重衰减 |
| `--factor` | float | `0.1` | 学习率衰减因子 |
| `--patience` | int | `3` | ReduceLROnPlateau 耐心值 |
| `--amp` | flag | False | 启用自动混合精度（AMP） |
| `--DALI` | flag | False | 使用 NVIDIA DALI 加速数据加载 |
| `--save_freq` | int | `50000` | 模型保存频率（iteration） |
| `--val_freq` | int | `10000` | 验证频率（iteration） |
| `--print_freq` | int | `1000` | 日志打印频率（iteration） |
| `--continue_train` | flag | False | 从 checkpoint 继续训练 |
| `--which_iters` | int | None | 加载第几个 iteration 的模型 |

### 使用 DALI 加速

```bash
python train.py --model ResNet18 --option A --DALI --amp
```

### 从 checkpoint 恢复训练

```bash
python train.py --model ResNet18 --option A --continue_train --which_iters 100000
```

### 查看训练曲线

```bash
tensorboard --logdir=runs/
```

## 验证

使用 10-crop 评估（标准 ResNet 论文评估方法），计算 Top-1 和 Top-5 错误率：

```bash
# 验证 ResNet18（Option A）
python Inference.py --model ResNet18 --option A

# 验证 ResNet34（Option B）
python Inference.py --model ResNet34 --option B
```

脚本将自动加载 `checkpoints/ImageNet/<model>/option_<option>/best_model.pth`。

## 实现细节

- **权重初始化**：Conv2d 使用 Kaiming 正态初始化，BatchNorm weight 初始化为 1、bias 为 0
- **优化器**：SGD with momentum (0.9)，weight decay (1e-4)
- **学习率策略**：ReduceLROnPlateau，验证集 error 不下降时衰减（factor=0.1）
- **数据增强（训练）**：RandomResizedCrop (224×224, scale∈[0.08,1]) + RandomHorizontalFlip
- **数据预处理（验证）**：Resize(256) + CenterCrop(224)
- **损失函数**：CrossEntropyLoss
- **AMP**：支持 CUDA 自动混合精度加速训练

## 论文引用

```bibtex
@inproceedings{he2016deep,
  title={Deep Residual Learning for Image Recognition},
  author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={770--778},
  year={2016}
}
```
