# ResNet Reproduction

PyTorch 实现的 ResNet 网络，基于论文 [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)（He et al., CVPR 2016）。本项目在 **ImageNet-1K** 和 **CIFAR-10** 两个数据集上分别复现了论文的两种网络结构，用于验证残差连接对深层网络训练的改善。

## 已实现的模型

### ImageNet-1K（论文 4.1 节结构）

| 模型 | 层数 | 说明 |
|------|------|------|
| PlainNet18 | 18 | 无残差连接的基础网络（对照组） |
| PlainNet34 | 34 | 无残差连接的基础网络（对照组） |
| ResNet18 | 18 | 带残差连接，支持 A/B/C 三种 shortcut |
| ResNet34 | 34 | 带残差连接，支持 A/B/C 三种 shortcut |

### CIFAR-10（论文 4.2 节结构，深度 = 6n+2）

| 模型 | n | 层数 | 说明 |
|------|---|------|------|
| PlainNet20 / 32 / 44 / 56 | 3/5/7/9 | 20/32/44/56 | 无残差连接（对照组） |
| ResNet20 / 32 / 44 / 56 | 3/5/7/9 | 20/32/44/56 | 带残差连接 |
| ResNet110 | 18 | 110 | 深层残差网络（需 warmup） |
| ResNet1202 | 200 | 1202 | 极深残差网络 |

CIFAR-10 版网络结构与 ImageNet 不同：首层为 3×3 conv（16 通道，stride 1，无池化），三个 stage（通道 16→32→64），`AvgPool(8)` + `FC(64, 10)`。

## Shortcut 选项

对于维度（通道数/特征图大小）发生变化的层，提供三种 shortcut 策略：

| 选项 | 维度不变时 | 维度变化时 |
|------|-----------|-----------|
| A | Identity | Zero-padding（补零填充通道，间隔采样缩小特征图） |
| B | Identity | 1×1 卷积投影 |
| C | 1×1 卷积投影 | 1×1 卷积投影 |

> ImageNet 版（`train.py`）支持 A/B/C 三种；CIFAR-10 版（`CIFAR10_train.py`）仅实现 Option A。

## 项目结构

```
ResNet-reproduction/
├── model/
│   ├── ImageNetBlock.py       # ImageNet 基础模块：BasicBlock / 残差Block / PlainNet Block
│   ├── ImageNetPlainNet.py    # PlainNet18 / PlainNet34
│   ├── ImageNetResNet.py      # ResNet18 / ResNet34（A/B/C shortcut）
│   └── CIFAR10model.py        # CIFAR-10 的 PlainNet / ResNet（20/32/44/56/110/1202）
├── datasets/
│   ├── base.py                # DALI DataLoader 封装
│   ├── dataloader.py          # ImageNet PyTorch DataLoader + 数据增强
│   ├── imagenet.py            # ImageNet NVIDIA DALI 数据管线
│   └── CIFAR10dataloader.py   # CIFAR-10 数据加载（自动下载、45k/5k/10k 划分）
├── scripts/
│   ├── move_valimg.py         # 整理 ImageNet 验证集
│   ├── train.sh / inference.sh            # ImageNet 训练/推理脚本（Linux）
│   └── CIFAR10_train.sh / CIFAR10_inference.sh / CIFAR10_layer_responses.sh
├── train.py                   # ImageNet 训练脚本
├── Inference.py               # ImageNet 推理/验证（10-crop 评估）
├── CIFAR10_train.py           # CIFAR-10 训练脚本（epoch 训练）
├── CIFAR10_Inference.py       # CIFAR-10 测试集评估（Top-1）
├── draw.py                    # ImageNet 训练曲线绘图
├── CIFAR10_draw_curve.py      # CIFAR-10 训练曲线绘图
├── CIFAR10_layer_responses.py # CIFAR-10 层响应分析（论文 Figure 7）
├── requirements.txt           # 依赖
└── README.md
```

## 环境要求

- Python 3.12+
- PyTorch 2.12+（CUDA 13.0）
- NVIDIA DALI（可选，仅 ImageNet 加速数据加载用）
- TensorBoard（日志记录）
- Matplotlib（绘图）

安装依赖：

```bash
pip install -r requirements.txt
```

核心依赖：
- `torch` / `torchvision`：模型训练与数据增强
- `nvidia-dali-cuda130`：GPU 加速数据管线（可选，仅 ImageNet）
- `tensorboard`：训练指标可视化

---

## ImageNet-1K

### 数据准备

1. 从 [image-net.org](https://image-net.org/) 下载 ILSVRC2012 数据集，按如下结构放置：

```
data/imagenet/
├── train/
│   ├── n01440764/
│   └── ...
├── val/
│   ├── ILSVRC2012_val_00000001.JPEG
│   └── ...
└── ILSVRC2012_devkit_t12/
    └── data/
        ├── meta.mat
        └── ILSVRC2012_validation_ground_truth.txt
```

2. 下载 [ILSVRC2012_devkit_t12](https://image-net.org/challenges/LSVRC/2012/index) 并解压后，整理验证集：

```bash
python scripts/move_valimg.py
```

### 训练

```bash
# 训练 ResNet18（Option A）
python train.py --model ResNet18 --option A

# 训练 ResNet34（Option B）
python train.py --model ResNet34 --option B

# 训练 PlainNet18（无残差连接对照）
python train.py --model PlainNet18

# 使用 DALI + AMP 加速
python train.py --model ResNet18 --option A --DALI --amp
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--model` | str | `PlainNet18` | PlainNet18/34、ResNet18/34 |
| `--option` | str | `A` | Shortcut 类型 A/B/C（仅 ResNet） |
| `--batch_size` | int | `256` | 批次大小 |
| `--iters` | int | `600000` | 总迭代数 |
| `--lr` | float | `0.1` | 初始学习率 |
| `--momentum` | float | `0.9` | SGD 动量 |
| `--weight_decay` | float | `0.0001` | 权重衰减 |
| `--amp` | flag | False | 混合精度训练 |
| `--DALI` | flag | False | DALI 加速数据加载 |
| `--val_freq` | int | `10000` | 验证频率 |
| `--save_freq` | int | `50000` | 保存频率 |
| `--continue_train` / `--which_iters` | — | — | 断点续训 |

### 验证（10-crop 评估）

```bash
python Inference.py --model ResNet18 --option A
```

输出 Top-1 和 Top-5 错误率，自动加载 `checkpoints/ImageNet/<model>/option_<option>/best_model.pth`。

---

## CIFAR-10

### 数据准备

无需手动下载，运行训练脚本时 `torchvision` 会自动下载 CIFAR-10 到 `./data`。数据划分为：

- 训练集 45,000 张
- 验证集 5,000 张（从原始 50k 训练集中随机划分）
- 测试集 10,000 张

数据增强（训练）：`RandomCrop(32, padding=4, padding_mode='reflect')` + `RandomHorizontalFlip`；预处理只**减去均值**（`Normalize(mean, std=(1,1,1))`，不做标准差归一化），与论文一致。

### 训练

训练采用 **epoch 训练**（默认 182 epochs ≈ 64k 次迭代），学习率在 32k / 48k 迭代处各衰减 0.1：

```bash
# 训练 ResNet20
python CIFAR10_train.py --model ResNet20 --amp

# 训练 ResNet56（深层网络加 warmup）
python CIFAR10_train.py --model ResNet56 --deep_Network --weight_decay 0.0003 --amp

# 训练 ResNet110（warmup + 更大 weight decay）
python CIFAR10_train.py --model ResNet110 --deep_Network --weight_decay 0.0005 --amp

# 训练 PlainNet 对照组
python CIFAR10_train.py --model PlainNet20 --amp
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--model` | str | `PlainNet20` | ResNet20/32/44/56/110/1202、PlainNet20/32/44/56 |
| `--n_epochs` | int | `182` | 总 epoch 数（≈64k 迭代） |
| `--batch_size` | int | `128` | 批次大小 |
| `--lr` | float | `0.1` | 初始学习率 |
| `--momentum` | float | `0.9` | SGD 动量 |
| `--weight_decay` | float | `0.0001` | 权重衰减（仅作用于 conv/fc 权重） |
| `--deep_Network` | flag | False | 深层网络 warmup（前 400 iter lr=0.01） |
| `--amp` | flag | False | 混合精度训练 |
| `--val_epochs` | int | `2` | 验证频率（epoch） |
| `--save_epochs` | int | `10` | 保存频率（epoch） |
| `--continue_train` / `--which_epoch` | — | — | 断点续训 |

> 学习率调度：普通网络用 `MultiStepLR`（32k/48k 衰减）；深层网络（`--deep_Network`）用 warmup + 阶梯衰减（前 400 iter lr=0.01，之后 0.1，再在 32k/48k 衰减）。

### 测试（Inference）

```bash
# 加载 best_model.pth
python CIFAR10_Inference.py --model ResNet20

# 加载指定 epoch 的 checkpoint
python CIFAR10_Inference.py --model ResNet56 --which_epoch 100
```

在 10,000 张测试集上输出 Top-1 Accuracy 与 Error。

### 绘制训练曲线

```bash
python CIFAR10_draw_curve.py
```

在文件顶部配置区选择要画的模型（`MODELS`）、画 loss 或 error（`PLOT_TYPE`）、纵轴上限（`Y_MAX_ERROR`，默认 30%）。图片默认保存为 `curve_cifar10.png`。

### 层响应分析（论文 Figure 7）

```bash
python CIFAR10_layer_responses.py --models PlainNet20 PlainNet56 ResNet20 ResNet56 ResNet110
```

通过 forward hook 捕捉各层 BatchNorm 输出特征图的标准差，绘制「原始顺序」与「按幅值降序」两个子图，用于观察深层 PlainNet 的响应退化现象。

---

## 实现细节

### ImageNet 与 CIFAR-10 共有

- **权重初始化**：Conv2d 使用 Kaiming 正态初始化，BatchNorm weight 初始化为 1、bias 为 0
- **优化器**：SGD with momentum (0.9)
- **损失函数**：CrossEntropyLoss
- **AMP**：支持 CUDA 混合精度训练

### 差异对比

| 维度 | ImageNet | CIFAR-10 |
|------|----------|----------|
| 学习率调度 | ReduceLROnPlateau（自适应） | MultiStepLR（32k/48k 固定衰减）+ 深层 warmup |
| 权重衰减 | 全部参数 | 仅 conv/fc 权重（BN/bias 不加） |
| 归一化 | 减均值并除以 std | 仅减均值（std=1） |
| shortcut | A/B/C | 仅 A |
| 评估 | 10-crop，Top-1/Top-5 | 单 crop，Top-1 |
| 数据加速 | 支持 DALI | 不需要 |

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
