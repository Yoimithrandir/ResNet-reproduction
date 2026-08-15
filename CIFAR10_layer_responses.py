import argparse
import os
import glob
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


from model.CIFAR10model import (
    ResNet20, ResNet32, ResNet44, ResNet56, ResNet110, ResNet1202,
    PlainNet20, PlainNet32, PlainNet44, PlainNet56
)
from datasets.CIFAR10dataloader import CIFAR10_get_dataloader

MODEL_DICT = {
    "ResNet20": ResNet20, "ResNet32": ResNet32, "ResNet44": ResNet44,
    "ResNet56": ResNet56, "ResNet110": ResNet110, "ResNet1202": ResNet1202,
    "PlainNet20": PlainNet20, "PlainNet32": PlainNet32, 
    "PlainNet44": PlainNet44, "PlainNet56": PlainNet56
}

COLORS = {
    "ResNet20": "#1f77b4",   # 蓝色
    "ResNet32": "#ff7f0e",   # 橙色
    "ResNet44": "#2ca02c",   # 绿色
    "ResNet56": "#d62728",   # 红色
    "ResNet110": "#9467bd",  # 紫色
    "PlainNet20": "#7f7f7f", # 灰色
    "PlainNet32": "#bcbd22", # 黄绿色
    "PlainNet56": "#17becf"  # 青色
}

class LayerResponseTracker:
    #使用Hook自动捕捉3x3卷积后BN的特征图响应标准差
    def __init__(self, model):
        self.model = model
        self.hooks = []
        self.layer_stds = []
        self._register_hooks()

    def _register_hooks(self):

        for name, module in self.model.named_modules():

            if isinstance(module, nn.BatchNorm2d):

                hook = module.register_forward_hook(self._make_hook(name))
                self.hooks.append(hook)

    def _make_hook(self, name):
        def hook_fn(module, input, output):
            # output: [N, C, H, W]
            
            with torch.no_grad():
                #计算当前batch所有样本、所有通道和空间位置的整体标准差
                std_val = output.std().item()
                self.layer_stds.append(std_val)
        return hook_fn

    def clear(self):
        self.layer_stds = []

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()


def load_best_checkpoint(model, model_name, device):
    """加载训练好的权重"""
    path = os.path.join("checkpoints", "CIFAR10", model_name, "best_model.pth")
    if os.path.exists(path):
        print(f"[加载权重] {path}")
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

    else:
        print(f"[未找到权重] 使用随机初始化的权重进行测试: {path}")


def collect_responses(model_name, dataloader, device, num_batches=10):
    """在验证集/测试集上前向传播并计算多 Batch 平均响应标准差"""
    model = MODEL_DICT[model_name]().to(device)
    load_best_checkpoint(model, model_name, device)
    model.eval()

    tracker = LayerResponseTracker(model)
    
    all_batch_stds = []
    
    with torch.no_grad():
        for i, (imgs, _) in enumerate(dataloader):
            if i >= num_batches:
                break
            imgs = imgs.to(device)
            tracker.clear()
            _ = model(imgs)
            all_batch_stds.append(tracker.layer_stds)

    tracker.remove_hooks()
    
    # 对多个batch取平均
    mean_layer_stds = np.mean(all_batch_stds, axis=0)
    return mean_layer_stds

"""绘制论文 Figure 7 的 Top 和 Bottom 两个子图"""
def plot_figure7(results, output_img="figure7_response_std.png"):

    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(10, 10))

    # ------------------------------------------------------------
    # Top: 原始网络层顺序
    # ------------------------------------------------------------
    for model_name, stds in results.items():
        color = COLORS.get(model_name, None)
        layers = np.arange(1, len(stds) + 1)
        ax_top.plot(layers, stds, label=model_name, color=color, linewidth=1.8, marker='o', markersize=3)

    ax_top.set_title("Original Order", fontsize=13, fontweight='bold')
    ax_top.set_xlabel("Layer Index", fontsize=11)
    ax_top.set_ylabel("Standard Deviation (std)", fontsize=11)
    ax_top.grid(True, linestyle=":", alpha=0.6)
    ax_top.legend(fontsize=10)
    ax_top.set_ylim(bottom=0)

    # ------------------------------------------------------------
    # Bottom: 按标准差大小降序排列
    # ------------------------------------------------------------
    for model_name, stds in results.items():
        color = COLORS.get(model_name, None)
        sorted_stds = np.sort(stds)[::-1]  # 降序排序
        ranks = np.arange(1, len(sorted_stds) + 1)
        ax_bottom.plot(ranks, sorted_stds, label=model_name, color=color, linewidth=1.8)

    ax_bottom.set_title("Sorted by magnitude", fontsize=13, fontweight='bold')
    ax_bottom.set_xlabel("Ranked Layers", fontsize=11)
    ax_bottom.set_ylabel("Standard Deviation (std)", fontsize=11)
    ax_bottom.grid(True, linestyle=":", alpha=0.6)
    ax_bottom.legend(fontsize=10)
    ax_bottom.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(output_img, dpi=300)
    print(f"\n图片绘制完成，已保存至: {output_img}")


def main():
    parser = argparse.ArgumentParser(description="Plot Layer Response Standard Deviations (Figure 7)")
    parser.add_argument("--models", nargs="+", default=["ResNet20", "ResNet32", "ResNet56", "PlainNet20"],
                        help="选择要对比的网络，支持多个，用空格隔开")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_batches", type=int, default=10, help="用于计算标准差平均值的测试批次数")
    parser.add_argument("--output", type=str, default="figure7_response_std.png")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 获取数据加载器
    _, _, test_loader = CIFAR10_get_dataloader(batch_size=args.batch_size, num_workers=4)

    results = {}
    for model_name in args.models:
        if model_name not in MODEL_DICT:
            print(f"跳过未知模型: {model_name}")
            continue
        print(f"\n分析模型特征响应: {model_name} ...")
        stds = collect_responses(model_name, test_loader, device, num_batches=args.num_batches)
        results[model_name] = stds

    if results:
        plot_figure7(results, output_img=args.output)


if __name__ == "__main__":
    main()