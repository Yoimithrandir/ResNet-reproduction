import os
import glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

# ============================================================
# 配置
# ============================================================

# 选择画error或loss
PLOT_TYPE = "error"

# 要画的模型列表
MODELS = [
    # "PlainNet20",
    # "PlainNet32",
    # "PlainNet44",
    # "PlainNet56",
    "ResNet20",
    "ResNet32",
    "ResNet44",
    "ResNet56",
    "ResNet110",
]

# 日志根目录
LOG_DIR = "runs/CIFAR10"

# 采样步长
TRAIN_STRIDE = 1
VAL_STRIDE = 1

# x 轴单位："1e4"或 "1e3"
X_UNIT = 1e4

# 画error时的纵轴上限（%）
Y_MAX_ERROR = 30

# 保存的图片文件名
OUTPUT_IMG = "curve_cifar10.png"

# ============================================================
# 颜色方案
# ============================================================
COLORS = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # gray
    "#bcbd22",  # olive
    "#17becf",  # cyan
]


def find_event_file(log_dir, model):
    """在log_dir下寻找包含指定model路径的最新tfevents文件"""
    pattern = os.path.join(log_dir, "**", model, "**", "*.tfevents.*")
    files = glob.glob(pattern, recursive=True)
    
    if not files:
        pattern_fallback = os.path.join(log_dir, model, "*.tfevents.*")
        files = glob.glob(pattern_fallback)

    if not files:
        return None

    # 按文件修改时间排序，取最新的一个event文件
    files = sorted(files, key=os.path.getmtime)
    return files[-1]


def load_scalars(log_dir, model):
    """读取scalar数据，并对acc/error乘以 100"""
    event_file = find_event_file(log_dir, model)
    if event_file is None:
        print(f"[跳过] 未找到 {model} 的 event 文件")
        return None

    print(f"[读取] 找到 {model} 的日志文件: {event_file}")
    ea = EventAccumulator(event_file)
    ea.Reload()

    tags = ea.Tags().get("scalars", [])
    result = {}

    targets = [
        ("train_loss",  ["train/loss", f"{model}/train_loss", f"{model}/train/loss"]),
        ("val_loss",    ["val/loss",   f"{model}/val_loss",   f"{model}/val/loss"]),
        ("train_error", ["train/error", f"{model}/train_error", f"{model}/train/error",
                         "train/acc", f"{model}/train_acc", f"{model}/train/acc"]),
        ("val_error",   ["val/error", f"{model}/val_error", f"{model}/val/error",
                         "val/acc", f"{model}/val_acc", f"{model}/val/acc"]),
    ]

    for key, possible_tags in targets:
        for t in possible_tags:
            if t in tags:
                scalars = ea.Scalars(t)
                steps = [s.step for s in scalars]
                raw_values = [s.value for s in scalars]

                # 如果是error/acc，直接乘以 100 转换成百分比
                if "error" in key:
                    values = [v * 100.0 for v in raw_values]
                else:
                    values = raw_values

                result[key] = (steps, values)
                break

    return result


def main():
    assert PLOT_TYPE in ("loss", "error")

    plt.figure(figsize=(12, 7))

    for i, model in enumerate(MODELS):
        color = COLORS[i % len(COLORS)]
        data = load_scalars(LOG_DIR, model)
        if not data:
            continue

        prefix_list = [("train", 1.2, "--", TRAIN_STRIDE), ("val", 2.5, "-", VAL_STRIDE)]

        for prefix, lw, ls, stride in prefix_list:
            key = f"{prefix}_{PLOT_TYPE}"
            if key not in data:
                print(f"警告: {model} 中未找到 {key} 标量数据")
                continue

            steps, values = data[key]
            
            # 步长切片采样
            steps = steps[::stride]
            values = values[::stride]

            # 缩放X轴单位
            steps = [s / X_UNIT for s in steps]
            label = f"{model} {prefix}"

            plt.plot(steps, values, color=color, linewidth=lw, linestyle=ls, label=label)

    # 坐标与图例设置
    if X_UNIT == 1e4:
        plt.xlabel(r"Iteration ($\times 10^4$)", fontsize=12)
    elif X_UNIT == 1e3:
        plt.xlabel(r"Iteration ($\times 10^3$)", fontsize=12)
    else:
        plt.xlabel("Iteration", fontsize=12)

    if PLOT_TYPE == "loss":
        plt.ylabel("Loss", fontsize=12)
    else:
        plt.ylabel("Error (%)", fontsize=12)
        plt.ylim(0, Y_MAX_ERROR)

    plt.legend(fontsize=9, ncol=2, loc="upper right")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    
    plt.savefig(OUTPUT_IMG, dpi=300)
    print(f"\n绘图完成,图片已保存至: {OUTPUT_IMG}")


if __name__ == "__main__":
    main()