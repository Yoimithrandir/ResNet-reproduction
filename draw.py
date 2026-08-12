from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os
import matplotlib.pyplot as plt

# ============================================================
# 用户配置 — 修改这里
# ============================================================

# 选择要画的网络（runs/ 下的文件夹名）
MODELS = [
    #"PlainNet18",
    #"PlainNet34",
    #"ResNet18",
    "ResNet34A",
    "ResNet34B",
    "ResNet34C",
]

# 画什么："loss" 或 "error"
PLOT_TYPE = "error"

# 日志目录
LOG_DIR = "runs"

# 采样步长（train 点密可跳着取，val 点少通常不用跳）
TRAIN_STRIDE = 5
VAL_STRIDE = 1

# x 轴单位："1e4"（万次迭代）或 "1e3"（千次迭代）
X_UNIT = 1e4

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


def load_scalars(log_dir, model):
    """读取指定模型 event 文件中的所有 scalar 数据"""
    model_dir = os.path.join(log_dir, model)
    if not os.path.isdir(model_dir):
        print(f"[跳过] 目录不存在: {model_dir}")
        return None

    # 找 event 文件
    event_files = []
    for f in os.listdir(model_dir):
        if f.startswith("events.out.tfevents"):
            event_files.append(os.path.join(model_dir, f))
    if not event_files:
        print(f"[跳过] 未找到 event 文件: {model_dir}")
        return None

    event_file = event_files[0]
    ea = EventAccumulator(event_file)
    ea.Reload()

    tags = ea.Tags().get("scalars", [])
    result = {}
    for tag in ["train/loss", "val/loss", "train/acc", "val/acc"]:
        if tag in tags:
            scalars = ea.Scalars(tag)
            result[tag] = ([s.step for s in scalars], [s.value for s in scalars])
    return result


def main():
    assert PLOT_TYPE in ("loss", "error"), "PLOT_TYPE 必须是 'loss' 或 'error'"

    plt.figure(figsize=(14, 8))

    for i, model in enumerate(MODELS):
        color = COLORS[i % len(COLORS)]
        data = load_scalars(LOG_DIR, model)
        if data is None:
            continue

        if PLOT_TYPE == "loss":
            # --- loss ---
            for prefix, lw, stride in [
                ("train", 1.2, TRAIN_STRIDE),
                ("val", 3.0, VAL_STRIDE),
            ]:
                tag = f"{prefix}/loss"
                if tag not in data:
                    continue
                steps, values = data[tag]
                steps = steps[::stride]
                values = values[::stride]
                steps = [s / X_UNIT for s in steps]
                label = f"{model} {prefix}"
                plt.plot(steps, values, color=color, linewidth=lw, label=label,
                         linestyle="-" if prefix == "val" else "--")

        else:
            # --- error = (1 - acc) * 100 ---
            for prefix, lw, stride in [
                ("train", 1.2, TRAIN_STRIDE),
                ("val", 3.0, VAL_STRIDE),
            ]:
                tag = f"{prefix}/acc"
                if tag not in data:
                    continue
                steps, values = data[tag]
                steps = steps[::stride]
                values = values[::stride]
                values = [(1 - v) * 100 for v in values]
                steps = [s / X_UNIT for s in steps]
                label = f"{model} {prefix}"
                plt.plot(steps, values, color=color, linewidth=lw, label=label,
                         linestyle="-" if prefix == "val" else "--")

    # 坐标标签
    if X_UNIT == 1e4:
        plt.xlabel(r"Iteration ($\times 10^4$)")
    else:
        plt.xlabel("Iteration")

    if PLOT_TYPE == "loss":
        plt.ylabel("Loss")
    else:
        plt.ylabel("Error (%)")

    #plt.title("Training Curve")
    plt.legend(fontsize=8, ncol=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("curve.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
