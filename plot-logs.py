from pathlib import Path
import json
import re
import subprocess

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


# ========================
# 路径设置
# ========================

PROJECT_DIR = Path(
    "."
)

LOGS_DIR = PROJECT_DIR / "training_logs"
OUTPUT_PATH = PROJECT_DIR / "training_metrics.png"


# ========================
# 读取训练日志
# ========================

pattern = re.compile(r"model_(\d+)\.json$")

logs = []

for log_path in LOGS_DIR.glob("model_*.json"):
    match = pattern.fullmatch(log_path.name)

    if match is None:
        continue

    with log_path.open(
        "r",
        encoding="utf-8",
    ) as log_file:
        data = json.load(log_file)

    epochs = data.get("epochs", [])

    if not epochs:
        continue

    logs.append(
        {
            "model_id": int(
                data.get(
                    "trained_model_id",
                    match.group(1),
                )
            ),
            "decision_count": int(
                data.get("decision_count", 0)
            ),
            "actor_valid_count": int(
                data.get("actor_valid_count", 0)
            ),
            "epochs": epochs,
        }
    )

logs.sort(
    key=lambda item: item["model_id"]
)

if not logs:
    raise RuntimeError(
        f"在 {LOGS_DIR} 中没有找到有效训练日志"
    )


# ========================
# 整理数据
# ========================

model_ids = [
    item["model_id"]
    for item in logs
]

decision_counts = [
    item["decision_count"]
    for item in logs
]

actor_valid_counts = [
    item["actor_valid_count"]
    for item in logs
]

epoch_numbers = sorted(
    {
        int(epoch["epoch"])
        for item in logs
        for epoch in item["epochs"]
    }
)


# ========================
# 创建图表
# ========================

plt.style.use("seaborn-v0_8-whitegrid")

fig, axes = plt.subplots(
    2,
    2,
    figsize=(16, 10),
)

policy_axis = axes[0, 0]
value_axis = axes[0, 1]
entropy_axis = axes[1, 0]
samples_axis = axes[1, 1]


# ========================
# 每个 epoch 分别画线
# ========================

for epoch_number in epoch_numbers:
    epoch_model_ids = []
    policy_losses = []
    value_losses = []
    entropies = []

    for item in logs:
        matching_epoch = next(
            (
                epoch
                for epoch in item["epochs"]
                if int(epoch["epoch"])
                == epoch_number
            ),
            None,
        )

        if matching_epoch is None:
            continue

        epoch_model_ids.append(
            item["model_id"]
        )

        policy_losses.append(
            float(
                matching_epoch["policy_loss"]
            )
        )

        value_losses.append(
            float(
                matching_epoch["value_loss"]
            )
        )

        entropies.append(
            float(
                matching_epoch["entropy"]
            )
        )

    label = f"Epoch {epoch_number}"

    policy_axis.plot(
        epoch_model_ids,
        policy_losses,
        marker="o",
        markersize=4,
        linewidth=1.5,
        label=label,
    )

    value_axis.plot(
        epoch_model_ids,
        value_losses,
        marker="o",
        markersize=4,
        linewidth=1.5,
        label=label,
    )

    entropy_axis.plot(
        epoch_model_ids,
        entropies,
        marker="o",
        markersize=4,
        linewidth=1.5,
        label=label,
    )


# ========================
# Policy loss
# ========================

policy_axis.axhline(
    0.0,
    color="black",
    linewidth=1,
    alpha=0.5,
)

policy_axis.set_title("PPO Policy Loss")
policy_axis.set_xlabel("Model ID")
policy_axis.set_ylabel("Policy loss")

policy_axis.ticklabel_format(
    axis="y",
    style="sci",
    scilimits=(-3, 3),
)

policy_axis.legend()


# ========================
# Value loss
# ========================

value_axis.set_title("Value Loss")
value_axis.set_xlabel("Model ID")
value_axis.set_ylabel("Value loss")
value_axis.legend()


# ========================
# Entropy
# ========================

entropy_axis.axhline(
    np.log(3),
    color="red",
    linestyle="--",
    linewidth=1.5,
    alpha=0.7,
    label="ln(3): maximum for 3 actions",
)

entropy_axis.set_title(
    "Masked Policy Entropy"
)

entropy_axis.set_xlabel("Model ID")
entropy_axis.set_ylabel("Entropy")
entropy_axis.legend()


# ========================
# 训练样本数量
# ========================

samples_axis.plot(
    model_ids,
    decision_counts,
    marker="o",
    linewidth=2,
    label="All model decisions",
)

samples_axis.plot(
    model_ids,
    actor_valid_counts,
    marker="o",
    linewidth=2,
    label="Actor-valid decisions",
)

samples_axis.fill_between(
    model_ids,
    actor_valid_counts,
    decision_counts,
    alpha=0.2,
    label="Actor-invalid decisions",
)

samples_axis.set_title("Training Samples")
samples_axis.set_xlabel("Model ID")
samples_axis.set_ylabel("Decision count")
samples_axis.legend()


# ========================
# 保存 PNG
# ========================

fig.suptitle(
    "Hungry Geese PPO Training Metrics",
    fontsize=18,
)

fig.tight_layout(
    rect=(0, 0, 1, 0.96),
)

fig.savefig(
    OUTPUT_PATH,
    dpi=200,
    bbox_inches="tight",
)

plt.close(fig)

print(f"PNG saved to: {OUTPUT_PATH.resolve()}")


# ========================
# macOS 自动打开图片
# ========================

subprocess.run(
    ["open", str(OUTPUT_PATH)],
    check=True,
)