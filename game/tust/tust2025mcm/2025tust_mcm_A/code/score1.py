#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
score1.py: 生成双柱状统计图对比每个待测五边形对两个标准模型的综合评分
使用标准柱状图对比五个观测五边形与两个标准模型的综合评分，在阈值λ=0.48处绘制水平虚线
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import MaxNLocator
import matplotlib.gridspec as gridspec
from matplotlib import ticker

# 设置字体为SimHei，以支持中文显示
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 使用SimHei字体
plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号

# 创建输出目录
FIGURES_DIR_Q1 = os.path.join("..", "figures")
ANALYSIS_DIR_Q1 = os.path.join(FIGURES_DIR_Q1, "analysis")
os.makedirs(ANALYSIS_DIR_Q1, exist_ok=True)


def plot_comparison_scores(save_path=None):
    """
    生成双柱状统计图，对比每个待测五边形对两个标准模型的综合评分，
    并在阈值λ=0.48处添加红色虚线。

    参数:
    save_path: 图像保存路径，默认为None（仅显示不保存）
    """
    # 对五个观测五边形的分类结果
    labels = [
        "观测五边形1",
        "观测五边形2",
        "观测五边形3",
        "观测五边形4",
        "观测五边形5",
    ]

    # 根据提供的数据填充每个五边形对两个模型的评分值
    # 综合评分S_I（与类别I的匹配度，越小越匹配）
    scores_class_I = [0.0578, 1.3265, 0.8485, 0.0581, 1.4862]

    # 综合评分S_II（与类别II的匹配度，越小越匹配）
    scores_class_II = [1.2782, 0.0924, 1.0704, 1.2747, 0.2965]

    # 创建一个新的图形
    fig = plt.figure(figsize=(12, 8))

    # 使用GridSpec创建更复杂的布局
    gs = gridspec.GridSpec(1, 1)
    ax = fig.add_subplot(gs[0])

    # 设置柱状图的宽度和位置
    width = 0.35
    x = np.arange(len(labels))

    # 绘制双柱状图 - 使用默认颜色
    rects1 = ax.bar(
        x - width / 2,
        scores_class_I,
        width,
        label="类别I评分",
        alpha=0.8,
    )
    rects2 = ax.bar(
        x + width / 2,
        scores_class_II,
        width,
        label="类别II评分",
        alpha=0.8,
    )

    # 绘制阈值线λ=0.79
    ax.axhline(
        y=0.79,
        linestyle="--",
        color="red",
        linewidth=2.0,
        alpha=0.7,
        label="λ阈值 = 0.79",
    )

    # 添加文本标签，表明此阈值的意义
    ax.text(4.7, 0.72, "λ = 0.79", color="red", fontsize=14, va="bottom", ha="right")

    # 添加箭头指示正确的分类区域
    ax.annotate(
        "低于阈值才被归类",
        xy=(2, 0.50),
        xytext=(2, 0.40),
        arrowprops=dict(
            facecolor="black", shrink=0.05, width=2, headwidth=10, alpha=0.7
        ),
        ha="center",
        fontsize=14,
    )

    # 在图上标记正确的分类结果
    classifications = ["类别I", "类别II", "未知类别", "类别I", "类别II"]
    for i, (c, s1, s2) in enumerate(
        zip(classifications, scores_class_I, scores_class_II)
    ):
        min_score = min(s1, s2)
        y_pos = min_score - 0.05
        if y_pos < 0.1:  # 如果太低，则在上方标注
            y_pos = min_score + 0.08
        ax.text(
            i,
            y_pos,
            f"{c}",
            ha="center",
            fontsize=14,
            color="darkblue" if c != "未知类别" else "darkred",
            fontweight="bold",
        )

    # 设置轴标签 - 增大字体
    ax.set_ylabel("综合评分值", fontsize=20)
  
    # 设置X轴刻度标签
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=20)

    # 设置Y轴范围，确保从0开始，并适当调整上限
    ax.set_ylim(0, 2.1)

    # 设置Y轴刻度标签字体大小
    ax.tick_params(axis="y", labelsize=20)

    # 添加网格线以提高可读性
    ax.grid(True, linestyle="--", alpha=0.3)

    # 添加一条水平基线
    ax.axhline(y=0, color="black", linewidth=0.8)

    # 添加图例 - 增大字体
    ax.legend(loc="upper right", fontsize=16)

    # 在每个柱子上方添加数值标签
    def add_labels(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height:.4f}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),  # 3点垂直偏移
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=16,
            )

    add_labels(rects1)
    add_labels(rects2)

    # 使用一个技巧来确保整数刻度
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.25))

    # 添加边框
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)

    # 调整布局
    plt.tight_layout()

    # 如果提供了保存路径，保存图形
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"评分对比图已保存至: {save_path}")
    else:
        # 否则显示图形
        plt.show()

    # 返回图形对象以便进一步自定义
    return fig, ax


if __name__ == "__main__":
    # 调用函数生成评分图表并保存
    save_path = os.path.join(ANALYSIS_DIR_Q1, "pentagon_classification_scores.png")
    plot_comparison_scores(save_path)
