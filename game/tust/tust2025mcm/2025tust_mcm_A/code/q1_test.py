import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams["font.sans-serif"] = ["SimHei"]  # 使用SimHei字体
plt.rcParams["axes.unicode_minus"] = False

# 创建figures文件夹（如果不存在）
figures_dir = "../figures"
if not os.path.exists(figures_dir):
    os.makedirs(figures_dir)


def normalize_pentagon(points):
    """
    对五边形进行归一化处理:
    1. 将重心移动到原点
    2. 按最大边长进行归一化
    """
    # 计算重心
    centroid = np.mean(points, axis=0)

    # 将重心平移到原点
    centered_points = points - centroid

    # 计算最大边长
    max_edge_length = 0
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            edge_length = np.linalg.norm(centered_points[i] - centered_points[j])
            max_edge_length = max(max_edge_length, edge_length)

    # 按最大边长归一化
    normalized_points = centered_points / max_edge_length

    return normalized_points, centroid, max_edge_length


def plot_pentagon(original, normalized, title="五边形归一化前后对比"):
    """绘制五边形归一化前后的对比图"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 原始五边形
    ax1.plot(
        np.append(original[:, 0], original[0, 0]),
        np.append(original[:, 1], original[0, 1]),
        "b-",
        linewidth=2,
    )
    ax1.scatter(original[:, 0], original[:, 1], color="red", s=50)
    ax1.set_title("原始五边形")
    ax1.grid(True)

    # 计算原始五边形的重心
    centroid = np.mean(original, axis=0)
    ax1.scatter(
        centroid[0], centroid[1], color="green", s=100, marker="*", label="重心"
    )
    ax1.legend()

    # 归一化后的五边形
    ax2.plot(
        np.append(normalized[:, 0], normalized[0, 0]),
        np.append(normalized[:, 1], normalized[0, 1]),
        "b-",
        linewidth=2,
    )
    ax2.scatter(normalized[:, 0], normalized[:, 1], color="red", s=50)
    ax2.set_title("归一化后的五边形")
    ax2.grid(True)

    # 归一化后重心应该在原点
    ax2.scatter(0, 0, color="green", s=100, marker="*", label="重心(原点)")
    ax2.legend()

    # 添加坐标轴
    ax2.axhline(y=0, color="k", linestyle="-", alpha=0.3)
    ax2.axvline(x=0, color="k", linestyle="-", alpha=0.3)

    # 设置坐标轴比例相等
    ax1.set_aspect("equal")
    ax2.set_aspect("equal")

    # 计算适当的x和y轴显示范围
    original_range_x = np.max(original[:, 0]) - np.min(original[:, 0])
    original_range_y = np.max(original[:, 1]) - np.min(original[:, 1])
    normalized_range_x = np.max(normalized[:, 0]) - np.min(normalized[:, 0])
    normalized_range_y = np.max(normalized[:, 1]) - np.min(normalized[:, 1])

    # 计算原始图形的中心
    original_midx = (np.max(original[:, 0]) + np.min(original[:, 0])) / 2
    original_midy = (np.max(original[:, 1]) + np.min(original[:, 1])) / 2

    # 计算归一化图形的中心
    normalized_midx = (np.max(normalized[:, 0]) + np.min(normalized[:, 0])) / 2
    normalized_midy = (np.max(normalized[:, 1]) + np.min(normalized[:, 1])) / 2

    # 确定两个图的最大范围
    max_range = (
        max(original_range_x, original_range_y, normalized_range_x, normalized_range_y)
        * 1.2
    )

    # 设置两个子图使用相同的显示范围
    ax1.set_xlim(original_midx - max_range / 2, original_midx + max_range / 2)
    ax1.set_ylim(original_midy - max_range / 2, original_midy + max_range / 2)
    ax2.set_xlim(normalized_midx - max_range / 2, normalized_midx + max_range / 2)
    ax2.set_ylim(normalized_midy - max_range / 2, normalized_midy + max_range / 2)

    plt.suptitle(title)
    plt.tight_layout()

    # 使用唯一的文件名保存不同的图像
    filename = f"../figures/{title.replace(' ', '_')}.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()


def main():
    # 标准类别I的五边形顶点坐标
    pentagon_class1 = np.array(
        [[0.74, 1.90], [1.26, 3.49], [3.54, 3.58], [3.13, 2.28], [4.45, 1.16]]
    )

    # 待测五边形1的顶点坐标
    test_pentagon1 = np.array(
        [[5.69, 3.94], [5.62, 4.31], [6.07, 4.61], [6.13, 4.29], [6.53, 4.21]]
    )

    # 归一化处理
    normalized_pentagon, centroid, max_edge = normalize_pentagon(test_pentagon1)

    print(f"原始五边形重心: {centroid}")
    print(f"最大边长: {max_edge}")

    # 绘制对比图
    plot_pentagon(test_pentagon1, normalized_pentagon, "五边形1归一化前后对比")

    # 也可以对标准类别I的五边形进行归一化处理并绘图
    normalized_class1, centroid_class1, max_edge_class1 = normalize_pentagon(
        pentagon_class1
    )
    plot_pentagon(pentagon_class1, normalized_class1, "标准类别I五边形归一化前后对比")


if __name__ == "__main__":
    main()
