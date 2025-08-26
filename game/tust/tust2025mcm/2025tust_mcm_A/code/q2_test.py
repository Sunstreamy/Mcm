import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull
import os

plt.rcParams["font.sans-serif"] = ["SimHei"]  # 使用SimHei字体
plt.rcParams["axes.unicode_minus"] = False

# 创建figures文件夹（如果不存在）
figures_dir = "../figures2/test_samples"
if not os.path.exists(figures_dir):
    os.makedirs(figures_dir)


def normalize_octahedron(points):
    """
    对八面体进行归一化处理:
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


def visualize_octahedron(points, ax, color="blue", alpha=0.2):
    """
    在给定的3D坐标轴上可视化八面体
    """
    try:
        # 绘制顶点
        ax.scatter(
            points[:, 0], points[:, 1], points[:, 2], color=color, s=50, alpha=1.0
        )

        # 使用ConvexHull计算八面体的面
        hull = ConvexHull(points)

        # 创建三角形面的集合
        faces = []
        for simplex in hull.simplices:
            # 收集构成一个面的三个顶点
            faces.append([points[simplex[0]], points[simplex[1]], points[simplex[2]]])

        # 创建Poly3DCollection并添加到坐标轴
        poly3d = Poly3DCollection(faces, alpha=alpha, linewidths=1, edgecolor="k")
        poly3d.set_facecolor(color)
        ax.add_collection3d(poly3d)

        return True
    except Exception as e:
        print(f"可视化八面体时出错: {e}")
        return False


def plot_octahedron(original, normalized, title="八面体归一化前后对比"):
    """绘制八面体归一化前后的对比图"""
    fig = plt.figure(figsize=(14, 6))

    # 计算原始八面体的重心
    centroid = np.mean(original, axis=0)

    # 计算原始八面体相对于重心的最大距离
    max_distance = max([np.linalg.norm(point - centroid) for point in original])

    # 使用一个统一的单位长度刻度
    # 计算合适的刻度（0.5单位）
    grid_unit = 0.5

    # 原始八面体
    ax1 = fig.add_subplot(121, projection="3d")
    visualize_octahedron(original, ax1, color="blue", alpha=0.3)

    # 添加重心
    ax1.scatter(
        centroid[0],
        centroid[1],
        centroid[2],
        color="green",
        s=100,
        marker="*",
        label="重心",
    )

    # 设置x轴范围和刻度
    x_mid = centroid[0]
    radius = max_distance * 1.2
    x_min = np.floor((x_mid - radius) / grid_unit) * grid_unit
    x_max = np.ceil((x_mid + radius) / grid_unit) * grid_unit
    ax1.set_xlim(x_min, x_max)
    ax1.set_xticks(np.arange(x_min, x_max + grid_unit, grid_unit))

    # 设置y轴范围和刻度
    y_mid = centroid[1]
    y_min = np.floor((y_mid - radius) / grid_unit) * grid_unit
    y_max = np.ceil((y_mid + radius) / grid_unit) * grid_unit
    ax1.set_ylim(y_min, y_max)
    ax1.set_yticks(np.arange(y_min, y_max + grid_unit, grid_unit))

    # 设置z轴范围和刻度
    z_mid = centroid[2]
    z_min = np.floor((z_mid - radius) / grid_unit) * grid_unit
    z_max = np.ceil((z_mid + radius) / grid_unit) * grid_unit
    ax1.set_zlim(z_min, z_max)
    ax1.set_zticks(np.arange(z_min, z_max + grid_unit, grid_unit))

    # 添加坐标轴标签
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    ax1.set_title("原始八面体")
    ax1.legend()

    # 归一化后的八面体
    ax2 = fig.add_subplot(122, projection="3d")
    visualize_octahedron(normalized, ax2, color="blue", alpha=0.3)

    # 归一化后重心应该在原点
    ax2.scatter(0, 0, 0, color="green", s=100, marker="*", label="重心(原点)")

    # 计算归一化后的半径，确保等比例
    norm_max = np.max([np.linalg.norm(point) for point in normalized])
    scale_ratio = max_distance / norm_max
    norm_radius = radius

    # 设置归一化后图形的刻度和范围，使用相同的grid_unit
    norm_min = np.floor(-norm_radius / grid_unit) * grid_unit
    norm_max = np.ceil(norm_radius / grid_unit) * grid_unit

    ax2.set_xlim(norm_min, norm_max)
    ax2.set_ylim(norm_min, norm_max)
    ax2.set_zlim(norm_min, norm_max)

    ax2.set_xticks(np.arange(norm_min, norm_max + grid_unit, grid_unit))
    ax2.set_yticks(np.arange(norm_min, norm_max + grid_unit, grid_unit))
    ax2.set_zticks(np.arange(norm_min, norm_max + grid_unit, grid_unit))

    # 添加坐标轴标签
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")
    ax2.set_title("归一化后的八面体")
    ax2.legend()

    # 设置相同的视角以便比较
    ax1.view_init(elev=20, azim=30)
    ax2.view_init(elev=20, azim=30)

    # 确保图形的大小相等
    ax1.set_box_aspect([1, 1, 1])
    ax2.set_box_aspect([1, 1, 1])

    # 启用网格
    ax1.grid(True)
    ax2.grid(True)

    plt.suptitle(title)
    plt.tight_layout()

    # 保存图像
    plt.savefig(
        f"{figures_dir}/octahedron_normalization.png", dpi=300, bbox_inches="tight"
    )
    plt.show()


def main():
    # 待测八面体1的顶点坐标
    octahedron1 = np.array(
        [
            [-2.1389, 0.4082, 2.3177],
            [-1.3502, 0.1494, 2.8754],
            [-1.9436, -0.4082, 3.4558],
            [-2.7323, -0.1494, 2.8981],
            [-1.9275, -0.5577, 2.4671],
            [-2.1549, 0.5577, 3.3064],
        ]
    )

    # 归一化处理
    normalized_octahedron, centroid, max_edge = normalize_octahedron(octahedron1)

    print(f"原始八面体重心: {centroid}")
    print(f"最大边长: {max_edge}")

    # 显示归一化前后的顶点坐标
    print("\n原始八面体顶点坐标:")
    for i, point in enumerate(octahedron1):
        print(f"顶点{i+1}: [{point[0]:.4f}, {point[1]:.4f}, {point[2]:.4f}]")

    print("\n归一化后八面体顶点坐标:")
    for i, point in enumerate(normalized_octahedron):
        print(f"顶点{i+1}: [{point[0]:.4f}, {point[1]:.4f}, {point[2]:.4f}]")

    # 绘制对比图
    plot_octahedron(octahedron1, normalized_octahedron, "八面体1归一化前后对比")


if __name__ == "__main__":
    main()
