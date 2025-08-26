import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull

from q2 import negative_samples_3d_list


def visualize_negative_sample(points, name):
    """
    可视化三维负样本

    参数:
    points: 形状为(N, 3)的点集
    name: 样本名称
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # 绘制点
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c="blue", marker="o", s=100)

    # 尝试绘制凸包
    try:
        hull = ConvexHull(points)
        for simplex in hull.simplices:
            vertices = points[simplex]
            # 创建三角形面
            poly = Poly3DCollection([vertices], alpha=0.3)
            poly.set_facecolor("cyan")
            poly.set_edgecolor("black")
            ax.add_collection3d(poly)
    except Exception as e:
        print(f"无法为 {name} 创建凸包: {e}")

    # 设置图表属性
    ax.set_title(f"负样本: {name}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # 使坐标轴比例相等
    max_range = np.max(np.ptp(points, axis=0)) / 2
    mid = np.mean(points, axis=0)
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

    plt.tight_layout()

    return fig, ax


def main():
    """测试并可视化所有负样本"""
    print("=== 三维负样本数据测试 ===")
    print(f"负样本总数: {len(negative_samples_3d_list)}")
    print("\n各负样本信息:")

    for i, (sample, name) in enumerate(negative_samples_3d_list):
        print(f"{i+1}. {name}: {sample.shape[0]}个顶点")

        # 计算点云的基本统计信息
        centroid = np.mean(sample, axis=0)
        min_coords = np.min(sample, axis=0)
        max_coords = np.max(sample, axis=0)

        print(
            f"   - 质心坐标: [{centroid[0]:.3f}, {centroid[1]:.3f}, {centroid[2]:.3f}]"
        )
        print(
            f"   - 坐标范围: X[{min_coords[0]:.3f}, {max_coords[0]:.3f}], "
            f"Y[{min_coords[1]:.3f}, {max_coords[1]:.3f}], "
            f"Z[{min_coords[2]:.3f}, {max_coords[2]:.3f}]"
        )

        # 尝试计算凸包体积
        try:
            hull = ConvexHull(sample)
            print(f"   - 凸包体积: {hull.volume:.6f}")
            print(f"   - 凸包表面积: {hull.area:.6f}")
        except Exception as e:
            print(f"   - 无法计算凸包: {e}")

        print()

        # 可视化负样本
        fig, ax = visualize_negative_sample(sample, name)
        plt.show()


if __name__ == "__main__":
    main()
