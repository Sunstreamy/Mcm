#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可视化函数模块：包含五边形和八面体的可视化函数 (简化版)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull


def plot_pentagon(original, normalized, title="五边形归一化前后对比"):
    """绘制五边形归一化前后的对比图"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # 原始五边形
    ax1.plot(
        np.append(original[:, 0], original[0, 0]),
        np.append(original[:, 1], original[0, 1]),
        "b-",
    )
    ax1.scatter(original[:, 0], original[:, 1], color="red")
    ax1.set_title("原始五边形")
    ax1.grid(True)

    # 计算并显示重心
    centroid = np.mean(original, axis=0)
    ax1.scatter(centroid[0], centroid[1], color="green", marker="*", label="重心")
    ax1.legend()

    # 归一化后的五边形
    ax2.plot(
        np.append(normalized[:, 0], normalized[0, 0]),
        np.append(normalized[:, 1], normalized[0, 1]),
        "b-",
    )
    ax2.scatter(normalized[:, 0], normalized[:, 1], color="red")
    ax2.scatter(0, 0, color="green", marker="*", label="重心(原点)")
    ax2.set_title("归一化后的五边形")
    ax2.grid(True)
    ax2.legend()

    # 设置坐标轴比例相等
    ax1.set_aspect("equal")
    ax2.set_aspect("equal")

    plt.tight_layout()
    plt.show()


def visualize_classification(pentagon, class_I, class_II, scores, save_path=None):
    """可视化五边形分类结果"""
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    # 绘制待测五边形
    ax[0].add_patch(Polygon(pentagon, fill=False, edgecolor="blue"))
    ax[0].scatter(pentagon[:, 0], pentagon[:, 1], color="blue", label="待测五边形")
    ax[0].set_title("待测五边形（归一化）")
    ax[0].set_aspect("equal")
    ax[0].set_xlim(-1.5, 1.5)
    ax[0].set_ylim(-1.5, 1.5)
    ax[0].grid(True)
    ax[0].legend()

    # 绘制与类别I的对比
    ax[1].add_patch(Polygon(pentagon, fill=False, edgecolor="blue"))
    ax[1].add_patch(Polygon(class_I, fill=False, edgecolor="red"))
    ax[1].scatter(pentagon[:, 0], pentagon[:, 1], color="blue", label="待测五边形")
    ax[1].scatter(class_I[:, 0], class_I[:, 1], color="red", label="标准类别I")
    ax[1].set_title(f"与类别I对比 (RMSD={scores['Class I']['RMSD']:.4f})")
    ax[1].set_aspect("equal")
    ax[1].set_xlim(-1.5, 1.5)
    ax[1].set_ylim(-1.5, 1.5)
    ax[1].grid(True)
    ax[1].legend()

    # 绘制与类别II的对比
    ax[2].add_patch(Polygon(pentagon, fill=False, edgecolor="blue"))
    ax[2].add_patch(Polygon(class_II, fill=False, edgecolor="green"))
    ax[2].scatter(pentagon[:, 0], pentagon[:, 1], color="blue", label="待测五边形")
    ax[2].scatter(class_II[:, 0], class_II[:, 1], color="green", label="标准类别II")
    ax[2].set_title(f"与类别II对比 (RMSD={scores['Class II']['RMSD']:.4f})")
    ax[2].set_aspect("equal")
    ax[2].set_xlim(-1.5, 1.5)
    ax[2].set_ylim(-1.5, 1.5)
    ax[2].grid(True)
    ax[2].legend()

    # 添加总标题
    plt.suptitle(f"五边形分类结果: {scores['Classification']}", fontsize=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()


def visualize_octahedron_simple(points, ax, color="blue", alpha=0.3):
    """简单绘制八面体"""
    # 绘制点云
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=color, marker="o")

    try:
        hull = ConvexHull(points)

        # 绘制边和面
        for simplex in hull.simplices:
            # 创建面
            verts = [points[simplex]]
            poly = Poly3DCollection(verts, alpha=alpha)
            poly.set_facecolor(color)
            ax.add_collection3d(poly)

            # 绘制边
            for i in range(len(simplex)):
                j = simplex[(i + 1) % len(simplex)]
                ax.plot(
                    [points[simplex[i], 0], points[j, 0]],
                    [points[simplex[i], 1], points[j, 1]],
                    [points[simplex[i], 2], points[j, 2]],
                    color=color,
                    alpha=0.7,
                )
    except Exception as e:
        print(f"绘制八面体面时出错: {e}")


def plot_octahedron(original, normalized, title="八面体归一化前后对比"):
    """绘制八面体归一化前后的对比图"""
    fig = plt.figure(figsize=(12, 5))

    # 计算原始八面体的重心
    centroid = np.mean(original, axis=0)
    max_distance = max([np.linalg.norm(point - centroid) for point in original])

    # 原始八面体
    ax1 = fig.add_subplot(121, projection="3d")
    visualize_octahedron_simple(original, ax1, color="blue", alpha=0.3)
    ax1.scatter(
        centroid[0], centroid[1], centroid[2], color="green", marker="*", label="重心"
    )

    # 设置坐标轴范围
    radius = max_distance * 1.2
    x_mid, y_mid, z_mid = centroid
    ax1.set_xlim(x_mid - radius, x_mid + radius)
    ax1.set_ylim(y_mid - radius, y_mid + radius)
    ax1.set_zlim(z_mid - radius, z_mid + radius)
    ax1.set_title("原始八面体")
    ax1.legend()

    # 归一化后的八面体
    ax2 = fig.add_subplot(122, projection="3d")
    visualize_octahedron_simple(normalized, ax2, color="blue", alpha=0.3)
    ax2.scatter(0, 0, 0, color="green", marker="*", label="重心(原点)")
    ax2.set_xlim(-1.5, 1.5)
    ax2.set_ylim(-1.5, 1.5)
    ax2.set_zlim(-1.5, 1.5)
    ax2.set_title("归一化后的八面体")
    ax2.legend()

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()


def visualize_alignment(detailed_results, title_prefix="", save_path=None):
    """可视化八面体点集的对齐结果"""
    fig = plt.figure(figsize=(15, 5))

    # 提取数据
    observed = detailed_results["observed_normalized"]
    class_I = detailed_results["class_I_normalized"]
    class_II = detailed_results["class_II_normalized"]
    rmsd_I = detailed_results["aligned_to_class_I"]["rmsd"]
    rmsd_II = detailed_results["aligned_to_class_II"]["rmsd"]
    classification = detailed_results["classification"]

    # 设置子图1 - 观测八面体与类别I的对比
    ax1 = fig.add_subplot(131, projection="3d")
    visualize_octahedron_simple(observed, ax1, color="blue", alpha=0.2)
    visualize_octahedron_simple(class_I, ax1, color="red", alpha=0.2)
    ax1.scatter([], [], [], c="blue", label="观测八面体")
    ax1.scatter([], [], [], c="red", label="标准类别I")
    ax1.set_title(f"与类别I对比\nRMSD: {rmsd_I:.4f}")
    ax1.legend()
    ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(-1.5, 1.5)
    ax1.set_zlim(-1.5, 1.5)

    # 设置子图2 - 观测八面体与类别II的对比
    ax2 = fig.add_subplot(132, projection="3d")
    visualize_octahedron_simple(observed, ax2, color="blue", alpha=0.2)
    visualize_octahedron_simple(class_II, ax2, color="green", alpha=0.2)
    ax2.scatter([], [], [], c="blue", label="观测八面体")
    ax2.scatter([], [], [], c="green", label="标准类别II")
    ax2.set_title(f"与类别II对比\nRMSD: {rmsd_II:.4f}")
    ax2.legend()
    ax2.set_xlim(-1.5, 1.5)
    ax2.set_ylim(-1.5, 1.5)
    ax2.set_zlim(-1.5, 1.5)

    # 设置子图3 - 根据分类结果选择最佳匹配
    ax3 = fig.add_subplot(133, projection="3d")
    visualize_octahedron_simple(observed, ax3, color="blue", alpha=0.2)

    if classification == "Class I":
        visualize_octahedron_simple(class_I, ax3, color="red", alpha=0.2)
        ax3.scatter([], [], [], c="blue", label="观测八面体")
        ax3.scatter([], [], [], c="red", label="标准类别I")
        ax3.set_title(f"最佳匹配: 类别I\nRMSD: {rmsd_I:.4f}")
    elif classification == "Class II":
        visualize_octahedron_simple(class_II, ax3, color="green", alpha=0.2)
        ax3.scatter([], [], [], c="blue", label="观测八面体")
        ax3.scatter([], [], [], c="green", label="标准类别II")
        ax3.set_title(f"最佳匹配: 类别II\nRMSD: {rmsd_II:.4f}")
    else:
        visualize_octahedron_simple(class_I, ax3, color="red", alpha=0.1)
        visualize_octahedron_simple(class_II, ax3, color="green", alpha=0.1)
        ax3.scatter([], [], [], c="blue", label="观测八面体")
        ax3.scatter([], [], [], c="red", label="标准类别I")
        ax3.scatter([], [], [], c="green", label="标准类别II")
        ax3.set_title("分类结果: 未知类别")

    ax3.legend()
    ax3.set_xlim(-1.5, 1.5)
    ax3.set_ylim(-1.5, 1.5)
    ax3.set_zlim(-1.5, 1.5)

    # 设置总标题
    plt.suptitle(f"{title_prefix} 八面体分类结果: {classification}")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    print("可视化函数模块 (简化版)")
