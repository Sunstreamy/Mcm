# -*- coding: utf-8 -*-
"""
问题一: 平面五边形归类判别
基于多尺度几何特征与Kabsch刚性配准算法
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from mpl_toolkits.mplot3d import Axes3D  # 用于3D图形
import os
import time
from tqdm import tqdm
import seaborn as sns

# 设置matplotlib的字体和显示参数
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 使用SimHei字体
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 14  # 增大默认字体大小
plt.rcParams["figure.figsize"] = [12, 8]  # 增大默认图形大小
plt.rcParams["lines.linewidth"] = 2.5  # 增大线条宽度
plt.rcParams["lines.markersize"] = 10  # 增大标记大小
plt.rcParams["axes.labelsize"] = 16  # 增大坐标轴标签字体大小
plt.rcParams["axes.titlesize"] = 18  # 增大标题字体大小
plt.rcParams["xtick.labelsize"] = 14  # 增大x轴刻度标签字体大小
plt.rcParams["ytick.labelsize"] = 14  # 增大y轴刻度标签字体大小
plt.rcParams["legend.fontsize"] = 14  # 增大图例字体大小
plt.rcParams["grid.alpha"] = 0.3  # 设置网格透明度

# === 全局输出目录常量 ===
FIGURES_DIR_Q1 = os.path.join("..", "figures")

# === 定义二维负样本数据 - 用于参数优化 ===
# 正三角形（3个顶点）
negative_sample_triangle = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3) / 2]])

# 正方形（4个顶点）
negative_sample_square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])

# 正六边形（6个顶点）
negative_sample_hexagon = np.array(
    [[np.cos(2 * np.pi * i / 6), np.sin(2 * np.pi * i / 6)] for i in range(6)]
)

# 不规则四边形
negative_sample_quad = np.array([[0, 0], [1.2, 0.3], [0.8, 1.5], [0.1, 0.9]])

# 不规则五边形（与标准类别不同的形状）
negative_sample_pentagon = np.array(
    [[0, 0], [1, 0.2], [0.7, 1.1], [0.3, 0.9], [-0.2, 0.4]]
)

# 随机生成的5顶点形状
np.random.seed(42)  # 设置随机种子以确保可重复性
negative_sample_random = np.random.uniform(-1, 1, (5, 2))

# 将所有负样本整合到一个列表中
negative_samples_2d_list = [
    (negative_sample_triangle, "正三角形"),
    (negative_sample_square, "正方形"),
    (negative_sample_hexagon, "正六边形"),
    (negative_sample_quad, "不规则四边形"),
    (negative_sample_pentagon, "不规则五边形"),
    (negative_sample_random, "随机五顶点形状"),
]


class PentagonClassifier:
    """平面五边形归类判别模型"""

    def __init__(
        self, lambda_threshold=0.15, alpha_weight=0.7, reference_shapes_for_scaling=None
    ):
        """
        初始化五边形分类器

        参数:
            lambda_threshold: 分类阈值，低于此值被认为匹配某一类别
            alpha_weight: 配准误差在综合评分中的权重
            reference_shapes_for_scaling: 用于特征标准化的参考形状列表，每个元素是一个形状的点集
        """
        self.lambda_threshold = lambda_threshold
        self.alpha_weight = alpha_weight

        # 特征标准化相关参数
        self.feature_means = None
        self.feature_stds = None
        self.use_feature_normalization = False

        # 计算并设置标准化参数
        self._calculate_and_set_normalization_params(reference_shapes_for_scaling)

    def _calculate_and_set_normalization_params(self, reference_shapes):
        """
        基于参考形状集计算特征标准化参数

        参数:
            reference_shapes: 参考形状列表，每个元素是一个形状的点集
        """
        # 初始化确保即使reference_shapes为空，这些属性也有定义
        self.feature_means = None
        self.feature_stds = None
        self.use_feature_normalization = False

        if reference_shapes is None or not reference_shapes:
            print("警告: 未提供用于标准化的参考形状，将不启用特征标准化。")
            return

        all_reference_features_raw = []
        for shape_points in reference_shapes:
            # 对形状进行归一化处理
            normalized_shape_points = self.normalize_pentagon(shape_points)
            # 确保 extract_features 返回的是原始特征
            raw_features = self.extract_features(normalized_shape_points)
            if raw_features is not None and len(raw_features) > 0:  # 确保特征提取成功
                all_reference_features_raw.append(raw_features)

        if not all_reference_features_raw:
            print("警告: 未能从参考形状中提取任何有效特征，将不启用特征标准化。")
            return

        reference_features_matrix = np.array(all_reference_features_raw)

        # 如果只有一个样本，确保是二维的 (n_samples, n_features)
        if reference_features_matrix.ndim == 1:
            # 假设特征提取总是返回相同长度的向量
            if reference_features_matrix.size > 0:
                reference_features_matrix = reference_features_matrix.reshape(1, -1)
            else:  # 如果是空的，也无法计算统计量
                print("警告: 提取的参考特征为空，将不启用特征标准化。")
                return

        if reference_features_matrix.shape[0] > 0:
            self.feature_means = np.mean(reference_features_matrix, axis=0)
            self.feature_stds = np.std(reference_features_matrix, axis=0)
            # 避免除以零或极小值
            self.feature_stds[self.feature_stds < 1e-10] = 1.0
            self.use_feature_normalization = True
            print(
                f"特征标准化参数已基于{reference_features_matrix.shape[0]}个参考形状计算并设置。"
            )
        else:
            print("警告: 参考特征集为空，将不启用特征标准化。")

    def normalize_pentagon(self, points):
        """
        归一化五边形（平移到质心，缩放到单位大小）

        参数:
            points: 五边形顶点坐标, shape=(5,2)

        返回:
            归一化后的顶点坐标
        """
        # 确保输入为NumPy数组
        points = np.array(points)

        # 计算质心
        centroid = np.mean(points, axis=0)

        # 去中心化
        centered = points - centroid

        # 计算最大边长
        max_dist = 0
        for i in range(len(centered)):
            j = (i + 1) % len(centered)  # 循环连接
            dist = np.linalg.norm(centered[i] - centered[j])
            max_dist = max(max_dist, dist)

        # 归一化
        if max_dist > 0:
            normalized = centered / max_dist
        else:
            normalized = centered

        return normalized

    def extract_features(self, pentagon_normalized_points):
        """
        提取五边形的几何特征

        参数:
            pentagon_normalized_points: 归一化后的五边形顶点坐标, shape=(5,2)

        返回:
            特征向量 - 未经过标准化的原始特征
        """
        # 确保输入为NumPy数组
        pentagon_normalized_points = np.array(pentagon_normalized_points)

        # 1. 计算边长序列
        edges = []
        for i in range(5):
            j = (i + 1) % 5
            edge_length = np.linalg.norm(
                pentagon_normalized_points[i] - pentagon_normalized_points[j]
            )
            edges.append(edge_length)
        edges = sorted(edges)

        # 2. 计算内角序列
        angles = []
        for i in range(5):
            prev = (i - 1) % 5
            next = (i + 1) % 5

            v1 = pentagon_normalized_points[prev] - pentagon_normalized_points[i]
            v2 = pentagon_normalized_points[next] - pentagon_normalized_points[i]

            # 归一化向量
            v1_norm = np.linalg.norm(v1)
            v2_norm = np.linalg.norm(v2)

            if v1_norm > 0 and v2_norm > 0:
                v1 = v1 / v1_norm
                v2 = v2 / v2_norm

                # 计算夹角余弦值
                cos_angle = np.clip(np.dot(v1, v2), -1.0, 1.0)

                # 计算内角（弧度）
                angle = np.arccos(cos_angle)
                angles.append(angle)
            else:
                angles.append(0)  # 处理退化情况

        angles = sorted(angles)

        # 3. 计算顶点到质心的距离
        centroid = np.mean(pentagon_normalized_points, axis=0)
        distances = [np.linalg.norm(p - centroid) for p in pentagon_normalized_points]
        distances = sorted(distances)

        # 4. 计算面积
        area = self.calculate_area(pentagon_normalized_points)

        # 组合所有特征
        features = np.concatenate([edges, angles, distances, [area]])

        # 注意：此处不对特征进行任何标准化，返回原始特征向量
        # 特征标准化将在分类器层面基于参考集进行

        return features

    def calculate_area(self, points):
        """
        计算多边形面积（使用叉积）

        参数:
            points: 多边形顶点坐标

        返回:
            面积值
        """
        area = 0
        n = len(points)

        for i in range(n):
            j = (i + 1) % n
            area += points[i][0] * points[j][1]
            area -= points[j][0] * points[i][1]

        return abs(area) / 2

    def try_all_rotations(self, source_points, target_points):
        """
        尝试不同的旋转和起始顶点排序，找到最佳的配准结果

        参数:
            source_points: 源点集，shape=(5,2)
            target_points: 目标点集，shape=(5,2)

        返回:
            best_R: 最佳旋转矩阵
            best_t: 最佳平移向量
            min_rmsd: 最小RMSD
            best_aligned: 最佳对齐后的点集
        """
        min_rmsd = float("inf")
        best_R = None
        best_t = None
        best_aligned = None

        # 尝试5种不同的起始顶点（旋转排序）
        for i in range(5):
            # 旋转顶点顺序
            rotated_points = np.roll(source_points, i, axis=0)

            # 正向排序
            R, t, rmsd, aligned = self.kabsch_algorithm(rotated_points, target_points)

            if rmsd < min_rmsd:
                min_rmsd = rmsd
                best_R = R
                best_t = t
                best_aligned = aligned

            # 反向排序（镜像）
            reversed_points = rotated_points[::-1]
            R, t, rmsd, aligned = self.kabsch_algorithm(reversed_points, target_points)

            if rmsd < min_rmsd:
                min_rmsd = rmsd
                best_R = R
                best_t = t
                best_aligned = aligned

        return best_R, best_t, min_rmsd, best_aligned

    def kabsch_algorithm(self, P, Q):
        """
        使用Kabsch算法计算最优旋转矩阵和RMSD

        参数:
            P, Q: 两组点集, shape=(n,2)

        返回:
            R: 旋转矩阵
            t: 平移向量
            rmsd: 均方根偏差
            P_aligned: 变换后的点集P
        """
        # 确保输入为NumPy数组
        P = np.array(P)
        Q = np.array(Q)

        # 确保点的数量相同
        if len(P) != len(Q):
            raise ValueError("点集P和Q必须具有相同数量的点")

        # 计算质心
        p_centroid = np.mean(P, axis=0)
        q_centroid = np.mean(Q, axis=0)

        # 中心化点集
        P_centered = P - p_centroid
        Q_centered = Q - q_centroid

        # 计算协方差矩阵
        H = P_centered.T @ Q_centered

        # SVD分解
        U, S, Vt = np.linalg.svd(H)

        # 计算旋转矩阵
        R = Vt.T @ U.T

        # 检查是否需要反射校正（确保是正旋转矩阵）
        if np.linalg.det(R) < 0:
            Vt[-1] = -Vt[-1]
            R = Vt.T @ U.T

        # 计算平移向量
        t = q_centroid - R @ p_centroid

        # 应用变换
        P_aligned = (R @ P.T).T + t

        # 计算RMSD
        rmsd = np.sqrt(np.mean(np.sum((P_aligned - Q) ** 2, axis=1)))

        return R, t, rmsd, P_aligned

    def classify(self, pentagon, class_I, class_II, verbose=False):
        """
        对五边形进行归类判别

        参数:
            pentagon: 待测五边形顶点坐标
            class_I: 标准类别I五边形顶点坐标
            class_II: 标准类别II五边形顶点坐标
            verbose: 是否打印详细信息

        返回:
            分类结果和评分
        """
        # 检查顶点数量
        if len(pentagon) != 5:
            if verbose:
                print(f"\n===== 五边形分类结果 =====")
                print(
                    f"输入图形顶点数量为 {len(pentagon)}，非五边形。直接判定为 Unknown 类别。"
                )
                print("===========================\n")

            # 创建表示"非常不匹配"的评分字典
            scores = {
                "Class I": {
                    "RMSD": float("inf"),
                    "Feature Distance": float("inf"),
                    "Combined Score": float("inf"),
                    "Aligned Points": pentagon.copy(),  # 使用原始点集
                },
                "Class II": {
                    "RMSD": float("inf"),
                    "Feature Distance": float("inf"),
                    "Combined Score": float("inf"),
                    "Aligned Points": pentagon.copy(),  # 使用原始点集
                },
                "Classification": "Unknown",
            }

            return "Unknown", scores

        # 归一化处理
        pentagon_norm = self.normalize_pentagon(pentagon)
        class_I_norm = self.normalize_pentagon(class_I)
        class_II_norm = self.normalize_pentagon(class_II)

        # 特征提取
        feat_pentagon = self.extract_features(pentagon_norm)
        feat_class_I = self.extract_features(class_I_norm)
        feat_class_II = self.extract_features(class_II_norm)

        # 特征标准化（如果启用）
        norm_feat_pentagon = self.normalize_features(feat_pentagon)
        norm_feat_class_I = self.normalize_features(feat_class_I)
        norm_feat_class_II = self.normalize_features(feat_class_II)

        # 计算特征相似度（欧氏距离）- 使用标准化后的特征
        d_feat_I = np.linalg.norm(norm_feat_pentagon - norm_feat_class_I)
        d_feat_II = np.linalg.norm(norm_feat_pentagon - norm_feat_class_II)

        # 考虑所有可能的旋转进行Kabsch配准
        _, _, rmsd_I, aligned_I = self.try_all_rotations(pentagon_norm, class_I_norm)
        _, _, rmsd_II, aligned_II = self.try_all_rotations(pentagon_norm, class_II_norm)

        # 计算综合评分
        score_I = self.alpha_weight * rmsd_I + (1 - self.alpha_weight) * d_feat_I
        score_II = self.alpha_weight * rmsd_II + (1 - self.alpha_weight) * d_feat_II

        # 归类判别
        if score_I < score_II and score_I < self.lambda_threshold:
            category = "Class I"
        elif score_II < score_I and score_II < self.lambda_threshold:
            category = "Class II"
        else:
            category = "Unknown"

        # 结果整理
        scores = {
            "Class I": {
                "RMSD": rmsd_I,
                "Feature Distance": d_feat_I,
                "Combined Score": score_I,
                "Aligned Points": aligned_I,
            },
            "Class II": {
                "RMSD": rmsd_II,
                "Feature Distance": d_feat_II,
                "Combined Score": score_II,
                "Aligned Points": aligned_II,
            },
            "Classification": category,
        }

        if verbose:
            self.print_classification_result(scores)

        return category, scores

    def print_classification_result(self, scores):
        """打印分类结果详情"""
        print("\n===== 五边形分类结果 =====")
        print(
            f"类别I评分: RMSD={scores['Class I']['RMSD']:.4f}, 特征距离={scores['Class I']['Feature Distance']:.4f}, 综合评分={scores['Class I']['Combined Score']:.4f}"
        )
        print(
            f"类别II评分: RMSD={scores['Class II']['RMSD']:.4f}, 特征距离={scores['Class II']['Feature Distance']:.4f}, 综合评分={scores['Class II']['Combined Score']:.4f}"
        )
        print(f"分类结果: {scores['Classification']}")
        print("===========================\n")

    def visualize_classification(
        self, pentagon, class_I, class_II, scores, save_path=None
    ):
        """
        可视化分类结果并保存到文件

        参数:
            pentagon: 待测五边形顶点坐标
            class_I: 标准类别I五边形顶点坐标
            class_II: 标准类别II五边形顶点坐标
            scores: 分类评分和结果
            save_path: 保存路径，如果为None则显示而不保存
        """
        # 归一化处理
        pentagon_norm = self.normalize_pentagon(pentagon)
        class_I_norm = self.normalize_pentagon(class_I)
        class_II_norm = self.normalize_pentagon(class_II)

        # 获取最佳配准后的点集
        aligned_I = scores["Class I"]["Aligned Points"]
        aligned_II = scores["Class II"]["Aligned Points"]

        # 创建图形
        plt.rcParams.update({"font.size": 16})  # 增大默认字体大小
        fig, ax = plt.subplots(1, 3, figsize=(18, 6))

        # 绘制待测五边形
        poly_test = Polygon(pentagon_norm, fill=False, edgecolor="blue", linewidth=2.5)
        ax[0].add_patch(poly_test)
        ax[0].scatter(pentagon_norm[:, 0], pentagon_norm[:, 1], color="blue", s=80)
        ax[0].set_title("待测五边形（归一化）")

        # 添加圆圈图例
        ax[0].scatter([], [], marker="o", color="blue", s=75, label="待测五边形")
        ax[0].legend(
            loc="upper right",
            fontsize=12,
            facecolor="white",
            edgecolor="gray",
            framealpha=0.9,
            borderpad=1,
            handletextpad=1,
        )

        # 绘制待测五边形与类别I比较
        poly_test1 = Polygon(aligned_I, fill=False, edgecolor="blue", linewidth=2.5)
        poly_class1 = Polygon(class_I_norm, fill=False, edgecolor="red", linewidth=2.5)
        ax[1].add_patch(poly_test1)
        ax[1].add_patch(poly_class1)
        ax[1].scatter(aligned_I[:, 0], aligned_I[:, 1], color="blue", s=80)
        ax[1].scatter(class_I_norm[:, 0], class_I_norm[:, 1], color="red", s=80)
        ax[1].set_title(f"与类别I比较")

        # 添加图例
        ax[1].scatter([], [], marker="o", color="red", s=75, label="标准类别I")
        ax[1].scatter([], [], marker="o", color="blue", s=75, label="待测五边形")
        ax[1].legend(
            loc="upper right",
            fontsize=12,
            facecolor="white",
            edgecolor="gray",
            framealpha=0.9,
            borderpad=1,
            handletextpad=1,
        )

        # 添加RMSD和评分文本
        score_text = f'RMSD={scores["Class I"]["RMSD"] if scores["Class I"]["RMSD"] != float("inf") else "未计算":.4f}\n评分={scores["Class I"]["Combined Score"] if scores["Class I"]["Combined Score"] != float("inf") else "未计算":.4f}'
        ax[1].text(
            0.05,
            0.95,
            score_text,
            transform=ax[1].transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        # 绘制待测五边形与类别II比较
        poly_test2 = Polygon(aligned_II, fill=False, edgecolor="blue", linewidth=2.5)
        poly_class2 = Polygon(
            class_II_norm, fill=False, edgecolor="green", linewidth=2.5
        )
        ax[2].add_patch(poly_test2)
        ax[2].add_patch(poly_class2)
        ax[2].scatter(aligned_II[:, 0], aligned_II[:, 1], color="blue", s=80)
        ax[2].scatter(class_II_norm[:, 0], class_II_norm[:, 1], color="green", s=80)
        ax[2].set_title(f"与类别II比较")

        # 添加图例
        ax[2].scatter([], [], marker="o", color="green", s=80, label="标准类别II")
        ax[2].scatter([], [], marker="o", color="blue", s=80, label="待测五边形")
        ax[2].legend(
            loc="upper right",
            fontsize=12,
            facecolor="white",
            edgecolor="gray",
            framealpha=0.9,
            borderpad=1,
            handletextpad=1,
        )

        # 添加RMSD和评分文本
        score_text = f'RMSD={scores["Class II"]["RMSD"] if scores["Class II"]["RMSD"] != float("inf") else "未计算":.4f}\n评分={scores["Class II"]["Combined Score"] if scores["Class II"]["Combined Score"] != float("inf") else "未计算":.4f}'
        ax[2].text(
            0.05,
            0.95,
            score_text,
            transform=ax[2].transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        # 设置轴范围和标签
        for a in ax:
            a.set_xlim(-1.5, 1.5)
            a.set_ylim(-1.5, 1.5)
            a.set_aspect("equal")
            a.grid(True)
            a.axhline(y=0, color="k", linestyle="-", alpha=0.3)
            a.axvline(x=0, color="k", linestyle="-", alpha=0.3)
            a.tick_params(axis="both", which="major", labelsize=16)  # 增大刻度字体


        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close(fig)  # 关闭图形而不是显示
        else:
            plt.show()

    def normalize_features(self, features):
        """
        对特征向量进行Z-score标准化

        参数:
            features: 原始特征向量

        返回:
            标准化后的特征向量
        """
        if (
            not self.use_feature_normalization
            or self.feature_means is None
            or self.feature_stds is None
        ):
            return features

        # Z-score标准化: (x - mean) / std
        normalized_features = (features - self.feature_means) / self.feature_stds

        return normalized_features


def find_optimal_parameters():
    """
    优化五边形分类器的alpha_p和lambda_p参数

    返回:
    best_params_dict: 包含最佳参数和评估结果的字典
    """
    print("=== 开始二维五边形分类器参数优化 ===")
    start_time = time.time()

    # 标准五边形样本
    class_I = np.array(
        [[0.74, 1.90], [1.26, 3.49], [3.54, 3.58], [3.13, 2.28], [4.45, 1.16]]
    )
    class_II = np.array(
        [[0.71, 2.75], [1.65, 4.08], [2.58, 3.09], [1.31, 2.66], [1.13, 1.75]]
    )

    # 创建用于特征标准化的参考形状集
    reference_shapes = [class_I, class_II]  # 先加入标准类别形状

    # 添加一些负样本作为参考
    for neg_sample, _ in negative_samples_2d_list:
        if len(neg_sample) == 5:  # 只添加顶点数为5的负样本
            reference_shapes.append(neg_sample)

    # 定义参数搜索范围
    alpha_values = np.linspace(0, 1, 20)
    lambda_values = np.linspace(0, 1, 20)

    print(f"参数搜索范围:")
    print(
        f"  alpha_p: {alpha_values[0]:.2f} 到 {alpha_values[-1]:.2f}，共{len(alpha_values)}个值"
    )
    print(
        f"  lambda_p: {lambda_values[0]:.2f} 到 {lambda_values[-1]:.2f}，共{len(lambda_values)}个值"
    )
    print(f"  总参数组合数: {len(alpha_values) * len(lambda_values)}")

    # 定义评估分数的权重参数
    weights = {
        "w_c1_correct": 18.0,  # 正确识别类别I的奖励
        "w_c2_correct": 18.0,  # 正确识别类别II的奖励
        "w_neg_reject": 8.0,  # 正确拒绝负样本的奖励
        "w_c1_to_c2_penalty": -12.0,  # 将类别I误判为类别II的惩罚
        "w_c2_to_c1_penalty": -12.0,  # 将类别II误判为类别I的惩罚
        "w_c1_unknown_penalty": -8.0,  # 将类别I误判为Unknown的惩罚
        "w_c2_unknown_penalty": -8.0,  # 将类别II误判为Unknown的惩罚
        "w_neg_to_c1_penalty": -15.0,  # 将负样本误判为类别I的惩罚
        "w_neg_to_c2_penalty": -15.0,  # 将负样本误判为类别II的惩罚
    }

    # 新的权重字典 weights_q1，风格和量级与问题二保持一致
    weights_q1 = {
        # 奖励项 - 较大的正数
        "w_c1_correct": 50.0,  # 正确识别类别I的奖励
        "w_c2_correct": 50.0,  # 正确识别类别II的奖励
        "w_neg_reject": 25.0,  # 正确拒绝负样本的奖励
        # 惩罚项 - 负数，严重程度不同
        "w_c1_to_c2_penalty": -35.0,  # 将类别I误判为类别II的惩罚（较重）
        "w_c2_to_c1_penalty": -35.0,  # 将类别II误判为类别I的惩罚（较重）
        "w_c1_unknown_penalty": -20.0,  # 将类别I误判为Unknown的惩罚（中等）
        "w_c2_unknown_penalty": -20.0,  # 将类别II误判为Unknown的惩罚（中等）
        "w_neg_to_c1_penalty": -40.0,  # 将负样本误判为类别I的惩罚（最重）
        "w_neg_to_c2_penalty": -40.0,  # 将负样本误判为类别II的惩罚（最重）
    }

    # 使用新的权重字典进行评分
    weights = weights_q1

    # 初始化最佳参数追踪
    best_score = float("-inf")
    best_params = {"alpha": None, "lambda": None}
    all_results = []

    # 定义扰动参数用于评估模型鲁棒性
    noise_levels_param_opt = [0.05, 0.1, 0.15]
    deformation_levels_param_opt = [0.05, 0.1, 0.15]
    num_trials_param_opt = 5  # 每个扰动水平下生成的测试样本数量

    # 预先计算固定的测试样本总数
    actual_c1_test_samples = (
        1  # 原始标准类别I样本
        + len(noise_levels_param_opt) * num_trials_param_opt  # 噪声干扰的类别I样本
        + len(deformation_levels_param_opt)
        * num_trials_param_opt  # 形变干扰的类别I样本
    )

    actual_c2_test_samples = (
        1  # 原始标准类别II样本
        + len(noise_levels_param_opt) * num_trials_param_opt  # 噪声干扰的类别II样本
        + len(deformation_levels_param_opt)
        * num_trials_param_opt  # 形变干扰的类别II样本
    )

    actual_neg_test_samples = len(negative_samples_2d_list)  # 负样本总数

    print(f"扰动参数:")
    print(f"  噪声水平: {noise_levels_param_opt}")
    print(f"  形变程度: {deformation_levels_param_opt}")
    print(f"  每种扰动的试验次数: {num_trials_param_opt}")

    # 计算测试样本总数
    total_positive_samples = actual_c1_test_samples + actual_c2_test_samples
    total_negative_samples = actual_neg_test_samples
    total_samples = total_positive_samples + total_negative_samples

    print(f"测试样本统计:")
    print(f"  类别I测试样本总数: {actual_c1_test_samples}")
    print(f"  类别II测试样本总数: {actual_c2_test_samples}")
    print(f"  负样本总数: {actual_neg_test_samples}")
    print(f"  正样本总数: {total_positive_samples}")
    print(f"  总样本数: {total_samples}")
    print("\n开始参数网格搜索...\n")

    # 设置随机种子以确保可重复性
    np.random.seed(42)

    # 网格搜索
    total_combinations = len(alpha_values) * len(lambda_values)
    combination_count = 0

    for alpha in alpha_values:
        for lambda_val in lambda_values:
            combination_count += 1
            progress = combination_count / total_combinations * 100
            print(
                f"评估参数组合 {combination_count}/{total_combinations} "
                f"(alpha={alpha:.2f}, lambda={lambda_val:.2f}) - "
                f"进度: {progress:.1f}%",
                end="\r",
            )

            # 初始化分类器 - 使用参考形状集进行特征标准化
            classifier = PentagonClassifier(
                lambda_threshold=lambda_val,
                alpha_weight=alpha,
                reference_shapes_for_scaling=reference_shapes,
            )

            # 初始化计数器
            counts = {
                "c1_correct": 0,  # 类别I正确识别
                "c1_as_c2": 0,  # 类别I识别为类别II
                "c1_as_unknown": 0,  # 类别I识别为未知
                "c2_correct": 0,  # 类别II正确识别
                "c2_as_c1": 0,  # 类别II识别为类别I
                "c2_as_unknown": 0,  # 类别II识别为未知
                "neg_as_c1": 0,  # 负样本识别为类别I
                "neg_as_c2": 0,  # 负样本识别为类别II
                "neg_correct": 0,  # 负样本正确拒绝
            }

            # 测试原始标准样本
            class_i_result, _ = classifier.classify(class_I, class_I, class_II)
            class_ii_result, _ = classifier.classify(class_II, class_I, class_II)

            # 更新计数
            if class_i_result == "Class I":
                counts["c1_correct"] += 1
            elif class_i_result == "Class II":
                counts["c1_as_c2"] += 1
            else:  # Unknown
                counts["c1_as_unknown"] += 1

            if class_ii_result == "Class II":
                counts["c2_correct"] += 1
            elif class_ii_result == "Class I":
                counts["c2_as_c1"] += 1
            else:  # Unknown
                counts["c2_as_unknown"] += 1

            # 测试带噪声的样本
            for noise_level in noise_levels_param_opt:
                for _ in range(num_trials_param_opt):
                    # 给类别I添加噪声
                    noisy_class_i = class_I.copy()
                    for i in range(len(noisy_class_i)):
                        noisy_class_i[i] += np.random.normal(0, noise_level, 2)

                    # 给类别II添加噪声
                    noisy_class_ii = class_II.copy()
                    for i in range(len(noisy_class_ii)):
                        noisy_class_ii[i] += np.random.normal(0, noise_level, 2)

                    # 分类并更新计数
                    result_i, _ = classifier.classify(noisy_class_i, class_I, class_II)
                    if result_i == "Class I":
                        counts["c1_correct"] += 1
                    elif result_i == "Class II":
                        counts["c1_as_c2"] += 1
                    else:  # Unknown
                        counts["c1_as_unknown"] += 1

                    result_ii, _ = classifier.classify(
                        noisy_class_ii, class_I, class_II
                    )
                    if result_ii == "Class II":
                        counts["c2_correct"] += 1
                    elif result_ii == "Class I":
                        counts["c2_as_c1"] += 1
                    else:  # Unknown
                        counts["c2_as_unknown"] += 1

            # 测试形变样本
            for deform_level in deformation_levels_param_opt:
                for _ in range(num_trials_param_opt):
                    # 随机形变类别I
                    deformed_class_i = class_I.copy()
                    for i in range(len(deformed_class_i)):
                        deformed_class_i[i] += np.random.uniform(
                            -deform_level, deform_level, 2
                        )

                    # 随机形变类别II
                    deformed_class_ii = class_II.copy()
                    for i in range(len(deformed_class_ii)):
                        deformed_class_ii[i] += np.random.uniform(
                            -deform_level, deform_level, 2
                        )

                    # 分类并更新计数
                    result_i, _ = classifier.classify(
                        deformed_class_i, class_I, class_II
                    )
                    if result_i == "Class I":
                        counts["c1_correct"] += 1
                    elif result_i == "Class II":
                        counts["c1_as_c2"] += 1
                    else:  # Unknown
                        counts["c1_as_unknown"] += 1

                    result_ii, _ = classifier.classify(
                        deformed_class_ii, class_I, class_II
                    )
                    if result_ii == "Class II":
                        counts["c2_correct"] += 1
                    elif result_ii == "Class I":
                        counts["c2_as_c1"] += 1
                    else:  # Unknown
                        counts["c2_as_unknown"] += 1

            # 测试负样本
            for neg_sample, _ in negative_samples_2d_list:
                result, _ = classifier.classify(neg_sample, class_I, class_II)
                if result == "Class I":
                    counts["neg_as_c1"] += 1
                elif result == "Class II":
                    counts["neg_as_c2"] += 1
                else:  # Unknown
                    counts["neg_correct"] += 1

            # 使用预先计算的固定总样本数计算准确率和错误率
            # 类别I的各项比率
            c1_acc_rate = (
                counts["c1_correct"] / actual_c1_test_samples
                if actual_c1_test_samples > 0
                else 0
            )
            c1_to_c2_error_rate = (
                counts["c1_as_c2"] / actual_c1_test_samples
                if actual_c1_test_samples > 0
                else 0
            )
            c1_to_unk_error_rate = (
                counts["c1_as_unknown"] / actual_c1_test_samples
                if actual_c1_test_samples > 0
                else 0
            )

            # 类别II的各项比率
            c2_acc_rate = (
                counts["c2_correct"] / actual_c2_test_samples
                if actual_c2_test_samples > 0
                else 0
            )
            c2_to_c1_error_rate = (
                counts["c2_as_c1"] / actual_c2_test_samples
                if actual_c2_test_samples > 0
                else 0
            )
            c2_to_unk_error_rate = (
                counts["c2_as_unknown"] / actual_c2_test_samples
                if actual_c2_test_samples > 0
                else 0
            )

            # 负样本的各项比率
            neg_acc_rate = (
                counts["neg_correct"] / actual_neg_test_samples
                if actual_neg_test_samples > 0
                else 0
            )
            neg_to_c1_error_rate = (
                counts["neg_as_c1"] / actual_neg_test_samples
                if actual_neg_test_samples > 0
                else 0
            )
            neg_to_c2_error_rate = (
                counts["neg_as_c2"] / actual_neg_test_samples
                if actual_neg_test_samples > 0
                else 0
            )

            # 使用权重计算分数
            score = (
                weights["w_c1_correct"] * c1_acc_rate
                + weights["w_c2_correct"] * c2_acc_rate
                + weights["w_neg_reject"] * neg_acc_rate
                + weights["w_c1_to_c2_penalty"] * c1_to_c2_error_rate
                + weights["w_c2_to_c1_penalty"] * c2_to_c1_error_rate
                + weights["w_c1_unknown_penalty"] * c1_to_unk_error_rate
                + weights["w_c2_unknown_penalty"] * c2_to_unk_error_rate
                + weights["w_neg_to_c1_penalty"] * neg_to_c1_error_rate
                + weights["w_neg_to_c2_penalty"] * neg_to_c2_error_rate
            )

            # 记录结果
            result_info = {
                "alpha": alpha,
                "lambda": lambda_val,
                "score": score,
                "c1_acc_rate": c1_acc_rate,
                "c2_acc_rate": c2_acc_rate,
                "neg_acc_rate": neg_acc_rate,
                "c1_to_c2_error_rate": c1_to_c2_error_rate,
                "c1_to_unk_error_rate": c1_to_unk_error_rate,
                "c2_to_c1_error_rate": c2_to_c1_error_rate,
                "c2_to_unk_error_rate": c2_to_unk_error_rate,
                "neg_to_c1_error_rate": neg_to_c1_error_rate,
                "neg_to_c2_error_rate": neg_to_c2_error_rate,
            }
            all_results.append(result_info)

            # 更新最佳参数
            if score > best_score:
                best_score = score
                best_params = {"alpha": alpha, "lambda": lambda_val}

    # 清除进度行
    print(" " * 100, end="\r")

    elapsed_time = time.time() - start_time
    print(f"\n参数优化完成，耗时: {elapsed_time:.2f}秒")
    print(
        f"最佳参数: alpha={best_params['alpha']:.2f}, lambda={best_params['lambda']:.2f}"
    )
    print(f"最高评分: {best_score:.4f}")

    # 返回最佳参数和所有结果
    best_params_dict = {
        "alpha": best_params["alpha"],
        "lambda": best_params["lambda"],
        "best_score": best_score,
        "results": all_results,
    }

    return best_params_dict


def plot_parameter_optimization_results(
    all_results_list, best_params_info, save_dir=None
):
    """
    可视化参数优化结果（综合评分、类别I准确率、类别II准确率、负样本拒绝率热力图），并保存到指定目录。

    参数:
        all_results_list: 所有评估结果的列表，每个元素包含alpha、lambda和得分信息
        best_params_info: 最佳参数信息字典，包含'alpha'和'lambda'
        save_dir: 保存目录，默认为None（不保存，只显示）
    """
    if save_dir is None:
        save_dir = os.path.join(FIGURES_DIR_Q1, "params")
        os.makedirs(save_dir, exist_ok=True)

    # 从结果列表中提取数据
    alphas = [result["alpha"] for result in all_results_list]
    lambdas = [result["lambda"] for result in all_results_list]
    scores = [result["score"] for result in all_results_list]

    # 找到最佳参数
    best_alpha = best_params_info["alpha"]
    best_lambda = best_params_info["lambda"]

    # 计算唯一的alpha和lambda值
    unique_alphas = sorted(list(set(alphas)))
    unique_lambdas = sorted(list(set(lambdas)))

    # 创建评分和准确率矩阵
    score_matrix = np.zeros((len(unique_alphas), len(unique_lambdas)))
    c1_acc_matrix = np.zeros((len(unique_alphas), len(unique_lambdas)))
    c2_acc_matrix = np.zeros((len(unique_alphas), len(unique_lambdas)))
    neg_acc_matrix = np.zeros((len(unique_alphas), len(unique_lambdas)))

    # 填充矩阵
    for result in all_results_list:
        alpha_idx = unique_alphas.index(result["alpha"])
        lambda_idx = unique_lambdas.index(result["lambda"])
        score_matrix[alpha_idx, lambda_idx] = result["score"]
        c1_acc_matrix[alpha_idx, lambda_idx] = result["c1_acc_rate"] * 100  # 转为百分比
        c2_acc_matrix[alpha_idx, lambda_idx] = result["c2_acc_rate"] * 100
        neg_acc_matrix[alpha_idx, lambda_idx] = result["neg_acc_rate"] * 100

    # 找到最佳点在矩阵中的索引
    best_alpha_idx = unique_alphas.index(best_alpha)
    best_lambda_idx = unique_lambdas.index(best_lambda)
    best_score = score_matrix[best_alpha_idx, best_lambda_idx]

    # 更新字体大小设置
    plt.rcParams.update({"font.size": 16})

    # === 1. 创建热力图（2x2布局） ===
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))

    # 综合评分热力图
    ax1 = axes[0, 0]
    sns.heatmap(
        score_matrix,
        annot=False,
        cmap="viridis",
        xticklabels=[f"{x:.2f}" for x in unique_lambdas],
        yticklabels=[f"{x:.2f}" for x in unique_alphas],
        ax=ax1,
    )
    # 设置标题
    ax1.set_title("综合评分", fontsize=25)
    ax1.set_xlabel("$\\lambda$", fontsize=25)
    ax1.set_ylabel("$\\alpha$", fontsize=25)
    ax1.tick_params(axis="both", which="major", labelsize=18)

    # 类别I准确率热力图
    ax2 = axes[0, 1]
    sns.heatmap(
        c1_acc_matrix,
        annot=False,
        cmap="Blues",
        xticklabels=[f"{x:.2f}" for x in unique_lambdas],
        yticklabels=[f"{x:.2f}" for x in unique_alphas],
        ax=ax2,
    )
    # 设置标题
    ax2.set_title("类别I准确率（%）", fontsize=25)
    ax2.set_xlabel("$\\lambda$", fontsize=25)
    ax2.set_ylabel("$\\alpha$", fontsize=25)
    ax2.tick_params(axis="both", which="major", labelsize=18)

    # 类别II准确率热力图
    ax3 = axes[1, 0]
    sns.heatmap(
        c2_acc_matrix,
        annot=False,
        cmap="Greens",
        xticklabels=[f"{x:.2f}" for x in unique_lambdas],
        yticklabels=[f"{x:.2f}" for x in unique_alphas],
        ax=ax3,
    )
    # 设置标题
    ax3.set_title("类别II准确率（%）", fontsize=25)
    ax3.set_xlabel("$\\lambda$", fontsize=25)
    ax3.set_ylabel("$\\alpha$", fontsize=25)
    ax3.tick_params(axis="both", which="major", labelsize=18)

    # 负样本拒绝率热力图
    ax4 = axes[1, 1]
    sns.heatmap(
        neg_acc_matrix,
        annot=False,
        cmap="Reds",
        xticklabels=[f"{x:.2f}" for x in unique_lambdas],
        yticklabels=[f"{x:.2f}" for x in unique_alphas],
        ax=ax4,
    )
    # 设置标题
    ax4.set_title("负样本拒绝率（%）", fontsize=25)
    ax4.set_xlabel("$\\lambda$", fontsize=25)
    ax4.set_ylabel("$\\alpha$", fontsize=25)
    ax4.tick_params(axis="both", which="major", labelsize=18)

    # 在每个热力图上标记最佳参数点
    for ax in [ax1, ax2, ax3, ax4]:
        ax.plot(
            best_lambda_idx + 0.5,  # +0.5是为了定位到格子中央
            best_alpha_idx + 0.5,
            "o",
            color="red",
            markersize=14,
            markerfacecolor="none",
            markeredgewidth=3,
        )
        # 添加总标题
    plt.suptitle(
        f"最佳参数: $\\alpha$={best_alpha:.2f}, $\\lambda$={best_lambda:.2f}, 评分={best_score:.2f}",
        fontsize=30,
    )
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)

    # 保存热力图
    save_path = os.path.join(save_dir, "pentagon_parameter_optimization_heatmaps.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    # === 2. 创建3D曲面图 ===
    try:
        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib import cm

        # 准备网格数据
        lambda_grid, alpha_grid = np.meshgrid(unique_lambdas, unique_alphas)

        fig3d = plt.figure(figsize=(12, 10))
        ax3d = fig3d.add_subplot(111, projection="3d")

        # 绘制3D曲面
        surf = ax3d.plot_surface(
            lambda_grid,
            alpha_grid,
            score_matrix,
            cmap=cm.viridis,
            alpha=0.85,
            antialiased=True,
        )

        # 轴标签 - 更大字体
        ax3d.set_xlabel("$\\lambda$", fontsize=20, labelpad=15)
        ax3d.set_ylabel("$\\alpha$", fontsize=20, labelpad=15)
        ax3d.set_zlabel("综合评分", fontsize=20, labelpad=15, rotation=90)
        # 移动Z轴标签到右侧，避免遮挡坐标轴
        ax3d.zaxis.set_rotate_label(False)  # 禁用自动旋转
        ax3d.zaxis.labelpad = 10  # 增加标签与轴的距离
        # 移除标题
        ax3d.tick_params(axis="both", which="major", labelsize=14)

        # 调整视角使Z轴标签更清晰
        ax3d.view_init(elev=20, azim=-60)

        # 添加颜色条
        cbar = fig3d.colorbar(surf, shrink=0.6, aspect=10, pad=0.1)
        cbar.set_label("综合评分", fontsize=16)
        cbar.ax.tick_params(labelsize=14)

        # 标记最佳点
        zmin = np.min(score_matrix)
        # 曲面上的红色星号
        ax3d.scatter(
            best_lambda,
            best_alpha,
            best_score,
            color="red",
            marker="*",
            s=200,
            depthshade=True,
            label="最佳参数",
        )
        # 底部投影红色叉号
        ax3d.scatter(
            best_lambda,
            best_alpha,
            zmin,
            color="red",
            marker="x",
            s=100,
            depthshade=True,
        )
        # 红色虚线连接
        ax3d.plot(
            [best_lambda, best_lambda],
            [best_alpha, best_alpha],
            [zmin, best_score],
            "r--",
            alpha=0.7,
            linewidth=3,
        )

        # 图例
        ax3d.legend(loc="upper right", fontsize=16)

        # 设置视角
        ax3d.view_init(elev=20, azim=-65)

        # 信息框
        info_text = (
            f"最佳参数:\n"
            f"λ = {best_lambda:.2f}\n"
            f"α = {best_alpha:.2f}\n"
            f"综合评分: {best_score:.2f}"
        )
        ax3d.text2D(
            0.02,
            0.98,
            info_text,
            transform=ax3d.transAxes,
            fontsize=12,
            verticalalignment="top",
            bbox=dict(
                boxstyle="round", facecolor="white", alpha=0.85, edgecolor="gray"
            ),
        )

        plt.tight_layout()

        # 保存3D曲面图
        save_path_3d = os.path.join(
            save_dir, "pentagon_parameter_optimization_3d.png"
        )
        plt.savefig(save_path_3d, dpi=300, bbox_inches="tight")
        plt.close(fig3d)

        print(f"参数优化可视化结果已保存至:")
        print(f"  热力图: {save_path}")
        print(f"  3D曲面图: {save_path_3d}")

    except Exception as e:
        print(f"3D曲面图绘制失败: {e}")
        print(f"热力图已保存至: {save_path}")

    return save_path


def comprehensive_robustness_analysis(classifier, class_I, class_II, save_dir):
    """
    全面的鲁棒性分析：测试分类器在不同噪声和形变条件下的性能

    参数:
        classifier: 已优化的PentagonClassifier实例
        class_I: 标准类别I五边形顶点坐标
        class_II: 标准类别II五边形顶点坐标
        save_dir: 图像保存目录
    """
    print("开始全面鲁棒性分析...")

    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)

    # 定义噪声水平
    noise_levels = np.linspace(0, 0.25, 11)  # 0到0.25之间的11个噪声水平

    # 定义形变程度
    deform_levels = np.linspace(0, 0.25, 11)  # 0到0.25之间的11个形变水平

    # 每个级别进行多次测试以获得统计显著性
    trials_per_level = 50

    # === 1. 噪声敏感性分析 ===
    print("进行噪声敏感性分析...")

    # 创建存储结果的字典
    noise_results = {
        "Class I": {"correct": [], "as_class_ii": [], "as_unknown": []},
        "Class II": {"correct": [], "as_class_ii": [], "as_unknown": []},
        "Negative": {"as_class_i": [], "as_class_ii": [], "correct_reject": []},
    }

    # 对每个噪声级别进行测试
    for noise_level in tqdm(noise_levels, desc="噪声水平"):
        # 对Class I样本进行噪声测试
        class1_counts = {"Class I": 0, "Class II": 0, "Unknown": 0}

        for _ in range(trials_per_level):
            # 添加高斯噪声
            noisy_pentagon = class_I.copy()
            for i in range(5):
                noisy_pentagon[i] += np.random.normal(0, noise_level, 2)

            # 分类
            category, _ = classifier.classify(noisy_pentagon, class_I, class_II)

            # 记录结果
            class1_counts[category] += 1

        # 计算每个类别的比例
        total = trials_per_level
        noise_results["Class I"]["correct"].append(
            class1_counts["Class I"] / total * 100
        )
        noise_results["Class I"]["as_class_ii"].append(
            class1_counts["Class II"] / total * 100
        )
        noise_results["Class I"]["as_unknown"].append(
            class1_counts["Unknown"] / total * 100
        )

        # 对Class II样本进行噪声测试
        class2_counts = {"Class I": 0, "Class II": 0, "Unknown": 0}

        for _ in range(trials_per_level):
            # 添加高斯噪声
            noisy_pentagon = class_II.copy()
            for i in range(5):
                noisy_pentagon[i] += np.random.normal(0, noise_level, 2)

            # 分类
            category, _ = classifier.classify(noisy_pentagon, class_I, class_II)

            # 记录结果
            class2_counts[category] += 1

        # 计算每个类别的比例
        noise_results["Class II"]["correct"].append(
            class2_counts["Class II"] / total * 100
        )
        noise_results["Class II"]["as_class_ii"].append(
            class2_counts["Class I"] / total * 100
        )
        noise_results["Class II"]["as_unknown"].append(
            class2_counts["Unknown"] / total * 100
        )

        # 对负样本进行噪声测试
        neg_counts = {"Class I": 0, "Class II": 0, "Unknown": 0}

        # 使用负样本列表中的第一个样本（如正三角形）
        neg_sample = negative_samples_2d_list[0][0]

        for _ in range(trials_per_level):
            # 添加高斯噪声
            noisy_pentagon = neg_sample.copy()
            for i in range(len(noisy_pentagon)):
                noisy_pentagon[i] += np.random.normal(0, noise_level, 2)

            # 分类
            category, _ = classifier.classify(noisy_pentagon, class_I, class_II)

            # 记录结果
            neg_counts[category] += 1

        # 计算每个类别的比例
        noise_results["Negative"]["as_class_i"].append(
            neg_counts["Class I"] / total * 100
        )
        noise_results["Negative"]["as_class_ii"].append(
            neg_counts["Class II"] / total * 100
        )
        noise_results["Negative"]["correct_reject"].append(
            neg_counts["Unknown"] / total * 100
        )

    # === 2. 形变敏感性分析 ===
    print("进行形变敏感性分析...")

    # 创建存储结果的字典
    deform_results = {
        "Class I": {"correct": [], "as_class_ii": [], "as_unknown": []},
        "Class II": {"correct": [], "as_class_ii": [], "as_unknown": []},
    }

    # 对每个形变级别进行测试
    for deform_level in tqdm(deform_levels, desc="形变水平"):
        # 对Class I样本进行形变测试
        class1_counts = {"Class I": 0, "Class II": 0, "Unknown": 0}

        for _ in range(trials_per_level):
            # 应用随机形变（移动每个顶点的位置）
            deformed_pentagon = class_I.copy()

            # 计算五边形的边长平均值作为缩放因子
            edge_lengths = []
            for j in range(5):
                j_next = (j + 1) % 5
                edge = np.linalg.norm(class_I[j] - class_I[j_next])
                edge_lengths.append(edge)

            # 使用边长平均值的一定比例作为形变幅度
            avg_edge = np.mean(edge_lengths)
            deform_magnitude = avg_edge * deform_level

            # 随机形变每个顶点
            for j in range(5):
                # 在x和y方向上应用随机位移
                deformed_pentagon[j] += np.random.uniform(
                    -deform_magnitude, deform_magnitude, 2
                )

            # 对形变后的五边形进行归类
            category, _ = classifier.classify(deformed_pentagon, class_I, class_II)

            # 记录结果
            class1_counts[category] += 1

        # 计算每个类别的比例
        total = trials_per_level
        deform_results["Class I"]["correct"].append(
            class1_counts["Class I"] / total * 100
        )
        deform_results["Class I"]["as_class_ii"].append(
            class1_counts["Class II"] / total * 100
        )
        deform_results["Class I"]["as_unknown"].append(
            class1_counts["Unknown"] / total * 100
        )

        # 对Class II样本进行形变测试
        class2_counts = {"Class I": 0, "Class II": 0, "Unknown": 0}

        for _ in range(trials_per_level):
            # 应用随机形变
            deformed_pentagon = class_II.copy()

            # 计算五边形的边长平均值作为缩放因子
            edge_lengths = []
            for j in range(5):
                j_next = (j + 1) % 5
                edge = np.linalg.norm(class_II[j] - class_II[j_next])
                edge_lengths.append(edge)

            # 使用边长平均值的一定比例作为形变幅度
            avg_edge = np.mean(edge_lengths)
            deform_magnitude = avg_edge * deform_level

            # 随机形变每个顶点
            for j in range(5):
                deformed_pentagon[j] += np.random.uniform(
                    -deform_magnitude, deform_magnitude, 2
                )

            # 分类
            category, _ = classifier.classify(deformed_pentagon, class_I, class_II)

            # 记录结果
            class2_counts[category] += 1

        # 计算每个类别的比例
        deform_results["Class II"]["correct"].append(
            class2_counts["Class II"] / total * 100
        )
        deform_results["Class II"]["as_class_ii"].append(
            class2_counts["Class I"] / total * 100
        )
        deform_results["Class II"]["as_unknown"].append(
            class2_counts["Unknown"] / total * 100
        )

    # === 3. 结果可视化 ===
    print("生成可视化结果...")

    # 计算平均性能
    y_avg_noise = (
        np.array(noise_results["Class I"]["correct"])
        + np.array(noise_results["Class II"]["correct"])
        + np.array(noise_results["Negative"]["correct_reject"])
    ) / 3

    y_avg_deform = (
        np.array(deform_results["Class I"]["correct"])
        + np.array(deform_results["Class II"]["correct"])
    ) / 2

    # 负样本在无形变时的正确拒绝率
    neg_baseline = noise_results["Negative"]["correct_reject"][0]

    # === 1. 单独的噪声敏感性分析图 ===
    plt.rcParams.update({"font.size": 16})  # 增大默认字体大小
    plt.figure(figsize=(12, 8))

    # 类别I正确识别率
    plt.plot(
        noise_levels,
        noise_results["Class I"]["correct"],
        color="#1f77b4",
        marker="o",
        markersize=10,
        linestyle="-",
        linewidth=3.0,
    )

    # 类别II正确识别率
    plt.plot(
        noise_levels,
        noise_results["Class II"]["correct"],
        color="#2ca02c",
        marker="s",
        markersize=10,
        linestyle="-",
        linewidth=3.0,
    )

    # 负样本正确拒绝率
    plt.plot(
        noise_levels,
        noise_results["Negative"]["correct_reject"],
        color="#d62728",
        marker="^",
        markersize=10,
        linestyle="-",
        linewidth=3.0,
    )

    # 平均性能
    plt.plot(
        noise_levels,
        y_avg_noise,
        marker="x",
        markersize=10,
        linestyle="--",
        color="gray",
        linewidth=2.5,
    )

    # 添加网格和标签
    plt.grid(True, alpha=0.3)
    plt.xlabel("噪声标准差 (σ)", fontsize=18)
    plt.ylabel("正确判断率 (%)", fontsize=18)

    # 明确设置坐标轴范围，确保最大值为100%
    plt.ylim(0, 105)

    # 设置刻度间隔
    plt.yticks([0, 20, 40, 60, 80, 100])
    plt.xticks([0, 0.05, 0.10, 0.15, 0.20, 0.25])

    # 修改坐标轴样式
    ax = plt.gca()
    # 设置边框
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)

    # 设置刻度字体大小
    plt.tick_params(axis="both", which="major", labelsize=16)

    # 调整X轴范围，使左边界向左移动一点，避免原点处的刻度标签重叠
    current_xlim = ax.get_xlim()
    new_left_limit = -0.005  # 设置一个小的负值
    ax.set_xlim(left=new_left_limit, right=current_xlim[1])

    # 保存图片
    noise_fig_path = os.path.join(save_dir, "pentagon_robustness_noise.png")
    plt.savefig(noise_fig_path, dpi=300, bbox_inches="tight")
    plt.close()

    # === 2. 单独的形变敏感性分析图 ===
    plt.figure(figsize=(12, 8))

    # 类别I正确识别率
    plt.plot(
        deform_levels,
        deform_results["Class I"]["correct"],
        color="#1f77b4",
        marker="o",
        markersize=10,
        linestyle="-",
        linewidth=3.0,
    )

    # 类别II正确识别率
    plt.plot(
        deform_levels,
        deform_results["Class II"]["correct"],
        color="#2ca02c",
        marker="s",
        markersize=10,
        linestyle="-",
        linewidth=3.0,
    )

    # 平均性能
    plt.plot(
        deform_levels,
        y_avg_deform,
        marker="x",
        markersize=10,
        linestyle="--",
        color="gray",
        linewidth=2.5,
    )

    # 添加网格和标签
    plt.grid(True, alpha=0.3)
    plt.xlabel("形变程度 (δ)", fontsize=18)
    plt.ylabel("正确判断率 (%)", fontsize=18)

    # 明确设置坐标轴范围，确保最大值为100%
    plt.ylim(0, 105)

    # 设置刻度间隔
    plt.yticks([0, 20, 40, 60, 80, 100])
    plt.xticks([0, 0.05, 0.10, 0.15, 0.20, 0.25])

    # 修改坐标轴样式
    ax = plt.gca()
    # 设置边框
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)

    # 设置刻度字体大小
    plt.tick_params(axis="both", which="major", labelsize=16)

    # 添加图例
    plt.legend(loc="best", fontsize=14)

    # 调整X轴范围，使左边界向左移动一点，避免原点处的刻度标签重叠
    current_xlim = ax.get_xlim()
    new_left_limit = -0.005  # 设置一个小的负值
    ax.set_xlim(left=new_left_limit, right=current_xlim[1])

    # 保存图片
    deform_fig_path = os.path.join(
        save_dir, "pentagon_robustness_deformation_vertex.png"
    )
    plt.savefig(deform_fig_path, dpi=300, bbox_inches="tight")
    plt.close()

    # === 3. 平均性能对比图 ===
    plt.figure(figsize=(12, 8))

    # 噪声干扰下的平均性能
    plt.plot(
        noise_levels,
        y_avg_noise,
        color="#1f77b4",
        marker="o",
        markersize=10,
        linestyle="-",
        linewidth=3.0,
    )

    # 形变干扰下的平均性能
    plt.plot(
        deform_levels,
        y_avg_deform,
        color="#d62728",
        marker="s",
        markersize=10,
        linestyle="-",
        linewidth=3.0,
    )

    # 添加网格和标签
    plt.grid(True, alpha=0.3)
    plt.xlabel("干扰强度", fontsize=18)
    plt.ylabel("平均正确率 (%)", fontsize=18)

    # 明确设置坐标轴范围，确保最大值为100%
    plt.ylim(0, 105)

    # 设置刻度间隔
    plt.yticks([0, 20, 40, 60, 80, 100])
    plt.xticks([0, 0.05, 0.10, 0.15, 0.20, 0.25])

    # 修改坐标轴样式
    ax = plt.gca()
    # 设置边框
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)

    # 设置刻度字体大小
    plt.tick_params(axis="both", which="major", labelsize=16)

    # 调整X轴范围，使左边界向左移动一点，避免原点处的刻度标签重叠
    current_xlim = ax.get_xlim()
    new_left_limit = -0.005  # 设置一个小的负值
    ax.set_xlim(left=new_left_limit, right=current_xlim[1])

    # 保存图片
    compare_fig_path = os.path.join(
        save_dir, "pentagon_average_robustness_comparison.png"
    )
    plt.savefig(compare_fig_path, dpi=300, bbox_inches="tight")
    plt.close()

    # 打印单独类别的详细性能
    print("\n各类别在低干扰条件下的详细性能:")
    print(
        f"  Class I 在噪声 σ={noise_levels[1]:.2f} 下的正确识别率: {noise_results['Class I']['correct'][1]:.2f}%"
    )
    print(
        f"  Class II 在噪声 σ={noise_levels[1]:.2f} 下的正确识别率: {noise_results['Class II']['correct'][1]:.2f}%"
    )
    print(
        f"  负样本在噪声 σ={noise_levels[1]:.2f} 下的正确拒绝率: {noise_results['Negative']['correct_reject'][1]:.2f}%"
    )

    print(
        f"  Class I 在形变水平 {deform_levels[1]:.2f} 下的正确识别率: {deform_results['Class I']['correct'][1]:.2f}%"
    )
    print(
        f"  Class II 在形变水平 {deform_levels[1]:.2f} 下的正确识别率: {deform_results['Class II']['correct'][1]:.2f}%"
    )

    print("\n鲁棒性分析图像已保存至:")
    print(f"  噪声敏感性分析: {noise_fig_path}")
    print(f"  形变敏感性分析: {deform_fig_path}")
    print(f"  平均鲁棒性对比: {compare_fig_path}")


def main():
    """主函数，按照顺序执行五边形分类的全部分析流程"""
    print("\n===== 开始五边形分类分析 =====")

    # === 加载标准数据 ===
    # 标准五边形样本
    class_I = np.array(
        [[0.74, 1.90], [1.26, 3.49], [3.54, 3.58], [3.13, 2.28], [4.45, 1.16]]
    )
    class_II = np.array(
        [[0.71, 2.75], [1.65, 4.08], [2.58, 3.09], [1.31, 2.66], [1.13, 1.75]]
    )

    # 创建用于特征标准化的参考形状集
    reference_shapes = [class_I, class_II]  # 先加入标准类别形状

    # 添加一些负样本作为参考
    for neg_sample, _ in negative_samples_2d_list:
        if len(neg_sample) == 5:  # 只添加顶点数为5的负样本
            reference_shapes.append(neg_sample)

    # === 参数优化部分 ===
    RUN_PARAMETER_OPTIMIZATION = True  # 如需优化设为True
    OPTIMIZED_ALPHA_P = 0.90  # 预设优化参数
    OPTIMIZED_LAMBDA_P = 0.40
    print(
        f"初始/预设优化参数: alpha_p={OPTIMIZED_ALPHA_P}, lambda_p={OPTIMIZED_LAMBDA_P}"
    )

    if RUN_PARAMETER_OPTIMIZATION:
        print("\n=== 开始参数优化 ===")
        best_params = find_optimal_parameters()
        OPTIMIZED_ALPHA_P = best_params["alpha"]
        OPTIMIZED_LAMBDA_P = best_params["lambda"]
        # 参数优化结果可视化
        plot_parameter_optimization_results(
            best_params["results"],
            {"alpha": OPTIMIZED_ALPHA_P, "lambda": OPTIMIZED_LAMBDA_P},
        )
        print(
            f"参数优化完成，最佳参数: alpha_p={OPTIMIZED_ALPHA_P:.4f}, lambda_p={OPTIMIZED_LAMBDA_P:.4f}"
        )
    else:
        print("\n使用预设优化参数进行后续分析。")

    # === 实例化最终分类器 ===
    final_classifier = PentagonClassifier(
        lambda_threshold=OPTIMIZED_LAMBDA_P,
        alpha_weight=OPTIMIZED_ALPHA_P,
        reference_shapes_for_scaling=reference_shapes,
    )

    # === 对表二观测五边形进行分类与可视化 ===
    test_pentagons_list = [
        (
            np.array(
                [[5.69, 3.94], [5.62, 4.31], [6.07, 4.61], [6.13, 4.29], [6.53, 4.21]]
            ),
            "pentagon_1",
        ),
        (
            np.array(
                [[1.14, 2.07], [2.55, 3.24], [5.07, 2.04], [5.03, 4.78], [1.88, 4.11]]
            ),
            "pentagon_2",
        ),
        (
            np.array(
                [[1.80, 2.76], [4.08, 3.04], [2.26, 3.25], [1.96, 4.54], [1.19, 3.61]]
            ),
            "pentagon_3",
        ),
        (
            np.array(
                [[5.74, 2.94], [3.96, 3.38], [3.69, 4.78], [1.68, 3.49], [1.98, 1.86]]
            ),
            "pentagon_4",
        ),
        (
            np.array(
                [[1.60, 4.79], [2.46, 1.93], [5.26, 3.42], [5.31, 5.80], [4.40, 4.25]]
            ),
            "pentagon_5",
        ),
    ]

    print("\n=== 对观测五边形样本进行分类与可视化 ===")
    for pentagon, pentagon_name in test_pentagons_list:
        print(f"--- 处理 {pentagon_name} ---")
        classification, scores = final_classifier.classify(
            pentagon, class_I, class_II, verbose=True
        )
        # 保存可视化结果
        save_path = os.path.join(
            test_samples_dir_q1, f"{pentagon_name.replace(' ', '_')}.png"
        )
        final_classifier.visualize_classification(
            pentagon, class_I, class_II, scores, save_path=save_path
        )
        print(f"分类图像已保存至: {save_path}\n")

    # === 执行全面鲁棒性分析 ===
    print("\n=== 开始全面鲁棒性分析 ===")
    comprehensive_robustness_analysis(
        final_classifier, class_I, class_II, robustness_dir_q1
    )
    print("\n=== 五边形分类分析完成 ===")


if __name__ == "__main__":
    # 创建输出目录
    params_dir_q1 = os.path.join(FIGURES_DIR_Q1, "params")
    robustness_dir_q1 = os.path.join(FIGURES_DIR_Q1, "robustness")
    test_samples_dir_q1 = os.path.join(FIGURES_DIR_Q1, "test_samples")
    for directory in [
        FIGURES_DIR_Q1,
        params_dir_q1,
        robustness_dir_q1,
        test_samples_dir_q1,
    ]:
        os.makedirs(directory, exist_ok=True)

    # 运行主函数执行完整的分析流程
    main()
