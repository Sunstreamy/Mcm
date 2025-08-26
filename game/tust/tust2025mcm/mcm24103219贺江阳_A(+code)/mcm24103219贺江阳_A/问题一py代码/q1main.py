#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
问题一：五边形归类判别模型核心代码 (简化版)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import os


class PentagonClassifier:
    """平面五边形归类判别模型"""

    def __init__(self, lambda_threshold=0.4, alpha_weight=0.9):
        """
        初始化五边形分类器

        参数:
            lambda_threshold: 分类阈值，低于此阈值的样本才会被归为某一类别
            alpha_weight: 几何配准与特征相似度的权重系数
        """
        self.lambda_threshold = lambda_threshold
        self.alpha_weight = alpha_weight

    def normalize_pentagon(self, points):
        """归一化五边形（平移到质心，缩放到单位大小）"""
        # 确保输入为NumPy数组
        points = np.array(points)

        # 计算质心
        centroid = np.mean(points, axis=0)

        # 去中心化
        centered = points - centroid

        # 计算最大边长
        max_dist = 0
        for i in range(len(centered)):
            j = (i + 1) % len(centered)
            dist = np.linalg.norm(centered[i] - centered[j])
            max_dist = max(max_dist, dist)

        # 归一化
        normalized = centered / max_dist if max_dist > 0 else centered

        return normalized

    def extract_features(self, pentagon_normalized_points):
        """提取五边形的几何特征"""
        # 确保输入为NumPy数组
        points = np.array(pentagon_normalized_points)

        # 1. 计算边长序列
        edges = []
        for i in range(5):
            j = (i + 1) % 5
            edge_length = np.linalg.norm(points[i] - points[j])
            edges.append(edge_length)
        edges = sorted(edges)

        # 2. 计算内角序列
        angles = []
        for i in range(5):
            prev = (i - 1) % 5
            next = (i + 1) % 5

            v1 = points[prev] - points[i]
            v2 = points[next] - points[i]

            # 归一化向量
            v1_norm = np.linalg.norm(v1)
            v2_norm = np.linalg.norm(v2)

            if v1_norm > 0 and v2_norm > 0:
                v1 = v1 / v1_norm
                v2 = v2 / v2_norm
                cos_angle = np.clip(np.dot(v1, v2), -1.0, 1.0)
                angle = np.arccos(cos_angle)
                angles.append(angle)
            else:
                angles.append(0)
        angles = sorted(angles)

        # 3. 计算顶点到质心的距离
        centroid = np.mean(points, axis=0)
        distances = [np.linalg.norm(p - centroid) for p in points]
        distances = sorted(distances)

        # 4. 计算面积
        area = self.calculate_area(points)

        # 组合所有特征
        features = np.concatenate([edges, angles, distances, [area]])

        return features

    def calculate_area(self, points):
        """计算多边形面积（使用叉积）"""
        area = 0
        n = len(points)

        for i in range(n):
            j = (i + 1) % n
            area += points[i][0] * points[j][1]
            area -= points[j][0] * points[i][1]

        return abs(area) / 2

    def kabsch_algorithm(self, P, Q):
        """使用Kabsch算法计算最优旋转矩阵和RMSD"""
        # 确保输入为NumPy数组
        P = np.array(P)
        Q = np.array(Q)

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

        # 检查是否需要反射校正
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

    def try_all_rotations(self, source_points, target_points):
        """尝试不同的旋转和起始顶点排序，找到最佳的配准结果"""
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

    def classify(self, pentagon, class_I, class_II, verbose=False):
        """对五边形进行归类判别"""
        # 检查顶点数量
        if len(pentagon) != 5:
            if verbose:
                print(
                    f"\n输入图形顶点数量为 {len(pentagon)}，非五边形。直接判定为 Unknown 类别。"
                )

            return "Unknown", {
                "Class I": {
                    "RMSD": float("inf"),
                    "Feature Distance": float("inf"),
                    "Combined Score": float("inf"),
                    "Aligned Points": pentagon.copy(),
                },
                "Class II": {
                    "RMSD": float("inf"),
                    "Feature Distance": float("inf"),
                    "Combined Score": float("inf"),
                    "Aligned Points": pentagon.copy(),
                },
                "Classification": "Unknown",
            }

        # 归一化处理
        pentagon_norm = self.normalize_pentagon(pentagon)
        class_I_norm = self.normalize_pentagon(class_I)
        class_II_norm = self.normalize_pentagon(class_II)

        # 特征提取
        feat_pentagon = self.extract_features(pentagon_norm)
        feat_class_I = self.extract_features(class_I_norm)
        feat_class_II = self.extract_features(class_II_norm)

        # 计算特征相似度（欧氏距离）
        d_feat_I = np.linalg.norm(feat_pentagon - feat_class_I)
        d_feat_II = np.linalg.norm(feat_pentagon - feat_class_II)

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
            print("\n===== 五边形分类结果 =====")
            print(
                f"类别I评分: RMSD={rmsd_I:.4f}, 特征距离={d_feat_I:.4f}, 综合评分={score_I:.4f}"
            )
            print(
                f"类别II评分: RMSD={rmsd_II:.4f}, 特征距离={d_feat_II:.4f}, 综合评分={score_II:.4f}"
            )
            print(f"分类结果: {category}")
            print("===========================\n")

        return category, scores


# 如果作为主程序运行
if __name__ == "__main__":
    print("五边形归类判别模型核心代码 (简化版)")
