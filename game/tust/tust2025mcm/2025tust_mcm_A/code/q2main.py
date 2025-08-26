#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
问题二：八面体归类判别模型核心代码 (简化版)
"""

import numpy as np
from scipy.spatial import ConvexHull
from scipy.linalg import svd
import os


class OctahedronClassifier:
    """三维八面体归类判别模型"""

    def __init__(
        self, alpha_o=0.75, lambda_o=0.45, icp_max_iterations=50, icp_tolerance=1e-6
    ):
        """
        初始化八面体分类器

        参数:
            alpha_o: 几何配准与特征相似度的权重系数
            lambda_o: 分类阈值，低于此阈值的样本才会被归为某一类别
            icp_max_iterations: ICP算法的最大迭代次数
            icp_tolerance: ICP算法的收敛阈值
        """
        self.alpha_o = alpha_o
        self.lambda_o = lambda_o
        self.icp_max_iterations = icp_max_iterations
        self.icp_tolerance = icp_tolerance

    def _normalize_points(self, points):
        """归一化点集（中心化和尺度归一化）"""
        points = np.array(points, dtype=float)
        if points.shape[0] == 0:
            return points  # 处理空输入

        centroid = np.mean(points, axis=0)
        centered_points = points - centroid

        try:
            # 尝试使用体积归一化
            hull = ConvexHull(centered_points)
            volume = hull.volume
            if volume > 1e-9:
                scale_factor = np.cbrt(volume)
            else:
                raise ValueError("体积太小")
        except Exception:
            # 使用RMS距离作为备选
            rms_dist = np.sqrt(np.mean(np.sum(centered_points**2, axis=1)))
            scale_factor = rms_dist if rms_dist > 1e-9 else 1.0

        normalized_points = (
            centered_points / scale_factor if scale_factor > 1e-9 else centered_points
        )

        return normalized_points

    def _extract_features(self, normalized_points):
        """提取八面体的几何特征"""
        # 确保输入为NumPy数组
        points = np.array(normalized_points)
        features = []

        # 1. 边长统计特征
        try:
            n_points = points.shape[0]
            if n_points >= 2:
                distances = []
                for i in range(n_points):
                    for j in range(i + 1, n_points):
                        dist = np.linalg.norm(points[i] - points[j])
                        distances.append(dist)

                distances = sorted(distances)
                features.extend(
                    [
                        np.min(distances),
                        np.max(distances),
                        np.median(distances),
                        np.mean(distances),
                        np.std(distances) if len(distances) > 1 else 0.0,
                    ]
                )
            else:
                features.extend([0.0] * 5)
        except Exception:
            features.extend([0.0] * 5)

        # 2. 顶点到原点的距离统计
        try:
            distances_to_origin = [np.linalg.norm(p) for p in points]
            if len(distances_to_origin) > 0:
                features.extend(
                    [
                        np.min(distances_to_origin),
                        np.max(distances_to_origin),
                        np.median(distances_to_origin),
                        np.mean(distances_to_origin),
                        np.std(distances_to_origin),
                    ]
                )
            else:
                features.extend([0.0] * 5)
        except Exception:
            features.extend([0.0] * 5)

        # 3. 体积与表面积相关特征
        try:
            hull = ConvexHull(points)
            volume = hull.volume if hull.volume is not None else 0.0
            area = hull.area if hull.area is not None else 0.0
            features.append(volume)

            # 计算球度
            sphericity = 0.0
            if area > 1e-9:
                sphericity = ((np.pi ** (1 / 3)) * (6 * volume) ** (2 / 3)) / area
            features.append(sphericity)
        except Exception:
            features.extend([0.0, 0.0])

        # 4. PCA 惯量特征
        try:
            if points.shape[0] >= points.shape[1]:
                centered = points - np.mean(points, axis=0)
                cov_matrix = np.cov(centered.T)
                eig_vals = sorted(np.abs(np.linalg.eigvalsh(cov_matrix)), reverse=True)

                current_eig_vals = list(eig_vals) + [0.0] * (3 - len(eig_vals))
                features.extend(current_eig_vals[:3])

                if current_eig_vals[0] > 1e-9:
                    features.append(current_eig_vals[1] / current_eig_vals[0])
                    features.append(current_eig_vals[2] / current_eig_vals[0])
                else:
                    features.extend([0.0, 0.0])
            else:
                features.extend([0.0] * 5)
        except Exception:
            features.extend([0.0] * 5)

        # 特征向量标准化
        feature_vector = np.array(features, dtype=float)
        mean_feat = np.mean(feature_vector)
        std_feat = np.std(feature_vector)

        if std_feat > 1e-9:
            return (feature_vector - mean_feat) / std_feat
        return feature_vector - mean_feat

    def _kabsch_algorithm(self, P, Q):
        """计算从点集P到点集Q的最佳旋转矩阵"""
        # 计算协方差矩阵H
        H = P.T @ Q

        # 对H进行SVD分解
        U, _, Vt = svd(H)

        # 确保旋转矩阵的行列式为正（避免反射）
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        return R

    def _iterative_closest_point(self, source_points, target_points):
        """实现迭代最近点(ICP)算法，用于点云配准"""
        # 确保输入为NumPy数组
        source = np.array(source_points, dtype=float)
        target = np.array(target_points, dtype=float)

        # 初始化变换参数
        total_R = np.eye(3)  # 单位旋转矩阵
        total_t = np.zeros(3)  # 零平移向量

        # 当前变换后的源点集
        current_source = source.copy()
        prev_error = float("inf")

        for iteration in range(self.icp_max_iterations):
            # 1. 对每个源点找到最近的目标点
            matched_target_points = np.zeros_like(current_source)
            for i, point in enumerate(current_source):
                distances = np.sqrt(np.sum((target - point) ** 2, axis=1))
                nearest_idx = np.argmin(distances)
                matched_target_points[i] = target[nearest_idx]

            # 2. 计算当前的均方根误差
            current_error = np.sqrt(
                np.mean(np.sum((current_source - matched_target_points) ** 2, axis=1))
            )

            # 3. 检查收敛条件
            if abs(prev_error - current_error) < self.icp_tolerance:
                break
            prev_error = current_error

            # 4. 计算质心
            source_centroid = np.mean(current_source, axis=0)
            target_centroid = np.mean(matched_target_points, axis=0)

            # 5. 中心化点集
            source_centered = current_source - source_centroid
            target_centered = matched_target_points - target_centroid

            # 6. 使用Kabsch算法计算最优旋转
            R = self._kabsch_algorithm(source_centered, target_centered)

            # 7. 计算平移向量
            t = target_centroid - R @ source_centroid

            # 8. 更新当前源点集
            current_source = (R @ current_source.T).T + t

            # 9. 更新累积变换
            total_R = R @ total_R
            total_t = R @ total_t + t

        # 应用最终变换得到结果
        transformed_source = (total_R @ source.T).T + total_t

        # 计算最终的RMSD
        final_rmsd = np.sqrt(
            np.mean(np.sum((transformed_source - matched_target_points) ** 2, axis=1))
        )

        return transformed_source, total_R, total_t, final_rmsd

    def classify(self, observed_vertices, std_class_I_vertices, std_class_II_vertices):
        """对观测到的八面体进行分类"""
        # 1. 预处理
        obs_norm = self._normalize_points(observed_vertices)
        std_I_norm = self._normalize_points(std_class_I_vertices)
        std_II_norm = self._normalize_points(std_class_II_vertices)

        # 2. 特征提取
        obs_features = self._extract_features(obs_norm)
        std_I_features = self._extract_features(std_I_norm)
        std_II_features = self._extract_features(std_II_norm)

        # 3. 计算特征距离
        dF_I = np.linalg.norm(obs_features - std_I_features)
        dF_II = np.linalg.norm(obs_features - std_II_features)

        # 4. ICP配准并计算RMSD
        aligned_obs_to_I, R_I, t_I, rmsd_I = self._iterative_closest_point(
            obs_norm, std_I_norm
        )
        aligned_obs_to_II, R_II, t_II, rmsd_II = self._iterative_closest_point(
            obs_norm, std_II_norm
        )

        # 5. 计算综合评分
        S_I = self.alpha_o * rmsd_I + (1 - self.alpha_o) * dF_I
        S_II = self.alpha_o * rmsd_II + (1 - self.alpha_o) * dF_II

        # 6. 分类决策
        classification_result = "Unknown"
        min_score = min(S_I, S_II)

        if min_score <= self.lambda_o:
            if S_I <= S_II:
                classification_result = "Class I"
            else:
                classification_result = "Class II"

        # 7. 准备详细结果字典
        results = {
            "classification": classification_result,
            "S_I": S_I,
            "S_II": S_II,
            "rmsd_I": rmsd_I,
            "rmsd_II": rmsd_II,
            "dF_I": dF_I,
            "dF_II": dF_II,
            "observed_normalized": obs_norm,
            "class_I_normalized": std_I_norm,
            "class_II_normalized": std_II_norm,
            "aligned_to_class_I": {
                "aligned_points": aligned_obs_to_I,
                "rotation_matrix": R_I,
                "translation_vector": t_I,
                "rmsd": rmsd_I,
            },
            "aligned_to_class_II": {
                "aligned_points": aligned_obs_to_II,
                "rotation_matrix": R_II,
                "translation_vector": t_II,
                "rmsd": rmsd_II,
            },
        }

        return classification_result, results


# 如果作为主程序运行
if __name__ == "__main__":
    print("八面体归类判别模型核心代码 (简化版)")
