import numpy as np
from scipy.spatial import ConvexHull
from scipy.linalg import svd, eig  # 用于SVD和特征值计算
from scipy import linalg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from scipy.spatial import ConvexHull
from itertools import combinations
import time
from tqdm import tqdm
import os
import seaborn as sns

plt.rcParams["font.sans-serif"] = ["SimHei"]  # 使用SimHei字体
plt.rcParams["axes.unicode_minus"] = False

# 考虑使用专用ICP库如open3d以获得更强健的ICP实现
# import open3d as o3d # 使用Open3D的示例

# 定义三维负样本数据 - 用于参数优化
# 立方体（8个顶点）
negative_sample_cube = np.array(
    [
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 1, 1],
    ]
)

# 正四面体（4个顶点）
negative_sample_tetrahedron = np.array(
    [
        [0, 0, 0],
        [1, 0, 0],
        [0.5, np.sqrt(3) / 2, 0],
        [0.5, np.sqrt(3) / 6, np.sqrt(6) / 3],
    ]
)

# 三角棱柱（6个顶点）
negative_sample_triangular_prism = np.array(
    [
        [0, 0, 0],
        [1, 0, 0],
        [0.5, np.sqrt(3) / 2, 0],  # 底面三角形
        [0, 0, 1],
        [1, 0, 1],
        [0.5, np.sqrt(3) / 2, 1],  # 顶面三角形
    ]
)

# 五角棱柱（10个顶点）
negative_sample_pentagonal_prism = np.array(
    [[np.cos(2 * np.pi * i / 5), np.sin(2 * np.pi * i / 5), 0] for i in range(5)]
    + [[np.cos(2 * np.pi * i / 5), np.sin(2 * np.pi * i / 5), 1] for i in range(5)]
)

# 不规则6顶点形状 - 扭曲的八面体
negative_sample_irregular_6pts = np.array(
    [
        [0.8, 0.1, 0.1],  # 扭曲的顶点
        [0.1, 1.2, -0.3],
        [-1.5, 0.2, 0.4],
        [0.3, -0.9, -0.7],
        [0.2, 0.3, 1.8],
        [-0.4, -0.2, -1.3],
    ]
)

# 随机生成的8顶点云
np.random.seed(42)  # 设置随机种子以确保可重复性
negative_sample_random_8pts = np.random.uniform(-1, 1, (8, 3))

# 球形分布的点云（近似球体，12个顶点）
phi = np.pi * (3.0 - np.sqrt(5.0))  # 黄金角
negative_sample_sphere_approx = np.array(
    [
        [
            np.cos(phi * i) * np.sin(np.pi * i / 6),
            np.sin(phi * i) * np.sin(np.pi * i / 6),
            np.cos(np.pi * i / 6),
        ]
        for i in range(12)
    ]
)

# 将所有负样本整合到一个列表中
negative_samples_3d_list = [
    (negative_sample_cube, "立方体"),
    (negative_sample_tetrahedron, "正四面体"),
    (negative_sample_triangular_prism, "三角棱柱"),
    (negative_sample_pentagonal_prism, "五角棱柱"),
    (negative_sample_irregular_6pts, "不规则6顶点形状"),
    (negative_sample_random_8pts, "随机8顶点云"),
    (negative_sample_sphere_approx, "近似球体(12顶点)"),
]


def find_optimal_parameters_3d(
    std_class_I_raw, std_class_II_raw, negative_samples_3d_list, classifier_params=None
):
    """
    优化八面体分类器的alpha_o和lambda_o参数

    参数:
    std_class_I_raw: 标准类别I八面体的原始顶点数据
    std_class_II_raw: 标准类别II八面体的原始顶点数据
    negative_samples_3d_list: 负样本列表，每个元素为(点集, 名称)的元组
    classifier_params: 传递给AdvancedOctahedronClassifier构造函数的其他固定参数

    返回:
    best_params_dict: 包含最佳参数和评估结果的字典
    """
    print("=== 开始三维八面体分类器参数优化 ===")
    start_time = time.time()

    # 设置默认的分类器参数
    if classifier_params is None:
        classifier_params = {"icp_max_iterations": 50, "icp_tolerance": 1e-6}

    # 定义参数搜索范围
    alpha_o_values = np.linspace(0, 1.0, 20)  # 从0到1
    lambda_o_values = np.linspace(0, 1.0, 20)  # 从0到1

    print(f"参数搜索范围:")
    print(
        f"  alpha_o: {alpha_o_values[0]:.2f} 到 {alpha_o_values[-1]:.2f}，共{len(alpha_o_values)}个值"
    )
    print(
        f"  lambda_o: {lambda_o_values[0]:.2f} 到 {lambda_o_values[-1]:.2f}，共{len(lambda_o_values)}个值"
    )
    print(f"  总参数组合数: {len(alpha_o_values) * len(lambda_o_values)}")

    # 定义评估分数的权重参数
    weights = {
        "w_c1_correct": 18.0,  # 正确识别类别I的奖励
        "w_c2_correct": 18.0,  # 正确识别类别II的奖励
        "w_neg_reject": 8.0,  # 正确拒绝负样本的奖励
        "w_c1_to_c2_penalty": -12.0,  # 将类别I误判为类别II的惩罚
        "w_c2_to_c1_penalty": -12.0,  # 将类别II误判为类别I的惩罚
        "w_c1_unknown_penalty": -8.0,  # 将类别I误判为Unknown的惩罚
        "w_c2_unknown_penalty": -8.0,  # 将类别II误判为Unknown的惩罚
        "w_neg_to_c1_penalty": -10.0,  # 将负样本误判为类别I的惩罚
        "w_neg_to_c2_penalty": -10.0,  # 将负样本误判为类别II的惩罚
    }

    # 初始化追踪最佳参数的变量
    best_params = {"alpha_o": None, "lambda_o": None}
    best_score = float("-inf")
    all_results_list = []

    # 定义扰动参数
    noise_levels_param_opt = [0.05, 0.1, 0.15]
    deformation_types_param_opt = ["uniaxial_stretch", "random_vertex_displacement"]
    deformation_levels_param_opt = [0.05, 0.1]
    num_trials_param_opt = 3  # 每个扰动水平下生成的测试样本数量

    print(f"扰动参数:")
    print(f"  噪声水平: {noise_levels_param_opt}")
    print(f"  形变类型: {deformation_types_param_opt}")
    print(f"  形变程度: {deformation_levels_param_opt}")
    print(f"  每种扰动的试验次数: {num_trials_param_opt}")

    # 计算测试样本总数
    total_positive_samples = (
        2  # 原始标准样本
        + len(noise_levels_param_opt) * 2 * num_trials_param_opt  # 噪声样本
        + len(deformation_types_param_opt)
        * len(deformation_levels_param_opt)
        * 2
        * num_trials_param_opt  # 形变样本
    )
    total_negative_samples = len(negative_samples_3d_list)
    total_samples = total_positive_samples + total_negative_samples

    print(f"测试样本统计:")
    print(f"  正样本总数: {total_positive_samples}")
    print(f"  负样本总数: {total_negative_samples}")
    print(f"  总样本数: {total_samples}")
    print("\n开始参数网格搜索...\n")

    # 设置随机种子以确保可重复性
    np.random.seed(42)

    # 计算类别I和类别II的总测试样本数
    c1_total_samples = (
        1
        + len(noise_levels_param_opt) * num_trials_param_opt
        + len(deformation_types_param_opt)
        * len(deformation_levels_param_opt)
        * num_trials_param_opt
    )
    c2_total_samples = (
        1
        + len(noise_levels_param_opt) * num_trials_param_opt
        + len(deformation_types_param_opt)
        * len(deformation_levels_param_opt)
        * num_trials_param_opt
    )

    # 使用tqdm创建进度条
    param_combinations = len(alpha_o_values) * len(lambda_o_values)
    progress_bar = tqdm(total=param_combinations, desc="参数优化进度")

    # 循环遍历所有参数组合
    for alpha_idx, alpha_val in enumerate(alpha_o_values):
        for lambda_idx, lambda_val in enumerate(lambda_o_values):
            # 使用当前参数实例化分类器
            classifier = AdvancedOctahedronClassifier(
                alpha_o=alpha_val, lambda_o=lambda_val, **classifier_params
            )

            # 初始化计数器
            # 类别I的计数器
            acc_C1_as_C1 = 0  # 正确识别为C1
            fpr_C1_as_C2 = 0  # 误判为C2
            fpr_C1_as_UNK = 0  # 误判为Unknown

            # 类别II的计数器
            acc_C2_as_C2 = 0  # 正确识别为C2
            fpr_C2_as_C1 = 0  # 误判为C1
            fpr_C2_as_UNK = 0  # 误判为Unknown

            # 负样本的计数器
            acc_NEG_as_UNK = 0  # 正确拒绝
            fpr_NEG_as_C1 = 0  # 误判为C1
            fpr_NEG_as_C2 = 0  # 误判为C2

            # 1. 评估类别I样本
            # 1.1 评估原始类别I样本
            classification, _ = classifier.classify(
                std_class_I_raw, std_class_I_raw, std_class_II_raw
            )
            if classification == "Class I":
                acc_C1_as_C1 += 1
            elif classification == "Class II":
                fpr_C1_as_C2 += 1
            else:  # Unknown
                fpr_C1_as_UNK += 1

            # 1.2 评估添加噪声的类别I样本
            for noise_level in noise_levels_param_opt:
                for trial in range(num_trials_param_opt):
                    # 添加噪声
                    noisy_sample = classifier.add_noise_3d(std_class_I_raw, noise_level)
                    # 分类
                    classification, _ = classifier.classify(
                        noisy_sample, std_class_I_raw, std_class_II_raw
                    )
                    # 更新计数器
                    if classification == "Class I":
                        acc_C1_as_C1 += 1
                    elif classification == "Class II":
                        fpr_C1_as_C2 += 1
                    else:  # Unknown
                        fpr_C1_as_UNK += 1

            # 1.3 评估形变后的类别I样本
            for deformation_type in deformation_types_param_opt:
                for deformation_level in deformation_levels_param_opt:
                    for trial in range(num_trials_param_opt):
                        # 应用形变
                        deformed_sample = classifier.apply_deformation_3d(
                            std_class_I_raw, deformation_type, deformation_level
                        )
                        # 分类
                        classification, _ = classifier.classify(
                            deformed_sample, std_class_I_raw, std_class_II_raw
                        )
                        # 更新计数器
                        if classification == "Class I":
                            acc_C1_as_C1 += 1
                        elif classification == "Class II":
                            fpr_C1_as_C2 += 1
                        else:  # Unknown
                            fpr_C1_as_UNK += 1

            # 2. 评估类别II样本
            # 2.1 评估原始类别II样本
            classification, _ = classifier.classify(
                std_class_II_raw, std_class_I_raw, std_class_II_raw
            )
            if classification == "Class II":
                acc_C2_as_C2 += 1
            elif classification == "Class I":
                fpr_C2_as_C1 += 1
            else:  # Unknown
                fpr_C2_as_UNK += 1

            # 2.2 评估添加噪声的类别II样本
            for noise_level in noise_levels_param_opt:
                for trial in range(num_trials_param_opt):
                    # 添加噪声
                    noisy_sample = classifier.add_noise_3d(
                        std_class_II_raw, noise_level
                    )
                    # 分类
                    classification, _ = classifier.classify(
                        noisy_sample, std_class_I_raw, std_class_II_raw
                    )
                    # 更新计数器
                    if classification == "Class II":
                        acc_C2_as_C2 += 1
                    elif classification == "Class I":
                        fpr_C2_as_C1 += 1
                    else:  # Unknown
                        fpr_C2_as_UNK += 1

            # 2.3 评估形变后的类别II样本
            for deformation_type in deformation_types_param_opt:
                for deformation_level in deformation_levels_param_opt:
                    for trial in range(num_trials_param_opt):
                        # 应用形变
                        deformed_sample = classifier.apply_deformation_3d(
                            std_class_II_raw, deformation_type, deformation_level
                        )
                        # 分类
                        classification, _ = classifier.classify(
                            deformed_sample, std_class_I_raw, std_class_II_raw
                        )
                        # 更新计数器
                        if classification == "Class II":
                            acc_C2_as_C2 += 1
                        elif classification == "Class I":
                            fpr_C2_as_C1 += 1
                        else:  # Unknown
                            fpr_C2_as_UNK += 1

            # 3. 评估负样本
            for neg_sample, neg_name in negative_samples_3d_list:
                # 3.1 评估原始负样本
                classification, _ = classifier.classify(
                    neg_sample, std_class_I_raw, std_class_II_raw
                )
                if classification == "Unknown":
                    acc_NEG_as_UNK += 1
                elif classification == "Class I":
                    fpr_NEG_as_C1 += 1
                else:  # Class II
                    fpr_NEG_as_C2 += 1

            # 计算比率
            c1_acc_rate = acc_C1_as_C1 / c1_total_samples
            c1_to_c2_error_rate = fpr_C1_as_C2 / c1_total_samples
            c1_to_unk_error_rate = fpr_C1_as_UNK / c1_total_samples

            c2_acc_rate = acc_C2_as_C2 / c2_total_samples
            c2_to_c1_error_rate = fpr_C2_as_C1 / c2_total_samples
            c2_to_unk_error_rate = fpr_C2_as_UNK / c2_total_samples

            neg_acc_rate = acc_NEG_as_UNK / total_negative_samples
            neg_to_c1_error_rate = fpr_NEG_as_C1 / total_negative_samples
            neg_to_c2_error_rate = fpr_NEG_as_C2 / total_negative_samples

            # 计算综合评分
            combined_score = (
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

            # 保存当前参数组合的评估结果
            result = {
                "alpha_o": alpha_val,
                "lambda_o": lambda_val,
                "c1_acc_rate": c1_acc_rate,
                "c1_to_c2_error_rate": c1_to_c2_error_rate,
                "c1_to_unk_error_rate": c1_to_unk_error_rate,
                "c2_acc_rate": c2_acc_rate,
                "c2_to_c1_error_rate": c2_to_c1_error_rate,
                "c2_to_unk_error_rate": c2_to_unk_error_rate,
                "neg_acc_rate": neg_acc_rate,
                "neg_to_c1_error_rate": neg_to_c1_error_rate,
                "neg_to_c2_error_rate": neg_to_c2_error_rate,
                "combined_score": combined_score,
            }
            all_results_list.append(result)

            # 更新最佳参数
            if combined_score > best_score:
                best_score = combined_score
                best_params["alpha_o"] = alpha_val
                best_params["lambda_o"] = lambda_val

            # 更新进度条
            progress_bar.update(1)

            # 可选：打印当前参数组合的评估结果
            if (alpha_idx * len(lambda_o_values) + lambda_idx + 1) % 5 == 0:
                print(
                    f"  参数组合 {alpha_idx * len(lambda_o_values) + lambda_idx + 1}/{param_combinations}: "
                    f"alpha_o={alpha_val:.2f}, lambda_o={lambda_val:.2f}, 评分={combined_score:.2f}"
                )

    # 关闭进度条
    progress_bar.close()

    # 计算优化用时
    optimization_time = time.time() - start_time

    print(f"\n参数优化完成！用时: {optimization_time:.2f}秒")
    print(
        f"最佳参数: alpha_o={best_params['alpha_o']:.4f}, lambda_o={best_params['lambda_o']:.4f}"
    )
    print(f"最佳评分: {best_score:.4f}")

    # 返回结果字典
    best_params_dict = {
        "alpha_o": best_params["alpha_o"],
        "lambda_o": best_params["lambda_o"],
        "best_score": best_score,
        "all_results": all_results_list,
        "weights": weights,
        "optimization_time": optimization_time,
    }

    return best_params_dict


class AdvancedOctahedronClassifier:
    def __init__(
        self,
        alpha_o=0.75,
        lambda_o=0.45,
        icp_max_iterations=50,
        icp_tolerance=1e-6,
        # 特征提取所需的参数（如果需要）
    ):
        self.alpha_o = alpha_o
        self.lambda_o = lambda_o
        self.icp_max_iterations = icp_max_iterations
        self.icp_tolerance = icp_tolerance
        # ...

    def _normalize_points(self, points):
        # ... (实现如上所述的中心化和基于体积/RMS距离的尺度归一化) ...
        points = np.array(points, dtype=float)
        if points.shape[0] == 0:
            return points  # 处理空输入
        centroid = np.mean(points, axis=0)
        centered_points = points - centroid

        try:
            # 尝试使用体积归一化，如果凸包计算失败则用RMS距离
            hull = ConvexHull(centered_points)  # 对中心化点集计算凸包
            volume = hull.volume
            if volume > 1e-9:  # 避免对平面形状除以零
                scale_factor = np.cbrt(volume)
            else:  # 对平面或退化形状使用备选方案
                raise ValueError("体积太小或凸包计算失败")
        except Exception:  # 如果凸包计算失败或体积为零，使用RMS距离作为备选
            if len(centered_points) > 0:
                rms_dist_to_centroid = np.sqrt(
                    np.mean(np.sum(centered_points**2, axis=1))
                )
                scale_factor = (
                    rms_dist_to_centroid if rms_dist_to_centroid > 1e-9 else 1.0
                )
            else:
                scale_factor = 1.0

        if scale_factor > 1e-9:  # 避免除以零
            normalized_points = centered_points / scale_factor
        else:
            normalized_points = centered_points  # 或作为错误处理
        return normalized_points

    def _extract_features_detailed(self, normalized_points):
        """提取更详细和鲁棒的特征集"""
        features = []
        points = normalized_points
        n_points = points.shape[0]
        if n_points < 4:  # 需要至少4个点来计算凸包和PCA
            return np.zeros(self.get_feature_vector_dim())  # 返回零向量或处理错误

        # 1. 边长统计特征 (例如，5维: 12条边的最小值、最大值、中值、均值、标准差)
        try:
            hull = ConvexHull(points)
            edge_lengths = []
            unique_edges = set()
            for simplex in hull.simplices:
                for i in range(3):
                    p1_idx, p2_idx = simplex[i], simplex[(i + 1) % 3]
                    edge = tuple(sorted((p1_idx, p2_idx)))
                    if edge not in unique_edges:
                        edge_lengths.append(
                            np.linalg.norm(points[p1_idx] - points[p2_idx])
                        )
                        unique_edges.add(edge)

            if not edge_lengths:
                raise ValueError("从凸包中未找到边")

            edge_lengths = np.array(edge_lengths)
            features.extend(
                [
                    np.min(edge_lengths),
                    np.max(edge_lengths),
                    np.median(edge_lengths),
                    np.mean(edge_lengths),
                    np.std(edge_lengths),
                ]
            )
        except Exception:
            features.extend([0.0] * 5)

        # 2. 顶点到质心距离统计特征 (例如，5维)
        distances_to_origin = np.linalg.norm(points, axis=1)
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

        # 3. 体积与表面积相关特征 (2维)
        try:
            if "hull" not in locals() or hull is None:
                hull = ConvexHull(points)
            volume = hull.volume if hull.volume is not None else 0.0
            area = hull.area if hull.area is not None else 0.0
            features.append(volume)
            sphericity = 0.0
            if area > 1e-9:  # 避免除以零
                sphericity = ((np.pi ** (1 / 3)) * (6 * volume) ** (2 / 3)) / area
            features.append(sphericity)
        except Exception:
            features.extend([0.0, 0.0])

        # 4. PCA 惯量特征 (3个特征值 + 2个比率 = 5维)
        try:
            if points.shape[0] >= points.shape[1]:  # 需要足够的点
                centered_again = points - np.mean(points, axis=0)  # 以防万一重新中心化
                cov_matrix = np.cov(centered_again.T)
                eig_vals_pca = sorted(
                    np.abs(linalg.eigvalsh(cov_matrix)), reverse=True
                )  # 对称矩阵用eigvalsh

                current_eig_vals = list(eig_vals_pca) + [0.0] * (3 - len(eig_vals_pca))
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

        # 5. 角度特征 (例如，面角均值和标准差, 二面角均值和标准差 = 4维)
        # 这在没有几何库的情况下很难健壮地实现。这里是占位符。
        # 你需要遍历hull.simplices（面），
        # 然后对每个面，找到其相邻面来计算二面角。
        # 对于面角，遍历每个面的顶点。
        try:
            if "hull" not in locals() or hull is None:
                hull = ConvexHull(points)
            # 实际角度计算的占位符
            face_angle_mean, face_angle_std = self._calculate_face_angle_stats(
                points, hull.simplices
            )
            dihedral_angle_mean, dihedral_angle_std = (
                self._calculate_dihedral_angle_stats(points, hull)
            )
            features.extend(
                [
                    face_angle_mean,
                    face_angle_std,
                    dihedral_angle_mean,
                    dihedral_angle_std,
                ]
            )
        except Exception:
            features.extend([0.0] * 4)

        # 总特征：5（边）+ 5（距离）+ 2（体积/面积）+ 5（PCA）+ 4（角度）= 21个特征
        # 根据需要调整以达到所需数量或基于有效性

        # 最终特征向量标准化（Z分数）
        feature_vector = np.array(features, dtype=float)
        mean_feat = np.mean(feature_vector)
        std_feat = np.std(feature_vector)
        if std_feat > 1e-9:
            return (feature_vector - mean_feat) / std_feat
        return feature_vector - mean_feat  # 如果标准差为零，则只进行中心化

    def _calculate_face_angle_stats(self, points, faces_simplices):
        """
        计算凸包中所有三角形面的内角统计特征

        参数:
        points: 形状为(N, 3)的点集
        faces_simplices: 从ConvexHull.simplices获取的面索引，每行是一个包含3个点索引的数组

        返回:
        face_angle_mean: 所有面角的均值
        face_angle_std: 所有面角的标准差
        """
        # 如果输入无效，返回默认值
        if len(faces_simplices) == 0 or len(points) < 3:
            return 0.0, 0.0

        all_angles = []

        for face_indices in faces_simplices:
            # 获取三角形面的三个顶点坐标
            p = [points[idx] for idx in face_indices]

            # 循环计算三个内角
            for i in range(3):
                # 从当前顶点到其他两个顶点的向量
                v1 = p[(i + 1) % 3] - p[i]
                v2 = p[(i + 2) % 3] - p[i]

                # 计算向量长度
                len_v1 = np.linalg.norm(v1)
                len_v2 = np.linalg.norm(v2)

                # 计算角度，如果是退化情况则填充0
                if len_v1 > 1e-9 and len_v2 > 1e-9:
                    dot_product = np.dot(v1, v2)
                    cos_angle = np.clip(dot_product / (len_v1 * len_v2), -1.0, 1.0)
                    angle = np.arccos(cos_angle)
                    all_angles.append(angle)
                else:
                    # 对于退化情况，添加0作为占位符
                    all_angles.append(0.0)

        # 计算所有角度的均值和标准差
        if len(all_angles) > 0:
            all_angles = np.array(all_angles)
            # 可选：过滤掉所有为0的占位符，仅计算有效角度的统计值
            valid_angles = all_angles[all_angles > 0]
            if len(valid_angles) > 0:
                face_angle_mean = np.mean(valid_angles)
                face_angle_std = np.std(valid_angles)
            else:
                face_angle_mean = 0.0
                face_angle_std = 0.0
            return face_angle_mean, face_angle_std
        else:
            return 0.0, 0.0

    def _calculate_dihedral_angle_stats(self, points, hull):
        """
        计算凸包中所有相邻面之间的二面角统计特征

        参数:
        points: 形状为(N, 3)的点集
        hull: 从scipy.spatial.ConvexHull计算得到的凸包对象

        返回:
        dihedral_angle_mean: 所有二面角的均值
        dihedral_angle_std: 所有二面角的标准差
        """
        # 1. 输入有效性检查
        if len(hull.simplices) == 0 or len(points) < 4:
            return 0.0, 0.0

        # 2. 获取并归一化面法向量
        face_normals = hull.equations[:, :3]  # 法向量是方程的前三个系数

        # 计算每个法向量的模长
        face_normals_norm = np.linalg.norm(face_normals, axis=1)

        # 创建有效法向量的掩码(模长大于阈值的法向量)
        valid_normals_mask = face_normals_norm > 1e-9

        # 如果没有有效的法向量，则返回默认值
        if not np.any(valid_normals_mask):
            return 0.0, 0.0

        # 初始化归一化后的法向量数组
        normalized_face_normals = np.zeros_like(face_normals)

        # 只对有效的法向量进行归一化
        normalized_face_normals[valid_normals_mask] = (
            face_normals[valid_normals_mask]
            / face_normals_norm[valid_normals_mask, np.newaxis]
        )

        # 3. 构建边到面的映射
        edge_to_faces = {}

        # 遍历所有面
        for face_idx, simplex_vertices in enumerate(hull.simplices):
            # 对面上的三条边
            for i in range(3):
                v_start_idx = simplex_vertices[i]
                v_end_idx = simplex_vertices[(i + 1) % 3]

                # 创建一个按顶点索引排序的边
                edge = tuple(sorted((v_start_idx, v_end_idx)))

                # 将当前面添加到与该边关联的面列表中
                if edge not in edge_to_faces:
                    edge_to_faces[edge] = []
                edge_to_faces[edge].append(face_idx)

        # 4. 计算二面角
        dihedral_angles = []

        # 遍历所有边及其相邻面
        for edge, face_indices_list in edge_to_faces.items():
            # 只处理内部边（被两个面共享的边）
            if len(face_indices_list) == 2:
                face_idx1, face_idx2 = face_indices_list

                # 检查两个面的法向量是否都有效
                if not (
                    valid_normals_mask[face_idx1] and valid_normals_mask[face_idx2]
                ):
                    continue

                # 获取两个面的法向量
                n1 = normalized_face_normals[face_idx1]
                n2 = normalized_face_normals[face_idx2]

                # 计算法向量的点积
                dot_product = np.dot(n1, n2)

                # 确保数值稳定性
                dot_product = np.clip(dot_product, -1.0, 1.0)

                # 计算二面角 (凸多面体的内部二面角)
                # 使用 arccos(-dot_product) 因为法向量指向外部
                dihedral_angle = np.arccos(-dot_product)

                # 添加到二面角列表
                dihedral_angles.append(dihedral_angle)

        # 5. 计算统计量并返回
        if len(dihedral_angles) > 0:
            dihedral_angles_arr = np.array(dihedral_angles)
            dihedral_angle_mean = np.mean(dihedral_angles_arr)
            dihedral_angle_std = np.std(dihedral_angles_arr)
            return dihedral_angle_mean, dihedral_angle_std
        else:
            return 0.0, 0.0

    def get_feature_vector_dim(self):
        # 根据_extract_features_detailed定义预期维度
        return 5 + 5 + 2 + 5 + 4  # 基于上述分解的示例

    def _kabsch_algorithm(self, P, Q):
        """
        计算从点集P到点集Q的最佳旋转矩阵

        参数:
        P, Q: 形状为(n, 3)的点集，已经中心化

        返回:
        R: 3x3旋转矩阵
        """
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
        """
        ICP算法的完整实现。

        参数:
        source_points: 待配准点集，shape为(n, 3)
        target_points: 目标点集，shape为(n, 3)

        返回:
        aligned_source: 配准后的源点集
        final_R: 从原始source_points到aligned_source的总旋转矩阵
        final_t: 从原始source_points到aligned_source的总平移向量
        final_rmsd: 最终的均方根距离
        """
        # 初始化
        transformed_source = source_points.copy()
        current_R = np.eye(3)  # 当前迭代的旋转
        current_t = np.zeros(3)  # 当前迭代的平移
        total_R = np.eye(3)  # 总旋转
        total_t = np.zeros(3)  # 总平移
        prev_rmsd = float("inf")

        # 迭代循环
        for iteration in range(self.icp_max_iterations):
            # 1. 寻找最近点对应
            matched_target_points = np.zeros_like(transformed_source)

            for i, point in enumerate(transformed_source):
                # 计算当前点到所有目标点的距离
                distances = np.linalg.norm(target_points - point, axis=1)
                # 找到距离最近的目标点索引
                closest_point_idx = np.argmin(distances)
                # 存储对应的目标点
                matched_target_points[i] = target_points[closest_point_idx]

            # 2. 计算当前迭代的变换
            # 计算质心
            centroid_transformed_source = np.mean(transformed_source, axis=0)
            centroid_matched_target = np.mean(matched_target_points, axis=0)

            # 将点集中心化
            P_centered = transformed_source - centroid_transformed_source
            Q_centered = matched_target_points - centroid_matched_target

            # 使用Kabsch算法计算旋转矩阵
            R_iter = self._kabsch_algorithm(P_centered, Q_centered)

            # 计算平移向量
            t_iter = centroid_matched_target - R_iter @ centroid_transformed_source

            # 3. 应用变换到源点集
            transformed_source = (R_iter @ transformed_source.T).T + t_iter

            # 4. 更新总变换
            # 非常重要：确保变换累积顺序正确
            total_t = R_iter @ total_t + t_iter
            total_R = R_iter @ total_R

            # 5. 计算当前RMSD
            current_rmsd = np.sqrt(
                np.mean(
                    np.sum((transformed_source - matched_target_points) ** 2, axis=1)
                )
            )

            # 6. 检查收敛
            if abs(prev_rmsd - current_rmsd) < self.icp_tolerance:
                break

            prev_rmsd = current_rmsd

        # 计算最终的RMSD
        final_rmsd = np.sqrt(
            np.mean(np.sum((transformed_source - matched_target_points) ** 2, axis=1))
        )

        return transformed_source, total_R, total_t, final_rmsd

    def classify(self, observed_vertices, std_class_I_vertices, std_class_II_vertices):
        """
        对观测到的八面体进行分类

        参数:
        observed_vertices: 观测八面体的顶点坐标
        std_class_I_vertices: 标准类别I八面体的顶点坐标
        std_class_II_vertices: 标准类别II八面体的顶点坐标

        返回:
        classification_result: 分类结果 ("Class I", "Class II" 或 "Unknown")
        results: 包含详细分类信息的字典
        """
        # 1. 预处理
        obs_norm = self._normalize_points(observed_vertices)
        std_I_norm = self._normalize_points(std_class_I_vertices)
        std_II_norm = self._normalize_points(std_class_II_vertices)

        # 2. 特征提取
        obs_features = self._extract_features_detailed(obs_norm)
        std_I_features = self._extract_features_detailed(std_I_norm)
        std_II_features = self._extract_features_detailed(std_II_norm)

        # 3. 计算特征距离
        dF_I = np.linalg.norm(obs_features - std_I_features)
        dF_II = np.linalg.norm(obs_features - std_II_features)

        # 4. ICP配准并计算RMSD
        # 正确调用ICP方法并获取返回值
        aligned_obs_to_I, R_I, t_I, rmsd_I = self._iterative_closest_point(
            obs_norm, std_I_norm
        )
        aligned_obs_to_II, R_II, t_II, rmsd_II = self._iterative_closest_point(
            obs_norm, std_II_norm
        )

        # 5. 计算综合评分，使用ICP返回的RMSD值
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

        # 7. 准备详细结果字典，包含对齐点集、变换矩阵和RMSD值
        results = {
            "classification": classification_result,
            "S_I": S_I,
            "S_II": S_II,
            "min_score": min_score,
            "rmsd_I": rmsd_I,
            "rmsd_II": rmsd_II,
            "dF_I": dF_I,
            "dF_II": dF_II,
            "obs_norm": obs_norm,
            "std_I_norm": std_I_norm,
            "std_II_norm": std_II_norm,
            # 存储ICP返回的对齐点集和变换信息
            "aligned_obs_to_I": aligned_obs_to_I,
            "R_I": R_I,  # 从观测八面体到类别I的旋转矩阵
            "t_I": t_I,  # 从观测八面体到类别I的平移向量
            "aligned_obs_to_II": aligned_obs_to_II,
            "R_II": R_II,  # 从观测八面体到类别II的旋转矩阵
            "t_II": t_II,  # 从观测八面体到类别II的平移向量
            # 特征向量，便于分析
            "obs_features": obs_features,
            "std_I_features": std_I_features,
            "std_II_features": std_II_features,
        }
        return classification_result, results

    def visualize_alignment(self, detailed_results, title_prefix="", save_path=None):
        """
        可视化八面体点集的对齐结果

        参数:
        detailed_results: classify方法返回的结果字典
        title_prefix: 图像标题前缀，例如观测八面体的名称
        save_path: 可选，若指定则将图像保存到该路径，否则显示
        """
        # 创建带有3个子图的图形
        fig = plt.figure(figsize=(20, 7))

        # 用于绘制八面体的函数
        def plot_octahedron(ax, points, color, alpha=0.2, label=None, draw_edges=True):
            # 绘制点云
            ax.scatter(
                points[:, 0],
                points[:, 1],
                points[:, 2],
                c=color,
                marker="o",
                s=100,  # 增大点的大小
                label=label,
            )

            # 计算凸包并绘制面
            try:
                hull = ConvexHull(points)
                for simplex in hull.simplices:
                    # 获取三角形面的顶点
                    vertices = points[simplex]
                    # 创建三角形面
                    poly = Poly3DCollection([vertices], alpha=alpha)
                    poly.set_facecolor(color)
                    poly.set_edgecolor("k" if draw_edges else color)
                    ax.add_collection3d(poly)
            except Exception as e:
                print(f"无法绘制凸包: {e}")

                # 回退绘图逻辑：当凸包计算失败时，绘制线框
                if (
                    points.shape[0] > 1 and points.shape[0] <= 10
                ):  # 检查点数是否适合线框绘制
                    # 生成所有顶点对的组合
                    vertex_pairs = list(combinations(range(points.shape[0]), 2))

                    # 创建线段列表
                    segments = []
                    for i, j in vertex_pairs:
                        segments.append([points[i], points[j]])

                    # 如果有线段，则创建线框
                    if segments:
                        # 创建线集合并设置样式
                        line_collection = Line3DCollection(
                            segments,
                            colors=color,
                            linewidths=2.0,  # 增大线宽
                            alpha=max(0.3, alpha * 0.5),
                        )
                        # 添加到轴对象
                        ax.add_collection3d(line_collection)
                        print(f"已绘制线框作为回退方案，共{len(segments)}条线段")

        # 设置坐标轴范围，使所有子图具有相同比例
        all_points = np.vstack(
            [
                detailed_results["obs_norm"],
                detailed_results["std_I_norm"],
                detailed_results["std_II_norm"],
                detailed_results["aligned_obs_to_I"],
                detailed_results["aligned_obs_to_II"],
            ]
        )
        min_vals = np.min(all_points, axis=0)
        max_vals = np.max(all_points, axis=0)

        # 确保坐标轴具有相同的比例
        max_range = np.max(max_vals - min_vals) * 0.6
        mid_vals = (max_vals + min_vals) / 2

        # 子图1: 观测八面体
        ax1 = fig.add_subplot(131, projection="3d")
        plot_octahedron(ax1, detailed_results["obs_norm"], "blue", label="观测八面体")
        ax1.set_title(f"观测八面体 (归一化)", fontsize=22)
        ax1.set_xlabel("X", fontsize=20)
        ax1.set_ylabel("Y", fontsize=20)
        ax1.set_zlabel("Z", fontsize=20)
        ax1.legend(fontsize=18)
        ax1.tick_params(axis="both", which="major", labelsize=18)

        # 设置相同的坐标轴范围
        ax1.set_xlim(mid_vals[0] - max_range, mid_vals[0] + max_range)
        ax1.set_ylim(mid_vals[1] - max_range, mid_vals[1] + max_range)
        ax1.set_zlim(mid_vals[2] - max_range, mid_vals[2] + max_range)

        # 子图2: 标准类别I八面体和对齐后的观测八面体
        ax2 = fig.add_subplot(132, projection="3d")
        plot_octahedron(ax2, detailed_results["std_I_norm"], "green", label="标准类别I")
        plot_octahedron(
            ax2,
            detailed_results["aligned_obs_to_I"],
            "blue",
            alpha=0.3,
            label="待测八面体",
        )
        ax2.set_title(f"对齐到类别I", fontsize=22)
        ax2.set_xlabel("X", fontsize=20)
        ax2.set_ylabel("Y", fontsize=20)
        ax2.set_zlabel("Z", fontsize=20)
        ax2.legend(fontsize=18)
        ax2.tick_params(axis="both", which="major", labelsize=18)

        # 添加RMSD和评分文本框 - 类别I
        score_text_I = (
            f'RMSD={detailed_results["rmsd_I"]:.4f}\n评分={detailed_results["S_I"]:.4f}'
        )
        # 使用 ax2.text2D 而不是 ax2.text 以便在3D图上添加2D文本
        ax2.text2D(
            0.05,
            0.95,
            score_text_I,
            transform=ax2.transAxes,
            verticalalignment="top",
            horizontalalignment="left",
            fontsize=18,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="gray"),
        )

        # 设置相同的坐标轴范围
        ax2.set_xlim(mid_vals[0] - max_range, mid_vals[0] + max_range)
        ax2.set_ylim(mid_vals[1] - max_range, mid_vals[1] + max_range)
        ax2.set_zlim(mid_vals[2] - max_range, mid_vals[2] + max_range)

        # 子图3: 标准类别II八面体和对齐后的观测八面体
        ax3 = fig.add_subplot(133, projection="3d")
        plot_octahedron(ax3, detailed_results["std_II_norm"], "red", label="标准类别II")
        plot_octahedron(
            ax3,
            detailed_results["aligned_obs_to_II"],
            "blue",
            alpha=0.3,
            label="待测八面体",
        )
        ax3.set_title(f"对齐到类别II", fontsize=22)
        ax3.set_xlabel("X", fontsize=20)
        ax3.set_ylabel("Y", fontsize=20)
        ax3.set_zlabel("Z", fontsize=20)
        ax3.legend(fontsize=18)
        ax3.tick_params(axis="both", which="major", labelsize=18)

        # 添加RMSD和评分文本框 - 类别II
        score_text_II = f'RMSD={detailed_results["rmsd_II"]:.4f}\n评分={detailed_results["S_II"]:.4f}'
        # 使用 ax3.text2D 而不是 ax3.text 以便在3D图上添加2D文本
        ax3.text2D(
            0.05,
            0.95,
            score_text_II,
            transform=ax3.transAxes,
            verticalalignment="top",
            horizontalalignment="left",
            fontsize=18,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="gray"),
        )

        # 设置相同的坐标轴范围
        ax3.set_xlim(mid_vals[0] - max_range, mid_vals[0] + max_range)
        ax3.set_ylim(mid_vals[1] - max_range, mid_vals[1] + max_range)
        ax3.set_zlim(mid_vals[2] - max_range, mid_vals[2] + max_range)

        # 添加总体标题和分类结果
        classification = detailed_results["classification"]
        min_score = detailed_results["min_score"]

        # 添加分类结果作为总标题
        if title_prefix:
            plt.suptitle(
                f"{title_prefix} - 分类结果: {classification}", fontsize=24, y=0.98
            )
        else:
            plt.suptitle(f"分类结果: {classification}", fontsize=24, y=0.98)

        # 确保子图布局合适
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        # 保存或显示图形
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()

    def add_noise_3d(self, points, noise_sigma):
        """
        为3D点集添加高斯噪声

        参数:
        points: 形状为(N, 3)的点集
        noise_sigma: 高斯噪声的标准差

        返回:
        添加了噪声的点集，形状与输入相同
        """
        # 复制原始点集，避免修改输入数据
        noisy_points = points.copy()

        # 确保输入点集不为空
        if points.shape[0] > 0:
            # 生成与点集形状相同的高斯噪声
            # 均值为0，标准差为noise_sigma
            noise = np.random.normal(0, noise_sigma, points.shape)

            # 将噪声添加到点集
            noisy_points = points + noise

        return noisy_points

    def apply_deformation_3d(
        self, points, deformation_type, deformation_level, axis=None
    ):
        """
        对3D点集应用不同类型的形变

        参数:
        points: 形状为(N, 3)的点集
        deformation_type: 形变类型，支持'uniaxial_stretch'和'random_vertex_displacement'
        deformation_level: 形变程度，根据形变类型有不同的含义
        axis: 可选参数，用于指定单轴拉伸/压缩的轴向('x', 'y', 'z')，若为None则随机选择

        返回:
        形变后的点集，形状与输入相同
        """
        # 复制原始点集并转换为浮点类型，避免修改输入数据并确保计算精度
        deformed_points = points.copy().astype(float)

        # 确保输入点集不为空
        if points.shape[0] == 0:
            return deformed_points

        # 单轴拉伸/压缩
        if deformation_type == "uniaxial_stretch":
            # 如果未指定轴向，随机选择一个轴
            if axis is None:
                axis = np.random.choice(["x", "y", "z"])

            # 根据指定的轴向应用拉伸/压缩
            if axis.lower() == "x":
                deformed_points[:, 0] = deformed_points[:, 0] * (1 + deformation_level)
            elif axis.lower() == "y":
                deformed_points[:, 1] = deformed_points[:, 1] * (1 + deformation_level)
            elif axis.lower() == "z":
                deformed_points[:, 2] = deformed_points[:, 2] * (1 + deformation_level)
            else:
                raise ValueError(f"不支持的轴向: {axis}，应为'x', 'y'或'z'")

        # 随机顶点位移
        elif deformation_type == "random_vertex_displacement":
            # 为每个顶点在每个坐标轴上生成随机位移
            # 位移范围为[-deformation_level, +deformation_level]的均匀分布
            random_displacements = np.random.uniform(
                -deformation_level, deformation_level, points.shape
            )

            # 应用随机位移
            deformed_points = deformed_points + random_displacements

        # 不支持的形变类型
        else:
            raise ValueError(f"不支持的形变类型: {deformation_type}")

        return deformed_points


def comprehensive_robustness_analysis_3d(
    classifier,  # 一个已用最优参数初始化的 AdvancedOctahedronClassifier 实例
    std_class_I_raw,
    std_class_II_raw,
    negative_samples_3d_list,  # 从 q2 模块全局导入或作为参数传入
    save_dir,  # 图像保存目录，例如 "figures_q2/robustness/"
):
    """
    对三维八面体分类器进行全面鲁棒性分析，包括噪声和形变敏感性。
    """
    print("\n=== 开始三维八面体鲁棒性分析 ===")

    # 1. 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)

    # 2. 设置随机种子，保证可重复性
    np.random.seed(42)

    # 3. 定义鲁棒性测试参数
    noise_levels_robustness = np.linspace(0, 0.25, 11)  # 0到0.25，11个等级
    deformation_types_robustness = ["uniaxial_stretch", "random_vertex_displacement"]
    deformation_levels_robustness = np.linspace(0, 0.25, 9)  # 0到0.25，9个等级
    trials_per_level_robustness = 30  # 每个扰动等级的重复试验次数

    # 4. 初始化结果存储结构
    noise_analysis_results = {
        "noise_levels": noise_levels_robustness,
        "Class I": {
            "correct_rate": np.zeros(len(noise_levels_robustness)),
            "to_class_II_rate": np.zeros(len(noise_levels_robustness)),
            "to_unknown_rate": np.zeros(len(noise_levels_robustness)),
        },
        "Class II": {
            "correct_rate": np.zeros(len(noise_levels_robustness)),
            "to_class_I_rate": np.zeros(len(noise_levels_robustness)),
            "to_unknown_rate": np.zeros(len(noise_levels_robustness)),
        },
        "Negative": {
            "reject_rate": np.zeros(len(noise_levels_robustness)),
            "to_class_I_rate": np.zeros(len(noise_levels_robustness)),
            "to_class_II_rate": np.zeros(len(noise_levels_robustness)),
        },
    }

    # 为每种形变类型分别初始化结果存储结构
    deformation_analysis_results_stretch = {
        "deformation_levels": deformation_levels_robustness,
        "Class I": {
            "correct_rate": np.zeros(len(deformation_levels_robustness)),
            "to_class_II_rate": np.zeros(len(deformation_levels_robustness)),
            "to_unknown_rate": np.zeros(len(deformation_levels_robustness)),
        },
        "Class II": {
            "correct_rate": np.zeros(len(deformation_levels_robustness)),
            "to_class_I_rate": np.zeros(len(deformation_levels_robustness)),
            "to_unknown_rate": np.zeros(len(deformation_levels_robustness)),
        },
        "Negative": {
            "reject_rate": np.zeros(len(deformation_levels_robustness)),
            "to_class_I_rate": np.zeros(len(deformation_levels_robustness)),
            "to_class_II_rate": np.zeros(len(deformation_levels_robustness)),
        },
    }

    deformation_analysis_results_random_disp = {
        "deformation_levels": deformation_levels_robustness,
        "Class I": {
            "correct_rate": np.zeros(len(deformation_levels_robustness)),
            "to_class_II_rate": np.zeros(len(deformation_levels_robustness)),
            "to_unknown_rate": np.zeros(len(deformation_levels_robustness)),
        },
        "Class II": {
            "correct_rate": np.zeros(len(deformation_levels_robustness)),
            "to_class_I_rate": np.zeros(len(deformation_levels_robustness)),
            "to_unknown_rate": np.zeros(len(deformation_levels_robustness)),
        },
        "Negative": {
            "reject_rate": np.zeros(len(deformation_levels_robustness)),
            "to_class_I_rate": np.zeros(len(deformation_levels_robustness)),
            "to_class_II_rate": np.zeros(len(deformation_levels_robustness)),
        },
    }

    # === 噪声敏感性分析 ===
    print("\n--- 开始噪声敏感性分析 ---")
    for idx_noise, noise_sigma in enumerate(noise_analysis_results["noise_levels"]):
        # a. 初始化临时计数器
        c1_correct_count = 0
        c1_to_c2_count = 0
        c1_to_unk_count = 0
        c2_correct_count = 0
        c2_to_c1_count = 0
        c2_to_unk_count = 0
        neg_reject_count = 0
        neg_to_c1_count = 0
        neg_to_c2_count = 0
        # b. 测试类别 I 样本
        for _ in tqdm(
            range(trials_per_level_robustness), desc=f"Class I, noise={noise_sigma:.3f}"
        ):
            noisy_sample = classifier.add_noise_3d(std_class_I_raw.copy(), noise_sigma)
            classification, _ = classifier.classify(
                noisy_sample, std_class_I_raw, std_class_II_raw
            )
            if classification == "Class I":
                c1_correct_count += 1
            elif classification == "Class II":
                c1_to_c2_count += 1
            else:
                c1_to_unk_count += 1
        # c. 测试类别 II 样本
        for _ in tqdm(
            range(trials_per_level_robustness),
            desc=f"Class II, noise={noise_sigma:.3f}",
        ):
            noisy_sample = classifier.add_noise_3d(std_class_II_raw.copy(), noise_sigma)
            classification, _ = classifier.classify(
                noisy_sample, std_class_I_raw, std_class_II_raw
            )
            if classification == "Class II":
                c2_correct_count += 1
            elif classification == "Class I":
                c2_to_c1_count += 1
            else:
                c2_to_unk_count += 1
        # d. 测试负样本
        total_neg_trials_for_this_noise_level = 0
        neg_trials_per_sample = max(
            1, trials_per_level_robustness // max(1, len(negative_samples_3d_list))
        )
        for neg_sample_points, neg_name in negative_samples_3d_list:
            for _ in range(neg_trials_per_sample):
                noisy_neg = classifier.add_noise_3d(
                    neg_sample_points.copy(), noise_sigma
                )
                classification, _ = classifier.classify(
                    noisy_neg, std_class_I_raw, std_class_II_raw
                )
                if classification == "Unknown":
                    neg_reject_count += 1
                elif classification == "Class I":
                    neg_to_c1_count += 1
                else:
                    neg_to_c2_count += 1
                total_neg_trials_for_this_noise_level += 1
        # e. 计算并存储比率
        noise_analysis_results["Class I"]["correct_rate"][idx_noise] = (
            c1_correct_count / trials_per_level_robustness
        )
        noise_analysis_results["Class I"]["to_class_II_rate"][idx_noise] = (
            c1_to_c2_count / trials_per_level_robustness
        )
        noise_analysis_results["Class I"]["to_unknown_rate"][idx_noise] = (
            c1_to_unk_count / trials_per_level_robustness
        )
        noise_analysis_results["Class II"]["correct_rate"][idx_noise] = (
            c2_correct_count / trials_per_level_robustness
        )
        noise_analysis_results["Class II"]["to_class_I_rate"][idx_noise] = (
            c2_to_c1_count / trials_per_level_robustness
        )
        noise_analysis_results["Class II"]["to_unknown_rate"][idx_noise] = (
            c2_to_unk_count / trials_per_level_robustness
        )
        if total_neg_trials_for_this_noise_level > 0:
            noise_analysis_results["Negative"]["reject_rate"][idx_noise] = (
                neg_reject_count / total_neg_trials_for_this_noise_level
            )
            noise_analysis_results["Negative"]["to_class_I_rate"][idx_noise] = (
                neg_to_c1_count / total_neg_trials_for_this_noise_level
            )
            noise_analysis_results["Negative"]["to_class_II_rate"][idx_noise] = (
                neg_to_c2_count / total_neg_trials_for_this_noise_level
            )
        else:
            noise_analysis_results["Negative"]["reject_rate"][idx_noise] = 0
            noise_analysis_results["Negative"]["to_class_I_rate"][idx_noise] = 0
            noise_analysis_results["Negative"]["to_class_II_rate"][idx_noise] = 0
    print("--- 噪声敏感性分析完成 ---\n")

    # === 形变敏感性分析 ===
    for deformation_type in deformation_types_robustness:
        print(f"\n--- 开始 {deformation_type} 形变敏感性分析 ---")
        # b. 选择对应的结果存储字典
        if deformation_type == "uniaxial_stretch":
            selected_deformation_results = deformation_analysis_results_stretch
        elif deformation_type == "random_vertex_displacement":
            selected_deformation_results = deformation_analysis_results_random_disp
        else:
            print(f"未知的形变类型: {deformation_type}")
            continue
        # c. 中层循环：遍历形变等级
        for idx_deform, deform_level in enumerate(
            tqdm(
                selected_deformation_results["deformation_levels"],
                desc=f"{deformation_type} 形变等级分析",
            )
        ):
            # d.i. 初始化临时计数器
            c1_correct_count = 0
            c1_to_c2_count = 0
            c1_to_unk_count = 0
            c2_correct_count = 0
            c2_to_c1_count = 0
            c2_to_unk_count = 0
            neg_reject_count = 0
            neg_to_c1_count = 0
            neg_to_c2_count = 0
            # d.ii. 测试类别 I 样本
            for _ in tqdm(
                range(trials_per_level_robustness),
                desc=f"Class I, {deformation_type}={deform_level:.3f}",
                leave=False,
            ):
                deformed_sample = classifier.apply_deformation_3d(
                    std_class_I_raw.copy(), deformation_type, deform_level
                )
                classification, _ = classifier.classify(
                    deformed_sample, std_class_I_raw, std_class_II_raw
                )
                if classification == "Class I":
                    c1_correct_count += 1
                elif classification == "Class II":
                    c1_to_c2_count += 1
                else:
                    c1_to_unk_count += 1
            # d.iii. 测试类别 II 样本
            for _ in tqdm(
                range(trials_per_level_robustness),
                desc=f"Class II, {deformation_type}={deform_level:.3f}",
                leave=False,
            ):
                deformed_sample = classifier.apply_deformation_3d(
                    std_class_II_raw.copy(), deformation_type, deform_level
                )
                classification, _ = classifier.classify(
                    deformed_sample, std_class_I_raw, std_class_II_raw
                )
                if classification == "Class II":
                    c2_correct_count += 1
                elif classification == "Class I":
                    c2_to_c1_count += 1
                else:
                    c2_to_unk_count += 1
            # d.iv. 测试负样本
            total_neg_trials_for_this_deform_level = 0
            neg_trials_per_sample = max(
                1, trials_per_level_robustness // max(1, len(negative_samples_3d_list))
            )
            for neg_sample_points, neg_name in negative_samples_3d_list:
                for _ in range(neg_trials_per_sample):
                    deformed_neg = classifier.apply_deformation_3d(
                        neg_sample_points.copy(), deformation_type, deform_level
                    )
                    classification, _ = classifier.classify(
                        deformed_neg, std_class_I_raw, std_class_II_raw
                    )
                    if classification == "Unknown":
                        neg_reject_count += 1
                    elif classification == "Class I":
                        neg_to_c1_count += 1
                    else:
                        neg_to_c2_count += 1
                    total_neg_trials_for_this_deform_level += 1
            # d.v. 计算并存储比率
            selected_deformation_results["Class I"]["correct_rate"][idx_deform] = (
                c1_correct_count / trials_per_level_robustness
            )
            selected_deformation_results["Class I"]["to_class_II_rate"][idx_deform] = (
                c1_to_c2_count / trials_per_level_robustness
            )
            selected_deformation_results["Class I"]["to_unknown_rate"][idx_deform] = (
                c1_to_unk_count / trials_per_level_robustness
            )
            selected_deformation_results["Class II"]["correct_rate"][idx_deform] = (
                c2_correct_count / trials_per_level_robustness
            )
            selected_deformation_results["Class II"]["to_class_I_rate"][idx_deform] = (
                c2_to_c1_count / trials_per_level_robustness
            )
            selected_deformation_results["Class II"]["to_unknown_rate"][idx_deform] = (
                c2_to_unk_count / trials_per_level_robustness
            )
            if total_neg_trials_for_this_deform_level > 0:
                selected_deformation_results["Negative"]["reject_rate"][idx_deform] = (
                    neg_reject_count / total_neg_trials_for_this_deform_level
                )
                selected_deformation_results["Negative"]["to_class_I_rate"][
                    idx_deform
                ] = (neg_to_c1_count / total_neg_trials_for_this_deform_level)
                selected_deformation_results["Negative"]["to_class_II_rate"][
                    idx_deform
                ] = (neg_to_c2_count / total_neg_trials_for_this_deform_level)
            else:
                selected_deformation_results["Negative"]["reject_rate"][idx_deform] = 0
                selected_deformation_results["Negative"]["to_class_I_rate"][
                    idx_deform
                ] = 0
                selected_deformation_results["Negative"]["to_class_II_rate"][
                    idx_deform
                ] = 0
    print("--- 形变敏感性分析完成 ---\n")

    print("\n--- 可视化鲁棒性结果并生成摘要 ---")

    # 1. 绘制噪声鲁棒性曲线图
    plt.figure(figsize=(12, 8))
    # 类别I正确识别率
    plt.plot(
        noise_levels_robustness,
        noise_analysis_results["Class I"]["correct_rate"] * 100,
        marker="o",
        linestyle="-",
        color="#1f77b4",
        markersize=10,
        linewidth=3.0,
    )
    # 类别II正确识别率
    plt.plot(
        noise_levels_robustness,
        noise_analysis_results["Class II"]["correct_rate"] * 100,
        marker="s",
        linestyle="-",
        color="#2ca02c",
        markersize=10,
        linewidth=3.0,
    )
    # 负样本正确拒绝率
    plt.plot(
        noise_levels_robustness,
        noise_analysis_results["Negative"]["reject_rate"] * 100,
        marker="^",
        linestyle="-",
        color="#d62728",
        markersize=10,
        linewidth=3.0,
    )
    # 平均性能线
    y_avg_noise = (
        noise_analysis_results["Class I"]["correct_rate"] * 100
        + noise_analysis_results["Class II"]["correct_rate"] * 100
        + noise_analysis_results["Negative"]["reject_rate"] * 100
    ) / 3
    plt.plot(
        noise_levels_robustness,
        y_avg_noise,
        marker="x",
        linestyle="--",
        color="gray",
        markersize=10,
        linewidth=2.5,
        label="平均性能",
    )

    # 添加网格和标签
    plt.grid(True, alpha=0.3)
    plt.xlabel("噪声标准差 (σ)", fontsize=18)
    plt.ylabel("正确判断率 (%)", fontsize=18)

    # 设置坐标轴范围，确保最大值为100%
    plt.ylim(0, 105)

    # 设置刻度间隔
    plt.yticks([0, 20, 40, 60, 80, 100])
    plt.xticks([0, 0.05, 0.10, 0.15, 0.20, 0.25])

    # 修改坐标轴样式
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)

    # 增大刻度标签字体
    plt.tick_params(axis="both", which="major", labelsize=16)

    # 调整X轴范围，使左边界向左移动一点，避免原点处的刻度标签重叠
    current_xlim = ax.get_xlim()
    new_left_limit = -0.01  # 设置一个小的负值
    ax.set_xlim(left=new_left_limit, right=current_xlim[1])

    # 保存图表
    noise_fig_path = os.path.join(save_dir, "octahedron_robustness_noise.png")
    plt.savefig(noise_fig_path, dpi=300, bbox_inches="tight")
    plt.close()

    # 2. 绘制形变鲁棒性曲线图
    for deformation_type in deformation_types_robustness:
        if deformation_type == "uniaxial_stretch":
            selected_deformation_results = deformation_analysis_results_stretch
            fname = "octahedron_robustness_uniaxial_stretch.png"
        elif deformation_type == "random_vertex_displacement":
            selected_deformation_results = deformation_analysis_results_random_disp
            fname = "octahedron_robustness_random_vertex_disp.png"
        else:
            continue

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111)

        x_deform = selected_deformation_results["deformation_levels"]
        y_c1 = selected_deformation_results["Class I"]["correct_rate"] * 100
        y_c2 = selected_deformation_results["Class II"]["correct_rate"] * 100
        y_neg = selected_deformation_results["Negative"]["reject_rate"] * 100

        # 类别I正确识别率
        ax.plot(
            x_deform,
            y_c1,
            marker="o",
            linestyle="-",
            color="#1f77b4",
            markersize=10,
            linewidth=3.0,
        )

        # 类别II正确识别率
        ax.plot(
            x_deform,
            y_c2,
            marker="s",
            linestyle="-",
            color="#2ca02c",
            markersize=10,
            linewidth=3.0,
        )

        # 负样本正确拒绝率
        ax.plot(
            x_deform,
            y_neg,
            marker="^",
            linestyle="-",
            color="#d62728",
            markersize=10,
            linewidth=3.0,
        )

        # 平均性能线
        y_avg = (y_c1 + y_c2 + y_neg) / 3
        ax.plot(
            x_deform,
            y_avg,
            marker="x",
            linestyle="--",
            color="gray",
            markersize=10,
            linewidth=2.5,
        )

        # 添加网格和标签
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("形变水平", fontsize=18)
        ax.set_ylabel("正确判断率 (%)", fontsize=18)

        # 设置坐标轴范围
        ax.set_ylim(0, 105)
        ax.set_yticks([0, 20, 40, 60, 80, 100])

        # 设置边框
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.5)

        # 增大刻度标签字体
        ax.tick_params(axis="both", which="major", labelsize=16)

        # 调整X轴范围，避免原点处的刻度标签重叠
        current_xlim = ax.get_xlim()
        new_left_limit = -0.01
        ax.set_xlim(left=new_left_limit, right=current_xlim[1])

        # 保存图表
        deform_fig_path = os.path.join(save_dir, fname)
        plt.savefig(deform_fig_path, dpi=300, bbox_inches="tight")
        plt.close()

    # 3. 打印鲁棒性分析结果摘要
    # a. 噪声分析摘要
    print("\n--- 噪声分析摘要 ---")
    for idx, sigma in enumerate(noise_analysis_results["noise_levels"]):
        c1 = noise_analysis_results["Class I"]["correct_rate"][idx] * 100
        c2 = noise_analysis_results["Class II"]["correct_rate"][idx] * 100
        neg = noise_analysis_results["Negative"]["reject_rate"][idx] * 100
        print(
            f"噪声σ={sigma:.3f}: 类别I正确率={c1:.1f}%, 类别II正确率={c2:.1f}%, 负样本拒绝率={neg:.1f}%"
        )
    # b. 形变分析摘要
    for deformation_type in deformation_types_robustness:
        if deformation_type == "uniaxial_stretch":
            selected_deformation_results = deformation_analysis_results_stretch
            title = "单轴拉伸"
        elif deformation_type == "random_vertex_displacement":
            selected_deformation_results = deformation_analysis_results_random_disp
            title = "随机顶点位移"
        else:
            continue
        print(f"\n--- {title} 形变分析摘要 ---")
        for idx, deform_level in enumerate(
            selected_deformation_results["deformation_levels"]
        ):
            c1 = selected_deformation_results["Class I"]["correct_rate"][idx] * 100
            c2 = selected_deformation_results["Class II"]["correct_rate"][idx] * 100
            neg = selected_deformation_results["Negative"]["reject_rate"][idx] * 100
            print(
                f"形变={deform_level:.3f}: 类别I正确率={c1:.1f}%, 类别II正确率={c2:.1f}%, 负样本拒绝率={neg:.1f}%"
            )
    print(f"\n鲁棒性分析图表和摘要已生成。图表保存在: {save_dir}")
    # 函数可不返回内容


def plot_parameter_optimization_results_3d(
    all_results_list, best_params_info, save_dir
):
    """
    可视化参数优化结果（综合评分、类别I准确率、类别II准确率、负样本拒绝率热力图），并保存到指定目录。

    参数：
        all_results_list: find_optimal_parameters_3d 返回的 all_results 列表
        best_params_info: 字典，包含 'alpha_o', 'lambda_o', 'best_score'
        save_dir: 图像保存目录
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os

    # 1. 提取所有 alpha_o 和 lambda_o 的唯一值
    alpha_values = sorted(list(set([result["alpha_o"] for result in all_results_list])))
    lambda_values = sorted(
        list(set([result["lambda_o"] for result in all_results_list]))
    )

    # 2. 创建矩阵
    score_matrix = np.zeros((len(alpha_values), len(lambda_values)))
    c1_acc_matrix = np.zeros((len(alpha_values), len(lambda_values)))
    c2_acc_matrix = np.zeros((len(alpha_values), len(lambda_values)))
    neg_acc_matrix = np.zeros((len(alpha_values), len(lambda_values)))

    # 3. 填充矩阵
    for result in all_results_list:
        alpha_idx = alpha_values.index(result["alpha_o"])
        lambda_idx = lambda_values.index(result["lambda_o"])
        score_matrix[alpha_idx, lambda_idx] = result["combined_score"]
        c1_acc_matrix[alpha_idx, lambda_idx] = result["c1_acc_rate"] * 100  # 转为百分比
        c2_acc_matrix[alpha_idx, lambda_idx] = result["c2_acc_rate"] * 100
        neg_acc_matrix[alpha_idx, lambda_idx] = result["neg_acc_rate"] * 100

    # 找到最佳点在矩阵中的索引
    best_alpha_idx = alpha_values.index(best_params_info["alpha_o"])
    best_lambda_idx = lambda_values.index(best_params_info["lambda_o"])
    best_score = best_params_info["best_score"]

    # 更新字体大小设置
    plt.rcParams.update({"font.size": 16})

    # 4. 创建热力图（2x2布局）
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))

    # 综合评分热力图
    ax1 = axes[0, 0]
    sns.heatmap(
        score_matrix,
        annot=False,
        cmap="viridis",
        xticklabels=[f"{x:.2f}" for x in lambda_values],
        yticklabels=[f"{x:.2f}" for x in alpha_values],
        ax=ax1,
    )
    ax1.set_title("综合评分", fontsize=25)
    ax1.set_xlabel("$\\lambda_o$", fontsize=25)
    ax1.set_ylabel("$\\alpha_o$", fontsize=25)
    ax1.tick_params(axis="both", which="major", labelsize=18)

    # 类别I准确率热力图
    ax2 = axes[0, 1]
    sns.heatmap(
        c1_acc_matrix,
        annot=False,
        cmap="Blues",
        xticklabels=[f"{x:.2f}" for x in lambda_values],
        yticklabels=[f"{x:.2f}" for x in alpha_values],
        ax=ax2,
    )
    ax2.set_title("类别I准确率（%）", fontsize=25)
    ax2.set_xlabel("$\\lambda_o$", fontsize=25)
    ax2.set_ylabel("$\\alpha_o$", fontsize=25)
    ax2.tick_params(axis="both", which="major", labelsize=18)

    # 类别II准确率热力图
    ax3 = axes[1, 0]
    sns.heatmap(
        c2_acc_matrix,
        annot=False,
        cmap="Greens",
        xticklabels=[f"{x:.2f}" for x in lambda_values],
        yticklabels=[f"{x:.2f}" for x in alpha_values],
        ax=ax3,
    )
    ax3.set_title("类别II准确率（%）", fontsize=25)
    ax3.set_xlabel("$\\lambda_o$", fontsize=25)
    ax3.set_ylabel("$\\alpha_o$", fontsize=25)
    ax3.tick_params(axis="both", which="major", labelsize=18)

    # 负样本拒绝率热力图
    ax4 = axes[1, 1]
    sns.heatmap(
        neg_acc_matrix,
        annot=False,
        cmap="Reds",
        xticklabels=[f"{x:.2f}" for x in lambda_values],
        yticklabels=[f"{x:.2f}" for x in alpha_values],
        ax=ax4,
    )
    ax4.set_title("负样本拒绝率（%）", fontsize=25)
    ax4.set_xlabel("$\\lambda_o$", fontsize=25)
    ax4.set_ylabel("$\\alpha_o$", fontsize=25)
    ax4.tick_params(axis="both", which="major", labelsize=18)

    # 5. 标记最佳参数点
    for ax in [ax1, ax2, ax3, ax4]:
        ax.plot(
            best_lambda_idx + 0.5,
            best_alpha_idx + 0.5,
            "o",
            color="red",
            markersize=14,
            markerfacecolor="none",
            markeredgewidth=3,
        )

    # 6. 添加总标题
    plt.suptitle(
        f"最佳参数: $\\alpha_o$={best_params_info['alpha_o']:.2f}, $\\lambda_o$={best_params_info['lambda_o']:.2f}, 评分={best_params_info['best_score']:.2f}",
        fontsize=30,
    )

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)

    # 7. 保存图像
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "octahedron_parameter_optimization_heatmaps.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"参数优化热力图已保存为: {save_path}")
    plt.close(fig)

    # 可选：绘制综合评分的3D曲面图
    try:
        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib import cm
        import matplotlib

        # meshgrid: lambda为X，alpha为Y，score为Z
        lambda_grid, alpha_grid = np.meshgrid(
            lambda_values, alpha_values, indexing="xy"
        )
        Z = score_matrix  # 形状与 meshgrid 匹配

        fig3d = plt.figure(figsize=(12, 10))
        ax3d = fig3d.add_subplot(111, projection="3d")
        surf = ax3d.plot_surface(
            lambda_grid,
            alpha_grid,
            Z,
            cmap=cm.viridis,
            alpha=0.85,
            antialiased=True,
        )

        # 轴标签 - 更大字体
        ax3d.set_xlabel("$\\lambda_o$", fontsize=20, labelpad=15)
        ax3d.set_ylabel("$\\alpha_o$", fontsize=20, labelpad=15)
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
        zmin = float(np.nanmin(Z))
        # 曲面上的红色星号
        ax3d.scatter(
            best_params_info["lambda_o"],
            best_params_info["alpha_o"],
            best_score,
            color="red",
            marker="*",
            s=200,
            depthshade=True,
            label="最佳参数",
        )
        # 底部投影红色叉号
        ax3d.scatter(
            best_params_info["lambda_o"],
            best_params_info["alpha_o"],
            zmin,
            color="red",
            marker="x",
            s=100,
            depthshade=True,
        )
        # 红色虚线连接
        ax3d.plot(
            [best_params_info["lambda_o"], best_params_info["lambda_o"]],
            [best_params_info["alpha_o"], best_params_info["alpha_o"]],
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
            f"λ_o = {best_params_info['lambda_o']:.2f}\n"
            f"α_o = {best_params_info['alpha_o']:.2f}\n"
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

        # 让3D曲面图保持交互状态，而不是自动保存
        save_path_3d = os.path.join(
            save_dir, "octahedron_parameter_optimization_3d.png"
        )

        print(f"3D曲面图已显示，请调整视角后手动保存")
        print(f"建议保存路径: {save_path_3d}")

        # 显示图形而不是保存，让用户可以旋转视角
        plt.show()

        print(f"参数优化可视化结果:")
        print(f"  热力图已保存: {save_path}")
        print(f"  3D曲面图已交互显示")

    except Exception as e:
        print(f"3D曲面图绘制失败: {e}")
        print(f"热力图已保存至: {save_path}")


def analyze_best_params(best_params):
    """分析最佳参数的详细性能指标（集成 test_parameter_optimization.py 的逻辑）"""
    best_result = None
    for result in best_params["all_results"]:
        if (
            result["alpha_o"] == best_params["alpha_o"]
            and result["lambda_o"] == best_params["lambda_o"]
        ):
            best_result = result
            break
    if best_result:
        print("\n=== 最佳参数的详细性能指标 ===")
        print(
            f"alpha_o = {best_result['alpha_o']:.2f}, lambda_o = {best_result['lambda_o']:.2f}"
        )
        print(f"综合评分: {best_result['combined_score']:.4f}")
        print("\n分类准确率:")
        print(f"  类别I准确率: {best_result['c1_acc_rate']:.4f}")
        print(f"  类别II准确率: {best_result['c2_acc_rate']:.4f}")
        print(f"  负样本拒绝率: {best_result['neg_acc_rate']:.4f}")
        print("\n错误率:")
        print(f"  类别I误判为类别II: {best_result['c1_to_c2_error_rate']:.4f}")
        print(f"  类别I误判为Unknown: {best_result['c1_to_unk_error_rate']:.4f}")
        print(f"  类别II误判为类别I: {best_result['c2_to_c1_error_rate']:.4f}")
        print(f"  类别II误判为Unknown: {best_result['c2_to_unk_error_rate']:.4f}")
        print(f"  负样本误判为类别I: {best_result['neg_to_c1_error_rate']:.4f}")
        print(f"  负样本误判为类别II: {best_result['neg_to_c2_error_rate']:.4f}")
    else:
        print("未找到最佳参数对应的详细结果")


if __name__ == "__main__":
    # === 全局输出目录常量 ===
    FIGURES_DIR_Q2 = os.path.join("..", "figures2")
    params_dir = os.path.join(FIGURES_DIR_Q2, "params")
    robustness_dir = os.path.join(FIGURES_DIR_Q2, "robustness")
    observed_samples_dir = os.path.join(
        FIGURES_DIR_Q2, "observed_samples_classification"
    )
    for directory in [FIGURES_DIR_Q2, params_dir, robustness_dir, observed_samples_dir]:
        os.makedirs(directory, exist_ok=True)

    # === 加载标准数据和负样本 ===
    std_class_I_raw = np.array(
        [
            [1.00, 0.00, 0.00],
            [0.00, 1.00, 0.00],
            [-1.00, 0.00, 0.00],
            [0.00, -1.00, 0.00],
            [0.00, 0.00, 1.00],
            [0.00, 0.00, -1.00],
        ]
    )
    std_class_II_raw = np.array(
        [
            [1.20, 0, 0.00],
            [0.20, 1.13, 0.00],
            [-1.03, 0.04, 0.00],
            [0.25, -2.03, 0.00],
            [0.12, -0.45, 2.01],
            [-0.09, 1.20, -1.05],
        ]
    )
    # negative_samples_3d_list 已在顶部定义

    # === 参数优化部分 ===
    RUN_PARAMETER_OPTIMIZATION = True  # 如需优化设为True
    OPTIMIZED_ALPHA_O = 0.80  # 预设优化参数
    OPTIMIZED_LAMBDA_O = 0.40
    print(
        f"初始/预设优化参数: alpha_o={OPTIMIZED_ALPHA_O}, lambda_o={OPTIMIZED_LAMBDA_O}"
    )

    if RUN_PARAMETER_OPTIMIZATION:
        print("\n=== 开始参数优化 ===")
        classifier_params_for_opt = {"icp_max_iterations": 50, "icp_tolerance": 1e-6}
        best_octa_params_dict = find_optimal_parameters_3d(
            std_class_I_raw,
            std_class_II_raw,
            negative_samples_3d_list,
            classifier_params_for_opt,
        )
        OPTIMIZED_ALPHA_O = best_octa_params_dict["alpha_o"]
        OPTIMIZED_LAMBDA_O = best_octa_params_dict["lambda_o"]
        # 参数优化结果可视化
        plot_parameter_optimization_results_3d(
            best_octa_params_dict["all_results"],
            {
                "alpha_o": OPTIMIZED_ALPHA_O,
                "lambda_o": OPTIMIZED_LAMBDA_O,
                "best_score": best_octa_params_dict["best_score"],
            },
            params_dir,
        )
        # 打印最佳参数详细性能
        analyze_best_params(best_octa_params_dict)
        print(
            f"参数优化完成，最佳参数: alpha_o={OPTIMIZED_ALPHA_O:.4f}, lambda_o={OPTIMIZED_LAMBDA_O:.4f}"
        )
    else:
        print("\n使用预设优化参数进行后续分析。")

    # === 实例化最终分类器 ===
    ICP_MAX_ITERATIONS = 50
    ICP_TOLERANCE = 1e-6
    final_classifier = AdvancedOctahedronClassifier(
        alpha_o=OPTIMIZED_ALPHA_O,
        lambda_o=OPTIMIZED_LAMBDA_O,
        icp_max_iterations=ICP_MAX_ITERATIONS,
        icp_tolerance=ICP_TOLERANCE,
    )

    # === 对表四观测八面体进行分类与可视化 ===
    observed_polyhedra_raw_list = [
        (
            np.array(
                [
                    [-2.1389, 0.4082, 2.3177],
                    [-1.3502, 0.1494, 2.8754],
                    [-1.9436, -0.4082, 3.4558],
                    [-2.7323, -0.1494, 2.8981],
                    [-1.9275, -0.5577, 2.4671],
                    [-2.1549, 0.5577, 3.3064],
                ]
            ),
            "八面体1",
        ),
        (
            np.array(
                [
                    [-0.42, 1.9658, -2.6577],
                    [3.1888, 1.0626, -0.9022],
                    [0.3517, -1.7738, 2.7681],
                    [-6.2263, -0.7056, -0.6016],
                    [-0.4835, -4.5869, -3.8465],
                    [3.2113, 3.2448, 2.0653],
                ]
            ),
            "八面体2",
        ),
        (
            np.array(
                [
                    [6.4945, 10.4945, 0.1942],
                    [-0.5402, 5.4598, -8.1474],
                    [-16.3093, -10.3093, 0.381],
                    [-5.2359, -1.2359, 14.6057],
                    [-6.6805, -1.6805, -8.7847],
                    [-2.3227, 2.6773, -2.458],
                ]
            ),
            "八面体3",
        ),
        (
            np.array(
                [
                    [2.4847, 3.6668, 0.1145],
                    [-0.0666, 2.053, -2.63],
                    [-5.2413, -3.1395, 0.4072],
                    [-1.6707, -0.0922, 4.9533],
                    [-1.9764, -0.3778, -2.6568],
                    [-0.6892, 0.9386, -0.7382],
                ]
            ),
            "八面体4",
        ),
        (
            np.array(
                [
                    [5.4945, 10.4945, 0.1942],
                    [-0.5402, 4.4598, -8.1474],
                    [-15.3093, -10.3093, 0.381],
                    [-5.2359, -0.2359, 14.6057],
                    [-6.6805, -1.6805, -9.7847],
                    [-2.3227, 2.6773, -1.458],
                ]
            ),
            "八面体5",
        ),
        (
            np.array(
                [
                    [1.4819, 7.921, 4.7339],
                    [0.482, 8.5766, 4.2529],
                    [1.4933, 6.8309, 4.616],
                    [0.1752, 5.7335, 5.5263],
                    [-0.6712, 7.9527, 4.6671],
                    [1.4278, 7.1234, 6.2662],
                ]
            ),
            "八面体6",
        ),
    ]
    print("\n=== 对观测八面体样本进行分类与可视化 ===")
    for obs_vertices, obs_name in observed_polyhedra_raw_list:
        print(f"--- 处理 {obs_name} ---")
        classification, detailed_results = final_classifier.classify(
            obs_vertices, std_class_I_raw, std_class_II_raw
        )
        print(f"分类结果: {classification}")
        print("\n关键指标:")
        print(f"  RMSD到类别I: {detailed_results['rmsd_I']:.4f}")
        print(f"  RMSD到类别II: {detailed_results['rmsd_II']:.4f}")
        print(f"  特征距离到类别I: {detailed_results['dF_I']:.4f}")
        print(f"  特征距离到类别II: {detailed_results['dF_II']:.4f}")
        print(f"  综合评分S_I: {detailed_results['S_I']:.4f}")
        print(f"  综合评分S_II: {detailed_results['S_II']:.4f}")
        print(f"  最小评分: {detailed_results['min_score']:.4f}")
        print(f"  阈值lambda_o: {OPTIMIZED_LAMBDA_O:.2f}")
        if detailed_results["min_score"] <= OPTIMIZED_LAMBDA_O:
            print(
                f"  结论: {obs_name}的最小评分 {detailed_results['min_score']:.4f} <= 阈值 {OPTIMIZED_LAMBDA_O}, 分类为 {classification}"
            )
        else:
            print(
                f"  结论: {obs_name}的最小评分 {detailed_results['min_score']:.4f} > 阈值 {OPTIMIZED_LAMBDA_O}, 分类为 Unknown"
            )
        save_path_align = os.path.join(
            observed_samples_dir, f"{obs_name.replace(' ', '_')}_alignment.png"
        )
        final_classifier.visualize_alignment(
            detailed_results, title_prefix=obs_name, save_path=save_path_align
        )
        print(f"对齐图像已保存至: {save_path_align}\n")

    # === 执行全面鲁棒性分析 ===
    print("\n=== 开始全面鲁棒性分析 ===")
    comprehensive_robustness_analysis_3d(
        final_classifier,
        std_class_I_raw,
        std_class_II_raw,
        negative_samples_3d_list,
        robustness_dir,
    )
    print("\n=== 全部流程结束 ===")
