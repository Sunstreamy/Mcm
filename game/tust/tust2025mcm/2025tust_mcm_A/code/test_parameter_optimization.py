import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from q2 import find_optimal_parameters_3d, negative_samples_3d_list
from tqdm import tqdm
import time
from q2 import AdvancedOctahedronClassifier


def visualize_optimization_results(best_params):
    """可视化参数优化结果"""
    # 提取所有结果
    all_results = best_params["all_results"]

    # 获取唯一的alpha_o和lambda_o值
    alpha_values = sorted(list(set([result["alpha_o"] for result in all_results])))
    lambda_values = sorted(list(set([result["lambda_o"] for result in all_results])))

    # 创建评分矩阵
    score_matrix = np.zeros((len(alpha_values), len(lambda_values)))
    c1_acc_matrix = np.zeros((len(alpha_values), len(lambda_values)))
    c2_acc_matrix = np.zeros((len(alpha_values), len(lambda_values)))
    neg_acc_matrix = np.zeros((len(alpha_values), len(lambda_values)))

    # 填充矩阵
    for result in all_results:
        alpha_idx = alpha_values.index(result["alpha_o"])
        lambda_idx = lambda_values.index(result["lambda_o"])
        score_matrix[alpha_idx, lambda_idx] = result["combined_score"]
        c1_acc_matrix[alpha_idx, lambda_idx] = result["c1_acc_rate"]
        c2_acc_matrix[alpha_idx, lambda_idx] = result["c2_acc_rate"]
        neg_acc_matrix[alpha_idx, lambda_idx] = result["neg_acc_rate"]
    # 创建热力图
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

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
    ax1.set_title("综合评分 (Combined Score)")
    ax1.set_xlabel("lambda_o")
    ax1.set_ylabel("alpha_o")

    # 类别I准确率热力图
    ax2 = axes[0, 1]
    sns.heatmap(
        c1_acc_matrix,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=[f"{x:.2f}" for x in lambda_values],
        yticklabels=[f"{x:.2f}" for x in alpha_values],
        ax=ax2,
    )
    ax2.set_title("类别I准确率 (Class I Accuracy)")
    ax2.set_xlabel("lambda_o")
    ax2.set_ylabel("alpha_o")

    # 类别II准确率热力图
    ax3 = axes[1, 0]
    sns.heatmap(
        c2_acc_matrix,
        annot=True,
        fmt=".2f",
        cmap="Greens",
        xticklabels=[f"{x:.2f}" for x in lambda_values],
        yticklabels=[f"{x:.2f}" for x in alpha_values],
        ax=ax3,
    )
    ax3.set_title("类别II准确率 (Class II Accuracy)")
    ax3.set_xlabel("lambda_o")
    ax3.set_ylabel("alpha_o")

    # 负样本拒绝率热力图
    ax4 = axes[1, 1]
    sns.heatmap(
        neg_acc_matrix,
        annot=True,
        fmt=".2f",
        cmap="Reds",
        xticklabels=[f"{x:.2f}" for x in lambda_values],
        yticklabels=[f"{x:.2f}" for x in alpha_values],
        ax=ax4,
    )
    ax4.set_title("负样本拒绝率 (Negative Sample Rejection Rate)")
    ax4.set_xlabel("lambda_o")
    ax4.set_ylabel("alpha_o")

    # 标记最佳参数点
    best_alpha_idx = alpha_values.index(best_params["alpha_o"])
    best_lambda_idx = lambda_values.index(best_params["lambda_o"])

    for ax in [ax1, ax2, ax3, ax4]:
        ax.plot(
            best_lambda_idx + 0.5, best_alpha_idx + 0.5, "o", color="red", markersize=10
        )

    # 添加总标题
    plt.suptitle(
        f"参数优化结果 - 最佳参数: alpha_o={best_params['alpha_o']:.2f}, lambda_o={best_params['lambda_o']:.2f}, 评分={best_params['best_score']:.2f}",
        fontsize=16,
    )

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)

    # 保存图像
    plt.savefig("parameter_optimization_results.png", dpi=300)
    print("结果可视化已保存为 parameter_optimization_results.png")

    # 显示图像
    plt.show()


def analyze_best_params(best_params):
    """分析最佳参数的详细性能指标"""
    # 找到最佳参数对应的结果
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
            f"alpha_o = {best_result['alpha_o']:.4f}, lambda_o = {best_result['lambda_o']:.4f}"
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


def test_optimized_parameters(
    optimized_alpha, optimized_lambda, negative_samples_list_from_main
):
    """
    测试优化后的参数对所有样本的分类效果
    """
    print("\n=== 测试优化后参数的分类效果 ===")

    # 1. 定义优化后的参数
    OPTIMIZED_ALPHA_O = optimized_alpha
    OPTIMIZED_LAMBDA_O = optimized_lambda
    ICP_MAX_ITERATIONS = 50
    ICP_TOLERANCE = 1e-6

    # 2. 创建分类器实例
    classifier = AdvancedOctahedronClassifier(
        alpha_o=OPTIMIZED_ALPHA_O,
        lambda_o=OPTIMIZED_LAMBDA_O,
        icp_max_iterations=ICP_MAX_ITERATIONS,
        icp_tolerance=ICP_TOLERANCE,
    )

    # 3. 加载标准类别八面体数据
    std_class_I_raw = np.array(
        [
            [1.00, 0.00, 0.00],  # 顶点一
            [0.00, 1.00, 0.00],  # 顶点二
            [-1.00, 0.00, 0.00],  # 顶点三
            [0.00, -1.00, 0.00],  # 顶点四
            [0.00, 0.00, 1.00],  # 顶点五
            [0.00, 0.00, -1.00],  # 顶点六
        ]
    )

    std_class_II_raw = np.array(
        [
            [1.20, 0, 0.00],  # 顶点一
            [0.20, 1.13, 0.00],  # 顶点二
            [-1.03, 0.04, 0.00],  # 顶点三
            [0.25, -2.03, 0.00],  # 顶点四
            [0.12, -0.45, 2.01],  # 顶点五
            [-0.09, 1.20, -1.05],  # 顶点六
        ]
    )

    # 4. 定义扰动参数
    noise_levels = [0.05, 0.1, 0.15]
    deformation_types = ["uniaxial_stretch", "random_vertex_displacement"]
    deformation_params = [0.1, 0.2, 0.3]  # 10%, 20%, 30%的变形程度

    # 5. 生成测试样本
    print("生成测试样本...")
    # 生成Class I样本
    class_I_samples = []
    for noise in noise_levels:
        for _ in range(5):  # 每个噪声水平生成5个样本
            # 添加噪声
            noisy_vertices = std_class_I_raw + np.random.normal(
                0, noise, std_class_I_raw.shape
            )
            class_I_samples.append((noisy_vertices, f"Class I (noise={noise})"))

    # 对Class I样本应用变形
    for def_type in deformation_types:
        for param in deformation_params:
            for _ in range(3):  # 每种变形生成3个样本
                # 添加变形
                deformed_vertices = classifier.apply_deformation_3d(
                    std_class_I_raw.copy(), def_type, param
                )
                class_I_samples.append(
                    (deformed_vertices, f"Class I ({def_type}, param={param})")
                )

    # 生成Class II样本
    class_II_samples = []
    for noise in noise_levels:
        for _ in range(5):  # 每个噪声水平生成5个样本
            # 添加噪声
            noisy_vertices = std_class_II_raw + np.random.normal(
                0, noise, std_class_II_raw.shape
            )
            class_II_samples.append((noisy_vertices, f"Class II (noise={noise})"))

    # 对Class II样本应用变形
    for def_type in deformation_types:
        for param in deformation_params:
            for _ in range(3):  # 每种变形生成3个样本
                # 添加变形
                deformed_vertices = classifier.apply_deformation_3d(
                    std_class_II_raw.copy(), def_type, param
                )
                class_II_samples.append(
                    (deformed_vertices, f"Class II ({def_type}, param={param})")
                )

    # 6. 测试分类效果
    print(
        f"测试样本数量: Class I={len(class_I_samples)}, Class II={len(class_II_samples)}, 负样本={len(negative_samples_3d_list)}"
    )
    print("开始测试分类效果...")

    # 初始化计数器和混淆矩阵
    class_I_correct = 0
    class_II_correct = 0
    negative_correct = 0

    # 初始化混淆矩阵 [真实类别][预测类别]
    confusion_matrix = np.zeros((3, 3), dtype=int)
    # 类别索引: 0=Class I, 1=Class II, 2=Unknown

    # 测试Class I样本
    print("\n测试Class I样本:")
    for vertices, name in class_I_samples:
        result, metrics = classifier.classify(
            vertices, std_class_I_raw, std_class_II_raw
        )
        print(f"样本 {name}: 分类为 {result}")
        if result == "Class I":
            class_I_correct += 1
            confusion_matrix[0, 0] += 1  # 真实Class I, 预测Class I
        elif result == "Class II":
            confusion_matrix[0, 1] += 1  # 真实Class I, 预测Class II
        else:  # Unknown
            confusion_matrix[0, 2] += 1  # 真实Class I, 预测Unknown

    # 测试Class II样本
    print("\n测试Class II样本:")
    for vertices, name in class_II_samples:
        result, metrics = classifier.classify(
            vertices, std_class_I_raw, std_class_II_raw
        )
        print(f"样本 {name}: 分类为 {result}")
        if result == "Class II":
            class_II_correct += 1
            confusion_matrix[1, 1] += 1  # 真实Class II, 预测Class II
        elif result == "Class I":
            confusion_matrix[1, 0] += 1  # 真实Class II, 预测Class I
        else:  # Unknown
            confusion_matrix[1, 2] += 1  # 真实Class II, 预测Unknown

    # 测试负样本
    print("\n测试负样本:")
    for vertices, name in negative_samples_3d_list:
        result, metrics = classifier.classify(
            vertices, std_class_I_raw, std_class_II_raw
        )
        print(f"样本 {name}: 分类为 {result}")
        if result == "Unknown":
            negative_correct += 1
            confusion_matrix[2, 2] += 1  # 真实Unknown, 预测Unknown
        elif result == "Class I":
            confusion_matrix[2, 0] += 1  # 真实Unknown, 预测Class I
        else:  # Class II
            confusion_matrix[2, 1] += 1  # 真实Unknown, 预测Class II

    # 7. 计算分类指标
    class_I_accuracy = (
        class_I_correct / len(class_I_samples) if len(class_I_samples) > 0 else 0
    )
    class_II_accuracy = (
        class_II_correct / len(class_II_samples) if len(class_II_samples) > 0 else 0
    )
    negative_rejection_rate = (
        negative_correct / len(negative_samples_3d_list)
        if len(negative_samples_3d_list) > 0
        else 0
    )

    # 8. 打印分类结果
    print("\n=== 分类结果汇总 ===")
    print(f"参数: alpha_o={OPTIMIZED_ALPHA_O}, lambda_o={OPTIMIZED_LAMBDA_O}")
    print(
        f"Class I 正确率: {class_I_correct}/{len(class_I_samples)} = {class_I_accuracy:.4f}"
    )
    print(
        f"Class II 正确率: {class_II_correct}/{len(class_II_samples)} = {class_II_accuracy:.4f}"
    )
    print(
        f"负样本拒绝率: {negative_correct}/{len(negative_samples_3d_list)} = {negative_rejection_rate:.4f}"
    )

    # 9. 计算综合评分
    weights = {
        "class_I_accuracy": 0.4,
        "class_II_accuracy": 0.4,
        "negative_rejection": 0.2,
    }

    combined_score = (
        weights["class_I_accuracy"] * class_I_accuracy
        + weights["class_II_accuracy"] * class_II_accuracy
        + weights["negative_rejection"] * negative_rejection_rate
    )

    print(f"综合评分: {combined_score:.4f}")

    # 10. 绘制混淆矩阵
    visualize_confusion_matrix(confusion_matrix)

    return {
        "alpha_o": OPTIMIZED_ALPHA_O,
        "lambda_o": OPTIMIZED_LAMBDA_O,
        "class_I_accuracy": class_I_accuracy,
        "class_II_accuracy": class_II_accuracy,
        "negative_rejection_rate": negative_rejection_rate,
        "combined_score": combined_score,
        "confusion_matrix": confusion_matrix,
    }


def visualize_confusion_matrix(confusion_matrix):
    """
    可视化混淆矩阵
    """
    class_names = ["Class I", "Class II", "Unknown"]

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        confusion_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("预测类别")
    plt.ylabel("真实类别")
    plt.title("分类混淆矩阵")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    print("混淆矩阵已保存为 confusion_matrix.png")


def main():
    """测试参数优化函数的基本功能"""
    print("=== 测试参数优化函数 ===")

    # 加载标准类别八面体数据
    std_class_I_raw = np.array(
        [
            [1.00, 0.00, 0.00],  # 顶点一
            [0.00, 1.00, 0.00],  # 顶点二
            [-1.00, 0.00, 0.00],  # 顶点三
            [0.00, -1.00, 0.00],  # 顶点四
            [0.00, 0.00, 1.00],  # 顶点五
            [0.00, 0.00, -1.00],  # 顶点六
        ]
    )

    std_class_II_raw = np.array(
        [
            [1.20, 0, 0.00],  # 顶点一
            [0.20, 1.13, 0.00],  # 顶点二
            [-1.03, 0.04, 0.00],  # 顶点三
            [0.25, -2.03, 0.00],  # 顶点四
            [0.12, -0.45, 2.01],  # 顶点五
            [-0.09, 1.20, -1.05],  # 顶点六
        ]
    )

    # 配置分类器参数
    classifier_params = {"icp_max_iterations": 50, "icp_tolerance": 1e-6}

    # 调用参数优化函数
    best_params = find_optimal_parameters_3d(
        std_class_I_raw, std_class_II_raw, negative_samples_3d_list, classifier_params
    )

    # 打印优化结果
    print("\n=== 参数优化结果 ===")
    print(f"最佳 alpha_o: {best_params['alpha_o']}")
    print(f"最佳 lambda_o: {best_params['lambda_o']}")
    print(f"最佳综合评分: {best_params['best_score']}")
    print(f"优化用时: {best_params['optimization_time']:.2f} 秒")

    # 分析最佳参数的详细性能
    analyze_best_params(best_params)

    # 可视化结果
    visualize_optimization_results(best_params)

    # 测试优化后的参数
    test_results = test_optimized_parameters(
        best_params["alpha_o"],
        best_params["lambda_o"],
        negative_samples_3d_list,  # 从main作用域传入
    )


if __name__ == "__main__":
    main()
