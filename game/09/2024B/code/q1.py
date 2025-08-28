import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import random

matplotlib.rcParams["font.sans-serif"] = ["SimHei"]  # 用来正常显示中文标签
matplotlib.rcParams["axes.unicode_minus"] = False  # 用来正常显示负号

# --- Step 1: Define SPRT Parameters ---
p0 = 0.10  # 合格质量水平 (H0)
p1 = 0.20  # 不合格质量水平 (H1)
alpha = 0.05  # 第一类错误率 (生产者风险)
beta = 0.10  # 第二类错误率 (消费者风险)
max_samples = 200  # 设置一个最大抽样数防止无限循环

# --- Step 2: Calculate Decision Boundaries ---
# 使用对数形式，计算更稳定
ln_A = np.log((1 - beta) / alpha)
ln_B = np.log(beta / (1 - alpha))

# 计算似然比对数的增量
log_p1_p0 = np.log(p1 / p0)
log_1p1_1p0 = np.log((1 - p1) / (1 - p0))

print(f"SPRT 模型参数:")
print(f"H0 (p0 - AQL): {p0}")
print(f"H1 (p1 - LTPD): {p1}")
print(f"Alpha (I类错误): {alpha}")
print(f"Beta (II类错误): {beta}")
print("-" * 30)
print(f"对数接受边界 ln(B): {ln_B:.4f}")
print(f"对数拒绝边界 ln(A): {ln_A:.4f}")
print("-" * 30)


# --- Part 1: Simulation to Demonstrate Efficiency ---


def run_sprt_simulation(true_p):
    """
    模拟一次完整的SPRT抽样过程
    :param true_p: 这批货物的真实次品率
    :return: (决策结果, 样本数量)
    """
    log_lambda = 0
    for m in range(1, max_samples + 1):
        # 模拟抽取一个样本
        is_defective = random.random() < true_p

        if is_defective:
            log_lambda += log_p1_p0
        else:
            log_lambda += log_1p1_1p0

        # 做出决策
        if log_lambda >= ln_A:
            return "拒绝", m
        if log_lambda <= ln_B:
            return "接受", m

    return "达到最大样本量，无结论", max_samples


def analyze_sprt_performance(true_p, num_simulations=10000):
    """
    运行多次模拟来分析SPRT在特定真实次品率下的性能
    """
    results = []
    sample_sizes = []

    for _ in range(num_simulations):
        result, n = run_sprt_simulation(true_p)
        results.append(result)
        sample_sizes.append(n)

    avg_sample_number = np.mean(sample_sizes)
    acceptance_rate = results.count("接受") / num_simulations
    rejection_rate = results.count("拒绝") / num_simulations

    print(f"--- 模拟分析: 真实次品率 p = {true_p:.2f} ---")
    print(f"平均抽样数 (ASN): {avg_sample_number:.2f}")
    print(f"接受率: {acceptance_rate:.2%}")
    print(f"拒绝率: {rejection_rate:.2%}")
    print("-" * 30)


# 运行三种情景的模拟
print("--- 1. 仿真分析开始 ---")
analyze_sprt_performance(true_p=0.05)  # 场景一：一批好货
analyze_sprt_performance(true_p=0.15)  # 场景二：一批边界货
analyze_sprt_performance(true_p=0.25)  # 场景三：一批坏货


# --- Part 2: Generate the Practical Decision Tool (Chart) ---


def generate_decision_chart():
    """
    生成SPRT决策图，这是给企业的最终方案
    """
    m = np.arange(1, max_samples + 1)  # 样本数 m

    # 决策边界是关于 (m, d) 的线性方程
    # ln(lambda) = d*ln(p1/p0) + (m-d)*ln((1-p1)/(1-p0))
    # 解出 d: d = (ln(lambda) - m*ln((1-p1)/(1-p0))) / (ln(p1/p0) - ln((1-p1)/(1-p0)))

    slope = log_1p1_1p0 / (log_1p1_1p0 - log_p1_p0)
    intercept_accept = ln_B / (log_p1_p0 - log_1p1_1p0)
    intercept_reject = ln_A / (log_p1_p0 - log_1p1_1p0)

    d_accept = slope * m + intercept_accept
    d_reject = slope * m + intercept_reject

    plt.figure(figsize=(12, 8))
    plt.plot(m, d_accept, "g-", label="接受线")
    plt.plot(m, d_reject, "r-", label="拒绝线")

    # 填充区域
    plt.fill_between(m, d_accept, -1, color="green", alpha=0.2, label="接受区域")
    plt.fill_between(
        m, d_reject, 1000, color="red", alpha=0.2, label="拒绝区域"
    )  # y-limit is large
    plt.fill_between(
        m, d_accept, d_reject, color="yellow", alpha=0.3, label="继续抽样区域"
    )

    plt.xlabel("抽样数量", fontsize=12)
    plt.ylabel("发现的次品数", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.xlim(0, 150)  # 可以调整X轴范围以便观察
    plt.ylim(0, 30)  # 可以调整Y轴范围以便观察

    # 在图中标注关键参数
    text_str = f"$p_0={p0}$, $p_1={p1}$\n" f"$\\alpha={alpha}$, $\\beta={beta}$"
    plt.text(
        0.65,
        0.1,
        text_str,
        transform=plt.gca().transAxes,
        fontsize=12,
        verticalalignment="bottom",
        bbox=dict(boxstyle="round,pad=0.5", fc="wheat", alpha=0.5),
    )

    plt.show()


print("\n--- 2. 生成决策工具 ---")
print("即将展示SPRT抽样决策图...")
generate_decision_chart()
