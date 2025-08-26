import numpy as np
import matplotlib.pyplot as plt
import random

# --- 模型参数 (保持不变) ---
p0 = 0.10
p1 = 0.20
alpha = 0.05
beta = 0.10
max_samples = 300 # 稍微增加最大样本量以应对边界情况

# --- 决策边界 (保持不变) ---
ln_A = np.log((1 - beta) / alpha)
ln_B = np.log(beta / (1 - alpha))
log_p1_p0 = np.log(p1 / p0)
log_1p1_1p0 = np.log((1 - p1) / (1 - p0))

# --- 仿真函数 (保持不变) ---
def run_sprt_simulation(true_p):
    log_lambda = 0
    for m in range(1, max_samples + 1):
        is_defective = random.random() < true_p
        if is_defective:
            log_lambda += log_p1_p0
        else:
            log_lambda += log_1p1_1p0
        if log_lambda >= ln_A:
            return "拒绝", m
        if log_lambda <= ln_B:
            return "接受", m
    return "达到最大样本量", max_samples

# --- 新增：系统性性能分析 ---
def comprehensive_analysis(num_simulations_per_point=2000):
    # 创建一个包含多个真实次品率p的数组
    true_p_values = np.linspace(0.01, 0.35, 35) # 测试从1%到35%的多种情况
    
    acceptance_rates = []
    avg_sample_numbers = []
    
    print("--- 开始进行系统性性能分析 ---")
    for i, p in enumerate(true_p_values):
        results = []
        sample_sizes = []
        for _ in range(num_simulations_per_point):
            result, n = run_sprt_simulation(p)
            results.append(result)
            sample_sizes.append(n)
            
        acceptance_rate = results.count("接受") / len(results)
        avg_sample_number = np.mean(sample_sizes)
        
        acceptance_rates.append(acceptance_rate)
        avg_sample_numbers.append(avg_sample_number)
        
        # 打印进度
        print(f"进度: {i+1}/{len(true_p_values)}, p={p:.2f}, Pa={acceptance_rate:.2%}, ASN={avg_sample_number:.2f}")

    return true_p_values, acceptance_rates, avg_sample_numbers

# --- 执行分析并绘图 ---
p_values, pa_values, asn_values = comprehensive_analysis()

# 创建图表
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), sharex=True)
plt.rcParams['font.sans-serif'] = ['SimHei'] # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False # 用来正常显示负号

# 绘制 OC 曲线
ax1.plot(p_values, pa_values, 'b-o', label='OC 曲线')
ax1.set_title('抽检特性曲线 (OC Curve)', fontsize=16)
ax1.set_ylabel('接受概率 (Pa)', fontsize=12)
ax1.grid(True, linestyle='--', alpha=0.6)
# 标注关键设计点
ax1.axhline(y=1-alpha, color='g', linestyle='--', label=f'1-α = {1-alpha:.2f}')
ax1.axvline(x=p0, color='g', linestyle='--')
ax1.axhline(y=beta, color='r', linestyle='--', label=f'β = {beta:.2f}')
ax1.axvline(x=p1, color='r', linestyle='--')
ax1.legend()

# 绘制 ASN 曲线
ax2.plot(p_values, asn_values, 'm-s', label='ASN 曲线')
ax2.set_title('平均样本数曲线 (ASN Curve)', fontsize=16)
ax2.set_xlabel('货物的真实次品率 (p)', fontsize=12)
ax2.set_ylabel('平均抽样数 (ASN)', fontsize=12)
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.legend()

plt.tight_layout()
plt.show()