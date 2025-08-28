# q4_for_q2_robust_only.py
# -*- coding: utf-8 -*-
"""
最终版 - 问题四针对问题二的稳健决策分析 (聚焦稳健最优)

本代码将贝叶斯-动态规划（B-DP）框架应用于问题二的所有六个情境，
专注于确定在参数不确定性下的最稳健决策策略及其风险-收益剖面。

核心流程:
1.  **模型封装**: 保持问题二的核心求解器不变。
2.  **批量处理循环**: 遍历所有六个基础情境，并对每个情境执行独立的蒙特卡洛仿真。
3.  **不确定性建模**: 动态地为每个情境的次品率定义其贝叶斯后验分布（Beta分布）。
4.  **稳健性分析**:
    a. 通过“最大频率准则”确定每个情境的最稳健策略。
    b. 详细计算该稳健策略在该情境不确定性下的“风险-收益剖面”（期望利润、利润标准差、胜出频率等）。
5.  **结果导出**:
    a. 生成一份简洁、聚焦的CSV文件，仅包含每个情境下的稳健最优策略及其性能指标。
    b. 文件将保存在 data/q4/ 目录下。
"""

import math
import itertools
import pandas as pd
import numpy as np
import os
from scipy.stats import beta
from tqdm import tqdm

# -----------------------------
# 数据：题目表1的 6 个情境
# -----------------------------
SCENARIOS = [
    {
        "情境": 1, "r1": 0.10, "p1": 4, "c1": 2, "r2": 0.10, "p2": 18, "c2": 3,
        "rf": 0.10, "a": 6, "cf": 3, "s": 56, "L": 6, "d": 5,
    },
    {
        "情境": 2, "r1": 0.20, "p1": 4, "c1": 2, "r2": 0.20, "p2": 18, "c2": 3,
        "rf": 0.20, "a": 6, "cf": 3, "s": 56, "L": 6, "d": 5,
    },
    {
        "情境": 3, "r1": 0.10, "p1": 4, "c1": 2, "r2": 0.10, "p2": 18, "c2": 3,
        "rf": 0.10, "a": 6, "cf": 3, "s": 56, "L": 30, "d": 5,
    },
    {
        "情境": 4, "r1": 0.20, "p1": 4, "c1": 1, "r2": 0.20, "p2": 18, "c2": 1,
        "rf": 0.20, "a": 6, "cf": 2, "s": 56, "L": 30, "d": 5,
    },
    {
        "情境": 5, "r1": 0.10, "p1": 4, "c1": 8, "r2": 0.20, "p2": 18, "c2": 1,
        "rf": 0.10, "a": 6, "cf": 2, "s": 56, "L": 10, "d": 5,
    },
    {
        "情境": 6, "r1": 0.05, "p1": 4, "c1": 2, "r2": 0.05, "p2": 18, "c2": 3,
        "rf": 0.05, "a": 6, "cf": 3, "s": 56, "L": 10, "d": 40,
    },
]

# ----------------------------------------------------------------------------
# 步骤 1: 模型封装
# ----------------------------------------------------------------------------
def enumerate_methods():
    """枚举所有16种完备的生产策略。"""
    methods = []
    m_id = 1
    for i1, i2, ip in itertools.product([0, 1], [0, 1], [0, 1]):
        if ip == 0: combos = [(1, "退货拆解"), (0, "退货报废")]
        else: combos = [(1, "厂内拆解"), (0, "厂内报废")]
        for dis, dis_name in combos:
            methods.append({
                "方法": f"M{m_id:02d}", "i1": i1, "i2": i2, "ip": ip, "dis": dis,
                "检零件1": "检" if i1 else "不检", "检零件2": "检" if i2 else "不检",
                "检成品": "检" if ip else "不检", "处理方式": dis_name,
            })
            m_id += 1
    return pd.DataFrame(methods)

def calculate_performance(params, i1, i2, ip, dis):
    """根据给定的参数和策略，计算单位成品的期望利润。"""
    r1, p1, c1 = params["r1"], params["p1"], params["c1"]
    r2, p2, c2 = params["r2"], params["p2"], params["c2"]
    rf, a, cf = params["rf"], params["a"], params["cf"]
    s, L, d = params["s"], params["L"], params["d"]

    if i1 == 0: r1_eff, c_upstream1 = r1, p1
    else: r1_eff, c_upstream1 = 0, (p1 + c1) / (1 - r1) if r1 < 1 else math.inf
    if i2 == 0: r2_eff, c_upstream2 = r2, p2
    else: r2_eff, c_upstream2 = 0, (p2 + c2) / (1 - r2) if r2 < 1 else math.inf
    c_upstream = c_upstream1 + c_upstream2

    p_parts_defect = 1 - (1 - r1_eff) * (1 - r2_eff)
    p_total_good = (1 - p_parts_defect) * (1 - rf)
    
    if p_total_good <= 1e-9: return -math.inf

    if ip == 0:
        c_try_total = c_upstream + a + (1 - p_total_good) * L
        e_cost_per_good = c_try_total / p_total_good
    else:
        p_total_defective = 1 - p_total_good
        v_salvage = 0
        if p_total_defective > 0:
            P_A, P_B, P_D1 = (1-r1_eff)*r2_eff, r1_eff*(1-r2_eff), (1-r1_eff)*(1-r2_eff)*rf
            P_p1_ok_given_def, P_p2_ok_given_def = (P_A + P_D1)/p_total_defective, (P_B + P_D1)/p_total_defective
            v_salvage = P_p1_ok_given_def * p1 + P_p2_ok_given_def * p2
        c_handle_factory = (d - v_salvage) if dis == 1 else 0
        c_try_total = c_upstream + a + cf + p_total_defective * c_handle_factory
        e_cost_per_good = c_try_total / p_total_good
        
    return s - e_cost_per_good

def EMV_Solver_For_Profit_Only(params, methods_df):
    """一个简化的求解器，仅用于快速找到纯利润最大化的策略名称。"""
    best_profit, best_policy_name = -math.inf, None
    for _, m in methods_df.iterrows():
        profit = calculate_performance(params, m["i1"], m["i2"], m["ip"], m["dis"])
        if profit > best_profit:
            best_profit, best_policy_name = profit, m["方法"]
    return best_policy_name

# ----------------------------------------------------------------------------
# 步骤 2, 3, 4: 批量仿真与稳健性分析
# ----------------------------------------------------------------------------
def main():
    # --- 全局参数设定 ---
    N_SIMULATIONS = 15000
    
    # --- 路径设置 ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "..", "data", "q4")
    os.makedirs(output_dir, exist_ok=True)
    
    # --- 预先生成策略列表 ---
    methods_df = enumerate_methods()
    
    # --- 存储所有情境的最终分析结果 ---
    robust_summary_data = []

    print("="*80)
    print("开始对所有六个情境进行批量稳健性分析")
    print(f"每个情境的仿真次数: {N_SIMULATIONS}")
    print("="*80)

    # --- 批量处理循环 ---
    for base_scenario_params in tqdm(SCENARIOS, desc="整体进度", unit="情境"):
        sc_num = base_scenario_params["情境"]
        
        # --- 不确定性建模 ---
        confidence_level = 100 
        r1_base, r2_base, rf_base = base_scenario_params['r1'], base_scenario_params['r2'], base_scenario_params['rf']
        BETA_PARAMS = {
            'r1': {'a': r1_base * confidence_level, 'b': (1 - r1_base) * confidence_level},
            'r2': {'a': r2_base * confidence_level, 'b': (1 - r2_base) * confidence_level},
            'rf': {'a': rf_base * confidence_level, 'b': (1 - rf_base) * confidence_level},
        }

        # --- 对当前情境执行蒙特卡洛仿真 ---
        policy_records = []
        profit_records_for_all_policies = {m['方法']: [] for _, m in methods_df.iterrows()}

        for _ in range(N_SIMULATIONS):
            random_params = base_scenario_params.copy()
            for param, dist_params in BETA_PARAMS.items():
                if dist_params['a'] > 0 and dist_params['b'] > 0:
                    random_params[param] = beta.rvs(dist_params['a'], dist_params['b'])

            best_policy_name = EMV_Solver_For_Profit_Only(random_params, methods_df)
            if best_policy_name:
                policy_records.append(best_policy_name)
            
            for _, m in methods_df.iterrows():
                profit = calculate_performance(random_params, m["i1"], m["i2"], m["ip"], m["dis"])
                profit_records_for_all_policies[m['方法']].append(profit)
                
        # --- 稳健性分析 ---
        if not policy_records: continue
        
        policy_counts = pd.Series(policy_records).value_counts()
        robust_policy_name = policy_counts.index[0]
        robust_policy_freq = policy_counts.iloc[0] / N_SIMULATIONS
        
        robust_policy_profits = np.array(profit_records_for_all_policies[robust_policy_name])
        robust_policy_profits = robust_policy_profits[np.isfinite(robust_policy_profits)]
        robust_expected_profit = robust_policy_profits.mean() if len(robust_policy_profits) > 0 else -math.inf
        robust_profit_std = robust_policy_profits.std() if len(robust_policy_profits) > 0 else 0

        # --- 汇总结果 ---
        rob_row = methods_df[methods_df['方法'] == robust_policy_name].iloc[0]
        
        robust_summary_data.append({
            "情境": sc_num,
            "稳健最优方法": robust_policy_name,
            "具体策略": f"{rob_row['检零件1']}/{rob_row['检零件2']}/{rob_row['检成品']}/{rob_row['处理方式']}",
            "期望利润": robust_expected_profit,
            "利润标准差": robust_profit_std,
            "胜出频率": robust_policy_freq
        })

    # --- 结果导出与打印 ---
    df_final_summary = pd.DataFrame(robust_summary_data)
    
    # 导出到CSV
    output_filename = "q2_稳健最优策略汇总报告.csv"
    full_path = os.path.join(output_dir, output_filename)
    df_final_summary.to_csv(full_path, index=False, encoding="utf-8-sig", float_format="%.4f")

    # --- 终端输出汇总报告 ---
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.unicode.east_asian_width", True)
    print("\n\n" + "=" * 110)
    print("问题四：各情境稳健最优策略分析报告")
    print("=" * 110)
    print(df_final_summary.to_string(index=False))
    print("\n" + "=" * 110)
    print(f"\n报告已成功导出至: {os.path.abspath(full_path)}")


if __name__ == "__main__":
    main()