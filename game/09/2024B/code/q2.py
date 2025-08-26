# -*- coding: utf-8 -*-
"""
最终版 - 问题二决策模型 (集成MCDA分析)

本代码旨在解决生产过程中的最优决策问题。其核心流程如下：
1.  定义6个不同的生产情境。
2.  严格枚举所有16种完备的生产策略。
3.  对每个情境，遍历所有策略：
    a. (第一层) 计算基础性能指标：期望利润、利润标准差、市场次品率等。
    b. (第二层) 基于第一层结果，进行数据归一化，并计算不同企业画像下的综合得分。
4.  输出三份核心结果文件：
    - 每个情境独立的、包含第一层和第二层全部分析数据的CSV文件。
    - 一个高度汇总的、基于不同决策依据的最优策略推荐CSV文件。
    - 一个将所有情境的详细数据合并在一起的总览CSV文件。
"""

import math
import itertools
import pandas as pd
import numpy as np
import os

# -----------------------------
# 数据：题目表1的 6 个情境
# -----------------------------
SCENARIOS = [
    {
        "情境": 1,
        "r1": 0.10,
        "p1": 4,
        "c1": 2,
        "r2": 0.10,
        "p2": 18,
        "c2": 3,
        "rf": 0.10,
        "a": 6,
        "cf": 3,
        "s": 56,
        "L": 6,
        "d": 5,
    },
    {
        "情境": 2,
        "r1": 0.20,
        "p1": 4,
        "c1": 2,
        "r2": 0.20,
        "p2": 18,
        "c2": 3,
        "rf": 0.20,
        "a": 6,
        "cf": 3,
        "s": 56,
        "L": 6,
        "d": 5,
    },
    {
        "情境": 3,
        "r1": 0.10,
        "p1": 4,
        "c1": 2,
        "r2": 0.10,
        "p2": 18,
        "c2": 3,
        "rf": 0.10,
        "a": 6,
        "cf": 3,
        "s": 56,
        "L": 30,
        "d": 5,
    },
    {
        "情境": 4,
        "r1": 0.20,
        "p1": 4,
        "c1": 1,
        "r2": 0.20,
        "p2": 18,
        "c2": 1,
        "rf": 0.20,
        "a": 6,
        "cf": 2,
        "s": 56,
        "L": 30,
        "d": 5,
    },
    {
        "情境": 5,
        "r1": 0.10,
        "p1": 4,
        "c1": 8,
        "r2": 0.20,
        "p2": 18,
        "c2": 1,
        "rf": 0.10,
        "a": 6,
        "cf": 2,
        "s": 56,
        "L": 10,
        "d": 5,
    },
    {
        "情境": 6,
        "r1": 0.05,
        "p1": 4,
        "c1": 2,
        "r2": 0.05,
        "p2": 18,
        "c2": 3,
        "rf": 0.05,
        "a": 6,
        "cf": 3,
        "s": 56,
        "L": 10,
        "d": 40,
    },
]


# -----------------------------
# 枚举16种完备策略
# -----------------------------
def enumerate_methods():
    methods = []
    m_id = 1
    for i1, i2, ip in itertools.product([0, 1], [0, 1], [0, 1]):
        if ip == 0:
            combos = [(1, "退货拆解"), (0, "退货报废")]
        else:
            combos = [(1, "厂内拆解"), (0, "厂内报废")]
        for dis, dis_name in combos:
            methods.append(
                {
                    "方法": f"M{m_id:02d}",
                    "i1": i1,
                    "i2": i2,
                    "ip": ip,
                    "dis": dis,
                    "检零件1": "检" if i1 else "不检",
                    "检零件2": "检" if i2 else "不检",
                    "检成品": "检" if ip else "不检",
                    "处理方式": dis_name,
                }
            )
            m_id += 1
    return methods


# -----------------------------
# 成本与利润模型（*** 已增加利润标准差的计算 ***）
# -----------------------------
def calculate_performance(sc, i1, i2, ip, dis):
    r1, p1, c1 = sc["r1"], sc["p1"], sc["c1"]
    r2, p2, c2 = sc["r2"], sc["p2"], sc["c2"]
    rf, a, cf = sc["rf"], sc["a"], sc["cf"]
    s, L, d = sc["s"], sc["L"], sc["d"]

    # 步骤1: 计算有效次品率和期望上游成本
    if i1 == 0:
        r1_eff, c_upstream1 = r1, p1
    else:
        r1_eff, c_upstream1 = 0, (p1 + c1) / (1 - r1) if r1 < 1 else math.inf
    if i2 == 0:
        r2_eff, c_upstream2 = r2, p2
    else:
        r2_eff, c_upstream2 = 0, (p2 + c2) / (1 - r2) if r2 < 1 else math.inf
    c_upstream = c_upstream1 + c_upstream2

    # 步骤2: 计算单次装配尝试的总成功率
    p_parts_defect = 1 - (1 - r1_eff) * (1 - r2_eff)
    p_total_good = (1 - p_parts_defect) * (1 - rf)
    p_total_defective = 1 - p_total_good

    if p_total_good <= 1e-9:
        return math.inf, -math.inf, p_total_defective, 0

    # 步骤3: 精确计算期望回收价值
    P_A = (1 - r1_eff) * r2_eff
    P_B = r1_eff * (1 - r2_eff)
    P_D1 = (1 - r1_eff) * (1 - r2_eff) * rf
    P_p1_ok_and_def = P_A + P_D1
    P_p2_ok_and_def = P_B + P_D1
    P_p1_ok_given_def = (
        P_p1_ok_and_def / p_total_defective if p_total_defective > 0 else 0
    )
    P_p2_ok_given_def = (
        P_p2_ok_and_def / p_total_defective if p_total_defective > 0 else 0
    )
    v_salvage = P_p1_ok_given_def * p1 + P_p2_ok_given_def * p2

    # 步骤4: 计算单位合格品期望总成本和市场次品率
    c_try_base = c_upstream + a
    market_defect_rate = 0
    if ip == 0:  # 成品不检
        c_handle_market = (d - v_salvage) if dis == 1 else 0
        c_try_total = c_try_base + p_total_defective * (L + c_handle_market)
        market_defect_rate = p_total_defective
    else:  # 成品检测
        c_handle_factory = (d - v_salvage) if dis == 1 else 0
        c_try_total = c_try_base + cf + p_total_defective * c_handle_factory

    e_cost_per_good = c_try_total / p_total_good
    e_profit = s - e_cost_per_good

    # 步骤5: 估算利润标准差 (主要风险源于市场调换损失L)
    std_profit = L * math.sqrt(market_defect_rate * (1 - market_defect_rate))

    return e_cost_per_good, e_profit, market_defect_rate, std_profit


# -----------------------------
# 主流程
# -----------------------------
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "..", "data", "q2")
    os.makedirs(output_dir, exist_ok=True)
    print(f"--- 所有输出文件将被保存到目录: {os.path.abspath(output_dir)} ---")

    methods_df = pd.DataFrame(enumerate_methods())

    # 用于存储最终结果
    all_scenarios_data = []
    best_strategies_summary = []

    # 定义企业画像
    personas = {
        "激进增长型": {"w_profit": 0.6, "w_stability": 0.2, "w_reputation": 0.2},
        "稳健品牌型": {"w_profit": 0.3, "w_stability": 0.3, "w_reputation": 0.4},
    }

    print("--- 正在执行第一层与第二层集成分析 ---")

    for sc in SCENARIOS:
        # --- 第一层: 基础性能计算 ---
        results = []
        for _, m in methods_df.iterrows():
            cost, profit, defect, std = calculate_performance(
                sc, m["i1"], m["i2"], m["ip"], m["dis"]
            )
            results.append(
                {
                    "期望成本": cost,
                    "期望利润": profit,
                    "市场次品率": defect,
                    "利润标准差": std,
                }
            )

        df_scenario = pd.concat([methods_df.copy(), pd.DataFrame(results)], axis=1)
        df_scenario["情境"] = sc["情境"]  # 增加情境标识

        df_scenario = df_scenario[np.isfinite(df_scenario["期望利润"])].copy()
        if df_scenario.empty:
            continue

        # --- 第二层: MCDA评分 ---
        df_scenario["归一化_利润"] = (
            (df_scenario["期望利润"] - df_scenario["期望利润"].min())
            / (df_scenario["期望利润"].max() - df_scenario["期望利润"].min())
            if df_scenario["期望利润"].nunique() > 1
            else 0
        )
        df_scenario["归一化_稳定性"] = (
            (df_scenario["利润标准差"].max() - df_scenario["利润标准差"])
            / (df_scenario["利润标准差"].max() - df_scenario["利润标准差"].min())
            if df_scenario["利润标准差"].nunique() > 1
            else 0
        )
        df_scenario["归一化_声誉"] = (
            1
            if df_scenario["市场次品率"].max() == 0
            else (df_scenario["市场次品率"].max() - df_scenario["市场次品率"])
            / df_scenario["市场次品率"].max()
        )

        for name, weights in personas.items():
            df_scenario[f"得分_{name}"] = (
                weights["w_profit"] * df_scenario["归一化_利润"]
                + weights["w_stability"] * df_scenario["归一化_稳定性"]
                + weights["w_reputation"] * df_scenario["归一化_声誉"]
            )

        # --- 结果导出与汇总 ---
        df_scenario = df_scenario.sort_values("期望利润", ascending=False).reset_index(
            drop=True
        )

        # 导出包含所有分析结果的CSV
        csv_filename = f'情景_{sc["情境"]}_全部分析结果.csv'
        full_path = os.path.join(output_dir, csv_filename)
        df_scenario.drop(columns=["i1", "i2", "ip", "dis"]).to_csv(
            full_path, index=False, encoding="utf-8-sig", float_format="%.3f"
        )
        print(f"情景 {sc['情境']} 计算完成，全部分析结果已导出至 -> {full_path}")

        all_scenarios_data.append(df_scenario)

        # --- 汇总最优策略 ---
        best_profit = df_scenario.iloc[0]
        best_strategies_summary.append(
            {
                "情境": sc["情境"],
                "决策依据": "纯利润最大化",
                "方法": best_profit["方法"],
                "具体策略": f'{best_profit["检零件1"]}/{best_profit["检零件2"]}/{best_profit["检成品"]}/{best_profit["处理方式"]}',
                "期望利润": best_profit["期望利润"],
                "市场次品率": best_profit["市场次品率"],
                "利润标准差": best_profit["利润标准差"],
            }
        )
        for name in personas.keys():
            best_persona = df_scenario.loc[df_scenario[f"得分_{name}"].idxmax()]
            best_strategies_summary.append(
                {
                    "情境": sc["情境"],
                    "决策依据": name,
                    "方法": best_persona["方法"],
                    "具体策略": f'{best_persona["检零件1"]}/{best_persona["检零件2"]}/{best_persona["检成品"]}/{best_persona["处理方式"]}',
                    "期望利润": best_persona["期望利润"],
                    "市场次品率": best_persona["市场次品率"],
                    "利润标准差": best_persona["利润标准差"],
                }
            )

    # --- 创建并导出最终的汇总文件到指定目录 ---
    # 1. 最优策略汇总
    df_summary = pd.DataFrame(best_strategies_summary)
    summary_path = os.path.join(output_dir, "各情景_最优策略汇总.csv")
    df_summary.to_csv(
        summary_path, index=False, encoding="utf-8-sig", float_format="%.3f"
    )

    # 2. 全部策略数据汇总
    df_all_combined = pd.concat(all_scenarios_data, ignore_index=True)
    combined_path = os.path.join(output_dir, "全部策略_六情境_详细数据_汇总版.csv")
    df_all_combined.drop(columns=["i1", "i2", "ip", "dis"]).to_csv(
        combined_path, index=False, encoding="utf-8-sig", float_format="%.3f"
    )

    # --- 终端输出汇总报告 ---
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.unicode.east_asian_width", True)
    print("\n\n" + "=" * 90)
    print("各情景最优策略汇总报告")
    print("=" * 90)
    print(df_summary.to_string(index=False))

    # *** --- 修改：更新最终的提示信息 --- ***
    print("\n\n已生成所有文件：")
    for i in range(1, len(SCENARIOS) + 1):
        # 构造每个文件的完整路径用于显示
        file_path = os.path.join(output_dir, f"情景_{i}_全部分析结果.csv")
        print(f" - {file_path}")
    print(f" - {summary_path}")
    print(f" - {combined_path}")


if __name__ == "__main__":
    main()
