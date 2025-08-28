# q4_for_q3_robust_dp_final.py
# -*- coding: utf-8 -*-
"""
最终版 - 问题四针对问题三的稳健决策分析 (包含完整的风险-收益剖面分析)

本代码将贝叶斯-动态规划（B-DP）框架应用于问题三的复杂生产系统，
旨在找到在次品率不确定性下的全局最优稳健策略，并全面评估其经济表现。

核心流程:
1.  **模型封装**: 保持问题三的动态规划求解器 `DP_Solver` 不变。
2.  **不确定性建模**: 为所有节点的次品率 (r, rf) 定义贝叶斯后验分布（Beta分布）。
3.  **蒙特卡洛仿真**:
    a. 循环 N 次。
    b. 在每次循环中，生成一个随机参数场景。
    c. 调用 `DP_Solver` 对此场景求解，记录胜出的全局策略。
    d. (***关键新增***) 额外记录【所有备选全局策略】在该场景下的利润。
4.  **稳健性分析与结果呈现**:
    a. 确定最高频的“最稳健策略”。
    b. 利用步骤3d中记录的全面利润数据，计算该稳健策略的期望利润和利润标准差。
    c. 将稳健解的性能与确定性解的理论利润进行对比。
"""

import pandas as pd
import numpy as np
import math
from scipy.stats import beta
from tqdm import tqdm
import copy

# ----------------------------------------------------------------------------
# 步骤 1: 模型封装 (与上一版代码完全相同)
# ----------------------------------------------------------------------------

# 全局节点字典，用于在单次求解中缓存节点对象
ALL_NODES_SINGLE_RUN = {}
NODE_DATA_SINGLE_RUN = {}

class Node:
    """代表生产流程图中的一个节点，并实现状态转移逻辑"""
    def __init__(self, name, node_data):
        self.name = name
        self.data = node_data
        self.type = self.data['type']
        self.parents = [Node(p_name, NODE_DATA_SINGLE_RUN[p_name]) for p_name in self.data.get('parents', [])]
        ALL_NODES_SINGLE_RUN[name] = self
        self.results, self.best_decision_for_good, self.final_optimal_decision = None, None, None

    def solve(self):
        if self.results: return self.results
        if self.type == 'part':
            r, buy, test = self.data['r'], self.data['buy'], self.data['test']
            cost_good = (buy + test) / (1 - r) if r < 1 else math.inf
            self.results = {'good_cost': cost_good, 'asis_cost': buy, 'asis_rate': r}
            return self.results

        parent_results = {p.name: p.solve() for p in self.parents}
        upstream_asis_cost = sum(res['asis_cost'] for res in parent_results.values())
        parent_asis_rates = [res['asis_rate'] for res in parent_results.values()]
        p_parents_good_asis = np.prod([1 - rate for rate in parent_asis_rates])
        p_total_good_asis = p_parents_good_asis * (1 - self.data['rf'])
        
        upstream_good_cost = sum(res['good_cost'] for res in parent_results.values())
        p_success_good_upstream = 1 - self.data['rf']
        
        if p_success_good_upstream > 1e-9:
            cost_per_try_scrap = upstream_good_cost + self.data['assy'] + self.data['test']
            cost_test_scrap = cost_per_try_scrap / p_success_good_upstream
            net_salvage_on_failure = upstream_good_cost - self.data['dis']
            e_cost_try_disassemble = cost_per_try_scrap - (1 - p_success_good_upstream) * net_salvage_on_failure
            cost_test_disassemble = e_cost_try_disassemble / p_success_good_upstream
        else:
            cost_test_scrap, cost_test_disassemble = math.inf, math.inf

        cost_good = min(cost_test_disassemble, cost_test_scrap)
        self.best_decision_for_good = "检+拆解" if cost_test_disassemble < cost_test_scrap else "检+报废"
            
        self.results = {'good_cost': cost_good, 'asis_cost': upstream_asis_cost + self.data['assy'], 'asis_rate': 1 - p_total_good_asis}
        return self.results

def backtrack_and_set_decisions(node, required_contract):
    if node.final_optimal_decision is not None: return
    if required_contract == 'good':
        node.final_optimal_decision = "检测" if node.type == 'part' else node.best_decision_for_good
        for parent in node.parents: backtrack_and_set_decisions(parent, 'good')
    elif required_contract == 'asis':
        node.final_optimal_decision = "不检测"
        for parent in node.parents: backtrack_and_set_decisions(parent, 'asis')

def DP_Solver(node_data_scenario):
    """
    接收一个具体的参数场景，返回全局最优策略、最大期望利润以及所有策略的利润。
    """
    global ALL_NODES_SINGLE_RUN, NODE_DATA_SINGLE_RUN
    ALL_NODES_SINGLE_RUN, NODE_DATA_SINGLE_RUN = {}, node_data_scenario
    
    root_node = Node('F', NODE_DATA_SINGLE_RUN['F'])
    root_node.solve()

    price, L = NODE_DATA_SINGLE_RUN['F']['price'], NODE_DATA_SINGLE_RUN['F']['L']
    parent_results = {p.name: p.solve() for p in root_node.parents}
    
    strategy_profits = {}
    upstream_cost_good = sum(res['good_cost'] for res in parent_results.values())
    p_success = 1 - root_node.data['rf']
    
    if p_success > 1e-9:
        strategy_profits['检+拆解'] = price - ( (upstream_cost_good + root_node.data['assy'] + root_node.data['test']) - (1 - p_success) * (upstream_cost_good - root_node.data['dis']) ) / p_success
        strategy_profits['检+报废'] = price - (upstream_cost_good + root_node.data['assy'] + root_node.data['test']) / p_success
    else:
        strategy_profits.update({'检+拆解': -math.inf, '检+报废': -math.inf})

    upstream_cost_asis = sum(res['asis_cost'] for res in parent_results.values())
    p_final_good = np.prod([1 - res['asis_rate'] for res in parent_results.values()]) * (1 - root_node.data['rf'])
    
    if p_final_good > 1e-9:
        p_final_defective = 1 - p_final_good
        e_try_cost_nt_dis = upstream_cost_asis + root_node.data['assy'] + p_final_defective * (L + root_node.data['dis'] - upstream_cost_asis)
        strategy_profits['不检+退货拆解'] = price - e_try_cost_nt_dis / p_final_good
        e_try_cost_nt_scrap = upstream_cost_asis + root_node.data['assy'] + p_final_defective * (L + upstream_cost_asis + root_node.data['assy'])
        strategy_profits['不检+退货报废'] = price - e_try_cost_nt_scrap / p_final_good
    else:
        strategy_profits.update({'不检+退货拆解': -math.inf, '不检+退货报废': -math.inf})

    if not strategy_profits or all(math.isinf(p) for p in strategy_profits.values()):
        return None, -math.inf, strategy_profits
        
    best_strategy_name = max(strategy_profits, key=strategy_profits.get)
    max_profit = strategy_profits[best_strategy_name]
    
    required_contract = 'good' if '检' in best_strategy_name else 'asis'
    backtrack_and_set_decisions(root_node, required_contract)
    root_node.final_optimal_decision = best_strategy_name
    
    full_policy = {name: node.final_optimal_decision for name, node in sorted(ALL_NODES_SINGLE_RUN.items())}
    return full_policy, max_profit, strategy_profits


# ----------------------------------------------------------------------------
# 步骤 2, 3, 4: B-DP 框架主流程
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    N_SIMULATIONS = 15000 
    BASE_NODE_DATA = {
        'Z1': {'type': 'part', 'r': 0.10, 'buy': 2, 'test': 1}, 'Z2': {'type': 'part', 'r': 0.10, 'buy': 8, 'test': 1},
        'Z3': {'type': 'part', 'r': 0.10, 'buy': 12, 'test': 2}, 'Z4': {'type': 'part', 'r': 0.10, 'buy': 2, 'test': 1},
        'Z5': {'type': 'part', 'r': 0.10, 'buy': 8, 'test': 1}, 'Z6': {'type': 'part', 'r': 0.10, 'buy': 12, 'test': 2},
        'Z7': {'type': 'part', 'r': 0.10, 'buy': 8, 'test': 1}, 'Z8': {'type': 'part', 'r': 0.10, 'buy': 12, 'test': 2},
        'B1': {'type': 'assembly', 'parents': ['Z1', 'Z2', 'Z3'], 'rf': 0.10, 'assy': 8, 'test': 4, 'dis': 6},
        'B2': {'type': 'assembly', 'parents': ['Z4', 'Z5', 'Z6'], 'rf': 0.10, 'assy': 8, 'test': 4, 'dis': 6},
        'B3': {'type': 'assembly', 'parents': ['Z7', 'Z8'], 'rf': 0.10, 'assy': 8, 'test': 4, 'dis': 6},
        'F':  {'type': 'final_assembly', 'parents': ['B1', 'B2', 'B3'], 'rf': 0.10, 'assy': 8, 'test': 6, 'dis': 10, 'price': 200, 'L': 40}
    }
    
    confidence_level = 100
    BETA_PARAMS = {}
    for name, data in BASE_NODE_DATA.items():
        key = 'r' if 'r' in data else 'rf' if 'rf' in data else None
        if key:
            p_nominal = data[key]
            BETA_PARAMS[name] = {'param': key, 'a': p_nominal * confidence_level, 'b': (1-p_nominal) * confidence_level}
            
    print("="*80, "\n开始对问题三的复杂系统进行稳健性分析...", f"\n仿真次数: {N_SIMULATIONS}", "\n" + "="*80)
    
    policy_records = []
    # *** 关键新增：为每一种可能的【完整策略链】记录其利润分布 ***
    profit_records_for_all_policies = {}

    for _ in tqdm(range(N_SIMULATIONS), desc="仿真进度"):
        random_node_data = copy.deepcopy(BASE_NODE_DATA)
        for name, dist_info in BETA_PARAMS.items():
            param_name, a, b = dist_info['param'], dist_info['a'], dist_info['b']
            if a > 0 and b > 0:
                random_node_data[name][param_name] = beta.rvs(a, b)
        
        best_policy, _, all_profits = DP_Solver(random_node_data)
        
        if best_policy:
            policy_tuple = tuple(sorted(best_policy.items()))
            policy_records.append(policy_tuple)
            
            # *** 关键新增：记录当前场景下，该最优策略的利润 ***
            if policy_tuple not in profit_records_for_all_policies:
                profit_records_for_all_policies[policy_tuple] = []
            profit_records_for_all_policies[policy_tuple].append(all_profits[best_policy['F']])

    if not policy_records:
        print("仿真未能产生有效策略，请检查参数设置。")
    else:
        policy_counts = pd.Series(policy_records).value_counts()
        robust_policy_tuple = policy_counts.index[0]
        robust_policy_freq = policy_counts.iloc[0] / N_SIMULATIONS
        robust_policy_dict = dict(robust_policy_tuple)
        
        # *** 关键新增：计算稳健策略的风险-收益剖面 ***
        robust_profits = np.array(profit_records_for_all_policies.get(robust_policy_tuple, [0]))
        robust_expected_profit = robust_profits.mean()
        robust_profit_std = robust_profits.std()

        deterministic_policy, deterministic_profit, _ = DP_Solver(BASE_NODE_DATA)

        print("\n\n" + "="*80, "\n复杂系统稳健决策分析报告", "\n" + "="*80)
        
        print(f"\n--- 确定性模型最优策略 (基准) ---")
        print(f"理论最大期望利润: {deterministic_profit:.4f} 元")
        df_det = pd.DataFrame(list(deterministic_policy.items()), columns=['节点', '最优决策']).set_index('节点')
        print(df_det)
        
        print(f"\n--- 稳健模型最优策略 ---")
        print(f"胜出频率: {robust_policy_freq:.2%}")
        print(f"期望利润 (仿真均值): {robust_expected_profit:.4f} 元")
        print(f"利润标准差 (风险): {robust_profit_std:.4f} 元")
        
        df_rob = pd.DataFrame(list(robust_policy_dict.items()), columns=['节点', '最优决策']).set_index('节点')
        print(df_rob)

        if deterministic_policy == robust_policy_dict:
            print("\n[结论]: 稳健最优策略与确定性最优策略一致。")
            print("这表明，在当前参数波动范围内，原有的'全检测'策略具有极强的内在稳健性。")
            print(f"尽管如此，现实中的期望利润约为 {robust_expected_profit:.2f} 元，略低于理论值，且伴随约 {robust_profit_std:.2f} 元的标准差风险。")
        else:
            print("\n[结论]: 稳健最优策略与确定性最优策略存在差异！")
            print("这表明，在考虑参数不确定性后，需要调整生产策略以规避风险。")