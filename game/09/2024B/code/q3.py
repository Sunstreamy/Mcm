# -*- coding: utf-8 -*-
"""
1.  DP逆向递推部分保持不变，其核心任务是为每个节点计算出包含两种“契约”的“成本菜单”
    (C_good 和 C_asis)，这部分逻辑是正确的。
2.  最终的全局决策与回溯逻辑被完全重构，以修复之前的根本性错误。
3.  顶层决策（在成品节点）不再是简单的宏观比较，而是显式地枚举成品本身所有可能的
    决策选项（检+拆解, 检+报废, 不检+退货拆解, 不检+退货报废）。
4.  为每一种成品决策，都从其上游节点的“菜单”中选择最优的“契约”来计算总成本。
5.  找出成本最低（利润最高）的成品决策作为全局最优解的起点。
6.  基于这个唯一正确的顶层决策，发起正确的回溯“订单”，构建全局最优策略链。
"""
import pandas as pd
import numpy as np
import math

# --- 步骤 1: 定义所有节点的数据 (来自题目表2) ---
# (与之前版本完全相同)
NODE_DATA = {
    'Z1': {'type': 'part', 'r': 0.10, 'buy': 2, 'test': 1}, 'Z2': {'type': 'part', 'r': 0.10, 'buy': 8, 'test': 1},
    'Z3': {'type': 'part', 'r': 0.10, 'buy': 12, 'test': 2}, 'Z4': {'type': 'part', 'r': 0.10, 'buy': 2, 'test': 1},
    'Z5': {'type': 'part', 'r': 0.10, 'buy': 8, 'test': 1}, 'Z6': {'type': 'part', 'r': 0.10, 'buy': 12, 'test': 2},
    'Z7': {'type': 'part', 'r': 0.10, 'buy': 8, 'test': 1}, 'Z8': {'type': 'part', 'r': 0.10, 'buy': 12, 'test': 2},
    'B1': {'type': 'assembly', 'parents': ['Z1', 'Z2', 'Z3'], 'rf': 0.10, 'assy': 8, 'test': 4, 'dis': 6},
    'B2': {'type': 'assembly', 'parents': ['Z4', 'Z5', 'Z6'], 'rf': 0.10, 'assy': 8, 'test': 4, 'dis': 6},
    'B3': {'type': 'assembly', 'parents': ['Z7', 'Z8'], 'rf': 0.10, 'assy': 8, 'test': 4, 'dis': 6},
    'F':  {'type': 'final_assembly', 'parents': ['B1', 'B2', 'B3'], 'rf': 0.10, 'assy': 8, 'test': 6, 'dis': 10, 'price': 200, 'L': 40}
}
ALL_NODES = {}

# --- 步骤 2: 创建核心的Node类 (与之前版本完全相同) ---
class Node:
    """代表生产流程图中的一个节点，并实现状态转移逻辑"""
    def __init__(self, name):
        if name in ALL_NODES: return
        self.name = name
        self.data = NODE_DATA[name]
        self.type = self.data['type']
        self.parents = [Node(p_name) for p_name in self.data.get('parents', [])]
        self.results = None
        self.best_decision_for_good = None
        self.final_optimal_decision = None
        ALL_NODES[name] = self

    def solve(self):
        if self.results: return self.results
        if self.type == 'part':
            r, buy, test = self.data['r'], self.data['buy'], self.data['test']
            cost_good = (buy + test) / (1 - r) if r < 1 else math.inf
            cost_asis = buy
            self.results = {'good_cost': cost_good, 'asis_cost': cost_asis, 'asis_rate': r}
            return self.results

        parent_results = {p.name: p.solve() for p in self.parents}
        upstream_asis_cost = sum(res['asis_cost'] for res in parent_results.values())
        cost_asis = upstream_asis_cost + self.data['assy']
        parent_asis_rates = [res['asis_rate'] for res in parent_results.values()]
        p_parents_good_asis = np.prod([1 - rate for rate in parent_asis_rates])
        p_total_good_asis = p_parents_good_asis * (1 - self.data['rf'])
        rate_asis = 1 - p_total_good_asis
        
        upstream_good_cost = sum(res['good_cost'] for res in parent_results.values())
        p_success_good_upstream = 1 - self.data['rf']
        
        if p_success_good_upstream <= 0:
            cost_test_scrap = math.inf
            cost_test_disassemble = math.inf
        else:
            cost_per_try_scrap = upstream_good_cost + self.data['assy'] + self.data['test']
            cost_test_scrap = cost_per_try_scrap / p_success_good_upstream
            
            v_salvage_good = upstream_good_cost
            net_salvage_on_failure = v_salvage_good - self.data['dis']
            e_cost_try_disassemble = cost_per_try_scrap - (1 - p_success_good_upstream) * net_salvage_on_failure
            cost_test_disassemble = e_cost_try_disassemble / p_success_good_upstream

        if cost_test_disassemble < cost_test_scrap:
            cost_good = cost_test_disassemble
            self.best_decision_for_good = "检+拆解"
        else:
            cost_good = cost_test_scrap
            self.best_decision_for_good = "检+报废"
            
        self.results = {'good_cost': cost_good, 'asis_cost': cost_asis, 'asis_rate': rate_asis}
        return self.results

def backtrack_and_set_decisions(node, required_contract):
    """递归回溯，为每个节点设置唯一的最终最优决策"""
    if node.final_optimal_decision is not None: return
    if required_contract == 'good':
        if node.type == 'part':
            node.final_optimal_decision = "检测"
        else:
            node.final_optimal_decision = node.best_decision_for_good
            for parent in node.parents: backtrack_and_set_decisions(parent, 'good')
    elif required_contract == 'asis':
        node.final_optimal_decision = "不检测"
        for parent in node.parents: backtrack_and_set_decisions(parent, 'asis')

# --- 步骤 3: 主执行流程 (全新重构，逻辑更清晰) ---
if __name__ == "__main__":
    print("--- 正在执行最终修正版DP求解器 ---")
    
    # 1. 逆向DP计算，为所有节点生成“成本菜单”
    root_node = Node('F')
    root_node.solve()

    # 2. 在顶层(成品节点)显式枚举所有全局策略，并计算其总成本/利润
    price = NODE_DATA['F']['price']
    L = NODE_DATA['F']['L']
    parent_results = {p.name: p.solve() for p in root_node.parents}
    
    # 初始化一个字典来存储所有全局策略的最终利润
    strategy_profits = {}

    # --- 全局策略 1 & 2: 成品经过检测，确保向市场提供“正品” ---
    # 此时，为降低风险，向上游索要的必然是“正品契约”
    upstream_cost_good = sum(res['good_cost'] for res in parent_results.values())
    p_success = 1 - root_node.data['rf']

    # 策略 1: 检+拆解 (使用DP内部为'good_cost'算出的最优决策)
    # 我们从菜单直接获取 C_good(F) 的值，因为它就是检+拆解和检+报废两者中的较优者
    cost_final_good = root_node.results['good_cost']
    # 重要的逻辑：best_decision_for_good 告诉我们C_good到底是由哪个策略产生的
    if root_node.best_decision_for_good == "检+拆解":
        strategy_profits['检+拆解'] = price - cost_final_good
        # 为了让比较更公平，我们也计算一下如果强制选报废的利润
        cost_final_good_scrap_alt = (upstream_cost_good + root_node.data['assy'] + root_node.data['test']) / p_success
        strategy_profits['检+报废'] = price - cost_final_good_scrap_alt
    else: # best_decision_for_good == "检+报废"
        strategy_profits['检+报废'] = price - cost_final_good
        # 计算强制选拆解的利润
        v_salvage_good = upstream_cost_good
        net_salvage_on_failure = v_salvage_good - root_node.data['dis']
        e_cost_try_disassemble = (upstream_cost_good + root_node.data['assy'] + root_node.data['test']) - (1-p_success) * net_salvage_on_failure
        cost_final_good_disassemble_alt = e_cost_try_disassemble / p_success
        strategy_profits['检+拆解'] = price - cost_final_good_disassemble_alt
    
    # --- 全局策略 3 & 4: 成品不检测，向市场提供“原样”产品 ---
    # 此时，从成本最优角度，向上游索要的应是“原样契约”
    upstream_cost_asis = sum(res['asis_cost'] for res in parent_results.values())
    
    # 计算在这种上游输入下，成品自身的成功率和次品率
    parent_asis_rates = [res['asis_rate'] for res in parent_results.values()]
    p_parents_good = np.prod([1 - rate for rate in parent_asis_rates])
    p_final_good = p_parents_good * (1 - root_node.data['rf'])
    
    if p_final_good > 1e-9:
        p_final_defective = 1 - p_final_good
        # 计算期望回收价值 (用于退货处理)
        v_salvage_asis = upstream_cost_asis # 近似处理
        
        # 策略 3: 不检+退货拆解
        market_loss_disassemble = L + root_node.data['dis'] - v_salvage_asis
        e_try_cost_nt_dis = upstream_cost_asis + root_node.data['assy'] + p_final_defective * market_loss_disassemble
        cost_final_nt_disassemble = e_try_cost_nt_dis / p_final_good
        strategy_profits['不检+退货拆解'] = price - cost_final_nt_disassemble
        
        # 策略 4: 不检+退货报废
        market_loss_scrap = L + (upstream_cost_asis + root_node.data['assy']) # 损失是L+全部投入
        e_try_cost_nt_scrap = upstream_cost_asis + root_node.data['assy'] + p_final_defective * market_loss_scrap
        cost_final_nt_scrap = e_try_cost_nt_scrap / p_final_good
        strategy_profits['不检+退货报废'] = price - cost_final_nt_scrap
    else:
        strategy_profits['不检+退货拆解'] = -math.inf
        strategy_profits['不检+退货报 Faroe'] = -math.inf

    # 3. 找出全局最优策略
    best_strategy_name = max(strategy_profits, key=strategy_profits.get)
    max_profit = strategy_profits[best_strategy_name]
    
    print(f"\n--- 全局最优策略分析 (显式比较) ---")
    print("所有全局策略的期望利润详情:")
    for strategy, profit in strategy_profits.items():
        print(f"  - {strategy}: {profit:.4f} 元")
        
    print(f"\n最优策略为: {best_strategy_name}")
    print(f"对应的最大期望利润: {max_profit:.2f} 元")
    
    # 4. 基于正确的全局最优策略，启动正确的回溯
    if '检' in best_strategy_name:
        required_final_contract = 'good'
    else:
        required_final_contract = 'asis'
    backtrack_and_set_decisions(root_node, required_final_contract)
    root_node.final_optimal_decision = best_strategy_name

    #    然后，调用回溯函数，让它来负责设置所有节点的决策，包括根节点
    backtrack_and_set_decisions(root_node, required_final_contract)

    #    在回溯完成后，手动校正根节点的决策名称（因为回溯函数内部对装配节点的决策是 '检+拆解' 或 '检+报废'）
    #    这一步是为了让最终表格的根节点决策名称与顶层分析的名称完全一致
    root_node.final_optimal_decision = best_strategy_name

    # 5. 整理并打印最终的决策与成本表格
    table_data = []
    for name, node in sorted(ALL_NODES.items()):
        table_data.append({
            "节点名称": name, "节点类型": node.type, "最优决策": node.final_optimal_decision,
            "C_good(正品成本)": node.results['good_cost'],
            "C_asis(原样成本)": node.results['asis_cost'],
        })
    results_df = pd.DataFrame(table_data).set_index("节点名称")
    
    print("\n--- 各节点最终决策与“成本菜单”详情 ---")
    pd.set_option('display.float_format', '{:.4f}'.format)
    print(results_df)