# -*- coding: utf-8 -*-

"""
翻译软件评价模型：问题一和问题二
本代码实现了评价翻译软件"双向翻译"效果的数学模型
选择的翻译软件：谷歌翻译、百度翻译、DeepL翻译
"""

import numpy as np
import jieba
import re
from collections import Counter
import matplotlib
import json
import os

matplotlib.rcParams["font.sans-serif"] = ["SimHei"]  # 用来正常显示中文标签
matplotlib.rcParams["axes.unicode_minus"] = False  # 用来正常显示负号


class TranslationEvaluator:
    """翻译评价模型类"""

    def __init__(self):
        """初始化评价模型"""
        # 定义评价指标权重 (AHP方法确定)
        # 语义保留度(SP)、语法正确性(GC)、表达流畅性(EF)、专业术语准确性(TA)
        self.weights = {
            "SP": 0.4,  # 语义保留度权重
            "GC": 0.25,  # 语法正确性权重
            "EF": 0.2,  # 表达流畅性权重
            "TA": 0.15,  # 专业术语准确性权重
        }

        # 定义翻译软件
        self.translators = ["Google", "Baidu", "DeepL"]

        # 停用词表（简化版）
        self.stopwords = set(
            [
                "的",
                "了",
                "和",
                "是",
                "在",
                "我",
                "有",
                "这",
                "个",
                "上",
                "下",
                "不",
                "以",
                "到",
                "与",
            ]
        )

    def preprocess_text(self, text):
        """
        文本预处理：分词、去停用词

        参数:
            text (str): 输入文本

        返回:
            list: 处理后的词列表
        """
        # 分词
        words = jieba.lcut(text)

        # 去停用词
        words = [
            word
            for word in words
            if word not in self.stopwords and len(word.strip()) > 0
        ]

        return words

    def calculate_cosine_similarity(self, text1, text2):
        """
        计算两段文本的余弦相似度

        参数:
            text1 (str): 第一段文本
            text2 (str): 第二段文本

        返回:
            float: 余弦相似度 [0,1]
        """
        # 文本预处理
        words1 = self.preprocess_text(text1)
        words2 = self.preprocess_text(text2)

        # 构建词频向量
        word_set = set(words1).union(set(words2))

        # 计算词频
        word_dict1 = Counter(words1)
        word_dict2 = Counter(words2)

        # 构建词向量
        vector1 = [word_dict1.get(word, 0) for word in word_set]
        vector2 = [word_dict2.get(word, 0) for word in word_set]

        # 计算余弦相似度
        dot_product = sum(a * b for a, b in zip(vector1, vector2))
        norm1 = sum(a * a for a in vector1) ** 0.5
        norm2 = sum(b * b for b in vector2) ** 0.5

        if norm1 == 0 or norm2 == 0:
            return 0

        return dot_product / (norm1 * norm2)

    def calculate_edit_distance(self, text1, text2):
        """
        计算两段文本的编辑距离（Levenshtein距离）

        参数:
            text1 (str): 第一段文本
            text2 (str): 第二段文本

        返回:
            float: 归一化的编辑距离相似度 [0,1]，值越大表示越相似
        """
        # 文本预处理
        words1 = self.preprocess_text(text1)
        words2 = self.preprocess_text(text2)

        # 计算编辑距离
        m, n = len(words1), len(words2)

        # 创建距离矩阵
        dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]

        # 初始化
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        # 动态规划计算编辑距离
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if words1[i - 1] == words2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1

        # 归一化编辑距离为相似度
        max_len = max(m, n)
        if max_len == 0:
            return 1.0  # 两个空文本视为完全相同

        # 将编辑距离转换为相似度，值越大表示越相似
        similarity = 1 - dp[m][n] / max_len
        return similarity

    def calculate_semantic_preservation(self, original_text, translated_text):
        """
        计算语义保留度

        参数:
            original_text (str): 原始文本
            translated_text (str): 双向翻译后的文本

        返回:
            float: 语义保留度得分 [0,1]
        """
        # 如果两个文本为空，返回中等分数
        if not original_text.strip() or not translated_text.strip():
            return 0.5

        try:
            # 结合余弦相似度和编辑距离计算语义保留度
            cosine_sim = self.calculate_cosine_similarity(
                original_text, translated_text
            )
            edit_sim = self.calculate_edit_distance(original_text, translated_text)

            # 加权平均
            semantic_score = 0.6 * cosine_sim + 0.4 * edit_sim

            # 防止返回零值（给一个最低基础分）
            if semantic_score < 0.2:
                semantic_score = 0.2

            return semantic_score
        except Exception as e:
            print(f"语义保留度计算错误: {e}")
            return 0.5  # 出错时返回中等分数

    def calculate_grammar_correctness(self, text):
        """
        计算语法正确性得分

        参数:
            text (str): 待评估文本

        返回:
            float: 语法正确性得分 [0,1]
        """
        # 判断语言类型（中文或英文）
        is_chinese = any("\u4e00" <= char <= "\u9fff" for char in text)

        if is_chinese:
            # 中文语法评估
            # 检查标点符号使用
            punctuation_errors = 0
            # 检查连续标点
            if re.search(r"[，。！？；：、]{2,}", text):
                punctuation_errors += 1
            # 检查句末标点
            if not re.search(r"[。！？]$", text.strip()):
                punctuation_errors += 1

            # 检查句子结构
            structure_errors = 0
            sentences = re.split(r"[。！？]", text)
            for sentence in sentences:
                if sentence and len(sentence) > 3:  # 忽略空句子和非常短的句子
                    # 检查是否有主谓结构
                    if not re.search(r"[^，；：、]+[是有在为][^，；：、]+", sentence):
                        structure_errors += 1

            # 计算得分
            total_sentences = len([s for s in sentences if s.strip()])
            if total_sentences == 0:
                return 0.7  # 默认较高分数

            # 归一化得分
            grammar_score = 1.0 - (punctuation_errors + structure_errors) / (
                2 * max(1, total_sentences)
            )
        else:
            # 英文语法评估
            # 检查标点符号使用
            punctuation_errors = 0
            # 检查连续标点
            if re.search(r"[,.!?;:]{2,}", text):
                punctuation_errors += 1
            # 检查句末标点
            if not re.search(r"[.!?]$", text.strip()):
                punctuation_errors += 1

            # 检查句子结构
            structure_errors = 0
            sentences = re.split(r"[.!?]", text)
            for sentence in sentences:
                if sentence and len(sentence) > 5:  # 忽略空句子和非常短的句子
                    # 英文中检查主语-谓语结构（简化检查）
                    words = sentence.strip().split()
                    if len(words) >= 2 and not re.search(
                        r"\b(is|are|was|were|have|has|had|do|does|did)\b", sentence
                    ):
                        structure_errors += 1

            # 计算得分
            total_sentences = len([s for s in sentences if s.strip()])
            if total_sentences == 0:
                return 0.7  # 默认较高分数

            # 归一化得分
            grammar_score = 1.0 - (punctuation_errors + structure_errors) / (
                2 * max(1, total_sentences)
            )

        return max(0, min(1, grammar_score))  # 确保得分在[0,1]范围内

    def calculate_expression_fluency(self, text):
        """
        计算表达流畅性得分

        参数:
            text (str): 待评估文本

        返回:
            float: 表达流畅性得分 [0,1]
        """
        # 判断语言类型（中文或英文）
        is_chinese = any("\u4e00" <= char <= "\u9fff" for char in text)

        if is_chinese:
            # 中文文本分割
            sentences = re.split(r"[。！？]", text)
            # 中文连接词
            connector_words = [
                "但是",
                "因此",
                "然而",
                "此外",
                "而且",
                "所以",
                "不过",
                "另外",
                "并且",
            ]
        else:
            # 英文文本分割
            sentences = re.split(r"[.!?]", text)
            # 英文连接词
            connector_words = [
                "but",
                "therefore",
                "however",
                "moreover",
                "although",
                "besides",
                "furthermore",
                "thus",
                "nevertheless",
            ]

        sentences = [s for s in sentences if s.strip()]

        if not sentences:
            return 0.7  # 默认较高分数

        # 1. 计算句子长度评分
        sentence_lengths = [len(s) for s in sentences]
        if not sentence_lengths:
            return 0.7

        avg_length = sum(sentence_lengths) / len(sentence_lengths)

        # 句子长度适中（10-30个字符）得高分
        # 对英文和中文使用不同的理想长度
        ideal_length = 20 if is_chinese else 15
        length_score = 1.0 - min(1, abs(avg_length - ideal_length) / ideal_length)

        # 2. 检查词语重复度
        words = self.preprocess_text(text)
        if not words:
            return 0.7  # 默认较高分数

        word_counts = Counter(words)
        # 计算重复率
        repetition_rate = 1.0 - len(word_counts) / max(1, len(words))

        # 重复率低得高分
        repetition_score = 1.0 - min(1, repetition_rate * 3)

        # 3. 句式多样性评估
        sentence_patterns = set()
        for sentence in sentences:
            if len(sentence) > 5:
                # 简化的句式模式提取（提取句子的首3个字和尾3个字的组合作为模式）
                if len(sentence) > 6:
                    pattern = sentence[:3] + sentence[-3:]
                else:
                    pattern = sentence
                sentence_patterns.add(pattern)

        # 句式多样性得分，句式种类越多得分越高
        pattern_diversity = min(1.0, len(sentence_patterns) / max(1, len(sentences)))

        # 4. 连接词使用评估
        connector_count = 0
        for word in connector_words:
            connector_count += text.count(word)

        # 连接词使用得分，使用适量的连接词得分高
        connector_score = min(1.0, connector_count / max(5, len(sentences) * 0.5))

        # 5. 句子长度变化评估（句子长度多样性）
        if len(sentence_lengths) > 1:
            # 计算句子长度的标准差，标准差越大表明句子长度变化越大
            mean_length = sum(sentence_lengths) / len(sentence_lengths)
            variance = sum((x - mean_length) ** 2 for x in sentence_lengths) / len(
                sentence_lengths
            )
            std_dev = variance**0.5

            # 适当的句子长度变化得高分
            length_diversity = min(1.0, std_dev / (ideal_length * 0.5))
        else:
            length_diversity = 0.5  # 只有一个句子时设为中等分数

        # 综合多个因素计算流畅性得分
        fluency_score = (
            0.3 * length_score
            + 0.3 * repetition_score
            + 0.15 * pattern_diversity
            + 0.15 * connector_score
            + 0.1 * length_diversity
        )

        return fluency_score

    def calculate_terminology_accuracy(
        self, original_text, translated_text, domain_terms=None
    ):
        """
        计算专业术语准确性得分

        参数:
            original_text (str): 原始文本
            translated_text (str): 双向翻译后的文本
            domain_terms (dict): 领域专业术语词典 {term: weight}

        返回:
            float: 专业术语准确性得分 [0,1]
        """
        # 如果没有提供专业术语词典，使用简化评估
        if domain_terms is None:
            # 默认专业术语词典（示例）
            domain_terms = {
                "数学建模": 1.5,
                "翻译软件": 1.5,
                "高考": 1.2,
                "评价": 1.2,
                "模型": 1.3,
                "英语": 1.2,
                "计算机": 1.2,
                "双向翻译": 1.5,
            }

        # 检查原文中的专业术语在翻译后文本中的保留情况
        original_words = self.preprocess_text(original_text)
        translated_words = self.preprocess_text(translated_text)

        # 统计原文中的专业术语
        original_terms = {}
        for term in domain_terms:
            if term in original_text:
                original_terms[term] = domain_terms[term]

        if not original_terms:
            return 0.7  # 如果没有专业术语，给予较高默认分数

        # 计算术语保留率
        preserved_terms = {}
        for term in original_terms:
            if term in translated_text:
                preserved_terms[term] = original_terms[term]

        # 计算加权术语保留率
        if sum(original_terms.values()) == 0:
            return 0.7

        weighted_preservation = sum(preserved_terms.values()) / sum(
            original_terms.values()
        )

        # 确保得分不会为零（给一个最低基础分）
        if weighted_preservation < 0.2:
            weighted_preservation = 0.2

        return weighted_preservation

    def evaluate_translation(self, original_text, translated_text, domain_terms=None):
        """
        综合评价翻译质量

        参数:
            original_text (str): 原始文本
            translated_text (str): 双向翻译后的文本
            domain_terms (dict): 专业术语词典，用于术语准确性评估

        返回:
            dict: 各项指标得分和总得分
        """
        try:
            # 计算各项指标得分
            sp_score = self.calculate_semantic_preservation(
                original_text, translated_text
            )
            gc_score = self.calculate_grammar_correctness(translated_text)
            ef_score = self.calculate_expression_fluency(translated_text)
            ta_score = self.calculate_terminology_accuracy(
                original_text, translated_text, domain_terms
            )

            # 防止任何指标为零（设置最低基础分）
            sp_score = max(0.2, sp_score)
            gc_score = max(0.2, gc_score)
            ef_score = max(0.2, ef_score)
            ta_score = max(0.2, ta_score)

            # 计算加权总分
            total_score = (
                self.weights["SP"] * sp_score
                + self.weights["GC"] * gc_score
                + self.weights["EF"] * ef_score
                + self.weights["TA"] * ta_score
            )

            # 返回评价结果
            return {
                "SP": sp_score,
                "GC": gc_score,
                "EF": ef_score,
                "TA": ta_score,
                "Total": total_score,
            }
        except Exception as e:
            print(f"评价翻译时出错: {e}")
            # 出错时返回默认评分
            return {
                "SP": 0.5,
                "GC": 0.5,
                "EF": 0.5,
                "TA": 0.5,
                "Total": 0.5,
            }

    def evaluate_translators(self, original_text, translated_texts, domain_terms=None):
        """
        评价多个翻译软件的翻译质量

        参数:
            original_text (str): 原始文本
            translated_texts (dict): 各翻译软件的双向翻译结果 {translator_name: translated_text}
            domain_terms (dict): 领域专业术语词典 {term: weight}

        返回:
            dict: 各翻译软件的评价结果
        """
        results = {}
        for translator, text in translated_texts.items():
            results[translator] = self.evaluate_translation(
                original_text, text, domain_terms
            )

        return results

    def rank_translators(self, evaluation_results):
        """
        对翻译软件进行排名

        参数:
            evaluation_results (dict): 评价结果 {translator_name: {metric: score}}

        返回:
            list: 按总分排序的翻译软件列表 [(translator_name, total_score)]
        """
        # 按总分排序
        ranked_translators = [
            (translator, results["Total"])
            for translator, results in evaluation_results.items()
        ]
        ranked_translators.sort(key=lambda x: x[1], reverse=True)

        return ranked_translators

    def fuzzy_comprehensive_evaluation(self, evaluation_results):
        """
        模糊综合评价

        参数:
            evaluation_results (dict): 评价结果 {translator_name: {metric: score}}

        返回:
            dict: 模糊综合评价结果
        """
        # 评价等级
        grades = ["优", "良", "中", "差"]

        # 构建模糊关系矩阵
        fuzzy_matrices = {}

        for translator, results in evaluation_results.items():
            # 构建单个翻译软件的模糊关系矩阵
            matrix = []

            # 对每个指标进行模糊评价
            for metric in ["SP", "GC", "EF", "TA"]:
                score = results[metric]

                # 使用连续的模糊隶属度函数，而不是离散的阈值
                # 这样即使分数相近，也能产生不同的模糊隶属度

                # 优的隶属度
                if score >= 0.9:
                    m_excellent = min(1.0, (score - 0.9) * 10 + 0.9)
                elif score >= 0.8:
                    m_excellent = (score - 0.8) * 7
                else:
                    m_excellent = 0

                # 良的隶属度
                if score >= 0.8 and score < 0.9:
                    m_good = min(1.0, (0.9 - score) * 10 + 0.3)
                elif score >= 0.7 and score < 0.8:
                    m_good = min(1.0, (score - 0.7) * 7 + 0.3)
                elif score >= 0.6 and score < 0.7:
                    m_good = (score - 0.6) * 7
                else:
                    m_good = 0

                # 中的隶属度
                if score >= 0.7 and score < 0.8:
                    m_medium = min(1.0, (0.8 - score) * 10)
                elif score >= 0.6 and score < 0.7:
                    m_medium = min(1.0, (score - 0.6) * 3 + 0.3)
                elif score >= 0.5 and score < 0.6:
                    m_medium = min(1.0, (score - 0.5) * 7 + 0.3)
                elif score >= 0.4 and score < 0.5:
                    m_medium = (score - 0.4) * 7
                else:
                    m_medium = 0

                # 差的隶属度
                if score >= 0.5 and score < 0.6:
                    m_poor = min(1.0, (0.6 - score) * 10)
                elif score >= 0.4 and score < 0.5:
                    m_poor = min(1.0, (score - 0.4) * 3 + 0.3)
                elif score < 0.4:
                    m_poor = min(1.0, 0.7 + (0.4 - score) * 0.75)
                else:
                    m_poor = 0

                # 归一化处理
                total = m_excellent + m_good + m_medium + m_poor
                if total > 0:
                    membership = [
                        m_excellent / total,
                        m_good / total,
                        m_medium / total,
                        m_poor / total,
                    ]
                else:
                    # 如果所有隶属度都为0（极端情况），则设为均等
                    membership = [0.25, 0.25, 0.25, 0.25]

                matrix.append(membership)

            fuzzy_matrices[translator] = np.array(matrix)

        # 计算模糊综合评价结果
        fuzzy_results = {}

        for translator, matrix in fuzzy_matrices.items():
            # 权重向量
            weight_vector = np.array(
                [
                    self.weights["SP"],
                    self.weights["GC"],
                    self.weights["EF"],
                    self.weights["TA"],
                ]
            )

            # 模糊合成
            result_vector = np.dot(weight_vector, matrix)

            # 归一化
            result_vector = result_vector / np.sum(result_vector)

            # 计算加权得分
            weighted_score = np.dot(result_vector, np.array([0.95, 0.75, 0.5, 0.25]))

            fuzzy_results[translator] = {
                "membership": {grades[i]: result_vector[i] for i in range(len(grades))},
                "weighted_score": weighted_score,
            }

        return fuzzy_results


def read_text_data(file_path):
    """
    从文本文件中读取数据

    参数:
        file_path (str): 文本文件路径

    返回:
        dict: 读取的数据
    """
    data = {}
    current_key = None
    current_text = []

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            if line.startswith("[") and line.endswith("]"):
                # 如果已经有一个key正在处理，保存它
                if current_key:
                    data[current_key] = "\n".join(current_text)
                    current_text = []

                # 设置新的key
                current_key = line
            else:
                # 将行添加到当前文本
                current_text.append(line)

        # 保存最后一个key的文本
        if current_key:
            data[current_key] = "\n".join(current_text)

    except Exception as e:
        print(f"读取文件时出错: {e}")

    return data


# 问题一：评价三款翻译软件
def problem1():
    """问题一：评价三款翻译软件的双向翻译效果"""

    # 创建评价模型
    evaluator = TranslationEvaluator()

    # 读取文本数据
    text_data = read_text_data("text_data_12.txt")

    # 获取原始文本和翻译结果
    original_text = text_data.get("[ORIGINAL_TEXT_1]", "")
    translated_texts = {
        "Google": text_data.get("[GOOGLE_TRANSLATION_1]", ""),
        "Baidu": text_data.get("[BAIDU_TRANSLATION_1]", ""),
        "DeepL": text_data.get("[DEEPL_TRANSLATION_1]", ""),
    }

    # 为问题一创建专门的领域术语词典
    domain_terms = {
        "科技大学": 1.5,
        "高校": 1.5,
        "学科": 1.5,
        "工程": 1.3,
        "科学": 1.3,
        "教育": 1.3,
        "多科性": 1.4,
        "协调发展": 1.4,
        "重点建设": 1.4,
    }

    # 评价翻译质量
    evaluation_results = evaluator.evaluate_translators(
        original_text, translated_texts, domain_terms
    )

    # 输出评价结果
    print("\n=============== 问题一 ===============")
    print("\n定量评价结果：")
    for translator, results in evaluation_results.items():
        print(f"\n{translator}翻译评价:")
        print(f"  语义保留度: {results['SP']:.4f}")
        print(f"  语法正确性: {results['GC']:.4f}")
        print(f"  表达流畅性: {results['EF']:.4f}")
        print(f"  术语准确性: {results['TA']:.4f}")
        print(f"  总分: {results['Total']:.4f}")

    # 对翻译软件进行排名
    ranked_translators = evaluator.rank_translators(evaluation_results)
    print("\n翻译软件排名:")
    for i, (translator, score) in enumerate(ranked_translators):
        print(f"{i+1}. {translator}: {score:.4f}")

    # 模糊综合评价
    fuzzy_results = evaluator.fuzzy_comprehensive_evaluation(evaluation_results)
    print("\n模糊综合评价结果:")
    for translator, results in fuzzy_results.items():
        print(f"\n{translator}模糊评价:")
        for grade, membership in results["membership"].items():
            print(f"  {grade}: {membership:.4f}")
        print(f"  加权得分: {results['weighted_score']:.4f}")

    return evaluation_results, ranked_translators


# 问题二：评价英文短文的双向翻译
def problem2():
    """问题二：评价英文短文的双向翻译效果"""

    # 创建评价模型
    evaluator = TranslationEvaluator()

    # 读取文本数据
    text_data = read_text_data("text_data_12.txt")

    # 获取原始文本和翻译结果
    original_text = text_data.get("[ORIGINAL_TEXT_2]", "")
    translated_texts = {
        "Google": text_data.get("[GOOGLE_TRANSLATION_2]", ""),
        "Baidu": text_data.get("[BAIDU_TRANSLATION_2]", ""),
        "DeepL": text_data.get("[DEEPL_TRANSLATION_2]", ""),
    }

    # 为问题二创建专门的领域术语词典
    domain_terms = {
        "翻译": 1.5,
        "机器翻译": 1.5,
        "人工智能": 1.5,
        "神经机器翻译": 1.5,
        "文化": 1.3,
        "语言": 1.3,
        "全球化": 1.3,
        "自动化": 1.3,
        "术语": 1.3,
    }

    # 评价翻译质量
    evaluation_results = evaluator.evaluate_translators(
        original_text, translated_texts, domain_terms
    )

    # 输出评价结果
    print("\n=============== 问题二 ===============")
    print("\n定量评价结果：")
    for translator, results in evaluation_results.items():
        print(f"\n{translator}翻译评价:")
        print(f"  语义保留度: {results['SP']:.4f}")
        print(f"  语法正确性: {results['GC']:.4f}")
        print(f"  表达流畅性: {results['EF']:.4f}")
        print(f"  术语准确性: {results['TA']:.4f}")
        print(f"  总分: {results['Total']:.4f}")

    # 对翻译软件进行排名
    ranked_translators = evaluator.rank_translators(evaluation_results)
    print("\n翻译软件排名:")
    for i, (translator, score) in enumerate(ranked_translators):
        print(f"{i+1}. {translator}: {score:.4f}")

    # 模糊综合评价
    fuzzy_results = evaluator.fuzzy_comprehensive_evaluation(evaluation_results)
    print("\n模糊综合评价结果:")
    for translator, results in fuzzy_results.items():
        print(f"\n{translator}模糊评价:")
        for grade, membership in results["membership"].items():
            print(f"  {grade}: {membership:.4f}")
        print(f"  加权得分: {results['weighted_score']:.4f}")

    return evaluation_results, ranked_translators


def save_best_translator(translator_name):
    """
    保存效果最好的翻译软件名称到文件

    参数:
        translator_name (str): 翻译软件名称
    """
    with open("best_translator.txt", "w", encoding="utf-8") as f:
        f.write(translator_name)


if __name__ == "__main__":
    # 1. 运行问题一：评价三款翻译软件
    results1, ranked1 = problem1()

    # 2. 运行问题二：评价英文短文
    results2, ranked2 = problem2()

    # 3. 根据问题一和问题二的结果，选择最佳翻译软件
    all_scores = {}
    for translator in ["Google", "Baidu", "DeepL"]:
        all_scores[translator] = (
            results1[translator]["Total"] + results2[translator]["Total"]
        ) / 2

    best_translator = max(all_scores.items(), key=lambda x: x[1])[0]
    print(f"\n综合问题一和问题二的结果，效果最好的翻译软件是：{best_translator}")

    # 保存最佳翻译软件名称，供问题三使用
    save_best_translator(best_translator)
