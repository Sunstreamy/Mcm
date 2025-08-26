# -*- coding: utf-8 -*-

"""
翻译软件评价模型：问题三
本代码使用问题一和问题二中表现最好的翻译软件评价不同类型文本的翻译效果
"""

import numpy as np
import jieba
import re
from collections import Counter
import matplotlib
import os

matplotlib.rcParams["font.sans-serif"] = ["SimHei"]  # 用来正常显示中文标签
matplotlib.rcParams["axes.unicode_minus"] = False  # 用来正常显示负号

# 从s12.py导入评价模型类
from s12 import TranslationEvaluator


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


def get_best_translator():
    """
    获取效果最好的翻译软件名称

    返回:
        str: 翻译软件名称
    """
    try:
        with open("best_translator.txt", "r", encoding="utf-8") as f:
            return f.read().strip()
    except:
        print("警告：无法读取最佳翻译软件名称，将使用默认值 'Google'")
        return "Google"


# 问题三：评价不同类型文本的翻译效果
def problem3():
    """
    问题三：使用效果最好的翻译软件评价不同类型文本的翻译效果
    """
    # 获取效果最好的翻译软件
    best_translator = get_best_translator()
    print(f"\n=============== 问题三 ===============")
    print(f"使用效果最好的翻译软件: {best_translator} 进行评价")

    # 创建评价模型
    evaluator = TranslationEvaluator()

    # 读取文本数据
    text_data = read_text_data("text_data_3.txt")

    # 获取原始文本
    texts = {
        "唐诗": text_data.get("[TANG_POEM]", ""),
        "古散文": text_data.get("[ANCIENT_PROSE]", ""),
        "数学教材": text_data.get("[MATH_TEXT]", ""),
        "专业课程": text_data.get("[PROFESSIONAL_COURSE]", ""),
    }

    # 获取翻译结果
    # 注：实际应用时需要替换为真实的翻译结果
    translated_texts = {
        "唐诗": text_data.get("[TANG_POEM_TRANSLATION]", ""),
        "古散文": text_data.get("[ANCIENT_PROSE_TRANSLATION]", ""),
        "数学教材": text_data.get("[MATH_TEXT_TRANSLATION]", ""),
        "专业课程": text_data.get("[PROFESSIONAL_COURSE_TRANSLATION]", ""),
    }

    # 为不同类型文本创建专门的术语词典
    domain_terms = {
        "唐诗": {"春眠": 1.5, "晓": 1.3, "啼鸟": 1.5, "风雨": 1.3, "花落": 1.4},
        "古散文": {
            "庖丁": 1.5,
            "解牛": 1.5,
            "文惠君": 1.5,
            "砉然": 1.4,
            "响然": 1.4,
            "奏刀": 1.4,
            "桑林": 1.3,
            "经首": 1.3,
        },
        "数学教材": {
            "向量空间": 1.5,
            "线性代数": 1.5,
            "向量": 1.4,
            "加法运算": 1.4,
            "标量乘法": 1.4,
            "向量公理": 1.5,
        },
        "专业课程": {
            "计算机网络": 1.5,
            "体系结构": 1.4,
            "通信协议": 1.4,
            "网络应用": 1.3,
            "核心课程": 1.3,
        },
    }

    # 保存所有类型文本的评价结果
    all_results = {}

    # 对每种类型的文本进行评价
    for text_type, original_text in texts.items():
        print(f"\n评价{text_type}的翻译效果：")
        print("\n原文：")
        print(original_text)

        # 获取该类型的翻译结果
        if text_type in translated_texts and translated_texts[text_type]:
            translated_text = translated_texts[text_type]
        else:
            # 如果没有预设的翻译结果，使用原文作为翻译结果（仅用于测试）
            translated_text = original_text
            print("警告：没有找到该类型文本的翻译结果，将使用原文作为翻译结果进行评估")

        print(f"\n{best_translator}翻译结果:")
        print(translated_text)

        # 创建翻译结果字典
        translator_result = {best_translator: translated_text}

        # 获取该类型文本的专业术语词典
        current_domain_terms = domain_terms.get(text_type, None)

        # 评价翻译质量
        evaluation_results = evaluator.evaluate_translators(
            original_text, translator_result, current_domain_terms
        )

        # 保存评价结果
        all_results[text_type] = evaluation_results[best_translator]

        # 输出评价结果
        print(f"\n{best_translator}翻译评价结果:")
        print(f"  语义保留度: {evaluation_results[best_translator]['SP']:.4f}")
        print(f"  语法正确性: {evaluation_results[best_translator]['GC']:.4f}")
        print(f"  表达流畅性: {evaluation_results[best_translator]['EF']:.4f}")
        print(f"  术语准确性: {evaluation_results[best_translator]['TA']:.4f}")
        print(f"  总分: {evaluation_results[best_translator]['Total']:.4f}")

    return all_results


if __name__ == "__main__":
    problem3()
