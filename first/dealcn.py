import math
from collections import Counter
import re
import os

# --- 配置 ---
# 确保这个文件名与你上一步生成的清理后文件名一致
INPUT_FILE = 'sina_scraper/100cn_cleaned.txt'

# 用于匹配所有汉字的正则表达式
HANZI_REGEX = re.compile(r'[\u4e00-\u9fa5]')

# 在报告中显示频率最高的前 N 个
TOP_N_TO_SHOW = 20

# --- 1. 汉字分析函数 (作业要求 ② - 汉字部分) ---
def analyze_hanzi(text):
    """
    计算所有单个汉字的频率、概率和信息熵
    """
    print("--- 汉字分析 (概率与熵) ---")
    
    # 找出所有符合汉字 Unicode 范围的字符
    hanzi_list = HANZI_REGEX.findall(text)
    
    if not hanzi_list:
        print("未在文件中找到汉字。")
        return None

    # 统计总字数
    total_hanzi_count = len(hanzi_list)
    
    # 使用 Counter 统计每个独立汉字的频次
    hanzi_counts = Counter(hanzi_list)
    
    # 统计独立汉字的数量
    unique_hanzi_count = len(hanzi_counts)
    
    entropy = 0.0
    probabilities = {}
    
    # 计算每个汉字的概率 和 整体的信息熵
    for char, count in hanzi_counts.items():
        probability = count / total_hanzi_count
        probabilities[char] = probability
        # 熵的计算公式: H = -Σ p(x) * log2(p(x))
        entropy -= probability * math.log2(probability)
        
    print(f"总汉字数量: {total_hanzi_count}")
    print(f"独立汉字数量 (字库大小): {unique_hanzi_count}")
    print(f"汉字信息熵 (Entropy): {entropy:.4f} bits/char")
    
    print(f"\n频率最高的 {TOP_N_TO_SHOW} 个汉字:")
    print(f"{'排名':<4} {'汉字':<4} {'次数':<6} {'概率':<10}")
    print("-" * 26)
    for i, (char, count) in enumerate(hanzi_counts.most_common(TOP_N_TO_SHOW), 1):
        print(f"{i:<4} {char:<4} {count:<6} {probabilities[char]:<10.4%}")
    
    # 返回汉字的频次统计结果，方便你后续（如作业要求⑤）使用
    return hanzi_counts

# --- 2. 主函数 ---
def main():
    if not os.path.exists(INPUT_FILE):
        print(f"错误: 输入文件 '{INPUT_FILE}' 未找到。")
        print(f"请先运行上一个清理脚本来生成此文件。")
        return

    print(f"正在读取文件: {INPUT_FILE}...\n")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        content = f.read()
        
    # 执行汉字分析
    hanzi_stats = analyze_hanzi(content)
    
    if hanzi_stats:
        print("\n--- 分析完成 ---")
        print("已输出汉字的频率、概率和信息熵。")

if __name__ == "__main__":
    main()