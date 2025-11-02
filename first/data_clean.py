import re
import sys

# --- 用户指定路径 ---
# 请在这里替换您的输入文件路径
inputfile = "tutorial/corpus/english_corpus.txt"
# 请在这里替换您的输出文件路径
outputfile = "tutorial/corpus/cleaned_corpus.txt"
# --------------------

# 用于跟踪已处理过的行，实现高效去重
seen_lines = set()

# 语料库中观察到的格式化残留前缀
# (例如 textbfmagellan -> magellan, sim6 -> 6)
ARTIFACT_PREFIXES = [
    'textbf', 'textit', 'emph', 'mathrm', 'sim', 
    'href', 'cdot', 'leq', 'geq', 'approx'
]

try:
    with open(inputfile, 'r', encoding='utf-8') as infile, \
         open(outputfile, 'w', encoding='utf-8') as outfile:

        print(f"开始清洗文件: {inputfile}...")
        
        line_count = 0
        for line in infile:
            line_count += 1
            if line_count % 200 == 0:
                print(f"  ...已处理 {line_count} 行", end='\r')

            # 步骤 1: 移除元数据标签 (例如 )
            cleaned_line = re.sub(r'\\s*', '', line)

            # 步骤 2: 文本标准化 (转为小写)
            cleaned_line = cleaned_line.lower()

            # 步骤 3: 移除格式化“标签”和特殊编码
            # 替换特定的unicode编码
            cleaned_line = cleaned_line.replace('unicodex2014', ' ') # 这是一个破折号

            # 移除特定前缀 (使用单词边界 \b 来确保只匹配单词开头)
            for prefix in ARTIFACT_PREFIXES:
                cleaned_line = re.sub(r'\b' + re.escape(prefix) + r'([a-z])', r'\1', cleaned_line)

            # 步骤 4: 移除数字和标点符号
            # 只保留英文字母 (a-z) 和空格 (\s)
            cleaned_line = re.sub(r'[^a-z\s]', '', cleaned_line)

            # 步骤 5: 移除多余空白
            # 将多个空格（或制表符）合并为一个空格，并移除首尾的空白
            cleaned_line = re.sub(r'\s+', ' ', cleaned_line).strip()

            # 步骤 6: 去重和写入
            # 只有当行不为空，并且是第一次见到时，才写入文件
            if cleaned_line and cleaned_line not in seen_lines:
                seen_lines.add(cleaned_line)
                outfile.write(cleaned_line + '\n')

        print(f"\n清洗完成。")
        print(f"总共处理了 {line_count} 行。")
        print(f"清洗后得到 {len(seen_lines)} 行独立内容。")
        print(f"文件已保存至: {outputfile}")

except FileNotFoundError:
    print(f"错误: 输入文件 {inputfile} 未找到。请检查路径是否正确。")
except Exception as e:
    print(f"处理文件时发生错误: {e}")