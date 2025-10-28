import re
import string
import os
import argparse # 用于命令行执行

def clean_text_block(text):
    """
    在去重、转小写和移除标点之前，清理单个文本块。
    移除 URL、特定的类 LaTeX 格式，并规范化内部空白。

    参数:
        text (str): 输入的文本块。

    返回:
        str: 清理后的文本块。
    """
    if not text:
        return ""

    # 1. 健壮地移除 URL 和链接占位符/短语
    # 更全面的 URL 匹配
    text = re.sub(r'https?://[^\s<>"]+|www\.[^\s<>"]+', '', text)
    # 常见的链接短语（不区分大小写，处理换行）
    link_phrases = [
        r'href\s+this\s+URL',
        # 匹配 "code/data/models is/are available/valuable at ..." 等模式，直到句末或换行
        r'(?:code|data|models?|codebase|repository|project\s+page|webpage)\b.*? (?:is|are)\s+(?:available|valuable)\s+at.*?(?:[\.\n]|$)',
        # 更通用的 "available/valuable at ..."
        r'\b(?:available|valuable)\s+at\s+.*?(?:[\.\n]|$)',
        # 处理 "code and/or data is/are available/valuable"
        r'code\s+(?:and|or)\s+data\s+(?:is|are)\s+(?:available|valuable)',
        r'code\s+available\s+at'
    ]
    for phrase in link_phrases:
         # 使用 DOTALL 进行多行匹配，IGNORECASE 忽略大小写
        text = re.sub(phrase, '', text, flags=re.IGNORECASE | re.DOTALL)

    # 2. 移除 LaTeX 风格格式 (textbf, textit, mathrm), 保留内容
    text = re.sub(r'\\(?:textbf|textit|mathrm)\{([^}]*?)\}', r'\1', text)

    # 3. 移除常见的数学分隔符 (非贪婪匹配)
    text = re.sub(r'\$.*?\$', '', text) # 内联数学公式 $...$
    text = re.sub(r'\\\(.*?\\\)', '', text) # \( ... \)
    text = re.sub(r'\\\[.*?\\\]', '', text) # \[ ... \]

    # 4. 移除简单的列表标记，如 i), (1), a) 等
    text = re.sub(r'\b(?:[ivx]+|\d+|[a-z])\)\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\(\s*(?:[ivx]+|\d+|[a-z])\s*\)\s*', '', text, flags=re.IGNORECASE)

    # 5. 将内部的空白字符规范化为单个空格
    text = re.sub(r'\s+', ' ', text)

    # 6. 去除块开头和结尾的空白字符
    text = text.strip()

    return text

def process_corpus_file(input_filepath, output_filepath):
    """
    读取原始语料库文件，根据分析需求（词频/熵）进行清理，
    并将清理后的文本写入输出文件。

    参数:
        input_filepath (str): 输入文本语料库文件的路径。
        output_filepath (str): 保存清理后输出文本文件的路径。
    """
    print(f"--- 开始语料库清理 ---")
    print(f"输入文件: {input_filepath}")
    print(f"输出文件: {output_filepath}")

    try:
        with open(input_filepath, 'r', encoding='utf-8') as infile:
            full_content = infile.read()
        print(f"成功读取输入文件 (共 {len(full_content):,} 字符)。")
    except FileNotFoundError:
        print(f"错误: 未找到输入文件 '{input_filepath}'。")
        return
    except Exception as e:
        print(f"读取文件 '{input_filepath}' 时出错: {e}")
        return

    # 1. 按 source 标签分割（丢弃标签本身）
    # 正则表达式 `\` 作为分隔符
    blocks = re.split(r'\\', full_content)
    print(f"分割成 {len(blocks)} 个原始块 (可能包含空块)。")

    # 2. 单独清理每个块，并基于清理后的内容去重
    unique_cleaned_content = []
    seen_content_hashes = set() # 使用哈希值可能在大规模数据时查找更快

    processed_blocks = 0
    empty_blocks = 0
    duplicate_blocks = 0

    for block in blocks:
        trimmed_block = block.strip()
        if not trimmed_block:
            empty_blocks += 1
            continue # 跳过空块或只有空白字符的块

        cleaned_content = clean_text_block(trimmed_block)

        if cleaned_content: # 确保清理后仍有内容
            # 使用哈希值进行高效检查
            content_hash = hash(cleaned_content)
            if content_hash not in seen_content_hashes:
                unique_cleaned_content.append(cleaned_content)
                seen_content_hashes.add(content_hash)
                processed_blocks += 1
            else:
                duplicate_blocks += 1
        else:
             # 清理后变为空块 (例如，块内只包含一个 URL)
             empty_blocks += 1

    print(f"处理块报告:")
    print(f"  - 添加的唯一非空块数量: {processed_blocks}")
    print(f"  - 跳过的重复块数量: {duplicate_blocks}")
    print(f"  - 跳过的空块或仅空白块数量: {empty_blocks}")

    if not unique_cleaned_content:
        print("警告: 清理和去重后没有剩余内容。")
        # 可以选择创建一个空的输出文件或直接停止
        try:
             with open(output_filepath, 'w', encoding='utf-8') as outfile:
                 outfile.write("")
             print(f"已创建空的输出文件: {output_filepath}")
        except Exception as e:
            print(f"写入空的输出文件时出错: {e}")
        return

    # 3. 将唯一的清理后块连接成一个大的文本字符串
    final_text = ' '.join(unique_cleaned_content)
    print(f"已将唯一块合并为单个文本 (共 {len(final_text):,} 字符)。")

    # 4. 全局转为小写
    final_text = final_text.lower()
    print("已将文本转换为小写。")

    # 5. 全局移除标点符号
    # 创建转换表以移除所有标点
    translator = str.maketrans('', '', string.punctuation)
    final_text = final_text.translate(translator)
    print("已移除标点符号。")

    # 6. 最终的空白字符规范化（移除标点后很重要）
    final_text = re.sub(r'\s+', ' ', final_text).strip()
    print("已全局规范化空白字符。")

    # 7. 将清理后的文本写入输出文件
    try:
        # 确保输出目录存在
        output_dir = os.path.dirname(output_filepath)
        if output_dir and not os.path.exists(output_dir): # 处理输出在当前目录的情况
            os.makedirs(output_dir)
            print(f"已创建输出目录: {output_dir}")

        with open(output_filepath, 'w', encoding='utf-8') as outfile:
            outfile.write(final_text)
        print(f"--- 清理完成 ---")
        print(f"清理后的文本已保存至: {output_filepath}")
        print(f"最终字符数: {len(final_text):,}")

    except Exception as e:
        print(f"写入输出文件 '{output_filepath}' 时出错: {e}")

# 添加一个接口函数，允许用户自定义输入和输出文件路径
def clean_corpus(input_file, output_file):
    """
    接口函数，允许用户通过代码调用自定义输入和输出文件路径。

    参数:
        input_file (str): 输入的文本语料库文件路径。
        output_file (str): 保存清理后输出文本文件的路径。
    """
    process_corpus_file(input_file, output_file)

# --- 主执行部分 ---
if __name__ == "__main__":
    # 直接在代码中指定输入和输出文件路径
    input_file = "tutorial/corpus/english_corpus.txt"  # 替换为你的输入文件路径
    output_file = "tutorial/corpus/cleaned_english_corpus.txt"  # 替换为你的输出文件路径

    # 调用接口函数进行清理
    clean_corpus(input_file, output_file)