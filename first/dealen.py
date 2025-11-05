import math
from collections import Counter
import re
import os
import nltk
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.data.path.append('C:/Users/zzx20/AppData/Roaming/nltk_data')
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, PunktSentenceTokenizer
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

# --- 1. 配置 ---

# !!! 关键：你准备好的三个不同规模的样本文件
INPUT_FILES = [
    'tutorial/corpus/100000en.txt',   # 示例：小规模样本
    'tutorial/corpus/150000en.txt',  # 示例：中规模样本
    'tutorial/corpus/200000en.txt'    # 示例：大规模样本
]

# --- 输出文件 ---
# 1. 最终对比图的文件名
OUTPUT_CHART_FILE = 'word_comparison_chart.png'
# 2. 独立高频词表格的前缀
OUTPUT_TABLE_PREFIX = 'word_top_n_'

# --- 字体配置 (!! 关键 !!) ---
FONT_PATH = 'C:/Windows/Fonts/arial.ttf' # (Arial 字体)
FONT_SIZE = 16
IMAGE_PADDING = 20

MATPLOTLIB_FONT_NAME = 'Arial'

# --- 分析配置 ---
WORD_REGEX = re.compile(r'\b\w+\b')
TOP_N_TO_SHOW = 20 # 在独立表格中显示前 N 个

lemmatizer = WordNetLemmatizer()
tokenizer = PunktSentenceTokenizer()

def analyze_words(text):
    """
    计算所有单词的频率、概率和信息熵
    """
    print("--- 单词分析 (概率与熵) ---")

    # 提取单词并进行词形还原
    words = [lemmatizer.lemmatize(word.lower()) for word in word_tokenize(text)]

    if not words:
        print("未在文件中找到单词。")
        return None, None

    total_word_count = len(words)
    word_counts = Counter(words)
    unique_word_count = len(word_counts)

    entropy = 0.0
    probabilities = {}

    for word, count in word_counts.items():
        probability = count / total_word_count
        probabilities[word] = probability
        entropy -= probability * math.log2(probability)

    print(f"总单词数量: {total_word_count}")
    print(f"独立单词数量 (词汇量): {unique_word_count}")
    print(f"单词信息熵 (Entropy): {entropy:.4f} bits/word")

    summary_stats = {
        "total": total_word_count,
        "unique": unique_word_count,
        "entropy": entropy
    }

    top_n_data = word_counts.most_common(TOP_N_TO_SHOW)

    return summary_stats, top_n_data, probabilities

def create_top_n_table_image(top_n_data, probabilities, filename, font):
    """
    为单个样本生成 Top-N 高频词表格图片
    """
    print(f"正在生成 Top-N 表格: {filename} ...")

    report_lines = []
    headers = f"{'排名':<4} {'单词':<10} {'次数':<8} {'概率':<10}"
    report_lines.append(headers)
    report_lines.append("-" * 40)

    for i, (word, count) in enumerate(top_n_data, 1):
        line = f"{i:<4} {word:<10} {count:<8} {probabilities[word]:<10.4%}"
        report_lines.append(line)

    report_text = "\n".join(report_lines)

    dummy_draw = ImageDraw.Draw(Image.new('RGB', (1, 1)))
    bbox = dummy_draw.textbbox((0, 0), report_text, font=font, spacing=5)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    img_width = text_width + (IMAGE_PADDING * 2)
    img_height = text_height + (IMAGE_PADDING * 2)

    image = Image.new('RGB', (img_width, img_height), 'white')
    draw = ImageDraw.Draw(image)

    draw.text(
        (IMAGE_PADDING, IMAGE_PADDING),
        report_text,
        fill='black',
        font=font,
        spacing=5
    )
    image.save(filename)
    print(f"成功保存表格: {filename}")

def create_comparison_chart(results_data, output_filename):
    """
    使用 Matplotlib 生成对比条形图
    """
    print(f"\n--- 正在生成对比图: {output_filename} ---")

    try:
        plt.rcParams['font.sans-serif'] = [MATPLOTLIB_FONT_NAME]
        plt.rcParams['axes.unicode_minus'] = False
    except Exception as e:
        print(f"警告: 设置字体 '{MATPLOTLIB_FONT_NAME}' 失败: {e}")

    labels = [os.path.basename(r['file']) for r in results_data]
    totals = [r['stats']['total'] for r in results_data]
    uniques = [r['stats']['unique'] for r in results_data]
    entropies = [r['stats']['entropy'] for r in results_data]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))
    fig.suptitle('不同规模语料库单词统计对比', fontsize=20, y=1.02)

    ax1.bar(labels, totals, color='skyblue')
    ax1.set_title('总单词数 (Total Count)')
    ax1.set_ylabel('数量')
    for i, v in enumerate(totals):
        ax1.text(i, v + 0.5, str(v), ha='center', va='bottom')

    ax2.bar(labels, uniques, color='lightgreen')
    ax2.set_title('独立单词数 (Unique Count)')
    ax2.set_ylabel('数量')
    for i, v in enumerate(uniques):
        ax2.text(i, v + 0.5, str(v), ha='center', va='bottom')

    ax3.bar(labels, entropies, color='salmon')
    ax3.set_title('单词信息熵 (Entropy)')
    ax3.set_ylabel('比特 (bits)')
    ax3.set_ylim(bottom=min(entropies) - 0.5)
    for i, v in enumerate(entropies):
        ax3.text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(output_filename)
    print(f"成功保存对比图: {output_filename}")

def main():
    all_results = []

    try:
        pil_font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
    except IOError:
        print(f"错误: 字体文件 '{FONT_PATH}' 未找到。")
        pil_font = None

    for filepath in INPUT_FILES:
        print(f"\n{'='*20}{filepath} {'='*20}")
        if not os.path.exists(filepath):
            print(f"\n{'='*20} 正在分析: {filepath} {'='*20}")
            continue

        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        summary_stats, top_n_data, probabilities = analyze_words(content)

        if summary_stats:
            all_results.append({
                "file": filepath,
                "stats": summary_stats
            })

            if pil_font and top_n_data:
                file_basename = os.path.basename(filepath)
                file_simple_name = os.path.splitext(file_basename)[0]
                table_filename = f"{OUTPUT_TABLE_PREFIX}{file_simple_name}.png"
                create_top_n_table_image(top_n_data, probabilities, table_filename, pil_font)

    if not all_results:
        print("\n--- 分析完成，但没有有效结果可生成图片。 ---")
        return

    create_comparison_chart(all_results, OUTPUT_CHART_FILE)

    print("\n--- 所有分析已完成 ---")

if __name__ == "__main__":
    main()

