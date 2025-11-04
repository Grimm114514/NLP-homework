import math
from collections import Counter
import re
import os
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

# --- 1. 配置 ---

# !!! 关键：你准备好的三个不同规模的样本文件
INPUT_FILES = [
    'sina_scraper/100cn_cleaned.txt',   # 示例：小规模样本
    'sina_scraper/150cn_cleaned.txt',  # 示例：中规模样本
    'sina_scraper/200cn_cleaned.txt'    # 示例：大规模样本
]

# --- 输出文件 ---
# 1. 最终对比图的文件名
OUTPUT_CHART_FILE = 'hanzi_comparison_chart.png'
# 2. 独立高频词表格的前缀
OUTPUT_TABLE_PREFIX = 'hanzi_top_n_'

# --- 字体配置 (!! 关键 !!) ---
# PIL (Pillow) 使用字体路径
FONT_PATH = 'C:/Windows/Fonts/msyh.ttc' # (微软雅黑)
FONT_SIZE = 16
IMAGE_PADDING = 20

# Matplotlib 使用字体名称
# (msyh.ttc 对应的字体名通常是 'Microsoft YaHei')
# (simsun.ttc 对应 'SimSun')
MATPLOTLIB_FONT_NAME = 'Microsoft YaHei' 

# --- 分析配置 ---
HANZI_REGEX = re.compile(r'[\u4e00-\u9fa5]')
TOP_N_TO_SHOW = 20 # 在独立表格中显示前 N 个

# --- 2. 汉字分析函数 (基本不变) ---
def analyze_hanzi(text):
    """
    计算所有单个汉字的频率、概率和信息熵
    """
    print("--- 汉字分析 (概率与熵) ---")
    
    hanzi_list = HANZI_REGEX.findall(text)
    
    if not hanzi_list:
        print("未在文件中找到汉字。")
        return None, None

    total_hanzi_count = len(hanzi_list)
    hanzi_counts = Counter(hanzi_list)
    unique_hanzi_count = len(hanzi_counts)
    
    entropy = 0.0
    probabilities = {}
    
    for char, count in hanzi_counts.items():
        probability = count / total_hanzi_count
        probabilities[char] = probability
        entropy -= probability * math.log2(probability)
        
    print(f"总汉字数量: {total_hanzi_count}")
    print(f"独立汉字数量 (字库大小): {unique_hanzi_count}")
    print(f"汉字信息熵 (Entropy): {entropy:.4f} bits/char")
    
    # 准备返回数据
    summary_stats = {
        "total": total_hanzi_count,
        "unique": unique_hanzi_count,
        "entropy": entropy
    }
    
    top_n_data = hanzi_counts.most_common(TOP_N_TO_SHOW)
    
    return summary_stats, top_n_data, probabilities

# --- 3. [新] 生成高频词表格图片 ---
def create_top_n_table_image(top_n_data, probabilities, filename, font):
    """
    为单个样本生成 Top-N 高频词表格图片
    """
    print(f"正在生成 Top-N 表格: {filename} ...")
    
    # 准备报告文本
    report_lines = []
    headers = f"{'排名':<4} {'汉字':<4} {'次数':<8} {'概率':<10}"
    report_lines.append(headers)
    report_lines.append("-" * 30)
    
    for i, (char, count) in enumerate(top_n_data, 1):
        line = f"{i:<4} {char:<4} {count:<8} {probabilities[char]:<10.4%}"
        report_lines.append(line)
        
    report_text = "\n".join(report_lines)
    
    # 动态计算图片尺寸
    dummy_draw = ImageDraw.Draw(Image.new('RGB', (1, 1)))
    bbox = dummy_draw.textbbox((0, 0), report_text, font=font, spacing=5)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    img_width = text_width + (IMAGE_PADDING * 2)
    img_height = text_height + (IMAGE_PADDING * 2)

    # 创建图片
    image = Image.new('RGB', (img_width, img_height), 'white')
    draw = ImageDraw.Draw(image)

    # 绘制文本
    draw.text(
        (IMAGE_PADDING, IMAGE_PADDING),
        report_text,
        fill='black',
        font=font,
        spacing=5
    )
    image.save(filename)
    print(f"成功保存表格: {filename}")

# --- 4. [新] 生成对比条形图 ---
def create_comparison_chart(results_data, output_filename):
    """
    使用 Matplotlib 生成对比条形图
    """
    print(f"\n--- 正在生成对比图: {output_filename} ---")
    
    # 配置 Matplotlib 字体
    try:
        plt.rcParams['font.sans-serif'] = [MATPLOTLIB_FONT_NAME]
        plt.rcParams['axes.unicode_minus'] = False # 解决负号显示问题
    except Exception as e:
        print(f"警告: 设置中文字体 '{MATPLOTLIB_FONT_NAME}' 失败: {e}")
        print("图表中的中文可能显示为方框。")

    # 准备数据
    labels = [os.path.basename(r['file']) for r in results_data]
    totals = [r['stats']['total'] for r in results_data]
    uniques = [r['stats']['unique'] for r in results_data]
    entropies = [r['stats']['entropy'] for r in results_data]
    
    # 创建 3 个子图 (3行, 1列)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))
    fig.suptitle('不同规模语料库汉字统计对比', fontsize=20, y=1.02)
    
    # 1. 总汉字数
    ax1.bar(labels, totals, color='skyblue')
    ax1.set_title('总汉字数 (Total Count)')
    ax1.set_ylabel('数量')
    for i, v in enumerate(totals):
        ax1.text(i, v + 0.5, str(v), ha='center', va='bottom')
        
    # 2. 独立汉字数
    ax2.bar(labels, uniques, color='lightgreen')
    ax2.set_title('独立汉字数 (Unique Count)')
    ax2.set_ylabel('数量')
    for i, v in enumerate(uniques):
        ax2.text(i, v + 0.5, str(v), ha='center', va='bottom')

    # 3. 汉字信息熵
    ax3.bar(labels, entropies, color='salmon')
    ax3.set_title('汉字信息熵 (Entropy)')
    ax3.set_ylabel('比特 (bits)')
    ax3.set_ylim(bottom=min(entropies) - 0.5) # 让Y轴起点更合理
    for i, v in enumerate(entropies):
        ax3.text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom')

    # 调整布局并保存
    plt.tight_layout()
    plt.savefig(output_filename)
    print(f"成功保存对比图: {output_filename}")

# --- 5. 主函数 ---
def main():
    all_results = []
    
    # 检查 PIL 字体
    try:
        pil_font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
    except IOError:
        print(f"错误: 字体文件 '{FONT_PATH}' 未找到。")
        print("请检查 FONT_PATH 变量是否设置正确。")
        print("独立表格将无法生成。")
        pil_font = None

    for filepath in INPUT_FILES:
        print(f"\n{'='*20}{filepath} {'='*20}")
        if not os.path.exists(filepath):
            print(f"\n{'='*20} 正在分析: {filepath} {'='*20}")
            continue

        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # 1. 执行分析
        summary_stats, top_n_data, probabilities = analyze_hanzi(content)
        
        if summary_stats:
            # 2. 保存对比图所需的结果
            all_results.append({
                "file": filepath,
                "stats": summary_stats
            })
            
            # 3. 生成该样本的 Top-N 表格图片
            if pil_font and top_n_data:
                file_basename = os.path.basename(filepath)
                file_simple_name = os.path.splitext(file_basename)[0]
                table_filename = f"{OUTPUT_TABLE_PREFIX}{file_simple_name}.png"
                create_top_n_table_image(top_n_data, probabilities, table_filename, pil_font)
    
    # 检查是否有结果
    if not all_results:
        print("\n--- 分析完成，但没有有效结果可生成图片。 ---")
        return

    # 4. 生成最终的对比图
    create_comparison_chart(all_results, OUTPUT_CHART_FILE)
    
    print("\n--- 所有分析已完成 ---")

if __name__ == "__main__":
    main()