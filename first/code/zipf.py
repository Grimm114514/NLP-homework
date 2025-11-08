import re
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import os

# 输入文件接口，可以根据需要修改这些路径
inputfiles = [
    "tutorial/corpus/100000en.txt",
    "tutorial/corpus/150000en.txt", 
    "tutorial/corpus/200000en.txt"
]

def clean_and_tokenize(text):
    """
    清理文本并分词
    """
    # 转换为小写
    text = text.lower()
    
    # 使用正则表达式提取单词（只保留字母）
    words = re.findall(r'\b[a-z]+\b', text)
    
    return words

def read_and_process_file(filepath):
    """
    读取并处理文件
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 分词并统计词频
        words = clean_and_tokenize(content)
        word_counts = Counter(words)
        
        return word_counts, len(words)
    
    except FileNotFoundError:
        print(f"文件未找到: {filepath}")
        return None, 0
    except Exception as e:
        print(f"处理文件 {filepath} 时出错: {e}")
        return None, 0

def calculate_zipf_metrics(word_counts):
    """
    计算Zipf定律相关指标
    """
    # 按频率排序
    sorted_counts = sorted(word_counts.values(), reverse=True)
    
    # 计算排名
    ranks = list(range(1, len(sorted_counts) + 1))
    
    # 计算log值用于拟合
    log_ranks = np.log(ranks)
    log_freqs = np.log(sorted_counts)
    
    # 线性拟合 log(frequency) = a * log(rank) + b
    coefficients = np.polyfit(log_ranks, log_freqs, 1)
    slope, intercept = coefficients
    
    # 计算R²值（拟合优度）
    r_squared = np.corrcoef(log_ranks, log_freqs)[0, 1] ** 2
    
    return ranks, sorted_counts, log_ranks, log_freqs, slope, intercept, r_squared

def plot_zipf_law(results, output_dir="zipf_results"):
    """
    为每个样本单独生成Zipf定律验证图表
    """
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    colors = ['blue', 'red', 'green']
    file_labels = ['100K词汇', '150K词汇', '200K词汇']
    
    # 为每个样本单独生成图表
    for i, (file_info, color, label) in enumerate(zip(results, colors, file_labels)):
        filepath, ranks, freqs, log_ranks, log_freqs, slope, intercept, r_squared, total_words = file_info
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        fig.suptitle(f'{label} - Zipf定律验证 (对数坐标拟合)', fontsize=16, fontweight='bold')
        
        # 对数坐标拟合图
        ax.scatter(log_ranks, log_freqs, color=color, alpha=0.6, s=1, label='数据点')
        
        # 拟合线
        fitted_line = slope * log_ranks + intercept
        ax.plot(log_ranks, fitted_line, 'r-', linewidth=2, 
               label=f'拟合线 (斜率={slope:.3f}, $R^2$={r_squared:.4f})')
        
        ax.set_title(f'对数坐标拟合\n总词数: {total_words:,}', fontweight='bold')
        ax.set_xlabel('log(排名)')
        ax.set_ylabel('log(词频)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        
        # 保存单独的图表
        filename = os.path.basename(filepath).replace('.txt', '')
        plt.savefig(os.path.join(output_dir, f'zipf_verification_{filename}.png'), 
                    dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()  # 关闭图表释放内存

def generate_report_table(results, output_dir="zipf_results"):
    """
    生成实验报告表格数据（移除Zipf指数相关内容）
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 准备表格数据
    table_data = []
    
    for i, file_info in enumerate(results):
        filepath, ranks, freqs, log_ranks, log_freqs, slope, intercept, r_squared, total_words = file_info
        
        # 计算统计指标
        unique_words = len(freqs)
        top_word_freq = freqs[0]
        coverage_top10 = sum(freqs[:10]) / sum(freqs) * 100
        coverage_top100 = sum(freqs[:100]) / sum(freqs) * 100
        
        table_data.append({
            '文件': os.path.basename(filepath),
            '总词数': total_words,
            '唯一词数': unique_words,
            '拟合优度R²': f"{r_squared:.4f}",
            '斜率': f"{slope:.3f}",
            '最高词频': top_word_freq,
            'Top10覆盖率(%)': f"{coverage_top10:.2f}",
            'Top100覆盖率(%)': f"{coverage_top100:.2f}"
        })
    
    # 输出表格到文件
    with open(os.path.join(output_dir, 'corpus_analysis_report.txt'), 'w', encoding='utf-8') as f:
        f.write("语料库统计分析报告\n")
        f.write("=" * 50 + "\n\n")
        
        for data in table_data:
            f.write(f"文件: {data['文件']}\n")
            f.write(f"总词数: {data['总词数']:,}\n")
            f.write(f"唯一词数: {data['唯一词数']:,}\n")
            f.write(f"拟合优度R²: {data['拟合优度R²']}\n")
            f.write(f"拟合斜率: {data['斜率']}\n")
            f.write(f"最高词频: {data['最高词频']:,}\n")
            f.write(f"Top10词汇覆盖率: {data['Top10覆盖率(%)']}%\n")
            f.write(f"Top100词汇覆盖率: {data['Top100覆盖率(%)']}%\n")
            f.write("-" * 30 + "\n\n")
        
        # 添加结论
        f.write("统计分析结论:\n")
        f.write("词频分布遵循幂律分布特征，高频词汇占据较大比例的语料覆盖率。\n")
        f.write("随着语料规模增大，词汇多样性增加，但高频词的覆盖率相对稳定。\n")
    
    # 打印表格
    print("\n" + "="*80)
    print("语料库统计分析结果")
    print("="*80)
    
    # 打印表头
    print(f"{'文件':<15} {'总词数':<10} {'唯一词数':<10} {'R²':<8} {'斜率':<8} {'Top10覆盖率%':<12}")
    print("-" * 75)
    
    # 打印数据
    for data in table_data:
        print(f"{data['文件']:<15} {data['总词数']:<10,} {data['唯一词数']:<10,} "
              f"{data['拟合优度R²']:<8} {data['斜率']:<8} {data['Top10覆盖率(%)']:<12}")
    
    return table_data

def main():
    """
    主函数 - 执行语料库统计分析
    """
    print("开始语料库统计分析...")
    print(f"分析文件: {inputfiles}")
    
    results = []
    
    for filepath in inputfiles:
        print(f"\n处理文件: {filepath}")
        
        # 处理文件
        word_counts, total_words = read_and_process_file(filepath)
        
        if word_counts is None:
            continue
        
        print(f"总词数: {total_words:,}")
        print(f"唯一词数: {len(word_counts):,}")
        
        # 计算统计指标
        ranks, freqs, log_ranks, log_freqs, slope, intercept, r_squared = calculate_zipf_metrics(word_counts)
        
        print(f"拟合优度R²: {r_squared:.4f}")
        print(f"拟合斜率: {slope:.3f}")
        
        results.append((filepath, ranks, freqs, log_ranks, log_freqs, slope, intercept, r_squared, total_words))
    
    if not results:
        print("没有成功处理的文件，请检查文件路径")
        return
    
    # 生成单独的图表
    print("\n生成对数拟合图表...")
    plot_zipf_law(results)
    
    # 生成报告
    print("生成统计报告...")
    generate_report_table(results)
    
    print(f"\n分析完成！结果已保存到 'zipf_results' 目录")
    print("生成的文件:")
    print("- zipf_verification_100000en.png: 100K样本的Zipf对数拟合图")
    print("- zipf_verification_150000en.png: 150K样本的Zipf对数拟合图")
    print("- zipf_verification_200000en.png: 200K样本的Zipf对数拟合图")
    print("- corpus_analysis_report.txt: 详细统计报告")

if __name__ == "__main__":
    main()
