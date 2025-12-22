import json
import numpy as np
import random
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def load_vectors(filepath: str) -> Dict[str, np.ndarray]:
    """加载词向量文件"""
    with open(filepath, 'r', encoding='utf-8') as f:
        vectors_dict = json.load(f)
    
    # 转换为numpy数组
    vectors = {word: np.array(vec) for word, vec in vectors_dict.items()}
    return vectors

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """计算余弦相似度"""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)

def find_most_similar(target_word: str, 
                     target_vec: np.ndarray, 
                     all_vectors: Dict[str, np.ndarray], 
                     top_k: int = 10) -> List[Tuple[str, float]]:
    """找到与目标词最相似的K个词"""
    similarities = []
    
    for word, vec in all_vectors.items():
        if word == target_word:  # 跳过自己
            continue
        sim = cosine_similarity(target_vec, vec)
        similarities.append((word, sim))
    
    # 按相似度降序排序，取前K个
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]

def plot_results(selected_words: List[str], vectors: Dict[str, np.ndarray], output_dir: str = 'figures'):
    """绘制相似度结果并保存为多张图片"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # 每张图显示5个词
    words_per_image = 5
    num_images = (len(selected_words) + words_per_image - 1) // words_per_image
    
    for img_idx in range(num_images):
        start_idx = img_idx * words_per_image
        end_idx = min(start_idx + words_per_image, len(selected_words))
        words_subset = selected_words[start_idx:end_idx]
        
        # 创建子图（1行5列或更少）
        num_words = len(words_subset)
        fig, axes = plt.subplots(1, num_words, figsize=(5*num_words, 6))
        
        # 如果只有一个词，axes不是数组
        if num_words == 1:
            axes = [axes]
        
        fig.suptitle(f'词向量相似度分析 - 第{img_idx+1}组', fontsize=16, fontweight='bold', y=0.98)
        
        for idx, word in enumerate(words_subset):
            ax = axes[idx]
            
            # 找到最相似的10个词
            similar_words = find_most_similar(word, vectors[word], vectors, top_k=10)
            
            # 提取词和相似度
            words_list = [w for w, _ in similar_words]
            similarities = [s for _, s in similar_words]
            
            # 绘制水平条形图
            y_pos = np.arange(len(words_list))
            colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(words_list)))
            
            ax.barh(y_pos, similarities, color=colors)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(words_list, fontsize=11)
            ax.set_xlabel('相似度', fontsize=11)
            ax.set_title(f'目标词: {word}', fontsize=13, fontweight='bold')
            ax.set_xlim([0, 1])
            ax.invert_yaxis()  # 最相似的在顶部
            ax.grid(axis='x', alpha=0.3)
            
            # 在条形上添加数值标签
            for i, (w, sim) in enumerate(similar_words):
                ax.text(sim + 0.01, i, f'{sim:.3f}', va='center', fontsize=9)
        
        plt.tight_layout()
        
        # 保存图片
        output_file = os.path.join(output_dir, f'word_similarity_group_{img_idx+1}.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"图片已保存至: {output_file}")
        plt.close()

def main():
    # 加载词向量
    print("正在加载词向量...")
    vectors = load_vectors('vector/rnn_vectors.json')
    print(f"已加载 {len(vectors)} 个词的向量\n")
    
    # 随机选择20个词（排除<UNK>）
    words = [w for w in vectors.keys() if w != '<UNK>']
    random.seed(42)  # 设置随机种子以便复现
    selected_words = random.sample(words, min(20, len(words)))
    
    print("=" * 80)
    print("随机抽取的20个词及其最相似的10个词")
    print("=" * 80)
    
    # 对每个选中的词，找出最相似的10个词
    all_results = []
    for idx, word in enumerate(selected_words, 1):
        print(f"\n【{idx}】目标词: {word}")
        print("-" * 60)
        
        similar_words = find_most_similar(word, vectors[word], vectors, top_k=10)
        all_results.append((word, similar_words))
        
        for rank, (similar_word, similarity) in enumerate(similar_words, 1):
            print(f"  {rank:2d}. {similar_word:8s}  相似度: {similarity:.6f}")
    
    print("\n" + "=" * 80)
    print("分析完成！")
    
    # 绘制并保存结果
    print("\n正在生成可视化图表...")
    plot_results(selected_words, vectors)

if __name__ == "__main__":
    main()

