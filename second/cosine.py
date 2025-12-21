import json
import numpy as np
import random
from numpy.linalg import norm

# ================= 配置 =================
RNN_FILE = 'vector/rnn_vectors.json'
CNN_FILE = 'vector/cnn_vectors.json'
TOP_K = 10     # 找前10个相似词
SAMPLE_NUM = 20 # 随机抽取多少个词进行测试

# ================= 核心函数 =================

def load_vectors(filepath):
    print(f"正在加载: {filepath} ...")
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # 将 list 转换为 numpy array，方便后续计算
    # 结果是一个字典: {'中国': np.array([0.1, ...]), ...}
    vectors = {k: np.array(v) for k, v in data.items()}
    return vectors

def cosine_similarity(vec_a, vec_b):
    # 余弦相似度公式: (A . B) / (|A| * |B|)
    # 加上 1e-8 防止分母为 0
    return np.dot(vec_a, vec_b) / (norm(vec_a) * norm(vec_b) + 1e-8)

def find_similar_words(target_word, all_vectors, top_k=10):
    """
    输入一个词，返回最相似的 top_k 个词及其相似度
    """
    if target_word not in all_vectors:
        return []

    target_vec = all_vectors[target_word]
    similarities = []

    for word, vec in all_vectors.items():
        if word == target_word: continue # 跳过自己
        
        score = cosine_similarity(target_vec, vec)
        similarities.append((word, score))

    # 按分数从高到低排序
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    return similarities[:top_k]

# ================= 主程序 =================
if __name__ == "__main__":
    # 1. 加载两个模型的向量
    rnn_vectors = load_vectors(RNN_FILE)
    cnn_vectors = load_vectors(CNN_FILE)
    
    # 确保两个模型的词表是一样的
    vocab_keys = list(rnn_vectors.keys())
    
    # 2. 随机选取 20 个词 (过滤掉 <UNK> 和 <PAD>)
    # 你也可以手动指定列表: test_words = ['中国', '经济', '发展'...]
    candidates = [w for w in vocab_keys if w not in ['<UNK>', '<PAD>']]
    test_words = random.sample(candidates, SAMPLE_NUM)
    
    # 如果想测试特定词，取消下面这行的注释
    # test_words = ['中国', '经济', '发展', '我们', '问题'] 

    print(f"\n开始对比分析 (共 {len(test_words)} 个测试词)...\n")
    print("="*80)

    for target in test_words:
        print(f"🔴 目标词: 【 {target} 】")
        
        # 计算 RNN 的结果
        rnn_sims = find_similar_words(target, rnn_vectors, TOP_K)
        # 计算 CNN 的结果
        cnn_sims = find_similar_words(target, cnn_vectors, TOP_K)
        
        # --- 格式化打印表格 ---
        print(f"{'Rank':<5} | {'RNN 预测结果':<25} | {'CNN 预测结果':<25}")
        print("-" * 60)
        
        for i in range(TOP_K):
            # 获取 RNN 的第 i 个结果
            r_word, r_score = rnn_sims[i] if i < len(rnn_sims) else ("-", 0)
            # 获取 CNN 的第 i 个结果
            c_word, c_score = cnn_sims[i] if i < len(cnn_sims) else ("-", 0)
            
            print(f"{i+1:<5} | {r_word:<15} ({r_score:.3f})   | {c_word:<15} ({c_score:.3f})")
            
        print("="*80 + "\n")