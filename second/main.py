import numpy as np
import random
import json
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# --- 1. 导入模块 ---
from data import load_data, build_vocab_and_dataset
from training import train_all_models, load_checkpoint
from FNN import FNNModel
from RNN import RNNModel
from LSTM import LSTMModel
from train import train_model

# --- 2. 相似度计算与分析工具函数 ---

def get_cosine_similarity(vec1, vec2):
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0: return 0.0
    return np.dot(vec1, vec2) / (norm1 * norm2)

def analyze_embeddings(model_name, embeddings, vocab, idx_to_word, test_words):
    print(f"\n======== 模型分析报告: {model_name} ========")
    for word in test_words:
        if word not in vocab: continue
        word_idx = vocab[word]
        word_vec = embeddings[word_idx]
        
        sim_scores = []
        for i in range(len(vocab)):
            if i == word_idx or i == 0: continue
            other_vec = embeddings[i]
            score = get_cosine_similarity(word_vec, other_vec)
            sim_scores.append((i, score))
        
        sim_scores.sort(key=lambda x: x[1], reverse=True)
        top_5 = sim_scores[:5] # 只看前5个，节省篇幅
        
        result_str = ", ".join([f"{idx_to_word[idx]}({score:.2f})" for idx, score in top_5])
        print(f"[{word}] -> {result_str}")

def visualize_embeddings(all_embeddings, vocab, idx_to_word, num_words=50):
    """使用PCA将词向量降维到2D并可视化"""
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    
    num_models = len(all_embeddings)
    fig, axes = plt.subplots(1, num_models, figsize=(8*num_models, 6))
    if num_models == 1:
        axes = [axes]
    
    for idx, (model_name, embeddings) in enumerate(all_embeddings.items()):
        # 使用PCA降维到2D
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embeddings[:num_words])
        
        # 绘制散点图
        axes[idx].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.6, s=100)
        
        # 标注词语
        for i in range(min(num_words, len(embeddings_2d))):
            word = idx_to_word.get(i, f'word_{i}')
            axes[idx].annotate(word, (embeddings_2d[i, 0], embeddings_2d[i, 1]), 
                             fontsize=8, alpha=0.7)
        
        axes[idx].set_title(f'{model_name} 词向量可视化 (PCA)', fontsize=14)
        axes[idx].set_xlabel('PCA维度1')
        axes[idx].set_ylabel('PCA维度2')
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('word_embeddings_visualization.png', dpi=150, bbox_inches='tight')
    print("\n📊 词向量可视化已保存至: word_embeddings_visualization.png")
    plt.show()

def visualize_similarity_matrix(embeddings, vocab, idx_to_word, model_name, num_words=30):
    """绘制词向量相似度矩阵热图"""
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 计算相似度矩阵
    selected_embeddings = embeddings[:num_words]
    similarity_matrix = np.zeros((num_words, num_words))
    
    for i in range(num_words):
        for j in range(num_words):
            similarity_matrix[i, j] = get_cosine_similarity(selected_embeddings[i], selected_embeddings[j])
    
    # 绘制热图
    plt.figure(figsize=(12, 10))
    plt.imshow(similarity_matrix, cmap='YlOrRd', aspect='auto')
    plt.colorbar(label='余弦相似度')
    
    # 设置标签
    words = [idx_to_word.get(i, f'word_{i}') for i in range(num_words)]
    plt.xticks(range(num_words), words, rotation=90, fontsize=8)
    plt.yticks(range(num_words), words, fontsize=8)
    
    plt.title(f'{model_name} 词向量相似度矩阵', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{model_name}_similarity_matrix.png', dpi=150, bbox_inches='tight')
    print(f"📊 {model_name} 相似度矩阵已保存至: {model_name}_similarity_matrix.png")
    plt.show()

def save_embeddings_to_json(all_embeddings, idx_to_word, filename='word_embeddings.json'):
    """将所有模型的词向量保存到JSON文件"""
    output_data = {}
    
    for model_name, embeddings in all_embeddings.items():
        model_data = {}
        for idx, word in idx_to_word.items():
            # 将numpy数组转换为列表以便JSON序列化
            vector = embeddings[idx].tolist()
            model_data[word] = {
                'index': idx,
                'vector': vector
            }
        output_data[model_name] = model_data
    
    # 保存到JSON文件
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"💾 词向量已保存至: {filename}")
    
    # 统计信息
    total_words = len(idx_to_word)
    vector_dim = len(embeddings[0])
    print(f"   包含 {len(all_embeddings)} 个模型")
    print(f"   每个模型 {total_words} 个词，每个词向量维度: {vector_dim}")

def save_embeddings_compact(all_embeddings, idx_to_word, filename='word_embeddings_compact.json'):
    """保存紧凑格式的词向量（只保存词和向量）"""
    output_data = {}
    
    for model_name, embeddings in all_embeddings.items():
        # 简化格式：{word: [vector]}
        model_data = {
            word: embeddings[idx].tolist() 
            for idx, word in idx_to_word.items()
        }
        output_data[model_name] = model_data
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"💾 紧凑格式词向量已保存至: {filename}")


# --- 3. 主程序 ---

if __name__ == "__main__":
    # 配置
    FILE_PATH = 'cleaned_corpus.txt'
    MODEL_DIR = './model'
    VOCAB_SIZE = 1000
    EMBED_DIM = 10
    HIDDEN_DIM = 64
    CONTEXT_SIZE = 3
    EPOCHS = 5000
    LR = 0.01

    print("Step 1: 处理数据...")
    raw_text = load_data(FILE_PATH)
    train_X, train_y, vocab, idx_to_word = build_vocab_and_dataset(raw_text, VOCAB_SIZE, CONTEXT_SIZE)
    print(f"词表大小: {len(vocab)}, 样本数: {len(train_X)}")
    
    # 显示词表（前50个词）
    print("\n[词表预览] 前50个高频词:")
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])[:50]
    vocab_preview = ", ".join([f"{word}({idx})" for word, idx in sorted_vocab])
    print(vocab_preview)

    print("\nStep 2: 训练所有模型...")
    all_embeddings = train_all_models(
        train_X, train_y, 
        len(vocab), EMBED_DIM, HIDDEN_DIM, CONTEXT_SIZE,
        EPOCHS, LR, MODEL_DIR
    )

    # === 显示词向量 ===
    print("\nStep 3: 显示词向量...")
    for model_name, embeds in all_embeddings.items():
        print(f"\n[{model_name}模型] 词向量维度: {embeds.shape}")
        # 显示前10个词的词向量
        print(f"前10个词的词向量:")
        for i in range(min(10, len(embeds))):
            word = idx_to_word[i]
            vec_str = np.array2string(embeds[i], precision=3, suppress_small=True, max_line_width=100)
            print(f"  {word:15s} -> {vec_str}")
    
    # === 相似度分析与结果对比 ===
    print("\nStep 4: 计算词向量相似度并分析...")
    valid_words = [w for w in vocab.keys() if w != "<UNK>"]
    test_words = random.sample(valid_words, min(10, len(valid_words)))
    
    for model_name, embeds in all_embeddings.items():
        analyze_embeddings(model_name, embeds, vocab, idx_to_word, test_words)

    # === 保存词向量到JSON ===
    print("\nStep 5: 保存词向量到JSON文件...")
    save_embeddings_to_json(all_embeddings, idx_to_word, 'word_embeddings.json')
    save_embeddings_compact(all_embeddings, idx_to_word, 'word_embeddings_compact.json')
    
    # === 可视化 ===
    print("\nStep 6: 生成可视化图表...")
    
    # 1. 词向量散点图（PCA降维）
    visualize_embeddings(all_embeddings, vocab, idx_to_word, num_words=50)
    
    # 2. 相似度矩阵热图
    for model_name, embeds in all_embeddings.items():
        visualize_similarity_matrix(embeds, vocab, idx_to_word, model_name, num_words=30)
    
    # === 演示：如何加载模型进行预测 ===
    print("\nStep 7: [演示] 加载已保存的模型...")
    loaded_lstm = LSTMModel(len(vocab), EMBED_DIM, HIDDEN_DIM)
    load_success = load_checkpoint(loaded_lstm, "lstm_model", folder=MODEL_DIR)
    
    if load_success:
        print("✅ 模型加载成功！可以进行相似度分析或预测任务。")