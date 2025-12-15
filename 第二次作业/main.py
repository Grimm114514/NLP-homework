import numpy as np
import random

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


# --- 3. 主程序 ---

if __name__ == "__main__":
    # 配置
    FILE_PATH = 'corpus.txt'
    MODEL_DIR = './model'
    VOCAB_SIZE = 1000
    EMBED_DIM = 10
    HIDDEN_DIM = 64
    CONTEXT_SIZE = 3
    EPOCHS = 50
    LR = 0.01

    print("Step 1: 处理数据...")
    raw_text = load_data(FILE_PATH)
    train_X, train_y, vocab, idx_to_word = build_vocab_and_dataset(raw_text, VOCAB_SIZE, CONTEXT_SIZE)
    print(f"词表大小: {len(vocab)}, 样本数: {len(train_X)}")

    print("\nStep 2: 训练所有模型...")
    all_embeddings = train_all_models(
        train_X, train_y, 
        len(vocab), EMBED_DIM, HIDDEN_DIM, CONTEXT_SIZE,
        EPOCHS, LR, MODEL_DIR
    )

    # === 相似度分析与结果对比 ===
    print("\nStep 3: 计算词向量相似度并分析...")
    valid_words = [w for w in vocab.keys() if w != "<UNK>"]
    test_words = random.sample(valid_words, min(10, len(valid_words)))
    
    for model_name, embeds in all_embeddings.items():
        analyze_embeddings(model_name, embeds, vocab, idx_to_word, test_words)

    # === 演示：如何加载模型进行预测 ===
    print("\nStep 4: [演示] 加载已保存的模型...")
    loaded_lstm = LSTMModel(len(vocab), EMBED_DIM, HIDDEN_DIM)
    load_success = load_checkpoint(loaded_lstm, "lstm_model", folder=MODEL_DIR)
    
    if load_success:
        print("✅ 模型加载成功！可以进行相似度分析或预测任务。")