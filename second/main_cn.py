import torch
import numpy as np
import random
import os
from collections import Counter

# --- 1. 导入原有模块 (保持名称不变) ---
from FNN import FNNModel
from RNN import RNNModel
from LSTM import LSTMModel
from train import train_model
from preprocess_cn import ChineseDataCleaner  # 导入刚才写的清洗器

# --- 2. 辅助工具 ---

def save_checkpoint(model, model_name, folder='./CN_model'):
    """保存模型，自动加入 CN 前缀文件夹"""
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f"创建文件夹: {folder}")
        
    # 保存文件名为 CN_xxx.pth
    filename = f"CN_{model_name}.pth"
    path = os.path.join(folder, filename)
    torch.save(model.state_dict(), path)
    print(f"✅ [CN] 模型已保存: {path}")

def build_vocab_and_dataset(text_content, vocab_size, context_size):
    """构建中文数据集"""
    tokens = text_content.split() # 已经是空格分隔的中文词了
    
    word_counts = Counter(tokens)
    most_common = word_counts.most_common(vocab_size - 1)
    
    vocab = {"<UNK>": 0}
    for word, _ in most_common:
        vocab[word] = len(vocab)
        
    idx_to_word = {i: w for w, i in vocab.items()}
    encoded_text = [vocab.get(t, 0) for t in tokens]
    
    # 截取一部分数据防止内存溢出 (视你电脑性能而定，可改为全量)
    limit = min(len(encoded_text), 500000)
    encoded_text = encoded_text[:limit]
    
    data_X = []
    data_y = []
    for i in range(len(encoded_text) - context_size):
        data_X.append(encoded_text[i : i + context_size])
        data_y.append(encoded_text[i + context_size])
        
    return torch.LongTensor(data_X), torch.LongTensor(data_y), vocab, idx_to_word

def analyze_embeddings(model_name, embeddings, vocab, idx_to_word, test_words):
    """分析相似度"""
    print(f"\n======== [CN] 模型分析: {model_name} ========")
    for word in test_words:
        if word not in vocab:
            continue
        
        word_idx = vocab[word]
        word_vec = embeddings[word_idx]
        
        sim_scores = []
        # 计算余弦相似度
        vec_norm = np.linalg.norm(word_vec)
        if vec_norm == 0: continue

        for i in range(len(vocab)):
            if i == word_idx or i == 0: continue
            other_vec = embeddings[i]
            other_norm = np.linalg.norm(other_vec)
            if other_norm == 0: continue
            
            score = np.dot(word_vec, other_vec) / (vec_norm * other_norm)
            sim_scores.append((i, score))
        
        sim_scores.sort(key=lambda x: x[1], reverse=True)
        top_5 = sim_scores[:5]
        
        result_str = ", ".join([f"{idx_to_word[idx]}({score:.2f})" for idx, score in top_5])
        print(f"词汇 [{word}] -> 近义词: {result_str}")

# --- 3. 主执行流程 ---

if __name__ == "__main__":
    # === 参数配置 ===
    RAW_FILE = 'ChineseCorpus199801_half.txt'
    CLEAN_FILE = 'CN_corpus_cleaned.txt'
    MODEL_DIR = './CN_model'  # 模型保存路径
    
    VOCAB_SIZE = 2500         # 中文常用词较多，建议设大一点
    EMBED_DIM = 20
    HIDDEN_DIM = 64
    CONTEXT_SIZE = 2
    EPOCHS = 2000
    LR = 0.01

    # === Step 0: 清洗数据 (只保留中文) ===
    print("Step 0: 执行中文清洗...")
    cleaner = ChineseDataCleaner(RAW_FILE, CLEAN_FILE)
    text_content = cleaner.clean_text()
    
    if not text_content:
        print("❌ 错误：数据为空，请检查原始文件路径。")
        exit()

    # === Step 1: 预处理 ===
    print("\nStep 1: 构建训练集...")
    train_X, train_y, vocab, idx_to_word = build_vocab_and_dataset(text_content, VOCAB_SIZE, CONTEXT_SIZE)
    print(f"词表大小: {len(vocab)}, 样本数: {len(train_X)}")

    all_embeddings = {}

    # === Step 2: 训练三个模型 (调用原始模块) ===
    
    # 1. CN_FNN
    print("\n>>> 开始训练 FNN (CN)...")
    fnn = FNNModel(len(vocab), EMBED_DIM, HIDDEN_DIM, context_size=CONTEXT_SIZE)
    train_model(fnn, train_X, train_y, epochs=EPOCHS, lr=LR)
    save_checkpoint(fnn, "fnn_model", folder=MODEL_DIR) # 保存为 CN_fnn_model.pth
    all_embeddings['CN_FNN'] = fnn.embeddings.weight.data.cpu().numpy()

    # 2. CN_RNN
    print("\n>>> 开始训练 RNN (CN)...")
    rnn = RNNModel(len(vocab), EMBED_DIM, HIDDEN_DIM)
    train_model(rnn, train_X, train_y, epochs=EPOCHS, lr=LR)
    save_checkpoint(rnn, "rnn_model", folder=MODEL_DIR) # 保存为 CN_rnn_model.pth
    all_embeddings['CN_RNN'] = rnn.embeddings.weight.data.cpu().numpy()

    # 3. CN_LSTM
    print("\n>>> 开始训练 LSTM (CN)...")
    lstm = LSTMModel(len(vocab), EMBED_DIM, HIDDEN_DIM)
    train_model(lstm, train_X, train_y, epochs=EPOCHS, lr=LR)
    save_checkpoint(lstm, "lstm_model", folder=MODEL_DIR) # 保存为 CN_lstm_model.pth
    all_embeddings['CN_LSTM'] = lstm.embeddings.weight.data.cpu().numpy()

    # === Step 3: 结果对比 ===
    print("\nStep 3: 中文词向量效果展示")
    
    # 选取一些典型的中文测试词
    test_words = ["中国", "经济", "发展", "北京", "生活", "学生"]
    # 再随机加几个
    valid_words = [w for w in vocab.keys() if w != "<UNK>"]
    if len(valid_words) > 5:
        test_words += random.sample(valid_words, 5)

    for name, embeds in all_embeddings.items():
        analyze_embeddings(name, embeds, vocab, idx_to_word, test_words)

    print("\nCN 训练任务全部完成！")