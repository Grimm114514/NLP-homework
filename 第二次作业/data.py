import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
import numpy as np

# 1. 模拟语料 (你需要替换成读取你的 txt 文件)
raw_text = """
The quick brown fox jumps over the lazy dog. 
I love natural language processing. 
Deep learning is fascinating and powerful.
The quick brown fox is very fast.
"""
# 实际使用时： with open('corpus.txt', 'r', encoding='utf-8') as f: raw_text = f.read()

# 2. 分词与构建词表
def build_vocab_and_data(text, vocab_size=1000):
    # 简单分词 (转小写，按空格切分)
    tokens = text.lower().replace('.', '').replace(',', '').split()
    
    # 统计词频
    word_counts = Counter(tokens)
    
    # 取最常用的 vocab_size - 1 个词 (留一个位置给 <UNK>)
    most_common = word_counts.most_common(vocab_size - 1)
    
    # 建立 word 到 index 的映射
    vocab = {"<UNK>": 0}
    for word, _ in most_common:
        vocab[word] = len(vocab)
        
    # 建立 index 到 word 的映射 (方便后面把向量转回文字)
    idx_to_word = {i: w for w, i in vocab.items()}
    
    # 把文章转成数字索引列表
    encoded_text = [vocab.get(t, 0) for t in tokens]
    
    return encoded_text, vocab, idx_to_word

# 3. 生成训练对 (X, y)
def create_dataset(encoded_text, context_size):
    data_X = []
    data_y = []
    
    # 滑动窗口
    for i in range(len(encoded_text) - context_size):
        # 输入：窗口内的词
        data_X.append(encoded_text[i : i + context_size])
        # 标签：下一个词
        data_y.append(encoded_text[i + context_size])
        
    return torch.LongTensor(data_X), torch.LongTensor(data_y)

# --- 执行预处理 ---
VOCAB_SIZE = 1000  # 题目要求
CONTEXT_SIZE = 2   # FNN用2，RNN/LSTM可以用更长，比如5或10

encoded_text, vocab, idx_to_word = build_vocab_and_data(raw_text, VOCAB_SIZE)
train_X, train_y = create_dataset(encoded_text, CONTEXT_SIZE)

print(f"训练数据样本数: {len(train_X)}")
print(f"示例 X: {train_X[0]} -> 对应单词: {[idx_to_word[i.item()] for i in train_X[0]]}")
print(f"示例 y: {train_y[0]} -> 对应单词: {idx_to_word[train_y[0].item()]}")