import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
import numpy as np
import re

# 词形还原工具
try:
    from nltk.stem import WordNetLemmatizer
    from nltk.corpus import wordnet
    import nltk
    # 确保下载了必要的数据
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        print("正在下载 WordNet 数据...")
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
    USE_LEMMATIZATION = True
except ImportError:
    print("⚠️ 未安装 NLTK，将不进行词形还原。安装方法: pip install nltk")
    USE_LEMMATIZATION = False

# 1. 加载数据
def load_data(file_path):
    """从文件中读取语料"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

# 2. 词形还原函数
def lemmatize_tokens(tokens):
    """对tokens进行词形还原"""
    if not USE_LEMMATIZATION:
        return tokens
    
    lemmatizer = WordNetLemmatizer()
    lemmatized = []
    for token in tokens:
        # 尝试作为名词、动词、形容词还原
        lemma = lemmatizer.lemmatize(token, pos='v')  # 动词
        if lemma == token:
            lemma = lemmatizer.lemmatize(token, pos='n')  # 名词
        if lemma == token:
            lemma = lemmatizer.lemmatize(token, pos='a')  # 形容词
        lemmatized.append(lemma)
    return lemmatized

# 3. 分词与构建词表
def build_vocab_and_data(text, vocab_size=1000, use_lemmatization=True):
    # 清理文本并分词 (转小写，移除标点，按空格切分)
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # 移除标点
    tokens = text.split()
    
    # 词形还原
    if use_lemmatization and USE_LEMMATIZATION:
        print("⚙️ 正在进行词形还原...")
        original_count = len(set(tokens))
        tokens = lemmatize_tokens(tokens)
        lemmatized_count = len(set(tokens))
        print(f"   词形还原前: {original_count} 个唯一词 -> 词形还原后: {lemmatized_count} 个唯一词")
    
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

# 5. 组合函数：构建词表并生成训练数据
def build_vocab_and_dataset(text, vocab_size, context_size, use_lemmatization=True):
    """
    整合词表构建和数据集生成
    返回: train_X, train_y, vocab, idx_to_word
    """
    encoded_text, vocab, idx_to_word = build_vocab_and_data(text, vocab_size, use_lemmatization)
    train_X, train_y = create_dataset(encoded_text, context_size)
    return train_X, train_y, vocab, idx_to_word

# --- 执行预处理 (可选的测试代码) ---
if __name__ == "__main__":
    VOCAB_SIZE = 1000  # 题目要求
    CONTEXT_SIZE = 2   # FNN用2，RNN/LSTM可以用更长，比如5或10

    raw_text = load_data('corpus_cleaned.txt')
    encoded_text, vocab, idx_to_word = build_vocab_and_data(raw_text, VOCAB_SIZE)
    train_X, train_y = create_dataset(encoded_text, CONTEXT_SIZE)

    print(f"训练数据样本数: {len(train_X)}")
    print(f"示例 X: {train_X[0]} -> 对应单词: {[idx_to_word[i.item()] for i in train_X[0]]}")
    print(f"示例 y: {train_y[0]} -> 对应单词: {idx_to_word[train_y[0].item()]}")