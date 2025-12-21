import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import time
import os
import numpy as np
import matplotlib.pyplot as plt

# ================= 配置参数 =================
class Config:
    # 输入文件
    corpus_file = 'corpus_cleaned_strict.txt'
    vocab_file = 'vocab.json'
    
    # 输出文件 (保存到 model 文件夹)
    model_save_path = 'model/rnn_model.pth'    
    json_save_path = 'vector/rnn_vectors.json' 
    
    # 训练参数
    embed_dim = 15    # 词向量维度
    hidden_dim = 32   # RNN隐层维度
    seq_len = 5       # 窗口大小
    batch_size = 64
    epochs = 30       # 训练轮数
    lr = 0.001

# 自动创建输出文件夹
os.makedirs('model', exist_ok=True)
os.makedirs('figures', exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"当前使用的设备: {device}")

# ================= 1. 数据加载与 Dataset =================
class LMDataset(Dataset):
    def __init__(self, data, seq_len):
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        # 输入: 连续的 seq_len 个词
        x = torch.tensor(self.data[idx : idx + self.seq_len], dtype=torch.long)
        # 标签: 下一个词
        y = torch.tensor(self.data[idx + self.seq_len], dtype=torch.long)
        return x, y

def load_data():
    print("正在加载词表和语料...")
    # 加载词表
    with open(Config.vocab_file, 'r', encoding='utf-8') as f:
        word_to_idx = json.load(f)
    idx_to_word = {v: k for k, v in word_to_idx.items()}
    
    # 加载语料
    with open(Config.corpus_file, 'r', encoding='utf-8') as f:
        words = f.read().split()
    
    # 数字化：找不到的词用 0 (<UNK>) 代替
    indices = [word_to_idx.get(w, 0) for w in words]
    
    print(f"加载完成！词表大小: {len(word_to_idx)}, 语料长度: {len(indices)}")
    return indices, word_to_idx, idx_to_word

# ================= 2. RNN 模型定义 =================
class RNNModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # batch_first=True: 输入形状为 (batch, seq, feature)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        embeds = self.embedding(x)
        out, _ = self.rnn(embeds)
        # 取最后一个时间步的输出用于预测
        last_out = out[:, -1, :] 
        return self.fc(last_out)

# ================= 3. 主训练流程 =================
if __name__ == '__main__':
    # 1. 准备数据
    data_indices, w2i, i2w = load_data()
    dataset = LMDataset(data_indices, Config.seq_len)
    dataloader = DataLoader(dataset, batch_size=Config.batch_size, shuffle=True)
    
    # 2. 初始化模型
    model = RNNModel(len(w2i), Config.embed_dim, Config.hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=Config.lr)
    criterion = nn.CrossEntropyLoss()

    # 3. 开始训练
    print(f">>> 开始训练 RNN ({Config.epochs} 轮) <<<")
    start_time = time.time()
    
    # 在训练流程中记录每轮的平均损失
    losses = []
    
    for epoch in range(Config.epochs):
        total_loss = 0
        model.train()
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        losses.append(avg_loss)  # 记录每轮的平均损失
        print(f"Epoch {epoch+1}/{Config.epochs} | Loss: {avg_loss:.4f}")

    print(f"训练耗时: {time.time()-start_time:.2f}s")

    # ================= 4. 保存结果 =================
    
    # A. 保存模型权重 (pth)
    torch.save(model.state_dict(), Config.model_save_path)
    print(f"RNN模型已保存至: {Config.model_save_path}")

    # B. 保存词向量 (JSON) - 方便查找
    weights = model.embedding.weight.data.cpu().numpy()
    vectors_dict = {}
    
    for i in range(len(w2i)):
        word = i2w[i]
        # numpy 数组转 list 才能存 json
        vectors_dict[word] = weights[i].tolist()
        
    with open(Config.json_save_path, 'w', encoding='utf-8') as f:
        json.dump(vectors_dict, f, ensure_ascii=False) # 去掉 indent 减小体积
        
    print(f"RNN词向量(JSON)已保存至: {Config.json_save_path}")

    # ================= 5. 绘制损失函数图像 =================
    def plot_loss_curve(losses):
        plt.figure(figsize=(8, 6))
        plt.plot(losses, label='Training Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss Curve')
        plt.legend()
        plt.grid()
        plt.savefig('figures/RNN_loss_curve.png')  # 保存图像
        plt.show()

    # 绘制损失函数图像
    plot_loss_curve(losses)