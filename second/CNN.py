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
    model_save_path = 'model/cnn_model.pth'    
    json_save_path = 'vector/cnn_vectors.json' 
    
    # 训练参数
    embed_dim = 15    # 必须和 RNN 保持一致以便对比
    seq_len = 5       # 窗口大小
    batch_size = 64
    epochs = 30
    lr = 0.001

os.makedirs('model', exist_ok=True)
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
        x = torch.tensor(self.data[idx : idx + self.seq_len], dtype=torch.long)
        y = torch.tensor(self.data[idx + self.seq_len], dtype=torch.long)
        return x, y

def load_data():
    print("正在加载词表和语料...")
    with open(Config.vocab_file, 'r', encoding='utf-8') as f:
        word_to_idx = json.load(f)
    idx_to_word = {v: k for k, v in word_to_idx.items()}
    
    with open(Config.corpus_file, 'r', encoding='utf-8') as f:
        words = f.read().split()
    
    indices = [word_to_idx.get(w, 0) for w in words]
    print(f"加载完成！词表大小: {len(word_to_idx)}, 语料长度: {len(indices)}")
    return indices, word_to_idx, idx_to_word

# ================= 2. CNN 模型定义 =================
class CNNModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, seq_len):
        super(CNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # 1D 卷积层
        # in_channels = 词向量维度
        # out_channels = 32 (提取出的特征数量)
        # kernel_size = 2 (每次看2个相邻词，类似Bigram)
        self.conv1 = nn.Conv1d(in_channels=embed_dim, out_channels=32, kernel_size=2)
        
        # 自适应最大池化，无论前面多长，最后都压缩成1个数
        self.pool = nn.AdaptiveMaxPool1d(1)
        
        # 全连接层
        self.fc = nn.Linear(32, vocab_size)

    def forward(self, x):
        # x: [batch, seq_len]
        embeds = self.embedding(x)          # -> [batch, seq_len, embed_dim]
        
        # 关键步骤：CNN 要求的输入格式是 [batch, channel, length]
        # 所以要把 embed_dim 换到中间去
        embeds = embeds.permute(0, 2, 1)    # -> [batch, embed_dim, seq_len]
        
        x = torch.relu(self.conv1(embeds))
        x = self.pool(x).squeeze(-1)        # -> [batch, 32]
        
        return self.fc(x)

# ================= 3. 主训练流程 =================
if __name__ == '__main__':
    # 1. 准备数据
    data_indices, w2i, i2w = load_data()
    dataset = LMDataset(data_indices, Config.seq_len)
    dataloader = DataLoader(dataset, batch_size=Config.batch_size, shuffle=True)
    
    # 2. 初始化模型
    model = CNNModel(len(w2i), Config.embed_dim, Config.seq_len).to(device)
    optimizer = optim.Adam(model.parameters(), lr=Config.lr)
    criterion = nn.CrossEntropyLoss()

    # 3. 开始训练
    print(f">>> 开始训练 CNN ({Config.epochs} 轮) <<<")
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
    print(f"CNN模型已保存至: {Config.model_save_path}")

    # B. 保存词向量 (JSON)
    weights = model.embedding.weight.data.cpu().numpy()
    vectors_dict = {}
    
    for i in range(len(w2i)):
        word = i2w[i]
        vectors_dict[word] = weights[i].tolist()
        
    with open(Config.json_save_path, 'w', encoding='utf-8') as f:
        json.dump(vectors_dict, f, ensure_ascii=False)
        
    print(f"CNN词向量(JSON)已保存至: {Config.json_save_path}")

    # ================= 5. 绘制损失函数图像 =================
    def plot_loss_curve(losses):
        plt.figure(figsize=(8, 6))
        plt.plot(losses, label='Training Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss Curve')
        plt.legend()
        plt.grid()
        plt.savefig('figures/CNN_loss_curve.png')  # 保存图像
        plt.show()

    # 绘制损失函数图像
    plot_loss_curve(losses)