import torch
import torch.nn as nn
import torch.nn.functional as F

class FNNModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, context_size=2):
        super(FNNModel, self).__init__()
        # 1. 词向量层 (这就是你要提取的结果)
        self.embeddings = nn.Embedding(vocab_size, embed_dim)
        
        # 2. 隐藏层 (第1层网络)
        # 输入维度是: 上下文单词数量 * 词向量维度
        self.linear1 = nn.Linear(context_size * embed_dim, hidden_dim)
        
        # 3. 输出层 (第2层网络)
        # 预测下一个词是词表中哪一个
        self.linear2 = nn.Linear(hidden_dim, vocab_size)

    def forward(self, inputs):
        # inputs shape: [batch_size, context_size] (例如: 这里的 context_size=2)
        
        # 获得词向量
        embeds = self.embeddings(inputs) 
        # embeds shape: [batch_size, context_size, embed_dim]
        
        # 展平 (Flatten)，把上下文的向量拼成一个长条
        embeds = embeds.view((inputs.shape[0], -1)) 
        
        # 前向传播
        out = F.relu(self.linear1(embeds)) # 激活函数
        out = self.linear2(out)            # 得到最终分数
        
        # log_softmax 用于计算交叉熵损失
        log_probs = F.log_softmax(out, dim=1)
        return log_probs

