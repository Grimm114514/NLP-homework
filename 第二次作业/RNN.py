import torch
import torch.nn as nn
import torch.nn.functional as F

class RNNModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(RNNModel, self).__init__()
        # 1. 词向量层
        self.embeddings = nn.Embedding(vocab_size, embed_dim)
        
        # 2. RNN 层
        # batch_first=True 让输入变成 [batch, seq_len, feature] 这种符合人类直觉的格式
        self.rnn = nn.RNN(input_size=embed_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        
        # 3. 全连接输出层
        self.linear = nn.Linear(hidden_dim, vocab_size)

    def forward(self, inputs):
        # inputs shape: [batch_size, seq_len] (句子长度可以变化)
        
        embeds = self.embeddings(inputs)
        # embeds shape: [batch_size, seq_len, embed_dim]
        
        # RNN 返回: output, h_n
        # output 包含所有时间步的输出，h_n 是最后一个时间步的隐藏状态
        output, h_n = self.rnn(embeds)
        
        # 我们取最后一个时间步的输出用于预测
        # output[:, -1, :] 表示取序列的最后一个
        last_output = output[:, -1, :]
        
        out = self.linear(last_output)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs

