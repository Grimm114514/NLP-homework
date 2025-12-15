import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(LSTMModel, self).__init__()
        # 1. 词向量层
        self.embeddings = nn.Embedding(vocab_size, embed_dim)
        
        # 2. LSTM 层
        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        
        # 3. 全连接输出层
        self.linear = nn.Linear(hidden_dim, vocab_size)

    def forward(self, inputs):
        # inputs shape: [batch_size, seq_len]
        
        embeds = self.embeddings(inputs)
        
        # LSTM 返回: output, (h_n, c_n)
        # LSTM 多了一个 cell state (c_n)，但在做分类时我们通常只关注 output 或 h_n
        output, (h_n, c_n) = self.lstm(embeds)
        
        # 取最后一个时间步
        last_output = output[:, -1, :]
        
        out = self.linear(last_output)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs
