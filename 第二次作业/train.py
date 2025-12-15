import torch
import torch.nn as nn
import torch.optim as optim

def train_model(model, train_X, train_y, epochs=50, lr=0.01):
    # 1. 定义损失函数 (多分类问题通常用交叉熵)
    criterion = nn.CrossEntropyLoss()
    
    # 2. 定义优化器 (Adam 通常比 SGD 收敛快)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print(f"开始训练 {model.__class__.__name__} ...")
    
    for epoch in range(epochs):
        optimizer.zero_grad() # 梯度清零
        
        # 前向传播
        output = model(train_X)
        
        # 计算损失
        loss = criterion(output, train_y)
        
        # 反向传播与更新参数
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
            
    print("训练完成！\n")