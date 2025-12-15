import torch
import os
from FNN import FNNModel
from RNN import RNNModel
from LSTM import LSTMModel
from train import train_model


def ensure_dir(directory):
    """如果目录不存在，则创建"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"创建文件夹: {directory}")


def save_checkpoint(model, model_name, folder='./model'):
    """保存模型参数"""
    ensure_dir(folder)
    filename = f"{model_name}.pth"
    path = os.path.join(folder, filename)
    torch.save(model.state_dict(), path)
    print(f"✅ 模型 [{model_name}] 已保存至: {path}")


def load_checkpoint(model, model_name, folder='./model'):
    """加载模型参数 (In-place 修改)"""
    path = os.path.join(folder, f"{model_name}.pth")
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))
        model.eval()
        print(f"✅ 模型 [{model_name}] 加载成功!")
        return True
    else:
        print(f"❌ 未找到模型文件: {path}")
        return False


def train_all_models(train_X, train_y, vocab_size, embed_dim, hidden_dim, context_size, epochs, lr, model_dir='./model'):
    """训练所有模型并返回训练后的embeddings (支持CUDA)"""
    all_embeddings = {}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 数据迁移到设备
    train_X = train_X.to(device) if hasattr(train_X, 'to') else train_X
    train_y = train_y.to(device) if hasattr(train_y, 'to') else train_y

    # 训练 FNN
    print("\n训练 FNN...")
    fnn = FNNModel(vocab_size, embed_dim, hidden_dim, context_size=context_size).to(device)
    train_model(fnn, train_X, train_y, epochs=epochs, lr=lr, device=device)
    save_checkpoint(fnn, "fnn_model", folder=model_dir)
    all_embeddings['FNN'] = fnn.embeddings.weight.data.cpu().numpy()

    # 训练 RNN
    print("\n训练 RNN...")
    rnn = RNNModel(vocab_size, embed_dim, hidden_dim).to(device)
    train_model(rnn, train_X, train_y, epochs=epochs, lr=lr, device=device)
    save_checkpoint(rnn, "rnn_model", folder=model_dir)
    all_embeddings['RNN'] = rnn.embeddings.weight.data.cpu().numpy()

    # 训练 LSTM
    print("\n训练 LSTM...")
    lstm = LSTMModel(vocab_size, embed_dim, hidden_dim).to(device)
    train_model(lstm, train_X, train_y, epochs=epochs, lr=lr, device=device)
    save_checkpoint(lstm, "lstm_model", folder=model_dir)
    all_embeddings['LSTM'] = lstm.embeddings.weight.data.cpu().numpy()

    return all_embeddings
