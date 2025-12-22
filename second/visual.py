import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib.font_manager as fm

# 1. 配置中文字体 (这是关键，否则汉字会乱码)
# Windows通常用 'SimHei', Mac通常用 'Arial Unicode MS' 或 'Heiti TC'
# 如果报错找不到字体，请手动指定一个你电脑上存在的 ttf 文件路径
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei'] 
plt.rcParams['axes.unicode_minus'] = False # 解决负号显示问题

def visualize_embeddings(file_path, model_name="Model"):
    # 2. 读取数据
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    words = list(data.keys())
    vectors = np.array(list(data.values()))
    
    # 为了图表清晰，如果词太多，我们可以只取前300个词来画，或者手动挑选特定类别的词
    # 这里演示取前 300 个词（假设你的 json 是按频率排序的）
    limit = 300 
    words = words[:limit]
    vectors = vectors[:limit]

    print(f"正在对 {model_name} 进行 t-SNE 降维计算，请稍候...")
    
    # 3. 使用 t-SNE 将 15维 降到 2维
    # n_components=2 表示降到2维
    # perplexity 建议在 5-50 之间，数据量少时建议设小一点，比如 5-10
    tsne = TSNE(n_components=2, perplexity=10, random_state=42, init='pca', learning_rate='auto')
    vectors_2d = tsne.fit_transform(vectors)

    # 4. 画图
    plt.figure(figsize=(14, 10))
    
    # 画散点
    x = vectors_2d[:, 0]
    y = vectors_2d[:, 1]
    plt.scatter(x, y, alpha=0.6, color='steelblue')

    # 给每个点标上文字
    for i, word in enumerate(words):
        plt.text(x[i], y[i], word, fontsize=9, alpha=0.8)

    plt.title(f"{model_name} 词向量可视化 (t-SNE)", fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.show()

# --- 使用示例 ---
# 请把下面的文件名换成你真实的 json 文件名
visualize_embeddings('vector/cnn_vectors.json', 'CNN Model')
visualize_embeddings('vector/rnn_vectors.json', 'RNN Model')