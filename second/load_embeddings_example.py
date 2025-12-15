"""
演示如何从JSON文件读取词向量
"""
import json
import numpy as np
import os

def load_embeddings_from_json(model_name='FNN', embeddings_dir='./embeddings'):
    """从JSON文件加载指定模型的词向量"""
    filename = os.path.join(embeddings_dir, f'{model_name}_embeddings_compact.json')
    
    if not os.path.exists(filename):
        print(f"文件不存在: {filename}")
        return None
    
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def load_all_models_embeddings(embeddings_dir='./embeddings'):
    """加载所有模型的词向量"""
    all_embeddings = {}
    
    # 查找所有的紧凑格式JSON文件
    if not os.path.exists(embeddings_dir):
        print(f"目录不存在: {embeddings_dir}")
        return all_embeddings
    
    for filename in os.listdir(embeddings_dir):
        if filename.endswith('_embeddings_compact.json'):
            model_name = filename.replace('_embeddings_compact.json', '')
            filepath = os.path.join(embeddings_dir, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                all_embeddings[model_name] = json.load(f)
    
    return all_embeddings

def get_word_vector(word, model_name='FNN', embeddings_dict=None, embeddings_dir='./embeddings'):
    """获取指定单词在指定模型中的词向量"""
    if embeddings_dict is None:
        embeddings_dict = load_embeddings_from_json(model_name, embeddings_dir)
        if embeddings_dict is None:
            return None
    
    if word not in embeddings_dict:
        print(f"单词 '{word}' 不在词表中")
        return None
    
    return np.array(embeddings_dict[word])

def cosine_similarity(vec1, vec2):
    """计算余弦相似度"""
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return np.dot(vec1, vec2) / (norm1 * norm2)

if __name__ == "__main__":
    print("=" * 60)
    print("从JSON文件读取词向量演示")
    print("=" * 60)
    
    # 加载所有模型的词向量
    embeddings = load_all_models_embeddings('./embeddings')
    
    if not embeddings:
        print("错误: 未找到词向量文件，请先运行 main.py 生成词向量")
        exit()
    
    print(f"\n可用模型: {list(embeddings.keys())}")
    
    # 查看每个模型的词表大小
    for model_name, model_data in embeddings.items():
        vocab_size = len(model_data)
        first_word = list(model_data.keys())[0]
        vector_dim = len(model_data[first_word])
        print(f"  {model_name}: {vocab_size} 个词, 向量维度: {vector_dim}")
    
    # 示例1：获取特定单词的词向量
    print("\n" + "=" * 60)
    print("示例1：查询单词的词向量")
    print("=" * 60)
    
    first_model = list(embeddings.keys())[0]
    test_word = list(embeddings[first_model].keys())[5]  # 取第6个词
    print(f"\n查询单词: '{test_word}'")
    
    for model_name in embeddings.keys():
        vec = get_word_vector(test_word, model_name, embeddings[model_name])
        if vec is not None:
            print(f"  {model_name}: {vec}")
    
    # 示例2：计算两个单词的相似度
    print("\n" + "=" * 60)
    print("示例2：计算单词相似度")
    print("=" * 60)
    
    words = list(embeddings[first_model].keys())[1:6]  # 取几个词
    print(f"\n选取的单词: {words}")
    
    for model_name in embeddings.keys():
        print(f"\n[{model_name}模型]")
        word1, word2 = words[0], words[1]
        vec1 = get_word_vector(word1, model_name, embeddings[model_name])
        vec2 = get_word_vector(word2, model_name, embeddings[model_name])
        
        if vec1 is not None and vec2 is not None:
            sim = cosine_similarity(vec1, vec2)
            print(f"  '{word1}' 和 '{word2}' 的相似度: {sim:.4f}")
    
    # 示例3：查找与给定词最相似的词
    print("\n" + "=" * 60)
    print("示例3：查找相似词")
    print("=" * 60)
    
    target_word = words[0]
    model_name = first_model
    target_vec = get_word_vector(target_word, model_name, embeddings[model_name])
    
    if target_vec is not None:
        print(f"\n查找与 '{target_word}' 最相似的词 (使用{model_name}模型):")
        
        similarities = []
        for word, vec in embeddings[model_name].items():
            if word != target_word:
                vec = np.array(vec)
                sim = cosine_similarity(target_vec, vec)
                similarities.append((word, sim))
        
        # 排序并显示前10个
        similarities.sort(key=lambda x: x[1], reverse=True)
        for word, sim in similarities[:10]:
            print(f"  {word:<20s} 相似度: {sim:.4f}")
