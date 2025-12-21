import json
import os

# 修改后的函数：增加了 save_to 路径参数
def build_vocab_and_indices(filepath, max_vocab, save_to='vocab.json'):
    print(f"正在读取文件: {filepath} ...")
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
        all_words = text.split()
    
    # 1. 统计与构建
    from collections import Counter
    word_counts = Counter(all_words)
    most_common = word_counts.most_common(max_vocab - 1)
    
    word_to_idx = {'<UNK>': 0}
    idx_to_word = {0: '<UNK>'}
    
    for idx, (word, count) in enumerate(most_common, start=1):
        word_to_idx[word] = idx
        idx_to_word[idx] = word
        
    # 2. === 新增：保存词表到本地 ===
    # 我们只需要保存 word_to_idx 即可，idx_to_word 可以反推
    print(f"正在保存词表到: {save_to} ...")
    with open(save_to, 'w', encoding='utf-8') as f:
        json.dump(word_to_idx, f, ensure_ascii=False, indent=2)
        
    # 3. 数字化
    corpus_indices = []
    for word in all_words:
        # 使用 get 方法，如果找不到词就返回 0 (<UNK>)
        idx = word_to_idx.get(word, 0)
        corpus_indices.append(idx)
            
    return corpus_indices, word_to_idx, idx_to_word

# ================= 使用示例 =================
if __name__ == "__main__":
    # 这一步运行完，目录下会多出一个 'vocab.json'
    indices, w2i, i2w = build_vocab_and_indices(
        'corpus_cleaned_strict.txt', 
        1000, 
        save_to='vocab.json'
    )
    print("数据准备完毕，词表已锁定！")