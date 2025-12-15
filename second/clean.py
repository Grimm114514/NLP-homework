import re

def aggressive_clean(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()

    # --- 1. 维基百科特定清洗 (去元数据) ---
    text = re.sub(r'\\', ' ', text)
    text = re.sub(r'\[.*?\]', ' ', text)
    text = re.sub(r'This .*? is a stub\.', '', text, flags=re.IGNORECASE)
    text = re.sub(r'You can help Wikipedia by expanding it\.', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Media related to .*? at Wikimedia Commons', '', text, flags=re.IGNORECASE)
    text = re.sub(r'http\S+', '', text)
    
    # --- 2. 预处理 ---
    # 转小写
    text = text.lower()
    # 连字符转空格 (例如 "co-operate" -> "co operate")
    text = text.replace('-', ' ')

    # --- 3. 去除数字 (新增) ---
    # 将所有数字替换为空格。
    # 例如 "born in 1998" -> "born in"
    text = re.sub(r'\d+', ' ', text)

    # --- 4. 去除标点和特殊符号 ---
    # 只保留小写字母 (a-z) 和空格
    text = re.sub(r'[^a-z\s]', ' ', text)

    # --- 5. 去除无意义单字母 (新增) ---
    words = text.split()
    cleaned_words = []
    
    # 保留逻辑：
    # 1. 长度大于1的词 (如 "is", "apple")
    # 2. 或者是特定的有意义单字母: "a" (一个) 和 "i" (我)
    # 3. 其他单字母如 "b", "c", "x" 等都会被过滤掉
    valid_single_letters = {'a', 'i'}
    
    for w in words:
        if len(w) > 1 or w in valid_single_letters:
            cleaned_words.append(w)

    # --- 6. 保存 ---
    final_text = ' '.join(cleaned_words)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(final_text)

    print(f"清洗完成！")
    print(f"原始词数(估): {len(words)}")
    print(f"清洗后词数: {len(cleaned_words)}")
    print(f"结果已保存至: {output_file}")
    
    # 打印效果预览
    print(f"\n预览 (前20个词): {cleaned_words[:20]}")

# ================= 运行 =================
input_filename = '200000en.txt'
output_filename = 'cleaned_corpus.txt'

try:
    aggressive_clean(input_filename, output_filename)
except FileNotFoundError:
    print(f"找不到文件 {input_filename}")