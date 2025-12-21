import re

def clean_and_cut_corpus_strict(input_file, output_file):
    print(f"正在读取文件: {input_file} ...")
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except UnicodeDecodeError:
        print("UTF-8 读取失败，尝试 GBK 编码...")
        with open(input_file, 'r', encoding='gbk') as f:
            lines = f.readlines()

    # 截断一半
    lines = lines[:len(lines) // 2]
    
    cleaned_data = []
    
    # === 正则表达式：只匹配中文字符 ===
    # \u4e00-\u9fa5 是常用汉字的 Unicode 范围
    chinese_pattern = re.compile(r'^[\u4e00-\u9fa5]+$')

    for line in lines:
        line = line.strip()
        if not line: continue
        
        tokens = line.split()
        line_words = []
        
        for token in tokens:
            # 1. 也是先去掉词性标注
            if '/' in token:
                # 处理像 [中国/ns 人民/n]nt 这种复杂情况，先去掉 [
                token = token.replace('[', '')
                word = token.split('/')[0]
            else:
                word = token
            
            # 2. 严格过滤：如果不是纯汉字，直接丢弃
            # 这会把 "1998", "50%", ",", "。" 全部过滤掉
            if chinese_pattern.match(word):
                line_words.append(word)
        
        # 如果这一行还有剩下的词，就写入
        # 我们设定一个阈值，比如至少有3个词才算一句话，太短的也没法训练上下文
        if len(line_words) > 2:
            cleaned_data.append(" ".join(line_words))

    with open(output_file, 'w', encoding='utf-8') as f:
        for line in cleaned_data:
            f.write(line + '\n')
            
    print(f"严格清洗完成！只保留了纯汉字。")
    print(f"结果已保存至: {output_file}")
    print(f"示例前两行: {cleaned_data[:2]}")

# ================= 运行 =================
input_filename = 'ChineseCorpus199801_half.txt' 
output_filename = 'corpus_cleaned_strict.txt'

clean_and_cut_corpus_strict(input_filename, output_filename)