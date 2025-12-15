import re
import os

class DataCleaner:
    def __init__(self, file_path, output_path):
        self.file_path = file_path
        self.output_path = output_path

    def clean_text(self):
        print(f"正在读取文件: {self.file_path} ...")
        
        if not os.path.exists(self.file_path):
            print(f"错误: 找不到文件 {self.file_path}")
            return False

        with open(self.file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # --- 1. 去除 标签 ---
        # 你的源文件中包含大量这种标记，必须去除
        # 匹配左括号 [ + source: + 空格 + 数字 + 右括号 ]
        text = re.sub(r'\[source: \d+\]', ' ', content)

        # --- 2. 统一转小写 ---
        # 减少词表大小，让 "The" 和 "the" 视为同一个词
        text = text.lower()

        # --- 3. 去除舞台提示和括号内容 ---
        # 针对剧本《不可儿戏》，去除如 [Exit Merriman] 这种动作提示
        text = re.sub(r'\[.*?\]', ' ', text)

        # --- 4. 处理特殊格式 ---
        # 去除用于强调的下划线 (如 _word_)
        text = text.replace('_', '')
        
        # --- 5. 去除标点符号和特殊字符 ---
        # 只保留英文字母和数字，将其他字符（标点、换行符）替换为空格
        # 如果你希望保留句号作为句子结束符，可以修改正则表达式
        text = re.sub(r'[^a-z0-9\s]', ' ', text)

        # --- 6. 压缩多余空格 ---
        # 将多个空格或换行符合并为一个空格
        text = re.sub(r'\s+', ' ', text).strip()

        # --- 7. 保存 ---
        # 此时 text 是一个巨大的长字符串，单词之间用空格分隔
        with open(self.output_path, 'w', encoding='utf-8') as f:
            f.write(text)
        
        print(f"清洗完成！已保存至: {self.output_path}")
        print(f"预览前100个字符: {text[:100]}...")
        
        # 返回清洗后的文本（字符串形式），方便直接调用
        return text

# --- 单元测试 ---
if __name__ == "__main__":
    # 假设你的原始文件名为 corpus.txt
    cleaner = DataCleaner('corpus.txt', 'corpus_cleaned.txt')
    cleaner.clean_text()