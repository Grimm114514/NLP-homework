import re
import os

class ChineseDataCleaner:
    def __init__(self, file_path, output_path):
        self.file_path = file_path
        self.output_path = output_path

    def clean_text(self):
        print(f"正在处理中文语料: {self.file_path} ...")
        
        if not os.path.exists(self.file_path):
            print(f"错误: 找不到文件 {self.file_path}")
            return None

        # 尝试不同的编码读取 (中文语料常见 utf-8 或 gb18030)
        content = ""
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            print("UTF-8 解码失败，尝试 GB18030...")
            with open(self.file_path, 'r', encoding='gb18030') as f:
                content = f.read()

        lines = content.split('\n')
        cleaned_words = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # 按空格切分每个 token
            # 格式示例: "19980101-01-001-001/m  迈向/v  充满/v  希望/n"
            tokens = line.split()

            # 处理该行的每一个 token
            for i, token in enumerate(tokens):
                # 1. 跳过行首的日期索引 (通常是每行的第一个元素，且包含数字和横杠)
                if i == 0 and '199801' in token:
                    continue

                # 2. 去除复合词标记 '[' 和 ']'
                # 示例: [中央/n 人民/n ...
                token = token.replace('[', '').replace(']', '')

                # 3. 分离词语和词性
                # 使用 rsplit 确保只从最后一个 / 切分 (防止词语本身含 /)
                if '/' in token:
                    word, tag = token.rsplit('/', 1)
                else:
                    word, tag = token, ''

                # 4. 过滤逻辑
                # - 跳过标点符号 (标记为 w)
                # - 跳过空字符串
                if tag == 'w' or not word.strip():
                    continue
                
                # 5. 加入结果列表
                cleaned_words.append(word)

        # 将所有词用空格连接成一个长字符串
        final_text = " ".join(cleaned_words)

        # 保存
        with open(self.output_path, 'w', encoding='utf-8') as f:
            f.write(final_text)
        
        print(f"中文语料清洗完成！已保存至: {self.output_path}")
        print(f"预览前50个词: {final_text[:100]}...")
        
        return final_text

# --- 单元测试 ---
if __name__ == "__main__":
    cleaner = ChineseDataCleaner('ChineseCorpus199801.txt', 'corpus_cn_cleaned.txt')
    cleaner.clean_text()