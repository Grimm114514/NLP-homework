import os
import re
from scrapy.exceptions import DropItem, CloseSpider
import logging

# 获取 Scrapy 的日志记录器
logger = logging.getLogger(__name__)

class TutorialPipeline:

    def __init__(self):
        # 【修改点③】: 设置 token 目标
        self.token_target = 100_000 # 10 万
        
        # 初始化计数器
        self.token_counts = {
            'en': 0,
            'zh': 0,
        }
        
        # 定义输出目录和文件名
        self.output_dir = 'corpus'
        self.filenames = {
            'en': os.path.join(self.output_dir, 'english_corpus.txt'),
            'zh': os.path.join(self.output_dir, 'chinese_corpus.txt'),
        }
        
        # 存储文件句柄
        self.files = {}

    def open_spider(self, spider):
        # 爬虫启动时
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 为每种语言打开一个文件，模式为 'a' (append, 追加)
        for lang, filename in self.filenames.items():
            # 检查文件是否已存在，如果存在，先粗略计算一下已有的token
            try:
                if os.path.exists(filename):
                    with open(filename, 'r', encoding='utf-8') as f:
                        existing_text = f.read()
                        self.token_counts[lang] = len(existing_text.split())
                    logger.info(f"已恢复 {filename}，检测到 {self.token_counts[lang]} 个 tokens。")
            except Exception as e:
                logger.warning(f"无法读取旧文件 {filename}: {e}")
                
            # 'a' 模式：追加写入
            self.files[lang] = open(filename, 'a', encoding='utf-8')
        
        logger.info(f"Pipeline 已启动。目标: {self.token_target} tokens/语言。")
        logger.info(f"当前状态: EN={self.token_counts['en']}, ZH={self.token_counts['zh']}")

    def close_spider(self, spider):
        # 爬虫关闭时，关闭所有文件
        for file in self.files.values():
            file.close()
        logger.info(f"Pipeline 已关闭。最终 Token 计数: EN={self.token_counts['en']}, ZH={self.token_counts['zh']}")

    def process_item(self, item, spider):
        language = item.get('language')
        
        # 如果 item 的语言不是 en 或 zh，或者该语言已达标，则丢弃
        if language not in self.token_counts or self.token_counts[language] >= self.token_target:
            raise DropItem(f"语言 {language} 已达标或不受支持。")

        # --- 1. 数据清洗 ---
        text = item.get('text_content', '')
        if language == 'en':
            text = re.sub(r'[^a-zA-Z0-9\s.,!?-]', '', text)
        elif language == 'zh':
            text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s，。！？、]', '', text) # 保留汉字和基本标点
        
        # --- 2. Tokenize 和 截断 ---
        tokens = text.split() # 按空白符分割
        if not tokens:
            raise DropItem("文本为空或无效。")
            
        current_count = self.token_counts[language]
        needed_tokens = self.token_target - current_count
        
        if len(tokens) > needed_tokens:
            # 这本书的 token 太多，我们只需要一部分
            tokens_to_write = tokens[:needed_tokens]
            self.token_counts[language] = self.token_target
        else:
            # 这本书的 token 不够，我们全要
            tokens_to_write = tokens
            self.token_counts[language] += len(tokens_to_write)
        
        # --- 3. 写入文件 ---
        text_to_write = " ".join(tokens_to_write)
        
        try:
            # 在文本末尾加一个换行符，用于分隔不同书籍
            self.files[language].write(text_to_write + "\n") 
        except Exception as e:
            raise DropItem(f"写入文件 {self.filenames[language]} 失败: {e}")

        # 打印进度
        logger.info(f"进度: {language.upper()} = {self.token_counts[language]} / {self.token_target} tokens")

        # --- 4. 检查是否所有目标都已达成 ---
        if all(count >= self.token_target for count in self.token_counts.values()):
            # 如果 en 和 zh 都达到了 200 万，就抛出 CloseSpider 异常来停止爬虫
            raise CloseSpider(f"所有语言均已达到 {self.token_target} token 目标。")

        return item