# myspider/pipelines.py
from scrapy.exceptions import CloseSpider

class GutenbergPipeline:
    def open_spider(self, spider):
        # 1. 打开文件
        self.file = open('corpus.txt', 'w', encoding='utf-8')
        
        # 2. 初始化计数器
        self.current_word_count = 0
        
        # 3. 从 settings.py 获取限制 (如果没设置，默认10万)
        self.limit = spider.settings.getint('MAX_WORDS_LIMIT', 100000)
        
        print(f"====== 爬虫启动: 目标收集 {self.limit} 个单词 ======")

    def close_spider(self, spider):
        self.file.close()
        print(f"====== 爬虫结束: 最终收集了 {self.current_word_count} 个单词 ======")

    def process_item(self, item, spider):
        # 如果已经超标，直接抛出异常停止 (防止并发请求多余处理)
        if self.current_word_count >= self.limit:
            raise CloseSpider(reason=f"已达到单词限制: {self.limit}")

        raw_text = item['text_content']
        
        # --- 清洗逻辑 (同之前) ---
        start_marker = "*** START OF"
        end_marker = "*** END OF"
        start_idx = raw_text.find(start_marker)
        end_idx = raw_text.find(end_marker)
        
        if start_idx != -1 and end_idx != -1:
            clean_text = raw_text[start_idx + len(start_marker):end_idx]
            first_newline = clean_text.find('\n')
            if first_newline != -1:
                clean_text = clean_text[first_newline:]
        else:
            clean_text = raw_text

        # 去除多余空白
        clean_text = " ".join(clean_text.split())
        
        # --- 统计与写入逻辑 ---
        
        # 1. 计算这本书有多少个词
        # (简单按空格切分统计)
        new_words = clean_text.split()
        num_new_words = len(new_words)
        
        # 2. 如果这本书内容太少（可能是空文件），跳过
        if num_new_words < 100:
            return item

        # 3. 写入文件
        self.file.write(clean_text + "\n")
        
        # [关键点] 强制刷新缓冲区，让你打开文件就能立刻看到内容，不用等
        self.file.flush()
        
        # 4. 更新全局计数器
        self.current_word_count += num_new_words
        
        # 5. 打印实时进度条
        progress = (self.current_word_count / self.limit) * 100
        print(f"Saving [{item['title'][:20]}...] | +{num_new_words} words | Total: {self.current_word_count}/{self.limit} ({progress:.2f}%)")

        # 6. 再次检查是否超标，如果超标，触发停止
        if self.current_word_count >= self.limit:
            # 这里的 reason 会显示在爬虫结束的 log 里
            raise CloseSpider(reason=f"Token Limit Reached: {self.current_word_count}")
            
        return item