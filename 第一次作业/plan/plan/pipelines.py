# plan/pipelines.py

class TxtWriterPipeline:

    def __init__(self):
        self.file = None

    def open_spider(self, spider):
        """
        在爬虫启动时被调用，用于打开文件。
        我们使用 'w' (write) 模式，这样每次运行都会覆盖旧文件，
        确保你得到一个全新的语料库。
        """
        self.file = open('corpus.txt', 'w', encoding='utf-8')

    def close_spider(self, spider):
        """
        在爬虫关闭时被调用，用于关闭文件。
        """
        if self.file:
            self.file.close()

    def process_item(self, item, spider):
        """
        爬虫每 yield 一个 item，这个方法就会被调用。
        """
        # 从 item 中获取 'text_content' 字段
        clean_text = item.get('text_content')
        
        if clean_text:
            # 将纯文本内容写入文件
            self.file.write(clean_text)
            
            # 在每本书之间添加一个换行符和分隔符
            # 这样你的 corpus.txt 就不会把所有书黏在一起
            self.file.write("\n\n" + "="*80 + "\n\n")
            
        return item # 必须返回 item