# pipelines.py
from scrapy.exceptions import DropItem

class SinaScraperPipeline:
    def open_spider(self, spider):
        # 1. 文件名和编码
        self.filename = f'{spider.name}_news.txt'
        self.file = open(self.filename, 'w', encoding='utf-8')
        spider.log(f"数据将保存到 {self.filename} (仅拼接正文内容)")

    def close_spider(self, spider):
        self.file.close()
        spider.log(f"纯文本语料库已保存到 {self.filename}")

    def process_item(self, item, spider):
        # 2. 只获取 'content' 字段
        content = item.get('content')

        # 3. 验证正文是否存在
        if content:
            # 4. 将正文内容直接写入文件
            self.file.write(content)
            
            # 5. 在不同文章的正文之间添加两个换行符
            self.file.write("\n\n")
            
            return item
        else:
            # 如果没有正文，则丢弃这个 item
            raise DropItem(f"文章缺少正文，已跳过: {item.get('url')}")