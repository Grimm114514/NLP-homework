# plan/items.py

import scrapy

class BookItem(scrapy.Item):
    # 定义你想要的数据字段
    title = scrapy.Field()       # 书名
    text_content = scrapy.Field()  # 书的纯文本内容
    url = scrapy.Field()         # 来源URL