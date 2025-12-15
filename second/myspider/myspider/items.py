# myspider/items.py
import scrapy

class GutenbergItem(scrapy.Item):
    # 我们只需要保存文本内容
    text_content = scrapy.Field()
    title = scrapy.Field() # 方便调试看进度，实际训练不用