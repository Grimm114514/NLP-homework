import scrapy

class TutorialItem(scrapy.Item):
    # 我们只需要三个字段：
    url = scrapy.Field()         # 文本来源的网址
    text_content = scrapy.Field() # 抓取到的纯文本内容
    language = scrapy.Field()    # 'en' 或 'zh'