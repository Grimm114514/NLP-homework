# items.py
import scrapy

class NewsArticleItem(scrapy.Item):
    # 我们只关心正文，但 Item 定义最好还是保留
    # 这样爬虫逻辑更清晰
    title = scrapy.Field()
    content = scrapy.Field()
    publish_time = scrapy.Field()
    source = scrapy.Field()
    url = scrapy.Field()