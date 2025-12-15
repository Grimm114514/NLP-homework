# myspider/spiders/gutenberg.py
import scrapy
from myspider.items import GutenbergItem

class GutenbergSpider(scrapy.Spider):
    name = "gutenberg"
    allowed_domains = ["gutenberg.org"]
    # 从 Top 100 书籍列表开始，保证语料足够多且经典
    start_urls = ["https://www.gutenberg.org/browse/scores/top"]

    def parse(self, response):
        """第一层：解析 Top 100 列表，找到每本书的链接"""
        # 这里的提取规则可能会随网站变动，目前通常在 <ol> 标签里
        # 我们只抓前 50 本书就足够完成作业了，以免给服务器太大压力或跑太久
        book_links = response.css('ol li a::attr(href)').getall()[:50]
        
        for link in book_links:
            # 链接通常是 /ebooks/12345 的形式
            if "/ebooks/" in link:
                yield response.follow(link, callback=self.parse_book_page)

    def parse_book_page(self, response):
        """第二层：在书籍详情页找到 Plain Text 的下载链接"""
        # 寻找包含 "Plain Text UTF-8" 的链接
        # 这里的 href 通常以 .txt 结尾
        txt_link = response.xpath('//a[contains(text(), "Plain Text UTF-8")]/@href').get()
        
        if txt_link:
            yield response.follow(txt_link, callback=self.parse_text_file)

    def parse_text_file(self, response):
        """第三层：直接获取 .txt 文件的内容"""
        # 此时 response.text 就是整本书的纯文本
        item = GutenbergItem()
        item['text_content'] = response.text
        # 只是为了控制台看着舒服，提取个大概的标题
        item['title'] = response.url.split('/')[-1] 
        yield item