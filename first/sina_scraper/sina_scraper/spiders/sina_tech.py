# spiders/sina_tech.py
import scrapy
# 确保从正确的 items 导入
from sina_scraper.items import NewsArticleItem 

class SinaTechSpider(scrapy.Spider):
    # 这个 name = "sina_tech" 是 genspider 自动生成的
    name = "sina_tech"
    allowed_domains = ["sina.com.cn"]
    start_urls = ["https://sina.com.cn/"
                  
                  
                  
                  ]

    def parse(self, response):
        """
        处理“列表页”（即科技首页）。
        任务：找到所有指向文章详情页的链接。
        """
        
        # 寻找文章链接。新浪的文章链接通常包含 'doc-i' 和 '.shtml'
        article_links = response.xpath('//a[contains(@href, "/doc-") and contains(@href, ".shtml")]/@href').getall()
        
        self.log(f"在 {response.url} 找到 {len(article_links)} 个潜在的文章链接。")
        
        for link in article_links:
            yield response.follow(link, callback=self.parse_article)

    def parse_article(self, response):
        """
        处理“详情页”，提取数据。
        """
        
        # 验证这是否是一个有效的文章页 (例如，必须包含标题和正文)
        title = response.xpath('//h1[@class="main-title"]/text()').get()
        content_paragraphs = response.xpath('//div[@id="artibody"]//p/text()').getall()

        # 如果没有标题或没有正文，说明这不是我们想要的文章页，直接返回
        if not title or not content_paragraphs:
            self.log(f"跳过页面 (非标准文章页): {response.url}")
            return

        item = NewsArticleItem()
        
        item['title'] = title.strip()
        item['url'] = response.url
        item['publish_time'] = response.xpath('//span[@class="date"]/text()').get()
        item['source'] = response.xpath('//span[@data-sudaclick="content_media_p"]/text()').get()
        item['content'] = "\n".join(paragraph.strip() for paragraph in content_paragraphs if paragraph.strip())

        # Yield Item，Scrapy 会开始计数
        yield item