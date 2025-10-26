import scrapy
from tutorial.items import TutorialItem

class GovCnSpider(scrapy.Spider):
    name = 'govcn' # 运行命令: scrapy crawl govcn
    
    # 从“最新政策”和“滚动新闻”开始
    start_urls = [
        'http://www.gov.cn/zhengce/zuixin.htm', 
        'http://www.gov.cn/xinwen/gundong.htm',
    ]
    
    # 限制爬虫只在 'gov.cn' 域名下活动
    allowed_domains = ['gov.cn']

    def parse(self, response):
        """
        解析列表页
        """
        
        # 1. 寻找文章链接
        # 匹配两种常见的列表样式
        article_links = response.css('ul.list-datainf li a::attr(href)').getall()
        article_links.extend(response.css('div.list.zhengce_list h4 a::attr(href)').getall())
        
        for link in article_links:
            # response.urljoin 会自动拼接 'http://www.gov.cn'
            link = response.urljoin(link)
            # 检查链接是否是文章页 (网址通常包含日期或 'content_')
            if '/content_' in link or '/20' in link.split('/')[-1]:
                yield response.follow(link, callback=self.parse_article)

        # 2. 寻找“下一页”
        next_page = response.css('a:contains("下一页")::attr(href)').get()
        if next_page:
            yield response.follow(response.urljoin(next_page), callback=self.parse)

    def parse_article(self, response):
        """
        解析文章页
        """
        
        # 主要内容通常在 <div class="pages_content">
        text_list = response.css('div.pages_content p::text').getall()
        
        if not text_list:
            # 尝试备用选择器
            text_list = response.css('div.TRS_Editor p::text').getall()

        if not text_list:
            # 最终备选
            text_list = response.css('p::text').getall()

        full_text = " ".join(text_list)
        
        # 忽略太短的页面（可能是空页或索引）
        if not full_text or len(full_text) < 100: 
            self.logger.debug(f"未找到有效文本: {response.url}")
            return
            
        item = TutorialItem()
        item['url'] = response.url
        item['language'] = 'zh'
        item['text_content'] = full_text
        yield item