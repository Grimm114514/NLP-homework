import scrapy
from tutorial.items import TutorialItem

class ArxivSpider(scrapy.Spider):
    name = 'arxiv' # 运行命令: scrapy crawl arxiv
    
    # 从几个计算机科学(CS)的“最近”列表页开始
    start_urls = [
        'https://arxiv.org/list/cs.AI/recent', # 人工智能
        'https://arxiv.org/list/cs.CL/recent', # 计算语言学
        'https://arxiv.org/list/cs.CV/recent', # 计算机视觉
    ]

    def parse(self, response):
        """
        解析列表页 (e.g., /list/cs.AI/recent)
        """
        
        # 1. 跟进到“摘要页” (Abstract)
        abstract_links = response.css('a[title="Abstract"]::attr(href)').getall()
        for link in abstract_links:
            # response.follow 会自动处理相对路径
            yield response.follow(link, callback=self.parse_abstract)

        # 2. 跟进到其他的“列表页”
        # 我们抓取页面上所有指向其他 /list/ 的链接，实现更广的浏览
        other_lists = response.css('div#dlpage a[href^="/list/"]::attr(href)').getall()
        for list_page in other_lists:
             yield response.follow(list_page, callback=self.parse)

    def parse_abstract(self, response):
        """
        解析摘要页 (e.g., /abs/2510.12345)
        """
        
        # 目标：寻找 "HTML" 版本的全文链接
        html_link = response.css('a[title="HTML (experimental)"]::attr(href)').get()
        
        if html_link:
            # 如果有 HTML 全文，跟进它
            yield response.follow(html_link, callback=self.parse_html_paper)
        else:
            # 备选方案：如果没 HTML，我们就只抓取“摘要”文本
            self.logger.info(f"未找到 HTML 版本 {response.url}, 仅使用摘要。")
            abstract_text = " ".join(response.css('blockquote.abstract::text').getall()).strip()
            
            # 简单清理
            abstract_text = abstract_text.replace("Abstract:", "").strip()

            if not abstract_text:
                return

            item = TutorialItem()
            item['url'] = response.url
            item['language'] = 'en'
            item['text_content'] = abstract_text
            yield item

    def parse_html_paper(self, response):
        """
        解析 HTML 全文页 (e.g., /html/2510.12345)
        """
        
        # 全文在 <div id="paper-body"> 中
        text_list = response.css('#paper-body p::text').getall()
        full_text = " ".join(text_list)
        
        if not full_text:
            self.logger.warning(f"无法从 HTML 页面提取文本: {response.url}")
            return
            
        item = TutorialItem()
        item['url'] = response.url
        item['language'] = 'en'
        item['text_content'] = full_text
        yield item