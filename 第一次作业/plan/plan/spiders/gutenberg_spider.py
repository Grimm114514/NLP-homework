# plan/spiders/gutenberg_spider.py

import scrapy
from plan.items import BookItem  # 导入你在 items.py 中定义的 Item

class GutenbergSpider(scrapy.Spider):
    name = 'gutenberg'
    allowed_domains = ['gutenberg.org']
    
    # 我们不使用 start_urls，而是重写 start_requests 来指定入口
    def start_requests(self):
        # 从 "Top 100 Ebooks" 页面开始
        yield scrapy.Request(url='https://www.gutenberg.org/browse/scores/top', callback=self.parse_top100)

    def parse_top100(self, response):
        """
        这个函数解析 "Top 100" 列表页面。
        它会找到每本书的链接，例如 /ebooks/1234
        """
        # 找到所有指向书本的链接 (形如 /ebooks/xxxx)
        # 这里的 CSS 选择器 'li.booklink a' 是根据古腾堡 "Top 100" 页面的HTML结构来的
        book_links = response.css('li.booklink a::attr(href)').getall()
        for link in book_links:
            # response.follow 会自动拼接成完整的 URL
            yield response.follow(link, callback=self.parse_book_page)

    def parse_book_page(self, response):
        """
        这个函数解析单个书本的详情页。
        它的目标是找到 "Plain Text UTF-8" 版本的链接。
        """
        # 提取书名
        # H1 标签的内容是书名
        title = response.css('h1[itemprop="name"]::text').get()
        if not title:
            # 备用方案（如果h1没取到）
            title = response.css('h1::text').get()
            
        # 找到 "Plain Text UTF-8" 文件的链接
        # 我们使用 XPath 查找文本内容包含 "Plain Text UTF-8" 的 <a> 标签
        text_link = response.xpath('//a[contains(text(), "Plain Text UTF-8")]/@href').get()

        if text_link:
            # 同样，使用 response.follow 来处理相对链接
            # 我们使用 meta 参数把书名传递给下一个解析函数
            yield response.follow(text_link, callback=self.parse_text, meta={'title': title, 'url': response.url})
        else:
            self.logger.warning(f"Could not find 'Plain Text UTF-8' link for: {response.url}")

    def parse_text(self, response):
        """
        这是最后一步，解析 .txt 文件。
        这个函数的核心是清理掉古腾堡的法律声明文本。
        """
        # 从 meta 中取回书名
        title = response.meta['title']
        
        # response.text 包含了 .txt 文件的所有内容
        full_text = response.text

        # --- 关键的清理步骤 ---
        # 找到正文的开始和结束标记
        start_marker = "*** START OF THIS PROJECT GUTENBERG EBOOK"
        end_marker = "*** END OF THIS PROJECT GUTENBERG EBOOK"
        
        # 有些文本可能是 "START OF THE PROJECT..."
        if start_marker not in full_text:
            start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK"

        if end_marker not in full_text:
            end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK"

        start_index = full_text.find(start_marker)
        end_index = full_text.find(end_marker)

        clean_text = ""

        if start_index != -1 and end_index != -1:
            # 找到了标记，提取它们之间的文本
            start_index += len(start_marker) # 移动到标记之后
            clean_text = full_text[start_index:end_index].strip()
        else:
            # 如果没找到标记（例如，某些非标准的文本），
            # 我们就记录一个警告，然后（不完美地）使用全部文本
            self.logger.warning(f"Could not find start/end markers in: {response.url}")
            clean_text = full_text # 回退方案
        
        # --- 清理完毕 ---

        # 创建 Item
        item = BookItem()
        item['title'] = title.strip() if title else 'Unknown Title'
        item['url'] = response.meta['url']
        item['text_content'] = clean_text
        
        yield item